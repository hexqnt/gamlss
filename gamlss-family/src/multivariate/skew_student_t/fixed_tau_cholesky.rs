#![allow(clippy::needless_range_loop, clippy::suboptimal_flops)]

//! Fixed-degrees-of-freedom Cholesky multivariate skew-Student-t.

use std::marker::PhantomData;

use gamlss_core::{
    CompilableFamily, Family, FixedDimensionalFamily, HasObservationDimension, Identity,
    InitialEtaFromTheta, Link, LocationCholesky, Log, ModelError, Nu, ObservationView,
    PositiveLink,
    shape::{Product, ShapeValues, Vector},
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};
use gamlss_special::ln_gamma;

use crate::multivariate::{
    elliptical, initial,
    matrix::FixedLowerTriangular,
    student_t::{nll_location_scale, valid_degrees_of_freedom},
};

#[cfg(feature = "rand")]
use super::try_sample_location_scale;
use super::{skew_student_t_gradient_terms, skew_student_t_nll};

/// Default-link fixed-`tau` multivariate skew-Student-t.
pub type MvSkewStudentTFixedTauCholeskyDefault<const D: usize> =
    MvSkewStudentTFixedTauCholesky<D, Identity, Log, Identity, Identity>;

/// Azzalini-style multivariate skew-Student-t with fixed degrees of freedom.
///
/// For `z = L^-1 (y - mu)` and `q = z'z`, the density is the multivariate
/// Student-t kernel multiplied by
/// `2 T_{tau + D}(alpha' z sqrt((tau + D) / (tau + q)))`.
/// `L L'` is a scale matrix, `alpha` acts in whitened coordinates, and `mu` is
/// a location rather than the response mean when `alpha != 0`.
///
/// `tau` belongs to the family instance and is fixed during fitting. This
/// removes the only unresolved score term—the derivative of a Student-t CDF
/// with respect to its degrees of freedom—while retaining analytic gradients
/// for all fitted location, Cholesky and shape predictors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MvSkewStudentTFixedTauCholesky<
    const D: usize,
    MuLink = Identity,
    DiagonalLink = Log,
    OffDiagonalLink = Identity,
    ShapeLink = Identity,
> {
    tau: f64,
    marker: PhantomData<(MuLink, DiagonalLink, OffDiagonalLink, ShapeLink)>,
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink, ShapeLink>
    MvSkewStudentTFixedTauCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, ShapeLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
    ShapeLink: Link<f64>,
{
    /// Creates a family with fixed positive degrees of freedom.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `D == 0` or `tau` is not
    /// finite and strictly positive.
    pub const fn try_new(tau: f64) -> Result<Self, ModelError> {
        if D == 0 {
            return Err(ModelError::InvalidParameter {
                parameter: "dimension",
                expected: "positive",
            });
        }
        if !(tau > 0.0 && tau.is_finite()) {
            return Err(ModelError::InvalidParameter {
                parameter: "tau",
                expected: "finite and > 0",
            });
        }
        Ok(Self {
            tau,
            marker: PhantomData,
        })
    }

    /// Creates a family with fixed positive degrees of freedom.
    ///
    /// # Panics
    ///
    /// Panics when `D == 0` or `tau` is not finite and strictly positive.
    #[must_use]
    pub const fn new(tau: f64) -> Self {
        assert!(
            D > 0,
            "multivariate skew-Student-t dimension must be positive"
        );
        assert!(
            tau > 0.0 && tau.is_finite(),
            "multivariate skew-Student-t tau must be finite and positive"
        );
        Self {
            tau,
            marker: PhantomData,
        }
    }

    /// Fixed degrees of freedom.
    #[must_use]
    pub const fn tau(&self) -> f64 {
        self.tau
    }

    /// Creates natural-scale parameters tied to this family's fixed `tau`.
    ///
    /// This is the preferred checked construction path when parameters will be
    /// passed directly to [`Family::nll`] or
    /// [`gamlss_core::TrySimulate::try_sample`].
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] unless location, Cholesky scale
    /// and shape are valid.
    pub fn try_theta(
        &self,
        mu: [f64; D],
        cholesky: FixedLowerTriangular<D>,
        shape: [f64; D],
    ) -> Result<MvSkewStudentTFixedTauCholeskyTheta<D>, ModelError> {
        MvSkewStudentTFixedTauCholeskyTheta::try_new(mu, cholesky, shape, self.tau)
    }

    fn theta_from_eta(
        &self,
        eta: &MvSkewStudentTFixedTauCholeskyEta<D>,
    ) -> MvSkewStudentTFixedTauCholeskyTheta<D> {
        let mu = eta.mu.map(MuLink::inverse);
        let shape = eta.shape.map(ShapeLink::inverse);
        let mut cholesky = FixedLowerTriangular::zeros();
        for row in 0..D {
            for col in 0..=row {
                let eta_value = eta.cholesky.lower(row, col);
                let value = if row == col {
                    DiagonalLink::inverse(eta_value)
                } else {
                    OffDiagonalLink::inverse(eta_value)
                };
                cholesky
                    .set_lower(row, col, value)
                    .expect("valid lower-triangular index");
            }
        }
        MvSkewStudentTFixedTauCholeskyTheta::from_parts_unchecked(mu, cholesky, shape, self.tau)
    }

    fn nan_eta() -> MvSkewStudentTFixedTauCholeskyEta<D> {
        MvSkewStudentTFixedTauCholeskyEta {
            mu: [f64::NAN; D],
            cholesky: FixedLowerTriangular::from_lower_rows([[f64::NAN; D]; D]),
            shape: [f64::NAN; D],
        }
    }

    fn nll_theta(
        &self,
        observation: [f64; D],
        theta: &MvSkewStudentTFixedTauCholeskyTheta<D>,
    ) -> f64 {
        if theta.tau.to_bits() != self.tau.to_bits() || !valid_theta(theta) {
            return f64::INFINITY;
        }
        let mut z = [0.0; D];
        let base_nll =
            nll_location_scale(observation, &theta.mu, &theta.cholesky, theta.tau, &mut z);
        skew_student_t_nll(base_nll, &z, &theta.shape, theta.tau)
    }

    fn nll_and_gradient_eta_values(
        &self,
        observation: [f64; D],
        eta: &MvSkewStudentTFixedTauCholeskyEta<D>,
    ) -> (f64, MvSkewStudentTFixedTauCholeskyEta<D>) {
        let theta = self.theta_from_eta(eta);
        if !valid_theta(&theta) {
            return (f64::INFINITY, Self::nan_eta());
        }
        let mut z = [0.0; D];
        let base_nll =
            nll_location_scale(observation, &theta.mu, &theta.cholesky, theta.tau, &mut z);
        let Some(terms) = skew_student_t_gradient_terms(base_nll, &z, &theta.shape, theta.tau)
        else {
            return (f64::INFINITY, Self::nan_eta());
        };
        let mut location_score = [0.0; D];
        if !elliptical::transpose_solve(
            D,
            &theta.cholesky,
            &terms.standardized,
            &mut location_score,
        ) {
            return (f64::INFINITY, Self::nan_eta());
        }

        let mut gradient = MvSkewStudentTFixedTauCholeskyEta {
            mu: std::array::from_fn(|component| {
                -location_score[component] * MuLink::derivative_inverse(eta.mu[component])
            }),
            cholesky: FixedLowerTriangular::zeros(),
            shape: std::array::from_fn(|component| {
                terms.shape[component] * ShapeLink::derivative_inverse(eta.shape[component])
            }),
        };
        for row in 0..D {
            for col in 0..=row {
                let eta_value = eta.cholesky.lower(row, col);
                let score = if row == col {
                    DiagonalLink::derivative_log_inverse(eta_value)
                        - location_score[row] * z[col] * DiagonalLink::derivative_inverse(eta_value)
                } else {
                    -location_score[row] * z[col] * OffDiagonalLink::derivative_inverse(eta_value)
                };
                gradient
                    .cholesky
                    .set_lower(row, col, score)
                    .expect("valid lower-triangular index");
            }
        }
        (terms.nll, gradient)
    }
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink, ShapeLink> Default
    for MvSkewStudentTFixedTauCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, ShapeLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
    ShapeLink: Link<f64>,
{
    fn default() -> Self {
        Self::new(5.0)
    }
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink, ShapeLink> Family
    for MvSkewStudentTFixedTauCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, ShapeLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
    ShapeLink: Link<f64>,
{
    type Eta = MvSkewStudentTFixedTauCholeskyEta<D>;
    type Theta = MvSkewStudentTFixedTauCholeskyTheta<D>;
    type GradientEta = MvSkewStudentTFixedTauCholeskyEta<D>;
    type Observation<'obs> = [f64; D];
    type Workspace = ();

    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut ()) -> Self::Theta {
        self.theta_from_eta(eta)
    }

    fn nll(&self, observation: [f64; D], theta: &Self::Theta, _workspace: &mut ()) -> f64 {
        self.nll_theta(observation, theta)
    }

    fn nll_eta(&self, observation: [f64; D], eta: &Self::Eta, _workspace: &mut ()) -> f64 {
        self.nll_theta(observation, &self.theta_from_eta(eta))
    }

    fn nll_and_gradient_eta(
        &self,
        observation: [f64; D],
        eta: &Self::Eta,
        _workspace: &mut (),
    ) -> (f64, Self::GradientEta) {
        self.nll_and_gradient_eta_values(observation, eta)
    }
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink, ShapeLink> FixedDimensionalFamily<D>
    for MvSkewStudentTFixedTauCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, ShapeLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
    ShapeLink: Link<f64>,
{
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink, ShapeLink> HasObservationDimension
    for MvSkewStudentTFixedTauCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, ShapeLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
    ShapeLink: Link<f64>,
{
    fn observation_dimension(&self) -> usize {
        D
    }
}

#[cfg(feature = "rand")]
impl<Rng, const D: usize, MuLink, DiagonalLink, OffDiagonalLink, ShapeLink> TrySimulate<Rng>
    for MvSkewStudentTFixedTauCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, ShapeLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
    ShapeLink: Link<f64>,
{
    type Sample = [f64; D];

    fn try_sample(
        &self,
        rng: &mut Rng,
        theta: &Self::Theta,
    ) -> Result<Self::Sample, SimulationError> {
        if theta.tau.to_bits() != self.tau.to_bits() || !valid_theta(theta) {
            return Err(SimulationError::InvalidParameters(
                "fixed-tau multivariate skew-Student-t theta",
            ));
        }
        try_sample_location_scale(rng, theta.mu, &theta.cholesky, &theta.shape, theta.tau)
    }
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink, ShapeLink> CompilableFamily
    for MvSkewStudentTFixedTauCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, ShapeLink>
where
    MuLink: InitialEtaFromTheta<f64>,
    DiagonalLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    OffDiagonalLink: InitialEtaFromTheta<f64>,
    ShapeLink: InitialEtaFromTheta<f64>,
{
    type Shape = Product<LocationCholesky<D>, Vector<Nu, D>>;

    fn eta_from_shape(values: ShapeValues<Self::Shape>) -> Self::Eta {
        MvSkewStudentTFixedTauCholeskyEta::new(
            values.0.0,
            FixedLowerTriangular::from_lower_rows(values.0.1),
            values.1,
        )
    }

    fn gradient_to_shape(gradient: &Self::GradientEta) -> ShapeValues<Self::Shape> {
        let lower = std::array::from_fn(|row| {
            std::array::from_fn(|col| gradient.cholesky.get(row, col).unwrap_or(0.0))
        });
        ((gradient.mu, lower), gradient.shape)
    }

    fn initial_shape<'obs, Obs>(&self, obs: &'obs Obs) -> ShapeValues<Self::Shape>
    where
        Obs: ObservationView<'obs, Observation = [f64; D]> + 'obs,
    {
        (
            initial::location_cholesky::<D, MuLink, DiagonalLink, OffDiagonalLink, Obs>(obs),
            [ShapeLink::initial_eta_from_theta(0.0); D],
        )
    }
}

/// Link-scale predictors for fixed-`tau` multivariate skew-Student-t.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MvSkewStudentTFixedTauCholeskyEta<const D: usize> {
    mu: [f64; D],
    cholesky: FixedLowerTriangular<D>,
    shape: [f64; D],
}

impl<const D: usize> MvSkewStudentTFixedTauCholeskyEta<D> {
    /// Creates link-scale predictors.
    #[must_use]
    pub const fn new(mu: [f64; D], cholesky: FixedLowerTriangular<D>, shape: [f64; D]) -> Self {
        Self {
            mu,
            cholesky,
            shape,
        }
    }

    /// Location predictors.
    #[must_use]
    pub const fn mu(&self) -> &[f64; D] {
        &self.mu
    }

    /// Mutable location predictors.
    #[must_use]
    pub const fn mu_mut(&mut self) -> &mut [f64; D] {
        &mut self.mu
    }

    /// Lower-triangular Cholesky predictors.
    #[must_use]
    pub const fn cholesky(&self) -> &FixedLowerTriangular<D> {
        &self.cholesky
    }

    /// Mutable lower-triangular Cholesky predictors.
    #[must_use]
    pub const fn cholesky_mut(&mut self) -> &mut FixedLowerTriangular<D> {
        &mut self.cholesky
    }

    /// Whitened-coordinate shape predictors.
    #[must_use]
    pub const fn shape(&self) -> &[f64; D] {
        &self.shape
    }

    /// Mutable whitened-coordinate shape predictors.
    #[must_use]
    pub const fn shape_mut(&mut self) -> &mut [f64; D] {
        &mut self.shape
    }
}

/// Natural-scale fixed-`tau` multivariate skew-Student-t parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MvSkewStudentTFixedTauCholeskyTheta<const D: usize> {
    mu: [f64; D],
    cholesky: FixedLowerTriangular<D>,
    shape: [f64; D],
    tau: f64,
}

impl<const D: usize> MvSkewStudentTFixedTauCholeskyTheta<D> {
    /// Creates checked natural-scale parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] unless dimension, location,
    /// Cholesky scale, shape and degrees of freedom are valid.
    pub fn try_new(
        mu: [f64; D],
        cholesky: FixedLowerTriangular<D>,
        shape: [f64; D],
        tau: f64,
    ) -> Result<Self, ModelError> {
        let theta = Self::from_parts_unchecked(mu, cholesky, shape, tau);
        if valid_theta(&theta) {
            Ok(theta)
        } else {
            Err(ModelError::InvalidParameter {
                parameter: "fixed-tau multivariate skew-Student-t theta",
                expected: "positive dimension, finite values, positive Cholesky diagonal and positive tau",
            })
        }
    }

    const fn from_parts_unchecked(
        mu: [f64; D],
        cholesky: FixedLowerTriangular<D>,
        shape: [f64; D],
        tau: f64,
    ) -> Self {
        Self {
            mu,
            cholesky,
            shape,
            tau,
        }
    }

    /// Location vector `mu`.
    #[must_use]
    pub const fn mu(&self) -> &[f64; D] {
        &self.mu
    }

    /// Cholesky factor of the Student-t scale matrix.
    #[must_use]
    pub const fn cholesky(&self) -> &FixedLowerTriangular<D> {
        &self.cholesky
    }

    /// Whitened-coordinate shape vector.
    #[must_use]
    pub const fn shape(&self) -> &[f64; D] {
        &self.shape
    }

    /// Fixed degrees of freedom.
    #[must_use]
    pub const fn tau(&self) -> f64 {
        self.tau
    }

    /// Actual response mean when `tau > 1`.
    #[must_use]
    pub fn mean(&self) -> Option<[f64; D]> {
        if self.tau <= 1.0 {
            return None;
        }
        let normalization = self
            .shape
            .iter()
            .fold(1.0_f64, |norm, shape| norm.hypot(*shape));
        let delta = self.shape.map(|shape| shape / normalization);
        let coefficient = (0.5 * (self.tau / std::f64::consts::PI).ln()
            + ln_gamma(0.5 * (self.tau - 1.0))
            - ln_gamma(0.5 * self.tau))
        .exp();
        Some(std::array::from_fn(|row| {
            self.mu[row]
                + coefficient
                    * (0..=row)
                        .map(|col| self.cholesky.lower(row, col) * delta[col])
                        .sum::<f64>()
        }))
    }

    /// Returns one covariance entry when `tau > 2`.
    #[must_use]
    pub fn covariance(&self, row: usize, col: usize) -> Option<f64> {
        if self.tau <= 2.0 || row >= D || col >= D {
            return None;
        }
        let mean = self.mean()?;
        let location_shift_row = mean[row] - self.mu[row];
        let location_shift_col = mean[col] - self.mu[col];
        let scale = (0..=row.min(col))
            .map(|index| self.cholesky.lower(row, index) * self.cholesky.lower(col, index))
            .sum::<f64>();
        Some(self.tau / (self.tau - 2.0) * scale - location_shift_row * location_shift_col)
    }
}

fn valid_theta<const D: usize>(theta: &MvSkewStudentTFixedTauCholeskyTheta<D>) -> bool {
    valid_degrees_of_freedom(theta.tau)
        && elliptical::valid_location_scale(D, &theta.mu, &theta.cholesky)
        && theta.shape.iter().all(|shape| shape.is_finite())
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{
        CholeskyScale, DenseDesign, Family, Gamlss, LinearPredictorBlock,
        LowerTriangularParameterBlock, ModelError, Mu, NoPenalty, Nu, ParameterBlocks,
        VectorParameterBlock,
    };

    #[test]
    fn constructors_and_family_instance_enforce_fixed_tau() {
        assert_eq!(
            MvSkewStudentTFixedTauCholeskyDefault::<0>::try_new(5.0),
            Err(ModelError::InvalidParameter {
                parameter: "dimension",
                expected: "positive",
            })
        );
        assert!(MvSkewStudentTFixedTauCholeskyDefault::<2>::try_new(0.0).is_err());
        assert!(MvSkewStudentTFixedTauCholeskyDefault::<2>::try_new(f64::NAN).is_err());

        let family = MvSkewStudentTFixedTauCholeskyDefault::<2>::new(5.0);
        let mismatched = MvSkewStudentTFixedTauCholeskyTheta::try_new(
            [0.0; 2],
            FixedLowerTriangular::from_lower_rows([[1.0, 0.0], [0.2, 0.8]]),
            [0.5, -0.3],
            6.0,
        )
        .unwrap();
        assert!(family.nll([0.1, -0.2], &mismatched, &mut ()).is_infinite());
    }

    use super::{
        MvSkewStudentTFixedTauCholeskyDefault, MvSkewStudentTFixedTauCholeskyEta,
        MvSkewStudentTFixedTauCholeskyTheta,
    };
    use crate::{
        SkewStudentTMuSigmaNuTau, SkewStudentTTheta,
        multivariate::{
            matrix::FixedLowerTriangular,
            student_t::{
                MvStudentTCholeskyDefault, MvStudentTCholeskyEta, MvStudentTCholeskyTheta,
            },
        },
    };

    #[test]
    fn zero_shape_matches_multivariate_student_t() {
        let skew = MvSkewStudentTFixedTauCholeskyDefault::<2>::new(6.0);
        let student = MvStudentTCholeskyDefault::<2>::new();
        let cholesky = FixedLowerTriangular::from_lower_rows([[1.1, 0.0], [0.2, 0.7]]);
        let skew_theta =
            MvSkewStudentTFixedTauCholeskyTheta::try_new([0.2, -0.4], cholesky, [0.0, 0.0], 6.0)
                .unwrap();
        let student_theta = MvStudentTCholeskyTheta::try_new([0.2, -0.4], cholesky, 6.0).unwrap();
        let observation = [1.0, -0.8];
        assert_relative_eq!(
            skew.nll(observation, &skew_theta, &mut ()),
            student.nll(observation, &student_theta, &mut ()),
            epsilon = 1.0e-12
        );

        let eta_cholesky =
            FixedLowerTriangular::from_lower_rows([[1.1_f64.ln(), 0.0], [0.2, 0.7_f64.ln()]]);
        let skew_eta =
            MvSkewStudentTFixedTauCholeskyEta::new([0.2, -0.4], eta_cholesky, [0.0, 0.0]);
        let student_eta =
            MvStudentTCholeskyEta::new([0.2, -0.4], eta_cholesky, (6.0_f64 - 2.0).ln());
        let (skew_nll, skew_gradient) = skew.nll_and_gradient_eta(observation, &skew_eta, &mut ());
        let (student_nll, student_gradient) =
            student.nll_and_gradient_eta(observation, &student_eta, &mut ());
        assert_relative_eq!(skew_nll, student_nll, epsilon = 1.0e-12);
        for component in 0..2 {
            assert_relative_eq!(
                skew_gradient.mu()[component],
                student_gradient.mu[component],
                epsilon = 1.0e-12
            );
        }
        for row in 0..2 {
            for col in 0..=row {
                assert_relative_eq!(
                    skew_gradient.cholesky().get(row, col).unwrap(),
                    student_gradient.cholesky.get(row, col).unwrap(),
                    epsilon = 1.0e-12
                );
            }
        }
    }

    #[test]
    fn one_dimensional_case_matches_scalar_skew_student_t() {
        let multivariate = MvSkewStudentTFixedTauCholeskyDefault::<1>::new(5.0);
        let scalar = SkewStudentTMuSigmaNuTau::new();
        let theta = MvSkewStudentTFixedTauCholeskyTheta::try_new(
            [0.4],
            FixedLowerTriangular::from_lower_rows([[0.8]]),
            [1.3],
            5.0,
        )
        .unwrap();
        assert_relative_eq!(
            multivariate.nll([1.1], &theta, &mut ()),
            scalar.nll(
                1.1,
                &SkewStudentTTheta {
                    mu: 0.4,
                    sigma: 0.8,
                    nu: 1.3,
                    tau: 5.0,
                },
                &mut ()
            ),
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn analytic_gradient_matches_finite_difference() {
        let family = MvSkewStudentTFixedTauCholeskyDefault::<2>::new(5.5);
        let eta = MvSkewStudentTFixedTauCholeskyEta::new(
            [0.1, -0.2],
            FixedLowerTriangular::from_lower_rows([[0.0, 0.0], [0.2, -0.1]]),
            [0.7, -0.4],
        );
        let observation = [0.7, -0.8];
        let (_, gradient) = family.nll_and_gradient_eta(observation, &eta, &mut ());
        let epsilon = 1.0e-6;
        for component in 0..2 {
            let mut plus = eta;
            plus.mu_mut()[component] += epsilon;
            let mut minus = eta;
            minus.mu_mut()[component] -= epsilon;
            let finite_difference = (family.nll_eta(observation, &plus, &mut ())
                - family.nll_eta(observation, &minus, &mut ()))
                / (2.0 * epsilon);
            assert_relative_eq!(
                gradient.mu()[component],
                finite_difference,
                epsilon = 1.0e-6
            );
        }
        for row in 0..2 {
            for col in 0..=row {
                let current = eta.cholesky().get(row, col).unwrap();
                let mut plus = eta;
                plus.cholesky_mut()
                    .set_lower(row, col, current + epsilon)
                    .unwrap();
                let mut minus = eta;
                minus
                    .cholesky_mut()
                    .set_lower(row, col, current - epsilon)
                    .unwrap();
                let finite_difference = (family.nll_eta(observation, &plus, &mut ())
                    - family.nll_eta(observation, &minus, &mut ()))
                    / (2.0 * epsilon);
                assert_relative_eq!(
                    gradient.cholesky().get(row, col).unwrap(),
                    finite_difference,
                    epsilon = 1.0e-6
                );
            }
        }
        for component in 0..2 {
            let mut plus = eta;
            plus.shape_mut()[component] += epsilon;
            let mut minus = eta;
            minus.shape_mut()[component] -= epsilon;
            let finite_difference = (family.nll_eta(observation, &plus, &mut ())
                - family.nll_eta(observation, &minus, &mut ()))
                / (2.0 * epsilon);
            assert_relative_eq!(
                gradient.shape()[component],
                finite_difference,
                epsilon = 1.0e-6
            );
        }
    }

    #[test]
    fn far_negative_skew_argument_keeps_finite_nll_and_gradient() {
        let family = MvSkewStudentTFixedTauCholeskyDefault::<1>::new(5.0);
        let eta = MvSkewStudentTFixedTauCholeskyEta::new(
            [0.0],
            FixedLowerTriangular::from_lower_rows([[0.0]]),
            [1.0e200],
        );
        let (nll, gradient) = family.nll_and_gradient_eta([-1.0], &eta, &mut ());
        assert!(nll.is_finite());
        assert!(gradient.mu()[0].is_finite());
        assert!(gradient.cholesky().get(0, 0).is_some_and(f64::is_finite));
        assert!(gradient.shape()[0].is_finite());
    }

    #[test]
    fn moments_obey_tau_existence_boundaries() {
        let cholesky = FixedLowerTriangular::from_lower_rows([[1.0, 0.0], [0.3, 0.8]]);
        let no_mean =
            MvSkewStudentTFixedTauCholeskyTheta::try_new([0.0, 0.0], cholesky, [0.5, -0.2], 0.8)
                .unwrap();
        assert!(no_mean.mean().is_none());
        let no_covariance =
            MvSkewStudentTFixedTauCholeskyTheta::try_new([0.0, 0.0], cholesky, [0.5, -0.2], 1.8)
                .unwrap();
        assert!(no_covariance.mean().is_some());
        assert!(no_covariance.covariance(0, 0).is_none());
        let finite =
            MvSkewStudentTFixedTauCholeskyTheta::try_new([0.0, 0.0], cholesky, [0.5, -0.2], 5.0)
                .unwrap();
        assert!(finite.covariance(0, 1).is_some_and(f64::is_finite));
    }

    #[test]
    fn static_shape_is_fit_ready() {
        let response = [[0.2, -0.3], [0.8, 0.4], [-0.5, 0.7]];
        let rows = response.len();
        let intercept = || LinearPredictorBlock::new(DenseDesign::intercept(rows));
        let mu = VectorParameterBlock::<Mu, 2, _, _>::new([intercept(), intercept()], NoPenalty, 0);
        let cholesky = LowerTriangularParameterBlock::<CholeskyScale, 2, _, _>::new(
            vec![intercept(), intercept(), intercept()],
            NoPenalty,
            0,
        );
        let shape =
            VectorParameterBlock::<Nu, 2, _, _>::new([intercept(), intercept()], NoPenalty, 0);
        let model = Gamlss::try_new_with_observations(
            MvSkewStudentTFixedTauCholeskyDefault::<2>::new(5.0),
            ParameterBlocks::new((mu, cholesky, shape)),
            response.as_slice(),
        )
        .unwrap();
        let beta: [f64; 7] = [0.1, -0.2, 0.0, 0.1, -0.1, 0.5, -0.3];
        let mut gradient: [f64; 7] = [0.0; 7];
        let value = model.try_value_gradient_into(&beta, &mut gradient).unwrap();
        assert!(value.is_finite());
        assert!(gradient.iter().all(|score| score.is_finite()));
    }

    #[cfg(feature = "rand")]
    #[test]
    fn sampling_returns_finite_vectors() {
        use gamlss_core::TrySimulate;
        use rand::SeedableRng;

        let family = MvSkewStudentTFixedTauCholeskyDefault::<2>::new(5.0);
        let theta = MvSkewStudentTFixedTauCholeskyTheta::try_new(
            [0.2, -0.1],
            FixedLowerTriangular::from_lower_rows([[1.0, 0.0], [0.3, 0.8]]),
            [1.2, -0.7],
            5.0,
        )
        .unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(47);
        assert!(
            family
                .try_sample(&mut rng, &theta)
                .is_ok_and(|sample| sample.iter().all(|value| value.is_finite()))
        );
    }
}
