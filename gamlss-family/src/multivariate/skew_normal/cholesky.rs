#![allow(clippy::needless_range_loop, clippy::suboptimal_flops)]

//! Cholesky-kernel multivariate skew-normal parameterization.

use std::marker::PhantomData;

use gamlss_core::{
    CompilableFamily, Family, FixedDimensionalFamily, HasObservationDimension, Identity,
    InitialEtaFromTheta, Link, LocationCholesky, Log, ModelError, Nu, ObservationView,
    PositiveLink,
    shape::{Product, ShapeValues, Vector},
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

use crate::multivariate::{
    elliptical,
    matrix::FixedLowerTriangular,
    normal::{MvNormalCholesky, kernel},
};

#[cfg(feature = "rand")]
use super::try_sample_location_scale;
use super::{skew_gradient_terms, skew_nll};

/// Default-link multivariate skew-normal with Cholesky scale.
pub type MvSkewNormalCholeskyDefault<const D: usize> =
    MvSkewNormalCholesky<D, Identity, Log, Identity, Identity>;

/// Multivariate skew-normal parameterized by location, Cholesky scale and shape.
///
/// For `z = L^-1 (y - mu)`, the density is
/// `2 phi_D(z) Phi(alpha' z) / det(L)`. The shape vector `alpha` therefore acts
/// in whitened coordinate order. `mu` is a location vector, not the
/// distribution mean when `alpha != 0`, and `L L'` is the Gaussian kernel
/// scale rather than the skew-normal covariance.
///
/// This form is useful for jointly skewed load, price, return and weather-error
/// vectors while retaining an unconstrained, fit-ready dependence
/// parameterization. It does not expose a joint or marginal CDF: those require
/// multivariate normal probability algorithms not used by the likelihood.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MvSkewNormalCholesky<
    const D: usize,
    MuLink = Identity,
    DiagonalLink = Log,
    OffDiagonalLink = Identity,
    ShapeLink = Identity,
> {
    marker: PhantomData<(MuLink, DiagonalLink, OffDiagonalLink, ShapeLink)>,
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink, ShapeLink>
    MvSkewNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, ShapeLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
    ShapeLink: Link<f64>,
{
    /// Creates a stateless family after checking the compile-time dimension.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `D == 0`.
    pub const fn try_new() -> Result<Self, ModelError> {
        if D == 0 {
            return Err(ModelError::InvalidParameter {
                parameter: "dimension",
                expected: "positive",
            });
        }
        Ok(Self {
            marker: PhantomData,
        })
    }

    /// Creates a stateless family.
    ///
    /// # Panics
    ///
    /// Panics when `D == 0`.
    #[must_use]
    pub const fn new() -> Self {
        assert!(D > 0, "multivariate skew-normal dimension must be positive");
        Self {
            marker: PhantomData,
        }
    }

    fn theta_from_eta(eta: &MvSkewNormalCholeskyEta<D>) -> MvSkewNormalCholeskyTheta<D> {
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
        MvSkewNormalCholeskyTheta::from_parts_unchecked(mu, cholesky, shape)
    }

    fn nan_eta() -> MvSkewNormalCholeskyEta<D> {
        MvSkewNormalCholeskyEta {
            mu: [f64::NAN; D],
            cholesky: FixedLowerTriangular::from_lower_rows([[f64::NAN; D]; D]),
            shape: [f64::NAN; D],
        }
    }

    fn nll_theta(observation: [f64; D], theta: &MvSkewNormalCholeskyTheta<D>) -> f64 {
        if !valid_theta(theta) {
            return f64::INFINITY;
        }
        let mut z = [0.0; D];
        let gaussian_nll = kernel::nll(D, &observation, &theta.mu, &theta.cholesky, &mut z);
        skew_nll(gaussian_nll, &z, &theta.shape)
    }

    fn nll_and_gradient_eta_values(
        observation: [f64; D],
        eta: &MvSkewNormalCholeskyEta<D>,
    ) -> (f64, MvSkewNormalCholeskyEta<D>) {
        let theta = Self::theta_from_eta(eta);
        if !valid_theta(&theta) {
            return (f64::INFINITY, Self::nan_eta());
        }

        let mut z = [0.0; D];
        let gaussian_nll = kernel::nll(D, &observation, &theta.mu, &theta.cholesky, &mut z);
        let Some(terms) = skew_gradient_terms(gaussian_nll, &z, &theta.shape) else {
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

        let mut gradient = MvSkewNormalCholeskyEta {
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
    for MvSkewNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, ShapeLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
    ShapeLink: Link<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink, ShapeLink> Family
    for MvSkewNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, ShapeLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
    ShapeLink: Link<f64>,
{
    type Eta = MvSkewNormalCholeskyEta<D>;
    type Theta = MvSkewNormalCholeskyTheta<D>;
    type GradientEta = MvSkewNormalCholeskyEta<D>;
    type Observation<'obs> = [f64; D];
    type Workspace = ();

    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut ()) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    fn nll(&self, observation: [f64; D], theta: &Self::Theta, _workspace: &mut ()) -> f64 {
        Self::nll_theta(observation, theta)
    }

    fn nll_eta(&self, observation: [f64; D], eta: &Self::Eta, _workspace: &mut ()) -> f64 {
        Self::nll_theta(observation, &Self::theta_from_eta(eta))
    }

    fn nll_and_gradient_eta(
        &self,
        observation: [f64; D],
        eta: &Self::Eta,
        _workspace: &mut (),
    ) -> (f64, Self::GradientEta) {
        Self::nll_and_gradient_eta_values(observation, eta)
    }
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink, ShapeLink> FixedDimensionalFamily<D>
    for MvSkewNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, ShapeLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
    ShapeLink: Link<f64>,
{
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink, ShapeLink> HasObservationDimension
    for MvSkewNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, ShapeLink>
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
    for MvSkewNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, ShapeLink>
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
        if !valid_theta(theta) {
            return Err(SimulationError::InvalidParameters(
                "multivariate skew-normal theta",
            ));
        }

        try_sample_location_scale(rng, theta.mu, &theta.cholesky, &theta.shape)
    }
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink, ShapeLink> CompilableFamily
    for MvSkewNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, ShapeLink>
where
    MuLink: InitialEtaFromTheta<f64>,
    DiagonalLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    OffDiagonalLink: InitialEtaFromTheta<f64>,
    ShapeLink: InitialEtaFromTheta<f64>,
{
    type Shape = Product<LocationCholesky<D>, Vector<Nu, D>>;

    fn eta_from_shape(values: ShapeValues<Self::Shape>) -> Self::Eta {
        MvSkewNormalCholeskyEta::new(
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
        let normal = MvNormalCholesky::<D, MuLink, DiagonalLink, OffDiagonalLink>::new();
        (
            normal.initial_shape(obs),
            [ShapeLink::initial_eta_from_theta(0.0); D],
        )
    }
}

/// Link-scale predictors for multivariate skew-normal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MvSkewNormalCholeskyEta<const D: usize> {
    mu: [f64; D],
    cholesky: FixedLowerTriangular<D>,
    shape: [f64; D],
}

impl<const D: usize> MvSkewNormalCholeskyEta<D> {
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

    /// Mutable shape predictors.
    #[must_use]
    pub const fn shape_mut(&mut self) -> &mut [f64; D] {
        &mut self.shape
    }
}

/// Natural-scale multivariate skew-normal parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MvSkewNormalCholeskyTheta<const D: usize> {
    mu: [f64; D],
    cholesky: FixedLowerTriangular<D>,
    shape: [f64; D],
}

impl<const D: usize> MvSkewNormalCholeskyTheta<D> {
    /// Creates checked natural-scale parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] unless `D > 0`, location and
    /// shape entries are finite, and the Cholesky factor is finite with a
    /// strictly positive diagonal.
    pub fn try_new(
        mu: [f64; D],
        cholesky: FixedLowerTriangular<D>,
        shape: [f64; D],
    ) -> Result<Self, ModelError> {
        let theta = Self {
            mu,
            cholesky,
            shape,
        };
        if valid_theta(&theta) {
            Ok(theta)
        } else {
            Err(ModelError::InvalidParameter {
                parameter: "multivariate skew-normal theta",
                expected: "positive dimension, finite values, and positive Cholesky diagonal",
            })
        }
    }

    const fn from_parts_unchecked(
        mu: [f64; D],
        cholesky: FixedLowerTriangular<D>,
        shape: [f64; D],
    ) -> Self {
        Self {
            mu,
            cholesky,
            shape,
        }
    }

    /// Location vector.
    #[must_use]
    pub const fn mu(&self) -> &[f64; D] {
        &self.mu
    }

    /// Cholesky factor of the Gaussian kernel scale.
    #[must_use]
    pub const fn cholesky(&self) -> &FixedLowerTriangular<D> {
        &self.cholesky
    }

    /// Whitened-coordinate shape vector.
    #[must_use]
    pub const fn shape(&self) -> &[f64; D] {
        &self.shape
    }
}

fn valid_theta<const D: usize>(theta: &MvSkewNormalCholeskyTheta<D>) -> bool {
    elliptical::valid_location_scale(D, &theta.mu, &theta.cholesky)
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

    use super::{MvSkewNormalCholeskyDefault, MvSkewNormalCholeskyEta, MvSkewNormalCholeskyTheta};
    use crate::{
        SkewNormalMuSigmaNu, SkewNormalTheta,
        multivariate::{matrix::FixedLowerTriangular, normal::MvNormalCholeskyDefault},
    };

    #[test]
    fn checked_constructor_rejects_zero_dimension() {
        assert_eq!(
            MvSkewNormalCholeskyDefault::<0>::try_new(),
            Err(ModelError::InvalidParameter {
                parameter: "dimension",
                expected: "positive",
            })
        );
    }

    #[test]
    fn zero_shape_matches_multivariate_normal() {
        let skew = MvSkewNormalCholeskyDefault::<2>::new();
        let normal = MvNormalCholeskyDefault::<2>::new();
        let eta = MvSkewNormalCholeskyEta::new(
            [0.2, -0.4],
            FixedLowerTriangular::from_lower_rows([[0.1, 0.0], [0.3, -0.2]]),
            [0.0, 0.0],
        );
        let normal_eta =
            crate::multivariate::normal::MvNormalCholeskyEta::new(*eta.mu(), *eta.cholesky());
        let observation = [0.8, -0.1];
        assert_relative_eq!(
            skew.nll_eta(observation, &eta, &mut ()),
            normal.nll_eta(observation, &normal_eta, &mut ()),
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn one_dimensional_case_matches_scalar_skew_normal() {
        let multivariate = MvSkewNormalCholeskyDefault::<1>::new();
        let scalar = SkewNormalMuSigmaNu::new();
        let theta = MvSkewNormalCholeskyTheta::try_new(
            [0.4],
            FixedLowerTriangular::from_lower_rows([[0.8]]),
            [1.3],
        )
        .unwrap();
        let scalar_theta = SkewNormalTheta {
            mu: 0.4,
            sigma: 0.8,
            nu: 1.3,
        };
        assert_relative_eq!(
            multivariate.nll([1.1], &theta, &mut ()),
            scalar.nll(1.1, &scalar_theta, &mut ()),
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn analytic_gradient_matches_finite_difference() {
        let family = MvSkewNormalCholeskyDefault::<2>::new();
        let eta = MvSkewNormalCholeskyEta::new(
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
                let mut plus = eta;
                let current = plus.cholesky().get(row, col).unwrap();
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
    fn invalid_domains_are_rejected() {
        assert!(
            MvSkewNormalCholeskyTheta::<0>::try_new([], FixedLowerTriangular::zeros(), [],)
                .is_err()
        );
        assert!(
            MvSkewNormalCholeskyTheta::try_new(
                [0.0, 0.0],
                FixedLowerTriangular::from_lower_rows([[1.0, 0.0], [0.0, 0.0]]),
                [0.0, 0.0],
            )
            .is_err()
        );
        assert!(
            MvSkewNormalCholeskyTheta::try_new(
                [0.0, 0.0],
                FixedLowerTriangular::from_lower_rows([[1.0, 0.0], [0.0, 1.0]]),
                [f64::NAN, 0.0],
            )
            .is_err()
        );
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
            MvSkewNormalCholeskyDefault::<2>::new(),
            ParameterBlocks::new((mu, cholesky, shape)),
            response.as_slice(),
        )
        .unwrap();
        let beta = [0.1, -0.2, 0.0, 0.1, -0.1, 0.5, -0.3];
        let mut gradient = [0.0; 7];
        let value = model.try_value_gradient_into(&beta, &mut gradient).unwrap();
        assert!(value.is_finite());
        assert!(gradient.iter().all(|score| score.is_finite()));
    }

    #[cfg(feature = "rand")]
    #[test]
    fn sampling_returns_finite_vectors() {
        use gamlss_core::TrySimulate;
        use rand::SeedableRng;

        let family = MvSkewNormalCholeskyDefault::<2>::new();
        let theta = MvSkewNormalCholeskyTheta::try_new(
            [0.2, -0.1],
            FixedLowerTriangular::from_lower_rows([[1.0, 0.0], [0.3, 0.8]]),
            [1.2, -0.7],
        )
        .unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(31);
        assert!(
            family
                .try_sample(&mut rng, &theta)
                .is_ok_and(|sample| sample.iter().all(|value| value.is_finite()))
        );
    }
}
