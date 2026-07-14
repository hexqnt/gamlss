#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::suboptimal_flops
)]

use std::marker::PhantomData;

use gamlss_core::{
    CompilableFamily, Family, FixedDimensionalFamily, HasMarginalCdf, HasObservationDimension,
    Identity, InitialEtaFromTheta, Link, LocationCholesky, Log, LogPlus, ModelError,
    ObservationView, PositiveLink, Tau,
    shape::{Product, Scalar, ShapeValues},
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};
use gamlss_special::{digamma, ln_gamma, student_t_cdf_standardized};

use crate::multivariate::matrix::FixedLowerTriangular;

/// Default-link generic multivariate Student-t with Cholesky scale.
pub type MvStudentTCholeskyDefault<const D: usize> =
    MvStudentTCholesky<D, Identity, Log, Identity, LogPlus<2>>;

/// Generic multivariate Student-t with a Cholesky scale factor and shared degrees of freedom.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MvStudentTCholesky<
    const D: usize,
    MuLink = Identity,
    DiagonalLink = Log,
    OffDiagonalLink = Identity,
    TauLink = LogPlus<2>,
> {
    marker: PhantomData<(MuLink, DiagonalLink, OffDiagonalLink, TauLink)>,
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink, TauLink>
    MvStudentTCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, TauLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
    TauLink: Link<f64>,
{
    /// Creates a stateless family value.
    ///
    /// # Panics
    ///
    /// Panics when the compile-time dimension is zero.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        assert!(D > 0, "multivariate Student-t dimension must be positive");
        Self {
            marker: PhantomData,
        }
    }

    fn theta_from_eta(eta: MvStudentTCholeskyEta<D>) -> MvStudentTCholeskyTheta<D> {
        let mut mu = [0.0; D];
        let mut cholesky = FixedLowerTriangular::zeros();
        for (mu, eta_mu) in mu.iter_mut().zip(eta.mu) {
            *mu = MuLink::inverse(eta_mu);
        }
        for row in 0..D {
            for col in 0..=row {
                let eta_value = eta.cholesky.get(row, col).expect("valid lower index");
                let value = if row == col {
                    DiagonalLink::inverse(eta_value)
                } else {
                    OffDiagonalLink::inverse(eta_value)
                };
                cholesky
                    .set_lower(row, col, value)
                    .expect("valid lower index");
            }
        }
        MvStudentTCholeskyTheta::from_parts_unchecked(mu, cholesky, TauLink::inverse(eta.tau))
    }

    fn nll_theta(observation: [f64; D], theta: MvStudentTCholeskyTheta<D>) -> f64 {
        let mut z = [0.0; D];
        nll_cholesky(observation, &theta, &mut z)
    }

    fn nan_eta() -> MvStudentTCholeskyEta<D> {
        MvStudentTCholeskyEta {
            mu: [f64::NAN; D],
            cholesky: FixedLowerTriangular::from_lower_rows([[f64::NAN; D]; D]),
            tau: f64::NAN,
        }
    }

    fn nll_and_gradient_eta_values(
        observation: [f64; D],
        eta: MvStudentTCholeskyEta<D>,
    ) -> (f64, MvStudentTCholeskyEta<D>) {
        let theta = Self::theta_from_eta(eta);
        let mut z = [0.0; D];
        let nll = nll_cholesky(observation, &theta, &mut z);
        if !nll.is_finite() {
            return (nll, Self::nan_eta());
        }

        let mut a = [0.0; D];
        transpose_solve(&theta.cholesky, &z, &mut a);
        let quadratic = z.iter().map(|value| value * value).sum::<f64>();
        let d = D as f64;
        let robust_weight = (theta.tau + d) / (theta.tau + quadratic);

        let mut gradient = MvStudentTCholeskyEta {
            mu: [0.0; D],
            cholesky: FixedLowerTriangular::zeros(),
            tau: 0.0,
        };

        for component in 0..D {
            gradient.mu[component] =
                -robust_weight * a[component] * MuLink::derivative_inverse(eta.mu[component]);
        }

        for row in 0..D {
            for col in 0..=row {
                let eta_value = lower(&eta.cholesky, row, col);
                let d_nll_d_l = if row == col {
                    DiagonalLink::derivative_log_inverse(eta_value)
                        - robust_weight
                            * a[row]
                            * z[col]
                            * DiagonalLink::derivative_inverse(eta_value)
                } else {
                    -robust_weight
                        * a[row]
                        * z[col]
                        * OffDiagonalLink::derivative_inverse(eta_value)
                };
                gradient
                    .cholesky
                    .set_lower(row, col, d_nll_d_l)
                    .expect("valid lower index");
            }
        }

        let d_nll_d_tau = 0.5 * digamma(0.5 * theta.tau)
            - 0.5 * digamma(f64::midpoint(theta.tau, d))
            + 0.5 * d / theta.tau
            + 0.5 * (quadratic / theta.tau).ln_1p()
            - f64::midpoint(theta.tau, d) * quadratic / (theta.tau * (theta.tau + quadratic));
        gradient.tau = d_nll_d_tau * TauLink::derivative_inverse(eta.tau);

        (nll, gradient)
    }
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink, TauLink> Default
    for MvStudentTCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, TauLink>
where
    MuLink: InitialEtaFromTheta<f64>,
    DiagonalLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    OffDiagonalLink: InitialEtaFromTheta<f64>,
    TauLink: InitialEtaFromTheta<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink, TauLink> Family
    for MvStudentTCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, TauLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
    TauLink: Link<f64>,
{
    type Eta = MvStudentTCholeskyEta<D>;
    type Theta = MvStudentTCholeskyTheta<D>;
    type GradientEta = MvStudentTCholeskyEta<D>;
    type Observation<'obs> = [f64; D];
    type Workspace = ();

    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    #[inline]
    fn nll(
        &self,
        observation: Self::Observation<'_>,
        theta: &Self::Theta,
        _workspace: &mut Self::Workspace,
    ) -> f64 {
        Self::nll_theta(observation, *theta)
    }

    #[inline]
    fn nll_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> f64 {
        Self::nll_theta(observation, Self::theta_from_eta(*eta))
    }

    #[inline]
    fn nll_and_gradient_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        Self::nll_and_gradient_eta_values(observation, *eta)
    }
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink, TauLink> FixedDimensionalFamily<D>
    for MvStudentTCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, TauLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
    TauLink: Link<f64>,
{
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink, TauLink> HasObservationDimension
    for MvStudentTCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, TauLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
    TauLink: Link<f64>,
{
    fn observation_dimension(&self) -> usize {
        D
    }
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink, TauLink> HasMarginalCdf
    for MvStudentTCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, TauLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
    TauLink: Link<f64>,
{
    fn marginal_cdf(&self, component: usize, y: f64, theta: &Self::Theta) -> f64 {
        if !y.is_finite() || component >= D || !valid_theta(theta) {
            return f64::NAN;
        }
        let Some(scale) = theta.marginal_scale(component) else {
            return f64::NAN;
        };
        student_t_cdf_standardized((y - theta.mu[component]) / scale, theta.tau)
    }
}

#[cfg(feature = "rand")]
impl<Rng, const D: usize, MuLink, DiagonalLink, OffDiagonalLink, TauLink> TrySimulate<Rng>
    for MvStudentTCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, TauLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
    TauLink: Link<f64>,
{
    type Sample = [f64; D];

    fn try_sample(
        &self,
        rng: &mut Rng,
        theta: &Self::Theta,
    ) -> Result<Self::Sample, SimulationError> {
        if !valid_theta(theta) {
            return Err(SimulationError::InvalidParameters(
                "multivariate Student-t theta",
            ));
        }
        let normal = rand_distr::StandardNormal;
        let Ok(chi_squared) = rand_distr::ChiSquared::new(theta.tau) else {
            return Err(SimulationError::BackendRejected(
                "Student-t degrees of freedom",
            ));
        };
        let scale = (theta.tau / rand_distr::Distribution::sample(&chi_squared, rng)).sqrt();
        let mut z = [0.0; D];
        for value in &mut z {
            *value = rand_distr::Distribution::sample(&normal, rng);
        }
        let mut out = theta.mu;
        for row in 0..D {
            for col in 0..=row {
                out[row] += scale * theta.cholesky.get(row, col).unwrap() * z[col];
            }
        }
        Ok(out)
    }
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink, TauLink> CompilableFamily
    for MvStudentTCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, TauLink>
where
    MuLink: InitialEtaFromTheta<f64>,
    DiagonalLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    OffDiagonalLink: InitialEtaFromTheta<f64>,
    TauLink: InitialEtaFromTheta<f64>,
{
    type Shape = Product<LocationCholesky<D>, Scalar<Tau>>;

    fn eta_from_shape(values: ShapeValues<Self::Shape>) -> MvStudentTCholeskyEta<D> {
        MvStudentTCholeskyEta::new(
            values.0.0,
            FixedLowerTriangular::from_lower_rows(values.0.1),
            values.1,
        )
    }

    fn gradient_to_shape(gradient: &MvStudentTCholeskyEta<D>) -> ShapeValues<Self::Shape> {
        let lower = std::array::from_fn(|row| {
            std::array::from_fn(|col| gradient.cholesky.get(row, col).unwrap_or(0.0))
        });
        ((gradient.mu, lower), gradient.tau)
    }

    fn initial_shape<'obs, Obs>(&self, obs: &'obs Obs) -> ShapeValues<Self::Shape>
    where
        Obs: ObservationView<'obs, Observation = [f64; D]> + 'obs,
    {
        let mut weights = [0.0; D];
        let mut means = [0.0; D];
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            let value = obs.observation_at(row);
            for component in 0..D {
                if weight > 0.0 && value[component].is_finite() {
                    weights[component] += weight;
                    means[component] += weight * value[component];
                }
            }
        }
        for component in 0..D {
            if weights[component] > 0.0 {
                means[component] /= weights[component];
            }
        }
        let mut variances = [0.0; D];
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            let value = obs.observation_at(row);
            for component in 0..D {
                if weight > 0.0 && value[component].is_finite() {
                    variances[component] += weight * (value[component] - means[component]).powi(2);
                }
            }
        }
        let mut lower = [[OffDiagonalLink::initial_eta_from_theta(0.0); D]; D];
        for component in 0..D {
            let scale = if weights[component] > 0.0 {
                (variances[component] / weights[component])
                    .sqrt()
                    .max(1.0e-6)
            } else {
                1.0
            };
            lower[component][component] = DiagonalLink::initial_eta_from_theta(scale);
        }
        (
            (means.map(MuLink::initial_eta_from_theta), lower),
            TauLink::initial_eta_from_theta(10.0),
        )
    }
}

/// Link-scale predictors for generic multivariate Student-t.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MvStudentTCholeskyEta<const D: usize> {
    /// Location predictors.
    pub mu: [f64; D],
    /// Lower-triangular Cholesky scale predictors.
    pub cholesky: FixedLowerTriangular<D>,
    /// Degrees-of-freedom predictor.
    pub tau: f64,
}

impl<const D: usize> MvStudentTCholeskyEta<D> {
    /// Creates link-scale predictors.
    #[must_use]
    #[inline]
    pub const fn new(mu: [f64; D], cholesky: FixedLowerTriangular<D>, tau: f64) -> Self {
        Self { mu, cholesky, tau }
    }
}

/// Natural-scale parameters for generic multivariate Student-t.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MvStudentTCholeskyTheta<const D: usize> {
    /// Location vector.
    mu: [f64; D],
    /// Lower-triangular Cholesky scale factor.
    cholesky: FixedLowerTriangular<D>,
    /// Shared degrees of freedom.
    tau: f64,
}

impl<const D: usize> MvStudentTCholeskyTheta<D> {
    /// Creates checked natural-scale parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] unless the dimension is
    /// positive, all entries are finite, every Cholesky diagonal entry is
    /// strictly positive, and `tau` is finite and positive.
    #[inline]
    pub fn try_new(
        mu: [f64; D],
        cholesky: FixedLowerTriangular<D>,
        tau: f64,
    ) -> Result<Self, ModelError> {
        let theta = Self { mu, cholesky, tau };
        if !valid_theta(&theta) {
            return Err(ModelError::InvalidParameter {
                parameter: "multivariate Student-t theta",
                expected: "positive dimension, finite values, positive Cholesky diagonal, and positive tau",
            });
        }
        Ok(theta)
    }

    #[inline]
    const fn from_parts_unchecked(
        mu: [f64; D],
        cholesky: FixedLowerTriangular<D>,
        tau: f64,
    ) -> Self {
        Self { mu, cholesky, tau }
    }

    /// Location vector.
    #[must_use]
    #[inline]
    pub const fn mu(&self) -> &[f64; D] {
        &self.mu
    }

    /// Lower-triangular Cholesky scale factor.
    #[must_use]
    #[inline]
    pub const fn cholesky(&self) -> &FixedLowerTriangular<D> {
        &self.cholesky
    }

    /// Shared degrees of freedom.
    #[must_use]
    #[inline]
    pub const fn tau(&self) -> f64 {
        self.tau
    }

    /// Returns one covariance-scale entry from `L L'`.
    #[must_use]
    pub fn scale_covariance(&self, row: usize, col: usize) -> Option<f64> {
        if row >= D || col >= D {
            return None;
        }
        let limit = row.min(col);
        Some(
            (0..=limit)
                .map(|index| self.cholesky.lower(row, index) * self.cholesky.lower(col, index))
                .sum(),
        )
    }

    /// Returns the marginal scale of one component.
    #[must_use]
    pub fn marginal_scale(&self, component: usize) -> Option<f64> {
        self.scale_covariance(component, component).map(f64::sqrt)
    }
}

fn valid_theta<const D: usize>(theta: &MvStudentTCholeskyTheta<D>) -> bool {
    D > 0
        && theta.tau > 0.0
        && theta.tau.is_finite()
        && theta.mu.iter().all(|value| value.is_finite())
        && (0..D).all(|row| {
            (0..=row).all(|col| {
                let value = theta.cholesky.get(row, col).unwrap();
                value.is_finite() && (row != col || value > 0.0)
            })
        })
}

fn nll_cholesky<const D: usize>(
    observation: [f64; D],
    theta: &MvStudentTCholeskyTheta<D>,
    z: &mut [f64; D],
) -> f64 {
    if !valid_theta(theta) || !observation.iter().all(|value| value.is_finite()) {
        return f64::INFINITY;
    }
    for row in 0..D {
        let mut value = observation[row] - theta.mu[row];
        for col in 0..row {
            value -= theta.cholesky.get(row, col).unwrap() * z[col];
        }
        z[row] = value / theta.cholesky.get(row, row).unwrap();
    }
    let quadratic = z.iter().map(|value| value * value).sum::<f64>();
    let log_det_scale = (0..D)
        .map(|index| theta.cholesky.get(index, index).unwrap().ln())
        .sum::<f64>();
    let d = D as f64;
    log_det_scale
        + f64::midpoint(theta.tau, d) * (quadratic / theta.tau).ln_1p()
        + ln_gamma(0.5 * theta.tau)
        - ln_gamma(f64::midpoint(theta.tau, d))
        + 0.5 * d * (theta.tau * std::f64::consts::PI).ln()
}

fn transpose_solve<const D: usize>(
    cholesky: &FixedLowerTriangular<D>,
    rhs: &[f64; D],
    out: &mut [f64; D],
) {
    for row in (0..D).rev() {
        let mut value = rhs[row];
        for col in (row + 1)..D {
            value -= lower(cholesky, col, row) * out[col];
        }
        out[row] = value / lower(cholesky, row, row);
    }
}

fn lower<const D: usize>(cholesky: &FixedLowerTriangular<D>, row: usize, col: usize) -> f64 {
    cholesky.get(row, col).expect("valid lower index")
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{
        CholeskyScale, DenseDesign, Family, Gamlss, HasMarginalCdf, LinearPredictorBlock,
        LowerTriangularParameterBlock, Mu, NoPenalty, ParameterBlock, ParameterBlocks, Tau,
        VectorParameterBlock,
    };

    use super::{MvStudentTCholeskyDefault, MvStudentTCholeskyEta, MvStudentTCholeskyTheta};
    use crate::multivariate::matrix::FixedLowerTriangular;
    use crate::multivariate::normal::{MvNormalCholeskyDefault, MvNormalCholeskyTheta};
    use crate::{StudentTMuSigmaTau, StudentTMuSigmaTauTheta};

    #[test]
    fn one_dimensional_case_matches_scalar_student_t() {
        let mv = MvStudentTCholeskyDefault::<1>::new();
        let scalar = StudentTMuSigmaTau::new();
        let theta = MvStudentTCholeskyTheta::try_new(
            [0.4],
            FixedLowerTriangular::from_lower_rows([[0.8]]),
            5.0,
        )
        .unwrap();
        let scalar_theta = StudentTMuSigmaTauTheta {
            mu: 0.4,
            sigma: 0.8,
            tau: 5.0,
        };
        assert_relative_eq!(
            mv.nll([1.7], &theta, &mut mv.workspace()),
            scalar.nll(1.7, &scalar_theta, &mut scalar.workspace()),
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn large_tau_approaches_multivariate_normal() {
        let student = MvStudentTCholeskyDefault::<2>::new();
        let normal = MvNormalCholeskyDefault::<2>::new();
        let cholesky = FixedLowerTriangular::from_lower_rows([[1.2, 0.0], [0.3, 0.9]]);
        let y = [0.7, -0.2];
        let student_theta = MvStudentTCholeskyTheta::try_new([0.1, -0.5], cholesky, 1.0e8).unwrap();
        let normal_theta = MvNormalCholeskyTheta::try_new([0.1, -0.5], cholesky).unwrap();
        assert_relative_eq!(
            student.nll(y, &student_theta, &mut student.workspace()),
            normal.nll(y, &normal_theta, &mut normal.workspace()),
            epsilon = 1.0e-7
        );
    }

    #[test]
    fn analytic_gradient_matches_finite_difference() {
        let family = MvStudentTCholeskyDefault::<2>::new();
        let eta = MvStudentTCholeskyEta::new(
            [0.1, -0.2],
            FixedLowerTriangular::from_lower_rows([[0.0, 0.0], [0.2, -0.1]]),
            1.0,
        );
        let (nll, gradient) =
            family.nll_and_gradient_eta([0.7, -0.8], &eta, &mut family.workspace());
        assert!(nll.is_finite());

        for component in 0..2 {
            let mut plus = eta;
            plus.mu[component] += 1.0e-6;
            let mut minus = eta;
            minus.mu[component] -= 1.0e-6;
            let fd = (family.nll_eta([0.7, -0.8], &plus, &mut family.workspace())
                - family.nll_eta([0.7, -0.8], &minus, &mut family.workspace()))
                / 2.0e-6;
            assert_relative_eq!(gradient.mu[component], fd, epsilon = 1.0e-6);
        }

        for row in 0..2 {
            for col in 0..=row {
                let current = eta.cholesky.get(row, col).unwrap();
                let mut plus = eta;
                plus.cholesky.set_lower(row, col, current + 1.0e-6).unwrap();
                let mut minus = eta;
                minus
                    .cholesky
                    .set_lower(row, col, current - 1.0e-6)
                    .unwrap();
                let fd = (family.nll_eta([0.7, -0.8], &plus, &mut family.workspace())
                    - family.nll_eta([0.7, -0.8], &minus, &mut family.workspace()))
                    / 2.0e-6;
                assert_relative_eq!(
                    gradient.cholesky.get(row, col).unwrap(),
                    fd,
                    epsilon = 1.0e-6
                );
            }
        }

        let mut plus = eta;
        plus.tau += 1.0e-6;
        let mut minus = eta;
        minus.tau -= 1.0e-6;
        let fd = (family.nll_eta([0.7, -0.8], &plus, &mut family.workspace())
            - family.nll_eta([0.7, -0.8], &minus, &mut family.workspace()))
            / 2.0e-6;
        assert_relative_eq!(gradient.tau, fd, epsilon = 1.0e-6);
    }

    #[test]
    fn invalid_domains_are_rejected() {
        let family = MvStudentTCholeskyDefault::<2>::new();
        assert!(
            MvStudentTCholeskyTheta::try_new(
                [0.0, 0.0],
                FixedLowerTriangular::from_lower_rows([[1.0, 0.0], [0.0, 0.0]]),
                5.0,
            )
            .is_err()
        );
        assert!(
            MvStudentTCholeskyTheta::try_new(
                [f64::NAN, 0.0],
                FixedLowerTriangular::from_lower_rows([[1.0, 0.0], [0.0, 1.0]]),
                5.0,
            )
            .is_err()
        );
        assert!(
            MvStudentTCholeskyTheta::<0>::try_new([], FixedLowerTriangular::zeros(), 5.0,).is_err()
        );
        let invalid_theta = MvStudentTCholeskyTheta::from_parts_unchecked(
            [0.0, 0.0],
            FixedLowerTriangular::from_lower_rows([[1.0, 0.0], [0.0, 0.0]]),
            5.0,
        );
        assert!(
            family
                .nll([0.0, 0.0], &invalid_theta, &mut family.workspace())
                .is_infinite()
        );
        assert!(
            MvStudentTCholeskyTheta::try_new(
                [0.0, 0.0],
                FixedLowerTriangular::from_lower_rows([[1.0, 0.0], [0.0, 1.0]]),
                0.0,
            )
            .is_err()
        );
        let invalid_tau = MvStudentTCholeskyTheta::from_parts_unchecked(
            [0.0, 0.0],
            FixedLowerTriangular::from_lower_rows([[1.0, 0.0], [0.0, 1.0]]),
            0.0,
        );
        assert!(
            family
                .nll([0.0, 0.0], &invalid_tau, &mut family.workspace())
                .is_infinite()
        );
    }

    #[test]
    fn marginal_cdf_uses_component_scale() {
        let family = MvStudentTCholeskyDefault::<2>::new();
        let theta = MvStudentTCholeskyTheta::try_new(
            [0.0, 1.0],
            FixedLowerTriangular::from_lower_rows([[2.0, 0.0], [3.0, 4.0]]),
            7.0,
        )
        .unwrap();
        assert_relative_eq!(family.marginal_cdf(0, 0.0, &theta), 0.5, epsilon = 1.0e-12);
        assert_eq!(theta.scale_covariance(1, 1), Some(25.0));
    }

    #[test]
    fn compiled_blocks_are_fit_ready() {
        let y = [[0.2, -0.3], [1.0, 0.4], [-0.5, 0.8]];
        let n = y.len();
        let mu = VectorParameterBlock::<Mu, 2, _, _>::new(
            [
                LinearPredictorBlock::new(DenseDesign::intercept(n)),
                LinearPredictorBlock::new(DenseDesign::intercept(n)),
            ],
            NoPenalty,
            99,
        );
        let cholesky = LowerTriangularParameterBlock::<CholeskyScale, 2, _, _>::new(
            vec![
                LinearPredictorBlock::new(DenseDesign::intercept(n)),
                LinearPredictorBlock::new(DenseDesign::intercept(n)),
                LinearPredictorBlock::new(DenseDesign::intercept(n)),
            ],
            NoPenalty,
            99,
        );
        let tau = ParameterBlock::<Tau, _, _>::linear(DenseDesign::intercept(n), NoPenalty, 99);
        let blocks = ParameterBlocks::new((mu, cholesky, tau));
        let model = Gamlss::try_new_with_observations(
            MvStudentTCholeskyDefault::<2>::new(),
            blocks,
            y.as_slice(),
        )
        .unwrap();
        let beta = vec![0.1, -0.2, 0.0, 0.2, -0.1, 1.0];
        let eta = model.predict_eta_row(&beta, 0).unwrap();

        assert_eq!(model.nparams(), 6);
        assert_relative_eq!(eta.mu[0], 0.1);
        assert_relative_eq!(eta.cholesky.get(1, 0).unwrap(), 0.2);
        assert_relative_eq!(eta.tau, 1.0);

        let mut gradient = vec![0.0; beta.len()];
        model.try_value_gradient_into(&beta, &mut gradient).unwrap();
        for index in 0..beta.len() {
            let mut plus = beta.clone();
            plus[index] += 1.0e-6;
            let mut minus = beta.clone();
            minus[index] -= 1.0e-6;
            let fd = (model.try_value(&plus).unwrap() - model.try_value(&minus).unwrap()) / 2.0e-6;
            assert_relative_eq!(gradient[index], fd, epsilon = 1.0e-6);
        }
    }
}
