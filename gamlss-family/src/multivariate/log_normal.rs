#![allow(clippy::needless_range_loop, clippy::suboptimal_flops)]

use std::marker::PhantomData;

use gamlss_core::{
    CholeskyScale, CompilableFamily, Family, FixedDimensionalFamily, HasConditionalCdf,
    HasMarginalCdf, HasObservationDimension, HasRosenblattTransform, Identity, InitialEtaFromTheta,
    Link, Log, LogLocation, ModelError, ObservationView, PositiveLink,
    shape::{Lower, Product, ShapeValues, Vector},
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};
use gamlss_special::unit_normal_cdf;

use crate::multivariate::{matrix::FixedLowerTriangular, normal::kernel};

/// Default-link multivariate log-normal with a Cholesky factor on the log scale.
pub type MvLogNormalCholeskyDefault<const D: usize> =
    MvLogNormalCholesky<D, Identity, Log, Identity>;

/// Multivariate log-normal parameterized by log-location and log-scale Cholesky factor.
///
/// `log(Y)` follows a multivariate normal distribution with location `m` and
/// covariance `L L'`. The likelihood on the original positive response scale
/// includes the Jacobian `sum(log(y_i))`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MvLogNormalCholesky<
    const D: usize,
    LogLocationLink = Identity,
    DiagonalLink = Log,
    OffDiagonalLink = Identity,
> {
    marker: PhantomData<(LogLocationLink, DiagonalLink, OffDiagonalLink)>,
}

impl<const D: usize, LogLocationLink, DiagonalLink, OffDiagonalLink>
    MvLogNormalCholesky<D, LogLocationLink, DiagonalLink, OffDiagonalLink>
where
    LogLocationLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
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
        assert!(D > 0, "multivariate log-normal dimension must be positive");
        Self {
            marker: PhantomData,
        }
    }

    fn theta_from_eta(eta: MvLogNormalCholeskyEta<D>) -> MvLogNormalCholeskyTheta<D> {
        let log_location = eta.log_location.map(LogLocationLink::inverse);
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
                    .expect("valid lower index");
            }
        }
        MvLogNormalCholeskyTheta::from_parts_unchecked(log_location, cholesky)
    }

    fn nan_eta() -> MvLogNormalCholeskyEta<D> {
        MvLogNormalCholeskyEta {
            log_location: [f64::NAN; D],
            cholesky: FixedLowerTriangular::from_lower_rows([[f64::NAN; D]; D]),
        }
    }

    fn nll_theta(observation: [f64; D], theta: &MvLogNormalCholeskyTheta<D>) -> f64 {
        let Some((logged, log_jacobian)) = log_observation(observation) else {
            return f64::INFINITY;
        };
        let mut z = [0.0; D];
        kernel::nll(D, &logged, &theta.log_location, &theta.cholesky, &mut z) + log_jacobian
    }

    fn nll_and_gradient_eta_values(
        observation: [f64; D],
        eta: MvLogNormalCholeskyEta<D>,
    ) -> (f64, MvLogNormalCholeskyEta<D>) {
        let Some((logged, log_jacobian)) = log_observation(observation) else {
            return (f64::INFINITY, Self::nan_eta());
        };
        let theta = Self::theta_from_eta(eta);
        let mut z = [0.0; D];
        let mut a = [0.0; D];
        let normal_nll = kernel::nll_and_score(
            D,
            &logged,
            &theta.log_location,
            &theta.cholesky,
            &mut z,
            &mut a,
        );
        let nll = normal_nll + log_jacobian;
        if !nll.is_finite() {
            return (nll, Self::nan_eta());
        }

        let mut gradient = MvLogNormalCholeskyEta {
            log_location: [0.0; D],
            cholesky: FixedLowerTriangular::zeros(),
        };
        for component in 0..D {
            gradient.log_location[component] =
                -a[component] * LogLocationLink::derivative_inverse(eta.log_location[component]);
        }
        for row in 0..D {
            for col in 0..=row {
                let eta_value = eta.cholesky.lower(row, col);
                let score = if row == col {
                    DiagonalLink::derivative_log_inverse(eta_value)
                        - a[row] * z[col] * DiagonalLink::derivative_inverse(eta_value)
                } else {
                    kernel::cholesky_score(row, col, z[col], &a, &theta.cholesky)
                        * OffDiagonalLink::derivative_inverse(eta_value)
                };
                gradient
                    .cholesky
                    .set_lower(row, col, score)
                    .expect("valid lower index");
            }
        }
        (nll, gradient)
    }
}

impl<const D: usize, LogLocationLink, DiagonalLink, OffDiagonalLink> Default
    for MvLogNormalCholesky<D, LogLocationLink, DiagonalLink, OffDiagonalLink>
where
    LogLocationLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize, LogLocationLink, DiagonalLink, OffDiagonalLink> Family
    for MvLogNormalCholesky<D, LogLocationLink, DiagonalLink, OffDiagonalLink>
where
    LogLocationLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    type Observation<'obs> = [f64; D];
    type Eta = MvLogNormalCholeskyEta<D>;
    type Theta = MvLogNormalCholeskyTheta<D>;
    type GradientEta = MvLogNormalCholeskyEta<D>;
    type Workspace = ();

    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut ()) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    fn nll(
        &self,
        observation: Self::Observation<'_>,
        theta: &Self::Theta,
        _workspace: &mut (),
    ) -> f64 {
        Self::nll_theta(observation, theta)
    }

    fn nll_and_gradient_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        _workspace: &mut (),
    ) -> (f64, Self::GradientEta) {
        Self::nll_and_gradient_eta_values(observation, *eta)
    }
}

impl<const D: usize, LogLocationLink, DiagonalLink, OffDiagonalLink> FixedDimensionalFamily<D>
    for MvLogNormalCholesky<D, LogLocationLink, DiagonalLink, OffDiagonalLink>
where
    LogLocationLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
}

impl<const D: usize, LogLocationLink, DiagonalLink, OffDiagonalLink> HasObservationDimension
    for MvLogNormalCholesky<D, LogLocationLink, DiagonalLink, OffDiagonalLink>
where
    LogLocationLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    fn observation_dimension(&self) -> usize {
        D
    }
}

impl<const D: usize, LogLocationLink, DiagonalLink, OffDiagonalLink> HasMarginalCdf
    for MvLogNormalCholesky<D, LogLocationLink, DiagonalLink, OffDiagonalLink>
where
    LogLocationLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    fn marginal_cdf(&self, component: usize, y: f64, theta: &Self::Theta) -> f64 {
        if component >= D || !y.is_finite() || !valid_theta(theta) {
            return f64::NAN;
        }
        if y <= 0.0 {
            return 0.0;
        }
        let scale = kernel::marginal_scale(D, component, &theta.log_location, &theta.cholesky);
        unit_normal_cdf((y.ln() - theta.log_location[component]) / scale)
    }
}

impl<const D: usize, LogLocationLink, DiagonalLink, OffDiagonalLink> HasConditionalCdf
    for MvLogNormalCholesky<D, LogLocationLink, DiagonalLink, OffDiagonalLink>
where
    LogLocationLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    fn conditional_cdf(
        &self,
        component: usize,
        y: f64,
        preceding: &[f64],
        theta: &Self::Theta,
    ) -> f64 {
        if component >= D
            || preceding.len() < component
            || preceding
                .iter()
                .take(component)
                .any(|value| !value.is_finite() || *value <= 0.0)
            || !valid_theta(theta)
            || !y.is_finite()
        {
            return f64::NAN;
        }
        if y <= 0.0 {
            return 0.0;
        }
        let mut logged_preceding = [0.0; D];
        for index in 0..component {
            logged_preceding[index] = preceding[index].ln();
        }
        let mut standardized = [0.0; D];
        kernel::conditional_cdf(
            D,
            component,
            y.ln(),
            &logged_preceding,
            &theta.log_location,
            &theta.cholesky,
            &mut standardized,
        )
    }
}

impl<const D: usize, LogLocationLink, DiagonalLink, OffDiagonalLink> HasRosenblattTransform
    for MvLogNormalCholesky<D, LogLocationLink, DiagonalLink, OffDiagonalLink>
where
    LogLocationLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    fn rosenblatt_into(
        &self,
        observation: Self::Observation<'_>,
        theta: &Self::Theta,
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        if out.len() != D {
            return Err(ModelError::ResponseLength {
                expected: D,
                actual: out.len(),
            });
        }
        let Some((logged, _)) = log_observation(observation) else {
            out.fill(f64::NAN);
            return Ok(());
        };
        kernel::rosenblatt_into(D, &logged, &theta.log_location, &theta.cholesky, out);
        Ok(())
    }
}

#[cfg(feature = "rand")]
impl<Rng, const D: usize, LogLocationLink, DiagonalLink, OffDiagonalLink> TrySimulate<Rng>
    for MvLogNormalCholesky<D, LogLocationLink, DiagonalLink, OffDiagonalLink>
where
    Rng: rand::Rng,
    LogLocationLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    type Sample = [f64; D];

    fn try_sample(
        &self,
        rng: &mut Rng,
        theta: &Self::Theta,
    ) -> Result<Self::Sample, SimulationError> {
        if !valid_theta(theta) {
            return Err(SimulationError::InvalidParameters(
                "multivariate log-normal theta",
            ));
        }
        let normal = rand_distr::StandardNormal;
        let z: [f64; D] = std::array::from_fn(|_| rand_distr::Distribution::sample(&normal, rng));
        let mut out = theta.log_location;
        for row in 0..D {
            for col in 0..=row {
                out[row] += theta.cholesky.lower(row, col) * z[col];
            }
            out[row] = out[row].exp();
        }
        if out.iter().all(|value| value.is_finite() && *value > 0.0) {
            Ok(out)
        } else {
            Err(SimulationError::NumericalFailure(
                "multivariate log-normal exponential transform",
            ))
        }
    }
}

impl<const D: usize, LogLocationLink, DiagonalLink, OffDiagonalLink> CompilableFamily
    for MvLogNormalCholesky<D, LogLocationLink, DiagonalLink, OffDiagonalLink>
where
    LogLocationLink: InitialEtaFromTheta<f64>,
    DiagonalLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    OffDiagonalLink: InitialEtaFromTheta<f64>,
{
    type Shape = Product<Vector<LogLocation, D>, Lower<CholeskyScale, D>>;

    fn eta_from_shape(values: ShapeValues<Self::Shape>) -> Self::Eta {
        MvLogNormalCholeskyEta::new(values.0, FixedLowerTriangular::from_lower_rows(values.1))
    }

    fn gradient_to_shape(gradient: &Self::GradientEta) -> ShapeValues<Self::Shape> {
        let lower = std::array::from_fn(|row| {
            std::array::from_fn(|col| gradient.cholesky.get(row, col).unwrap_or(0.0))
        });
        (gradient.log_location, lower)
    }

    fn initial_shape<'obs, Obs>(&self, obs: &'obs Obs) -> ShapeValues<Self::Shape>
    where
        Obs: ObservationView<'obs, Observation = [f64; D]> + 'obs,
    {
        let mut weight_sum = 0.0;
        let mut means = [0.0; D];
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            let observation = obs.observation_at(row);
            if weight > 0.0 && valid_observation(&observation) {
                weight_sum += weight;
                for component in 0..D {
                    means[component] += weight * observation[component].ln();
                }
            }
        }
        if weight_sum > 0.0 {
            for value in &mut means {
                *value /= weight_sum;
            }
        }
        let mut variance = [0.0; D];
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            let observation = obs.observation_at(row);
            if weight > 0.0 && valid_observation(&observation) {
                for component in 0..D {
                    variance[component] +=
                        weight * (observation[component].ln() - means[component]).powi(2);
                }
            }
        }
        let mut lower = [[OffDiagonalLink::initial_eta_from_theta(0.0); D]; D];
        for component in 0..D {
            let scale = if weight_sum > 0.0 {
                (variance[component] / weight_sum).sqrt().max(1.0e-6)
            } else {
                1.0
            };
            lower[component][component] = DiagonalLink::initial_eta_from_theta(scale);
        }
        (means.map(LogLocationLink::initial_eta_from_theta), lower)
    }

    fn validate_compiled(&self) -> Result<(), ModelError> {
        Self::try_new().map(|_| ())
    }
}

/// Link-scale predictors for multivariate log-normal Cholesky parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MvLogNormalCholeskyEta<const D: usize> {
    log_location: [f64; D],
    cholesky: FixedLowerTriangular<D>,
}

impl<const D: usize> MvLogNormalCholeskyEta<D> {
    /// Creates fixed-dimensional predictors.
    #[must_use]
    pub const fn new(log_location: [f64; D], cholesky: FixedLowerTriangular<D>) -> Self {
        Self {
            log_location,
            cholesky,
        }
    }

    /// Log-location predictors.
    #[must_use]
    pub const fn log_location(&self) -> &[f64; D] {
        &self.log_location
    }

    /// Log-scale Cholesky predictors.
    #[must_use]
    pub const fn cholesky(&self) -> &FixedLowerTriangular<D> {
        &self.cholesky
    }

    /// Mutably borrows log-location predictors.
    #[must_use]
    pub const fn log_location_mut(&mut self) -> &mut [f64; D] {
        &mut self.log_location
    }

    /// Mutably borrows log-scale Cholesky predictors.
    #[must_use]
    pub const fn cholesky_mut(&mut self) -> &mut FixedLowerTriangular<D> {
        &mut self.cholesky
    }
}

/// Natural-scale parameters of `log(Y)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MvLogNormalCholeskyTheta<const D: usize> {
    log_location: [f64; D],
    cholesky: FixedLowerTriangular<D>,
}

impl<const D: usize> MvLogNormalCholeskyTheta<D> {
    /// Creates checked log-location and Cholesky parameters.
    pub fn try_new(
        log_location: [f64; D],
        cholesky: FixedLowerTriangular<D>,
    ) -> Result<Self, ModelError> {
        let theta = Self {
            log_location,
            cholesky,
        };
        if valid_theta(&theta) {
            Ok(theta)
        } else {
            Err(ModelError::InvalidParameter {
                parameter: "multivariate log-normal theta",
                expected: "positive dimension, finite log locations and a finite Cholesky factor with positive diagonal",
            })
        }
    }

    const fn from_parts_unchecked(
        log_location: [f64; D],
        cholesky: FixedLowerTriangular<D>,
    ) -> Self {
        Self {
            log_location,
            cholesky,
        }
    }

    /// Location vector of `log(Y)`.
    #[must_use]
    pub const fn log_location(&self) -> &[f64; D] {
        &self.log_location
    }

    /// Cholesky factor of `Cov(log(Y))`.
    #[must_use]
    pub const fn cholesky(&self) -> &FixedLowerTriangular<D> {
        &self.cholesky
    }

    /// Returns one covariance entry of `log(Y)`.
    #[must_use]
    pub fn log_covariance(&self, row: usize, col: usize) -> Option<f64> {
        if row >= D || col >= D {
            return None;
        }
        Some(
            (0..=row.min(col))
                .map(|index| self.cholesky.lower(row, index) * self.cholesky.lower(col, index))
                .sum(),
        )
    }

    /// Returns the marginal mean of one original-scale component.
    #[must_use]
    pub fn mean(&self, component: usize) -> Option<f64> {
        self.log_covariance(component, component)
            .map(|variance| (self.log_location[component] + 0.5 * variance).exp())
    }

    /// Returns one original-scale covariance entry.
    #[must_use]
    pub fn covariance(&self, row: usize, col: usize) -> Option<f64> {
        let log_covariance = self.log_covariance(row, col)?;
        let mean_product = self.mean(row)? * self.mean(col)?;
        Some(mean_product * log_covariance.exp_m1())
    }
}

fn valid_observation<const D: usize>(observation: &[f64; D]) -> bool {
    D > 0
        && observation
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
}

fn log_observation<const D: usize>(observation: [f64; D]) -> Option<([f64; D], f64)> {
    if !valid_observation(&observation) {
        return None;
    }
    let logged = observation.map(f64::ln);
    Some((logged, logged.iter().sum()))
}

fn valid_theta<const D: usize>(theta: &MvLogNormalCholeskyTheta<D>) -> bool {
    kernel::valid_theta(D, &theta.log_location, &theta.cholesky)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use approx::assert_relative_eq;
    use gamlss_core::{Family, HasConditionalCdf, HasMarginalCdf, HasRosenblattTransform};

    use super::{MvLogNormalCholeskyDefault, MvLogNormalCholeskyEta, MvLogNormalCholeskyTheta};
    use crate::{
        LogNormalLogLocationLogSd, LogNormalLogLocationLogSdTheta,
        multivariate::matrix::FixedLowerTriangular,
    };

    #[test]
    fn one_dimension_matches_scalar_log_normal() {
        let family = MvLogNormalCholeskyDefault::<1>::new();
        let theta = MvLogNormalCholeskyTheta::try_new(
            [0.4],
            FixedLowerTriangular::from_lower_rows([[0.8]]),
        )
        .unwrap();
        let scalar = LogNormalLogLocationLogSd::new();
        let scalar_theta = LogNormalLogLocationLogSdTheta {
            log_location: 0.4,
            log_sd: 0.8,
        };
        assert_relative_eq!(
            family.nll([1.7], &theta, &mut ()),
            scalar.nll(1.7, &scalar_theta, &mut ()),
            epsilon = 1.0e-14
        );
        assert_relative_eq!(
            family.marginal_cdf(0, 1.7, &theta),
            gamlss_core::HasCdf::cdf(&scalar, 1.7, &scalar_theta),
            epsilon = 1.0e-14
        );
    }

    #[test]
    fn multivariate_density_is_normal_density_plus_log_jacobian() {
        use crate::multivariate::normal::{MvNormalCholeskyDefault, MvNormalCholeskyTheta};

        let family = MvLogNormalCholeskyDefault::<2>::new();
        let cholesky = FixedLowerTriangular::from_lower_rows([[0.8, 0.0], [0.25, 0.7]]);
        let theta = MvLogNormalCholeskyTheta::try_new([0.2, -0.3], cholesky).unwrap();
        let normal = MvNormalCholeskyDefault::<2>::new();
        let normal_theta = MvNormalCholeskyTheta::try_new([0.2, -0.3], cholesky).unwrap();
        let y = [1.4_f64, 0.7];
        let logged = y.map(f64::ln);
        let expected = normal.nll(logged, &normal_theta, &mut ()) + logged.iter().sum::<f64>();
        assert_relative_eq!(family.nll(y, &theta, &mut ()), expected, epsilon = 1.0e-14);
    }

    #[test]
    fn analytic_gradient_matches_finite_difference() {
        let family = MvLogNormalCholeskyDefault::<2>::new();
        let eta = MvLogNormalCholeskyEta::new(
            [0.2, -0.3],
            FixedLowerTriangular::from_lower_rows([[-0.1, 0.0], [0.25, 0.2]]),
        );
        let y = [1.4, 0.7];
        let (_, gradient) = family.nll_and_gradient_eta(y, &eta, &mut ());
        for component in 0..2 {
            let mut lower = eta;
            let mut upper = eta;
            lower.log_location_mut()[component] -= 1.0e-6;
            upper.log_location_mut()[component] += 1.0e-6;
            let numeric =
                (family.nll_eta(y, &upper, &mut ()) - family.nll_eta(y, &lower, &mut ())) / 2.0e-6;
            assert_relative_eq!(
                gradient.log_location()[component],
                numeric,
                epsilon = 1.0e-6
            );
        }
        for row in 0..2 {
            for col in 0..=row {
                let mut lower = eta;
                let mut upper = eta;
                let value = lower.cholesky().get(row, col).unwrap();
                lower
                    .cholesky_mut()
                    .set_lower(row, col, value - 1.0e-6)
                    .unwrap();
                upper
                    .cholesky_mut()
                    .set_lower(row, col, value + 1.0e-6)
                    .unwrap();
                let numeric = (family.nll_eta(y, &upper, &mut ())
                    - family.nll_eta(y, &lower, &mut ()))
                    / 2.0e-6;
                assert_relative_eq!(
                    gradient.cholesky().get(row, col).unwrap(),
                    numeric,
                    epsilon = 1.0e-6
                );
            }
        }
    }

    #[test]
    fn transformed_diagnostics_match_underlying_normal_ordering() {
        let family = MvLogNormalCholeskyDefault::<2>::new();
        let theta = MvLogNormalCholeskyTheta::try_new(
            [0.2, -0.3],
            FixedLowerTriangular::from_lower_rows([[0.8, 0.0], [0.25, 0.7]]),
        )
        .unwrap();
        let y = [1.4, 0.7];
        let mut transform = [0.0; 2];
        family.rosenblatt_into(y, &theta, &mut transform).unwrap();
        assert_relative_eq!(
            transform[0],
            family.marginal_cdf(0, y[0], &theta),
            epsilon = 1.0e-14
        );
        assert_relative_eq!(
            transform[1],
            family.conditional_cdf(1, y[1], &y, &theta),
            epsilon = 1.0e-14
        );
        assert_eq!(family.conditional_cdf(1, 0.0, &y, &theta), 0.0);
    }

    #[test]
    fn moments_and_domains_are_checked() {
        let family = MvLogNormalCholeskyDefault::<2>::new();
        let theta = MvLogNormalCholeskyTheta::try_new(
            [0.2, -0.3],
            FixedLowerTriangular::from_lower_rows([[0.8, 0.0], [0.25, 0.7]]),
        )
        .unwrap();
        assert!(theta.mean(0).unwrap() > 0.0);
        assert!(theta.covariance(0, 0).unwrap() > 0.0);
        assert!(family.nll([0.0, 1.0], &theta, &mut ()).is_infinite());
        assert!(family.marginal_cdf(2, 1.0, &theta).is_nan());
        assert!(MvLogNormalCholeskyDefault::<0>::try_new().is_err());
    }

    #[cfg(feature = "rand")]
    #[test]
    fn sampling_is_strictly_positive() {
        use gamlss_core::TrySimulate;
        use rand::SeedableRng;

        let family = MvLogNormalCholeskyDefault::<2>::new();
        let theta = MvLogNormalCholeskyTheta::try_new(
            [0.2, -0.3],
            FixedLowerTriangular::from_lower_rows([[0.8, 0.0], [0.25, 0.7]]),
        )
        .unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        let sample = family.try_sample(&mut rng, &theta).unwrap();
        assert!(sample.iter().all(|value| value.is_finite() && *value > 0.0));
    }
}
