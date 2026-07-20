#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::suboptimal_flops
)]

//! Mean/SD/partial-correlation multivariate power-exponential parameterization.

use std::marker::PhantomData;

use gamlss_core::{
    CompilableFamily, Family, FixedDimensionalFamily, HasObservationDimension, Identity,
    InitialEtaFromTheta, Link, Log, ModelError, Mu, ObservationView, PartialCorrelation,
    PositiveLink, Power, Sigma,
    shape::{Product, Scalar, ShapeValues, StrictLower, Vector},
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

use crate::multivariate::{
    correlation::{
        FixedPartialCorrelations, correlation_cholesky_from_partial, covariance_from_cholesky,
        partial_corr_from_eta, partial_corr_gradient_from_cholesky_score,
        scale_cholesky_from_correlation,
    },
    elliptical, initial,
    matrix::FixedLowerTriangular,
};

use super::{
    d_log_covariance_multiplier_d_power, direct_power_score, log_covariance_multiplier,
    radial_nll_constant,
};

/// Default-link multivariate power-exponential with explicit marginal SDs.
pub type MvPowerExponentialMeanStdPartialCorrDefault<const D: usize> =
    MvPowerExponentialMeanStdPartialCorr<D, Identity, Log, Log>;

/// Multivariate power-exponential with mean, marginal SD, partial correlation and power.
///
/// The modeled covariance is `diag(sigma) R diag(sigma)`. Internally, the
/// Cholesky scatter factor is rescaled by the power-dependent radial second
/// moment, so `sigma` remains the actual response standard deviation for every
/// positive power. Ordered partial correlations are mapped through `tanh` and
/// construct a valid correlation matrix without projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MvPowerExponentialMeanStdPartialCorr<
    const D: usize,
    MuLink = Identity,
    SigmaLink = Log,
    PowerLink = Log,
> {
    marker: PhantomData<(MuLink, SigmaLink, PowerLink)>,
}

impl<const D: usize, MuLink, SigmaLink, PowerLink>
    MvPowerExponentialMeanStdPartialCorr<D, MuLink, SigmaLink, PowerLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    PowerLink: PositiveLink<f64>,
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
        assert!(
            D > 0,
            "multivariate power-exponential dimension must be positive"
        );
        Self {
            marker: PhantomData,
        }
    }

    fn theta_from_eta(
        eta: &MvPowerExponentialMeanStdPartialCorrEta<D>,
    ) -> MvPowerExponentialMeanStdPartialCorrTheta<D> {
        let mu = eta.mu.map(MuLink::inverse);
        let sigma = eta.sigma.map(SigmaLink::inverse);
        let power = PowerLink::inverse(eta.power);
        let partial_corr = partial_corr_from_eta(&eta.partial_corr);
        MvPowerExponentialMeanStdPartialCorrTheta::from_canonical_parts(
            mu,
            sigma,
            partial_corr,
            power,
        )
    }

    const fn nan_eta() -> MvPowerExponentialMeanStdPartialCorrEta<D> {
        MvPowerExponentialMeanStdPartialCorrEta {
            mu: [f64::NAN; D],
            sigma: [f64::NAN; D],
            partial_corr: FixedPartialCorrelations::filled_strict_lower(f64::NAN),
            power: f64::NAN,
        }
    }

    fn nll_theta(
        observation: [f64; D],
        theta: &MvPowerExponentialMeanStdPartialCorrTheta<D>,
    ) -> f64 {
        if !valid_theta(theta) {
            return f64::INFINITY;
        }
        let mut z = [0.0; D];
        let Some((quadratic, log_det_scale)) =
            elliptical::standardize(D, &observation, &theta.mu, &theta.scatter_cholesky, &mut z)
        else {
            return f64::INFINITY;
        };
        let radial_power = quadratic.powf(0.5 * theta.power);
        let nll = log_det_scale + radial_nll_constant(D as f64, theta.power) + 0.5 * radial_power;
        if nll.is_finite() { nll } else { f64::INFINITY }
    }

    fn nll_and_gradient_eta_values(
        observation: [f64; D],
        eta: &MvPowerExponentialMeanStdPartialCorrEta<D>,
    ) -> (f64, MvPowerExponentialMeanStdPartialCorrEta<D>) {
        let theta = Self::theta_from_eta(eta);
        if !valid_theta(&theta) {
            return (f64::INFINITY, Self::nan_eta());
        }
        let mut z = [0.0; D];
        let Some((quadratic, log_det_scale)) =
            elliptical::standardize(D, &observation, &theta.mu, &theta.scatter_cholesky, &mut z)
        else {
            return (f64::INFINITY, Self::nan_eta());
        };
        let radial_power = quadratic.powf(0.5 * theta.power);
        let nll = log_det_scale + radial_nll_constant(D as f64, theta.power) + 0.5 * radial_power;
        if !nll.is_finite() {
            return (f64::INFINITY, Self::nan_eta());
        }

        let radial_weight = if quadratic == 0.0 {
            0.0
        } else {
            0.5 * theta.power * quadratic.powf(0.5 * theta.power - 1.0)
        };
        let standardized_gradient = z.map(|standardized| radial_weight * standardized);
        let mut location_score = [0.0; D];
        if !elliptical::transpose_solve(
            D,
            &theta.scatter_cholesky,
            &standardized_gradient,
            &mut location_score,
        ) {
            return (f64::INFINITY, Self::nan_eta());
        }

        let mut gradient = zero_eta();
        let mut correlation_score = [[0.0; D]; D];
        for row in 0..D {
            gradient.mu[row] = -location_score[row] * MuLink::derivative_inverse(eta.mu[row]);
            let residual_score = (0..=row)
                .map(|col| {
                    -location_score[row]
                        * z[col]
                        * theta.correlation_cholesky.lower(row, col)
                        * theta.scatter_per_standard_deviation
                })
                .sum::<f64>();
            gradient.sigma[row] = SigmaLink::derivative_log_inverse(eta.sigma[row])
                + residual_score * SigmaLink::derivative_inverse(eta.sigma[row]);

            for col in 0..=row {
                correlation_score[row][col] = -theta.scatter_per_standard_deviation
                    * theta.sigma[row]
                    * location_score[row]
                    * z[col];
                if row == col {
                    correlation_score[row][col] += 1.0 / theta.correlation_cholesky.lower(row, row);
                }
            }
        }
        gradient.partial_corr = partial_corr_gradient_from_cholesky_score(
            &eta.partial_corr,
            &theta.correlation_cholesky,
            &correlation_score,
        );

        let dimension = D as f64;
        let scale_score = dimension - radial_weight * quadratic;
        let covariance_chain =
            -0.5 * d_log_covariance_multiplier_d_power(dimension, theta.power) * scale_score;
        gradient.power = (direct_power_score(dimension, theta.power, quadratic, radial_power)
            + covariance_chain)
            * PowerLink::derivative_inverse(eta.power);
        (nll, gradient)
    }
}

impl<const D: usize, MuLink, SigmaLink, PowerLink> Default
    for MvPowerExponentialMeanStdPartialCorr<D, MuLink, SigmaLink, PowerLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    PowerLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize, MuLink, SigmaLink, PowerLink> Family
    for MvPowerExponentialMeanStdPartialCorr<D, MuLink, SigmaLink, PowerLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    PowerLink: PositiveLink<f64>,
{
    type Eta = MvPowerExponentialMeanStdPartialCorrEta<D>;
    type Theta = MvPowerExponentialMeanStdPartialCorrTheta<D>;
    type GradientEta = MvPowerExponentialMeanStdPartialCorrEta<D>;
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

impl<const D: usize, MuLink, SigmaLink, PowerLink> FixedDimensionalFamily<D>
    for MvPowerExponentialMeanStdPartialCorr<D, MuLink, SigmaLink, PowerLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    PowerLink: PositiveLink<f64>,
{
}

impl<const D: usize, MuLink, SigmaLink, PowerLink> HasObservationDimension
    for MvPowerExponentialMeanStdPartialCorr<D, MuLink, SigmaLink, PowerLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    PowerLink: PositiveLink<f64>,
{
    fn observation_dimension(&self) -> usize {
        D
    }
}

#[cfg(feature = "rand")]
impl<Rng, const D: usize, MuLink, SigmaLink, PowerLink> TrySimulate<Rng>
    for MvPowerExponentialMeanStdPartialCorr<D, MuLink, SigmaLink, PowerLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    PowerLink: PositiveLink<f64>,
{
    type Sample = [f64; D];

    fn try_sample(
        &self,
        rng: &mut Rng,
        theta: &Self::Theta,
    ) -> Result<Self::Sample, SimulationError> {
        if !valid_theta(theta) {
            return Err(SimulationError::InvalidParameters(
                "MV power-exponential mean/SD/partial-correlation theta",
            ));
        }
        let dimension = D as f64;
        let gamma = rand_distr::Gamma::new(dimension / theta.power, 1.0)
            .map_err(|_| SimulationError::BackendRejected("power-exponential radial shape"))?;
        let radius = (2.0 * rand_distr::Distribution::sample(&gamma, rng)).powf(1.0 / theta.power);
        let normal = rand_distr::StandardNormal;
        let direction: [f64; D] =
            std::array::from_fn(|_| rand_distr::Distribution::sample(&normal, rng));
        let direction_norm = direction
            .iter()
            .fold(0.0_f64, |norm, value| norm.hypot(*value));
        if !radius.is_finite() || direction_norm == 0.0 || !direction_norm.is_finite() {
            return Err(SimulationError::NumericalFailure(
                "power-exponential radial transform",
            ));
        }
        let standardized = direction.map(|value| radius * value / direction_norm);
        let mut out = theta.mu;
        for row in 0..D {
            for col in 0..=row {
                out[row] += theta.scatter_cholesky.lower(row, col) * standardized[col];
            }
        }
        if out.iter().all(|value| value.is_finite()) {
            Ok(out)
        } else {
            Err(SimulationError::NumericalFailure(
                "MV power-exponential mean/SD transform",
            ))
        }
    }
}

impl<const D: usize, MuLink, SigmaLink, PowerLink> CompilableFamily
    for MvPowerExponentialMeanStdPartialCorr<D, MuLink, SigmaLink, PowerLink>
where
    MuLink: InitialEtaFromTheta<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    PowerLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Shape = Product<
        Product<Product<Vector<Mu, D>, Vector<Sigma, D>>, StrictLower<PartialCorrelation, D>>,
        Scalar<Power>,
    >;

    fn eta_from_shape(values: ShapeValues<Self::Shape>) -> Self::Eta {
        MvPowerExponentialMeanStdPartialCorrEta::new(
            values.0.0.0,
            values.0.0.1,
            FixedPartialCorrelations::from_lower_rows(values.0.1),
            values.1,
        )
    }

    fn gradient_to_shape(gradient: &Self::GradientEta) -> ShapeValues<Self::Shape> {
        (
            (
                (gradient.mu, gradient.sigma),
                gradient.partial_corr.lower_rows(),
            ),
            gradient.power,
        )
    }

    fn initial_shape<'obs, Obs>(&self, obs: &'obs Obs) -> ShapeValues<Self::Shape>
    where
        Obs: ObservationView<'obs, Observation = [f64; D]> + 'obs,
    {
        let (mu, sigma, partial_corr) =
            initial::location_scale_partial_correlation::<D, MuLink, SigmaLink, Obs>(obs);
        (
            ((mu, sigma), partial_corr),
            PowerLink::initial_eta_from_theta(2.0),
        )
    }

    fn validate_compiled(&self) -> Result<(), ModelError> {
        Self::try_new().map(|_| ())
    }
}

/// Link-scale predictors for mean/SD/partial-correlation power-exponential.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MvPowerExponentialMeanStdPartialCorrEta<const D: usize> {
    /// Mean predictors.
    pub mu: [f64; D],
    /// Marginal standard-deviation predictors.
    pub sigma: [f64; D],
    /// Ordered strict-lower partial-correlation predictors.
    pub partial_corr: FixedPartialCorrelations<D>,
    /// Positive radial-power predictor.
    pub power: f64,
}

impl<const D: usize> MvPowerExponentialMeanStdPartialCorrEta<D> {
    /// Creates link-scale predictors.
    #[must_use]
    pub const fn new(
        mu: [f64; D],
        sigma: [f64; D],
        partial_corr: FixedPartialCorrelations<D>,
        power: f64,
    ) -> Self {
        Self {
            mu,
            sigma,
            partial_corr,
            power,
        }
    }
}

/// Natural-scale mean/SD/partial-correlation power-exponential parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MvPowerExponentialMeanStdPartialCorrTheta<const D: usize> {
    mu: [f64; D],
    sigma: [f64; D],
    partial_corr: FixedPartialCorrelations<D>,
    correlation_cholesky: FixedLowerTriangular<D>,
    scatter_cholesky: FixedLowerTriangular<D>,
    scatter_per_standard_deviation: f64,
    power: f64,
}

impl<const D: usize> MvPowerExponentialMeanStdPartialCorrTheta<D> {
    /// Creates checked natural-scale parameters and their derived scatter geometry.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] for zero dimension, non-finite
    /// means, non-positive SDs or power, partial correlations outside `(-1, 1)`,
    /// or an unrepresentable derived scatter factor.
    pub fn try_new(
        mu: [f64; D],
        sigma: [f64; D],
        partial_corr: FixedPartialCorrelations<D>,
        power: f64,
    ) -> Result<Self, ModelError> {
        let theta = Self::from_canonical_parts(mu, sigma, partial_corr, power);
        if valid_theta(&theta) {
            Ok(theta)
        } else {
            Err(ModelError::InvalidParameter {
                parameter: "MV power-exponential mean/SD/partial-correlation theta",
                expected: "positive dimension, finite means, positive SDs and power, partial correlations in (-1, 1), and representable scatter geometry",
            })
        }
    }

    fn from_canonical_parts(
        mu: [f64; D],
        sigma: [f64; D],
        partial_corr: FixedPartialCorrelations<D>,
        power: f64,
    ) -> Self {
        let correlation_cholesky = correlation_cholesky_from_partial(&partial_corr);
        let covariance_cholesky = scale_cholesky_from_correlation(&sigma, &correlation_cholesky);
        let scatter_per_standard_deviation =
            (-0.5 * log_covariance_multiplier(D as f64, power)).exp();
        let mut scatter_cholesky = FixedLowerTriangular::zeros();
        for row in 0..D {
            for col in 0..=row {
                scatter_cholesky
                    .set_lower(
                        row,
                        col,
                        scatter_per_standard_deviation * covariance_cholesky.lower(row, col),
                    )
                    .expect("valid lower-triangular index");
            }
        }
        Self {
            mu,
            sigma,
            partial_corr,
            correlation_cholesky,
            scatter_cholesky,
            scatter_per_standard_deviation,
            power,
        }
    }

    /// Mean vector.
    #[must_use]
    pub const fn mu(&self) -> &[f64; D] {
        &self.mu
    }

    /// Marginal standard deviations.
    #[must_use]
    pub const fn sigma(&self) -> &[f64; D] {
        &self.sigma
    }

    /// Ordered partial correlations.
    #[must_use]
    pub const fn partial_corr(&self) -> &FixedPartialCorrelations<D> {
        &self.partial_corr
    }

    /// Correlation Cholesky factor.
    #[must_use]
    pub const fn correlation_cholesky(&self) -> &FixedLowerTriangular<D> {
        &self.correlation_cholesky
    }

    /// Internal power-exponential scatter Cholesky factor.
    #[must_use]
    pub const fn scatter_cholesky(&self) -> &FixedLowerTriangular<D> {
        &self.scatter_cholesky
    }

    /// Positive radial power.
    #[must_use]
    pub const fn power(&self) -> f64 {
        self.power
    }

    /// Returns one covariance entry from `diag(sigma) R diag(sigma)`.
    #[must_use]
    pub fn covariance(&self, row: usize, col: usize) -> Option<f64> {
        covariance_from_cholesky(&self.correlation_cholesky, row, col)
            .map(|correlation| self.sigma[row] * self.sigma[col] * correlation)
    }
}

fn valid_theta<const D: usize>(theta: &MvPowerExponentialMeanStdPartialCorrTheta<D>) -> bool {
    D > 0
        && theta.power > 0.0
        && theta.power.is_finite()
        && theta.mu.iter().all(|value| value.is_finite())
        && theta
            .sigma
            .iter()
            .all(|value| *value > 0.0 && value.is_finite())
        && theta
            .partial_corr
            .iter()
            .all(|value| value.is_finite() && value.abs() < 1.0)
        && theta.scatter_per_standard_deviation > 0.0
        && theta.scatter_per_standard_deviation.is_finite()
        && elliptical::valid_location_scale(D, &theta.mu, &theta.scatter_cholesky)
}

fn zero_eta<const D: usize>() -> MvPowerExponentialMeanStdPartialCorrEta<D> {
    MvPowerExponentialMeanStdPartialCorrEta {
        mu: [0.0; D],
        sigma: [0.0; D],
        partial_corr: FixedPartialCorrelations::zeros(),
        power: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{
        DenseDesign, Family, Gamlss, LinearPredictorBlock, ModelError, Mu, NoPenalty,
        ParameterBlock, ParameterBlocks, PartialCorrelation, Power, Sigma,
        StrictLowerTriangularParameterBlock, VectorParameterBlock,
    };

    use super::{
        MvPowerExponentialMeanStdPartialCorrDefault, MvPowerExponentialMeanStdPartialCorrEta,
        MvPowerExponentialMeanStdPartialCorrTheta,
    };
    use crate::multivariate::{
        FixedPartialCorrelations,
        power_exponential::{MvPowerExponentialCholeskyDefault, MvPowerExponentialCholeskyTheta},
    };

    #[test]
    fn checked_constructors_reject_invalid_domains() {
        assert_eq!(
            MvPowerExponentialMeanStdPartialCorrDefault::<0>::try_new(),
            Err(ModelError::InvalidParameter {
                parameter: "dimension",
                expected: "positive",
            })
        );
        assert!(
            MvPowerExponentialMeanStdPartialCorrTheta::<0>::try_new(
                [],
                [],
                FixedPartialCorrelations::zeros(),
                2.0,
            )
            .is_err()
        );
        assert!(
            MvPowerExponentialMeanStdPartialCorrTheta::try_new(
                [0.0; 2],
                [1.0, 0.0],
                FixedPartialCorrelations::zeros(),
                2.0,
            )
            .is_err()
        );
        assert!(
            MvPowerExponentialMeanStdPartialCorrTheta::try_new(
                [0.0; 2],
                [1.0; 2],
                FixedPartialCorrelations::try_new(vec![1.0]).unwrap(),
                2.0,
            )
            .is_err()
        );
        assert!(
            MvPowerExponentialMeanStdPartialCorrTheta::try_new(
                [0.0; 2],
                [1.0; 2],
                FixedPartialCorrelations::zeros(),
                0.0,
            )
            .is_err()
        );
    }

    #[test]
    fn canonical_scatter_form_has_identical_likelihood() {
        let family = MvPowerExponentialMeanStdPartialCorrDefault::<2>::new();
        let theta = MvPowerExponentialMeanStdPartialCorrTheta::try_new(
            [0.2, -0.3],
            [1.2, 0.8],
            FixedPartialCorrelations::try_new(vec![0.35]).unwrap(),
            1.4,
        )
        .unwrap();
        let canonical = MvPowerExponentialCholeskyDefault::<2>::new();
        let canonical_theta = MvPowerExponentialCholeskyTheta::try_new(
            *theta.mu(),
            *theta.scatter_cholesky(),
            theta.power(),
        )
        .unwrap();
        let observation = [0.7, -0.1];
        assert_relative_eq!(
            family.nll(observation, &theta, &mut ()),
            canonical.nll(observation, &canonical_theta, &mut ()),
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn covariance_uses_modeled_standard_deviations_and_correlation() {
        let theta = MvPowerExponentialMeanStdPartialCorrTheta::try_new(
            [0.0; 2],
            [2.0, 3.0],
            FixedPartialCorrelations::try_new(vec![0.4]).unwrap(),
            0.8,
        )
        .unwrap();
        assert_relative_eq!(theta.covariance(0, 0).unwrap(), 4.0, epsilon = 1.0e-12);
        assert_relative_eq!(theta.covariance(1, 1).unwrap(), 9.0, epsilon = 1.0e-12);
        assert_relative_eq!(theta.covariance(1, 0).unwrap(), 2.4, epsilon = 1.0e-12);
    }

    #[test]
    fn normal_power_uses_covariance_cholesky_without_radial_rescaling() {
        let theta = MvPowerExponentialMeanStdPartialCorrTheta::try_new(
            [0.0; 2],
            [2.0, 3.0],
            FixedPartialCorrelations::try_new(vec![0.4]).unwrap(),
            2.0,
        )
        .unwrap();
        assert_relative_eq!(
            theta.scatter_cholesky().get(0, 0).unwrap(),
            2.0,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            theta.scatter_cholesky().get(1, 0).unwrap(),
            1.2,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            theta.scatter_cholesky().get(1, 1).unwrap(),
            3.0 * (1.0_f64 - 0.4_f64.powi(2)).sqrt(),
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn eta_gradient_matches_finite_difference() {
        let family = MvPowerExponentialMeanStdPartialCorrDefault::<2>::new();
        let eta = MvPowerExponentialMeanStdPartialCorrEta::new(
            [0.1, -0.2],
            [0.2, -0.1],
            FixedPartialCorrelations::try_new(vec![0.3]).unwrap(),
            1.4_f64.ln(),
        );
        let observation = [0.8, -0.5];
        let (_, gradient) = family.nll_and_gradient_eta(observation, &eta, &mut ());
        let epsilon = 1.0e-6;
        let mut coordinates = [
            eta.mu[0],
            eta.mu[1],
            eta.sigma[0],
            eta.sigma[1],
            0.3,
            eta.power,
        ];
        let actual = [
            gradient.mu[0],
            gradient.mu[1],
            gradient.sigma[0],
            gradient.sigma[1],
            gradient.partial_corr.get(1, 0).unwrap(),
            gradient.power,
        ];
        for index in 0..coordinates.len() {
            let original = coordinates[index];
            coordinates[index] = original + epsilon;
            let plus = eta_from_coordinates(coordinates);
            coordinates[index] = original - epsilon;
            let minus = eta_from_coordinates(coordinates);
            coordinates[index] = original;
            let finite_difference = (family.nll_eta(observation, &plus, &mut ())
                - family.nll_eta(observation, &minus, &mut ()))
                / (2.0 * epsilon);
            assert_relative_eq!(actual[index], finite_difference, epsilon = 2.0e-6);
        }
    }

    #[test]
    fn static_shape_is_fit_ready() {
        let response = [[0.2, -0.3], [0.8, 0.4], [-0.5, 0.7]];
        let rows = response.len();
        let intercept = || LinearPredictorBlock::new(DenseDesign::intercept(rows));
        let mu = VectorParameterBlock::<Mu, 2, _, _>::new([intercept(), intercept()], NoPenalty, 0);
        let sigma =
            VectorParameterBlock::<Sigma, 2, _, _>::new([intercept(), intercept()], NoPenalty, 0);
        let partial = StrictLowerTriangularParameterBlock::<PartialCorrelation, 2, _, _>::new(
            vec![intercept()],
            NoPenalty,
            0,
        );
        let power = ParameterBlock::<Power, _, _>::new(intercept(), NoPenalty, 0);
        let model = Gamlss::try_new_with_observations(
            MvPowerExponentialMeanStdPartialCorrDefault::<2>::new(),
            ParameterBlocks::new(((mu, sigma, partial), power)),
            response.as_slice(),
        )
        .unwrap();
        let beta: [f64; 6] = [0.1, -0.2, 0.0, -0.1, 0.2, 2.0_f64.ln()];
        let mut gradient: [f64; 6] = [0.0; 6];
        let value = model.try_value_gradient_into(&beta, &mut gradient).unwrap();
        assert!(value.is_finite());
        assert!(gradient.iter().all(|score| score.is_finite()));
        assert_eq!(
            model.parameter_layout().unique_slice("sigma").unwrap(),
            Some(2..4)
        );
        assert_eq!(
            model.parameter_layout().unique_slice("power").unwrap(),
            Some(5..6)
        );
    }

    #[cfg(feature = "rand")]
    #[test]
    fn sampling_returns_finite_vectors() {
        use gamlss_core::TrySimulate;
        use rand::SeedableRng;

        let family = MvPowerExponentialMeanStdPartialCorrDefault::<2>::new();
        let theta = MvPowerExponentialMeanStdPartialCorrTheta::try_new(
            [0.2, -0.1],
            [1.0, 0.8],
            FixedPartialCorrelations::try_new(vec![0.3]).unwrap(),
            1.4,
        )
        .unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(37);
        assert!(
            family
                .try_sample(&mut rng, &theta)
                .is_ok_and(|sample| sample.iter().all(|value| value.is_finite()))
        );
    }

    fn eta_from_coordinates(values: [f64; 6]) -> MvPowerExponentialMeanStdPartialCorrEta<2> {
        MvPowerExponentialMeanStdPartialCorrEta::new(
            [values[0], values[1]],
            [values[2], values[3]],
            FixedPartialCorrelations::try_new(vec![values[4]]).unwrap(),
            values[5],
        )
    }
}
