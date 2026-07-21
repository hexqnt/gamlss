#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::suboptimal_flops
)]

//! Multivariate Poisson distribution induced by one shared latent shock.

use std::marker::PhantomData;

use gamlss_core::{
    CompilableFamily, Family, FixedDimensionalFamily, HasMarginalCdf, HasObservationDimension,
    IdiosyncraticRate, InitialEtaFromTheta, Log, ModelError, ObservationView, PositiveLink,
    SharedRate,
    shape::{Product, Scalar, ShapeValues, Vector},
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};
use gamlss_special::{included_count, is_nonnegative_integer, ln_gamma, log_add_exp};

use crate::{initial::positive_floor, univariate::poisson::PoissonKernel};

/// Default-link common-shock multivariate Poisson family.
pub type MvPoissonCommonShockDefault<const D: usize> = MvPoissonCommonShock<D, Log, Log>;

#[derive(Debug, Clone, Copy)]
struct LatentPosteriorSummary {
    log_kernel_sum: f64,
    shared_count_mean: f64,
}

#[derive(Debug, Clone, Copy)]
struct CountMomentSummary<const D: usize> {
    marginal_mean: [f64; D],
    average_pair_covariance: Option<f64>,
}

/// Multivariate Poisson model with one common latent count.
///
/// The construction is
/// `Y_i = X_i + Z`, where independent `X_i ~ Poisson(lambda_i)` and
/// `Z ~ Poisson(lambda_shared)`. Consequently,
/// `E[Y_i] = Var[Y_i] = lambda_i + lambda_shared` and every distinct pair has
/// covariance `lambda_shared`. This is a compact model for positive synchronous
/// count dependence, such as coincident outages, price jumps, weather events or
/// threshold exceedances.
///
/// The joint likelihood sums exactly over the latent shared count from zero to
/// `min(Y_i)`. To keep one extreme observation from causing unbounded work,
/// observations requiring more than 1,000,001 terms are rejected with infinite
/// NLL; the exact bound is exposed as [`Self::MAX_EXACT_SHARED_COUNT`]. The model
/// cannot express negative covariance or pair-specific shared shocks; those are
/// structural limitations of this family, not limitations of the surrounding
/// family architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MvPoissonCommonShock<const D: usize, IdiosyncraticRateLink = Log, SharedRateLink = Log> {
    marker: PhantomData<(IdiosyncraticRateLink, SharedRateLink)>,
}

impl<const D: usize, IdiosyncraticRateLink, SharedRateLink>
    MvPoissonCommonShock<D, IdiosyncraticRateLink, SharedRateLink>
where
    IdiosyncraticRateLink: PositiveLink<f64>,
    SharedRateLink: PositiveLink<f64>,
{
    /// Largest `min(Y_i)` accepted by the exact latent-count likelihood sum.
    ///
    /// The number of evaluated terms is one larger because the sum includes
    /// both zero and this upper endpoint.
    pub const MAX_EXACT_SHARED_COUNT: u64 = 1_000_000;

    /// Creates a stateless family after checking the compile-time dimension.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `D < 2`.
    pub const fn try_new() -> Result<Self, ModelError> {
        if D < 2 {
            return Err(ModelError::InvalidParameter {
                parameter: "dimension",
                expected: "at least two response components",
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
    /// Panics when `D < 2`.
    #[must_use]
    pub const fn new() -> Self {
        assert!(
            D >= 2,
            "common-shock multivariate Poisson dimension must be at least two"
        );
        Self {
            marker: PhantomData,
        }
    }

    fn theta_from_eta(eta: &MvPoissonCommonShockEta<D>) -> MvPoissonCommonShockTheta<D> {
        MvPoissonCommonShockTheta {
            idiosyncratic_rate: eta.idiosyncratic_rate.map(IdiosyncraticRateLink::inverse),
            shared_rate: SharedRateLink::inverse(eta.shared_rate),
        }
    }

    const fn nan_eta() -> MvPoissonCommonShockEta<D> {
        MvPoissonCommonShockEta {
            idiosyncratic_rate: [f64::NAN; D],
            shared_rate: f64::NAN,
        }
    }

    fn latent_log_sum_and_mean(
        observation: &[f64; D],
        theta: &MvPoissonCommonShockTheta<D>,
    ) -> Option<LatentPosteriorSummary> {
        if !valid_theta(theta)
            || observation
                .iter()
                .any(|count| !is_nonnegative_integer(*count))
        {
            return None;
        }
        let min_count = observation.iter().copied().fold(f64::INFINITY, f64::min);
        let upper = included_count(min_count, Self::MAX_EXACT_SHARED_COUNT)?;
        let log_rates = theta.idiosyncratic_rate.map(f64::ln);
        let log_shared_rate = theta.shared_rate.ln();
        let mut log_term = observation
            .iter()
            .zip(log_rates)
            .map(|(count, log_rate)| count * log_rate - ln_gamma(count + 1.0))
            .sum::<f64>();
        let mut log_sum = f64::NEG_INFINITY;
        let mut latent_mean = 0.0;

        for shared_count in 0..=upper {
            let next_log_sum = log_add_exp(log_sum, log_term);
            if !next_log_sum.is_finite() {
                return None;
            }
            let previous_weight = if log_sum == f64::NEG_INFINITY {
                0.0
            } else {
                (log_sum - next_log_sum).exp()
            };
            let current_weight = (log_term - next_log_sum).exp();
            latent_mean = previous_weight * latent_mean + current_weight * shared_count as f64;
            log_sum = next_log_sum;

            if shared_count < upper {
                let shared_count_f = shared_count as f64;
                let log_ratio = log_shared_rate - shared_count_f.ln_1p()
                    + observation
                        .iter()
                        .zip(log_rates)
                        .map(|(count, log_rate)| (count - shared_count_f).ln() - log_rate)
                        .sum::<f64>();
                log_term += log_ratio;
            }
        }
        (log_sum.is_finite() && latent_mean.is_finite()).then_some(LatentPosteriorSummary {
            log_kernel_sum: log_sum,
            shared_count_mean: latent_mean,
        })
    }

    fn nll_theta(observation: [f64; D], theta: &MvPoissonCommonShockTheta<D>) -> f64 {
        let Some(summary) = Self::latent_log_sum_and_mean(&observation, theta) else {
            return f64::INFINITY;
        };
        let nll = theta.idiosyncratic_rate.iter().sum::<f64>() + theta.shared_rate
            - summary.log_kernel_sum;
        if nll.is_finite() { nll } else { f64::INFINITY }
    }

    fn nll_and_gradient_eta_values(
        observation: [f64; D],
        eta: &MvPoissonCommonShockEta<D>,
    ) -> (f64, MvPoissonCommonShockEta<D>) {
        let theta = Self::theta_from_eta(eta);
        let Some(summary) = Self::latent_log_sum_and_mean(&observation, &theta) else {
            return (f64::INFINITY, Self::nan_eta());
        };
        let nll = theta.idiosyncratic_rate.iter().sum::<f64>() + theta.shared_rate
            - summary.log_kernel_sum;
        if !nll.is_finite() {
            return (f64::INFINITY, Self::nan_eta());
        }

        let gradient = MvPoissonCommonShockEta {
            idiosyncratic_rate: std::array::from_fn(|component| {
                let natural_score = 1.0
                    - (observation[component] - summary.shared_count_mean)
                        / theta.idiosyncratic_rate[component];
                natural_score
                    * IdiosyncraticRateLink::derivative_inverse(eta.idiosyncratic_rate[component])
            }),
            shared_rate: (1.0 - summary.shared_count_mean / theta.shared_rate)
                * SharedRateLink::derivative_inverse(eta.shared_rate),
        };
        (nll, gradient)
    }
}

impl<const D: usize, IdiosyncraticRateLink, SharedRateLink> Default
    for MvPoissonCommonShock<D, IdiosyncraticRateLink, SharedRateLink>
where
    IdiosyncraticRateLink: PositiveLink<f64>,
    SharedRateLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize, IdiosyncraticRateLink, SharedRateLink> Family
    for MvPoissonCommonShock<D, IdiosyncraticRateLink, SharedRateLink>
where
    IdiosyncraticRateLink: PositiveLink<f64>,
    SharedRateLink: PositiveLink<f64>,
{
    type Eta = MvPoissonCommonShockEta<D>;
    type Theta = MvPoissonCommonShockTheta<D>;
    type GradientEta = MvPoissonCommonShockEta<D>;
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

impl<const D: usize, IdiosyncraticRateLink, SharedRateLink> FixedDimensionalFamily<D>
    for MvPoissonCommonShock<D, IdiosyncraticRateLink, SharedRateLink>
where
    IdiosyncraticRateLink: PositiveLink<f64>,
    SharedRateLink: PositiveLink<f64>,
{
}

impl<const D: usize, IdiosyncraticRateLink, SharedRateLink> HasObservationDimension
    for MvPoissonCommonShock<D, IdiosyncraticRateLink, SharedRateLink>
where
    IdiosyncraticRateLink: PositiveLink<f64>,
    SharedRateLink: PositiveLink<f64>,
{
    fn observation_dimension(&self) -> usize {
        D
    }
}

impl<const D: usize, IdiosyncraticRateLink, SharedRateLink> HasMarginalCdf
    for MvPoissonCommonShock<D, IdiosyncraticRateLink, SharedRateLink>
where
    IdiosyncraticRateLink: PositiveLink<f64>,
    SharedRateLink: PositiveLink<f64>,
{
    fn marginal_cdf(&self, component: usize, y: f64, theta: &Self::Theta) -> f64 {
        let Some(mean) = theta.marginal_mean(component) else {
            return f64::NAN;
        };
        if !valid_theta(theta) {
            return f64::NAN;
        }
        PoissonKernel::cdf(y, mean)
    }
}

#[cfg(feature = "rand")]
impl<Rng, const D: usize, IdiosyncraticRateLink, SharedRateLink> TrySimulate<Rng>
    for MvPoissonCommonShock<D, IdiosyncraticRateLink, SharedRateLink>
where
    Rng: rand::Rng,
    IdiosyncraticRateLink: PositiveLink<f64>,
    SharedRateLink: PositiveLink<f64>,
{
    type Sample = [f64; D];

    fn try_sample(
        &self,
        rng: &mut Rng,
        theta: &Self::Theta,
    ) -> Result<Self::Sample, SimulationError> {
        if !valid_theta(theta) {
            return Err(SimulationError::InvalidParameters(
                "common-shock multivariate Poisson theta",
            ));
        }
        let shared_distribution = rand_distr::Poisson::new(theta.shared_rate)
            .map_err(|_| SimulationError::BackendRejected("shared Poisson rate"))?;
        let shared = rand_distr::Distribution::sample(&shared_distribution, rng);
        let mut out = [0.0; D];
        for component in 0..D {
            let distribution = rand_distr::Poisson::new(theta.idiosyncratic_rate[component])
                .map_err(|_| SimulationError::BackendRejected("idiosyncratic Poisson rate"))?;
            let idiosyncratic = rand_distr::Distribution::sample(&distribution, rng);
            out[component] = shared + idiosyncratic;
        }
        if out.iter().all(|value| value.is_finite()) {
            Ok(out)
        } else {
            Err(SimulationError::NumericalFailure(
                "common-shock multivariate Poisson sample",
            ))
        }
    }
}

impl<const D: usize, IdiosyncraticRateLink, SharedRateLink> CompilableFamily
    for MvPoissonCommonShock<D, IdiosyncraticRateLink, SharedRateLink>
where
    IdiosyncraticRateLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    SharedRateLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Shape = Product<Vector<IdiosyncraticRate, D>, Scalar<SharedRate>>;

    fn eta_from_shape(values: ShapeValues<Self::Shape>) -> Self::Eta {
        MvPoissonCommonShockEta::new(values.0, values.1)
    }

    fn gradient_to_shape(gradient: &Self::GradientEta) -> ShapeValues<Self::Shape> {
        (gradient.idiosyncratic_rate, gradient.shared_rate)
    }

    fn initial_shape<'obs, Obs>(&self, obs: &'obs Obs) -> ShapeValues<Self::Shape>
    where
        Obs: ObservationView<'obs, Observation = [f64; D]> + 'obs,
    {
        let summary = weighted_count_moments(obs);
        let minimum_mean = summary
            .marginal_mean
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let fallback = 0.05 * minimum_mean;
        let empirical = summary.average_pair_covariance.unwrap_or(fallback);
        let upper = 0.5 * minimum_mean;
        let shared_rate = if upper > 0.0 {
            positive_floor(empirical.max(fallback).min(upper))
        } else {
            positive_floor(0.0)
        };
        let idiosyncratic_rate = summary
            .marginal_mean
            .map(|mean| positive_floor(mean - shared_rate));

        (
            idiosyncratic_rate.map(IdiosyncraticRateLink::initial_eta_from_theta),
            SharedRateLink::initial_eta_from_theta(shared_rate),
        )
    }
}

/// Link-scale predictors for the common-shock multivariate Poisson family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MvPoissonCommonShockEta<const D: usize> {
    /// Predictors for component-specific latent rates.
    pub idiosyncratic_rate: [f64; D],
    /// Predictor for the common latent rate.
    pub shared_rate: f64,
}

impl<const D: usize> MvPoissonCommonShockEta<D> {
    /// Creates link-scale predictors.
    #[must_use]
    pub const fn new(idiosyncratic_rate: [f64; D], shared_rate: f64) -> Self {
        Self {
            idiosyncratic_rate,
            shared_rate,
        }
    }
}

/// Natural-scale parameters for the common-shock multivariate Poisson family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MvPoissonCommonShockTheta<const D: usize> {
    idiosyncratic_rate: [f64; D],
    shared_rate: f64,
}

impl<const D: usize> MvPoissonCommonShockTheta<D> {
    /// Creates checked positive latent rates.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] unless `D >= 2` and all rates
    /// are finite and strictly positive.
    pub fn try_new(idiosyncratic_rate: [f64; D], shared_rate: f64) -> Result<Self, ModelError> {
        let theta = Self {
            idiosyncratic_rate,
            shared_rate,
        };
        if valid_theta(&theta) {
            Ok(theta)
        } else {
            Err(ModelError::InvalidParameter {
                parameter: "common-shock multivariate Poisson theta",
                expected: "at least two components and finite, strictly positive latent rates",
            })
        }
    }

    /// Component-specific latent rates.
    #[must_use]
    pub const fn idiosyncratic_rate(&self) -> &[f64; D] {
        &self.idiosyncratic_rate
    }

    /// Common latent rate and off-diagonal covariance.
    #[must_use]
    pub const fn shared_rate(&self) -> f64 {
        self.shared_rate
    }

    /// Marginal response mean for one component.
    #[must_use]
    pub fn marginal_mean(&self, component: usize) -> Option<f64> {
        self.idiosyncratic_rate
            .get(component)
            .map(|rate| rate + self.shared_rate)
    }

    /// Marginal response variance for one component.
    #[must_use]
    pub fn marginal_variance(&self, component: usize) -> Option<f64> {
        self.marginal_mean(component)
    }

    /// Returns a correlation-matrix entry.
    #[must_use]
    pub fn correlation(&self, row: usize, col: usize) -> Option<f64> {
        let row_variance = self.marginal_variance(row)?;
        let col_variance = self.marginal_variance(col)?;
        if row == col {
            Some(1.0)
        } else {
            Some(self.shared_rate / (row_variance * col_variance).sqrt())
        }
    }

    /// Returns a covariance-matrix entry.
    #[must_use]
    pub fn covariance(&self, row: usize, col: usize) -> Option<f64> {
        if row >= D || col >= D {
            return None;
        }
        if row == col {
            self.marginal_mean(row)
        } else {
            Some(self.shared_rate)
        }
    }
}

fn valid_theta<const D: usize>(theta: &MvPoissonCommonShockTheta<D>) -> bool {
    D >= 2
        && theta
            .idiosyncratic_rate
            .iter()
            .all(|rate| rate.is_finite() && *rate > 0.0)
        && theta.shared_rate.is_finite()
        && theta.shared_rate > 0.0
}

fn weighted_count_moments<'obs, Obs, const D: usize>(obs: &'obs Obs) -> CountMomentSummary<D>
where
    Obs: ObservationView<'obs, Observation = [f64; D]> + 'obs,
{
    let mut weight_sum = 0.0;
    let mut marginal_mean = [0.0; D];
    let mut pair_m2 = [[0.0; D]; D];
    for row in 0..obs.len() {
        let weight = obs.weight_at(row);
        let observation = obs.observation_at(row);
        if weight <= 0.0
            || !weight.is_finite()
            || observation
                .iter()
                .any(|count| !is_nonnegative_integer(*count))
        {
            continue;
        }
        let next_weight_sum = weight_sum + weight;
        if !next_weight_sum.is_finite() {
            continue;
        }
        let delta: [f64; D] =
            std::array::from_fn(|component| observation[component] - marginal_mean[component]);
        let next_mean: [f64; D] = std::array::from_fn(|component| {
            marginal_mean[component] + weight / next_weight_sum * delta[component]
        });
        for left in 1..D {
            for right in 0..left {
                pair_m2[left][right] +=
                    weight * delta[left] * (observation[right] - next_mean[right]);
            }
        }
        marginal_mean = next_mean;
        weight_sum = next_weight_sum;
    }

    if weight_sum <= 0.0 {
        return CountMomentSummary {
            marginal_mean: [1.0; D],
            average_pair_covariance: None,
        };
    }
    let mut covariance_sum = 0.0;
    let mut covariance_pairs = 0usize;
    for (left, row) in pair_m2.iter().enumerate().skip(1) {
        for value in row.iter().take(left) {
            covariance_sum += value / weight_sum;
            covariance_pairs += 1;
        }
    }
    let average_pair_covariance = (covariance_pairs > 0)
        .then(|| covariance_sum / covariance_pairs as f64)
        .filter(|covariance| covariance.is_finite());
    CountMomentSummary {
        marginal_mean,
        average_pair_covariance,
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{
        DenseDesign, Family, Gamlss, HasCdf, HasMarginalCdf, IdiosyncraticRate,
        LinearPredictorBlock, NoPenalty, ParameterBlock, ParameterBlocks, SharedRate,
        VectorParameterBlock,
    };
    use gamlss_special::is_nonnegative_integer;

    use super::{MvPoissonCommonShockDefault, MvPoissonCommonShockEta, MvPoissonCommonShockTheta};
    use crate::{IndependentVec, PoissonEta, PoissonMean, PoissonTheta};

    #[test]
    fn bivariate_likelihood_matches_direct_latent_sum() {
        let family = MvPoissonCommonShockDefault::<2>::new();
        let theta = MvPoissonCommonShockTheta::try_new([1.2, 0.7], 0.4).unwrap();
        let probability = (-2.3_f64).exp() * (1.2_f64.powi(2) * 0.7 / 2.0 + 0.4 * 1.2);
        assert_relative_eq!(
            family.nll([2.0, 1.0], &theta, &mut ()),
            -probability.ln(),
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn analytic_gradient_matches_finite_difference() {
        let family = MvPoissonCommonShockDefault::<3>::new();
        let eta = MvPoissonCommonShockEta::new([0.2, -0.4, 0.1], -0.7);
        let observation = [3.0, 1.0, 2.0];
        let (_, gradient) = family.nll_and_gradient_eta(observation, &eta, &mut ());
        let epsilon = 1.0e-6;
        for component in 0..3 {
            let mut plus = eta;
            plus.idiosyncratic_rate[component] += epsilon;
            let mut minus = eta;
            minus.idiosyncratic_rate[component] -= epsilon;
            let finite_difference = (family.nll_eta(observation, &plus, &mut ())
                - family.nll_eta(observation, &minus, &mut ()))
                / (2.0 * epsilon);
            assert_relative_eq!(
                gradient.idiosyncratic_rate[component],
                finite_difference,
                epsilon = 1.0e-6
            );
        }
        let mut plus = eta;
        plus.shared_rate += epsilon;
        let mut minus = eta;
        minus.shared_rate -= epsilon;
        let finite_difference = (family.nll_eta(observation, &plus, &mut ())
            - family.nll_eta(observation, &minus, &mut ()))
            / (2.0 * epsilon);
        assert_relative_eq!(gradient.shared_rate, finite_difference, epsilon = 1.0e-6);
    }

    #[test]
    fn zero_component_reduces_to_independent_poissons_plus_zero_shock_probability() {
        let family = MvPoissonCommonShockDefault::<3>::new();
        let independent = IndependentVec::<_, 3>::new(PoissonMean::new());
        let eta =
            MvPoissonCommonShockEta::new([1.2_f64.ln(), 0.7_f64.ln(), 2.1_f64.ln()], 0.4_f64.ln());
        let independent_eta = eta.idiosyncratic_rate.map(|mu| PoissonEta { mu });
        let observation = [0.0, 2.0, 1.0];
        let (nll, gradient) = family.nll_and_gradient_eta(observation, &eta, &mut ());
        let (independent_nll, independent_gradient) = independent.nll_and_gradient_eta(
            observation,
            &independent_eta,
            &mut independent.workspace(),
        );

        assert_relative_eq!(nll, independent_nll + 0.4, epsilon = 1.0e-14);
        for component in 0..3 {
            assert_relative_eq!(
                gradient.idiosyncratic_rate[component],
                independent_gradient[component].mu,
                epsilon = 1.0e-14
            );
        }
        assert_relative_eq!(gradient.shared_rate, 0.4, epsilon = 1.0e-14);
    }

    #[test]
    fn likelihood_and_gradient_are_permutation_equivariant() {
        let family = MvPoissonCommonShockDefault::<3>::new();
        let eta = MvPoissonCommonShockEta::new([0.2, -0.4, 0.1], -0.7);
        let observation = [3.0, 1.0, 2.0];
        let (nll, gradient) = family.nll_and_gradient_eta(observation, &eta, &mut ());

        let permutation = [2, 0, 1];
        let permuted_eta = MvPoissonCommonShockEta::new(
            permutation.map(|index| eta.idiosyncratic_rate[index]),
            eta.shared_rate,
        );
        let permuted_observation = permutation.map(|index| observation[index]);
        let (permuted_nll, permuted_gradient) =
            family.nll_and_gradient_eta(permuted_observation, &permuted_eta, &mut ());

        assert_relative_eq!(permuted_nll, nll, epsilon = 1.0e-14);
        for (component, original) in permutation.into_iter().enumerate() {
            assert_relative_eq!(
                permuted_gradient.idiosyncratic_rate[component],
                gradient.idiosyncratic_rate[original],
                epsilon = 1.0e-14
            );
        }
        assert_relative_eq!(
            permuted_gradient.shared_rate,
            gradient.shared_rate,
            epsilon = 1.0e-14
        );
    }

    #[test]
    fn moments_and_marginal_cdf_follow_poisson_construction() {
        let family = MvPoissonCommonShockDefault::<2>::new();
        let theta = MvPoissonCommonShockTheta::try_new([1.2, 0.7], 0.4).unwrap();
        assert_eq!(theta.marginal_mean(0), Some(1.6));
        assert_eq!(theta.marginal_variance(0), Some(1.6));
        assert_eq!(theta.covariance(0, 1), Some(0.4));
        assert_eq!(theta.covariance(1, 1), Some(1.1));
        assert_relative_eq!(
            theta.correlation(0, 1).unwrap(),
            0.4 / (1.6_f64 * 1.1).sqrt(),
            epsilon = 1.0e-14
        );
        assert_eq!(theta.correlation(1, 1), Some(1.0));
        assert_relative_eq!(
            family.marginal_cdf(0, 2.0, &theta),
            PoissonMean::new().cdf(2.0, &PoissonTheta { mu: 1.6 }),
            epsilon = 1.0e-14
        );
    }

    #[test]
    fn invalid_dimensions_counts_and_rates_are_rejected() {
        assert!(MvPoissonCommonShockDefault::<1>::try_new().is_err());
        assert!(MvPoissonCommonShockTheta::<2>::try_new([1.0, 0.0], 0.5).is_err());
        let family = MvPoissonCommonShockDefault::<2>::new();
        let theta = MvPoissonCommonShockTheta::try_new([1.0, 2.0], 0.5).unwrap();
        assert!(family.nll([-1.0, 2.0], &theta, &mut ()).is_infinite());
        assert!(family.nll([1.5, 2.0], &theta, &mut ()).is_infinite());
        let excessive = (MvPoissonCommonShockDefault::<2>::MAX_EXACT_SHARED_COUNT + 1) as f64;
        assert!(
            family
                .nll([excessive, excessive], &theta, &mut ())
                .is_infinite()
        );
    }

    #[test]
    fn static_shape_is_fit_ready() {
        let response = [[0.0, 1.0], [2.0, 3.0], [1.0, 1.0]];
        let rows = response.len();
        let intercept = || LinearPredictorBlock::new(DenseDesign::intercept(rows));
        let rates = VectorParameterBlock::<IdiosyncraticRate, 2, _, _>::new(
            [intercept(), intercept()],
            NoPenalty,
            0,
        );
        let shared = ParameterBlock::<SharedRate, _, _>::new(intercept(), NoPenalty, 0);
        let model = Gamlss::try_new_with_observations(
            MvPoissonCommonShockDefault::<2>::new(),
            ParameterBlocks::new((rates, shared)),
            response.as_slice(),
        )
        .unwrap();
        let beta: [f64; 3] = [0.1, -0.2, -0.7];
        let mut gradient: [f64; 3] = [0.0; 3];
        let value = model.try_value_gradient_into(&beta, &mut gradient).unwrap();
        assert!(value.is_finite());
        assert!(gradient.iter().all(|score| score.is_finite()));
        assert_eq!(
            model
                .parameter_layout()
                .unique_slice("idiosyncratic_rate")
                .unwrap(),
            Some(0..2)
        );
        assert_eq!(
            model
                .parameter_layout()
                .unique_slice("shared_rate")
                .unwrap(),
            Some(2..3)
        );
        assert!(
            model
                .initial_parameters()
                .is_ok_and(|initial| initial.iter().all(|value| value.is_finite()))
        );
    }

    #[cfg(feature = "rand")]
    #[test]
    fn sampling_returns_finite_count_vectors() {
        use gamlss_core::TrySimulate;
        use rand::SeedableRng;

        let family = MvPoissonCommonShockDefault::<2>::new();
        let theta = MvPoissonCommonShockTheta::try_new([1.2, 0.7], 0.4).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(41);
        let sample = family.try_sample(&mut rng, &theta).unwrap();
        assert!(sample.iter().all(|count| is_nonnegative_integer(*count)));
    }
}
