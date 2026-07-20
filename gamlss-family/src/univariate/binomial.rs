use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasQuantile, InitialEtaFromObservations, InitialEtaFromTheta, Logit,
    ModelError, ObservationView, ParameterParts, Probability, UnitIntervalLink,
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};
use gamlss_special::{
    bernoulli_kl, is_nonnegative_integer, ln_gamma, ln_gamma_delta, ln_gamma_stirling_residual,
    regularized_beta,
};

use crate::domain::{is_probability, is_strict_probability};
use crate::initial::probability_floor;

/// Binomial family with one fixed positive number of trials for every row.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/distributions/binomial_fixed_trials.svg")
)]
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinomialFixedTrials<ProbabilityLink = Logit> {
    trials: u32,
    marker: PhantomData<ProbabilityLink>,
}

/// Binomial family whose observations carry `[successes, trials]`.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/distributions/binomial_varying_trials.svg")
)]
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinomialVaryingTrials<ProbabilityLink = Logit> {
    marker: PhantomData<ProbabilityLink>,
}

/// Fixed-trials binomial family with the default logit probability link.
pub type BinomialFixedTrialsProbability = BinomialFixedTrials<Logit>;
/// Varying-trials binomial family with the default logit probability link.
pub type BinomialVaryingTrialsProbability = BinomialVaryingTrials<Logit>;

#[derive(Debug, Clone, Copy)]
pub(super) struct BinomialKernel;

impl BinomialKernel {
    #[inline]
    pub(super) fn valid_observation(successes: f64, trials: f64) -> bool {
        is_nonnegative_integer(successes) && is_nonnegative_integer(trials) && successes <= trials
    }

    #[inline]
    pub(super) fn log_choose(trials: f64, successes: f64) -> f64 {
        let smaller = successes.min(trials - successes);
        if smaller <= 0.0 {
            return 0.0;
        }
        let larger = trials - smaller;
        ln_gamma_delta(larger + 1.0, smaller) - ln_gamma(smaller + 1.0)
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn nll(successes: f64, trials: f64, probability: f64) -> f64 {
        if !Self::valid_observation(successes, trials) || !is_strict_probability(probability) {
            return f64::INFINITY;
        }
        let failures = trials - successes;
        if successes <= 0.0 {
            return -trials * (-probability).ln_1p();
        }
        if failures <= 0.0 {
            return -trials * probability.ln();
        }
        let success_fraction = successes / trials;
        ln_gamma_stirling_residual(successes) + ln_gamma_stirling_residual(failures)
            - ln_gamma_stirling_residual(trials)
            + trials.ln()
            + success_fraction.ln()
            + (-success_fraction).ln_1p()
            + trials * bernoulli_kl(success_fraction, probability)
    }

    #[inline]
    fn d_probability(successes: f64, trials: f64, probability: f64) -> f64 {
        trials.mul_add(probability, -successes) / (probability * (1.0 - probability))
    }

    fn cdf(successes: f64, trials: f64, probability: f64) -> f64 {
        if !successes.is_finite()
            || !is_nonnegative_integer(trials)
            || trials < 0.0
            || !is_strict_probability(probability)
        {
            return f64::NAN;
        }
        if successes < 0.0 {
            return 0.0;
        }
        let successes = successes.floor();
        if successes >= trials {
            return 1.0;
        }
        regularized_beta(trials - successes, successes + 1.0, 1.0 - probability)
    }
}

impl<ProbabilityLink> BinomialFixedTrials<ProbabilityLink>
where
    ProbabilityLink: UnitIntervalLink<f64>,
{
    /// Creates a fixed-trials family after validating `trials > 0`.
    pub const fn try_new(trials: u32) -> Result<Self, ModelError> {
        if trials == 0 {
            return Err(ModelError::InvalidParameter {
                parameter: "binomial trials",
                expected: "positive",
            });
        }
        Ok(Self {
            trials,
            marker: PhantomData,
        })
    }

    /// Returns the common number of trials.
    #[must_use]
    pub const fn trials(&self) -> u32 {
        self.trials
    }

    #[inline]
    fn trials_f64(&self) -> f64 {
        f64::from(self.trials)
    }

    #[inline]
    fn theta_from_eta(eta: BinomialEta) -> BinomialTheta {
        BinomialTheta {
            probability: ProbabilityLink::inverse(eta.probability),
        }
    }
}

impl<ProbabilityLink> BinomialVaryingTrials<ProbabilityLink>
where
    ProbabilityLink: UnitIntervalLink<f64>,
{
    /// Creates a stateless varying-trials binomial family.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(eta: BinomialEta) -> BinomialTheta {
        BinomialTheta {
            probability: ProbabilityLink::inverse(eta.probability),
        }
    }
}

impl<ProbabilityLink> Default for BinomialVaryingTrials<ProbabilityLink>
where
    ProbabilityLink: UnitIntervalLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<ProbabilityLink> for BinomialFixedTrials<ProbabilityLink>;
    parameters = (Probability,);
    arity = 1;
);

gamlss_core::impl_scalar_compilable_family!(
    impl<ProbabilityLink> for BinomialVaryingTrials<ProbabilityLink>;
    parameters = (Probability,);
    arity = 1;
);

macro_rules! impl_family {
    ($family:ident, $observation:ty, $parts:expr) => {
        impl<ProbabilityLink> Family for $family<ProbabilityLink>
        where
            ProbabilityLink: UnitIntervalLink<f64>,
        {
            type Eta = BinomialEta;
            type Theta = BinomialTheta;
            type GradientEta = BinomialEta;
            type Observation<'obs> = $observation;
            type Workspace = ();

            fn workspace(&self) -> Self::Workspace {}

            fn theta(&self, eta: &Self::Eta, _workspace: &mut ()) -> Self::Theta {
                Self::theta_from_eta(*eta)
            }

            fn nll(
                &self,
                observation: $observation,
                theta: &Self::Theta,
                _workspace: &mut (),
            ) -> f64 {
                let (successes, trials) = $parts(self, observation);
                BinomialKernel::nll(successes, trials, theta.probability)
            }

            fn nll_and_gradient_eta(
                &self,
                observation: $observation,
                eta: &Self::Eta,
                _workspace: &mut (),
            ) -> (f64, Self::GradientEta) {
                let theta = Self::theta_from_eta(*eta);
                let (successes, trials) = $parts(self, observation);
                let nll = BinomialKernel::nll(successes, trials, theta.probability);
                if !nll.is_finite() {
                    return (
                        nll,
                        BinomialEta {
                            probability: f64::NAN,
                        },
                    );
                }
                let gradient = BinomialKernel::d_probability(successes, trials, theta.probability)
                    * ProbabilityLink::derivative_inverse(eta.probability);
                (
                    nll,
                    BinomialEta {
                        probability: gradient,
                    },
                )
            }
        }
    };
}

impl_family!(
    BinomialFixedTrials,
    f64,
    |family: &BinomialFixedTrials<ProbabilityLink>, successes| { (successes, family.trials_f64()) }
);
impl_family!(
    BinomialVaryingTrials,
    [f64; 2],
    |_family, observation: [f64; 2]| { observation.into() }
);

impl<ProbabilityLink> InitialEtaFromObservations<1> for BinomialFixedTrials<ProbabilityLink>
where
    ProbabilityLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = f64> + 'obs,
    {
        let trials = self.trials_f64();
        let mut successes = 0.0;
        let mut total_trials = 0.0;
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            let value = obs.observation_at(row);
            if weight > 0.0 && BinomialKernel::valid_observation(value, trials) {
                successes = weight.mul_add(value, successes);
                total_trials = weight.mul_add(trials, total_trials);
            }
        }
        initial_eta::<ProbabilityLink>(successes, total_trials)
    }
}

impl<ProbabilityLink> InitialEtaFromObservations<1> for BinomialVaryingTrials<ProbabilityLink>
where
    ProbabilityLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = [f64; 2]> + 'obs,
    {
        let mut successes = 0.0;
        let mut total_trials = 0.0;
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            let [value, trials] = obs.observation_at(row);
            if weight > 0.0 && BinomialKernel::valid_observation(value, trials) {
                successes = weight.mul_add(value, successes);
                total_trials = weight.mul_add(trials, total_trials);
            }
        }
        initial_eta::<ProbabilityLink>(successes, total_trials)
    }
}

fn initial_eta<ProbabilityLink>(successes: f64, trials: f64) -> BinomialEta
where
    ProbabilityLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
{
    let probability = if trials > 0.0 {
        probability_floor((successes + 0.5) / (trials + 1.0))
    } else {
        0.5
    };
    BinomialEta {
        probability: ProbabilityLink::initial_eta_from_theta(probability),
    }
}

impl<ProbabilityLink> HasCdf for BinomialFixedTrials<ProbabilityLink>
where
    ProbabilityLink: UnitIntervalLink<f64>,
{
    fn cdf(&self, successes: f64, theta: &Self::Theta) -> f64 {
        BinomialKernel::cdf(successes, self.trials_f64(), theta.probability)
    }
}

impl<ProbabilityLink> HasCdf for BinomialVaryingTrials<ProbabilityLink>
where
    ProbabilityLink: UnitIntervalLink<f64>,
{
    fn cdf(&self, observation: [f64; 2], theta: &Self::Theta) -> f64 {
        BinomialKernel::cdf(observation[0], observation[1], theta.probability)
    }
}

impl<ProbabilityLink> HasQuantile for BinomialFixedTrials<ProbabilityLink>
where
    ProbabilityLink: UnitIntervalLink<f64>,
{
    fn quantile(&self, probability: f64, theta: &Self::Theta) -> f64 {
        if !is_probability(probability) || !is_strict_probability(theta.probability) {
            return f64::NAN;
        }
        let mut lower = 0_u32;
        let mut upper = self.trials;
        while lower < upper {
            let middle = lower + (upper - lower) / 2;
            if self.cdf(f64::from(middle), theta) >= probability {
                upper = middle;
            } else {
                lower = middle + 1;
            }
        }
        f64::from(lower)
    }
}

#[cfg(feature = "rand")]
impl<Rng, ProbabilityLink> TrySimulate<Rng> for BinomialFixedTrials<ProbabilityLink>
where
    Rng: rand::Rng,
    ProbabilityLink: UnitIntervalLink<f64>,
{
    type Sample = f64;

    #[allow(clippy::cast_precision_loss)]
    fn try_sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Result<f64, SimulationError> {
        if !is_strict_probability(theta.probability) {
            return Err(SimulationError::InvalidParameters("Binomial theta"));
        }
        let distribution = rand_distr::Binomial::new(u64::from(self.trials), theta.probability)
            .map_err(|_| SimulationError::BackendRejected("Binomial trials/probability"))?;
        Ok(rand_distr::Distribution::sample(&distribution, rng) as f64)
    }
}

/// Link-scale probability predictor shared by binomial families.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinomialEta {
    /// Success-probability predictor.
    pub probability: f64,
}

impl ParameterParts<1> for BinomialEta {
    fn from_array(values: [f64; 1]) -> Self {
        Self {
            probability: values[0],
        }
    }
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.probability,
            _ => unreachable!("binomial eta only has index 0"),
        }
    }
}

/// Natural-scale probability shared by binomial families.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinomialTheta {
    /// Success probability in `(0, 1)`.
    pub probability: f64,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use approx::assert_relative_eq;
    use gamlss_core::{Family, HasCdf, HasQuantile, Logit, ParameterParts};

    use super::{BinomialFixedTrials, BinomialTheta, BinomialVaryingTrials};
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn fixed_and_varying_gradients_match_finite_difference() {
        let fixed = BinomialFixedTrials::<Logit>::try_new(10).unwrap();
        let varying = BinomialVaryingTrials::<Logit>::new();
        assert_gradient_matches_finite_difference::<_, 1>(&fixed, 4.0, [0.2]);
        let eta = super::BinomialEta::from_array([0.2]);
        let epsilon = 1.0e-6;
        let lower = super::BinomialEta::from_array([0.2 - epsilon]);
        let upper = super::BinomialEta::from_array([0.2 + epsilon]);
        let (_, gradient) = varying.nll_and_gradient_eta([4.0, 10.0], &eta, &mut ());
        let numeric = (varying.nll_eta([4.0, 10.0], &upper, &mut ())
            - varying.nll_eta([4.0, 10.0], &lower, &mut ()))
            / (2.0 * epsilon);
        assert_relative_eq!(gradient.probability, numeric, epsilon = 1.0e-7);
    }

    #[test]
    fn fixed_and_varying_likelihoods_match() {
        let fixed = BinomialFixedTrials::<Logit>::try_new(10).unwrap();
        let varying = BinomialVaryingTrials::<Logit>::new();
        let theta = BinomialTheta { probability: 0.4 };
        assert_relative_eq!(
            fixed.nll(3.0, &theta, &mut ()),
            varying.nll([3.0, 10.0], &theta, &mut ()),
            epsilon = 1.0e-14
        );
        assert_relative_eq!(
            fixed.cdf(3.0, &theta),
            varying.cdf([3.0, 10.0], &theta),
            epsilon = 1.0e-14
        );
        assert_eq!(fixed.quantile(fixed.cdf(3.0, &theta), &theta), 3.0);
    }

    #[test]
    fn rejects_invalid_counts() {
        let family = BinomialVaryingTrials::<Logit>::new();
        let theta = BinomialTheta { probability: 0.4 };
        assert!(family.nll([11.0, 10.0], &theta, &mut ()).is_infinite());
        assert!(family.nll([3.5, 10.0], &theta, &mut ()).is_infinite());
        assert_relative_eq!(family.nll([0.0, 0.0], &theta, &mut ()), 0.0);
        assert_relative_eq!(family.cdf([0.0, 0.0], &theta), 1.0);
    }

    #[test]
    fn concentrated_large_trial_likelihood_preserves_normalizer() {
        let family = BinomialVaryingTrials::<Logit>::new();
        let trials = 1.0e16;
        let probability = 0.4;
        let successes = trials * probability;
        let theta = BinomialTheta { probability };
        let nll = family.nll([successes, trials], &theta, &mut ());
        let expected =
            0.5 * (std::f64::consts::TAU * trials * probability * (1.0 - probability)).ln();
        assert_relative_eq!(nll, expected, epsilon = 2.0e-13);
    }
}
