use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasCrps, InitialEtaFromObservations, InitialEtaFromTheta, Log, Logit,
    ObservationView, ParameterParts, PositiveLink, Precision, Probability, UnitIntervalLink,
};
use gamlss_special::{digamma_delta, included_count, ln_gamma_delta, log_add_exp};

use crate::crps::finite_discrete_crps_from_log_pmf;
use crate::domain::{is_positive_finite, is_strict_probability};
use crate::initial::{positive_floor, probability_floor};

use super::binomial::BinomialKernel;

const MAX_CDF_TERMS: u64 = 1_000_000;

/// Beta-binomial mean/precision parameterization with logit/log links.
pub type BetaBinomialMeanPrecision = BetaBinomial<Logit, Log>;

/// Beta-binomial family whose observations carry `[successes, trials]`.
///
/// The beta mixing distribution is parameterized by mean `probability` and
/// positive `precision`, with $\alpha=\mu\phi$ and
/// $\beta=(1-\mu)\phi$.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/distributions/beta_binomial.svg")
)]
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BetaBinomial<ProbabilityLink = Logit, PrecisionLink = Log> {
    marker: PhantomData<(ProbabilityLink, PrecisionLink)>,
}

impl<ProbabilityLink, PrecisionLink> BetaBinomial<ProbabilityLink, PrecisionLink>
where
    ProbabilityLink: UnitIntervalLink<f64>,
    PrecisionLink: PositiveLink<f64>,
{
    /// Creates a stateless beta-binomial family.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    fn theta_from_eta(eta: BetaBinomialEta) -> BetaBinomialTheta {
        BetaBinomialTheta {
            probability: ProbabilityLink::inverse(eta.probability),
            precision: PrecisionLink::inverse(eta.precision),
        }
    }

    fn valid_theta(theta: BetaBinomialTheta) -> bool {
        if !is_strict_probability(theta.probability) || !is_positive_finite(theta.precision) {
            return false;
        }
        let (alpha, beta) = Self::alpha_beta(theta);
        is_positive_finite(alpha) && is_positive_finite(beta)
    }

    fn alpha_beta(theta: BetaBinomialTheta) -> (f64, f64) {
        (
            theta.probability * theta.precision,
            (1.0 - theta.probability) * theta.precision,
        )
    }

    fn log_pmf(successes: f64, trials: f64, theta: BetaBinomialTheta) -> f64 {
        if !BinomialKernel::valid_observation(successes, trials) || !Self::valid_theta(theta) {
            return f64::NEG_INFINITY;
        }
        let failures = trials - successes;
        let (alpha, beta) = Self::alpha_beta(theta);
        BinomialKernel::log_choose(trials, successes)
            + ln_gamma_delta(alpha, successes)
            + ln_gamma_delta(beta, failures)
            - ln_gamma_delta(theta.precision, trials)
    }

    fn nll_theta(successes: f64, trials: f64, theta: BetaBinomialTheta) -> f64 {
        let log_pmf = Self::log_pmf(successes, trials, theta);
        if log_pmf.is_finite() {
            -log_pmf
        } else {
            f64::INFINITY
        }
    }

    fn gradient_theta(successes: f64, trials: f64, theta: BetaBinomialTheta) -> (f64, f64) {
        let (alpha, beta) = Self::alpha_beta(theta);
        let common = digamma_delta(theta.precision, trials);
        let d_alpha = common - digamma_delta(alpha, successes);
        let d_beta = common - digamma_delta(beta, trials - successes);
        (
            theta.precision * (d_alpha - d_beta),
            theta
                .probability
                .mul_add(d_alpha, (1.0 - theta.probability) * d_beta),
        )
    }
}

impl<ProbabilityLink, PrecisionLink> Default for BetaBinomial<ProbabilityLink, PrecisionLink>
where
    ProbabilityLink: UnitIntervalLink<f64>,
    PrecisionLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<ProbabilityLink, PrecisionLink> for BetaBinomial<ProbabilityLink, PrecisionLink>;
    parameters = (Probability, Precision);
    arity = 2;
);

impl<ProbabilityLink, PrecisionLink> Family for BetaBinomial<ProbabilityLink, PrecisionLink>
where
    ProbabilityLink: UnitIntervalLink<f64>,
    PrecisionLink: PositiveLink<f64>,
{
    type Eta = BetaBinomialEta;
    type Theta = BetaBinomialTheta;
    type GradientEta = BetaBinomialEta;
    type Observation<'obs> = [f64; 2];
    type Workspace = ();

    fn workspace(&self) -> Self::Workspace {}
    fn theta(&self, eta: &Self::Eta, _workspace: &mut ()) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }
    fn nll(&self, observation: [f64; 2], theta: &Self::Theta, _workspace: &mut ()) -> f64 {
        Self::nll_theta(observation[0], observation[1], *theta)
    }
    fn nll_and_gradient_eta(
        &self,
        observation: [f64; 2],
        eta: &Self::Eta,
        _workspace: &mut (),
    ) -> (f64, Self::GradientEta) {
        let theta = Self::theta_from_eta(*eta);
        let nll = Self::nll_theta(observation[0], observation[1], theta);
        if !nll.is_finite() {
            return (nll, BetaBinomialEta::from_array([f64::NAN; 2]));
        }
        let (d_probability, d_precision) =
            Self::gradient_theta(observation[0], observation[1], theta);
        (
            nll,
            BetaBinomialEta {
                probability: d_probability * ProbabilityLink::derivative_inverse(eta.probability),
                precision: d_precision * PrecisionLink::derivative_inverse(eta.precision),
            },
        )
    }
}

impl<ProbabilityLink, PrecisionLink> InitialEtaFromObservations<2>
    for BetaBinomial<ProbabilityLink, PrecisionLink>
where
    ProbabilityLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
    PrecisionLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = [f64; 2]> + 'obs,
    {
        let mut weight_sum = 0.0;
        let mut successes = 0.0;
        let mut trials = 0.0;
        let mut proportions = Vec::with_capacity(obs.len());
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            let [value, row_trials] = obs.observation_at(row);
            if weight > 0.0
                && row_trials > 0.0
                && BinomialKernel::valid_observation(value, row_trials)
            {
                weight_sum += weight;
                successes = weight.mul_add(value, successes);
                trials = weight.mul_add(row_trials, trials);
                proportions.push((value / row_trials, row_trials, weight));
            }
        }
        let probability = if trials > 0.0 {
            probability_floor((successes + 0.5) / (trials + 1.0))
        } else {
            0.5
        };
        let precision = if weight_sum > 0.0 && proportions.len() > 1 {
            let variance = proportions
                .iter()
                .map(|(value, _, weight)| weight * (value - probability).powi(2))
                .sum::<f64>()
                / weight_sum;
            let inverse_trials = proportions
                .iter()
                .map(|(_, trials, weight)| weight / trials)
                .sum::<f64>()
                / weight_sum;
            let ratio = variance / (probability * (1.0 - probability));
            let estimate = (1.0 - ratio) / (ratio - inverse_trials);
            if estimate.is_finite() && estimate > 0.0 {
                positive_floor(estimate.min(1.0e6))
            } else {
                10.0
            }
        } else {
            10.0
        };
        BetaBinomialEta {
            probability: ProbabilityLink::initial_eta_from_theta(probability),
            precision: PrecisionLink::initial_eta_from_theta(precision),
        }
    }
}

impl<ProbabilityLink, PrecisionLink> HasCdf for BetaBinomial<ProbabilityLink, PrecisionLink>
where
    ProbabilityLink: UnitIntervalLink<f64>,
    PrecisionLink: PositiveLink<f64>,
{
    #[allow(clippy::cast_precision_loss)]
    fn cdf(&self, observation: [f64; 2], theta: &Self::Theta) -> f64 {
        let [successes, trials] = observation;
        if !successes.is_finite()
            || !trials.is_finite()
            || trials < 0.0
            || trials.fract() != 0.0
            || !Self::valid_theta(*theta)
        {
            return f64::NAN;
        }
        if successes < 0.0 {
            return 0.0;
        }
        if successes >= trials {
            return 1.0;
        }
        let Some(upper) = included_count(successes, MAX_CDF_TERMS) else {
            return f64::NAN;
        };
        let (alpha, beta) = Self::alpha_beta(*theta);
        let mut log_term = Self::log_pmf(0.0, trials, *theta);
        let mut log_sum = log_term;
        for successes in 0..upper {
            let successes = successes as f64;
            let remaining = trials - successes;
            log_term += remaining.ln() - successes.ln_1p() + (alpha + successes).ln()
                - (beta + (remaining - 1.0)).ln();
            log_sum = log_add_exp(log_sum, log_term);
        }
        log_sum.exp().clamp(0.0, 1.0)
    }
}

impl<ProbabilityLink, PrecisionLink> HasCrps for BetaBinomial<ProbabilityLink, PrecisionLink>
where
    ProbabilityLink: UnitIntervalLink<f64>,
    PrecisionLink: PositiveLink<f64>,
{
    #[allow(clippy::cast_precision_loss)]
    fn crps(&self, observation: [f64; 2], theta: &Self::Theta) -> f64 {
        let [successes, trials] = observation;
        if !BinomialKernel::valid_observation(successes, trials) || !Self::valid_theta(*theta) {
            return f64::NAN;
        }
        let Some(successes) = included_count(successes, MAX_CDF_TERMS) else {
            return f64::NAN;
        };
        let Some(trials) = included_count(trials, MAX_CDF_TERMS) else {
            return f64::NAN;
        };
        finite_discrete_crps_from_log_pmf(successes, trials, |value| {
            Self::log_pmf(value as f64, trials as f64, *theta)
        })
    }
}

/// Link-scale predictors for [`BetaBinomial`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BetaBinomialEta {
    /// Mean success-probability predictor.
    pub probability: f64,
    /// Beta precision predictor.
    pub precision: f64,
}

impl ParameterParts<2> for BetaBinomialEta {
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            probability: values[0],
            precision: values[1],
        }
    }
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.probability,
            1 => self.precision,
            _ => unreachable!("beta-binomial eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale mean/precision parameters for [`BetaBinomial`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BetaBinomialTheta {
    /// Mean success probability in `(0, 1)`.
    pub probability: f64,
    /// Positive precision of the beta mixing distribution.
    pub precision: f64,
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{Family, ParameterParts};

    use super::{BetaBinomialEta, BetaBinomialMeanPrecision, BetaBinomialTheta};

    #[test]
    fn gradient_matches_finite_difference() {
        let family = BetaBinomialMeanPrecision::new();
        let eta = BetaBinomialEta::from_array([0.3, 1.4]);
        let (_, gradient) = family.nll_and_gradient_eta([4.0, 10.0], &eta, &mut ());
        for index in 0..2 {
            let epsilon = 1.0e-6;
            let mut lower = [eta.probability, eta.precision];
            let mut upper = lower;
            lower[index] -= epsilon;
            upper[index] += epsilon;
            let lower = BetaBinomialEta::from_array(lower);
            let upper = BetaBinomialEta::from_array(upper);
            let numeric = (family.nll_eta([4.0, 10.0], &upper, &mut ())
                - family.nll_eta([4.0, 10.0], &lower, &mut ()))
                / (2.0 * epsilon);
            assert_relative_eq!(gradient.part(index), numeric, epsilon = 2.0e-7);
        }
    }

    #[test]
    fn cdf_sums_to_one() {
        use gamlss_core::HasCdf;
        let family = BetaBinomialMeanPrecision::new();
        let theta = BetaBinomialTheta {
            probability: 0.4,
            precision: 8.0,
        };
        assert_relative_eq!(family.cdf([10.0, 10.0], &theta), 1.0, epsilon = 1.0e-12);
        let expected = (0_u32..=4)
            .map(|successes| (-family.nll([f64::from(successes), 10.0], &theta, &mut ())).exp())
            .sum::<f64>();
        assert_relative_eq!(family.cdf([4.0, 10.0], &theta), expected, epsilon = 1.0e-13);
        assert_relative_eq!(family.nll([0.0, 0.0], &theta, &mut ()), 0.0);
        assert_relative_eq!(family.cdf([0.0, 0.0], &theta), 1.0);
        assert!(family.nll([11.0, 10.0], &theta, &mut ()).is_infinite());
    }

    #[test]
    fn large_precision_approaches_binomial_likelihood() {
        use gamlss_core::{Family, Logit};

        let family = BetaBinomialMeanPrecision::new();
        let binomial = super::super::binomial::BinomialFixedTrials::<Logit>::try_new(10).unwrap();
        let binomial_theta = super::super::binomial::BinomialTheta { probability: 0.4 };
        for precision in [1.0e10, 1.0e16] {
            let theta = BetaBinomialTheta {
                probability: 0.4,
                precision,
            };
            assert_relative_eq!(
                family.nll([4.0, 10.0], &theta, &mut ()),
                binomial.nll(4.0, &binomial_theta, &mut ()),
                epsilon = 2.0e-8
            );
        }

        let eta = BetaBinomialEta {
            probability: (0.4_f64 / 0.6).ln(),
            precision: 1.0e16_f64.ln(),
        };
        let (_, gradient) = family.nll_and_gradient_eta([3.0, 10.0], &eta, &mut ());
        assert_relative_eq!(gradient.probability, 1.0, epsilon = 2.0e-14);
        assert!(gradient.precision.abs() <= 2.0e-14);
    }
}
