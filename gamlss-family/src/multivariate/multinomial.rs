#![allow(clippy::doc_markdown)]

use gamlss_core::{
    CompilableFamily, Family, FixedDimensionalFamily, HasConditionalCdf, HasMarginalCdf,
    HasObservationDimension, HasRosenblattTransform, ModelError, ObservationView, Probability,
    shape::{ShapeValues, Simplex},
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};
use gamlss_special::{baseline_softmax, is_nonnegative_integer};

#[cfg(feature = "rand")]
use crate::multivariate::count::try_sample_multinomial;
use crate::{
    domain::is_interior_simplex,
    multivariate::count::{
        PerObservationTrials, PreparedFixedTrials, TrialPolicy, validated_total,
    },
    univariate::binomial::BinomialKernel,
};

/// Multinomial family with one fixed positive number of trials for every row.
///
/// Each observation is a `K`-component count vector whose entries must be nonnegative integers summing to [`Self::trials`]. The `K - 1` free predictors are baseline-softmax logits; the final category is the structural baseline.
///
/// For counts $y_1,\ldots,y_K$ and probabilities $p_1,\ldots,p_K$, the probability mass is
///
/// $$
/// P(Y=y)=\frac{n!}{\prod_{k=1}^{K}y_k!}\prod_{k=1}^{K}p_k^{y_k},
/// \qquad n=\sum_{k=1}^{K}y_k.
/// $$
///
/// The moments are $\operatorname{E}(Y_k)=np_k$, $\operatorname{Var}(Y_k)=np_k(1-p_k)$, and $\operatorname{Cov}(Y_j,Y_k)=-np_jp_k$ for $j\ne k$.
///
/// Marginal and ordered conditional CDFs reduce to binomial CDFs. [`HasRosenblattTransform`] exposes their non-randomized discrete PIT values; consequently the final coordinate is exactly one for every valid count vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultinomialFixedTrials<const K: usize> {
    trials: PreparedFixedTrials,
}

impl<const K: usize> MultinomialFixedTrials<K> {
    /// Creates a fixed-trials multinomial family after validating its category count and number of trials.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `K < 2` or `trials == 0`.
    pub fn try_new(trials: u32) -> Result<Self, ModelError> {
        if K < 2 {
            return Err(ModelError::InvalidParameter {
                parameter: "multinomial category count",
                expected: "at least two",
            });
        }
        if trials == 0 {
            return Err(ModelError::InvalidParameter {
                parameter: "multinomial trials",
                expected: "positive",
            });
        }
        Ok(Self {
            trials: PreparedFixedTrials::new(trials),
        })
    }

    /// Creates a fixed-trials multinomial family.
    ///
    /// # Panics
    ///
    /// Panics when `K < 2` or `trials == 0`.
    #[must_use]
    pub fn new(trials: u32) -> Self {
        assert!(
            K >= 2,
            "multinomial family requires at least two categories"
        );
        assert!(trials > 0, "multinomial trials must be positive");
        Self {
            trials: PreparedFixedTrials::new(trials),
        }
    }

    /// Returns the common number of trials.
    #[must_use]
    pub const fn trials(&self) -> u32 {
        self.trials.trials()
    }

    #[inline]
    const fn trial_policy(&self) -> &PreparedFixedTrials {
        &self.trials
    }
}

/// Multinomial family whose number of trials is the sum of each observed count vector.
///
/// This form is the multinomial counterpart of `BinomialVaryingTrials`: rows may have different totals, including zero. Because the row total is not a distribution parameter, this family does not implement simulation or unconditional marginal CDF capabilities.
///
/// Its probability mass and baseline-softmax parameterization are the same as [`MultinomialFixedTrials`], but $n=\sum_k y_k$ is inferred independently for every observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultinomialVaryingTrials<const K: usize> {
    private: (),
}

impl<const K: usize> MultinomialVaryingTrials<K> {
    /// Creates a varying-trials family after checking that at least two categories exist.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `K < 2`.
    pub const fn try_new() -> Result<Self, ModelError> {
        if K < 2 {
            return Err(ModelError::InvalidParameter {
                parameter: "multinomial category count",
                expected: "at least two",
            });
        }
        Ok(Self { private: () })
    }

    /// Creates a varying-trials multinomial family.
    ///
    /// # Panics
    ///
    /// Panics when `K < 2`.
    #[must_use]
    pub const fn new() -> Self {
        assert!(
            K >= 2,
            "multinomial family requires at least two categories"
        );
        Self { private: () }
    }

    #[inline]
    const fn trial_policy(self) -> PerObservationTrials {
        match self {
            Self { private: () } => PerObservationTrials,
        }
    }
}

impl<const K: usize> Default for MultinomialVaryingTrials<K> {
    fn default() -> Self {
        Self::new()
    }
}

/// Baseline-softmax predictors shared by the multinomial families.
///
/// The final array entry is a structural zero retained by the fixed-size carrier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultinomialEta<const K: usize> {
    /// `K - 1` free logits followed by the zero baseline slot.
    pub logits: [f64; K],
}

impl<const K: usize> MultinomialEta<K> {
    /// Creates a predictor carrier and enforces a zero baseline slot.
    #[must_use]
    #[inline]
    pub const fn new(mut logits: [f64; K]) -> Self {
        if K > 0 {
            logits[K - 1] = 0.0;
        }
        Self { logits }
    }
}

/// Natural-scale category probabilities shared by the multinomial families.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultinomialTheta<const K: usize> {
    probabilities: [f64; K],
}

impl<const K: usize> MultinomialTheta<K> {
    /// Creates valid natural-scale multinomial probabilities.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] unless `K >= 2`, every component is finite and strictly positive, and the `f64` component sum is close to one.
    pub fn try_new(probabilities: [f64; K]) -> Result<Self, ModelError> {
        let theta = Self { probabilities };
        if valid_theta(&theta) {
            Ok(theta)
        } else {
            Err(ModelError::InvalidParameter {
                parameter: "multinomial theta",
                expected: "K >= 2, strictly positive finite probabilities, and an f64 sum close to one",
            })
        }
    }

    /// Returns all category probabilities.
    #[must_use]
    #[inline]
    pub const fn probabilities(&self) -> &[f64; K] {
        &self.probabilities
    }

    /// Returns one category probability.
    #[must_use]
    #[inline]
    pub fn probability(&self, category: usize) -> Option<f64> {
        self.probabilities.get(category).copied()
    }
}

#[inline]
fn theta_from_eta<const K: usize>(eta: &MultinomialEta<K>) -> MultinomialTheta<K> {
    MultinomialTheta {
        probabilities: baseline_softmax(eta.logits),
    }
}

#[inline]
fn valid_theta<const K: usize>(theta: &MultinomialTheta<K>) -> bool {
    is_interior_simplex(&theta.probabilities)
}

fn nll_validated<const K: usize>(
    counts: &[f64; K],
    theta: &MultinomialTheta<K>,
    total: f64,
    trial_policy: impl TrialPolicy,
) -> f64 {
    trial_policy.multinomial_nll(counts, total, &theta.probabilities)
}

fn nll_theta<const K: usize>(
    counts: [f64; K],
    theta: &MultinomialTheta<K>,
    trial_policy: impl TrialPolicy,
) -> f64 {
    let Some(total) = validated_total(&counts, trial_policy) else {
        return f64::INFINITY;
    };
    if !valid_theta(theta) {
        return f64::INFINITY;
    }
    nll_validated(&counts, theta, total, trial_policy)
}

const fn nan_eta<const K: usize>() -> MultinomialEta<K> {
    MultinomialEta {
        logits: [f64::NAN; K],
    }
}

fn nll_and_gradient_eta<const K: usize>(
    counts: [f64; K],
    eta: &MultinomialEta<K>,
    trial_policy: impl TrialPolicy,
) -> (f64, MultinomialEta<K>) {
    let theta = theta_from_eta(eta);
    let Some(total) = validated_total(&counts, trial_policy) else {
        return (f64::INFINITY, nan_eta());
    };
    if !valid_theta(&theta) {
        return (f64::INFINITY, nan_eta());
    }
    let nll = nll_validated(&counts, &theta, total, trial_policy);
    if !nll.is_finite() {
        return (nll, nan_eta());
    }

    let mut logits = std::array::from_fn(|category| {
        total.mul_add(theta.probabilities[category], -counts[category])
    });
    logits[K - 1] = 0.0;
    (nll, MultinomialEta { logits })
}

macro_rules! impl_family {
    ($family:ident) => {
        impl<const K: usize> Family for $family<K> {
            type Eta = MultinomialEta<K>;
            type Theta = MultinomialTheta<K>;
            type GradientEta = MultinomialEta<K>;
            type Observation<'obs> = [f64; K];
            type Workspace = ();

            #[inline]
            fn workspace(&self) -> Self::Workspace {}

            fn theta(&self, eta: &Self::Eta, _workspace: &mut ()) -> Self::Theta {
                theta_from_eta(eta)
            }

            fn nll(
                &self,
                counts: Self::Observation<'_>,
                theta: &Self::Theta,
                _workspace: &mut (),
            ) -> f64 {
                nll_theta(counts, theta, self.trial_policy())
            }

            fn nll_and_gradient_eta(
                &self,
                counts: Self::Observation<'_>,
                eta: &Self::Eta,
                _workspace: &mut (),
            ) -> (f64, Self::GradientEta) {
                nll_and_gradient_eta(counts, eta, self.trial_policy())
            }
        }

        impl<const K: usize> FixedDimensionalFamily<K> for $family<K> {}

        impl<const K: usize> HasObservationDimension for $family<K> {
            fn observation_dimension(&self) -> usize {
                K
            }
        }
    };
}

impl_family!(MultinomialFixedTrials);
impl_family!(MultinomialVaryingTrials);

fn initial_logits<'obs, const K: usize, Obs>(
    obs: &'obs Obs,
    trial_policy: impl TrialPolicy,
) -> [f64; K]
where
    Obs: ObservationView<'obs, Observation = [f64; K]> + 'obs,
{
    let mut totals = [0.5; K];
    for row in 0..obs.len() {
        let weight = obs.weight_at(row);
        let counts = obs.observation_at(row);
        if weight > 0.0 && validated_total(&counts, trial_policy).is_some() {
            for (total, count) in totals.iter_mut().zip(counts) {
                *total = weight.mul_add(count, *total);
            }
        }
    }
    let baseline = totals[K - 1].ln();
    let mut logits = totals.map(|total| total.ln() - baseline);
    logits[K - 1] = 0.0;
    logits
}

macro_rules! impl_compilable {
    ($family:ident, $validate:expr) => {
        impl<const K: usize> CompilableFamily for $family<K> {
            type Shape = Simplex<Probability, K>;

            fn eta_from_shape(values: ShapeValues<Self::Shape>) -> Self::Eta {
                MultinomialEta::new(values)
            }

            fn gradient_to_shape(gradient: &Self::GradientEta) -> ShapeValues<Self::Shape> {
                gradient.logits
            }

            fn initial_shape<'obs, Obs>(&self, obs: &'obs Obs) -> ShapeValues<Self::Shape>
            where
                Obs: ObservationView<'obs, Observation = [f64; K]> + 'obs,
            {
                initial_logits(obs, self.trial_policy())
            }

            fn validate_compiled(&self) -> Result<(), ModelError> {
                $validate(self)
            }
        }
    };
}

impl_compilable!(MultinomialFixedTrials, |family: &Self| {
    Self::try_new(family.trials()).map(|_| ())
});
impl_compilable!(MultinomialVaryingTrials, |_family: &Self| {
    Self::try_new().map(|_| ())
});

impl<const K: usize> HasMarginalCdf for MultinomialFixedTrials<K> {
    fn marginal_cdf(&self, component: usize, y: f64, theta: &Self::Theta) -> f64 {
        let Some(probability) = theta.probability(component) else {
            return f64::NAN;
        };
        if !valid_theta(theta) {
            return f64::NAN;
        }
        binomial_cdf_allow_degenerate(y, self.trials.total(), probability)
    }
}

impl<const K: usize> HasConditionalCdf for MultinomialFixedTrials<K> {
    fn conditional_cdf(
        &self,
        component: usize,
        y: f64,
        preceding: &[f64],
        theta: &Self::Theta,
    ) -> f64 {
        if component >= K || preceding.len() < component || !y.is_finite() || !valid_theta(theta) {
            return f64::NAN;
        }
        let preceding_total = preceding
            .iter()
            .copied()
            .take(component)
            .try_fold(0.0, |total, count| {
                is_nonnegative_integer(count).then_some(total + count)
            });
        let Some(preceding_total) = preceding_total else {
            return f64::NAN;
        };
        let trials = self.trials.total();
        if !preceding_total.is_finite() || preceding_total > trials {
            return f64::NAN;
        }
        let remaining_probability = theta.probabilities[component..].iter().sum::<f64>();
        let conditional_probability = theta.probabilities[component] / remaining_probability;
        binomial_cdf_allow_degenerate(
            y,
            trials - preceding_total,
            conditional_probability.clamp(0.0, 1.0),
        )
    }
}

impl<const K: usize> HasRosenblattTransform for MultinomialFixedTrials<K> {
    fn rosenblatt_into(
        &self,
        observation: Self::Observation<'_>,
        theta: &Self::Theta,
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        if out.len() != K {
            return Err(ModelError::ResponseLength {
                expected: K,
                actual: out.len(),
            });
        }
        if validated_total(&observation, self.trial_policy()).is_none() || !valid_theta(theta) {
            out.fill(f64::NAN);
            return Ok(());
        }
        for component in 0..K {
            out[component] =
                self.conditional_cdf(component, observation[component], &observation, theta);
        }
        Ok(())
    }
}

#[cfg(feature = "rand")]
impl<Rng, const K: usize> TrySimulate<Rng> for MultinomialFixedTrials<K>
where
    Rng: rand::Rng,
{
    type Sample = [f64; K];

    fn try_sample(
        &self,
        rng: &mut Rng,
        theta: &Self::Theta,
    ) -> Result<Self::Sample, SimulationError> {
        if !valid_theta(theta) {
            return Err(SimulationError::InvalidParameters("Multinomial theta"));
        }
        try_sample_multinomial(
            rng,
            self.trials(),
            &theta.probabilities,
            "Multinomial conditional binomial",
            "Multinomial sequential sampling",
        )
    }
}

fn binomial_cdf_allow_degenerate(successes: f64, trials: f64, probability: f64) -> f64 {
    if !successes.is_finite()
        || !is_nonnegative_integer(trials)
        || !(0.0..=1.0).contains(&probability)
        || !probability.is_finite()
    {
        return f64::NAN;
    }
    if successes < 0.0 {
        return 0.0;
    }
    let successes = successes.floor();
    if successes >= trials || trials == 0.0 || probability <= 0.0 {
        return 1.0;
    }
    if probability >= 1.0 {
        return 0.0;
    }
    BinomialKernel::cdf(successes, trials, probability)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use approx::assert_relative_eq;
    use gamlss_core::{
        CompilableFamily, DenseDesign, Family, Gamlss, HasConditionalCdf, HasMarginalCdf,
        HasRosenblattTransform, LinearPredictorBlock, NoPenalty, ParameterBlocks, Probability,
        SimplexLogitParameterBlock,
    };
    use gamlss_special::ln_gamma;

    use super::{
        MultinomialEta, MultinomialFixedTrials, MultinomialTheta, MultinomialVaryingTrials,
    };
    use crate::univariate::{BinomialEta, BinomialTheta, BinomialVaryingTrialsProbability};

    fn direct_nll<const K: usize>(counts: [f64; K], probabilities: [f64; K]) -> f64 {
        let total = counts.iter().sum::<f64>();
        counts
            .iter()
            .zip(probabilities)
            .fold(-ln_gamma(total + 1.0), |nll, (count, probability)| {
                (nll + ln_gamma(*count + 1.0)) - count * probability.ln()
            })
    }

    fn assert_gradient<const K: usize, F>(family: &F, counts: [f64; K], eta: MultinomialEta<K>)
    where
        F: for<'obs> Family<
                Observation<'obs> = [f64; K],
                Eta = MultinomialEta<K>,
                GradientEta = MultinomialEta<K>,
            >,
    {
        let (_, gradient) = family.nll_and_gradient_eta(counts, &eta, &mut family.workspace());
        for category in 0..K - 1 {
            let mut lower = eta;
            let mut upper = eta;
            lower.logits[category] -= 1.0e-6;
            upper.logits[category] += 1.0e-6;
            let numeric = (family.nll_eta(counts, &upper, &mut family.workspace())
                - family.nll_eta(counts, &lower, &mut family.workspace()))
                / 2.0e-6;
            assert_relative_eq!(gradient.logits[category], numeric, epsilon = 1.0e-8);
        }
        assert_eq!(gradient.logits[K - 1], 0.0);
    }

    #[test]
    fn fixed_and_varying_gradients_match_finite_difference() {
        let eta = MultinomialEta::new([0.2, -0.4, 7.0]);
        assert_gradient(&MultinomialFixedTrials::<3>::new(10), [2.0, 3.0, 5.0], eta);
        assert_gradient(&MultinomialVaryingTrials::<3>::new(), [4.0, 2.0, 1.0], eta);
    }

    #[test]
    fn likelihood_matches_closed_form_and_binomial_special_case() {
        let family = MultinomialFixedTrials::<3>::new(10);
        let varying = MultinomialVaryingTrials::<3>::new();
        let theta = MultinomialTheta::try_new([0.2, 0.3, 0.5]).unwrap();
        let probability = 2_520.0_f64 * 0.2_f64.powi(2) * 0.3_f64.powi(3) * 0.5_f64.powi(5);
        assert_relative_eq!(
            family.nll([2.0, 3.0, 5.0], &theta, &mut ()),
            -probability.ln(),
            epsilon = 2.0e-14
        );
        for counts in [[2.0, 3.0, 5.0], [0.0, 4.0, 6.0], [9.0, 0.0, 1.0]] {
            assert_eq!(
                family.nll(counts, &theta, &mut ()),
                varying.nll(counts, &theta, &mut ())
            );
        }

        let multinomial = MultinomialVaryingTrials::<2>::new();
        let binomial = BinomialVaryingTrialsProbability::new();
        let multinomial_theta = MultinomialTheta::try_new([0.4, 0.6]).unwrap();
        let binomial_theta = BinomialTheta { probability: 0.4 };
        assert_relative_eq!(
            multinomial.nll([3.0, 7.0], &multinomial_theta, &mut ()),
            binomial.nll([3.0, 10.0], &binomial_theta, &mut ()),
            epsilon = 1.0e-14
        );

        let multinomial_eta = MultinomialEta::new([0.3, 0.0]);
        let binomial_eta = BinomialEta { probability: 0.3 };
        let (multinomial_nll, multinomial_gradient) =
            multinomial.nll_and_gradient_eta([3.0, 7.0], &multinomial_eta, &mut ());
        let (binomial_nll, binomial_gradient) =
            binomial.nll_and_gradient_eta([3.0, 10.0], &binomial_eta, &mut ());
        assert_relative_eq!(multinomial_nll, binomial_nll, epsilon = 1.0e-14);
        assert_relative_eq!(
            multinomial_gradient.logits[0],
            binomial_gradient.probability,
            epsilon = 1.0e-14
        );
        assert_eq!(multinomial_gradient.logits[1], 0.0);
    }

    #[test]
    fn likelihood_matches_direct_log_gamma_formula_across_sparse_counts() {
        let family = MultinomialVaryingTrials::<3>::new();
        for (counts, probabilities) in [
            ([0.0, 0.0, 1.0], [0.2, 0.3, 0.5]),
            ([0.0, 4.0, 6.0], [0.1, 0.35, 0.55]),
            ([1.0, 1.0, 1.0], [0.25, 0.25, 0.5]),
            ([20.0, 30.0, 50.0], [0.21, 0.29, 0.5]),
        ] {
            let theta = MultinomialTheta::try_new(probabilities).unwrap();
            assert_relative_eq!(
                family.nll(counts, &theta, &mut ()),
                direct_nll(counts, probabilities),
                epsilon = 3.0e-13
            );
        }
    }

    #[test]
    fn eta_and_theta_carriers_enforce_simplex_invariants() {
        let eta = MultinomialEta::new([0.2, -0.4, 123.0]);
        assert_eq!(eta.logits[2], 0.0);
        let theta = MultinomialVaryingTrials::<3>::new().theta(&eta, &mut ());
        assert!(theta.probabilities().iter().all(|value| *value > 0.0));
        assert_relative_eq!(theta.probabilities().iter().sum::<f64>(), 1.0);
        assert_eq!(theta.probability(1), Some(theta.probabilities()[1]));
        assert_eq!(theta.probability(3), None);

        assert!(MultinomialTheta::<1>::try_new([1.0]).is_err());
        assert!(MultinomialTheta::try_new([0.0, 0.5, 0.5]).is_err());
        assert!(MultinomialTheta::try_new([0.2, 0.3, 0.4]).is_err());
        assert!(MultinomialTheta::try_new([f64::NAN, 0.5, 0.5]).is_err());
        assert!(MultinomialTheta::try_new([f64::INFINITY, 0.5, 0.5]).is_err());
    }

    #[test]
    fn extreme_finite_logits_keep_nll_and_gradient_finite() {
        let family = MultinomialVaryingTrials::<3>::new();
        let eta = MultinomialEta::new([f64::MAX, -f64::MAX, 0.0]);
        let (nll, gradient) = family.nll_and_gradient_eta([0.0, 1.0, 0.0], &eta, &mut ());
        assert!(nll.is_finite());
        assert!(gradient.logits.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn rejects_invalid_counts_and_accepts_empty_varying_trial_row() {
        let fixed = MultinomialFixedTrials::<3>::new(10);
        let varying = MultinomialVaryingTrials::<3>::new();
        let theta = MultinomialTheta::try_new([0.2, 0.3, 0.5]).unwrap();
        assert!(fixed.nll([2.0, 3.0, 4.0], &theta, &mut ()).is_infinite());
        assert!(varying.nll([2.5, 3.0, 4.0], &theta, &mut ()).is_infinite());
        assert!(varying.nll([-1.0, 2.0, 3.0], &theta, &mut ()).is_infinite());
        let eta = MultinomialEta::new([0.2, -0.4, 0.0]);
        let (nll, gradient) = varying.nll_and_gradient_eta([0.0; 3], &eta, &mut ());
        assert_eq!(nll, 0.0);
        assert_eq!(gradient.logits, [0.0; 3]);
        assert!(MultinomialFixedTrials::<1>::try_new(4).is_err());
        assert!(MultinomialFixedTrials::<3>::try_new(0).is_err());
        assert!(MultinomialVaryingTrials::<1>::try_new().is_err());
    }

    #[test]
    fn large_balanced_likelihood_preserves_multinomial_normalizer() {
        let family = MultinomialVaryingTrials::<3>::new();
        let total = 1.0e16;
        let probabilities = [0.2, 0.3, 0.5];
        let theta = MultinomialTheta::try_new(probabilities).unwrap();
        let counts = probabilities.map(|probability| total * probability);
        let nll = family.nll(counts, &theta, &mut ());
        let expected = f64::midpoint(
            2.0 * (std::f64::consts::TAU * total).ln(),
            probabilities.iter().map(|value| value.ln()).sum::<f64>(),
        );
        assert_relative_eq!(nll, expected, epsilon = 3.0e-13);
    }

    #[test]
    fn initialization_aggregates_category_counts() {
        let family = MultinomialVaryingTrials::<3>::new();
        let observations = [[4.0, 1.0, 0.0], [2.0, 1.0, 2.0], [0.0, 0.0, 0.0]];
        let logits = family.initial_shape(&observations.as_slice());
        assert!(logits[0] > logits[1]);
        assert_relative_eq!(logits[1], logits[2], epsilon = 1.0e-14);
        assert_eq!(logits[2], 0.0);
    }

    #[test]
    fn fixed_trials_compiles_with_one_predictor_per_free_logit() {
        let observations = [
            [3.0, 1.0, 1.0],
            [1.0, 3.0, 1.0],
            [2.0, 1.0, 2.0],
            [0.0, 2.0, 3.0],
        ];
        let logits = SimplexLogitParameterBlock::<Probability, 3, _, _>::new(
            vec![
                LinearPredictorBlock::new(DenseDesign::intercept(observations.len())),
                LinearPredictorBlock::new(DenseDesign::intercept(observations.len())),
            ],
            NoPenalty,
            0,
        );
        let model = Gamlss::try_new_with_observations(
            MultinomialFixedTrials::<3>::new(5),
            ParameterBlocks::new(logits),
            observations.as_slice(),
        )
        .unwrap();
        let beta = [0.2, -0.1];
        let mut gradient = [0.0; 2];
        model.try_value_gradient_into(&beta, &mut gradient).unwrap();
        assert_eq!(model.nparams(), 2);
        for category in 0..2 {
            let mut lower = beta;
            let mut upper = beta;
            lower[category] -= 1.0e-6;
            upper[category] += 1.0e-6;
            let numeric =
                (model.try_value(&upper).unwrap() - model.try_value(&lower).unwrap()) / 2.0e-6;
            assert_relative_eq!(gradient[category], numeric, epsilon = 1.0e-8);
        }
    }

    #[test]
    fn fixed_trials_exposes_binomial_marginals_and_conditionals() {
        let family = MultinomialFixedTrials::<3>::new(5);
        let theta = MultinomialTheta::try_new([0.2, 0.3, 0.5]).unwrap();
        assert_relative_eq!(
            family.marginal_cdf(0, 1.0, &theta),
            0.8_f64.powi(4).mul_add(1.0, 0.8_f64.powi(5)),
            epsilon = 1.0e-14
        );

        let binomial = crate::univariate::BinomialFixedTrialsProbability::try_new(4).unwrap();
        let conditional_theta = BinomialTheta {
            probability: 0.3 / 0.8,
        };
        assert_relative_eq!(
            family.conditional_cdf(1, 2.0, &[1.0], &theta),
            gamlss_core::HasCdf::cdf(&binomial, 2.0, &conditional_theta),
            epsilon = 1.0e-14
        );

        let observation = [1.0, 2.0, 2.0];
        let mut transform = [0.0; 3];
        family
            .rosenblatt_into(observation, &theta, &mut transform)
            .unwrap();
        for component in 0..3 {
            assert_relative_eq!(
                transform[component],
                family.conditional_cdf(component, observation[component], &observation, &theta),
                epsilon = 1.0e-14
            );
        }
        assert_eq!(transform[2], 1.0);
    }

    #[test]
    fn fixed_trial_diagnostics_reject_invalid_queries_without_panicking() {
        let family = MultinomialFixedTrials::<3>::new(5);
        let theta = MultinomialTheta::try_new([0.2, 0.3, 0.5]).unwrap();
        assert_eq!(family.marginal_cdf(0, -1.0, &theta), 0.0);
        assert!(family.marginal_cdf(3, 1.0, &theta).is_nan());
        assert!(family.marginal_cdf(0, f64::NAN, &theta).is_nan());
        assert!(
            family
                .conditional_cdf(3, 1.0, &[1.0, 1.0, 1.0], &theta)
                .is_nan()
        );
        assert!(family.conditional_cdf(2, 1.0, &[1.0], &theta).is_nan());
        assert!(family.conditional_cdf(1, 1.0, &[0.5], &theta).is_nan());
        assert!(family.conditional_cdf(1, 1.0, &[6.0], &theta).is_nan());

        assert!(
            family
                .rosenblatt_into([1.0, 2.0, 2.0], &theta, &mut [0.0; 2])
                .is_err()
        );
        let mut invalid = [0.0; 3];
        family
            .rosenblatt_into([1.0, 2.0, 1.0], &theta, &mut invalid)
            .unwrap();
        assert!(invalid.iter().all(|value| value.is_nan()));
    }

    #[cfg(feature = "rand")]
    #[test]
    fn fixed_trials_sampling_returns_valid_count_vectors() {
        use gamlss_core::TrySimulate;
        use rand::SeedableRng;

        let family = MultinomialFixedTrials::<3>::new(100);
        let theta = MultinomialTheta::try_new([0.2, 0.3, 0.5]).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family.try_sample(&mut rng, &theta).unwrap();
        assert!(
            sample
                .iter()
                .all(|count| super::is_nonnegative_integer(*count))
        );
        assert_eq!(sample.iter().sum::<f64>(), 100.0);

        let invalid = MultinomialTheta {
            probabilities: [0.2, 0.3, 0.4],
        };
        assert!(family.try_sample(&mut rng, &invalid).is_err());
    }
}
