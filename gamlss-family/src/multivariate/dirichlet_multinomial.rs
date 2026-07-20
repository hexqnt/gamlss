#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::needless_range_loop,
    clippy::suboptimal_flops
)]

use std::marker::PhantomData;

use gamlss_core::{
    CompilableFamily, Family, FixedDimensionalFamily, HasCdf, HasConditionalCdf, HasMarginalCdf,
    HasObservationDimension, HasRosenblattTransform, InitialEtaFromTheta, Log, Mean, ModelError,
    ObservationView, PositiveLink, Precision,
    shape::{Product, Scalar, ShapeValues, Simplex},
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};
use gamlss_special::{baseline_softmax, digamma_delta, is_nonnegative_integer, ln_gamma_delta};

#[cfg(feature = "rand")]
use crate::multivariate::{count::try_sample_multinomial, simplex::try_sample_dirichlet};
use crate::{
    domain::is_interior_simplex,
    multivariate::count::{TrialPolicy, negative_log_multinomial_coefficient, validated_total},
    univariate::{BetaBinomialMeanPrecision, BetaBinomialTheta},
};

/// Dirichlet-multinomial family with one fixed positive number of trials per row.
///
/// The simplex mean `mu` and positive precision `phi` define Dirichlet
/// concentrations `alpha_k = mu_k * phi`. Integrating multinomial category
/// probabilities over that Dirichlet distribution produces overdispersed count
/// vectors. As `phi` grows, the distribution approaches the multinomial with
/// probabilities `mu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirichletMultinomialFixedTrials<const K: usize, PrecisionLink = Log> {
    trials: u32,
    marker: PhantomData<PrecisionLink>,
}

impl<const K: usize, PrecisionLink> DirichletMultinomialFixedTrials<K, PrecisionLink>
where
    PrecisionLink: PositiveLink<f64>,
{
    /// Creates a fixed-trials family after validating its category count and trial count.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `K < 2` or `trials == 0`.
    pub const fn try_new(trials: u32) -> Result<Self, ModelError> {
        if K < 2 {
            return Err(ModelError::InvalidParameter {
                parameter: "Dirichlet-multinomial category count",
                expected: "at least two",
            });
        }
        if trials == 0 {
            return Err(ModelError::InvalidParameter {
                parameter: "Dirichlet-multinomial trials",
                expected: "positive",
            });
        }
        Ok(Self {
            trials,
            marker: PhantomData,
        })
    }

    /// Creates a fixed-trials family.
    ///
    /// # Panics
    ///
    /// Panics when `K < 2` or `trials == 0`.
    #[must_use]
    pub const fn new(trials: u32) -> Self {
        assert!(
            K >= 2,
            "Dirichlet-multinomial requires at least two categories"
        );
        assert!(trials > 0, "Dirichlet-multinomial trials must be positive");
        Self {
            trials,
            marker: PhantomData,
        }
    }

    /// Returns the common number of trials.
    #[must_use]
    pub const fn trials(&self) -> u32 {
        self.trials
    }

    #[inline]
    const fn trial_policy(&self) -> TrialPolicy {
        TrialPolicy::Fixed(self.trials)
    }
}

/// Dirichlet-multinomial family whose trial count is inferred from each row.
///
/// Rows may have different totals, including zero. Because the total is
/// observation-side exposure rather than a distribution parameter, this form
/// does not expose unconditional marginal CDFs or simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirichletMultinomialVaryingTrials<const K: usize, PrecisionLink = Log> {
    marker: PhantomData<PrecisionLink>,
}

impl<const K: usize, PrecisionLink> DirichletMultinomialVaryingTrials<K, PrecisionLink>
where
    PrecisionLink: PositiveLink<f64>,
{
    /// Creates a varying-trials family after validating its category count.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `K < 2`.
    pub const fn try_new() -> Result<Self, ModelError> {
        if K < 2 {
            return Err(ModelError::InvalidParameter {
                parameter: "Dirichlet-multinomial category count",
                expected: "at least two",
            });
        }
        Ok(Self {
            marker: PhantomData,
        })
    }

    /// Creates a varying-trials family.
    ///
    /// # Panics
    ///
    /// Panics when `K < 2`.
    #[must_use]
    pub const fn new() -> Self {
        assert!(
            K >= 2,
            "Dirichlet-multinomial requires at least two categories"
        );
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    const fn trial_policy(&self) -> TrialPolicy {
        match self {
            Self { marker: _ } => TrialPolicy::PerObservation,
        }
    }
}

impl<const K: usize, PrecisionLink> Default for DirichletMultinomialVaryingTrials<K, PrecisionLink>
where
    PrecisionLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Link-scale simplex-mean and precision predictors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirichletMultinomialMeanPrecisionEta<const K: usize> {
    /// `K - 1` free mean logits followed by the structural zero baseline.
    pub logits: [f64; K],
    /// Precision predictor.
    pub precision: f64,
}

impl<const K: usize> DirichletMultinomialMeanPrecisionEta<K> {
    /// Creates predictors and normalizes the final baseline slot to zero.
    #[must_use]
    pub const fn new(mut logits: [f64; K], precision: f64) -> Self {
        if K > 0 {
            logits[K - 1] = 0.0;
        }
        Self { logits, precision }
    }
}

/// Natural-scale simplex mean and Dirichlet precision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirichletMultinomialMeanPrecisionTheta<const K: usize> {
    mean: [f64; K],
    precision: f64,
}

impl<const K: usize> DirichletMultinomialMeanPrecisionTheta<K> {
    /// Creates checked natural-scale parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] unless `K >= 2`, `mean` is an
    /// interior simplex and all concentrations `mean_k * precision` are finite
    /// and positive.
    pub fn try_new(mean: [f64; K], precision: f64) -> Result<Self, ModelError> {
        let theta = Self { mean, precision };
        if valid_theta(&theta) {
            Ok(theta)
        } else {
            Err(ModelError::InvalidParameter {
                parameter: "Dirichlet-multinomial theta",
                expected: "K >= 2, an interior simplex mean, and finite positive concentrations",
            })
        }
    }

    /// Returns the category mean vector.
    #[must_use]
    pub const fn mean(&self) -> &[f64; K] {
        &self.mean
    }

    /// Returns the positive Dirichlet precision.
    #[must_use]
    pub const fn precision(&self) -> f64 {
        self.precision
    }

    /// Returns one Dirichlet concentration `mean_k * precision`.
    #[must_use]
    pub fn alpha(&self, category: usize) -> Option<f64> {
        self.mean
            .get(category)
            .map(|probability| probability * self.precision)
    }
}

#[inline]
fn valid_theta<const K: usize>(theta: &DirichletMultinomialMeanPrecisionTheta<K>) -> bool {
    K >= 2
        && theta.precision > 0.0
        && theta.precision.is_finite()
        && is_interior_simplex(&theta.mean)
        && theta.mean.iter().all(|mean| {
            let alpha = mean * theta.precision;
            alpha > 0.0 && alpha.is_finite()
        })
}

fn theta_from_eta<const K: usize, PrecisionLink>(
    eta: &DirichletMultinomialMeanPrecisionEta<K>,
) -> DirichletMultinomialMeanPrecisionTheta<K>
where
    PrecisionLink: PositiveLink<f64>,
{
    DirichletMultinomialMeanPrecisionTheta {
        mean: baseline_softmax(eta.logits),
        precision: PrecisionLink::inverse(eta.precision),
    }
}

fn nll_validated<const K: usize>(
    counts: &[f64; K],
    theta: &DirichletMultinomialMeanPrecisionTheta<K>,
    total: f64,
) -> f64 {
    let mut nll = negative_log_multinomial_coefficient(counts, total)
        + ln_gamma_delta(theta.precision, total);
    for (count, mean) in counts.iter().zip(theta.mean) {
        nll -= ln_gamma_delta(mean * theta.precision, *count);
    }
    nll
}

fn nll_theta<const K: usize>(
    counts: [f64; K],
    theta: &DirichletMultinomialMeanPrecisionTheta<K>,
    policy: TrialPolicy,
) -> f64 {
    let Some(total) = validated_total(&counts, policy) else {
        return f64::INFINITY;
    };
    if !valid_theta(theta) {
        return f64::INFINITY;
    }
    let nll = nll_validated(&counts, theta, total);
    if nll.is_finite() { nll } else { f64::INFINITY }
}

const fn nan_eta<const K: usize>() -> DirichletMultinomialMeanPrecisionEta<K> {
    DirichletMultinomialMeanPrecisionEta {
        logits: [f64::NAN; K],
        precision: f64::NAN,
    }
}

fn nll_and_gradient_eta<const K: usize, PrecisionLink>(
    counts: [f64; K],
    eta: &DirichletMultinomialMeanPrecisionEta<K>,
    policy: TrialPolicy,
) -> (f64, DirichletMultinomialMeanPrecisionEta<K>)
where
    PrecisionLink: PositiveLink<f64>,
{
    let theta = theta_from_eta::<K, PrecisionLink>(eta);
    let Some(total) = validated_total(&counts, policy) else {
        return (f64::INFINITY, nan_eta());
    };
    if !valid_theta(&theta) {
        return (f64::INFINITY, nan_eta());
    }
    let nll = nll_validated(&counts, &theta, total);
    if !nll.is_finite() {
        return (f64::INFINITY, nan_eta());
    }

    let common = digamma_delta(theta.precision, total);
    let d_alpha: [f64; K] = std::array::from_fn(|category| {
        common - digamma_delta(theta.mean[category] * theta.precision, counts[category])
    });
    let d_precision = theta
        .mean
        .iter()
        .zip(d_alpha)
        .map(|(mean, gradient)| mean * gradient)
        .sum::<f64>();
    let mut logits = std::array::from_fn(|category| {
        theta.precision * theta.mean[category] * (d_alpha[category] - d_precision)
    });
    logits[K - 1] = 0.0;

    (
        nll,
        DirichletMultinomialMeanPrecisionEta {
            logits,
            precision: d_precision * PrecisionLink::derivative_inverse(eta.precision),
        },
    )
}

macro_rules! impl_family {
    ($family:ident) => {
        impl<const K: usize, PrecisionLink> Family for $family<K, PrecisionLink>
        where
            PrecisionLink: PositiveLink<f64>,
        {
            type Observation<'obs> = [f64; K];
            type Eta = DirichletMultinomialMeanPrecisionEta<K>;
            type Theta = DirichletMultinomialMeanPrecisionTheta<K>;
            type GradientEta = DirichletMultinomialMeanPrecisionEta<K>;
            type Workspace = ();

            fn workspace(&self) -> Self::Workspace {}

            fn theta(&self, eta: &Self::Eta, _workspace: &mut ()) -> Self::Theta {
                theta_from_eta::<K, PrecisionLink>(eta)
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
                nll_and_gradient_eta::<K, PrecisionLink>(counts, eta, self.trial_policy())
            }
        }

        impl<const K: usize, PrecisionLink> FixedDimensionalFamily<K>
            for $family<K, PrecisionLink>
        where
            PrecisionLink: PositiveLink<f64>,
        {
        }

        impl<const K: usize, PrecisionLink> HasObservationDimension
            for $family<K, PrecisionLink>
        where
            PrecisionLink: PositiveLink<f64>,
        {
            fn observation_dimension(&self) -> usize {
                K
            }
        }
    };
}

impl_family!(DirichletMultinomialFixedTrials);
impl_family!(DirichletMultinomialVaryingTrials);

fn initial_shape<'obs, const K: usize, PrecisionLink, Obs>(
    obs: &'obs Obs,
    policy: TrialPolicy,
) -> ([f64; K], f64)
where
    PrecisionLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    Obs: ObservationView<'obs, Observation = [f64; K]> + 'obs,
{
    let mut category_totals = [0.5; K];
    let mut grand_total = 0.5 * K as f64;
    for row in 0..obs.len() {
        let weight = obs.weight_at(row);
        let counts = obs.observation_at(row);
        if weight > 0.0 && validated_total(&counts, policy).is_some() {
            for (category_total, count) in category_totals.iter_mut().zip(counts) {
                *category_total = weight.mul_add(count, *category_total);
                grand_total = weight.mul_add(count, grand_total);
            }
        }
    }
    let mean = category_totals.map(|total| total / grand_total);

    let mut precision_sum = 0.0;
    let mut precision_count = 0.0;
    for category in 0..K {
        let mut weight_sum = 0.0;
        let mut inverse_trials = 0.0;
        let mut variance = 0.0;
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            let counts = obs.observation_at(row);
            let Some(total) = validated_total(&counts, policy) else {
                continue;
            };
            if weight > 0.0 && total > 0.0 {
                let proportion = counts[category] / total;
                weight_sum += weight;
                inverse_trials += weight / total;
                variance += weight * (proportion - mean[category]).powi(2);
            }
        }
        if weight_sum > 0.0 && mean[category] > 0.0 && mean[category] < 1.0 {
            let ratio = variance / weight_sum / (mean[category] * (1.0 - mean[category]));
            let inverse_trials = inverse_trials / weight_sum;
            let estimate = (1.0 - ratio) / (ratio - inverse_trials);
            if estimate > 0.0 && estimate.is_finite() {
                precision_sum += estimate.clamp(0.1, 1.0e6);
                precision_count += 1.0;
            }
        }
    }
    let precision = if precision_count > 0.0 {
        precision_sum / precision_count
    } else {
        10.0
    };
    let baseline = mean[K - 1].ln();
    let mut logits = mean.map(|value| value.ln() - baseline);
    logits[K - 1] = 0.0;
    (logits, PrecisionLink::initial_eta_from_theta(precision))
}

macro_rules! impl_compilable {
    ($family:ident, $validate:expr) => {
        impl<const K: usize, PrecisionLink> CompilableFamily for $family<K, PrecisionLink>
        where
            PrecisionLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
        {
            type Shape = Product<Simplex<Mean, K>, Scalar<Precision>>;

            fn eta_from_shape(values: ShapeValues<Self::Shape>) -> Self::Eta {
                DirichletMultinomialMeanPrecisionEta::new(values.0, values.1)
            }

            fn gradient_to_shape(gradient: &Self::GradientEta) -> ShapeValues<Self::Shape> {
                (gradient.logits, gradient.precision)
            }

            fn initial_shape<'obs, Obs>(&self, obs: &'obs Obs) -> ShapeValues<Self::Shape>
            where
                Obs: ObservationView<'obs, Observation = [f64; K]> + 'obs,
            {
                initial_shape::<K, PrecisionLink, Obs>(obs, self.trial_policy())
            }

            fn validate_compiled(&self) -> Result<(), ModelError> {
                $validate(self)
            }
        }
    };
}

impl_compilable!(DirichletMultinomialFixedTrials, |family: &Self| {
    Self::try_new(family.trials).map(|_| ())
});
impl_compilable!(DirichletMultinomialVaryingTrials, |_family: &Self| {
    Self::try_new().map(|_| ())
});

impl<const K: usize, PrecisionLink> HasMarginalCdf
    for DirichletMultinomialFixedTrials<K, PrecisionLink>
where
    PrecisionLink: PositiveLink<f64>,
{
    fn marginal_cdf(&self, component: usize, y: f64, theta: &Self::Theta) -> f64 {
        let Some(mean) = theta.mean.get(component).copied() else {
            return f64::NAN;
        };
        if !valid_theta(theta) {
            return f64::NAN;
        }
        BetaBinomialMeanPrecision::new().cdf(
            [y, f64::from(self.trials)],
            &BetaBinomialTheta {
                probability: mean,
                precision: theta.precision,
            },
        )
    }
}

fn degenerate_count_cdf(y: f64, count: f64) -> f64 {
    if !y.is_finite() {
        f64::NAN
    } else if y < count {
        0.0
    } else {
        1.0
    }
}

impl<const K: usize, PrecisionLink> HasConditionalCdf
    for DirichletMultinomialFixedTrials<K, PrecisionLink>
where
    PrecisionLink: PositiveLink<f64>,
{
    fn conditional_cdf(
        &self,
        component: usize,
        y: f64,
        preceding: &[f64],
        theta: &Self::Theta,
    ) -> f64 {
        if component >= K || preceding.len() < component || !valid_theta(theta) {
            return f64::NAN;
        }
        let Some(used) = preceding
            .iter()
            .copied()
            .take(component)
            .try_fold(0.0, |sum, count| {
                is_nonnegative_integer(count).then_some(sum + count)
            })
        else {
            return f64::NAN;
        };
        let remaining_trials = f64::from(self.trials) - used;
        if remaining_trials < 0.0 || !remaining_trials.is_finite() {
            return f64::NAN;
        }
        if component + 1 == K {
            return degenerate_count_cdf(y, remaining_trials);
        }
        let remaining_mean = theta.mean[component..].iter().sum::<f64>();
        BetaBinomialMeanPrecision::new().cdf(
            [y, remaining_trials],
            &BetaBinomialTheta {
                probability: theta.mean[component] / remaining_mean,
                precision: theta.precision * remaining_mean,
            },
        )
    }
}

impl<const K: usize, PrecisionLink> HasRosenblattTransform
    for DirichletMultinomialFixedTrials<K, PrecisionLink>
where
    PrecisionLink: PositiveLink<f64>,
{
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
impl<Rng, const K: usize, PrecisionLink> TrySimulate<Rng>
    for DirichletMultinomialFixedTrials<K, PrecisionLink>
where
    Rng: rand::Rng,
    PrecisionLink: PositiveLink<f64>,
{
    type Sample = [f64; K];

    fn try_sample(
        &self,
        rng: &mut Rng,
        theta: &Self::Theta,
    ) -> Result<Self::Sample, SimulationError> {
        if !valid_theta(theta) {
            return Err(SimulationError::InvalidParameters(
                "Dirichlet-multinomial theta",
            ));
        }
        let concentrations = theta.mean.map(|mean| mean * theta.precision);
        let probabilities = try_sample_dirichlet(
            rng,
            &concentrations,
            "Dirichlet-multinomial concentration",
            "Dirichlet-multinomial gamma normalization",
        )?;
        try_sample_multinomial(
            rng,
            self.trials,
            &probabilities,
            "Dirichlet-multinomial conditional binomial",
            "Dirichlet-multinomial sequential sampling",
        )
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use approx::assert_relative_eq;
    use gamlss_core::{CompilableFamily, Family, HasConditionalCdf, HasMarginalCdf};

    use super::{
        DirichletMultinomialFixedTrials, DirichletMultinomialMeanPrecisionEta,
        DirichletMultinomialMeanPrecisionTheta, DirichletMultinomialVaryingTrials,
    };
    use crate::{BetaBinomialEta, BetaBinomialMeanPrecision, BetaBinomialTheta};

    fn assert_gradient<const K: usize, F>(
        family: &F,
        counts: [f64; K],
        eta: DirichletMultinomialMeanPrecisionEta<K>,
    ) where
        F: for<'obs> Family<
                Observation<'obs> = [f64; K],
                Eta = DirichletMultinomialMeanPrecisionEta<K>,
                GradientEta = DirichletMultinomialMeanPrecisionEta<K>,
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
            assert_relative_eq!(gradient.logits[category], numeric, epsilon = 2.0e-7);
        }
        let mut lower = eta;
        let mut upper = eta;
        lower.precision -= 1.0e-6;
        upper.precision += 1.0e-6;
        let numeric = (family.nll_eta(counts, &upper, &mut family.workspace())
            - family.nll_eta(counts, &lower, &mut family.workspace()))
            / 2.0e-6;
        assert_relative_eq!(gradient.precision, numeric, epsilon = 2.0e-7);
        assert_eq!(gradient.logits[K - 1], 0.0);
    }

    #[test]
    fn fixed_and_varying_gradients_match_finite_difference() {
        let eta = DirichletMultinomialMeanPrecisionEta::new([0.2, -0.3, 0.0], 8.0_f64.ln());
        assert_gradient(
            &DirichletMultinomialFixedTrials::<3>::new(10),
            [2.0, 3.0, 5.0],
            eta,
        );
        assert_gradient(
            &DirichletMultinomialVaryingTrials::<3>::new(),
            [4.0, 2.0, 1.0],
            eta,
        );
    }

    #[test]
    fn two_categories_match_beta_binomial() {
        let family = DirichletMultinomialFixedTrials::<2>::new(10);
        let theta = DirichletMultinomialMeanPrecisionTheta::try_new([0.4, 0.6], 8.0).unwrap();
        let beta = BetaBinomialMeanPrecision::new();
        let beta_theta = BetaBinomialTheta {
            probability: 0.4,
            precision: 8.0,
        };
        assert_relative_eq!(
            family.nll([3.0, 7.0], &theta, &mut ()),
            beta.nll([3.0, 10.0], &beta_theta, &mut ()),
            epsilon = 2.0e-14
        );

        let eta = DirichletMultinomialMeanPrecisionEta::new([0.3, 0.0], 8.0_f64.ln());
        let beta_eta = BetaBinomialEta {
            probability: 0.3,
            precision: 8.0_f64.ln(),
        };
        let (dirichlet_multinomial_nll, dirichlet_multinomial_gradient) =
            family.nll_and_gradient_eta([3.0, 7.0], &eta, &mut ());
        let (beta_binomial_nll, beta_binomial_gradient) =
            beta.nll_and_gradient_eta([3.0, 10.0], &beta_eta, &mut ());
        assert_relative_eq!(
            dirichlet_multinomial_nll,
            beta_binomial_nll,
            epsilon = 2.0e-14
        );
        assert_relative_eq!(
            dirichlet_multinomial_gradient.logits[0],
            beta_binomial_gradient.probability,
            epsilon = 2.0e-14
        );
        assert_relative_eq!(
            dirichlet_multinomial_gradient.precision,
            beta_binomial_gradient.precision,
            epsilon = 2.0e-14
        );
        assert_eq!(dirichlet_multinomial_gradient.logits[1], 0.0);
    }

    #[test]
    fn likelihood_matches_direct_multivariate_log_gamma_formula() {
        let family = DirichletMultinomialFixedTrials::<3>::new(10);
        let theta = DirichletMultinomialMeanPrecisionTheta::try_new([0.2, 0.3, 0.5], 8.0).unwrap();
        let counts = [2.0, 3.0, 5.0];
        let alpha = [1.6, 2.4, 4.0];
        let log_probability = gamlss_special::ln_gamma(11.0)
            - counts
                .iter()
                .map(|count| gamlss_special::ln_gamma(*count + 1.0))
                .sum::<f64>()
            + gamlss_special::ln_gamma(8.0)
            - gamlss_special::ln_gamma(18.0)
            + counts
                .iter()
                .zip(alpha)
                .map(|(count, alpha)| {
                    gamlss_special::ln_gamma(*count + alpha) - gamlss_special::ln_gamma(alpha)
                })
                .sum::<f64>();
        assert_relative_eq!(
            family.nll(counts, &theta, &mut ()),
            -log_probability,
            epsilon = 2.0e-14
        );
    }

    #[test]
    fn concentrated_limit_approaches_multinomial() {
        use crate::multivariate::{MultinomialFixedTrials, MultinomialTheta};

        let family = DirichletMultinomialFixedTrials::<3>::new(10);
        let theta =
            DirichletMultinomialMeanPrecisionTheta::try_new([0.2, 0.3, 0.5], 1.0e16).unwrap();
        let multinomial = MultinomialFixedTrials::<3>::new(10);
        let multinomial_theta = MultinomialTheta::try_new([0.2, 0.3, 0.5]).unwrap();
        assert_relative_eq!(
            family.nll([2.0, 3.0, 5.0], &theta, &mut ()),
            multinomial.nll([2.0, 3.0, 5.0], &multinomial_theta, &mut ()),
            epsilon = 2.0e-8
        );
    }

    #[test]
    fn fixed_marginal_and_conditionals_reduce_to_beta_binomial() {
        let family = DirichletMultinomialFixedTrials::<3>::new(10);
        let theta = DirichletMultinomialMeanPrecisionTheta::try_new([0.2, 0.3, 0.5], 8.0).unwrap();
        let beta = BetaBinomialMeanPrecision::new();
        assert_relative_eq!(
            family.marginal_cdf(0, 2.0, &theta),
            gamlss_core::HasCdf::cdf(
                &beta,
                [2.0, 10.0],
                &BetaBinomialTheta {
                    probability: 0.2,
                    precision: 8.0,
                },
            ),
            epsilon = 1.0e-14
        );
        assert_relative_eq!(
            family.conditional_cdf(1, 3.0, &[2.0], &theta),
            gamlss_core::HasCdf::cdf(
                &beta,
                [3.0, 8.0],
                &BetaBinomialTheta {
                    probability: 0.3 / 0.8,
                    precision: 6.4,
                },
            ),
            epsilon = 1.0e-14
        );
        assert_eq!(family.conditional_cdf(2, 5.0, &[2.0, 3.0], &theta), 1.0);
    }

    #[test]
    fn domains_and_initialization_are_checked() {
        let fixed = DirichletMultinomialFixedTrials::<3>::new(10);
        let varying = DirichletMultinomialVaryingTrials::<3>::new();
        let theta = DirichletMultinomialMeanPrecisionTheta::try_new([0.2, 0.3, 0.5], 8.0).unwrap();
        assert!(fixed.nll([2.0, 3.0, 4.0], &theta, &mut ()).is_infinite());
        assert!(varying.nll([2.5, 3.0, 4.0], &theta, &mut ()).is_infinite());
        assert_eq!(varying.nll([0.0; 3], &theta, &mut ()), 0.0);
        assert!(DirichletMultinomialFixedTrials::<1>::try_new(2).is_err());
        assert!(DirichletMultinomialFixedTrials::<2>::try_new(0).is_err());
        assert!(DirichletMultinomialMeanPrecisionTheta::try_new([0.0, 1.0], 8.0).is_err());

        let observations = [[8.0, 1.0, 1.0], [6.0, 2.0, 2.0], [7.0, 1.0, 2.0]];
        let (logits, precision) = fixed.initial_shape(&observations.as_slice());
        assert!(logits[0] > logits[1]);
        assert_eq!(logits[2], 0.0);
        assert!(precision.is_finite());
    }

    #[cfg(feature = "rand")]
    #[test]
    fn sampling_returns_valid_fixed_total_counts() {
        use gamlss_core::TrySimulate;
        use rand::SeedableRng;

        let family = DirichletMultinomialFixedTrials::<3>::new(100);
        let theta = DirichletMultinomialMeanPrecisionTheta::try_new([0.2, 0.3, 0.5], 8.0).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(21);
        let sample = family.try_sample(&mut rng, &theta).unwrap();
        assert_eq!(sample.iter().sum::<f64>(), 100.0);
        assert!(
            sample
                .iter()
                .all(|value| value.fract() == 0.0 && *value >= 0.0)
        );
    }
}
