//! Shared count-vector helpers for multinomial-like families.

use gamlss_special::{is_nonnegative_integer, ln_gamma_stirling_residual};

#[cfg(feature = "rand")]
use gamlss_core::SimulationError;

/// Prepared invariants for count vectors with one trial count per model.
///
/// The multinomial saddle-point normalizer contains a term depending only on
/// the fixed trial count. Keeping it beside `trials` avoids one logarithm and
/// one Stirling-residual evaluation per observation.
#[derive(Debug, Clone, Copy)]
pub(super) struct PreparedFixedTrials {
    trials: u32,
    total: f64,
    multinomial_nll_constant: f64,
}

impl PreparedFixedTrials {
    pub(super) fn new(trials: u32) -> Self {
        debug_assert!(trials > 0);
        let total = f64::from(trials);
        let log_total = total.ln();
        let residual = ln_gamma_stirling_residual(total);
        Self {
            trials,
            total,
            multinomial_nll_constant: -residual - log_total,
        }
    }

    #[inline]
    pub(super) const fn trials(self) -> u32 {
        self.trials
    }

    #[inline]
    pub(super) const fn total(self) -> f64 {
        self.total
    }

    #[inline]
    const fn cached_multinomial_nll_constant(self) -> f64 {
        self.multinomial_nll_constant
    }
}

impl PartialEq for PreparedFixedTrials {
    fn eq(&self, other: &Self) -> bool {
        self.trials == other.trials
    }
}

impl Eq for PreparedFixedTrials {}

/// Count-total policy used by multinomial-like likelihood kernels.
pub(super) trait TrialPolicy: Copy {
    fn expected_total(self) -> Option<f64>;

    fn multinomial_nll_constant(self, total: f64) -> f64 {
        -ln_gamma_stirling_residual(total) - total.ln()
    }

    fn multinomial_nll_at_empirical_probabilities<const K: usize>(
        self,
        counts: &[f64; K],
        total: f64,
    ) -> f64 {
        if total == 0.0 {
            return 0.0;
        }
        let mut nll = self.multinomial_nll_constant(total);
        for count in counts.iter().copied().filter(|count| *count > 0.0) {
            nll += ln_gamma_stirling_residual(count) + count.ln();
        }
        nll
    }

    /// Returns `sum(ln(y_k!)) - ln(n!)` without subtracting large log-gamma values.
    fn negative_log_multinomial_coefficient<const K: usize>(
        self,
        counts: &[f64; K],
        total: f64,
    ) -> f64 {
        let empirical_nll = self.multinomial_nll_at_empirical_probabilities(counts, total);
        if total == 0.0 {
            return empirical_nll;
        }
        counts
            .iter()
            .copied()
            .filter(|count| *count > 0.0)
            .fold(empirical_nll, |value, count| {
                count.mul_add((count / total).ln(), value)
            })
    }

    fn multinomial_nll<const K: usize>(
        self,
        counts: &[f64; K],
        total: f64,
        probabilities: &[f64; K],
    ) -> f64 {
        let empirical_nll = self.multinomial_nll_at_empirical_probabilities(counts, total);
        if total == 0.0 {
            return empirical_nll;
        }
        let proportions = counts.map(|count| count / total);
        total.mul_add(
            gamlss_special::categorical_kl(&proportions, probabilities),
            empirical_nll,
        )
    }
}

impl TrialPolicy for &PreparedFixedTrials {
    fn expected_total(self) -> Option<f64> {
        Some(self.total())
    }

    fn multinomial_nll_constant(self, _total: f64) -> f64 {
        self.cached_multinomial_nll_constant()
    }
}

/// A fixed trial count without a prepared likelihood normalizer.
#[derive(Debug, Clone, Copy)]
pub(super) struct FixedTrialCount(u32);

impl FixedTrialCount {
    pub(super) const fn new(trials: u32) -> Self {
        Self(trials)
    }
}

impl TrialPolicy for FixedTrialCount {
    fn expected_total(self) -> Option<f64> {
        Some(f64::from(self.0))
    }
}

/// A trial count inferred independently from every observation.
#[derive(Debug, Clone, Copy)]
pub(super) struct PerObservationTrials;

impl TrialPolicy for PerObservationTrials {
    fn expected_total(self) -> Option<f64> {
        None
    }
}

/// Validates an integer-valued count vector and returns its total.
#[allow(clippy::float_cmp)] // Fixed u32 totals are represented exactly by f64.
pub(super) fn validated_total<const K: usize>(
    counts: &[f64; K],
    policy: impl TrialPolicy,
) -> Option<f64> {
    if K < 2 || counts.iter().any(|count| !is_nonnegative_integer(*count)) {
        return None;
    }
    let total = counts.iter().sum::<f64>();
    if !total.is_finite()
        || policy
            .expected_total()
            .is_some_and(|expected| total != expected)
    {
        return None;
    }
    Some(total)
}

/// Samples multinomial counts through sequential conditional binomials.
#[cfg(feature = "rand")]
pub(super) fn try_sample_multinomial<Rng, const K: usize>(
    rng: &mut Rng,
    trials: u32,
    probabilities: &[f64; K],
    backend_context: &'static str,
    numerical_context: &'static str,
) -> Result<[f64; K], SimulationError>
where
    Rng: rand::Rng,
{
    let mut out = [0.0; K];
    let mut remaining_trials = u64::from(trials);
    for category in 0..K.saturating_sub(1) {
        let remaining_probability = probabilities[category..].iter().sum::<f64>();
        if remaining_probability <= 0.0 || !remaining_probability.is_finite() {
            return Err(SimulationError::NumericalFailure(numerical_context));
        }
        let conditional = probabilities[category] / remaining_probability;
        if !conditional.is_finite() {
            return Err(SimulationError::NumericalFailure(numerical_context));
        }
        let distribution = rand_distr::Binomial::new(remaining_trials, conditional.clamp(0.0, 1.0))
            .map_err(|_| SimulationError::BackendRejected(backend_context))?;
        let count = rand_distr::Distribution::sample(&distribution, rng);
        let count = u32::try_from(count)
            .map_err(|_| SimulationError::NumericalFailure(numerical_context))?;
        out[category] = f64::from(count);
        remaining_trials -= u64::from(count);
    }
    let remaining_trials = u32::try_from(remaining_trials)
        .map_err(|_| SimulationError::NumericalFailure(numerical_context))?;
    if K > 0 {
        out[K - 1] = f64::from(remaining_trials);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::PreparedFixedTrials;

    #[test]
    fn cached_constant_matches_direct_saddle_point_term() {
        for trials in [1, 2, 7, 20, 1_000, u32::MAX] {
            let fixed = PreparedFixedTrials::new(trials);
            let total = f64::from(trials);
            assert_relative_eq!(
                fixed.cached_multinomial_nll_constant(),
                -gamlss_special::ln_gamma_stirling_residual(total) - total.ln(),
                epsilon = 1.0e-15,
            );
        }
    }
}
