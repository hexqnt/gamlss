//! Probability transforms, count domains and stable divergences.

#[inline]
pub fn is_probability(value: f64) -> bool {
    (0.0..=1.0).contains(&value) && value.is_finite()
}

#[inline]
pub const fn clamp_probability(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

/// Baseline softmax with a representable strictly-positive result.
///
/// The final logit is fixed to zero. Finite logit differences that would make
/// `exp` underflow are saturated at the smallest normal exponent, preserving
/// the interior-simplex invariant in `f64`.
#[must_use]
pub fn baseline_softmax<const D: usize>(mut logits: [f64; D]) -> [f64; D] {
    if D == 0 || logits.iter().any(|value| !value.is_finite()) {
        return [f64::NAN; D];
    }
    logits[D - 1] = 0.0;
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_exponent = f64::MIN_POSITIVE.ln();
    let mut weights = [0.0; D];
    let mut sum = 0.0;
    for (weight, logit) in weights.iter_mut().zip(logits) {
        *weight = (logit - max).max(min_exponent).exp();
        sum += *weight;
    }
    for weight in &mut weights {
        *weight /= sum;
    }
    weights
}

/// Returns `true` for finite counts represented on the shared `f64` observation path.
#[must_use]
#[inline]
pub fn is_nonnegative_integer(value: f64) -> bool {
    value >= 0.0 && value.is_finite() && value.fract() == 0.0
}

/// Converts a finite CDF query point into the largest included count.
///
/// Returning `None` for non-finite or excessively large query points keeps
/// discrete CDF implementations from doing unbounded work.
#[must_use]
#[inline]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn included_count(value: f64, max_terms: u64) -> Option<u64> {
    if !value.is_finite() {
        return None;
    }
    if value < 0.0 {
        return Some(0);
    }

    let count = value.floor();
    if count > max_terms as f64 {
        None
    } else {
        Some(count as u64)
    }
}

/// Stable two-term log-space addition.
#[must_use]
#[inline]
pub fn log_add_exp(log_left: f64, log_right: f64) -> f64 {
    if log_left.is_nan() || log_right.is_nan() {
        return f64::NAN;
    }
    if log_left == f64::NEG_INFINITY {
        return log_right;
    }
    if log_right == f64::NEG_INFINITY {
        return log_left;
    }
    if log_left == f64::INFINITY || log_right == f64::INFINITY {
        return f64::INFINITY;
    }

    let max = log_left.max(log_right);
    max + (-(log_left - log_right).abs()).exp().ln_1p()
}

/// Kullback-Leibler divergence between Bernoulli probabilities.
///
/// Both probabilities must lie strictly inside `(0, 1)`. A local series
/// preserves the quadratic divergence when the probabilities are nearly
/// equal and the two first-order log terms would otherwise cancel.
#[must_use]
pub fn bernoulli_kl(probability: f64, reference: f64) -> f64 {
    if !(0.0..1.0).contains(&probability)
        || !(0.0..1.0).contains(&reference)
        || !probability.is_finite()
        || !reference.is_finite()
    {
        return f64::NAN;
    }

    let complement = 1.0 - probability;
    let difference = reference - probability;
    let relative_probability = difference / probability;
    let relative_complement = -difference / complement;
    if relative_probability.abs().max(relative_complement.abs()) <= 0.25 {
        let mut probability_power = relative_probability * relative_probability;
        let mut complement_power = relative_complement * relative_complement;
        let mut sum = 0.0;
        for order in 2..=128 {
            let order_f = f64::from(order);
            let sign = if order % 2 == 0 { 1.0 } else { -1.0 };
            let term = sign * probability.mul_add(probability_power, complement * complement_power)
                / order_f;
            let magnitude = probability
                .mul_add(probability_power.abs(), complement * complement_power.abs())
                / order_f;
            sum += term;
            if magnitude <= f64::EPSILON * sum.abs() {
                break;
            }
            probability_power *= relative_probability;
            complement_power *= relative_complement;
        }
        sum.max(0.0)
    } else {
        (-probability).mul_add(
            relative_probability.ln_1p(),
            -complement * relative_complement.ln_1p(),
        )
    }
}

/// Kullback-Leibler divergence between categorical probability vectors.
///
/// `probability` may lie on the simplex boundary, while every `reference` component must be strictly positive. The vectors are expected to have the same normalization; a local series preserves the divergence when their components are nearly equal and direct log-ratio terms would cancel.
///
/// Returns `NaN` for an empty vector, a negative or non-finite probability, a non-positive or non-finite reference component, a non-finite component sum, or an all-zero probability vector.
#[must_use]
pub fn categorical_kl<const K: usize>(probability: &[f64; K], reference: &[f64; K]) -> f64 {
    let probability_sum = probability.iter().sum::<f64>();
    let reference_sum = reference.iter().sum::<f64>();
    if K == 0
        || probability
            .iter()
            .any(|value| *value < 0.0 || !value.is_finite())
        || reference
            .iter()
            .any(|value| *value <= 0.0 || !value.is_finite())
        || probability_sum <= 0.0
        || !probability_sum.is_finite()
        || !reference_sum.is_finite()
    {
        return f64::NAN;
    }

    let relative: [f64; K] = std::array::from_fn(|category| {
        let value = probability[category];
        if value > 0.0 {
            (reference[category] - value) / value
        } else {
            0.0
        }
    });
    let use_series = probability
        .iter()
        .zip(relative.iter())
        .all(|(value, relative)| *value <= 0.0 || relative.abs() <= 0.25);
    if use_series {
        let mut powers = relative;
        let mut sum = 0.0;
        for order in 1..=128 {
            let order_f = f64::from(order);
            let weighted_power = probability
                .iter()
                .zip(powers.iter())
                .map(|(weight, power)| weight * power)
                .sum::<f64>();
            let magnitude = probability
                .iter()
                .zip(powers.iter())
                .map(|(weight, power)| weight * power.abs())
                .sum::<f64>()
                / order_f;
            let sign = if order % 2 == 0 { 1.0 } else { -1.0 };
            sum += sign * weighted_power / order_f;
            if order > 1 && magnitude <= f64::EPSILON * sum.abs() {
                break;
            }
            for (power, relative) in powers.iter_mut().zip(relative.iter()) {
                *power *= relative;
            }
        }
        sum
    } else {
        probability
            .iter()
            .zip(reference)
            .filter(|(value, _)| **value > 0.0)
            .map(|(value, reference)| value * (value.ln() - reference.ln()))
            .sum()
    }
}
