//! Generalized inverses and bracketed CDF inversion.

use crate::probability::is_probability;

/// Generalized inverse for a discrete CDF on non-negative integer support.
///
/// Returns `NaN` when `p` is outside `[0, 1]`, when the supplied CDF returns a
/// non-finite value, or when `max_count` is reached before the CDF reaches `p`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn discrete_quantile<F>(p: f64, max_count: u64, mut cdf: F) -> f64
where
    F: FnMut(u64) -> f64,
{
    if !is_probability(p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }

    let cdf_zero = cdf(0);
    if !cdf_zero.is_finite() {
        return f64::NAN;
    }
    if cdf_zero >= p {
        return 0.0;
    }
    if max_count == 0 {
        return f64::NAN;
    }

    let mut high = 1_u64.min(max_count);
    loop {
        let cdf_high = cdf(high);
        if !cdf_high.is_finite() {
            return f64::NAN;
        }
        if cdf_high >= p {
            break;
        }
        if high == max_count {
            return f64::NAN;
        }
        high = high.saturating_mul(2).min(max_count);
    }

    let mut low = 0_u64;
    while low < high {
        let mid = low + (high - low) / 2;
        let cdf_mid = cdf(mid);
        if !cdf_mid.is_finite() {
            return f64::NAN;
        }
        if cdf_mid < p {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    low as f64
}

/// Inverts a monotone CDF on a finite closed interval by bisection.
///
/// Returns the interval boundary for `p == 0` or `p == 1`, and `NaN` for
/// invalid probabilities, non-finite bounds, reversed bounds, non-finite CDF
/// evaluations, or endpoints that do not bracket `p`.
#[must_use]
#[allow(clippy::float_cmp)]
pub fn invert_bounded_cdf<F>(p: f64, lower: f64, upper: f64, mut cdf: F) -> f64
where
    F: FnMut(f64) -> f64,
{
    if !is_probability(p) || !lower.is_finite() || !upper.is_finite() || lower > upper {
        return f64::NAN;
    }
    if p == 0.0 {
        return lower;
    }
    if p == 1.0 {
        return upper;
    }

    let mut low = lower;
    let mut high = upper;
    let mut cdf_low = cdf(low);
    let mut cdf_high = cdf(high);
    if !cdf_low.is_finite()
        || !cdf_high.is_finite()
        || cdf_low > cdf_high
        || p < cdf_low
        || p > cdf_high
    {
        return f64::NAN;
    }

    for _ in 0..120 {
        let mid = low.midpoint(high);
        let cdf_mid = cdf(mid);
        if !cdf_mid.is_finite() {
            return f64::NAN;
        }
        if cdf_mid < p {
            low = mid;
            cdf_low = cdf_mid;
        } else {
            high = mid;
            cdf_high = cdf_mid;
        }
    }

    let mid = low.midpoint(high);
    let cdf_mid = cdf(mid);
    if !cdf_mid.is_finite() {
        return f64::NAN;
    }

    let low_error = (cdf_low - p).abs();
    let mid_error = (cdf_mid - p).abs();
    let high_error = (cdf_high - p).abs();
    if low_error <= mid_error && low_error <= high_error {
        low
    } else if high_error <= mid_error {
        high
    } else {
        mid
    }
}

/// Inverts a monotone CDF on `[0, +inf)` by bracketing and bisection.
///
/// Returns `0` for `p == 0`, `+inf` for `p == 1`, and `NaN` for invalid
/// probabilities, non-finite CDF values, or failed finite bracketing for
/// interior probabilities.
#[must_use]
pub fn invert_positive_cdf<F>(p: f64, mut cdf: F) -> f64
where
    F: FnMut(f64) -> f64,
{
    if !is_probability(p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    #[allow(clippy::float_cmp)]
    if p == 1.0 {
        return f64::INFINITY;
    }

    let mut high = 1.0;
    loop {
        let cdf_high = cdf(high);
        if !cdf_high.is_finite() {
            return f64::NAN;
        }
        if cdf_high >= p {
            break;
        }
        high *= 2.0;
        if !high.is_finite() {
            return f64::NAN;
        }
    }

    invert_bounded_cdf(p, 0.0, high, cdf)
}

/// Inverts a monotone CDF on the real line by bracketing and bisection.
///
/// Returns infinite tails for boundary probabilities and `NaN` for invalid
/// probabilities, non-finite CDF values, or failed finite bracketing for
/// interior probabilities.
#[must_use]
pub fn invert_real_cdf<F>(p: f64, mut cdf: F) -> f64
where
    F: FnMut(f64) -> f64,
{
    if !is_probability(p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    #[allow(clippy::float_cmp)]
    if p == 1.0 {
        return f64::INFINITY;
    }

    let mut low = -1.0;
    loop {
        let cdf_low = cdf(low);
        if !cdf_low.is_finite() {
            return f64::NAN;
        }
        if cdf_low <= p {
            break;
        }
        low *= 2.0;
        if !low.is_finite() {
            return f64::NAN;
        }
    }

    let mut high = 1.0;
    loop {
        let cdf_high = cdf(high);
        if !cdf_high.is_finite() {
            return f64::NAN;
        }
        if cdf_high >= p {
            break;
        }
        high *= 2.0;
        if !high.is_finite() {
            return f64::NAN;
        }
    }

    invert_bounded_cdf(p, low, high, cdf)
}
