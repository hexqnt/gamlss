//! Gamma, digamma and incomplete-gamma numerical kernels.

use crate::normal::{unit_normal_cdf, unit_normal_log_pdf};
use crate::probability::clamp_probability;
use crate::{HALF_LOG_2_PI, INV_SQRT_2_PI};

const LANCZOS_SHIFT: f64 = 6.5;

const LANCZOS_COEFFICIENTS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

#[inline]
fn lanczos_sum(value: f64) -> f64 {
    let shifted = value - 1.0;
    let mut sum = LANCZOS_COEFFICIENTS[0];

    #[allow(clippy::cast_precision_loss)]
    for (index, coefficient) in LANCZOS_COEFFICIENTS.iter().copied().enumerate().skip(1) {
        sum += coefficient / (shifted + index as f64);
    }

    sum
}

/// Natural logarithm of the absolute gamma function via the Lanczos approximation.
///
/// Returns `NaN` at poles and for non-finite negative inputs. For positive
/// inputs this is the usual `ln(Gamma(x))`; for negative non-integers it is
/// `ln(abs(Gamma(x)))`.
#[must_use]
#[inline]
pub fn ln_gamma(value: f64) -> f64 {
    if value == f64::INFINITY {
        return f64::INFINITY;
    }
    if !value.is_finite() || (value <= 0.0 && value.fract() == 0.0) {
        return f64::NAN;
    }

    if value < 0.5 {
        let sin_pi = (std::f64::consts::PI * value).sin();
        if sin_pi == 0.0 {
            return f64::NAN;
        }
        return std::f64::consts::PI.ln() - sin_pi.abs().ln() - ln_gamma(1.0 - value);
    }

    let x = lanczos_sum(value);
    let t = value + LANCZOS_SHIFT;

    (value - 0.5).mul_add(t.ln(), HALF_LOG_2_PI) - t + x.ln()
}

/// Difference `ln(Gamma(base + increment)) - ln(Gamma(base))` for positive arguments.
///
/// This avoids cancellation when `base` is large and `increment` is small, which is common in beta normalizing constants and gamma-ratio terms.
#[must_use]
#[inline]
pub fn ln_gamma_delta(base: f64, increment: f64) -> f64 {
    if base <= 0.0 || increment < 0.0 || !base.is_finite() || !increment.is_finite() {
        return f64::NAN;
    }
    if increment == 0.0 {
        return 0.0;
    }

    let target = base + increment;
    if !target.is_finite() {
        return f64::INFINITY;
    }
    if base < 0.5 {
        return ln_gamma(target) - ln_gamma(base);
    }

    let t = base + LANCZOS_SHIFT;
    let log_power = increment.mul_add(t.ln(), (target - 0.5) * (increment / t).ln_1p());
    log_power - increment + (lanczos_sum(target) / lanczos_sum(base)).ln()
}

/// Residual `ln(Gamma(x)) - x * ln(x) + x` for positive finite `x`.
///
/// The direct expression catastrophically cancels for large `x`; the
/// Stirling series keeps the logarithmic remainder representable.
#[must_use]
#[inline]
pub fn ln_gamma_stirling_residual(value: f64) -> f64 {
    if value <= 0.0 || !value.is_finite() {
        return f64::NAN;
    }
    if value < 8.0 {
        return value.mul_add(1.0 - value.ln(), ln_gamma(value));
    }

    let inverse = 1.0 / value;
    let inverse2 = inverse * inverse;
    let inverse3 = inverse2 * inverse;
    let inverse5 = inverse3 * inverse2;
    let inverse7 = inverse5 * inverse2;
    let inverse9 = inverse7 * inverse2;
    let inverse11 = inverse9 * inverse2;
    0.5f64.mul_add(-value.ln(), HALF_LOG_2_PI) + inverse / 12.0 - inverse3 / 360.0
        + inverse5 / 1_260.0
        - inverse7 / 1_680.0
        + inverse9 / 1_188.0
        - 691.0 * inverse11 / 360_360.0
}

/// Digamma function approximation for positive arguments.
#[must_use]
#[inline]
pub fn digamma(value: f64) -> f64 {
    if value <= 0.0 || !value.is_finite() {
        return f64::NAN;
    }

    let mut x = value;
    let mut result = 0.0;

    #[allow(clippy::while_float)]
    while x < 8.0 {
        result -= 1.0 / x;
        x += 1.0;
    }

    let inv = 1.0 / x;
    let inv_sq = inv * inv;
    let inv_fourth = inv_sq * inv_sq;
    let inv_sixth = inv_fourth * inv_sq;
    let inv_eighth = inv_fourth * inv_fourth;
    let inv_tenth = inv_eighth * inv_sq;
    let inv_twelfth = inv_tenth * inv_sq;
    0.5f64.mul_add(-inv, result + x.ln()) - inv_sq / 12.0 + inv_fourth / 120.0 - inv_sixth / 252.0
        + inv_eighth / 240.0
        - 5.0 * inv_tenth / 660.0
        + 691.0 * inv_twelfth / 32_760.0
}

/// Difference `digamma(x) - ln(x)` for positive finite `x`.
///
/// The asymptotic form avoids subtracting nearly equal logarithms for large
/// arguments.
#[must_use]
#[inline]
pub fn digamma_minus_ln(value: f64) -> f64 {
    if value <= 0.0 || !value.is_finite() {
        return f64::NAN;
    }
    if value < 8.0 {
        return digamma(value) - value.ln();
    }

    let inverse = 1.0 / value;
    let inverse_squared = inverse * inverse;
    let inverse_fourth = inverse_squared * inverse_squared;
    let inverse_sixth = inverse_fourth * inverse_squared;
    let inverse_eighth = inverse_fourth * inverse_fourth;
    let inverse_tenth = inverse_eighth * inverse_squared;
    let inverse_twelfth = inverse_tenth * inverse_squared;
    (-0.5_f64).mul_add(inverse, -inverse_squared / 12.0) + inverse_fourth / 120.0
        - inverse_sixth / 252.0
        + inverse_eighth / 240.0
        - 5.0 * inverse_tenth / 660.0
        + 691.0 * inverse_twelfth / 32_760.0
}

/// Difference `digamma(base + increment) - digamma(base)` for positive arguments.
///
/// This evaluates the large-argument asymptotic series term by term, avoiding
/// cancellation when `increment` is smaller than the spacing between nearby
/// representable values at `base`.
#[must_use]
pub fn digamma_delta(base: f64, increment: f64) -> f64 {
    if base <= 0.0 || increment < 0.0 || !base.is_finite() || !increment.is_finite() {
        return f64::NAN;
    }
    if increment == 0.0 {
        return 0.0;
    }

    let target = base + increment;
    if !target.is_finite() {
        return f64::INFINITY;
    }
    let mut shifted = base;
    let mut recurrence = 0.0;
    #[allow(clippy::while_float)]
    while shifted < 8.0 {
        let shifted_target = shifted + increment;
        recurrence += if increment > shifted {
            (1.0 - shifted / shifted_target) / shifted
        } else {
            increment / shifted / shifted_target
        };
        shifted += 1.0;
    }

    let ratio = increment / shifted;
    let log_ratio = ratio.ln_1p();
    let inv = 1.0 / shifted;
    let inv_sq = inv * inv;
    let inv_fourth = inv_sq * inv_sq;
    let inv_sixth = inv_fourth * inv_sq;
    let inv_eighth = inv_fourth * inv_fourth;
    let inv_tenth = inv_eighth * inv_sq;
    let inv_twelfth = inv_tenth * inv_sq;
    let reciprocal_delta = 0.5 * inv * ratio / (1.0 + ratio);

    recurrence + log_ratio + reciprocal_delta - inv_sq * (-2.0 * log_ratio).exp_m1() / 12.0
        + inv_fourth * (-4.0 * log_ratio).exp_m1() / 120.0
        - inv_sixth * (-6.0 * log_ratio).exp_m1() / 252.0
        + inv_eighth * (-8.0 * log_ratio).exp_m1() / 240.0
        - 5.0 * inv_tenth * (-10.0 * log_ratio).exp_m1() / 660.0
        + 691.0 * inv_twelfth * (-12.0 * log_ratio).exp_m1() / 32_760.0
}

/// Regularized lower incomplete gamma function `P(a, x)`.
#[must_use]
#[inline]
pub fn regularized_gamma_lower(a: f64, x: f64) -> f64 {
    clamp_probability(regularized_gamma_unchecked::<false>(a, x))
}

/// Regularized upper incomplete gamma function `Q(a, x) = 1 - P(a, x)`.
#[must_use]
#[inline]
pub fn regularized_gamma_upper(a: f64, x: f64) -> f64 {
    clamp_probability(regularized_gamma_unchecked::<true>(a, x))
}

/// The tail selector is static so lower-only and upper-only calls keep their original arithmetic.
#[inline]
fn regularized_gamma_unchecked<const UPPER: bool>(a: f64, x: f64) -> f64 {
    if a <= 0.0 || !a.is_finite() || x < 0.0 || !x.is_finite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return if UPPER { 1.0 } else { 0.0 };
    }

    if use_gamma_saddlepoint(a, x) {
        let (lower, upper) = gamma_saddlepoint_pair(a, x);
        return if UPPER { upper } else { lower };
    }

    if x < a + 1.0 {
        let lower = gamma_lower_series(a, x);
        if UPPER { 1.0 - lower } else { lower }
    } else {
        let upper = gamma_upper_continued_fraction(a, x);
        if UPPER { upper } else { 1.0 - upper }
    }
}

#[inline]
fn use_gamma_saddlepoint(a: f64, x: f64) -> bool {
    const MIN_SHAPE: f64 = 50_000.0;
    const MAX_CENTERED_DISTANCE: f64 = 10.0;

    a >= MIN_SHAPE && (x - a).abs() <= MAX_CENTERED_DISTANCE * a.sqrt()
}

fn gamma_saddlepoint_pair(a: f64, x: f64) -> (f64, f64) {
    let sqrt_a = a.sqrt();
    let centered = (x - a) / sqrt_a;
    if centered.abs() <= 1.0e-5 {
        let central_correction = centered + 1.0 / (3.0 * sqrt_a);
        let lower = INV_SQRT_2_PI.mul_add(central_correction, 0.5);
        return (lower, 1.0 - lower);
    }

    let lambda = x / a;
    let eta = lambda - 1.0 - lambda.ln();
    if eta <= 0.0 {
        return (0.5, 0.5);
    }

    let r = (2.0 * a * eta).sqrt().copysign(lambda - 1.0);
    let correction = unit_normal_log_pdf(r).exp() * (1.0 / r - 1.0 / centered);
    let lower = unit_normal_cdf(r) + correction;
    (lower, 1.0 - lower)
}

fn gamma_lower_series(a: f64, x: f64) -> f64 {
    const MAX_ITERATIONS: usize = 10_000;
    const EPSILON: f64 = 1.0e-14;

    let mut term = 1.0 / a;
    let mut sum = term;
    let mut ap = a;
    for _ in 0..MAX_ITERATIONS {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if term.abs() <= sum.abs() * EPSILON {
            break;
        }
    }

    sum * (a.mul_add(x.ln(), -x) - ln_gamma(a)).exp()
}

#[allow(clippy::cast_precision_loss)]
fn gamma_upper_continued_fraction(a: f64, x: f64) -> f64 {
    const MAX_ITERATIONS: usize = 1_000;
    const EPSILON: f64 = 1.0e-14;
    const TINY: f64 = 1.0e-300;

    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    if b.abs() < TINY {
        b = TINY;
    }
    let mut d = 1.0 / b;
    let mut h = d;

    for iteration in 1..=MAX_ITERATIONS {
        let i = iteration as f64;
        let an = -i * (i - a);
        b += 2.0;
        d = an.mul_add(d, b);
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < EPSILON {
            break;
        }
    }

    (a.mul_add(x.ln(), -x) - ln_gamma(a)).exp() * h
}
