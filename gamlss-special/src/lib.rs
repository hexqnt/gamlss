#![forbid(unsafe_code)]
//! Special functions and numerical helpers for GAMLSS crates.
//!
//! This crate intentionally exposes small scalar `f64` free functions. Keeping
//! the API as free functions lets distribution crates call them without runtime
//! dispatch while still sharing one implementation of gamma/beta, normal-tail
//! and CDF inversion helpers.
//!
//! Invalid domains return `NaN` for helper-style APIs and non-finite likelihood
//! scale values where that convention is already part of the surrounding
//! distribution code.

#[inline(always)]
fn is_probability(value: f64) -> bool {
    (0.0..=1.0).contains(&value) && value.is_finite()
}

/// Natural logarithm of the gamma function via the Lanczos approximation.
#[must_use]
#[inline]
pub fn ln_gamma(value: f64) -> f64 {
    const COEFFICIENTS: [f64; 9] = [
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

    if value < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * value).sin().ln()
            - ln_gamma(1.0 - value);
    }

    let shifted = value - 1.0;
    let mut x = COEFFICIENTS[0];
    for (index, coefficient) in COEFFICIENTS.iter().copied().enumerate().skip(1) {
        x += coefficient / (shifted + index as f64);
    }
    let t = shifted + 7.5;

    0.5 * (2.0 * std::f64::consts::PI).ln() + (shifted + 0.5) * t.ln() - t + x.ln()
}

/// Natural logarithm of the beta function for positive finite arguments.
#[must_use]
#[inline]
pub fn ln_beta(a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 || !a.is_finite() || !b.is_finite() {
        return f64::NAN;
    }

    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Returns `true` for finite counts represented on the shared `f64` observation path.
#[must_use]
#[inline]
pub fn is_nonnegative_integer(value: f64) -> bool {
    value >= 0.0 && value.is_finite() && value.fract() == 0.0
}

/// Converts a finite CDF query point into the largest included count.
///
/// Returning `None` keeps discrete CDF implementations from doing unbounded
/// work for pathologically large query points.
#[must_use]
#[inline]
pub fn included_count(value: f64, max_terms: u64) -> Option<u64> {
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
    if log_left == f64::NEG_INFINITY {
        return log_right;
    }
    if log_right == f64::NEG_INFINITY {
        return log_left;
    }

    let max = log_left.max(log_right);
    max + ((log_left - max).exp() + (log_right - max).exp()).ln()
}

/// Standard normal log-density.
#[must_use]
#[inline]
pub fn unit_normal_log_pdf(z: f64) -> f64 {
    const HALF_LOG_2_PI: f64 = 0.918_938_533_204_672_7;
    -HALF_LOG_2_PI - 0.5 * z * z
}

/// Student-t normalizing constant on the negative log-density scale.
#[must_use]
#[inline]
pub fn student_t_nll_constant(nu: f64) -> f64 {
    0.5 * (nu.ln() + std::f64::consts::PI.ln()) + ln_gamma(0.5 * nu) - ln_gamma(0.5 * (nu + 1.0))
}

/// Standard Student-t log-density.
#[must_use]
#[inline]
pub fn student_t_log_pdf_standardized(t: f64, nu: f64) -> f64 {
    if nu <= 0.0 || !nu.is_finite() || !t.is_finite() {
        return f64::NEG_INFINITY;
    }

    -student_t_nll_constant(nu) - 0.5 * (nu + 1.0) * (t * t / nu).ln_1p()
}

/// Standard Student-t CDF.
#[must_use]
#[inline]
pub fn student_t_cdf_standardized(t: f64, nu: f64) -> f64 {
    if nu <= 0.0 || !nu.is_finite() {
        return f64::NAN;
    }
    if !t.is_finite() {
        return if t.is_sign_negative() { 0.0 } else { 1.0 };
    }
    if t == 0.0 {
        return 0.5;
    }

    let beta = regularized_beta(0.5 * nu, 0.5, nu / (nu + t * t));
    if t < 0.0 {
        0.5 * beta
    } else {
        1.0 - 0.5 * beta
    }
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
    while x < 8.0 {
        result -= 1.0 / x;
        x += 1.0;
    }

    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + x.ln() - 0.5 * inv - inv2 / 12.0 + inv2 * inv2 / 120.0 - inv2 * inv2 * inv2 / 252.0
        + inv2 * inv2 * inv2 * inv2 / 240.0
}

/// Regularized incomplete beta function `I_x(a, b)` for positive `a`, `b`.
#[must_use]
#[inline]
pub fn regularized_beta(a: f64, b: f64, x: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 || !a.is_finite() || !b.is_finite() || !(0.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }

    let log_front = -ln_beta(a, b) + a * x.ln() + b * (1.0 - x).ln();
    let front = log_front.exp();

    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_continued_fraction(a, b, x) / a
    } else {
        1.0 - front * beta_continued_fraction(b, a, 1.0 - x) / b
    }
    .clamp(0.0, 1.0)
}

/// Regularized lower incomplete gamma function `P(a, x)`.
#[must_use]
#[inline]
pub fn regularized_gamma_lower(a: f64, x: f64) -> f64 {
    if a <= 0.0 || !a.is_finite() || x < 0.0 || !x.is_finite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }

    if x < a + 1.0 {
        gamma_lower_series(a, x)
    } else {
        1.0 - gamma_upper_continued_fraction(a, x)
    }
    .clamp(0.0, 1.0)
}

fn gamma_lower_series(a: f64, x: f64) -> f64 {
    const MAX_ITERATIONS: usize = 1_000;
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

    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

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
        d = an * d + b;
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

    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// Generalized inverse for a discrete CDF on non-negative integer support.
///
/// Returns `NaN` when `p` is outside `[0, 1]` or when `max_count` is reached
/// before the supplied CDF reaches `p`.
#[must_use]
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

    let mut high = 1_u64;
    while high < max_count && cdf(high) < p {
        high = high.saturating_mul(2).min(max_count);
    }
    if cdf(high) < p {
        return f64::NAN;
    }

    let mut low = 0_u64;
    while low < high {
        let mid = low + (high - low) / 2;
        if cdf(mid) < p {
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
/// invalid probabilities or non-finite bounds.
#[must_use]
pub fn invert_bounded_cdf<F>(p: f64, lower: f64, upper: f64, mut cdf: F) -> f64
where
    F: FnMut(f64) -> f64,
{
    if !is_probability(p) || !lower.is_finite() || !upper.is_finite() {
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
    for _ in 0..120 {
        let mid = 0.5 * (low + high);
        if cdf(mid) < p {
            low = mid;
        } else {
            high = mid;
        }
    }

    0.5 * (low + high)
}

/// Inverts a monotone CDF on `[0, +inf)` by bracketing and bisection.
///
/// Returns `0` for `p == 0`, `+inf` for `p == 1`, and `NaN` for invalid
/// probabilities.
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
    if p == 1.0 {
        return f64::INFINITY;
    }

    let mut high = 1.0;
    while cdf(high) < p {
        high *= 2.0;
        if !high.is_finite() {
            return f64::INFINITY;
        }
    }

    invert_bounded_cdf(p, 0.0, high, cdf)
}

/// Inverts a monotone CDF on the real line by bracketing and bisection.
///
/// Returns infinite tails for boundary probabilities and `NaN` for invalid
/// probabilities.
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
    if p == 1.0 {
        return f64::INFINITY;
    }

    let mut low = -1.0;
    while cdf(low) > p {
        low *= 2.0;
        if !low.is_finite() {
            return f64::NEG_INFINITY;
        }
    }

    let mut high = 1.0;
    while cdf(high) < p {
        high *= 2.0;
        if !high.is_finite() {
            return f64::INFINITY;
        }
    }

    invert_bounded_cdf(p, low, high, cdf)
}

/// Integrates a finite interval with adaptive Simpson quadrature.
///
/// Returns `NaN` for non-finite bounds, reversed bounds, or non-finite function
/// evaluations.
#[must_use]
pub fn integrate_finite<F>(lower: f64, upper: f64, mut function: F) -> f64
where
    F: FnMut(f64) -> f64,
{
    const EPSILON: f64 = 1.0e-10;
    const MAX_DEPTH: u32 = 24;

    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return f64::NAN;
    }
    if lower == upper {
        return 0.0;
    }

    let mid = 0.5 * (lower + upper);
    let f_lower = function(lower);
    let f_mid = function(mid);
    let f_upper = function(upper);
    if !f_lower.is_finite() || !f_mid.is_finite() || !f_upper.is_finite() {
        return f64::NAN;
    }

    let whole = simpson(lower, upper, f_lower, f_mid, f_upper);
    adaptive_simpson(
        &mut function,
        lower,
        upper,
        [f_lower, f_mid, f_upper],
        whole,
        EPSILON,
        MAX_DEPTH,
    )
}

fn adaptive_simpson<F>(
    function: &mut F,
    lower: f64,
    upper: f64,
    values: [f64; 3],
    whole: f64,
    epsilon: f64,
    depth: u32,
) -> f64
where
    F: FnMut(f64) -> f64,
{
    let [f_lower, f_mid, f_upper] = values;
    let mid = 0.5 * (lower + upper);
    let left_mid = 0.5 * (lower + mid);
    let right_mid = 0.5 * (mid + upper);
    let f_left_mid = function(left_mid);
    let f_right_mid = function(right_mid);
    if !f_left_mid.is_finite() || !f_right_mid.is_finite() {
        return f64::NAN;
    }

    let left = simpson(lower, mid, f_lower, f_left_mid, f_mid);
    let right = simpson(mid, upper, f_mid, f_right_mid, f_upper);
    let delta = left + right - whole;
    if depth == 0 || delta.abs() <= 15.0 * epsilon {
        return left + right + delta / 15.0;
    }

    let left_integral = adaptive_simpson(
        function,
        lower,
        mid,
        [f_lower, f_left_mid, f_mid],
        left,
        0.5 * epsilon,
        depth - 1,
    );
    let right_integral = adaptive_simpson(
        function,
        mid,
        upper,
        [f_mid, f_right_mid, f_upper],
        right,
        0.5 * epsilon,
        depth - 1,
    );

    left_integral + right_integral
}

#[inline(always)]
fn simpson(lower: f64, upper: f64, f_lower: f64, f_mid: f64, f_upper: f64) -> f64 {
    (upper - lower) * (f_lower + 4.0 * f_mid + f_upper) / 6.0
}

fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITERATIONS: usize = 200;
    const EPSILON: f64 = 3.0e-14;
    const TINY: f64 = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;

    for iteration in 1..=MAX_ITERATIONS {
        let m = iteration as f64;
        let m2 = 2.0 * m;

        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;

        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
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

    h
}

/// Standard normal CDF approximation.
#[must_use]
#[inline]
pub fn unit_normal_cdf(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    if z == f64::NEG_INFINITY {
        return 0.0;
    }
    if z == f64::INFINITY {
        return 1.0;
    }

    let x = z.abs();
    let t = 1.0 / (1.0 + 0.231_641_9 * x);
    let polynomial =
        ((((1.330_274_429 * t - 1.821_255_978) * t + 1.781_477_937) * t - 0.356_563_782) * t
            + 0.319_381_530)
            * t;
    let tail = (-0.5 * x * x).exp() * polynomial / (2.0 * std::f64::consts::PI).sqrt();

    if z >= 0.0 { 1.0 - tail } else { tail }.clamp(0.0, 1.0)
}

/// Natural logarithm of the standard normal CDF.
#[must_use]
#[inline]
pub fn log_ndtr(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    if z == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    if z == f64::INFINITY {
        return 0.0;
    }
    if z <= -5.0 {
        return log_ndtr_left_tail(z);
    }

    unit_normal_cdf(z).ln()
}

fn log_ndtr_left_tail(z: f64) -> f64 {
    let x = -z;
    let inv2 = 1.0 / (x * x);
    let correction = 1.0 - inv2 + 3.0 * inv2 * inv2 - 15.0 * inv2 * inv2 * inv2
        + 105.0 * inv2 * inv2 * inv2 * inv2;

    unit_normal_log_pdf(z) - x.ln() + correction.max(f64::MIN_POSITIVE).ln()
}

/// Standard normal Mills ratio `phi(z) / Phi(z)`.
#[must_use]
#[inline]
pub fn normal_mills_ratio(z: f64) -> f64 {
    (unit_normal_log_pdf(z) - log_ndtr(z)).exp()
}

/// Owen's T function `T(h, a)`.
///
/// This is primarily used for skew-normal CDF evaluation. The implementation
/// uses adaptive Simpson integration, which is accurate enough for family
/// helper APIs while keeping production dependencies unchanged.
#[must_use]
pub fn owens_t(h: f64, a: f64) -> f64 {
    if !h.is_finite() || !a.is_finite() {
        return f64::NAN;
    }
    if a == 0.0 {
        return 0.0;
    }

    let sign = a.signum();
    let upper = a.abs();
    if upper > 50.0 {
        return sign * 0.5 * (1.0 - unit_normal_cdf(h.abs()));
    }

    let h2 = h * h;
    let integral = integrate_finite(0.0, upper, |x| {
        (-0.5 * h2 * (1.0 + x * x)).exp() / (1.0 + x * x)
    });
    sign * integral / (2.0 * std::f64::consts::PI)
}

/// Standard normal quantile approximation.
#[must_use]
#[inline]
pub fn unit_normal_quantile(p: f64) -> f64 {
    if p < 0.0 || !p.is_finite() || p > 1.0 {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }

    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    const P_LOW: f64 = 0.024_25;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (-p).ln_1p()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use statrs::distribution::{Continuous, ContinuousCDF, Normal as StatrsNormal, StudentsT};

    use super::{
        digamma, discrete_quantile, integrate_finite, invert_bounded_cdf, invert_positive_cdf,
        invert_real_cdf, ln_beta, ln_gamma, log_add_exp, log_ndtr, normal_mills_ratio, owens_t,
        regularized_beta, regularized_gamma_lower, student_t_cdf_standardized,
        student_t_log_pdf_standardized, student_t_nll_constant, unit_normal_cdf,
        unit_normal_quantile,
    };

    fn assert_close(actual: f64, expected: f64, rel_tol: f64, abs_tol: f64) {
        let diff = (actual - expected).abs();
        let scale = actual.abs().max(expected.abs()).max(1.0);
        assert!(
            diff <= abs_tol.max(rel_tol * scale),
            "actual {actual:?} differs from expected {expected:?}; diff={diff:?}, rel_tol={rel_tol:?}, abs_tol={abs_tol:?}"
        );
    }

    #[test]
    fn ln_gamma_matches_known_constants() {
        assert_relative_eq!(ln_gamma(1.0), 0.0, epsilon = 1.0e-12);
        assert_relative_eq!(
            ln_gamma(0.5),
            0.5 * std::f64::consts::PI.ln(),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(ln_gamma(5.0), 24.0_f64.ln(), epsilon = 1.0e-12);
    }

    #[test]
    fn ln_gamma_matches_statrs_reference_grid() {
        for value in [
            0.001_f64, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 25.0, 75.0, 150.0,
        ] {
            assert_close(
                ln_gamma(value),
                statrs::function::gamma::ln_gamma(value),
                2.0e-14,
                2.0e-13,
            );
        }
    }

    #[test]
    fn ln_beta_matches_known_constants() {
        assert_relative_eq!(ln_beta(1.0, 1.0), 0.0, epsilon = 1.0e-12);
        assert_relative_eq!(
            ln_beta(0.5, 0.5),
            std::f64::consts::PI.ln(),
            epsilon = 1.0e-12
        );
        assert!(ln_beta(0.0, 1.0).is_nan());
    }

    #[test]
    fn digamma_matches_known_constants_and_recurrence() {
        let euler_gamma = 0.577_215_664_901_532_9;

        assert_relative_eq!(digamma(1.0), -euler_gamma, epsilon = 1.0e-10);
        assert_relative_eq!(
            digamma(0.5),
            -euler_gamma - 2.0 * 2.0_f64.ln(),
            epsilon = 1.0e-10
        );
        assert_relative_eq!(digamma(4.25), digamma(3.25) + 1.0 / 3.25, epsilon = 1.0e-12);
    }

    #[test]
    fn digamma_matches_statrs_reference_grid() {
        for value in [0.001_f64, 0.01, 0.1, 0.5, 1.0, 1.5, 8.0, 30.0, 120.0] {
            assert_close(
                digamma(value),
                statrs::function::gamma::digamma(value),
                2.0e-12,
                2.0e-11,
            );
        }
    }

    #[test]
    fn unit_normal_cdf_matches_reference_points() {
        assert_relative_eq!(unit_normal_cdf(0.0), 0.5, epsilon = 1.0e-7);
        assert_relative_eq!(unit_normal_cdf(1.0), 0.841_344_746, epsilon = 1.0e-7);
        assert_relative_eq!(unit_normal_cdf(-1.0), 0.158_655_254, epsilon = 1.0e-7);
    }

    #[test]
    fn unit_normal_cdf_and_quantile_match_statrs_reference_grid() {
        let reference = StatrsNormal::new(0.0, 1.0).unwrap();

        for z in [-8.0_f64, -6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0, 8.0] {
            assert_close(unit_normal_cdf(z), reference.cdf(z), 0.0, 8.0e-8);
        }

        for p in [
            1.0e-12,
            1.0e-10,
            1.0e-8,
            1.0e-4,
            0.01,
            0.5,
            0.99,
            1.0 - 1.0e-4,
            1.0 - 1.0e-8,
            1.0 - 1.0e-10,
            1.0 - 1.0e-12,
        ] {
            let quantile = unit_normal_quantile(p);
            assert_close(quantile, reference.inverse_cdf(p), 0.0, 1.0e-8);
            assert_close(unit_normal_cdf(quantile), p, 0.0, 8.0e-8);
        }
    }

    #[test]
    fn log_ndtr_and_mills_ratio_stay_finite_in_left_tail() {
        for z in [-2.0, 0.0, 3.0] {
            assert_relative_eq!(log_ndtr(z).exp(), unit_normal_cdf(z), epsilon = 1.0e-14);
        }

        let left_tail = log_ndtr(-40.0);
        assert!(left_tail.is_finite());
        assert!(left_tail < -800.0);

        let mills = normal_mills_ratio(-40.0);
        assert!(mills.is_finite());
        assert!(mills > 40.0 && mills < 40.1, "mills was {mills}");

        assert_eq!(log_ndtr(f64::INFINITY), 0.0);
        assert_eq!(log_ndtr(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert!(log_ndtr(f64::NAN).is_nan());
    }

    #[test]
    fn log_ndtr_matches_statrs_reference_where_cdf_is_representable() {
        let reference = StatrsNormal::new(0.0, 1.0).unwrap();
        for z in [-10.0_f64, -8.0, -5.0, -2.0, 0.0, 2.0, 5.0] {
            assert_close(log_ndtr(z), reference.cdf(z).ln(), 0.0, 1.0e-4);
        }
    }

    #[test]
    fn standardized_student_t_helpers_match_statrs_reference() {
        for nu in [1.5_f64, 2.5, 5.0, 30.0] {
            let reference = StudentsT::new(0.0, 1.0, nu).unwrap();
            assert_close(
                (-student_t_nll_constant(nu)).exp(),
                reference.pdf(0.0),
                2.0e-13,
                2.0e-13,
            );

            for t in [-4.0_f64, -1.0, 0.0, 1.0, 4.0] {
                assert_close(
                    student_t_log_pdf_standardized(t, nu).exp(),
                    reference.pdf(t),
                    2.0e-13,
                    2.0e-13,
                );
                assert_close(
                    student_t_cdf_standardized(t, nu),
                    reference.cdf(t),
                    2.0e-12,
                    2.0e-12,
                );
            }
        }

        assert!(student_t_log_pdf_standardized(0.0, 0.0).is_infinite());
        assert!(student_t_cdf_standardized(0.0, 0.0).is_nan());
    }

    #[test]
    fn unit_normal_quantile_matches_reference_points() {
        assert_relative_eq!(unit_normal_quantile(0.5), 0.0, epsilon = 1.0e-9);
        assert_relative_eq!(unit_normal_quantile(0.841_344_746), 1.0, epsilon = 1.0e-6);
        assert_relative_eq!(unit_normal_quantile(0.158_655_254), -1.0, epsilon = 1.0e-6);

        let upper = unit_normal_quantile(1.0 - 1.0e-12);
        let lower = unit_normal_quantile(1.0e-12);
        assert!(upper.is_finite());
        assert_relative_eq!(upper, -lower, epsilon = 5.0e-6);
    }

    #[test]
    fn regularized_beta_matches_simple_cases() {
        assert_relative_eq!(regularized_beta(1.0, 1.0, 0.25), 0.25, epsilon = 1.0e-14);
        assert_relative_eq!(regularized_beta(2.0, 1.0, 0.5), 0.25, epsilon = 1.0e-14);
        assert_relative_eq!(regularized_beta(1.0, 2.0, 0.5), 0.75, epsilon = 1.0e-14);
    }

    #[test]
    fn regularized_beta_matches_statrs_reference_grid() {
        for (a, b, x) in [
            (0.1_f64, 0.2_f64, 1.0e-8_f64),
            (0.1, 0.2, 0.01),
            (0.1, 5.0, 0.8),
            (0.5, 0.5, 0.99),
            (2.0, 7.0, 0.25),
            (25.0, 30.0, 0.45),
            (75.0, 80.0, 0.55),
        ] {
            assert_close(
                regularized_beta(a, b, x),
                statrs::function::beta::beta_reg(a, b, x),
                2.0e-11,
                2.0e-12,
            );
        }
    }

    #[test]
    fn regularized_gamma_lower_matches_simple_cases() {
        assert_relative_eq!(regularized_gamma_lower(1.0, 2.0), 1.0 - (-2.0_f64).exp());
        assert_relative_eq!(
            regularized_gamma_lower(2.0, 2.0),
            1.0 - 3.0 * (-2.0_f64).exp(),
            epsilon = 1.0e-14
        );
        assert_eq!(regularized_gamma_lower(2.0, 0.0), 0.0);
    }

    #[test]
    fn regularized_gamma_lower_matches_statrs_reference_grid() {
        for (a, x) in [
            (0.1_f64, 1.0e-8_f64),
            (0.1, 0.01),
            (0.5, 0.5),
            (1.0, 20.0),
            (2.5, 1.5),
            (10.0, 12.0),
            (50.0, 45.0),
            (100.0, 130.0),
        ] {
            assert_close(
                regularized_gamma_lower(a, x),
                statrs::function::gamma::gamma_lr(a, x),
                2.0e-11,
                2.0e-12,
            );
        }
    }

    #[test]
    fn regularized_gamma_lower_is_bounded_near_continued_fraction_singularity() {
        let value = regularized_gamma_lower(2.5, 1.5);

        assert!(value.is_finite());
        assert!((0.0..=1.0).contains(&value));
    }

    #[test]
    fn owens_t_satisfies_basic_symmetry_and_special_cases() {
        for h in [-4.0_f64, -1.0, 0.0, 1.0, 4.0] {
            assert_eq!(owens_t(h, 0.0), 0.0);
            assert_close(owens_t(h, 1.3), owens_t(-h, 1.3), 0.0, 1.0e-14);
            assert_close(owens_t(h, -1.3), -owens_t(h, 1.3), 0.0, 1.0e-14);
        }

        for a in [-2.0_f64, -0.5, 0.5, 2.0] {
            assert_close(
                owens_t(0.0, a),
                a.atan() / (2.0 * std::f64::consts::PI),
                0.0,
                1.0e-12,
            );
        }
    }

    #[test]
    fn discrete_quantile_returns_generalized_inverse() {
        assert_eq!(discrete_quantile(0.1, 10, |k| (k + 1) as f64 / 10.0), 0.0);
        assert_eq!(discrete_quantile(0.2, 10, |k| (k + 1) as f64 / 10.0), 1.0);
        assert_eq!(discrete_quantile(1.0, 9, |k| (k + 1) as f64 / 10.0), 9.0);
        assert!(discrete_quantile(1.0, 8, |k| (k + 1) as f64 / 10.0).is_nan());
    }

    #[test]
    fn cdf_inversion_helpers_find_quantiles() {
        assert_relative_eq!(
            invert_bounded_cdf(0.25, 0.0, 1.0, |x| x),
            0.25,
            epsilon = 1.0e-14
        );
        assert_relative_eq!(
            invert_positive_cdf(0.75, |x| 1.0 - (-x).exp()),
            -(0.25_f64).ln(),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            invert_real_cdf(0.75, unit_normal_cdf),
            unit_normal_quantile(0.75),
            epsilon = 1.0e-6
        );
    }

    #[test]
    fn integrate_finite_matches_polynomial_and_handles_invalid_bounds() {
        assert_relative_eq!(
            integrate_finite(0.0, 1.0, |x| x * x),
            1.0 / 3.0,
            epsilon = 1.0e-10
        );
        assert_eq!(integrate_finite(2.0, 2.0, |_| 1.0), 0.0);
        assert!(integrate_finite(2.0, 1.0, |_| 1.0).is_nan());
    }

    #[test]
    fn log_add_exp_combines_log_terms_without_underflow() {
        assert_relative_eq!(
            log_add_exp(-1000.0, -1001.0),
            -1000.0 + (-1.0_f64).exp().ln_1p(),
            epsilon = 1.0e-12
        );

        let combined = log_add_exp(f64::NEG_INFINITY, -3.0);
        assert_relative_eq!(combined, -3.0, epsilon = 1.0e-12);
    }
}
