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

#[inline]
fn is_probability(value: f64) -> bool {
    (0.0..=1.0).contains(&value) && value.is_finite()
}

#[inline]
const fn clamp_probability(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

#[inline]
fn polynomial_ascending(value: f64, coefficients: &[f64]) -> f64 {
    // AS241 coefficients below are stored from constant term to highest degree.
    coefficients
        .iter()
        .rev()
        .fold(0.0, |accumulator, coefficient| {
            accumulator * value + coefficient
        })
}

#[inline]
fn polynomial_ascending_with_constant_one(value: f64, coefficients: &[f64]) -> f64 {
    value.mul_add(polynomial_ascending(value, coefficients), 1.0)
}

#[inline]
fn polynomial_descending(value: f64, coefficients: &[f64]) -> f64 {
    // Cody/Cephes coefficients below are stored from highest degree to constant term.
    coefficients.iter().fold(0.0, |accumulator, coefficient| {
        accumulator * value + coefficient
    })
}

#[inline]
fn polynomial_descending_with_implicit_leading_one(value: f64, coefficients: &[f64]) -> f64 {
    let Some((&first, rest)) = coefficients.split_first() else {
        return value;
    };

    rest.iter().fold(value + first, |accumulator, coefficient| {
        accumulator * value + coefficient
    })
}

/// Natural logarithm of the absolute gamma function via the Lanczos approximation.
///
/// Returns `NaN` at poles and for non-finite negative inputs. For positive
/// inputs this is the usual `ln(Gamma(x))`; for negative non-integers it is
/// `ln(abs(Gamma(x)))`.
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

    let shifted = value - 1.0;
    let mut x = COEFFICIENTS[0];
    for (index, coefficient) in COEFFICIENTS.iter().copied().enumerate().skip(1) {
        x += coefficient / (shifted + index as f64);
    }
    let t = shifted + 7.5;

    (shifted + 0.5).mul_add(t.ln(), 0.5 * (2.0 * std::f64::consts::PI).ln()) - t + x.ln()
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
/// Returning `None` for non-finite or excessively large query points keeps
/// discrete CDF implementations from doing unbounded work.
#[must_use]
#[inline]
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
    max + ((log_left - max).exp() + (log_right - max).exp()).ln()
}

/// Standard normal log-density.
#[must_use]
#[inline]
pub fn unit_normal_log_pdf(z: f64) -> f64 {
    const HALF_LOG_2_PI: f64 = 0.918_938_533_204_672_7;
    (0.5 * z).mul_add(-z, -HALF_LOG_2_PI)
}

/// Student-t normalizing constant on the negative log-density scale.
#[must_use]
#[inline]
pub fn student_t_nll_constant(nu: f64) -> f64 {
    f64::midpoint(nu.ln(), std::f64::consts::PI.ln()) + ln_gamma(0.5 * nu)
        - ln_gamma(f64::midpoint(nu, 1.0))
}

/// Standard Student-t log-density.
///
/// Returns `NaN` for invalid degrees of freedom or a `NaN` variate, and
/// negative infinity at either infinite tail.
#[must_use]
#[inline]
pub fn student_t_log_pdf_standardized(t: f64, nu: f64) -> f64 {
    if nu <= 0.0 || !nu.is_finite() || t.is_nan() {
        return f64::NAN;
    }
    if t.is_infinite() {
        return f64::NEG_INFINITY;
    }

    -student_t_nll_constant(nu) - f64::midpoint(nu, 1.0) * (t * t / nu).ln_1p()
}

/// Standard Student-t CDF.
///
/// Returns `NaN` for invalid degrees of freedom or a `NaN` variate. Infinite
/// tails map to zero and one.
#[must_use]
#[inline]
pub fn student_t_cdf_standardized(t: f64, nu: f64) -> f64 {
    if nu <= 0.0 || !nu.is_finite() || t.is_nan() {
        return f64::NAN;
    }
    if !t.is_finite() {
        return if t.is_sign_negative() { 0.0 } else { 1.0 };
    }
    if t == 0.0 {
        return 0.5;
    }

    let beta = regularized_beta(0.5 * nu, 0.5, nu / t.mul_add(t, nu));
    if t < 0.0 {
        0.5 * beta
    } else {
        0.5f64.mul_add(-beta, 1.0)
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

    #[allow(clippy::while_float)]
    while x < 8.0 {
        result -= 1.0 / x;
        x += 1.0;
    }

    let inv = 1.0 / x;
    let inv2 = inv * inv;
    0.5f64.mul_add(-inv, result + x.ln()) - inv2 / 12.0 + inv2 * inv2 / 120.0
        - inv2 * inv2 * inv2 / 252.0
        + inv2 * inv2 * inv2 * inv2 / 240.0
}

/// Regularized incomplete beta function `I_x(a, b)` for positive `a`, `b`.
#[must_use]
#[inline]
pub fn regularized_beta(a: f64, b: f64, x: f64) -> f64 {
    clamp_probability(regularized_beta_unchecked(a, b, x))
}

fn regularized_beta_unchecked(a: f64, b: f64, x: f64) -> f64 {
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
}

/// Regularized lower incomplete gamma function `P(a, x)`.
#[must_use]
#[inline]
pub fn regularized_gamma_lower(a: f64, x: f64) -> f64 {
    clamp_probability(regularized_gamma_lower_unchecked(a, x))
}

fn regularized_gamma_lower_unchecked(a: f64, x: f64) -> f64 {
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

    sum * (a.mul_add(x.ln(), -x) - ln_gamma(a)).exp()
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

/// Generalized inverse for a discrete CDF on non-negative integer support.
///
/// Returns `NaN` when `p` is outside `[0, 1]`, when the supplied CDF returns a
/// non-finite value, or when `max_count` is reached before the CDF reaches `p`.
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

    let mid = lower.midpoint(upper);
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
    let mid = lower.midpoint(upper);
    let left_mid = lower.midpoint(mid);
    let right_mid = mid.midpoint(upper);
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

#[inline]
fn simpson(lower: f64, upper: f64, f_lower: f64, f_mid: f64, f_upper: f64) -> f64 {
    (upper - lower) * (4.0f64.mul_add(f_mid, f_lower) + f_upper) / 6.0
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

/// Standard normal CDF.
#[must_use]
#[inline]
pub fn unit_normal_cdf(z: f64) -> f64 {
    unit_normal_sf(-z)
}

fn unit_normal_sf(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    if z == f64::NEG_INFINITY {
        return 1.0;
    }
    if z == f64::INFINITY {
        return 0.0;
    }

    clamp_probability(0.5 * unit_normal_erfc(z * std::f64::consts::FRAC_1_SQRT_2))
}

fn unit_normal_erfc(x: f64) -> f64 {
    const ERF_NUMERATOR: [f64; 5] = [
        9.604_973_739_870_516,
        90.026_019_720_384_27,
        2_232.005_345_946_843,
        7_003.325_141_128_051,
        55_592.301_301_039_49,
    ];
    const ERF_DENOMINATOR: [f64; 5] = [
        33.561_714_164_750_31,
        521.357_949_780_152_7,
        4_594.323_829_709_801,
        22_629.000_061_389_09,
        49_267.394_260_863_59,
    ];
    const ERFC_NUMERATOR: [f64; 9] = [
        2.461_969_814_735_305e-10,
        0.564_189_564_831_068_8,
        7.463_210_564_422_699,
        48.637_197_098_568_14,
        196.520_832_956_077_1,
        526.445_194_995_477_3,
        934.528_527_171_957_6,
        1_027.551_886_895_157,
        557.535_335_369_399_4,
    ];
    const ERFC_DENOMINATOR: [f64; 8] = [
        13.228_195_115_474_499,
        86.707_214_088_598_97,
        354.937_778_887_819_9,
        975.708_501_743_205_5,
        1_823.909_166_879_097_3,
        2_246.337_608_187_109_7,
        1_656.663_091_941_613_5,
        557.535_340_817_727_7,
    ];
    const ERFC_TAIL_NUMERATOR: [f64; 6] = [
        0.564_189_583_547_755_1,
        1.275_366_707_599_781,
        5.019_050_422_511_805,
        6.160_210_979_930_536,
        7.409_742_699_504_489,
        2.978_866_653_721_002,
    ];
    const ERFC_TAIL_DENOMINATOR: [f64; 6] = [
        2.260_528_632_201_172_6,
        9.396_035_249_380_014,
        12.048_953_980_809_665,
        17.081_445_074_756_59,
        9.608_968_090_632_859,
        3.369_076_451_000_815,
    ];
    const SQRT_HALF: f64 = std::f64::consts::FRAC_1_SQRT_2;
    const ERFC_UNDERFLOW_X: f64 = 27.3;

    if x.abs() <= SQRT_HALF {
        let square = x * x;
        let erf = x * polynomial_descending(square, &ERF_NUMERATOR)
            / polynomial_descending_with_implicit_leading_one(square, &ERF_DENOMINATOR);
        return 1.0 - erf;
    }

    let abs_x = x.abs();
    if abs_x >= ERFC_UNDERFLOW_X {
        return if x < 0.0 { 2.0 } else { 0.0 };
    }

    let (numerator, denominator) = if abs_x < 8.0 {
        (
            polynomial_descending(abs_x, &ERFC_NUMERATOR),
            polynomial_descending_with_implicit_leading_one(abs_x, &ERFC_DENOMINATOR),
        )
    } else {
        (
            polynomial_descending(abs_x, &ERFC_TAIL_NUMERATOR),
            polynomial_descending_with_implicit_leading_one(abs_x, &ERFC_TAIL_DENOMINATOR),
        )
    };

    let erfc = (-abs_x * abs_x).exp() * numerator / denominator;
    if x < 0.0 { 2.0 - erfc } else { erfc }
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
    if z <= -10.0 {
        return log_ndtr_left_tail(z);
    }

    unit_normal_cdf(z).ln()
}

fn log_ndtr_left_tail(z: f64) -> f64 {
    let x = -z;
    let inv2 = 1.0 / (x * x);
    let correction = polynomial_descending(
        inv2,
        &[
            -34_459_425.0,
            2_027_025.0,
            -135_135.0,
            10_395.0,
            -945.0,
            105.0,
            -15.0,
            3.0,
            -1.0,
            1.0,
        ],
    );

    unit_normal_log_pdf(z) - x.ln() + correction.max(f64::MIN_POSITIVE).ln()
}

/// Standard normal Mills ratio `phi(z) / Phi(z)`.
///
/// Infinite tails map to positive infinity and zero, respectively; `NaN`
/// propagates.
#[must_use]
#[inline]
pub fn normal_mills_ratio(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    if z == f64::NEG_INFINITY {
        return f64::INFINITY;
    }
    if z == f64::INFINITY {
        return 0.0;
    }

    let log_ratio = unit_normal_log_pdf(z) - log_ndtr(z);
    if log_ratio.is_nan() && z < 0.0 {
        // Both log terms may underflow to `-inf` for extreme finite left-tail
        // inputs. The leading asymptotic term remains representable.
        -z
    } else {
        log_ratio.exp()
    }
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
        return sign * 0.5 * unit_normal_sf(h.abs());
    }

    let h2 = h * h;
    let integral = integrate_finite(0.0, upper, |x| {
        (-0.5 * h2 * (1.0 + x * x)).exp() / (1.0 + x * x)
    });
    sign * integral / (2.0 * std::f64::consts::PI)
}

/// Standard normal quantile using Wichura's AS241 rational approximation.
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

    const CENTER_NUMERATOR: [f64; 8] = [
        3.387_132_872_796_366_5,
        133.141_667_891_784_38,
        1_971.590_950_306_551_3,
        13_731.693_765_509_46,
        45_921.953_931_549_87,
        67_265.770_927_008_7,
        33_430.575_583_588_13,
        2_509.080_928_730_122_7,
    ];
    const CENTER_DENOMINATOR: [f64; 7] = [
        42.313_330_701_600_91,
        687.187_007_492_057_9,
        5_394.196_021_424_751,
        21_213.794_301_586_597,
        39_307.895_800_092_71,
        28_729.085_735_721_943,
        5_226.495_278_852_855,
    ];
    const TAIL_NUMERATOR: [f64; 8] = [
        1.423_437_110_749_683_5,
        4.630_337_846_156_545,
        5.769_497_221_460_691,
        3.647_848_324_763_204_5,
        1.270_458_252_452_368_4,
        0.241_780_725_177_450_6,
        0.022_723_844_989_269_184,
        0.000_774_545_014_278_341_4,
    ];
    const TAIL_DENOMINATOR: [f64; 7] = [
        2.053_191_626_637_759,
        1.676_384_830_183_803_8,
        0.689_767_334_985_1,
        0.148_103_976_427_480_08,
        0.015_198_666_563_616_457,
        0.000_547_593_808_499_534_5,
        1.050_750_071_644_416_8e-9,
    ];
    const FAR_TAIL_NUMERATOR: [f64; 8] = [
        6.657_904_643_501_104,
        5.463_784_911_164_114,
        1.784_826_539_917_291_3,
        0.296_560_571_828_504_87,
        0.026_532_189_526_576_124,
        0.001_242_660_947_388_078_4,
        0.000_027_115_555_687_434_876,
        0.000_000_201_033_439_929_228_82,
    ];
    const FAR_TAIL_DENOMINATOR: [f64; 7] = [
        0.599_832_206_555_887_9,
        0.136_929_880_922_735_8,
        0.014_875_361_290_850_615,
        0.000_786_869_131_145_613_3,
        0.000_018_463_183_175_100_547,
        0.000_000_142_151_175_831_644_6,
        2.044_263_103_389_939_7e-15,
    ];

    let centered = p - 0.5;
    if centered.abs() <= 0.425 {
        let r = 0.180_625 - centered * centered;
        centered * polynomial_ascending(r, &CENTER_NUMERATOR)
            / polynomial_ascending_with_constant_one(r, &CENTER_DENOMINATOR)
    } else {
        let tail_probability = if centered < 0.0 { p } else { 1.0 - p };
        let mut r = (-tail_probability.ln()).sqrt();
        let quantile = if r <= 5.0 {
            r -= 1.6;
            polynomial_ascending(r, &TAIL_NUMERATOR)
                / polynomial_ascending_with_constant_one(r, &TAIL_DENOMINATOR)
        } else {
            r -= 5.0;
            polynomial_ascending(r, &FAR_TAIL_NUMERATOR)
                / polynomial_ascending_with_constant_one(r, &FAR_TAIL_DENOMINATOR)
        };
        if centered < 0.0 { -quantile } else { quantile }
    }
}
