/// Natural logarithm of the gamma function via the Lanczos approximation.
pub(crate) fn ln_gamma(value: f64) -> f64 {
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

/// Returns `true` for finite counts represented on the shared `f64` observation path.
pub(crate) fn is_nonnegative_integer(value: f64) -> bool {
    value >= 0.0 && value.is_finite() && value.fract() == 0.0
}

/// Converts a finite CDF query point into the largest included count.
///
/// Returning `None` keeps discrete CDF implementations from doing unbounded
/// work for pathologically large query points.
pub(crate) fn included_count(value: f64, max_terms: u64) -> Option<u64> {
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
pub(crate) fn log_add_exp(log_left: f64, log_right: f64) -> f64 {
    if log_left == f64::NEG_INFINITY {
        return log_right;
    }
    if log_right == f64::NEG_INFINITY {
        return log_left;
    }

    let max = log_left.max(log_right);
    max + ((log_left - max).exp() + (log_right - max).exp()).ln()
}

/// Digamma function approximation for positive arguments.
pub(crate) fn digamma(value: f64) -> f64 {
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

/// Standard normal CDF approximation.
pub(crate) fn unit_normal_cdf(z: f64) -> f64 {
    (0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2))).clamp(0.0, 1.0)
}

fn erf_approx(value: f64) -> f64 {
    const P: f64 = 0.327_591_1;
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;

    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let x = value.abs();
    let t = 1.0 / (1.0 + P * x);
    let polynomial = (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t;

    sign * (1.0 - polynomial * (-x * x).exp())
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::{digamma, ln_gamma, log_add_exp, unit_normal_cdf};

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
    fn unit_normal_cdf_matches_reference_points() {
        assert_relative_eq!(unit_normal_cdf(0.0), 0.5, epsilon = 1.0e-7);
        assert_relative_eq!(unit_normal_cdf(1.0), 0.841_344_746, epsilon = 1.0e-7);
        assert_relative_eq!(unit_normal_cdf(-1.0), 0.158_655_254, epsilon = 1.0e-7);
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
