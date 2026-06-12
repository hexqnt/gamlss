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

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::{digamma, ln_gamma};

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
}
