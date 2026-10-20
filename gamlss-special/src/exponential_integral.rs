//! Exponential integral with series and continued-fraction regimes.

const EULER_MASCHERONI: f64 = std::f64::consts::EULER_GAMMA;

/// Exponential integral `E1(x) = integral_x^inf exp(-t) / t dt`.
///
/// Returns `NaN` for negative or `NaN` inputs, positive infinity at zero, and
/// zero at positive infinity.
#[must_use]
pub fn exponential_integral_e1(x: f64) -> f64 {
    if x < 0.0 || x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x == f64::INFINITY {
        return 0.0;
    }

    if x <= 1.0 {
        exponential_integral_e1_series(x)
    } else {
        exponential_integral_e1_continued_fraction(x)
    }
}

#[allow(clippy::cast_precision_loss)]
fn exponential_integral_e1_series(x: f64) -> f64 {
    const MAX_ITERATIONS: usize = 1_000;
    const EPSILON: f64 = 1.0e-16;

    let mut factorial_term = -x;
    let mut sum = factorial_term;
    for iteration in 2..=MAX_ITERATIONS {
        let k = iteration as f64;
        factorial_term *= -x / k;
        let term = factorial_term / k;
        sum += term;
        if term.abs() <= EPSILON * sum.abs().max(1.0) {
            break;
        }
    }

    -EULER_MASCHERONI - x.ln() - sum
}

#[allow(clippy::cast_precision_loss)]
fn exponential_integral_e1_continued_fraction(x: f64) -> f64 {
    const MAX_ITERATIONS: usize = 1_000;
    const EPSILON: f64 = 1.0e-14;
    const TINY: f64 = 1.0e-300;

    let mut b = x + 1.0;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;

    for iteration in 1..=MAX_ITERATIONS {
        let i = iteration as f64;
        let a = -(i * i);
        b += 2.0;

        d = a.mul_add(d, b);
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + a / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = c * d;
        h *= delta;
        if (delta - 1.0).abs() <= EPSILON {
            break;
        }
    }

    (-x).exp() * h
}
