//! Beta normalizers and incomplete-beta numerical kernels.

use crate::gamma::{ln_gamma, ln_gamma_delta};
use crate::probability::clamp_probability;

/// Natural logarithm of the beta function for positive finite arguments.
#[must_use]
#[inline]
pub fn ln_beta(a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 || !a.is_finite() || !b.is_finite() {
        return f64::NAN;
    }

    let small = a.min(b);
    let large = a.max(b);
    ln_gamma(small) - ln_gamma_delta(large, small)
}

/// Logarithm of the multivariate beta function.
///
/// The sequential beta identity avoids subtracting one large `ln_gamma(sum)`
/// from a separately accumulated sum of large gamma terms.
#[must_use]
pub fn ln_multivariate_beta(alpha: &[f64]) -> f64 {
    if alpha.len() < 2
        || alpha
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return f64::NAN;
    }

    let mut prefix = alpha[0];
    let mut value = 0.0;
    for component in alpha.iter().copied().skip(1) {
        value += ln_beta(prefix, component);
        prefix += component;
        if !prefix.is_finite() {
            return f64::NAN;
        }
    }
    value
}

/// Regularized incomplete beta function `I_x(a, b)` for positive `a`, `b`.
#[must_use]
#[inline]
pub fn regularized_beta(a: f64, b: f64, x: f64) -> f64 {
    clamp_probability(regularized_beta_pair_unchecked(a, b, x).0)
}

/// Complement of the regularized incomplete beta function, `1 - I_x(a, b)`.
///
/// This avoids subtracting from one in the upper tail when the continued
/// fraction can compute the complement directly.
#[must_use]
#[inline]
pub fn regularized_beta_complement(a: f64, b: f64, x: f64) -> f64 {
    clamp_probability(regularized_beta_pair_unchecked(a, b, x).1)
}

/// Logarithm of the regularized incomplete beta function `I_x(a, b)`.
///
/// The lower-tail branch is evaluated directly in log space so callers can
/// retain probabilities smaller than the normal floating-point range.
#[must_use]
#[inline]
pub fn log_regularized_beta(a: f64, b: f64, x: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 || !a.is_finite() || !b.is_finite() || !(0.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    #[allow(clippy::float_cmp)]
    if x == 1.0 {
        return 0.0;
    }

    let log_front = b.mul_add((-x).ln_1p(), a.mul_add(x.ln(), -ln_beta(a, b)));
    if x < (a + 1.0) / (a + b + 2.0) {
        log_front + beta_continued_fraction(a, b, x).ln() - a.ln()
    } else {
        let log_complement = log_front + beta_continued_fraction(b, a, 1.0 - x).ln() - b.ln();
        (-log_complement.exp().clamp(0.0, 1.0)).ln_1p()
    }
}

#[allow(clippy::suboptimal_flops)]
fn regularized_beta_pair_unchecked(a: f64, b: f64, x: f64) -> (f64, f64) {
    if a <= 0.0 || b <= 0.0 || !a.is_finite() || !b.is_finite() || !(0.0..=1.0).contains(&x) {
        return (f64::NAN, f64::NAN);
    }
    if x == 0.0 {
        return (0.0, 1.0);
    }
    #[allow(clippy::float_cmp)]
    if x == 1.0 {
        return (1.0, 0.0);
    }

    let front = beta_front(a, b, x);
    let term = if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_continued_fraction(a, b, x) / a
    } else {
        front * beta_continued_fraction(b, a, 1.0 - x) / b
    };

    if x < (a + 1.0) / (a + b + 2.0) {
        (term, 1.0 - term)
    } else {
        (1.0 - term, term)
    }
}

#[inline]
fn beta_front(a: f64, b: f64, x: f64) -> f64 {
    b.mul_add((-x).ln_1p(), a.mul_add(x.ln(), -ln_beta(a, b)))
        .exp()
}

#[allow(clippy::cast_precision_loss)]
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
        d = aa.mul_add(d, 1.0);
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
        d = aa.mul_add(d, 1.0);
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
