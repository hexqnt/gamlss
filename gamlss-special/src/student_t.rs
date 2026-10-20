//! Prepared Student-t density and stable CDF evaluation.

use crate::beta::{ln_beta, log_regularized_beta};
use crate::gamma::ln_gamma_delta;

/// Student-t normalizing constant on the negative log-density scale.
#[must_use]
#[inline]
pub fn student_t_nll_constant(nu: f64) -> f64 {
    f64::midpoint(nu.ln(), std::f64::consts::PI.ln()) - ln_gamma_delta(0.5 * nu, 0.5)
}

/// Prepared standard Student-t density for one fixed number of degrees of freedom.
///
/// Construction validates the degrees of freedom and evaluates the
/// normalizing constant once. Reuse the value when many variates share `nu`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StandardStudentTKernel {
    degrees_of_freedom: f64,
    nll_constant: f64,
}

impl StandardStudentTKernel {
    /// Creates a kernel for finite, strictly positive degrees of freedom.
    #[must_use]
    #[inline]
    pub fn try_new(degrees_of_freedom: f64) -> Option<Self> {
        if degrees_of_freedom <= 0.0 || !degrees_of_freedom.is_finite() {
            return None;
        }
        Some(Self {
            degrees_of_freedom,
            nll_constant: student_t_nll_constant(degrees_of_freedom),
        })
    }

    /// Returns the prepared number of degrees of freedom.
    #[must_use]
    #[inline]
    pub const fn degrees_of_freedom(self) -> f64 {
        self.degrees_of_freedom
    }

    /// Returns the standard density's negative-log normalizing constant.
    #[must_use]
    #[inline]
    pub const fn nll_constant(self) -> f64 {
        self.nll_constant
    }

    /// Evaluates the standard Student-t log-density.
    ///
    /// Returns `NaN` for a `NaN` variate and negative infinity at either
    /// infinite tail.
    #[must_use]
    #[inline]
    pub fn log_pdf(self, t: f64) -> f64 {
        if t.is_nan() {
            return f64::NAN;
        }
        if t.is_infinite() {
            return f64::NEG_INFINITY;
        }
        f64::midpoint(self.degrees_of_freedom, 1.0).mul_add(
            -log_one_plus_square_over_positive(t, self.degrees_of_freedom),
            -self.nll_constant,
        )
    }
}

/// Standard Student-t log-density.
///
/// Returns `NaN` for invalid degrees of freedom or a `NaN` variate, and
/// negative infinity at either infinite tail.
#[must_use]
#[inline]
pub fn student_t_log_pdf_standardized(t: f64, nu: f64) -> f64 {
    StandardStudentTKernel::try_new(nu).map_or(f64::NAN, |kernel| kernel.log_pdf(t))
}

/// Standard Student-t CDF.
///
/// Returns `NaN` for invalid degrees of freedom or a `NaN` variate. Infinite
/// tails map to zero and one.
#[must_use]
#[inline]
pub fn student_t_cdf_standardized(t: f64, nu: f64) -> f64 {
    student_t_log_cdf_standardized(t, nu).exp()
}

/// Logarithm of the standard Student-t CDF.
///
/// Unlike taking the logarithm of [`student_t_cdf_standardized`], this keeps
/// finite log-probabilities in the far negative tail after the probability
/// itself has underflowed.
#[must_use]
#[inline]
pub fn student_t_log_cdf_standardized(t: f64, nu: f64) -> f64 {
    if nu <= 0.0 || !nu.is_finite() || t.is_nan() {
        return f64::NAN;
    }
    if !t.is_finite() {
        return if t.is_sign_negative() {
            f64::NEG_INFINITY
        } else {
            0.0
        };
    }
    if t == 0.0 {
        return -std::f64::consts::LN_2;
    }

    let log_x = -log_one_plus_square_over_positive(t, nu);
    let x = log_x.exp();
    let a = 0.5 * nu;
    let log_beta = if x == 0.0 {
        a.mul_add(log_x, -ln_beta(a, 0.5)) - a.ln()
    } else {
        log_regularized_beta(a, 0.5, x)
    };
    if t < 0.0 {
        log_beta - std::f64::consts::LN_2
    } else {
        (-0.5 * log_beta.exp()).ln_1p()
    }
}

#[inline]
fn log_one_plus_square_over_positive(value: f64, positive: f64) -> f64 {
    let log_ratio = 0.5_f64.mul_add(-positive.ln(), value.abs().ln());
    if log_ratio <= 0.0 {
        (2.0 * log_ratio).exp().ln_1p()
    } else {
        2.0_f64.mul_add(log_ratio, (-2.0 * log_ratio).exp().ln_1p())
    }
}
