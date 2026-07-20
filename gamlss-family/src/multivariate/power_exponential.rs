#![allow(clippy::suboptimal_flops)]

//! Multivariate power-exponential distributions.

use gamlss_special::{digamma, ln_gamma};

pub use cholesky::{
    MvPowerExponentialCholesky, MvPowerExponentialCholeskyDefault, MvPowerExponentialCholeskyEta,
    MvPowerExponentialCholeskyTheta,
};
pub use mean_std_partial_corr::{
    MvPowerExponentialMeanStdPartialCorr, MvPowerExponentialMeanStdPartialCorrDefault,
    MvPowerExponentialMeanStdPartialCorrEta, MvPowerExponentialMeanStdPartialCorrTheta,
};

mod cholesky;
mod mean_std_partial_corr;

pub(super) fn radial_nll_constant(dimension: f64, power: f64) -> f64 {
    -power.ln() - ln_gamma(0.5 * dimension)
        + (1.0 + dimension / power) * std::f64::consts::LN_2
        + 0.5 * dimension * std::f64::consts::PI.ln()
        + ln_gamma(dimension / power)
}

pub(super) fn direct_power_score(
    dimension: f64,
    power: f64,
    quadratic: f64,
    radial_power: f64,
) -> f64 {
    let radial_derivative = if quadratic == 0.0 {
        0.0
    } else {
        0.25 * radial_power * quadratic.ln()
    };
    -1.0 / power
        - dimension * (std::f64::consts::LN_2 + digamma(dimension / power)) / (power * power)
        + radial_derivative
}

pub(super) fn log_covariance_multiplier(dimension: f64, power: f64) -> f64 {
    2.0 * std::f64::consts::LN_2 / power + ln_gamma((dimension + 2.0) / power)
        - ln_gamma(dimension / power)
        - dimension.ln()
}

pub(super) fn d_log_covariance_multiplier_d_power(dimension: f64, power: f64) -> f64 {
    (-2.0 * std::f64::consts::LN_2 - (dimension + 2.0) * digamma((dimension + 2.0) / power)
        + dimension * digamma(dimension / power))
        / (power * power)
}
