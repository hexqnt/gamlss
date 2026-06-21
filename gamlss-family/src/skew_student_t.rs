//! Skew Student-t distribution parameterizations.

use crate::special::{
    integrate_finite, invert_bounded_cdf, ln_gamma, student_t_cdf_standardized,
    student_t_log_pdf_standardized,
};

pub use mean_sd::{
    SkewStudentTMeanSd, SkewStudentTMeanSdEta, SkewStudentTMeanSdNuTau, SkewStudentTMeanSdTheta,
};
pub use mu_sigma_nu_tau::{
    SkewStudentT, SkewStudentTEta, SkewStudentTMuSigmaNuTau, SkewStudentTTheta,
};

mod mean_sd;
mod mu_sigma_nu_tau;

const LOG_2: f64 = std::f64::consts::LN_2;

#[inline(always)]
fn valid_location_scale(mu: f64, sigma: f64, nu: f64, tau: f64) -> bool {
    mu.is_finite()
        && sigma > 0.0
        && sigma.is_finite()
        && nu.is_finite()
        && tau > 0.0
        && tau.is_finite()
}

#[inline(always)]
fn skew_argument(z: f64, nu: f64, tau: f64) -> f64 {
    nu * z * ((tau + 1.0) / (tau + z * z)).sqrt()
}

#[inline(always)]
fn standard_density(z: f64, nu: f64, tau: f64) -> f64 {
    let skew = student_t_cdf_standardized(skew_argument(z, nu, tau), tau + 1.0);
    if skew <= 0.0 {
        return 0.0;
    }

    (LOG_2 + student_t_log_pdf_standardized(z, tau) + skew.ln()).exp()
}

#[inline(always)]
fn nll_location_scale(y: f64, mu: f64, sigma: f64, nu: f64, tau: f64) -> f64 {
    if !y.is_finite() || !valid_location_scale(mu, sigma, nu, tau) {
        return f64::INFINITY;
    }

    let z = (y - mu) / sigma;
    let skew = student_t_cdf_standardized(skew_argument(z, nu, tau), tau + 1.0);
    if skew <= 0.0 {
        return f64::INFINITY;
    }

    sigma.ln() - LOG_2 - student_t_log_pdf_standardized(z, tau) - skew.ln()
}

#[inline(always)]
fn cdf_location_scale(y: f64, mu: f64, sigma: f64, nu: f64, tau: f64) -> f64 {
    if !y.is_finite() || !valid_location_scale(mu, sigma, nu, tau) {
        return f64::NAN;
    }

    let z = (y - mu) / sigma;
    if z <= -100.0 {
        return 0.0;
    }
    if z >= 100.0 {
        return 1.0;
    }

    integrate_finite(-100.0, z, |t| standard_density(t, nu, tau)).clamp(0.0, 1.0)
}

#[inline(always)]
fn quantile_location_scale(p: f64, mu: f64, sigma: f64, nu: f64, tau: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) || !valid_location_scale(mu, sigma, nu, tau) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }

    invert_bounded_cdf(p, mu - 100.0 * sigma, mu + 100.0 * sigma, |y| {
        cdf_location_scale(y, mu, sigma, nu, tau)
    })
}

#[inline(always)]
fn mean_sd_to_location_scale(mean: f64, sigma: f64, nu: f64, tau: f64) -> Option<(f64, f64)> {
    if !mean.is_finite()
        || sigma <= 0.0
        || !sigma.is_finite()
        || !nu.is_finite()
        || tau <= 2.0
        || !tau.is_finite()
    {
        return None;
    }

    let delta = nu / nu.hypot(1.0);
    let log_mean_factor = 0.5 * tau.ln() + ln_gamma(0.5 * (tau - 1.0))
        - 0.5 * std::f64::consts::PI.ln()
        - ln_gamma(0.5 * tau);
    let standardized_mean = delta * log_mean_factor.exp();
    let standardized_variance = tau / (tau - 2.0) - standardized_mean * standardized_mean;
    if standardized_variance <= 0.0 || !standardized_variance.is_finite() {
        return None;
    }

    let scale = sigma / standardized_variance.sqrt();
    Some((mean - scale * standardized_mean, scale))
}
