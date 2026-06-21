//! Skew-normal distribution parameterizations.

use crate::special::{invert_real_cdf, owens_t, unit_normal_cdf, unit_normal_log_pdf};

pub use mean_sd::{
    SkewNormalMeanSd, SkewNormalMeanSdEta, SkewNormalMeanSdNu, SkewNormalMeanSdTheta,
};
pub use mu_sigma_nu::{SkewNormal, SkewNormalEta, SkewNormalMuSigmaNu, SkewNormalTheta};

mod mean_sd;
mod mu_sigma_nu;

const LOG_2: f64 = std::f64::consts::LN_2;
const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;

#[inline(always)]
fn valid_location_scale(mu: f64, sigma: f64, nu: f64) -> bool {
    mu.is_finite() && sigma > 0.0 && sigma.is_finite() && nu.is_finite()
}

#[inline(always)]
fn nll_location_scale(y: f64, mu: f64, sigma: f64, nu: f64) -> f64 {
    if !y.is_finite() || !valid_location_scale(mu, sigma, nu) {
        return f64::INFINITY;
    }

    let z = (y - mu) / sigma;
    let skew_cdf = unit_normal_cdf(nu * z);
    if skew_cdf <= 0.0 {
        return f64::INFINITY;
    }

    sigma.ln() - LOG_2 - unit_normal_log_pdf(z) - skew_cdf.ln()
}

#[inline(always)]
fn standard_cdf(z: f64, nu: f64) -> f64 {
    (unit_normal_cdf(z) - 2.0 * owens_t(z, nu)).clamp(0.0, 1.0)
}

#[inline(always)]
fn cdf_location_scale(y: f64, mu: f64, sigma: f64, nu: f64) -> f64 {
    if !y.is_finite() || !valid_location_scale(mu, sigma, nu) {
        return f64::NAN;
    }

    standard_cdf((y - mu) / sigma, nu)
}

#[inline(always)]
fn quantile_location_scale(p: f64, mu: f64, sigma: f64, nu: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) || !valid_location_scale(mu, sigma, nu) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }

    invert_real_cdf(p, |z| standard_cdf(z, nu)) * sigma + mu
}

#[inline(always)]
fn mean_sd_to_location_scale(mean: f64, sigma: f64, nu: f64) -> Option<(f64, f64)> {
    if !mean.is_finite() || sigma <= 0.0 || !sigma.is_finite() || !nu.is_finite() {
        return None;
    }

    let delta = nu / nu.hypot(1.0);
    let standardized_mean = SQRT_2_OVER_PI * delta;
    let standardized_variance = 1.0 - standardized_mean * standardized_mean;
    if standardized_variance <= 0.0 || !standardized_variance.is_finite() {
        return None;
    }

    let scale = sigma / standardized_variance.sqrt();
    Some((mean - scale * standardized_mean, scale))
}
