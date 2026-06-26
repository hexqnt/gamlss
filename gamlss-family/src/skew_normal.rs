//! Skew-normal distribution parameterizations.

use gamlss_special::{
    invert_real_cdf, log_ndtr, normal_mills_ratio, owens_t, unit_normal_cdf, unit_normal_log_pdf,
};

pub use mean_sd::{
    SkewNormalMeanSd, SkewNormalMeanSdEta, SkewNormalMeanSdNu, SkewNormalMeanSdTheta,
};
pub use mu_sigma_nu::{SkewNormal, SkewNormalEta, SkewNormalMuSigmaNu, SkewNormalTheta};

mod mean_sd;
mod mu_sigma_nu;

const LOG_2: f64 = std::f64::consts::LN_2;
const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;

#[derive(Debug, Clone, Copy)]
struct SkewNormalGradient {
    mu: f64,
    sigma: f64,
    nu: f64,
}

impl SkewNormalGradient {
    #[inline]
    const fn nan() -> Self {
        Self {
            mu: f64::NAN,
            sigma: f64::NAN,
            nu: f64::NAN,
        }
    }
}
#[inline]
fn valid_location_scale(mu: f64, sigma: f64, nu: f64) -> bool {
    mu.is_finite() && sigma > 0.0 && sigma.is_finite() && nu.is_finite()
}

#[inline]
fn nll_location_scale(y: f64, mu: f64, sigma: f64, nu: f64) -> f64 {
    if !y.is_finite() || !valid_location_scale(mu, sigma, nu) {
        return f64::INFINITY;
    }

    let z = (y - mu) / sigma;
    let log_skew_cdf = log_ndtr(nu * z);
    if !log_skew_cdf.is_finite() {
        return f64::INFINITY;
    }

    sigma.ln() - LOG_2 - unit_normal_log_pdf(z) - log_skew_cdf
}

#[inline]
fn nll_gradient_location_scale(y: f64, mu: f64, sigma: f64, nu: f64) -> SkewNormalGradient {
    if !y.is_finite() || !valid_location_scale(mu, sigma, nu) {
        return SkewNormalGradient::nan();
    }

    let z = (y - mu) / sigma;
    let mills = normal_mills_ratio(nu * z);
    if !mills.is_finite() {
        return SkewNormalGradient::nan();
    }

    SkewNormalGradient {
        mu: (nu * mills - z) / sigma,
        sigma: (1.0 - z * z + nu * z * mills) / sigma,
        nu: -z * mills,
    }
}

#[inline]
fn standard_cdf(z: f64, nu: f64) -> f64 {
    (unit_normal_cdf(z) - 2.0 * owens_t(z, nu)).clamp(0.0, 1.0)
}

#[inline]
fn cdf_location_scale(y: f64, mu: f64, sigma: f64, nu: f64) -> f64 {
    if !y.is_finite() || !valid_location_scale(mu, sigma, nu) {
        return f64::NAN;
    }

    standard_cdf((y - mu) / sigma, nu)
}

#[inline]
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

    invert_real_cdf(p, |z| standard_cdf(z, nu)).mul_add(sigma, mu)
}

#[inline]
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
