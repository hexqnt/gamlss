//! Skew Student-t distribution parameterizations.

use gamlss_special::{
    integrate_finite, invert_real_cdf, ln_gamma, student_t_cdf_standardized,
    student_t_log_pdf_standardized,
};

use crate::constants::LOG_2;

pub use mean_sd::{
    SkewStudentTMeanSd, SkewStudentTMeanSdEta, SkewStudentTMeanSdNuTau, SkewStudentTMeanSdTheta,
};
pub use mu_sigma_nu_tau::{
    SkewStudentT, SkewStudentTEta, SkewStudentTMuSigmaNuTau, SkewStudentTTheta,
};

mod mean_sd;
mod mu_sigma_nu_tau;

#[inline]
fn valid_location_scale(mu: f64, sigma: f64, nu: f64, tau: f64) -> bool {
    mu.is_finite()
        && sigma > 0.0
        && sigma.is_finite()
        && nu.is_finite()
        && tau > 0.0
        && tau.is_finite()
}

#[inline]
#[allow(clippy::suboptimal_flops)]
fn skew_argument(z: f64, nu: f64, tau: f64) -> f64 {
    let standardized = if z.is_infinite() {
        z.signum()
    } else {
        z / z.hypot(tau.sqrt())
    };
    nu * (tau + 1.0).sqrt() * standardized
}

#[inline]
fn standard_density(z: f64, nu: f64, tau: f64) -> f64 {
    let skew = student_t_cdf_standardized(skew_argument(z, nu, tau), tau + 1.0);
    if skew <= 0.0 {
        return 0.0;
    }

    (LOG_2 + student_t_log_pdf_standardized(z, tau) + skew.ln()).exp()
}

#[inline]
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

#[inline]
fn cdf_location_scale(y: f64, mu: f64, sigma: f64, nu: f64, tau: f64) -> f64 {
    if !y.is_finite() || !valid_location_scale(mu, sigma, nu, tau) {
        return f64::NAN;
    }

    let z = (y - mu) / sigma;
    let base_cdf = student_t_cdf_standardized(z, tau);
    if nu == 0.0 {
        return base_cdf;
    }

    let cdf_at_zero = 0.5 - nu.atan() / std::f64::consts::PI;
    let transformed_density = |s: f64| {
        let t = s.sinh();
        standard_density(t, nu, tau) * s.cosh()
    };
    let cdf = if z < 0.0 {
        cdf_at_zero - integrate_finite(z.asinh(), 0.0, transformed_density)
    } else {
        cdf_at_zero + integrate_finite(0.0, z.asinh(), transformed_density)
    };
    cdf.clamp(0.0, 1.0)
}

#[inline]
fn quantile_location_scale(p: f64, mu: f64, sigma: f64, nu: f64, tau: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) || !valid_location_scale(mu, sigma, nu, tau) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    #[allow(clippy::float_cmp)]
    if p == 1.0 {
        return f64::INFINITY;
    }

    invert_real_cdf(p, |y| cdf_location_scale(y, mu, sigma, nu, tau))
}

#[inline]
#[allow(clippy::suboptimal_flops)]
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

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{HasCdf, HasQuantile};

    use super::SkewStudentTMuSigmaNuTau;
    use crate::{SkewStudentTTheta, StudentTMuSigmaTau, StudentTMuSigmaTauTheta};

    #[test]
    fn symmetric_heavy_tail_cdf_and_quantile_are_not_truncated_at_one_hundred_scales() {
        let skew = SkewStudentTMuSigmaNuTau::new();
        let student = StudentTMuSigmaTau::new();
        let skew_theta = SkewStudentTTheta {
            mu: 0.0,
            sigma: 1.0,
            nu: 0.0,
            tau: 0.5,
        };
        let student_theta = StudentTMuSigmaTauTheta {
            mu: 0.0,
            sigma: 1.0,
            tau: 0.5,
        };

        let cdf = skew.cdf(-100.0, &skew_theta);
        assert!(cdf > 0.0);
        assert_relative_eq!(cdf, student.cdf(-100.0, &student_theta), epsilon = 1.0e-14);

        let quantile = skew.quantile(0.01, &skew_theta);
        assert!(quantile < -100.0);
        assert_relative_eq!(skew.cdf(quantile, &skew_theta), 0.01, epsilon = 1.0e-10);
    }

    #[test]
    fn skew_cdf_obeys_reflection_identity_in_heavy_tails() {
        let family = SkewStudentTMuSigmaNuTau::new();
        let left = family.cdf(
            -250.0,
            &SkewStudentTTheta {
                mu: 0.0,
                sigma: 1.0,
                nu: 2.0,
                tau: 0.7,
            },
        );
        let reflected = family.cdf(
            250.0,
            &SkewStudentTTheta {
                mu: 0.0,
                sigma: 1.0,
                nu: -2.0,
                tau: 0.7,
            },
        );

        assert!(left > 0.0);
        assert_relative_eq!(left + reflected, 1.0, epsilon = 2.0e-9);
    }
}
