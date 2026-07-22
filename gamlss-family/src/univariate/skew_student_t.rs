//! Skew Student-t distribution parameterizations.

use gamlss_special::{
    integrate_finite, invert_real_cdf, ln_gamma_delta, student_t_cdf_standardized,
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
fn crps_location_scale(y: f64, mu: f64, sigma: f64, nu: f64, tau: f64) -> f64 {
    if !y.is_finite() || !valid_location_scale(mu, sigma, nu, tau) || tau <= 1.0 {
        return f64::NAN;
    }
    crate::crps::integrate_cdf_crps(y, sigma, |x| cdf_location_scale(x, mu, sigma, nu, tau))
}

#[cfg(feature = "rand")]
fn try_sample_location_scale<Rng>(
    rng: &mut Rng,
    mu: f64,
    sigma: f64,
    nu: f64,
    tau: f64,
) -> Result<f64, gamlss_core::SimulationError>
where
    Rng: rand::Rng,
{
    if !valid_location_scale(mu, sigma, nu, tau) {
        return Err(gamlss_core::SimulationError::InvalidParameters(
            "skew Student-t location/scale",
        ));
    }

    let chi_squared = rand_distr::ChiSquared::new(tau).map_err(|_| {
        gamlss_core::SimulationError::BackendRejected("skew Student-t degrees of freedom")
    })?;
    let mixing = rand_distr::Distribution::sample(&chi_squared, rng);
    let denominator = (mixing / tau).sqrt();
    let delta = nu / nu.hypot(1.0);
    let u = crate::simulation::standard_normal(rng).abs();
    let v = crate::simulation::standard_normal(rng);
    let numerator = (1.0 - delta * delta).max(0.0).sqrt().mul_add(v, delta * u);
    let sample = sigma.mul_add(numerator / denominator, mu);
    if sample.is_finite() {
        Ok(sample)
    } else {
        Err(gamlss_core::SimulationError::NumericalFailure(
            "skew Student-t scale mixture",
        ))
    }
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
    let log_mean_factor =
        0.5 * (tau.ln() - std::f64::consts::PI.ln()) - ln_gamma_delta(0.5 * (tau - 1.0), 0.5);
    let standardized_mean = delta * log_mean_factor.exp();
    let standardized_variance =
        2.0_f64.mul_add(1.0 / (tau - 2.0), 1.0) - standardized_mean * standardized_mean;
    if standardized_variance <= 0.0 || !standardized_variance.is_finite() {
        return None;
    }

    let scale = sigma / standardized_variance.sqrt();
    Some((mean - scale * standardized_mean, scale))
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    #[cfg(feature = "rand")]
    use gamlss_core::TrySimulate;
    use gamlss_core::{Family, HasCdf, HasQuantile};

    use super::{SkewStudentTMeanSdNuTau, SkewStudentTMeanSdTheta, SkewStudentTMuSigmaNuTau};
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

    #[test]
    fn mean_sd_parameterization_preserves_large_tau_normal_limit() {
        let theta = SkewStudentTMeanSdTheta {
            mean: 3.0,
            sigma: 2.0,
            nu: 1.0,
            tau: 1.0e16,
        };
        let standardized_mean = 1.0 / std::f64::consts::PI.sqrt();
        let scale = theta.sigma / (1.0 - standardized_mean * standardized_mean).sqrt();
        let location_scale = SkewStudentTTheta {
            mu: theta.mean - scale * standardized_mean,
            sigma: scale,
            nu: theta.nu,
            tau: theta.tau,
        };
        let mean_sd_family = SkewStudentTMeanSdNuTau::new();
        let location_scale_family = SkewStudentTMuSigmaNuTau::new();
        let actual = mean_sd_family.nll(theta.mean, &theta, &mut ());
        let expected = location_scale_family.nll(theta.mean, &location_scale, &mut ());

        assert!(actual.is_finite(), "nll was {actual}");
        assert_relative_eq!(actual, expected, epsilon = 1.0e-14);
    }

    #[cfg(feature = "rand")]
    #[test]
    fn both_skew_student_t_parameterizations_support_fallible_sampling() {
        use rand::SeedableRng;

        let mut rng = rand::rngs::StdRng::seed_from_u64(23);
        let location_scale = SkewStudentTMuSigmaNuTau::new();
        let sample = location_scale
            .try_sample(
                &mut rng,
                &crate::SkewStudentTTheta {
                    mu: 0.0,
                    sigma: 1.0,
                    nu: 1.5,
                    tau: 5.0,
                },
            )
            .unwrap();
        assert!(sample.is_finite());

        let mean_sd = SkewStudentTMeanSdNuTau::new();
        assert!(
            mean_sd
                .try_sample(
                    &mut rng,
                    &SkewStudentTMeanSdTheta {
                        mean: 0.0,
                        sigma: 1.0,
                        nu: 1.5,
                        tau: 5.0,
                    },
                )
                .is_ok_and(f64::is_finite)
        );
        assert!(
            mean_sd
                .try_sample(
                    &mut rng,
                    &SkewStudentTMeanSdTheta {
                        mean: 0.0,
                        sigma: 1.0,
                        nu: 1.5,
                        tau: 2.0,
                    },
                )
                .is_err()
        );
    }
}
