//! Skew-normal distribution parameterizations.

use gamlss_special::{
    invert_real_cdf, log_ndtr, normal_mills_ratio, owens_t, unit_normal_cdf, unit_normal_log_pdf,
};

use crate::constants::{INV_SQRT_2_PI, LOG_2, SQRT_2_OVER_PI};

pub use mean_sd::{
    SkewNormalMeanSd, SkewNormalMeanSdEta, SkewNormalMeanSdNu, SkewNormalMeanSdTheta,
};
pub use mu_sigma_nu::{SkewNormal, SkewNormalEta, SkewNormalMuSigmaNu, SkewNormalTheta};

mod mean_sd;
mod mu_sigma_nu;

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

    #[allow(clippy::suboptimal_flops)]
    SkewNormalGradient {
        mu: (nu * mills - z) / sigma,
        sigma: (1.0 - z * z + nu * z * mills) / sigma,
        nu: -z * mills,
    }
}

#[inline]
#[allow(clippy::suboptimal_flops)]
fn standard_cdf(z: f64, nu: f64) -> f64 {
    if z == f64::NEG_INFINITY {
        return 0.0;
    }
    if z == f64::INFINITY {
        return 1.0;
    }
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
#[allow(clippy::suboptimal_flops)]
fn crps_location_scale(y: f64, mu: f64, sigma: f64, nu: f64) -> f64 {
    if !y.is_finite() || !valid_location_scale(mu, sigma, nu) {
        return f64::NAN;
    }

    let z = (y - mu) / sigma;
    let delta = nu / nu.hypot(1.0);
    let shape_norm = nu.hypot(1.0);
    let cdf = standard_cdf(z, nu);
    let density = if z.is_finite() {
        2.0 * INV_SQRT_2_PI * (-0.5 * z * z).exp() * unit_normal_cdf(nu * z)
    } else {
        0.0
    };
    let standardized_mean = delta * SQRT_2_OVER_PI;

    // The closed form follows from CRPS = E|X - y| - E|X - X'| / 2.
    // The final term below is half the Gini mean difference of a standard
    // skew-normal variate.
    let half_gini = 2.0 * SQRT_2_OVER_PI / std::f64::consts::PI
        * (std::f64::consts::SQRT_2 * shape_norm.atan()
            - delta * (nu / std::f64::consts::SQRT_2).atan());
    let standardized_correction = 2.0 * density
        - 2.0 * standardized_mean * unit_normal_cdf(z * shape_norm)
        + standardized_mean
        - half_gini;
    (y - mu)
        .mul_add(2.0 * cdf - 1.0, sigma * standardized_correction)
        .max(0.0)
}

#[inline]
fn quantile_location_scale(p: f64, mu: f64, sigma: f64, nu: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) || !valid_location_scale(mu, sigma, nu) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    #[allow(clippy::float_cmp)]
    if p == 1.0 {
        return f64::INFINITY;
    }

    invert_real_cdf(p, |z| standard_cdf(z, nu)).mul_add(sigma, mu)
}

#[cfg(feature = "rand")]
#[inline]
fn try_sample_location_scale<Rng>(
    rng: &mut Rng,
    mu: f64,
    sigma: f64,
    nu: f64,
) -> Result<f64, gamlss_core::SimulationError>
where
    Rng: rand::Rng,
{
    if !valid_location_scale(mu, sigma, nu) {
        return Err(gamlss_core::SimulationError::InvalidParameters(
            "skew-normal location/scale",
        ));
    }

    let delta = nu / nu.hypot(1.0);
    let u = crate::simulation::standard_normal(rng).abs();
    let v = crate::simulation::standard_normal(rng);
    let standardized = (1.0 - delta * delta).max(0.0).sqrt().mul_add(v, delta * u);
    crate::simulation::ensure_finite(sigma.mul_add(standardized, mu), "skew-normal transform")
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

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::HasCrps;

    use super::{
        SQRT_2_OVER_PI, SkewNormalMeanSdNu, SkewNormalMeanSdTheta, SkewNormalMuSigmaNu,
        SkewNormalTheta,
    };
    use crate::{NormalMuSigma, NormalTheta};

    #[test]
    fn skew_normal_crps_matches_fixed_values() {
        let family = SkewNormalMuSigmaNu::new();
        let cases = [
            (0.0, 0.0, 1.0, 0.0, 0.233_694_977_255_109_1),
            (0.7, -0.3, 1.4, 2.0, 0.228_347_498_427_710_96),
            (-1.2, 0.4, 0.8, -3.5, 0.728_503_282_263_107_8),
            (3.0, 1.0, 2.0, 100.0, 0.409_765_909_347_208_3),
        ];

        for (y, mu, sigma, nu, expected) in cases {
            assert_relative_eq!(
                family.crps(y, &SkewNormalTheta { mu, sigma, nu }),
                expected,
                epsilon = 1.0e-12
            );
        }
    }

    #[test]
    fn symmetric_skew_normal_crps_matches_normal() {
        let skew_normal = SkewNormalMuSigmaNu::new();
        let normal = NormalMuSigma::new();
        let skew_theta = SkewNormalTheta {
            mu: -0.4,
            sigma: 1.7,
            nu: 0.0,
        };
        let normal_theta = NormalTheta {
            mu: skew_theta.mu,
            sigma: skew_theta.sigma,
        };

        for y in [-5.0, -0.4, 0.7, 8.0] {
            assert_relative_eq!(
                skew_normal.crps(y, &skew_theta),
                normal.crps(y, &normal_theta),
                epsilon = 2.0e-14
            );
        }
    }

    #[test]
    fn skew_normal_parameterizations_have_identical_crps() {
        let location_scale = SkewNormalMuSigmaNu::new();
        let mean_sd = SkewNormalMeanSdNu::new();
        let theta = SkewNormalTheta {
            mu: -0.3,
            sigma: 1.4,
            nu: 2.0,
        };
        let delta = theta.nu / theta.nu.hypot(1.0);
        let standardized_mean = SQRT_2_OVER_PI * delta;
        let equivalent = SkewNormalMeanSdTheta {
            mean: theta.sigma.mul_add(standardized_mean, theta.mu),
            sigma: theta.sigma * (1.0 - standardized_mean * standardized_mean).sqrt(),
            nu: theta.nu,
        };

        for y in [-2.0, 0.0, 0.7, 3.0] {
            assert_relative_eq!(
                mean_sd.crps(y, &equivalent),
                location_scale.crps(y, &theta),
                epsilon = 2.0e-14
            );
        }
    }

    #[test]
    fn skew_normal_crps_rejects_invalid_domains() {
        let location_scale = SkewNormalMuSigmaNu::new();
        let mean_sd = SkewNormalMeanSdNu::new();

        assert!(
            location_scale
                .crps(
                    f64::NAN,
                    &SkewNormalTheta {
                        mu: 0.0,
                        sigma: 1.0,
                        nu: 2.0,
                    },
                )
                .is_nan()
        );
        assert!(
            location_scale
                .crps(
                    0.0,
                    &SkewNormalTheta {
                        mu: 0.0,
                        sigma: 0.0,
                        nu: 2.0,
                    },
                )
                .is_nan()
        );
        assert!(
            mean_sd
                .crps(
                    0.0,
                    &SkewNormalMeanSdTheta {
                        mean: 0.0,
                        sigma: 1.0,
                        nu: f64::INFINITY,
                    },
                )
                .is_nan()
        );

        let extreme = location_scale.crps(
            f64::MAX,
            &SkewNormalTheta {
                mu: 0.0,
                sigma: f64::MIN_POSITIVE,
                nu: 2.0,
            },
        );
        assert!(extreme.is_finite());
    }
}
