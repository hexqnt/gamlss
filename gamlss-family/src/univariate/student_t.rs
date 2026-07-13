//! Student's t distribution parameterizations.

use gamlss_core::{Identity, Log, LogPlus};

use gamlss_special::{digamma, invert_real_cdf, ln_beta, ln_gamma, regularized_beta};

pub use dynamic::{StudentTDynamic, StudentTMuSigmaTauEta, StudentTMuSigmaTauTheta};
pub use fixed::{StudentT, StudentTEta};
pub use stddev::{StudentTMuSdTauEta, StudentTMuSdTauTheta, StudentTStdDev};

mod dynamic;
mod fixed;
mod stddev;

/// Student's t distribution with `Identity` link for `mu` and `Log` link
/// for `sigma`.
pub type StudentTMuSigma = StudentT<Identity, Log>;
/// Student's t location-scale distribution with estimated degrees of freedom `tau > 0`.
pub type StudentTMuSigmaTau = StudentTDynamic<Identity, Log, Log>;
/// Student's t distribution parameterized by mean, standard deviation, and `tau > 2`.
pub type StudentTMuSdTau = StudentTStdDev<Identity, Log, LogPlus<2>>;

/// Student's t distribution parameters on the natural location-scale surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudentTTheta {
    /// Location parameter.
    pub mu: f64,
    /// Positive scale parameter.
    pub sigma: f64,
}

pub(super) struct StudentTGradientTheta {
    pub(super) mu: f64,
    pub(super) sigma: f64,
    pub(super) tau: f64,
}

pub(super) fn student_t_nll_theta(nu: f64, y: f64, theta: StudentTTheta) -> f64 {
    if !y.is_finite()
        || !theta.mu.is_finite()
        || theta.sigma <= 0.0
        || !theta.sigma.is_finite()
        || nu <= 0.0
        || !nu.is_finite()
    {
        return f64::INFINITY;
    }

    let z = (y - theta.mu) / theta.sigma;
    student_t_constant(nu) + theta.sigma.ln() + f64::midpoint(nu, 1.0) * (z * z / nu).ln_1p()
}

#[allow(clippy::suboptimal_flops)]
pub(super) fn student_t_nll_gradient_theta(
    nu: f64,
    y: f64,
    theta: StudentTTheta,
) -> StudentTGradientTheta {
    if !y.is_finite()
        || !theta.mu.is_finite()
        || theta.sigma <= 0.0
        || !theta.sigma.is_finite()
        || nu <= 0.0
        || !nu.is_finite()
    {
        return StudentTGradientTheta {
            mu: f64::NAN,
            sigma: f64::NAN,
            tau: f64::NAN,
        };
    }

    let z = (y - theta.mu) / theta.sigma;
    let z2 = z * z;
    let denominator = nu + z2;
    let mu = -(nu + 1.0) * z / (theta.sigma * denominator);
    let sigma = (1.0 - (nu + 1.0) * z2 / denominator) / theta.sigma;
    let tau = 0.5 / nu + 0.5 * digamma(0.5 * nu) - 0.5 * digamma(f64::midpoint(nu, 1.0))
        + 0.5 * (z2 / nu).ln_1p()
        - f64::midpoint(nu, 1.0) * z2 / (nu * denominator);

    StudentTGradientTheta { mu, sigma, tau }
}

pub(super) fn student_t_standard_cdf(nu: f64, t: f64) -> f64 {
    if !t.is_finite() {
        return if t.is_sign_negative() { 0.0 } else { 1.0 };
    }
    if t == 0.0 {
        return 0.5;
    }

    let beta = regularized_beta(0.5 * nu, 0.5, nu / t.mul_add(t, nu));
    if t < 0.0 {
        0.5 * beta
    } else {
        0.5f64.mul_add(-beta, 1.0)
    }
}

#[allow(clippy::float_cmp)]
pub(super) fn student_t_standard_quantile(nu: f64, p: f64) -> f64 {
    if p < 0.0 || !p.is_finite() || p > 1.0 || nu <= 0.0 || !nu.is_finite() {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }

    invert_real_cdf(p, |t| student_t_standard_cdf(nu, t))
}

fn student_t_standard_density(nu: f64, t: f64) -> f64 {
    if !t.is_finite() {
        return 0.0;
    }

    (-student_t_constant(nu) - f64::midpoint(nu, 1.0) * (t * t / nu).ln_1p()).exp()
}

#[allow(clippy::suboptimal_flops)]
fn student_t_standard_crps_constant(nu: f64) -> f64 {
    let log_beta_half_nu_minus_half = ln_beta(0.5, nu - 0.5);
    let log_beta_half_nu_half = ln_beta(0.5, 0.5 * nu);

    2.0 * nu.sqrt() / (nu - 1.0) * (log_beta_half_nu_minus_half - 2.0 * log_beta_half_nu_half).exp()
}

#[allow(clippy::suboptimal_flops)]
pub(super) fn student_t_crps_theta(nu: f64, y: f64, theta: StudentTTheta) -> f64 {
    if nu <= 1.0
        || !nu.is_finite()
        || !y.is_finite()
        || !theta.mu.is_finite()
        || theta.sigma <= 0.0
        || !theta.sigma.is_finite()
    {
        return f64::NAN;
    }

    let z = (y - theta.mu) / theta.sigma;
    let cdf = student_t_standard_cdf(nu, z);
    let density = student_t_standard_density(nu, z);
    let tail_moment = 2.0 * density * z.mul_add(z, nu) / (nu - 1.0);

    theta.sigma * (z * (2.0 * cdf - 1.0) + tail_moment - student_t_standard_crps_constant(nu))
}

/// Normalization constant for the logarithm of the Student's t density.
fn student_t_constant(nu: f64) -> f64 {
    f64::midpoint(nu.ln(), std::f64::consts::PI.ln()) + ln_gamma(0.5 * nu)
        - ln_gamma(f64::midpoint(nu, 1.0))
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;
    use gamlss_core::{Family, HasCdf, HasCrps, HasQuantile};
    use statrs::distribution::{ContinuousCDF, StudentsT};

    use super::{
        StudentTEta, StudentTMuSdTau, StudentTMuSdTauEta, StudentTMuSdTauTheta, StudentTMuSigma,
        StudentTMuSigmaTau, StudentTMuSigmaTauTheta, StudentTTheta,
    };
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn student_t_rejects_invalid_degrees_of_freedom() {
        assert!(StudentTMuSigma::try_new(0.0).is_err());
        assert!(StudentTMuSigma::try_new(f64::INFINITY).is_err());
    }

    #[test]
    fn student_t_gradient_matches_finite_difference() {
        let family = StudentTMuSigma::try_new(5.0).unwrap();
        assert_gradient_matches_finite_difference::<_, 2>(&family, 1.7, [0.4, -0.2]);
    }

    #[test]
    fn student_t_rejects_non_finite_domain_and_returns_nan_gradient() {
        let family = StudentTMuSigma::try_new(5.0).unwrap();
        let theta = StudentTTheta {
            mu: 0.4,
            sigma: 0.8,
        };

        assert!(family.nll(1.7, &theta, &mut family.workspace()).is_finite());
        assert!(
            family
                .nll(f64::NEG_INFINITY, &theta, &mut family.workspace())
                .is_infinite()
        );
        assert!(
            family
                .nll(
                    1.7,
                    &StudentTTheta {
                        mu: f64::NAN,
                        sigma: theta.sigma,
                    },
                    &mut family.workspace(),
                )
                .is_infinite()
        );
        assert!(
            family
                .nll(
                    1.7,
                    &StudentTTheta {
                        mu: theta.mu,
                        sigma: 0.0,
                    },
                    &mut family.workspace(),
                )
                .is_infinite()
        );

        let (nll, gradient) = family.nll_and_gradient_eta(
            1.7,
            &StudentTEta {
                mu: 0.4,
                sigma: f64::NEG_INFINITY,
            },
            &mut family.workspace(),
        );
        assert!(nll.is_infinite());
        assert!(gradient.mu.is_nan());
        assert!(gradient.sigma.is_nan());
    }

    #[test]
    fn student_t_cdf_matches_reference_points() {
        let family = StudentTMuSigma::try_new(5.0).unwrap();
        let theta = StudentTTheta {
            mu: 0.4,
            sigma: 0.8,
        };

        assert_relative_eq!(family.cdf(theta.mu, &theta), 0.5, epsilon = 1.0e-12);
        assert_relative_eq!(
            family.cdf(theta.mu + theta.sigma, &theta),
            0.818_391_266_175_438_7,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            family.cdf(theta.mu - theta.sigma, &theta),
            0.181_608_733_824_561_27,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn student_t_cdf_and_quantile_match_statrs_reference() {
        let family = StudentTMuSigma::try_new(7.0).unwrap();
        let theta = StudentTTheta {
            mu: 0.4,
            sigma: 0.8,
        };
        let reference = StudentsT::new(theta.mu, theta.sigma, 7.0).unwrap();

        for y in [-2.0, -0.3, 0.4, 1.2, 3.0] {
            assert_relative_eq!(family.cdf(y, &theta), reference.cdf(y), epsilon = 1.0e-11);
        }

        for p in [0.01, 0.1, 0.5, 0.9, 0.99] {
            assert_relative_eq!(
                family.quantile(p, &theta),
                reference.inverse_cdf(p),
                epsilon = 1.0e-10
            );
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn student_t_quantile_inverts_cdf() {
        let family = StudentTMuSigma::try_new(5.0).unwrap();
        let theta = StudentTTheta {
            mu: 0.4,
            sigma: 0.8,
        };

        let y = family.quantile(0.75, &theta);

        assert_relative_eq!(family.cdf(y, &theta), 0.75, epsilon = 1.0e-12);
        assert_eq!(family.quantile(0.0, &theta), f64::NEG_INFINITY);
        assert_eq!(family.quantile(1.0, &theta), f64::INFINITY);
        assert!(family.quantile(f64::NAN, &theta).is_nan());
    }

    #[test]
    fn student_t_cdf_returns_nan_for_invalid_domains() {
        let family = StudentTMuSigma::try_new(5.0).unwrap();

        assert!(
            family
                .cdf(
                    1.0,
                    &StudentTTheta {
                        mu: 0.0,
                        sigma: 0.0,
                    },
                )
                .is_nan()
        );
        assert!(
            family
                .cdf(
                    f64::NAN,
                    &StudentTTheta {
                        mu: 0.0,
                        sigma: 1.0,
                    },
                )
                .is_nan()
        );
    }

    #[test]
    fn student_t_crps_matches_fixed_values() {
        let family = StudentTMuSigma::try_new(5.0).unwrap();

        assert_relative_eq!(
            family.crps(
                1.0,
                &StudentTTheta {
                    mu: 0.0,
                    sigma: 1.0,
                },
            ),
            0.603_830_562_748_23,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            family.crps(
                0.0,
                &StudentTTheta {
                    mu: 0.0,
                    sigma: 1.0,
                },
            ),
            0.257_025_362_900_647_5,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn student_t_crps_scales_with_sigma() {
        let family = StudentTMuSigma::try_new(5.0).unwrap();

        assert_relative_eq!(
            family.crps(
                2.0,
                &StudentTTheta {
                    mu: 0.0,
                    sigma: 2.0,
                },
            ),
            2.0 * family.crps(
                1.0,
                &StudentTTheta {
                    mu: 0.0,
                    sigma: 1.0,
                },
            ),
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn student_t_crps_returns_nan_for_invalid_domains() {
        let family = StudentTMuSigma::try_new(5.0).unwrap();

        assert!(
            family
                .crps(
                    f64::NAN,
                    &StudentTTheta {
                        mu: 0.0,
                        sigma: 1.0,
                    },
                )
                .is_nan()
        );
        assert!(
            family
                .crps(
                    1.0,
                    &StudentTTheta {
                        mu: 0.0,
                        sigma: 0.0,
                    },
                )
                .is_nan()
        );
        assert!(
            StudentTMuSigma::try_new(1.0)
                .unwrap()
                .crps(
                    1.0,
                    &StudentTTheta {
                        mu: 0.0,
                        sigma: 1.0,
                    },
                )
                .is_nan()
        );
        assert!(
            StudentTMuSigmaTau::new()
                .crps(
                    1.0,
                    &StudentTMuSigmaTauTheta {
                        mu: 0.0,
                        sigma: 1.0,
                        tau: 0.5,
                    },
                )
                .is_nan()
        );
    }

    #[test]
    fn student_t_crps_is_nonnegative_for_valid_domains() {
        let family = StudentTMuSigma::try_new(5.0).unwrap();

        assert!(
            family.crps(
                1.0,
                &StudentTTheta {
                    mu: 0.0,
                    sigma: 1.0,
                },
            ) >= 0.0
        );
    }

    #[cfg(feature = "rand")]
    #[test]
    fn student_t_sampling_returns_finite_values_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = StudentTMuSigma::try_new(5.0).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        assert!(
            family
                .sample(
                    &mut rng,
                    &StudentTTheta {
                        mu: 0.0,
                        sigma: 1.0
                    }
                )
                .is_finite()
        );
        assert!(
            family
                .sample(
                    &mut rng,
                    &StudentTTheta {
                        mu: 0.0,
                        sigma: 0.0
                    }
                )
                .is_nan()
        );
    }

    #[test]
    fn student_t_stddev_parameterization_matches_scale_form() {
        let stddev_family = StudentTMuSdTau::new();
        let scale_family = StudentTMuSigmaTau::new();
        let theta = StudentTMuSdTauTheta {
            mu: 0.3,
            sigma: 1.4,
            tau: 7.0,
        };
        let scale_theta = StudentTMuSigmaTauTheta {
            mu: theta.mu,
            sigma: theta.sigma * ((theta.tau - 2.0) / theta.tau).sqrt(),
            tau: theta.tau,
        };

        for y in [-2.0, 0.3, 1.7, 4.0] {
            assert_relative_eq!(
                stddev_family.nll(y, &theta, &mut stddev_family.workspace()),
                scale_family.nll(y, &scale_theta, &mut scale_family.workspace()),
                epsilon = 1.0e-12
            );
            assert_relative_eq!(
                stddev_family.cdf(y, &theta),
                scale_family.cdf(y, &scale_theta),
                epsilon = 1.0e-12
            );
            assert_relative_eq!(
                stddev_family.crps(y, &theta),
                scale_family.crps(y, &scale_theta),
                epsilon = 1.0e-12
            );
        }

        for p in [0.01, 0.5, 0.95] {
            assert_relative_eq!(
                stddev_family.quantile(p, &theta),
                scale_family.quantile(p, &scale_theta),
                epsilon = 1.0e-10
            );
        }
    }

    #[test]
    fn student_t_dynamic_accepts_heavy_tails_without_finite_variance() {
        let family = StudentTMuSigmaTau::new();
        let theta = StudentTMuSigmaTauTheta {
            mu: 0.0,
            sigma: 1.0,
            tau: 1.5,
        };

        assert!(family.nll(0.2, &theta, &mut family.workspace()).is_finite());
        assert!(family.cdf(0.2, &theta).is_finite());
        assert!(family.quantile(0.5, &theta).is_finite());
    }

    #[test]
    fn student_t_stddev_gradient_matches_finite_difference() {
        let family = StudentTMuSdTau::new();
        assert_gradient_matches_finite_difference::<_, 3>(&family, 1.7, [0.4, -0.2, 5.0_f64.ln()]);
    }

    #[test]
    fn student_t_stddev_rejects_invalid_domains() {
        let family = StudentTMuSdTau::new();
        let valid = StudentTMuSdTauTheta {
            mu: 0.0,
            sigma: 1.0,
            tau: 5.0,
        };

        assert!(family.nll(0.2, &valid, &mut family.workspace()).is_finite());
        assert!(
            family
                .nll(
                    0.2,
                    &StudentTMuSdTauTheta {
                        sigma: 0.0,
                        ..valid
                    },
                    &mut family.workspace(),
                )
                .is_infinite()
        );
        assert!(
            family
                .cdf(0.2, &StudentTMuSdTauTheta { tau: 2.0, ..valid },)
                .is_nan()
        );
        assert!(
            family
                .quantile(
                    0.5,
                    &StudentTMuSdTauTheta {
                        tau: f64::INFINITY,
                        ..valid
                    },
                )
                .is_nan()
        );

        let (nll, gradient) = family.nll_and_gradient_eta(
            1.7,
            &StudentTMuSdTauEta {
                mu: 0.4,
                sigma: f64::NEG_INFINITY,
                tau: 5.0_f64.ln(),
            },
            &mut family.workspace(),
        );
        assert!(nll.is_infinite());
        assert!(gradient.mu.is_nan());
        assert!(gradient.sigma.is_nan());
        assert!(gradient.tau.is_nan());
    }
}
