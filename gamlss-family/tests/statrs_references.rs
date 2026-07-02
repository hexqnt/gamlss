#![allow(
    clippy::float_cmp,
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
use gamlss_core::{HasCdf, HasDensity, HasQuantile};
use gamlss_family::*;
use statrs::distribution::{
    Bernoulli as StatrsBernoulli, Beta as StatrsBeta, ContinuousCDF, Discrete, DiscreteCDF,
    Exp as StatrsExp, Gamma as StatrsGamma, Gumbel as StatrsGumbel, Laplace as StatrsLaplace,
    LogNormal as StatrsLogNormal, NegativeBinomial as StatrsNegativeBinomial,
    Normal as StatrsNormal, Poisson as StatrsPoisson, StudentsT as StatrsStudentsT,
    Weibull as StatrsWeibull,
};

use common::{
    ContinuousReferenceTolerances, POSITIVE_REFERENCE_POINTS, REAL_REFERENCE_POINTS,
    TAIL_PROBABILITIES, assert_close, assert_continuous_statrs_reference,
    assert_discrete_statrs_reference, nb_success_probability, statrs_discrete_quantile,
};

#[path = "common/helpers.rs"]
mod common;

#[test]
fn cdf_quantile_and_density_match_statrs_references() {
    let bernoulli = BernoulliProbability::new();
    let bernoulli_theta = BernoulliTheta { mu: 0.35 };
    let statrs_bernoulli = StatrsBernoulli::new(bernoulli_theta.mu).unwrap();
    for y in [0.0, 1.0] {
        assert_close(
            bernoulli.density(y, &bernoulli_theta),
            statrs_bernoulli.pmf(y as u64),
            0.0,
            1.0e-14,
        );
    }
    for y in [-1.0_f64, 0.0, 0.5, 1.0] {
        let expected = if y < 0.0 {
            0.0
        } else {
            statrs_bernoulli.cdf(y.floor() as u64)
        };
        assert_close(bernoulli.cdf(y, &bernoulli_theta), expected, 0.0, 1.0e-14);
    }

    let beta = BetaMeanPrecision::new();
    let beta_theta = BetaTheta {
        mu: 0.4,
        precision: 5.0,
    };
    let statrs_beta = StatrsBeta::new(
        beta_theta.mu * beta_theta.precision,
        (1.0 - beta_theta.mu) * beta_theta.precision,
    )
    .unwrap();
    assert_continuous_statrs_reference(
        &beta,
        beta_theta,
        &statrs_beta,
        &[0.05, 0.2, 0.6, 0.95],
        ContinuousReferenceTolerances {
            cdf_abs: 2.0e-11,
            density_rel: 1.0e-10,
            density_abs: 1.0e-10,
            quantile_abs: 2.0e-10,
        },
    );

    let exponential = ExponentialRate::new();
    let exponential_theta = ExponentialRateTheta { rate: 1.7 };
    let statrs_exponential = StatrsExp::new(exponential_theta.rate).unwrap();
    assert_continuous_statrs_reference(
        &exponential,
        exponential_theta,
        &statrs_exponential,
        &POSITIVE_REFERENCE_POINTS,
        ContinuousReferenceTolerances {
            cdf_abs: 1.0e-14,
            density_rel: 0.0,
            density_abs: 1.0e-14,
            quantile_abs: 1.0e-12,
        },
    );

    let gamma = GammaShapeRate::new();
    let gamma_theta = GammaTheta {
        shape: 2.3,
        rate: 1.4,
    };
    let statrs_gamma = StatrsGamma::new(gamma_theta.shape, gamma_theta.rate).unwrap();
    assert_continuous_statrs_reference(
        &gamma,
        gamma_theta,
        &statrs_gamma,
        &POSITIVE_REFERENCE_POINTS,
        ContinuousReferenceTolerances {
            cdf_abs: 2.0e-10,
            density_rel: 1.0e-10,
            density_abs: 1.0e-10,
            quantile_abs: 2.0e-8,
        },
    );

    let gumbel = GumbelMuSigma::new();
    let gumbel_theta = GumbelTheta {
        mu: -0.3,
        sigma: 1.2,
    };
    let statrs_gumbel = StatrsGumbel::new(gumbel_theta.mu, gumbel_theta.sigma).unwrap();
    assert_continuous_statrs_reference(
        &gumbel,
        gumbel_theta,
        &statrs_gumbel,
        &REAL_REFERENCE_POINTS,
        ContinuousReferenceTolerances {
            cdf_abs: 1.0e-14,
            density_rel: 0.0,
            density_abs: 1.0e-14,
            quantile_abs: 1.0e-12,
        },
    );

    let laplace = LaplaceMuSigma::new();
    let laplace_theta = LaplaceTheta {
        mu: 0.4,
        sigma: 0.8,
    };
    let statrs_laplace = StatrsLaplace::new(laplace_theta.mu, laplace_theta.sigma).unwrap();
    assert_continuous_statrs_reference(
        &laplace,
        laplace_theta,
        &statrs_laplace,
        &REAL_REFERENCE_POINTS,
        ContinuousReferenceTolerances {
            cdf_abs: 1.0e-14,
            density_rel: 0.0,
            density_abs: 1.0e-14,
            quantile_abs: 1.0e-12,
        },
    );

    let log_normal = LogNormalLogLocationLogSd::new();
    let log_normal_theta = LogNormalTheta {
        log_location: 0.2,
        log_sd: 0.7,
    };
    let statrs_log_normal =
        StatrsLogNormal::new(log_normal_theta.log_location, log_normal_theta.log_sd).unwrap();
    assert_continuous_statrs_reference(
        &log_normal,
        log_normal_theta,
        &statrs_log_normal,
        &POSITIVE_REFERENCE_POINTS,
        ContinuousReferenceTolerances {
            cdf_abs: 2.0e-7,
            density_rel: 1.0e-10,
            density_abs: 1.0e-10,
            quantile_abs: 5.0e-7,
        },
    );

    let normal = NormalMuSigma::new();
    let normal_theta = NormalTheta {
        mu: -0.2,
        sigma: 1.1,
    };
    let statrs_normal = StatrsNormal::new(normal_theta.mu, normal_theta.sigma).unwrap();
    assert_continuous_statrs_reference(
        &normal,
        normal_theta,
        &statrs_normal,
        &REAL_REFERENCE_POINTS,
        ContinuousReferenceTolerances {
            cdf_abs: 2.0e-7,
            density_rel: 1.0e-10,
            density_abs: 1.0e-10,
            quantile_abs: 5.0e-7,
        },
    );

    let poisson = PoissonMean::new();
    let poisson_theta = PoissonTheta { mu: 4.0 };
    let statrs_poisson = StatrsPoisson::new(poisson_theta.mu).unwrap();
    assert_discrete_statrs_reference(&poisson, poisson_theta, &statrs_poisson, 0_u64..12, 1.0e-12);

    let negative_binomial = NegativeBinomialMeanSize::new();
    let negative_binomial_theta = NegativeBinomialTheta {
        mu: 5.0,
        shape: 2.5,
    };
    let statrs_negative_binomial = StatrsNegativeBinomial::new(
        negative_binomial_theta.shape,
        nb_success_probability(negative_binomial_theta),
    )
    .unwrap();
    assert_discrete_statrs_reference(
        &negative_binomial,
        negative_binomial_theta,
        &statrs_negative_binomial,
        0_u64..16,
        1.0e-12,
    );

    let student_t = StudentTMuSigma::default();
    let student_t_theta = StudentTTheta {
        mu: 0.2,
        sigma: 1.3,
    };
    let statrs_student_t =
        StatrsStudentsT::new(student_t_theta.mu, student_t_theta.sigma, 5.0).unwrap();
    assert_continuous_statrs_reference(
        &student_t,
        student_t_theta,
        &statrs_student_t,
        &REAL_REFERENCE_POINTS,
        ContinuousReferenceTolerances {
            cdf_abs: 2.0e-10,
            density_rel: 1.0e-10,
            density_abs: 1.0e-10,
            quantile_abs: 3.0e-7,
        },
    );

    let weibull = WeibullScaleShape::new();
    let weibull_theta = WeibullTheta {
        shape: 1.7,
        scale: 1.2,
    };
    let statrs_weibull = StatrsWeibull::new(weibull_theta.shape, weibull_theta.scale).unwrap();
    assert_continuous_statrs_reference(
        &weibull,
        weibull_theta,
        &statrs_weibull,
        &POSITIVE_REFERENCE_POINTS,
        ContinuousReferenceTolerances {
            cdf_abs: 1.0e-14,
            density_rel: 1.0e-12,
            density_abs: 1.0e-12,
            quantile_abs: 1.0e-12,
        },
    );
}

#[test]
fn tail_cdf_and_quantile_match_statrs_references() {
    let normal = NormalMuSigma::new();
    let normal_theta = NormalTheta {
        mu: -0.2,
        sigma: 1.1,
    };
    let statrs_normal = StatrsNormal::new(normal_theta.mu, normal_theta.sigma).unwrap();
    for p in TAIL_PROBABILITIES {
        let actual = normal.quantile(p, &normal_theta);
        let expected = statrs_normal.inverse_cdf(p);
        assert_close(actual, expected, 0.0, 2.0e-8);
        assert_close(normal.cdf(actual, &normal_theta), p, 0.0, 8.0e-8);
    }

    let log_normal = LogNormalLogLocationLogSd::new();
    let log_normal_theta = LogNormalTheta {
        log_location: 0.2,
        log_sd: 0.7,
    };
    let statrs_log_normal =
        StatrsLogNormal::new(log_normal_theta.log_location, log_normal_theta.log_sd).unwrap();
    for p in TAIL_PROBABILITIES {
        let actual = log_normal.quantile(p, &log_normal_theta);
        let expected = statrs_log_normal.inverse_cdf(p);
        assert_close(actual, expected, 2.0e-8, 1.0e-8);
        assert_close(log_normal.cdf(actual, &log_normal_theta), p, 0.0, 8.0e-8);
    }

    let student_t = StudentTMuSigma::try_new(5.0).unwrap();
    let student_t_theta = StudentTTheta {
        mu: 0.2,
        sigma: 1.3,
    };
    let statrs_student_t =
        StatrsStudentsT::new(student_t_theta.mu, student_t_theta.sigma, 5.0).unwrap();
    for p in TAIL_PROBABILITIES {
        let actual = student_t.quantile(p, &student_t_theta);
        let expected = statrs_student_t.inverse_cdf(p);
        assert_close(actual, expected, 3.0e-5, 2.0e-5);
        assert_close(student_t.cdf(actual, &student_t_theta), p, 0.0, 2.0e-7);
    }
}

#[test]
fn gamma_and_beta_extreme_shape_cases_match_statrs_references() {
    let gamma = GammaShapeRate::new();
    for theta in [
        GammaTheta {
            shape: 0.15,
            rate: 2.0,
        },
        GammaTheta {
            shape: 75.0,
            rate: 3.0,
        },
    ] {
        let reference = StatrsGamma::new(theta.shape, theta.rate).unwrap();
        for y in [0.01_f64, 0.1, 1.0, 10.0, 40.0] {
            assert_close(gamma.cdf(y, &theta), reference.cdf(y), 0.0, 2.0e-9);
        }
        for p in [0.01_f64, 0.1, 0.5, 0.9, 0.99] {
            let actual = gamma.quantile(p, &theta);
            let expected = reference.inverse_cdf(p);
            if expected.is_finite() {
                assert_close(actual, expected, 2.0e-8, 2.0e-8);
            } else {
                assert!(
                    actual.is_finite(),
                    "gamma quantile({p}) returned {actual:?}"
                );
                assert_close(gamma.cdf(actual, &theta), p, 0.0, 2.0e-8);
            }
        }
    }

    let beta = BetaMeanPrecision::new();
    for theta in [
        BetaTheta {
            mu: 0.2,
            precision: 0.75,
        },
        BetaTheta {
            mu: 0.45,
            precision: 200.0,
        },
    ] {
        let reference = StatrsBeta::new(
            theta.mu * theta.precision,
            (1.0 - theta.mu) * theta.precision,
        )
        .unwrap();
        for y in [1.0e-8_f64, 0.01, 0.1, 0.5, 0.9, 1.0 - 1.0e-8] {
            assert_close(beta.cdf(y, &theta), reference.cdf(y), 0.0, 2.0e-9);
        }
        for p in [0.01_f64, 0.1, 0.5, 0.9, 0.99] {
            let actual = beta.quantile(p, &theta);
            let expected = reference.inverse_cdf(p);
            if expected.is_finite() {
                assert_close(actual, expected, 2.0e-8, 2.0e-8);
            } else {
                assert!(actual.is_finite(), "beta quantile({p}) returned {actual:?}");
                assert_close(beta.cdf(actual, &theta), p, 0.0, 2.0e-8);
            }
        }
    }
}

#[test]
fn large_discrete_cdf_and_quantile_cases_match_statrs_references() {
    let poisson = PoissonMean::new();
    for theta in [PoissonTheta { mu: 80.0 }, PoissonTheta { mu: 250.0 }] {
        let reference = StatrsPoisson::new(theta.mu).unwrap();
        for count in [
            (theta.mu - 3.0 * theta.mu.sqrt()).floor().max(0.0) as u64,
            theta.mu.floor() as u64,
            (theta.mu + 3.0 * theta.mu.sqrt()).ceil() as u64,
        ] {
            assert_close(
                poisson.cdf(count as f64, &theta),
                reference.cdf(count),
                0.0,
                2.0e-11,
            );
        }
        for p in [0.001_f64, 0.01, 0.5, 0.99, 0.999] {
            assert_eq!(
                poisson.quantile(p, &theta),
                statrs_discrete_quantile(p, |count| reference.cdf(count)) as f64
            );
        }
    }

    let negative_binomial = NegativeBinomialMeanSize::new();
    for theta in [
        NegativeBinomialTheta {
            mu: 80.0,
            shape: 3.5,
        },
        NegativeBinomialTheta {
            mu: 250.0,
            shape: 120.0,
        },
    ] {
        let reference =
            StatrsNegativeBinomial::new(theta.shape, nb_success_probability(theta)).unwrap();
        for count in [
            (0.5 * theta.mu).floor() as u64,
            theta.mu.floor() as u64,
            (1.5 * theta.mu).ceil() as u64,
        ] {
            assert_close(
                negative_binomial.cdf(count as f64, &theta),
                reference.cdf(count),
                0.0,
                3.0e-11,
            );
        }
        for p in [0.001_f64, 0.01, 0.5, 0.99, 0.999] {
            assert_eq!(
                negative_binomial.quantile(p, &theta),
                statrs_discrete_quantile(p, |count| reference.cdf(count)) as f64
            );
        }
    }
}

#[test]
fn skew_student_t_symmetric_case_matches_student_t_reference() {
    let skew_t = SkewStudentTMuSigmaNuTau::new();
    let theta = SkewStudentTTheta {
        mu: 0.3,
        sigma: 1.4,
        nu: 0.0,
        tau: 8.0,
    };
    let reference = StatrsStudentsT::new(theta.mu, theta.sigma, theta.tau).unwrap();

    for y in [-4.0_f64, -1.0, 0.3, 1.0, 5.0] {
        assert_close(skew_t.cdf(y, &theta), reference.cdf(y), 0.0, 2.0e-6);
    }
    for p in [0.01_f64, 0.1, 0.5, 0.9, 0.99] {
        assert_close(
            skew_t.quantile(p, &theta),
            reference.inverse_cdf(p),
            0.0,
            2.0e-5,
        );
    }
}
