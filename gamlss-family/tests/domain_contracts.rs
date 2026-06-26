#![allow(clippy::float_cmp)]
use gamlss_core::{Family, HasCdf, HasQuantile};
use gamlss_family::*;

#[test]
fn invalid_domains_and_boundaries_follow_public_contracts() {
    let normal = NormalMuSigma::new();
    assert!(
        normal
            .cdf(
                f64::NEG_INFINITY,
                NormalTheta {
                    mu: 0.0,
                    sigma: 1.0,
                },
            )
            .is_nan()
    );
    assert_eq!(
        normal.quantile(
            0.0,
            NormalTheta {
                mu: 0.0,
                sigma: 1.0,
            },
        ),
        f64::NEG_INFINITY
    );
    assert!(
        normal
            .nll(
                0.0,
                NormalTheta {
                    mu: 0.0,
                    sigma: 0.0,
                },
            )
            .is_infinite()
    );

    let beta = BetaMeanPrecision::new();
    let beta_theta = BetaTheta {
        mu: 0.4,
        precision: 3.0,
    };
    assert!(beta.nll(0.0, beta_theta).is_infinite());
    assert!(beta.nll(1.0, beta_theta).is_infinite());
    assert_eq!(beta.cdf(0.0, beta_theta), 0.0);
    assert_eq!(beta.cdf(1.0, beta_theta), 1.0);
    assert_eq!(beta.quantile(0.0, beta_theta), 0.0);
    assert_eq!(beta.quantile(1.0, beta_theta), 1.0);
    assert!(beta.quantile(f64::NAN, beta_theta).is_nan());

    let exponential = ExponentialRate::new();
    assert_eq!(
        exponential.cdf(-1.0, ExponentialRateTheta { rate: 1.0 }),
        0.0
    );
    assert_eq!(
        exponential.quantile(0.0, ExponentialRateTheta { rate: 1.0 }),
        0.0
    );
    assert!(
        exponential
            .quantile(1.0, ExponentialRateTheta { rate: 1.0 })
            .is_infinite()
    );
    assert!(
        exponential
            .nll(-1.0, ExponentialRateTheta { rate: 1.0 })
            .is_infinite()
    );

    let poisson = PoissonMean::new();
    assert!(poisson.nll(1.5, PoissonTheta { mu: 2.0 }).is_infinite());
    assert_eq!(poisson.cdf(-1.0, PoissonTheta { mu: 2.0 }), 0.0);
    assert_eq!(poisson.quantile(0.0, PoissonTheta { mu: 2.0 }), 0.0);
    assert!(
        poisson
            .quantile(f64::NAN, PoissonTheta { mu: 2.0 })
            .is_nan()
    );

    let bernoulli = BernoulliProbability::new();
    assert!(bernoulli.nll(0.5, BernoulliTheta { mu: 0.5 }).is_infinite());
    assert_eq!(bernoulli.cdf(-1.0, BernoulliTheta { mu: 0.5 }), 0.0);
    assert_eq!(bernoulli.quantile(0.0, BernoulliTheta { mu: 0.5 }), 0.0);
    assert_eq!(bernoulli.quantile(1.0, BernoulliTheta { mu: 0.5 }), 1.0);
    assert!(
        bernoulli
            .quantile(f64::NAN, BernoulliTheta { mu: 0.5 })
            .is_nan()
    );
}

#[test]
fn extended_family_invalid_domains_return_non_finite_likelihoods() {
    assert!(
        SkewStudentTMuSigmaNuTau::new()
            .nll(
                0.0,
                SkewStudentTTheta {
                    mu: 0.0,
                    sigma: 0.0,
                    nu: 0.0,
                    tau: 5.0,
                },
            )
            .is_infinite()
    );
    assert!(
        SkewNormalMeanSdNu::new()
            .nll(
                0.0,
                SkewNormalMeanSdTheta {
                    mean: 0.0,
                    sigma: 0.0,
                    nu: 0.0,
                },
            )
            .is_infinite()
    );
    assert!(
        SkewNormalMeanSdNu::new()
            .cdf(
                0.0,
                SkewNormalMeanSdTheta {
                    mean: f64::NAN,
                    sigma: 1.0,
                    nu: 0.0,
                },
            )
            .is_nan()
    );
    assert!(
        SkewStudentTMeanSdNuTau::new()
            .nll(
                0.0,
                SkewStudentTMeanSdTheta {
                    mean: 0.0,
                    sigma: 1.0,
                    nu: 0.0,
                    tau: 2.0,
                },
            )
            .is_infinite()
    );
    assert!(
        SkewStudentTMeanSdNuTau::new()
            .quantile(
                0.5,
                SkewStudentTMeanSdTheta {
                    mean: 0.0,
                    sigma: 1.0,
                    nu: f64::INFINITY,
                    tau: 5.0,
                },
            )
            .is_nan()
    );
    assert!(
        TweedieMeanDispersionPower::new()
            .nll(
                -1.0,
                TweedieTheta {
                    mean: 1.0,
                    dispersion: 1.0,
                    power: 1.5,
                },
            )
            .is_infinite()
    );
    assert!(
        TweedieMeanCvPower::new()
            .nll(
                1.0,
                TweedieMeanCvPowerTheta {
                    mean: 1.0,
                    cv: 0.0,
                    power: 1.5,
                },
            )
            .is_infinite()
    );
    assert!(
        TweedieMeanDispersionPower::new()
            .nll(
                1.0,
                TweedieTheta {
                    mean: 1.0,
                    dispersion: 1.0,
                    power: 2.0,
                },
            )
            .is_infinite()
    );
    assert!(
        StudentTMuSigmaTau::new()
            .nll(
                1.0,
                StudentTMuSigmaTauTheta {
                    mu: 0.0,
                    sigma: 1.0,
                    tau: 0.0,
                },
            )
            .is_infinite()
    );
    assert!(
        StudentTMuSigmaTau::new()
            .cdf(
                1.0,
                StudentTMuSigmaTauTheta {
                    mu: 0.0,
                    sigma: 0.0,
                    tau: 5.0,
                },
            )
            .is_nan()
    );
}

#[test]
fn theta_construction_from_eta_stays_inside_expected_domains() {
    assert_eq!(
        BernoulliProbability::new()
            .theta(BernoulliEta { mu: 0.0 })
            .mu,
        0.5
    );
    assert_eq!(
        BetaMeanPrecision::new()
            .theta(BetaEta {
                mu: 0.0,
                precision: 0.0,
            })
            .mu,
        0.5
    );
    assert_eq!(
        ExponentialRate::new()
            .theta(ExponentialRateEta { rate: 0.0 })
            .rate,
        1.0
    );
    assert_eq!(PoissonMean::new().theta(PoissonEta { mu: 0.0 }).mu, 1.0);
    assert_eq!(
        NormalMuSigma::new()
            .theta(NormalEta {
                mu: 0.0,
                sigma: 0.0,
            })
            .sigma,
        1.0
    );
    assert_eq!(
        GammaShapeRate::new()
            .theta(GammaEta {
                shape: 0.0,
                rate: 0.0,
            })
            .shape,
        1.0
    );
    assert_eq!(
        GumbelMuSigma::new()
            .theta(GumbelEta {
                mu: 0.0,
                sigma: 0.0,
            })
            .sigma,
        1.0
    );
    assert_eq!(
        InverseGaussianMuShape::new()
            .theta(InverseGaussianEta {
                mu: 0.0,
                shape: 0.0,
            })
            .shape,
        1.0
    );
    assert_eq!(
        LaplaceMuSigma::new()
            .theta(LaplaceEta {
                mu: 0.0,
                sigma: 0.0,
            })
            .sigma,
        1.0
    );
    assert_eq!(
        LogNormalLogLocationLogSd::new()
            .theta(LogNormalEta {
                log_location: 0.0,
                log_sd: 0.0,
            })
            .log_sd,
        1.0
    );
    assert_eq!(
        LogisticMuSigma::new()
            .theta(LogisticEta {
                mu: 0.0,
                sigma: 0.0,
            })
            .sigma,
        1.0
    );
    assert_eq!(
        LomaxShapeScale::new()
            .theta(LomaxEta {
                shape: 0.0,
                scale: 0.0,
            })
            .scale,
        1.0
    );
    assert_eq!(
        NegativeBinomialMeanSize::new()
            .theta(NegativeBinomialEta {
                mu: 0.0,
                shape: 0.0,
            })
            .shape,
        1.0
    );
    assert_eq!(
        StudentTMuSigma::default()
            .theta(StudentTEta {
                mu: 0.0,
                sigma: 0.0,
            })
            .sigma,
        1.0
    );
    assert_eq!(
        WeibullScaleShape::new()
            .theta(WeibullEta {
                shape: 0.0,
                scale: 0.0,
            })
            .scale,
        1.0
    );
}
