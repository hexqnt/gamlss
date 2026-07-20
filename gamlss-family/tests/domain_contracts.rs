#![allow(clippy::float_cmp)]
use gamlss_core::{Family, HasCdf, HasQuantile};
use gamlss_family::*;

fn nll<F>(family: &F, y: f64, theta: &F::Theta) -> f64
where
    F: for<'obs> Family<Observation<'obs> = f64>,
{
    let mut workspace = family.workspace();
    family.nll(y, theta, &mut workspace)
}

#[allow(clippy::needless_pass_by_value)]
fn theta<F>(family: &F, eta: F::Eta) -> F::Theta
where
    F: Family,
{
    let mut workspace = family.workspace();
    family.theta(&eta, &mut workspace)
}

#[test]
fn invalid_domains_and_boundaries_follow_public_contracts() {
    let normal = NormalMuSigma::new();
    assert!(
        normal
            .cdf(
                f64::NEG_INFINITY,
                &NormalTheta {
                    mu: 0.0,
                    sigma: 1.0,
                },
            )
            .is_nan()
    );
    assert_eq!(
        normal.quantile(
            0.0,
            &NormalTheta {
                mu: 0.0,
                sigma: 1.0,
            },
        ),
        f64::NEG_INFINITY
    );
    assert!(
        nll(
            &normal,
            0.0,
            &NormalTheta {
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
    assert!(nll(&beta, 0.0, &beta_theta).is_infinite());
    assert!(nll(&beta, 1.0, &beta_theta).is_infinite());
    assert_eq!(beta.cdf(0.0, &beta_theta), 0.0);
    assert_eq!(beta.cdf(1.0, &beta_theta), 1.0);
    assert_eq!(beta.quantile(0.0, &beta_theta), 0.0);
    assert_eq!(beta.quantile(1.0, &beta_theta), 1.0);
    assert!(beta.quantile(f64::NAN, &beta_theta).is_nan());

    let exponential = ExponentialRate::new();
    assert_eq!(
        exponential.cdf(-1.0, &ExponentialRateTheta { rate: 1.0 }),
        0.0
    );
    assert_eq!(
        exponential.quantile(0.0, &ExponentialRateTheta { rate: 1.0 }),
        0.0
    );
    assert!(
        exponential
            .quantile(1.0, &ExponentialRateTheta { rate: 1.0 })
            .is_infinite()
    );
    assert!(nll(&exponential, -1.0, &ExponentialRateTheta { rate: 1.0 }).is_infinite());

    let poisson = PoissonMean::new();
    assert!(nll(&poisson, 1.5, &PoissonTheta { mu: 2.0 }).is_infinite());
    assert_eq!(poisson.cdf(-1.0, &PoissonTheta { mu: 2.0 }), 0.0);
    assert_eq!(poisson.quantile(0.0, &PoissonTheta { mu: 2.0 }), 0.0);
    assert!(
        poisson
            .quantile(f64::NAN, &PoissonTheta { mu: 2.0 })
            .is_nan()
    );

    let bernoulli = BernoulliProbability::new();
    assert!(nll(&bernoulli, 0.5, &BernoulliTheta { mu: 0.5 }).is_infinite());
    assert_eq!(bernoulli.cdf(-1.0, &BernoulliTheta { mu: 0.5 }), 0.0);
    assert_eq!(bernoulli.quantile(0.0, &BernoulliTheta { mu: 0.5 }), 0.0);
    assert_eq!(bernoulli.quantile(1.0, &BernoulliTheta { mu: 0.5 }), 1.0);
    assert!(
        bernoulli
            .quantile(f64::NAN, &BernoulliTheta { mu: 0.5 })
            .is_nan()
    );
}

#[test]
fn extended_family_invalid_domains_return_non_finite_likelihoods() {
    assert!(
        nll(
            &SkewStudentTMuSigmaNuTau::new(),
            0.0,
            &SkewStudentTTheta {
                mu: 0.0,
                sigma: 0.0,
                nu: 0.0,
                tau: 5.0,
            },
        )
        .is_infinite()
    );
    assert!(
        nll(
            &SkewNormalMeanSdNu::new(),
            0.0,
            &SkewNormalMeanSdTheta {
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
                &SkewNormalMeanSdTheta {
                    mean: f64::NAN,
                    sigma: 1.0,
                    nu: 0.0,
                },
            )
            .is_nan()
    );
    assert!(
        nll(
            &SkewStudentTMeanSdNuTau::new(),
            0.0,
            &SkewStudentTMeanSdTheta {
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
                &SkewStudentTMeanSdTheta {
                    mean: 0.0,
                    sigma: 1.0,
                    nu: f64::INFINITY,
                    tau: 5.0,
                },
            )
            .is_nan()
    );
    assert!(
        nll(
            &TweedieMeanDispersionPower::new(),
            -1.0,
            &TweedieTheta {
                mean: 1.0,
                dispersion: 1.0,
                power: 1.5,
            },
        )
        .is_infinite()
    );
    assert!(
        nll(
            &TweedieMeanCvPower::new(),
            1.0,
            &TweedieMeanCvPowerTheta {
                mean: 1.0,
                cv: 0.0,
                power: 1.5,
            },
        )
        .is_infinite()
    );
    assert!(
        nll(
            &TweedieMeanDispersionPower::new(),
            1.0,
            &TweedieTheta {
                mean: 1.0,
                dispersion: 1.0,
                power: 2.0,
            },
        )
        .is_infinite()
    );
    assert!(
        nll(
            &StudentTMuSigmaTau::new(),
            1.0,
            &StudentTMuSigmaTauTheta {
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
                &StudentTMuSigmaTauTheta {
                    mu: 0.0,
                    sigma: 0.0,
                    tau: 5.0,
                },
            )
            .is_nan()
    );
}

#[test]
fn new_univariate_family_boundaries_follow_public_contracts() {
    let geometric = GeometricMean::new();
    let geometric_theta = GeometricTheta { mean: 2.0 };
    assert!(nll(&geometric, 0.5, &geometric_theta).is_infinite());
    assert_eq!(geometric.cdf(-1.0, &geometric_theta), 0.0);
    assert!(geometric.quantile(1.0, &geometric_theta).is_infinite());

    let rayleigh = RayleighScale::new();
    let rayleigh_theta = RayleighTheta { scale: 1.0 };
    assert!(nll(&rayleigh, 0.0, &rayleigh_theta).is_infinite());
    assert_eq!(rayleigh.cdf(0.0, &rayleigh_theta), 0.0);
    assert!(rayleigh.quantile(1.0, &rayleigh_theta).is_infinite());

    let log_logistic = LogLogisticScaleShape::new();
    let log_logistic_theta = LogLogisticTheta {
        scale: 1.0,
        shape: 2.0,
    };
    assert!(nll(&log_logistic, 0.0, &log_logistic_theta).is_infinite());
    assert_eq!(log_logistic.cdf(0.0, &log_logistic_theta), 0.0);
    assert_eq!(log_logistic.quantile(0.0, &log_logistic_theta), 0.0);
    assert!(
        log_logistic
            .quantile(1.0, &log_logistic_theta)
            .is_infinite()
    );

    let chi = ChiDegreesOfFreedom::new();
    let chi_theta = ChiTheta {
        degrees_of_freedom: 3.0,
    };
    assert!(nll(&chi, 0.0, &chi_theta).is_infinite());
    assert_eq!(chi.cdf(0.0, &chi_theta), 0.0);
    assert!(
        chi.cdf(
            1.0,
            &ChiTheta {
                degrees_of_freedom: 0.0,
            },
        )
        .is_nan()
    );

    let chi_squared = ChiSquaredDegreesOfFreedom::new();
    let chi_squared_theta = ChiSquaredTheta {
        degrees_of_freedom: 3.0,
    };
    assert!(nll(&chi_squared, 0.0, &chi_squared_theta).is_infinite());
    assert_eq!(chi_squared.cdf(0.0, &chi_squared_theta), 0.0);

    let generalized_pareto = GeneralizedParetoScaleShape::new();
    let generalized_pareto_theta = GeneralizedParetoTheta {
        scale: 2.0,
        shape: -0.5,
    };
    assert!(nll(&generalized_pareto, 4.0, &generalized_pareto_theta).is_infinite());
    assert_eq!(generalized_pareto.cdf(4.0, &generalized_pareto_theta), 1.0);
    assert_eq!(
        generalized_pareto.quantile(1.0, &generalized_pareto_theta),
        4.0
    );

    let binomial = BinomialVaryingTrialsProbability::new();
    let binomial_theta = BinomialTheta { probability: 0.4 };
    assert!(
        binomial
            .nll([2.5, 10.0], &binomial_theta, &mut ())
            .is_infinite()
    );
    assert!(binomial.cdf([2.0, -1.0], &binomial_theta).is_nan());

    let beta_binomial = BetaBinomialMeanPrecision::new();
    let beta_binomial_theta = BetaBinomialTheta {
        probability: 0.4,
        precision: 5.0,
    };
    assert!(
        beta_binomial
            .nll([11.0, 10.0], &beta_binomial_theta, &mut ())
            .is_infinite()
    );

    let categorical = Categorical::<3>::new();
    let categorical_theta = CategoricalTheta {
        probabilities: [0.2, 0.3, 0.5],
    };
    assert!(
        categorical
            .nll(3.0, &categorical_theta, &mut ())
            .is_infinite()
    );
    assert!(categorical.quantile(f64::NAN, &categorical_theta).is_nan());
}

#[test]
fn theta_construction_from_eta_stays_inside_expected_domains() {
    assert_eq!(
        theta(&BernoulliProbability::new(), BernoulliEta { mu: 0.0 }).mu,
        0.5
    );
    assert_eq!(
        theta(
            &BetaMeanPrecision::new(),
            BetaEta {
                mu: 0.0,
                precision: 0.0,
            },
        )
        .mu,
        0.5
    );
    assert_eq!(
        theta(&ExponentialRate::new(), ExponentialRateEta { rate: 0.0 }).rate,
        1.0
    );
    assert_eq!(theta(&PoissonMean::new(), PoissonEta { mu: 0.0 }).mu, 1.0);
    assert_eq!(
        theta(
            &NormalMuSigma::new(),
            NormalEta {
                mu: 0.0,
                sigma: 0.0,
            },
        )
        .sigma,
        1.0
    );
    assert_eq!(
        theta(
            &GammaShapeRate::new(),
            GammaShapeRateEta {
                shape: 0.0,
                rate: 0.0,
            },
        )
        .shape,
        1.0
    );
    assert_eq!(
        theta(
            &GumbelMuSigma::new(),
            GumbelEta {
                mu: 0.0,
                sigma: 0.0,
            },
        )
        .sigma,
        1.0
    );
    assert_eq!(
        theta(
            &InverseGaussianMuShape::new(),
            InverseGaussianEta {
                mu: 0.0,
                shape: 0.0,
            },
        )
        .shape,
        1.0
    );
    assert_eq!(
        theta(
            &LaplaceMuSigma::new(),
            LaplaceEta {
                mu: 0.0,
                sigma: 0.0,
            },
        )
        .sigma,
        1.0
    );
    assert_eq!(
        theta(
            &LogNormalLogLocationLogSd::new(),
            LogNormalLogLocationLogSdEta {
                log_location: 0.0,
                log_sd: 0.0,
            },
        )
        .log_sd,
        1.0
    );
    assert_eq!(
        theta(
            &LogisticMuSigma::new(),
            LogisticEta {
                mu: 0.0,
                sigma: 0.0,
            },
        )
        .sigma,
        1.0
    );
    assert_eq!(
        theta(
            &LomaxShapeScale::new(),
            LomaxEta {
                shape: 0.0,
                scale: 0.0,
            },
        )
        .scale,
        1.0
    );
    assert_eq!(
        theta(
            &NegativeBinomialMeanSize::new(),
            NegativeBinomialEta {
                mu: 0.0,
                shape: 0.0,
            },
        )
        .shape,
        1.0
    );
    assert_eq!(
        theta(
            &StudentTMuSigma::default(),
            StudentTEta {
                mu: 0.0,
                sigma: 0.0,
            },
        )
        .sigma,
        1.0
    );
    assert_eq!(
        theta(
            &WeibullScaleShape::new(),
            WeibullScaleShapeEta {
                shape: 0.0,
                scale: 0.0,
            },
        )
        .scale,
        1.0
    );
}
