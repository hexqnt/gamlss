use gamlss_core::{Family, HasCdf, HasDensity, HasQuantile, ParameterParts};
use gamlss_family::{
    BernoulliEta, BernoulliProbability, BernoulliTheta, BetaEta, BetaMeanPrecision, BetaTheta,
    ExponentialRate, ExponentialRateEta, ExponentialRateTheta, GammaEta, GammaShapeRate,
    GammaTheta, GumbelEta, GumbelMuSigma, GumbelTheta, InverseGaussianEta, InverseGaussianMuShape,
    InverseGaussianTheta, LaplaceEta, LaplaceMuSigma, LaplaceTheta, LogNormalEta,
    LogNormalLogLocationLogSd, LogNormalTheta, LogisticEta, LogisticMuSigma, LogisticTheta,
    LomaxEta, LomaxShapeScale, LomaxTheta, NegativeBinomialEta, NegativeBinomialMeanSize,
    NegativeBinomialTheta, NormalEta, NormalMuSigma, NormalTheta, PoissonEta, PoissonMean,
    PoissonTheta, StudentTEta, StudentTMuSigma, StudentTTheta, WeibullEta, WeibullScaleShape,
    WeibullTheta,
};
use proptest::prelude::*;
use statrs::distribution::{
    Bernoulli as StatrsBernoulli, Beta as StatrsBeta, Continuous, ContinuousCDF, Discrete,
    DiscreteCDF, Exp as StatrsExp, Gamma as StatrsGamma, Gumbel as StatrsGumbel,
    Laplace as StatrsLaplace, LogNormal as StatrsLogNormal,
    NegativeBinomial as StatrsNegativeBinomial, Normal as StatrsNormal, Poisson as StatrsPoisson,
    StudentsT as StatrsStudentsT, Weibull as StatrsWeibull,
};

const CASES: u32 = 32;
const PROB_MIN: f64 = 1.0e-8;
const PROB_MAX: f64 = 1.0 - PROB_MIN;
const FD_REL_TOL: f64 = 2.0e-5;
const FD_ABS_TOL: f64 = 2.0e-5;
const REAL_REFERENCE_POINTS: [f64; 5] = [-3.0, -1.0, 0.0, 1.0, 3.0];
const POSITIVE_REFERENCE_POINTS: [f64; 5] = [0.1, 0.5, 1.0, 2.0, 4.0];
const REFERENCE_PROBABILITIES: [f64; 5] = [0.01, 0.1, 0.5, 0.9, 0.99];

struct ContinuousReferenceTolerances {
    cdf_abs: f64,
    density_rel: f64,
    density_abs: f64,
    quantile_abs: f64,
}

fn proptest_config() -> ProptestConfig {
    ProptestConfig {
        cases: CASES,
        ..ProptestConfig::default()
    }
}

fn assert_close(actual: f64, expected: f64, rel_tol: f64, abs_tol: f64) {
    let diff = (actual - expected).abs();
    let scale = actual.abs().max(expected.abs()).max(1.0);
    assert!(
        diff <= abs_tol.max(rel_tol * scale),
        "actual {actual:?} differs from expected {expected:?}; diff={diff:?}, rel_tol={rel_tol:?}, abs_tol={abs_tol:?}"
    );
}

fn assert_gradient_matches_finite_difference<F, const K: usize>(family: &F, y: f64, eta: [f64; K])
where
    F: for<'obs> Family<Observation<'obs> = f64>,
    F::Eta: Copy + ParameterParts<K>,
    F::NllGradientEta: ParameterParts<K>,
{
    let (_, gradient) = family.nll_and_gradient_eta(y, F::Eta::from_array(eta));

    for index in 0..K {
        let epsilon = f64::EPSILON.sqrt() * eta[index].abs().max(1.0);
        let mut plus = eta;
        plus[index] += epsilon;
        let mut minus = eta;
        minus[index] -= epsilon;

        let finite_difference = (family.nll_eta(y, F::Eta::from_array(plus))
            - family.nll_eta(y, F::Eta::from_array(minus)))
            / (2.0 * epsilon);
        let actual = gradient.part(index);

        assert!(
            actual.is_finite(),
            "gradient component {index} is not finite: {actual:?}"
        );
        assert!(
            finite_difference.is_finite(),
            "finite-difference component {index} is not finite: {finite_difference:?}"
        );
        assert_close(actual, finite_difference, FD_REL_TOL, FD_ABS_TOL);
    }
}

fn assert_continuous_inverse<F>(family: &F, p: f64, theta: F::Theta, tolerance: f64)
where
    F: for<'obs> Family<Observation<'obs> = f64> + HasCdf + HasQuantile,
    F::Theta: Copy,
{
    let y = family.quantile(p, theta);
    assert!(y.is_finite(), "quantile({p}) returned {y:?}");
    assert_close(family.cdf(y, theta), p, 0.0, tolerance);
}

fn assert_discrete_inverse<F>(family: &F, p: f64, theta: F::Theta)
where
    F: for<'obs> Family<Observation<'obs> = f64> + HasCdf + HasQuantile,
    F::Theta: Copy,
{
    let q = family.quantile(p, theta);
    assert!(q.is_finite(), "quantile({p}) returned {q:?}");
    assert_eq!(q.fract(), 0.0, "discrete quantile should be integral");

    let cdf_at_q = family.cdf(q, theta);
    assert!(
        cdf_at_q + 1.0e-14 >= p,
        "cdf(q) must be at least p; p={p:?}, q={q:?}, cdf={cdf_at_q:?}"
    );
    if q > 0.0 {
        let cdf_below_q = family.cdf(q - 1.0, theta);
        assert!(
            cdf_below_q < p + 1.0e-14,
            "cdf(q - 1) must be below p; p={p:?}, q={q:?}, cdf={cdf_below_q:?}"
        );
    }
}

fn assert_cdf_monotone<F>(family: &F, theta: F::Theta, ys: &[f64])
where
    F: for<'obs> Family<Observation<'obs> = f64> + HasCdf,
    F::Theta: Copy,
{
    let mut previous = f64::NEG_INFINITY;
    for &y in ys {
        let cdf = family.cdf(y, theta);
        assert!(cdf.is_finite(), "cdf({y}) returned {cdf:?}");
        assert!(
            cdf + 1.0e-14 >= previous,
            "cdf is not monotone at y={y:?}: {cdf:?} < {previous:?}"
        );
        previous = cdf;
    }
}

fn integrate_simpson<F>(lower: f64, upper: f64, intervals: usize, mut f: F) -> f64
where
    F: FnMut(f64) -> f64,
{
    assert_eq!(intervals % 2, 0);
    let step = (upper - lower) / intervals as f64;
    let mut sum = f(lower) + f(upper);
    for index in 1..intervals {
        let weight = if index % 2 == 0 { 2.0 } else { 4.0 };
        sum += weight * f(lower + index as f64 * step);
    }
    sum * step / 3.0
}

fn assert_density_integrates_over_quantile_bracket<F>(
    family: &F,
    theta: F::Theta,
    tail_probability: f64,
    tolerance: f64,
) where
    F: for<'obs> Family<Observation<'obs> = f64> + HasCdf + HasQuantile + HasDensity,
    F::Theta: Copy,
{
    let lower = family.quantile(tail_probability, theta);
    let upper = family.quantile(1.0 - tail_probability, theta);
    assert!(
        lower.is_finite() && upper.is_finite() && lower < upper,
        "invalid integration bracket [{lower:?}, {upper:?}]"
    );

    let integral = integrate_simpson(lower, upper, 1024, |y| family.density(y, theta));
    let expected = family.cdf(upper, theta) - family.cdf(lower, theta);
    assert_close(integral, expected, 0.0, tolerance);
    assert_close(integral, 1.0, 0.0, tolerance + 2.0 * tail_probability);
}

fn assert_discrete_mass_sums_to_one<F>(family: &F, theta: F::Theta, upper_p: f64, tolerance: f64)
where
    F: for<'obs> Family<Observation<'obs> = f64> + HasQuantile + HasDensity,
    F::Theta: Copy,
{
    let upper = family.quantile(upper_p, theta);
    assert!(
        upper.is_finite() && upper >= 0.0,
        "invalid discrete upper quantile {upper:?}"
    );
    let upper = upper as u64;
    let mass = (0..=upper)
        .map(|count| family.density(count as f64, theta))
        .sum::<f64>();
    assert_close(mass, upper_p, 0.0, tolerance + (1.0 - upper_p));
    assert_close(mass, 1.0, 0.0, tolerance + (1.0 - upper_p));
}

fn nb_success_probability(theta: NegativeBinomialTheta) -> f64 {
    theta.shape / (theta.shape + theta.mu)
}

fn assert_continuous_statrs_reference<F, R>(
    family: &F,
    theta: F::Theta,
    reference: &R,
    ys: &[f64],
    tolerances: ContinuousReferenceTolerances,
) where
    F: for<'obs> Family<Observation<'obs> = f64> + HasCdf + HasDensity + HasQuantile,
    F::Theta: Copy,
    R: Continuous<f64, f64> + ContinuousCDF<f64, f64>,
{
    for &y in ys {
        assert_close(
            family.cdf(y, theta),
            reference.cdf(y),
            0.0,
            tolerances.cdf_abs,
        );
        assert_close(
            family.density(y, theta),
            reference.pdf(y),
            tolerances.density_rel,
            tolerances.density_abs,
        );
    }

    for p in REFERENCE_PROBABILITIES {
        assert_close(
            family.quantile(p, theta),
            reference.inverse_cdf(p),
            0.0,
            tolerances.quantile_abs,
        );
    }
}

fn assert_discrete_statrs_reference<F, R>(
    family: &F,
    theta: F::Theta,
    reference: &R,
    counts: impl Iterator<Item = u64>,
    tolerance: f64,
) where
    F: for<'obs> Family<Observation<'obs> = f64> + HasCdf + HasDensity,
    F::Theta: Copy,
    R: Discrete<u64, f64> + DiscreteCDF<u64, f64>,
{
    for count in counts {
        assert_close(
            family.cdf(count as f64, theta),
            reference.cdf(count),
            0.0,
            tolerance,
        );
        assert_close(
            family.density(count as f64, theta),
            reference.pmf(count),
            tolerance,
            tolerance,
        );
    }
}

proptest! {
    #![proptest_config(proptest_config())]

    #[test]
    fn finite_difference_gradients_match_for_continuous_families(
        y_real in -5.0_f64..5.0,
        y_positive in 0.01_f64..20.0,
        y_unit in 0.001_f64..0.999,
        eta1 in -3.0_f64..3.0,
        eta2 in -3.0_f64..3.0,
    ) {
        assert_gradient_matches_finite_difference::<_, 2>(&NormalMuSigma::new(), y_real, [eta1, eta2]);
        assert_gradient_matches_finite_difference::<_, 2>(&GumbelMuSigma::new(), y_real, [eta1, eta2]);
        assert_gradient_matches_finite_difference::<_, 2>(&LaplaceMuSigma::new(), y_real, [eta1, eta2]);
        assert_gradient_matches_finite_difference::<_, 2>(&LogisticMuSigma::new(), y_real, [eta1, eta2]);
        assert_gradient_matches_finite_difference::<_, 2>(&StudentTMuSigma::default(), y_real, [eta1, eta2]);

        assert_gradient_matches_finite_difference::<_, 1>(&ExponentialRate::new(), y_positive, [eta1]);
        assert_gradient_matches_finite_difference::<_, 2>(&GammaShapeRate::new(), y_positive, [eta1, eta2]);
        assert_gradient_matches_finite_difference::<_, 2>(&InverseGaussianMuShape::new(), y_positive, [eta1, eta2]);
        assert_gradient_matches_finite_difference::<_, 2>(&LogNormalLogLocationLogSd::new(), y_positive, [eta1, eta2]);
        assert_gradient_matches_finite_difference::<_, 2>(&LomaxShapeScale::new(), y_positive, [eta1, eta2]);
        assert_gradient_matches_finite_difference::<_, 2>(&WeibullScaleShape::new(), y_positive, [eta1, eta2]);

        assert_gradient_matches_finite_difference::<_, 2>(&BetaMeanPrecision::new(), y_unit, [eta1, eta2]);
    }

    #[test]
    fn finite_difference_gradients_match_for_discrete_families(
        count in 0_u32..40,
        binary in 0_u32..2,
        eta1 in -3.0_f64..3.0,
        eta2 in -3.0_f64..3.0,
    ) {
        assert_gradient_matches_finite_difference::<_, 1>(&BernoulliProbability::new(), f64::from(binary), [eta1]);
        assert_gradient_matches_finite_difference::<_, 1>(&PoissonMean::new(), f64::from(count), [eta1]);
        assert_gradient_matches_finite_difference::<_, 2>(&NegativeBinomialMeanSize::new(), f64::from(count), [eta1, eta2]);
    }

    #[test]
    fn continuous_quantiles_invert_cdfs(
        p in PROB_MIN..PROB_MAX,
        location in -2.0_f64..2.0,
        log_scale in -2.0_f64..2.0,
        log_shape in -1.5_f64..2.0,
        mu_unit in 0.05_f64..0.95,
    ) {
        let scale = log_scale.exp();
        let shape = log_shape.exp();

        assert_continuous_inverse(&NormalMuSigma::new(), p, NormalTheta { mu: location, sigma: scale }, 2.0e-7);
        assert_continuous_inverse(&GumbelMuSigma::new(), p, GumbelTheta { mu: location, sigma: scale }, 2.0e-10);
        assert_continuous_inverse(&LaplaceMuSigma::new(), p, LaplaceTheta { mu: location, sigma: scale }, 2.0e-10);
        assert_continuous_inverse(&LogisticMuSigma::new(), p, LogisticTheta { mu: location, sigma: scale }, 2.0e-10);
        assert_continuous_inverse(&StudentTMuSigma::default(), p, StudentTTheta { mu: location, sigma: scale }, 2.0e-7);

        assert_continuous_inverse(&ExponentialRate::new(), p, ExponentialRateTheta { rate: shape }, 2.0e-10);
        assert_continuous_inverse(&GammaShapeRate::new(), p, GammaTheta { shape, rate: scale }, 2.0e-7);
        assert_continuous_inverse(&InverseGaussianMuShape::new(), p, InverseGaussianTheta { mu: scale, shape }, 2.0e-7);
        assert_continuous_inverse(&LogNormalLogLocationLogSd::new(), p, LogNormalTheta { log_location: location, log_sd: scale }, 2.0e-7);
        assert_continuous_inverse(&LomaxShapeScale::new(), p, LomaxTheta { shape, scale }, 2.0e-10);
        assert_continuous_inverse(&WeibullScaleShape::new(), p, WeibullTheta { shape, scale }, 2.0e-10);

        assert_continuous_inverse(&BetaMeanPrecision::new(), p, BetaTheta { mu: mu_unit, precision: shape + 2.0 }, 2.0e-7);
    }

    #[test]
    fn discrete_quantiles_are_generalized_inverses(
        p in PROB_MIN..PROB_MAX,
        mu in 0.05_f64..20.0,
        shape in 0.2_f64..20.0,
        bernoulli_mu in 0.01_f64..0.99,
    ) {
        assert_discrete_inverse(&BernoulliProbability::new(), p, BernoulliTheta { mu: bernoulli_mu });
        assert_discrete_inverse(&PoissonMean::new(), p, PoissonTheta { mu });
        assert_discrete_inverse(&NegativeBinomialMeanSize::new(), p, NegativeBinomialTheta { mu, shape });
    }
}

#[test]
fn cdfs_are_monotone_on_representative_grids() {
    assert_cdf_monotone(
        &NormalMuSigma::new(),
        NormalTheta {
            mu: 0.3,
            sigma: 1.2,
        },
        &[-4.0, -1.0, 0.0, 1.0, 4.0],
    );
    assert_cdf_monotone(
        &BetaMeanPrecision::new(),
        BetaTheta {
            mu: 0.4,
            precision: 5.0,
        },
        &[0.0, 0.05, 0.4, 0.9, 1.0],
    );
    assert_cdf_monotone(
        &PoissonMean::new(),
        PoissonTheta { mu: 3.5 },
        &[-1.0, 0.0, 1.0, 3.0, 10.0],
    );
    assert_cdf_monotone(
        &NegativeBinomialMeanSize::new(),
        NegativeBinomialTheta {
            mu: 4.0,
            shape: 2.0,
        },
        &[-1.0, 0.0, 1.0, 3.0, 10.0],
    );
}

#[test]
fn continuous_density_integrates_approximately_to_one() {
    assert_density_integrates_over_quantile_bracket(
        &NormalMuSigma::new(),
        NormalTheta {
            mu: 0.2,
            sigma: 1.1,
        },
        1.0e-4,
        2.0e-5,
    );
    assert_density_integrates_over_quantile_bracket(
        &BetaMeanPrecision::new(),
        BetaTheta {
            mu: 0.5,
            precision: 6.0,
        },
        1.0e-4,
        5.0e-5,
    );
    assert_density_integrates_over_quantile_bracket(
        &ExponentialRate::new(),
        ExponentialRateTheta { rate: 1.4 },
        1.0e-4,
        2.0e-5,
    );
    assert_density_integrates_over_quantile_bracket(
        &GammaShapeRate::new(),
        GammaTheta {
            shape: 2.4,
            rate: 1.3,
        },
        1.0e-4,
        7.0e-5,
    );
    assert_density_integrates_over_quantile_bracket(
        &GumbelMuSigma::new(),
        GumbelTheta {
            mu: -0.2,
            sigma: 1.1,
        },
        1.0e-4,
        2.0e-5,
    );
    assert_density_integrates_over_quantile_bracket(
        &InverseGaussianMuShape::new(),
        InverseGaussianTheta {
            mu: 1.3,
            shape: 2.0,
        },
        1.0e-4,
        3.0e-4,
    );
    assert_density_integrates_over_quantile_bracket(
        &LaplaceMuSigma::new(),
        LaplaceTheta {
            mu: 0.2,
            sigma: 0.9,
        },
        1.0e-4,
        2.0e-5,
    );
    assert_density_integrates_over_quantile_bracket(
        &LogNormalLogLocationLogSd::new(),
        LogNormalTheta {
            log_location: 0.1,
            log_sd: 0.7,
        },
        1.0e-4,
        8.0e-5,
    );
    assert_density_integrates_over_quantile_bracket(
        &LogisticMuSigma::new(),
        LogisticTheta {
            mu: -0.3,
            sigma: 1.4,
        },
        1.0e-4,
        2.0e-5,
    );
    assert_density_integrates_over_quantile_bracket(
        &LomaxShapeScale::new(),
        LomaxTheta {
            shape: 2.2,
            scale: 1.1,
        },
        1.0e-4,
        1.0e-4,
    );
    assert_density_integrates_over_quantile_bracket(
        &StudentTMuSigma::default(),
        StudentTTheta {
            mu: 0.0,
            sigma: 1.2,
        },
        1.0e-4,
        1.5e-4,
    );
    assert_density_integrates_over_quantile_bracket(
        &WeibullScaleShape::new(),
        WeibullTheta {
            shape: 2.0,
            scale: 1.4,
        },
        1.0e-4,
        5.0e-5,
    );
}

#[test]
fn discrete_mass_sums_approximately_to_one() {
    assert_discrete_mass_sums_to_one(
        &BernoulliProbability::new(),
        BernoulliTheta { mu: 0.35 },
        1.0,
        1.0e-14,
    );
    assert_discrete_mass_sums_to_one(
        &PoissonMean::new(),
        PoissonTheta { mu: 4.0 },
        1.0 - 1.0e-10,
        2.0e-12,
    );
    assert_discrete_mass_sums_to_one(
        &NegativeBinomialMeanSize::new(),
        NegativeBinomialTheta {
            mu: 5.0,
            shape: 2.5,
        },
        1.0 - 1.0e-10,
        2.0e-12,
    );
}

#[test]
fn cdf_quantile_and_density_match_statrs_references() {
    let bernoulli = BernoulliProbability::new();
    let bernoulli_theta = BernoulliTheta { mu: 0.35 };
    let statrs_bernoulli = StatrsBernoulli::new(bernoulli_theta.mu).unwrap();
    for y in [0.0, 1.0] {
        assert_close(
            bernoulli.density(y, bernoulli_theta),
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
        assert_close(bernoulli.cdf(y, bernoulli_theta), expected, 0.0, 1.0e-14);
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
