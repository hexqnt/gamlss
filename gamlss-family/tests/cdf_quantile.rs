#![allow(clippy::float_cmp)]
use gamlss_core::{Family, HasCdf, HasQuantile};
use gamlss_family::*;
use proptest::prelude::*;

use common::{
    PROB_MAX, PROB_MIN, TAIL_PROBABILITIES, assert_cdf_monotone, assert_close,
    assert_continuous_inverse, assert_density_integrates_over_quantile_bracket,
    assert_discrete_inverse, assert_discrete_mass_sums_to_one,
    assert_discrete_or_continuous_generalized_inverse, integrate_simpson, proptest_config,
};

#[path = "common/helpers.rs"]
mod common;

proptest! {
    #![proptest_config(proptest_config())]

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
        assert_continuous_inverse(&StudentTMuSigma::default(), p, StudentTTheta { mu: location, sigma: scale }, 2.0e-6);

        assert_continuous_inverse(&ExponentialRate::new(), p, ExponentialRateTheta { rate: shape }, 2.0e-10);
        assert_continuous_inverse(&RayleighScale::new(), p, RayleighTheta { scale }, 2.0e-10);
        assert_continuous_inverse(&ChiDegreesOfFreedom::new(), p, ChiTheta { degrees_of_freedom: shape }, 2.0e-7);
        assert_continuous_inverse(&ChiSquaredDegreesOfFreedom::new(), p, ChiSquaredTheta { degrees_of_freedom: shape }, 2.0e-7);
        assert_continuous_inverse(&GammaShapeRate::new(), p, GammaShapeRateTheta { shape, rate: scale }, 2.0e-7);
        assert_continuous_inverse(&InverseGaussianMuShape::new(), p, InverseGaussianTheta { mu: scale, shape }, 2.0e-7);
        assert_continuous_inverse(&LogLogisticScaleShape::new(), p, LogLogisticTheta { scale, shape }, 2.0e-10);
        assert_continuous_inverse(&LogNormalLogLocationLogSd::new(), p, LogNormalLogLocationLogSdTheta { log_location: location, log_sd: scale }, 2.0e-7);
        assert_continuous_inverse(&LomaxShapeScale::new(), p, LomaxTheta { shape, scale }, 2.0e-10);
        assert_continuous_inverse(&WeibullScaleShape::new(), p, WeibullScaleShapeTheta { shape, scale }, 2.0e-10);
        assert_continuous_inverse(&GeneralizedParetoScaleShape::new(), p, GeneralizedParetoTheta { scale, shape }, 2.0e-10);

        assert_continuous_inverse(&BetaMeanPrecision::new(), p, BetaTheta { mu: mu_unit, precision: shape + 2.0 }, 5.0e-6);
    }

    #[test]
    fn discrete_quantiles_are_generalized_inverses(
        p in PROB_MIN..PROB_MAX,
        mu in 0.05_f64..20.0,
        shape in 0.2_f64..20.0,
        bernoulli_mu in 0.01_f64..0.99,
    ) {
        assert_discrete_inverse(&BernoulliProbability::new(), p, BernoulliTheta { mu: bernoulli_mu });
        assert_discrete_inverse(&BinomialFixedTrialsProbability::try_new(40).unwrap(), p, BinomialTheta { probability: bernoulli_mu });
        assert_discrete_inverse(&GeometricMean::new(), p, GeometricTheta { mean: mu });
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
        GammaShapeRateTheta {
            shape: 2.4,
            rate: 1.3,
        },
        1.0e-4,
        7.0e-5,
    );
    assert_density_integrates_over_quantile_bracket(
        &RayleighScale::new(),
        RayleighTheta { scale: 1.3 },
        1.0e-4,
        2.0e-5,
    );
    assert_density_integrates_over_quantile_bracket(
        &LogLogisticScaleShape::new(),
        LogLogisticTheta {
            scale: 1.3,
            shape: 2.4,
        },
        1.0e-4,
        1.0e-4,
    );
    assert_density_integrates_over_quantile_bracket(
        &ChiDegreesOfFreedom::new(),
        ChiTheta {
            degrees_of_freedom: 3.7,
        },
        1.0e-4,
        7.0e-5,
    );
    assert_density_integrates_over_quantile_bracket(
        &ChiSquaredDegreesOfFreedom::new(),
        ChiSquaredTheta {
            degrees_of_freedom: 4.5,
        },
        1.0e-4,
        7.0e-5,
    );
    assert_density_integrates_over_quantile_bracket(
        &GeneralizedParetoScaleShape::new(),
        GeneralizedParetoTheta {
            scale: 1.3,
            shape: 0.2,
        },
        1.0e-4,
        2.0e-4,
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
        LogNormalLogLocationLogSdTheta {
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
        WeibullScaleShapeTheta {
            shape: 2.0,
            scale: 1.4,
        },
        1.0e-4,
        5.0e-5,
    );
}

#[test]
fn extended_continuous_densities_integrate_to_their_cdf_mass() {
    assert_density_integrates_over_quantile_bracket(
        &SkewNormalMuSigmaNu::new(),
        SkewNormalTheta {
            mu: 0.1,
            sigma: 1.2,
            nu: 0.7,
        },
        1.0e-4,
        2.0e-4,
    );
    assert_density_integrates_over_quantile_bracket(
        &PowerExponentialMuSigmaNu::new(),
        PowerExponentialTheta {
            mu: 0.1,
            sigma: 1.2,
            nu: 1.5,
        },
        1.0e-4,
        2.0e-4,
    );
    assert_density_integrates_over_quantile_bracket(
        &ShashMuSigmaNuTau::new(),
        ShashTheta {
            mu: 0.1,
            sigma: 1.2,
            nu: 0.7,
            tau: 1.2,
        },
        1.0e-4,
        3.0e-4,
    );
    assert_density_integrates_over_quantile_bracket(
        &JohnsonSuMuSigmaNuTau::new(),
        JohnsonSuTheta {
            mu: 0.1,
            sigma: 1.2,
            nu: 0.3,
            tau: 1.1,
        },
        1.0e-4,
        2.0e-4,
    );
    assert_density_integrates_over_quantile_bracket(
        &GeneralizedGammaScaleSigmaNu::new(),
        GeneralizedGammaTheta {
            scale: 1.2,
            sigma: 0.6,
            nu: -0.8,
        },
        1.0e-4,
        5.0e-4,
    );
    assert_density_integrates_over_quantile_bracket(
        &GevMuSigmaShape::new(),
        GevTheta {
            mu: 0.1,
            sigma: 1.2,
            nu: -0.2,
        },
        1.0e-4,
        3.0e-4,
    );
    assert_density_integrates_over_quantile_bracket(
        &SkewStudentTMuSigmaNuTau::new(),
        SkewStudentTTheta {
            mu: 0.1,
            sigma: 1.2,
            nu: 0.7,
            tau: 5.0,
        },
        1.0e-3,
        1.5e-3,
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
    assert_discrete_mass_sums_to_one(
        &GeometricMean::new(),
        GeometricTheta { mean: 3.0 },
        1.0 - 1.0e-10,
        2.0e-12,
    );
    assert_discrete_mass_sums_to_one(
        &BinomialFixedTrialsProbability::try_new(20).unwrap(),
        BinomialTheta { probability: 0.35 },
        1.0,
        2.0e-12,
    );
    assert_discrete_mass_sums_to_one(
        &ZipComponentMeanZeroProbability::new(),
        ZipComponentMeanZeroProbabilityTheta {
            component_mean: 4.0,
            zero_probability: 0.25,
        },
        1.0 - 1.0e-10,
        2.0e-12,
    );
    assert_discrete_mass_sums_to_one(
        &ZinbComponentMeanSizeZeroProbability::new(),
        ZinbComponentMeanSizeZeroProbabilityTheta {
            component_mean: 5.0,
            size: 2.5,
            zero_probability: 0.25,
        },
        1.0 - 1.0e-10,
        2.0e-12,
    );
}

#[test]
fn extended_continuous_family_quantiles_invert_cdfs() {
    assert_continuous_inverse(
        &SkewNormalMuSigmaNu::new(),
        0.4,
        SkewNormalTheta {
            mu: 0.1,
            sigma: 1.2,
            nu: 0.5,
        },
        3.0e-5,
    );
    assert_continuous_inverse(
        &PowerExponentialMuSigmaNu::new(),
        0.4,
        PowerExponentialTheta {
            mu: 0.1,
            sigma: 1.2,
            nu: 1.5,
        },
        2.0e-7,
    );
    assert_continuous_inverse(
        &ShashMuSigmaNuTau::new(),
        0.4,
        ShashTheta {
            mu: 0.1,
            sigma: 1.2,
            nu: 0.5,
            tau: 0.8,
        },
        2.0e-7,
    );
    assert_continuous_inverse(
        &JohnsonSuMuSigmaNuTau::new(),
        0.4,
        JohnsonSuTheta {
            mu: 0.1,
            sigma: 1.2,
            nu: 0.3,
            tau: 1.1,
        },
        2.0e-7,
    );
    assert_continuous_inverse(
        &GeneralizedGammaScaleSigmaNu::new(),
        0.4,
        GeneralizedGammaTheta {
            scale: 1.2,
            sigma: 0.6,
            nu: 0.8,
        },
        2.0e-7,
    );
    assert_continuous_inverse(
        &GevMuSigmaShape::new(),
        0.4,
        GevTheta {
            mu: 0.1,
            sigma: 1.2,
            nu: 0.1,
        },
        2.0e-7,
    );
}

#[test]
fn tweedie_cdf_quantile_and_positive_density_are_consistent() {
    let tweedie = TweedieMeanDispersionPower::new();
    let theta = TweedieTheta {
        mean: 2.0,
        dispersion: 0.8,
        power: 1.5,
    };
    let lambda = theta.mean.powf(2.0 - theta.power) / (theta.dispersion * (2.0 - theta.power));
    let atom = (-lambda).exp();

    assert_close(tweedie.cdf(0.0, &theta), atom, 0.0, 1.0e-14);
    assert_eq!(tweedie.quantile(0.5 * atom, &theta), 0.0);

    let grid = [0.0_f64, 0.05, 0.25, 1.0, 2.0, 5.0, 10.0];
    assert_cdf_monotone(&tweedie, theta, &grid);
    for p in [atom + 0.01, 0.25, 0.5, 0.75, 0.95] {
        assert_discrete_or_continuous_generalized_inverse(&tweedie, p, theta, 2.0e-7);
    }

    let lower = 1.0e-4;
    let upper = tweedie.quantile(0.95, &theta);
    let integral = integrate_simpson(lower, upper, 2048, |y| {
        (-tweedie.nll(y, &theta, &mut tweedie.workspace())).exp()
    });
    let expected = tweedie.cdf(upper, &theta) - tweedie.cdf(lower, &theta);
    assert_close(integral, expected, 0.0, 3.0e-4);
}

#[test]
fn tail_quantiles_follow_distribution_contracts() {
    for p in TAIL_PROBABILITIES {
        assert_continuous_inverse(
            &NormalMuSigma::new(),
            p,
            NormalTheta {
                mu: 0.2,
                sigma: 1.1,
            },
            5.0e-7,
        );
        assert_continuous_inverse(
            &GumbelMuSigma::new(),
            p,
            GumbelTheta {
                mu: -0.2,
                sigma: 0.9,
            },
            1.0e-8,
        );
        assert_continuous_inverse(
            &LogisticMuSigma::new(),
            p,
            LogisticTheta {
                mu: 0.1,
                sigma: 1.3,
            },
            1.0e-8,
        );
    }

    for p in TAIL_PROBABILITIES {
        assert_discrete_inverse(&PoissonMean::new(), p, PoissonTheta { mu: 4.0 });
        assert_discrete_inverse(
            &NegativeBinomialMeanSize::new(),
            p,
            NegativeBinomialTheta {
                mu: 4.0,
                shape: 1.7,
            },
        );
    }
}

#[test]
fn mixed_distribution_quantiles_are_generalized_inverses() {
    for p in [1.0e-12, 0.05, 0.25, 0.5, 0.9, 1.0 - 1.0e-12] {
        assert_discrete_or_continuous_generalized_inverse(
            &ZipComponentMeanZeroProbability::new(),
            p,
            ZipComponentMeanZeroProbabilityTheta {
                component_mean: 3.0,
                zero_probability: 0.25,
            },
            1.0e-12,
        );
        assert_discrete_or_continuous_generalized_inverse(
            &ZinbComponentMeanSizeZeroProbability::new(),
            p,
            ZinbComponentMeanSizeZeroProbabilityTheta {
                component_mean: 3.0,
                size: 1.5,
                zero_probability: 0.25,
            },
            1.0e-12,
        );
        assert_discrete_or_continuous_generalized_inverse(
            &ZagaComponentMeanCvZeroProbability::new(),
            p,
            ZagaComponentMeanCvZeroProbabilityTheta {
                component_mean: 2.0,
                cv: 0.5,
                zero_probability: 0.25,
            },
            1.0e-7,
        );
    }
}
