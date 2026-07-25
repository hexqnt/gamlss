use approx::assert_relative_eq;
use gamlss_core::HasCrps;
use gamlss_family::{
    BeinfMuSigmaNuTau, BeinfTheta, BetaBinomialMeanPrecision, BetaBinomialTheta,
    BinomialFixedTrialsProbability, BinomialTheta, BinomialVaryingTrialsProbability, Categorical,
    CategoricalTheta, ChiDegreesOfFreedom, ChiTheta, GammaShapeRate, GammaShapeRateTheta,
    GeneralizedGammaScaleSigmaNu, GeneralizedGammaTheta, GevMuSigmaShape, GevTheta,
    JohnsonSuMuSigmaNuTau, JohnsonSuTheta, NegativeBinomialMeanDispersion,
    NegativeBinomialMeanDispersionTheta, NegativeBinomialMeanSize, NegativeBinomialTheta,
    PoissonMean, PoissonTheta, PowerExponentialMuSigmaNu, PowerExponentialTheta, ShashMuSigmaNuTau,
    ShashTheta, SkewPowerExponentialMuSigmaSkewPower, SkewPowerExponentialTheta,
    SkewStudentTMuSigmaNuTau, SkewStudentTTheta, TweedieMeanCvPower, TweedieMeanCvPowerTheta,
    TweedieMeanDispersionPower, TweedieTheta, ZagaComponentMeanCvZeroProbability,
    ZagaComponentMeanCvZeroProbabilityTheta, ZagaTotalMeanCvZeroProbability,
    ZagaTotalMeanCvZeroProbabilityTheta, ZinbComponentMeanSizeZeroProbability,
    ZinbComponentMeanSizeZeroProbabilityTheta, ZinbTotalMeanSizeZeroProbability,
    ZinbTotalMeanSizeZeroProbabilityTheta, ZipComponentMeanZeroProbability,
    ZipComponentMeanZeroProbabilityTheta, ZipTotalMeanZeroProbability,
    ZipTotalMeanZeroProbabilityTheta,
};

#[test]
fn finite_and_count_crps_match_independent_reference_values() {
    let binomial = BinomialFixedTrialsProbability::try_new(10).unwrap();
    let binomial_theta = BinomialTheta { probability: 0.5 };
    assert_relative_eq!(
        binomial.crps(4.0, &binomial_theta),
        0.595_577_239_990_234_4,
        epsilon = 1.0e-14
    );

    let varying = BinomialVaryingTrialsProbability::new();
    assert_relative_eq!(
        varying.crps([4.0, 10.0], &binomial_theta),
        binomial.crps(4.0, &binomial_theta),
        epsilon = 1.0e-14
    );

    let beta_binomial = BetaBinomialMeanPrecision::new();
    assert_relative_eq!(
        beta_binomial.crps(
            [4.0, 10.0],
            &BetaBinomialTheta {
                probability: 0.4,
                precision: 8.0,
            },
        ),
        0.534_067_958_729_645_2,
        epsilon = 2.0e-14
    );

    let categorical = Categorical::<3>::new();
    assert_relative_eq!(
        categorical.crps(
            1.0,
            &CategoricalTheta {
                probabilities: [0.2, 0.3, 0.5],
            },
        ),
        0.29,
        epsilon = 1.0e-14
    );

    let negative_binomial = NegativeBinomialMeanSize::new();
    assert_relative_eq!(
        negative_binomial.crps(
            2.0,
            &NegativeBinomialTheta {
                mu: 5.0,
                shape: 5.0,
            },
        ),
        1.553_362_990_905_857_7,
        epsilon = 2.0e-10
    );
    assert_relative_eq!(
        negative_binomial.crps(
            3.0,
            &NegativeBinomialTheta {
                mu: 3.45,
                shape: 2.3,
            },
        ),
        0.627_403_553_856_612_4,
        epsilon = 2.0e-10
    );
    assert_relative_eq!(
        negative_binomial.crps(
            0.0,
            &NegativeBinomialTheta {
                mu: 0.175,
                shape: 0.7,
            },
        ),
        0.021_551_617_349_051_536,
        epsilon = 2.0e-10
    );

    let poisson = PoissonMean::new();
    let poisson_score = poisson.crps(1.0, &PoissonTheta { mu: 2.0 });
    let concentrated_nb = negative_binomial.crps(
        1.0,
        &NegativeBinomialTheta {
            mu: 2.0,
            shape: 1.0e12,
        },
    );
    assert_relative_eq!(concentrated_nb, poisson_score, epsilon = 2.0e-6);
}

#[test]
fn transformed_continuous_crps_match_scipy_quadrature_references() {
    assert_relative_eq!(
        ChiDegreesOfFreedom::new().crps(
            1.2,
            &ChiTheta {
                degrees_of_freedom: 3.5,
            },
        ),
        0.299_048_750_515_216_57,
        epsilon = 3.0e-9
    );

    let generalized_gamma = GeneralizedGammaScaleSigmaNu::new();
    assert_relative_eq!(
        generalized_gamma.crps(
            1.1,
            &GeneralizedGammaTheta {
                scale: 1.3,
                sigma: 0.7,
                nu: 0.8,
            },
        ),
        0.199_099_518_305_626_83,
        epsilon = 4.0e-9
    );
    assert_relative_eq!(
        generalized_gamma.crps(
            1.1,
            &GeneralizedGammaTheta {
                scale: 1.3,
                sigma: 0.8,
                nu: -0.5,
            },
        ),
        0.365_446_132_878_106_7,
        epsilon = 5.0e-9
    );

    assert_relative_eq!(
        GevMuSigmaShape::new().crps(
            0.3,
            &GevTheta {
                mu: -0.2,
                sigma: 1.4,
                nu: 0.2,
            },
        ),
        0.432_477_849_861_784_17,
        epsilon = 4.0e-9
    );

    assert_relative_eq!(
        JohnsonSuMuSigmaNuTau::new().crps(
            0.8,
            &JohnsonSuTheta {
                mu: 0.2,
                sigma: 1.3,
                nu: -0.4,
                tau: 1.7,
            },
        ),
        0.234_095_070_791_210_97,
        epsilon = 4.0e-9
    );

    assert_relative_eq!(
        PowerExponentialMuSigmaNu::new().crps(
            0.7,
            &PowerExponentialTheta {
                mu: -0.3,
                sigma: 1.4,
                nu: 1.3,
            },
        ),
        0.607_937_732_240_242_6,
        epsilon = 4.0e-9
    );

    assert_relative_eq!(
        ShashMuSigmaNuTau::new().crps(
            0.7,
            &ShashTheta {
                mu: 0.2,
                sigma: 1.3,
                nu: 1.5,
                tau: 0.8,
            },
        ),
        0.459_548_921_907_022,
        epsilon = 5.0e-9
    );

    assert_relative_eq!(
        SkewPowerExponentialMuSigmaSkewPower::new().crps(
            0.7,
            &SkewPowerExponentialTheta {
                mu: 0.2,
                sigma: 1.3,
                skew_ratio: 1.7,
                power: 1.4,
            },
        ),
        0.366_195_144_151_163_33,
        epsilon = 5.0e-9
    );
}

#[test]
fn expensive_mixed_and_heavy_tail_crps_match_reference_values() {
    assert_relative_eq!(
        SkewStudentTMuSigmaNuTau::new().crps(
            0.7,
            &SkewStudentTTheta {
                mu: 0.2,
                sigma: 1.3,
                nu: 1.1,
                tau: 4.5,
            },
        ),
        0.308_810_059_758_956_65,
        epsilon = 8.0e-9
    );

    assert_relative_eq!(
        TweedieMeanDispersionPower::new().crps(
            0.8,
            &TweedieTheta {
                mean: 1.2,
                dispersion: 0.7,
                power: 1.5,
            },
        ),
        0.233_477_165_905_599_47,
        epsilon = 8.0e-9
    );

    assert_relative_eq!(
        BeinfMuSigmaNuTau::new().crps(
            0.6,
            &BeinfTheta {
                mu: 0.4,
                sigma: 0.3,
                nu: 0.2,
                tau: 0.5,
            },
        ),
        0.137_193_852_055_258_58,
        epsilon = 4.0e-9
    );

    assert_relative_eq!(
        ZagaComponentMeanCvZeroProbability::new().crps(
            0.8,
            &ZagaComponentMeanCvZeroProbabilityTheta {
                component_mean: 1.5,
                cv: 0.7,
                zero_probability: 0.2,
            },
        ),
        0.266_330_703_900_746_4,
        epsilon = 3.0e-12
    );

    assert_relative_eq!(
        ZipComponentMeanZeroProbability::new().crps(
            1.0,
            &ZipComponentMeanZeroProbabilityTheta {
                component_mean: 2.0,
                zero_probability: 0.3,
            },
        ),
        0.391_431_691_019_364_2,
        epsilon = 3.0e-12
    );

    assert_relative_eq!(
        ZinbComponentMeanSizeZeroProbability::new().crps(
            1.0,
            &ZinbComponentMeanSizeZeroProbabilityTheta {
                component_mean: 2.0,
                size: 1.5,
                zero_probability: 0.3,
            },
        ),
        0.434_263_823_003_445,
        epsilon = 3.0e-10
    );
}

#[test]
fn equivalent_parameterizations_have_identical_crps() {
    let mean_size = NegativeBinomialMeanSize::new();
    let mean_dispersion = NegativeBinomialMeanDispersion::new();
    assert_relative_eq!(
        mean_size.crps(
            3.0,
            &NegativeBinomialTheta {
                mu: 2.3,
                shape: 5.0,
            },
        ),
        mean_dispersion.crps(
            3.0,
            &NegativeBinomialMeanDispersionTheta {
                mean: 2.3,
                dispersion: 0.2,
            },
        ),
        epsilon = 2.0e-10
    );

    let mean = 1.2_f64;
    let dispersion = 0.7_f64;
    let power = 1.5_f64;
    let cv = (dispersion / mean.powf(2.0 - power)).sqrt();
    assert_relative_eq!(
        TweedieMeanDispersionPower::new().crps(
            0.8,
            &TweedieTheta {
                mean,
                dispersion,
                power,
            },
        ),
        TweedieMeanCvPower::new().crps(0.8, &TweedieMeanCvPowerTheta { mean, cv, power },),
        epsilon = 8.0e-9
    );

    let zero_probability = 0.2;
    let component_mean = 1.5;
    let total_mean = component_mean * (1.0 - zero_probability);
    assert_relative_eq!(
        ZagaComponentMeanCvZeroProbability::new().crps(
            0.8,
            &ZagaComponentMeanCvZeroProbabilityTheta {
                component_mean,
                cv: 0.7,
                zero_probability,
            },
        ),
        ZagaTotalMeanCvZeroProbability::new().crps(
            0.8,
            &ZagaTotalMeanCvZeroProbabilityTheta {
                total_mean,
                cv: 0.7,
                zero_probability,
            },
        ),
        epsilon = 3.0e-12
    );

    assert_relative_eq!(
        ZipComponentMeanZeroProbability::new().crps(
            1.0,
            &ZipComponentMeanZeroProbabilityTheta {
                component_mean,
                zero_probability,
            },
        ),
        ZipTotalMeanZeroProbability::new().crps(
            1.0,
            &ZipTotalMeanZeroProbabilityTheta {
                total_mean,
                zero_probability,
            },
        ),
        epsilon = 3.0e-12
    );

    assert_relative_eq!(
        ZinbComponentMeanSizeZeroProbability::new().crps(
            1.0,
            &ZinbComponentMeanSizeZeroProbabilityTheta {
                component_mean,
                size: 1.5,
                zero_probability,
            },
        ),
        ZinbTotalMeanSizeZeroProbability::new().crps(
            1.0,
            &ZinbTotalMeanSizeZeroProbabilityTheta {
                total_mean,
                size: 1.5,
                zero_probability,
            },
        ),
        epsilon = 3.0e-10
    );
}

#[test]
fn crps_rejects_distributions_without_a_finite_first_moment() {
    assert!(
        GeneralizedGammaScaleSigmaNu::new()
            .crps(
                1.0,
                &GeneralizedGammaTheta {
                    scale: 1.0,
                    sigma: 2.0,
                    nu: -0.5,
                },
            )
            .is_nan()
    );
    assert!(
        GevMuSigmaShape::new()
            .crps(
                1.0,
                &GevTheta {
                    mu: 0.0,
                    sigma: 1.0,
                    nu: 1.0,
                },
            )
            .is_nan()
    );
    assert!(
        GevMuSigmaShape::new()
            .crps(
                -6.0,
                &GevTheta {
                    mu: 0.0,
                    sigma: 1.0,
                    nu: 0.2,
                },
            )
            .is_nan()
    );
    assert!(
        SkewStudentTMuSigmaNuTau::new()
            .crps(
                1.0,
                &SkewStudentTTheta {
                    mu: 0.0,
                    sigma: 1.0,
                    nu: 0.5,
                    tau: 1.0,
                },
            )
            .is_nan()
    );

    assert_relative_eq!(
        GammaShapeRate::new().crps(
            1.0,
            &GammaShapeRateTheta {
                shape: 2.0,
                rate: 1.0,
            },
        ),
        0.457_276_647_028_654_84,
        epsilon = 1.0e-12
    );
}
