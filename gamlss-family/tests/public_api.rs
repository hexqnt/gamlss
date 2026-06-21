use gamlss_core::Family;
use gamlss_family::{
    BetaMeanPrecision, ExponentialMean, ExponentialMeanTheta, ExponentialRate,
    ExponentialRateTheta, GammaMeanCv, GammaMeanCvTheta, GammaMeanShape, GammaMeanShapeTheta,
    GammaShapeRate, GammaShapeRateTheta, InverseGaussianMeanCv, InverseGaussianMeanCvTheta,
    LogNormalLogLocationLogSd, LogNormalLogLocationLogSdTheta, LogNormalMeanCv,
    LogNormalMeanCvTheta, LogNormalMeanLogSd, LogNormalMeanLogSdTheta, LogNormalMedianLogSd,
    LogNormalMedianLogSdTheta, NegativeBinomialMeanDispersion, NegativeBinomialMeanDispersionTheta,
    NegativeBinomialMeanSize, NegativeBinomialTheta, SkewNormalMeanSdNu, SkewNormalMeanSdTheta,
    SkewStudentTMeanSdNuTau, SkewStudentTMeanSdTheta, StudentTMuSdTau, StudentTMuSdTauTheta,
    StudentTMuSigmaTau, StudentTMuSigmaTauTheta, TweedieMeanCvPower, TweedieMeanCvPowerTheta,
    TweedieMeanDispersionPower, TweedieTheta, WeibullMeanShape, WeibullMeanShapeTheta,
    WeibullScaleShape, WeibullScaleShapeTheta, ZagaTotalMeanCvZeroProbability,
    ZagaTotalMeanCvZeroProbabilityTheta, ZinbTotalMeanSizeZeroProbability,
    ZinbTotalMeanSizeZeroProbabilityTheta, ZipTotalMeanZeroProbability,
    ZipTotalMeanZeroProbabilityTheta,
};

#[test]
fn semantic_parameterization_aliases_construct_without_type_annotations() {
    assert!(
        GammaMeanCv::new()
            .nll(1.2, GammaMeanCvTheta { mean: 1.0, cv: 0.5 })
            .is_finite()
    );
    assert!(
        GammaMeanShape::new()
            .nll(
                1.2,
                GammaMeanShapeTheta {
                    mean: 1.0,
                    shape: 4.0,
                },
            )
            .is_finite()
    );
    assert!(
        GammaShapeRate::new()
            .nll(
                1.2,
                GammaShapeRateTheta {
                    shape: 4.0,
                    rate: 4.0,
                },
            )
            .is_finite()
    );

    assert!(
        ExponentialMean::new()
            .nll(1.2, ExponentialMeanTheta { mean: 1.0 })
            .is_finite()
    );
    assert!(
        ExponentialRate::new()
            .nll(1.2, ExponentialRateTheta { rate: 1.0 })
            .is_finite()
    );

    assert!(
        LogNormalMeanLogSd::new()
            .nll(
                1.2,
                LogNormalMeanLogSdTheta {
                    mean: 1.0,
                    log_sd: 0.5,
                },
            )
            .is_finite()
    );
    assert!(
        LogNormalMeanCv::new()
            .nll(1.2, LogNormalMeanCvTheta { mean: 1.0, cv: 0.5 },)
            .is_finite()
    );
    assert!(
        LogNormalMedianLogSd::new()
            .nll(
                1.2,
                LogNormalMedianLogSdTheta {
                    median: 1.0,
                    log_sd: 0.5,
                },
            )
            .is_finite()
    );
    assert!(
        LogNormalLogLocationLogSd::new()
            .nll(
                1.2,
                LogNormalLogLocationLogSdTheta {
                    log_location: 0.0,
                    log_sd: 0.5,
                },
            )
            .is_finite()
    );

    assert!(
        WeibullMeanShape::new()
            .nll(
                1.2,
                WeibullMeanShapeTheta {
                    mean: 1.0,
                    shape: 1.5,
                },
            )
            .is_finite()
    );
    assert!(
        WeibullScaleShape::new()
            .nll(
                1.2,
                WeibullScaleShapeTheta {
                    scale: 1.0,
                    shape: 1.5,
                },
            )
            .is_finite()
    );
    assert!(
        TweedieMeanDispersionPower::new()
            .nll(
                1.2,
                TweedieTheta {
                    mean: 1.0,
                    dispersion: 0.5,
                    power: 1.5,
                },
            )
            .is_finite()
    );
    assert!(
        TweedieMeanCvPower::new()
            .nll(
                1.2,
                TweedieMeanCvPowerTheta {
                    mean: 1.0,
                    cv: 0.5,
                    power: 1.5,
                },
            )
            .is_finite()
    );
    assert!(
        StudentTMuSigmaTau::new()
            .nll(
                1.2,
                StudentTMuSigmaTauTheta {
                    mu: 0.0,
                    sigma: 1.0,
                    tau: 5.0,
                },
            )
            .is_finite()
    );
    assert!(
        StudentTMuSdTau::new()
            .nll(
                1.2,
                StudentTMuSdTauTheta {
                    mu: 0.0,
                    sigma: 1.0,
                    tau: 5.0,
                },
            )
            .is_finite()
    );
    assert!(
        SkewNormalMeanSdNu::new()
            .nll(
                1.2,
                SkewNormalMeanSdTheta {
                    mean: 0.0,
                    sigma: 1.0,
                    nu: 0.5,
                },
            )
            .is_finite()
    );
    assert!(
        SkewStudentTMeanSdNuTau::new()
            .nll(
                1.2,
                SkewStudentTMeanSdTheta {
                    mean: 0.0,
                    sigma: 1.0,
                    nu: 0.5,
                    tau: 5.0,
                },
            )
            .is_finite()
    );
}

#[test]
fn existing_mean_style_aliases_construct_without_type_annotations() {
    assert!(
        BetaMeanPrecision::new()
            .nll(
                0.4,
                gamlss_family::BetaTheta {
                    mu: 0.4,
                    precision: 5.0,
                },
            )
            .is_finite()
    );
    assert!(
        NegativeBinomialMeanSize::new()
            .nll(
                2.0,
                NegativeBinomialTheta {
                    mu: 3.0,
                    shape: 4.0,
                },
            )
            .is_finite()
    );
    assert!(
        NegativeBinomialMeanDispersion::new()
            .nll(
                2.0,
                NegativeBinomialMeanDispersionTheta {
                    mean: 3.0,
                    dispersion: 0.25,
                },
            )
            .is_finite()
    );
    assert!(
        InverseGaussianMeanCv::new()
            .nll(1.2, InverseGaussianMeanCvTheta { mean: 1.0, cv: 0.5 })
            .is_finite()
    );
    assert!(
        ZipTotalMeanZeroProbability::new()
            .nll(
                2.0,
                ZipTotalMeanZeroProbabilityTheta {
                    total_mean: 2.0,
                    zero_probability: 0.2,
                },
            )
            .is_finite()
    );
    assert!(
        ZagaTotalMeanCvZeroProbability::new()
            .nll(
                1.2,
                ZagaTotalMeanCvZeroProbabilityTheta {
                    total_mean: 1.0,
                    cv: 0.5,
                    zero_probability: 0.2,
                },
            )
            .is_finite()
    );
    assert!(
        ZinbTotalMeanSizeZeroProbability::new()
            .nll(
                2.0,
                ZinbTotalMeanSizeZeroProbabilityTheta {
                    total_mean: 2.0,
                    size: 4.0,
                    zero_probability: 0.2,
                },
            )
            .is_finite()
    );
}
