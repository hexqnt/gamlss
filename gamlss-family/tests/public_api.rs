use gamlss_core::Family;
use gamlss_family::{
    BetaMeanPrecision, ExponentialMean, ExponentialMeanTheta, ExponentialRate,
    ExponentialRateTheta, GammaMeanCv, GammaMeanCvTheta, GammaMeanShape, GammaMeanShapeTheta,
    GammaShapeRate, GammaShapeRateTheta, LogNormalLogLocationLogSd, LogNormalLogLocationLogSdTheta,
    LogNormalMeanLogSd, LogNormalMeanLogSdTheta, LogNormalMedianLogSd, LogNormalMedianLogSdTheta,
    NegativeBinomialMeanSize, NegativeBinomialTheta, WeibullMeanShape, WeibullMeanShapeTheta,
    WeibullScaleShape, WeibullScaleShapeTheta,
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
}
