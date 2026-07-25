#![allow(clippy::suboptimal_flops)]

use gamlss_core::{Family, HasCdf, HasQuantile};
use gamlss_family::{
    BetaBinomialMeanPrecision, BetaBinomialTheta, GeneralizedParetoScaleShape,
    GeneralizedParetoTheta, LogLogisticScaleShape, LogLogisticTheta,
};

use common::assert_close;

#[path = "common/helpers.rs"]
mod common;

// Reference values were generated with scipy.stats from SciPy 1.17.1.
#[test]
fn beta_binomial_matches_scipy_reference_values() {
    let family = BetaBinomialMeanPrecision::new();
    let theta = BetaBinomialTheta {
        probability: 0.4,
        precision: 8.0,
    };
    for (successes, log_pmf, cdf) in [
        (0.0, -3.203_498_371_876_435, 0.040_619_851_776_000_12),
        (4.0, -1.806_707_017_116_06, 0.601_520_869_376_001_5),
        (10.0, -5.371_312_721_660_349, 1.0),
    ] {
        assert_close(
            family.nll([successes, 10.0], &theta, &mut ()),
            -log_pmf,
            0.0,
            2.0e-13,
        );
        assert_close(family.cdf([successes, 10.0], &theta), cdf, 0.0, 2.0e-13);
    }
}

#[test]
fn log_logistic_matches_scipy_fisk_reference_values() {
    let family = LogLogisticScaleShape::new();
    let theta = LogLogisticTheta {
        scale: 1.3,
        shape: 2.4,
    };
    for (y, log_pdf, cdf) in [
        (0.2, -2.029_683_350_756_803_5, 0.011_070_652_260_374_894),
        (1.7, -1.143_342_370_719_097, 0.655_619_528_989_406_2),
        (10.0, -6.338_535_718_313_023_5, 0.992_582_955_937_998_3),
    ] {
        assert_close(family.nll(y, &theta, &mut ()), -log_pdf, 0.0, 2.0e-14);
        assert_close(family.cdf(y, &theta), cdf, 0.0, 2.0e-14);
    }
    for (probability, quantile) in [
        (0.01, 0.191_614_640_144_991_88),
        (0.5, 1.3),
        (0.99, 8.819_785_370_894_406),
    ] {
        assert_close(
            family.quantile(probability, &theta),
            quantile,
            2.0e-14,
            2.0e-14,
        );
    }
}

#[test]
fn generalized_pareto_matches_scipy_reference_values() {
    let family = GeneralizedParetoScaleShape::new();
    for (shape, rows, quantiles) in [
        (
            0.4,
            [
                (0.0, -0.530_628_251_062_170_4, 0.0),
                (0.7, -1.064_268_328_817_127_5, 0.316_939_254_488_476_8),
                (3.0, -2.399_916_951_818_073_4, 0.736_896_086_066_282_5),
            ],
            [
                (0.01, 0.017_119_960_163_856_404),
                (0.5, 1.357_908_620_784_800_6),
                (0.99, 22.565_687_140_408_198),
            ],
        ),
        (
            -0.2,
            [
                (0.0, -0.530_628_251_062_170_4, 0.0),
                (0.7, -0.874_397_970_265_069_3, 0.349_303_628_633_024_33),
                (3.0, -2.271_900_536_093_553, 0.886_572_380_176_313_5),
            ],
            [
                (0.01, 0.017_068_410_877_880_694),
                (0.5, 1.100_320_211_982_944_5),
                (0.99, 5.116_089_050_295_272_5),
            ],
        ),
        (
            0.0,
            [
                (0.0, -0.530_628_251_062_170_4, 0.0),
                (0.7, -0.942_392_956_944_523_3, 0.337_519_864_606_073_75),
                (3.0, -2.295_334_133_415_111_4, 0.828_762_857_055_211_8),
            ],
            [
                (0.01, 0.017_085_570_950_952_45),
                (0.5, 1.178_350_206_951_907),
                (0.99, 7.828_789_316_179_753),
            ],
        ),
    ] {
        let theta = GeneralizedParetoTheta { scale: 1.7, shape };
        for (y, log_pdf, cdf) in rows {
            assert_close(family.nll(y, &theta, &mut ()), -log_pdf, 0.0, 3.0e-14);
            assert_close(family.cdf(y, &theta), cdf, 0.0, 3.0e-14);
        }
        for (probability, quantile) in quantiles {
            assert_close(
                family.quantile(probability, &theta),
                quantile,
                3.0e-14,
                3.0e-14,
            );
        }
    }
}
