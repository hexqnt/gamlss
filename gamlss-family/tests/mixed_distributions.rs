use gamlss_core::{HasCdf, HasQuantile};
use gamlss_family::*;

use common::assert_new_family_gradient_matches_finite_difference;

#[path = "common/helpers.rs"]
mod common;

#[test]
fn zero_adjusted_and_inflated_gradients_match_finite_differences() {
    let zip = ZipMeanZeroProbability::new();
    assert_new_family_gradient_matches_finite_difference::<_, 2>(&zip, 0.0, [1.2_f64.ln(), -1.0]);
    assert_new_family_gradient_matches_finite_difference::<_, 2>(&zip, 3.0, [1.2_f64.ln(), -1.0]);

    let zinb = ZinbMeanSizeZeroProbability::new();
    assert_new_family_gradient_matches_finite_difference::<_, 3>(
        &zinb,
        0.0,
        [1.2_f64.ln(), 2.0_f64.ln(), -1.0],
    );
    assert_new_family_gradient_matches_finite_difference::<_, 3>(
        &zinb,
        3.0,
        [1.2_f64.ln(), 2.0_f64.ln(), -1.0],
    );

    let zinb_total = ZinbTotalMeanSizeZeroProbability::new();
    assert_new_family_gradient_matches_finite_difference::<_, 3>(
        &zinb_total,
        0.0,
        [0.84_f64.ln(), 2.0_f64.ln(), -1.0],
    );
    assert_new_family_gradient_matches_finite_difference::<_, 3>(
        &zinb_total,
        3.0,
        [0.84_f64.ln(), 2.0_f64.ln(), -1.0],
    );

    let zaga = ZagaMeanSigmaZeroProbability::new();
    assert_new_family_gradient_matches_finite_difference::<_, 3>(
        &zaga,
        0.0,
        [1.2_f64.ln(), 0.8_f64.ln(), -1.0],
    );
    assert_new_family_gradient_matches_finite_difference::<_, 3>(
        &zaga,
        1.4,
        [1.2_f64.ln(), 0.8_f64.ln(), -1.0],
    );

    let zaga_total = ZagaTotalMeanCvZeroProbability::new();
    assert_new_family_gradient_matches_finite_difference::<_, 3>(
        &zaga_total,
        0.0,
        [0.84_f64.ln(), 0.8_f64.ln(), -1.0],
    );
    assert_new_family_gradient_matches_finite_difference::<_, 3>(
        &zaga_total,
        1.4,
        [0.84_f64.ln(), 0.8_f64.ln(), -1.0],
    );

    let beinf = BeinfMuSigmaNuTau::new();
    assert_new_family_gradient_matches_finite_difference::<_, 4>(
        &beinf,
        0.0,
        [0.1, -1.0, -2.0, -2.2],
    );
    assert_new_family_gradient_matches_finite_difference::<_, 4>(
        &beinf,
        0.4,
        [0.1, -1.0, -2.0, -2.2],
    );
    assert_new_family_gradient_matches_finite_difference::<_, 4>(
        &beinf,
        1.0,
        [0.1, -1.0, -2.0, -2.2],
    );
}

#[test]
fn mixed_family_cdfs_and_quantiles_handle_atoms() {
    let zip = ZipMeanZeroProbability::new();
    let zip_theta = ZipTheta {
        mu: 2.0,
        sigma: 0.3,
    };
    assert!(zip.cdf(0.0, zip_theta) > 0.3);
    let large_zip_cdf = zip.cdf(
        1000.0,
        ZipTheta {
            mu: 1000.0,
            sigma: 0.3,
        },
    );
    assert!(
        large_zip_cdf > 0.6 && large_zip_cdf < 0.7,
        "large ZIP CDF was {large_zip_cdf}"
    );
    assert_eq!(zip.quantile(0.1, zip_theta), 0.0);
    assert!(
        zip.quantile(
            0.5,
            ZipTheta {
                mu: 0.0,
                sigma: 0.3,
            },
        )
        .is_nan()
    );

    let zinb = ZinbMeanSizeZeroProbability::new();
    let zinb_theta = ZinbTheta {
        mu: 2.0,
        shape: 1.5,
        nu: 0.25,
    };
    assert!(zinb.cdf(0.0, zinb_theta) > 0.25);
    let large_zinb_cdf = zinb.cdf(
        1000.0,
        ZinbTheta {
            mu: 1000.0,
            shape: 2000.0,
            nu: 0.25,
        },
    );
    assert!(
        large_zinb_cdf > 0.55 && large_zinb_cdf < 0.65,
        "large ZINB CDF was {large_zinb_cdf}"
    );
    assert_eq!(zinb.quantile(0.1, zinb_theta), 0.0);
    assert!(
        zinb.quantile(
            0.5,
            ZinbTheta {
                mu: 2.0,
                shape: 0.0,
                nu: 0.25,
            },
        )
        .is_nan()
    );

    let zaga = ZagaMeanSigmaZeroProbability::new();
    let zaga_theta = ZagaTheta {
        mu: 1.2,
        sigma: 0.7,
        nu: 0.2,
    };
    assert_eq!(zaga.cdf(0.0, zaga_theta), 0.2);
    assert_eq!(zaga.quantile(0.1, zaga_theta), 0.0);

    let beinf = BeinfMuSigmaNuTau::new();
    let beinf_theta = BeinfTheta {
        mu: 0.4,
        sigma: 0.2,
        nu: 0.2,
        tau: 0.3,
    };
    assert!(beinf.cdf(0.0, beinf_theta) > 0.0);
    assert_eq!(beinf.quantile(0.01, beinf_theta), 0.0);
    assert_eq!(beinf.quantile(0.99, beinf_theta), 1.0);
}
