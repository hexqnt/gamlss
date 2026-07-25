#![allow(clippy::float_cmp)]
use gamlss_core::{Family, HasCdf, HasQuantile};
use gamlss_family::*;

use common::{assert_close, assert_new_family_gradient_matches_finite_difference};

#[path = "common/helpers.rs"]
mod common;

#[test]
fn zero_adjusted_and_inflated_gradients_match_finite_differences() {
    let zip = ZipComponentMeanZeroProbability::new();
    assert_new_family_gradient_matches_finite_difference::<_, 2>(&zip, 0.0, [1.2_f64.ln(), -1.0]);
    assert_new_family_gradient_matches_finite_difference::<_, 2>(&zip, 3.0, [1.2_f64.ln(), -1.0]);

    let zinb = ZinbComponentMeanSizeZeroProbability::new();
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

    let zaga = ZagaComponentMeanCvZeroProbability::new();
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
fn mixed_likelihoods_match_their_component_and_atom_decompositions() {
    let zip = ZipComponentMeanZeroProbability::new();
    let poisson = PoissonMean::new();
    let zip_theta = ZipComponentMeanZeroProbabilityTheta {
        component_mean: 2.3,
        zero_probability: 0.25,
    };
    let poisson_theta = PoissonTheta {
        mu: zip_theta.component_mean,
    };
    let zip_zero_mass = (1.0 - zip_theta.zero_probability).mul_add(
        (-zip_theta.component_mean).exp(),
        zip_theta.zero_probability,
    );
    assert_close(
        zip.nll(0.0, &zip_theta, &mut zip.workspace()),
        -zip_zero_mass.ln(),
        0.0,
        2.0e-14,
    );
    assert_close(
        zip.nll(3.0, &zip_theta, &mut zip.workspace()),
        poisson.nll(3.0, &poisson_theta, &mut poisson.workspace())
            - (1.0 - zip_theta.zero_probability).ln(),
        0.0,
        2.0e-14,
    );
    assert_close(
        zip.cdf(3.0, &zip_theta),
        (1.0 - zip_theta.zero_probability)
            .mul_add(poisson.cdf(3.0, &poisson_theta), zip_theta.zero_probability),
        0.0,
        2.0e-14,
    );

    let zinb = ZinbComponentMeanSizeZeroProbability::new();
    let negative_binomial = NegativeBinomialMeanSize::new();
    let zinb_theta = ZinbComponentMeanSizeZeroProbabilityTheta {
        component_mean: 2.3,
        size: 1.7,
        zero_probability: 0.25,
    };
    let negative_binomial_theta = NegativeBinomialTheta {
        mu: zinb_theta.component_mean,
        shape: zinb_theta.size,
    };
    let nb_zero_mass = (-negative_binomial.nll(
        0.0,
        &negative_binomial_theta,
        &mut negative_binomial.workspace(),
    ))
    .exp();
    let zinb_zero_mass =
        (1.0 - zinb_theta.zero_probability).mul_add(nb_zero_mass, zinb_theta.zero_probability);
    assert_close(
        zinb.nll(0.0, &zinb_theta, &mut zinb.workspace()),
        -zinb_zero_mass.ln(),
        0.0,
        2.0e-14,
    );
    assert_close(
        zinb.nll(3.0, &zinb_theta, &mut zinb.workspace()),
        negative_binomial.nll(
            3.0,
            &negative_binomial_theta,
            &mut negative_binomial.workspace(),
        ) - (1.0 - zinb_theta.zero_probability).ln(),
        0.0,
        2.0e-14,
    );
    assert_close(
        zinb.cdf(3.0, &zinb_theta),
        (1.0 - zinb_theta.zero_probability).mul_add(
            negative_binomial.cdf(3.0, &negative_binomial_theta),
            zinb_theta.zero_probability,
        ),
        0.0,
        2.0e-14,
    );

    let zaga = ZagaComponentMeanCvZeroProbability::new();
    let gamma = GammaMeanCv::new();
    let zaga_theta = ZagaComponentMeanCvZeroProbabilityTheta {
        component_mean: 1.8,
        cv: 0.6,
        zero_probability: 0.2,
    };
    let gamma_theta = GammaMeanCvTheta {
        mean: zaga_theta.component_mean,
        cv: zaga_theta.cv,
    };
    assert_close(
        zaga.nll(0.0, &zaga_theta, &mut zaga.workspace()),
        -zaga_theta.zero_probability.ln(),
        0.0,
        2.0e-14,
    );
    assert_close(
        zaga.nll(1.4, &zaga_theta, &mut zaga.workspace()),
        gamma.nll(1.4, &gamma_theta, &mut gamma.workspace())
            - (1.0 - zaga_theta.zero_probability).ln(),
        0.0,
        2.0e-14,
    );
    assert_close(
        zaga.cdf(1.4, &zaga_theta),
        (1.0 - zaga_theta.zero_probability)
            .mul_add(gamma.cdf(1.4, &gamma_theta), zaga_theta.zero_probability),
        0.0,
        2.0e-14,
    );

    let beinf = BeinfMuSigmaNuTau::new();
    let beta = BetaMeanPrecision::new();
    let beinf_theta = BeinfTheta {
        mu: 0.35,
        sigma: 0.2,
        nu: 0.3,
        tau: 0.4,
    };
    let denominator = 1.0 + beinf_theta.nu + beinf_theta.tau;
    let beta_theta = BetaTheta {
        mu: beinf_theta.mu,
        precision: 1.0 / beinf_theta.sigma - 1.0,
    };
    assert_close(
        beinf.nll(0.0, &beinf_theta, &mut beinf.workspace()),
        (denominator / beinf_theta.nu).ln(),
        0.0,
        2.0e-14,
    );
    assert_close(
        beinf.nll(1.0, &beinf_theta, &mut beinf.workspace()),
        (denominator / beinf_theta.tau).ln(),
        0.0,
        2.0e-14,
    );
    assert_close(
        beinf.nll(0.4, &beinf_theta, &mut beinf.workspace()),
        beta.nll(0.4, &beta_theta, &mut beta.workspace()) + denominator.ln(),
        0.0,
        2.0e-14,
    );

    let reflected = BeinfTheta {
        mu: 1.0 - beinf_theta.mu,
        sigma: beinf_theta.sigma,
        nu: beinf_theta.tau,
        tau: beinf_theta.nu,
    };
    for y in [0.0_f64, 0.2, 0.7, 1.0] {
        assert_close(
            beinf.nll(y, &beinf_theta, &mut beinf.workspace()),
            beinf.nll(1.0 - y, &reflected, &mut beinf.workspace()),
            0.0,
            2.0e-14,
        );
    }
    let zero_mass = beinf_theta.nu / denominator;
    let one_mass = beinf_theta.tau / denominator;
    let cdf_at_zero = beinf.cdf(0.0, &beinf_theta);
    assert_close(cdf_at_zero, zero_mass, 0.0, 2.0e-14);
    assert_eq!(beinf.quantile(cdf_at_zero, &beinf_theta), 0.0);
    assert_eq!(
        beinf.quantile(0.5_f64.mul_add(-one_mass, 1.0), &beinf_theta),
        1.0
    );
}

#[test]
fn mixed_family_cdfs_and_quantiles_handle_atoms() {
    let zip = ZipComponentMeanZeroProbability::new();
    let zip_theta = ZipComponentMeanZeroProbabilityTheta {
        component_mean: 2.0,
        zero_probability: 0.3,
    };
    assert!(zip.cdf(0.0, &zip_theta) > 0.3);
    let large_zip_cdf = zip.cdf(
        1000.0,
        &ZipComponentMeanZeroProbabilityTheta {
            component_mean: 1000.0,
            zero_probability: 0.3,
        },
    );
    assert!(
        large_zip_cdf > 0.6 && large_zip_cdf < 0.7,
        "large ZIP CDF was {large_zip_cdf}"
    );
    assert_eq!(zip.quantile(0.1, &zip_theta), 0.0);
    assert!(
        zip.quantile(
            0.5,
            &ZipComponentMeanZeroProbabilityTheta {
                component_mean: 0.0,
                zero_probability: 0.3,
            },
        )
        .is_nan()
    );

    let zinb = ZinbComponentMeanSizeZeroProbability::new();
    let zinb_theta = ZinbComponentMeanSizeZeroProbabilityTheta {
        component_mean: 2.0,
        size: 1.5,
        zero_probability: 0.25,
    };
    assert!(zinb.cdf(0.0, &zinb_theta) > 0.25);
    let large_zinb_cdf = zinb.cdf(
        1000.0,
        &ZinbComponentMeanSizeZeroProbabilityTheta {
            component_mean: 1000.0,
            size: 2000.0,
            zero_probability: 0.25,
        },
    );
    assert!(
        large_zinb_cdf > 0.55 && large_zinb_cdf < 0.65,
        "large ZINB CDF was {large_zinb_cdf}"
    );
    assert_eq!(zinb.quantile(0.1, &zinb_theta), 0.0);
    assert!(
        zinb.quantile(
            0.5,
            &ZinbComponentMeanSizeZeroProbabilityTheta {
                component_mean: 2.0,
                size: 0.0,
                zero_probability: 0.25,
            },
        )
        .is_nan()
    );

    let zaga = ZagaComponentMeanCvZeroProbability::new();
    let zaga_theta = ZagaComponentMeanCvZeroProbabilityTheta {
        component_mean: 1.2,
        cv: 0.7,
        zero_probability: 0.2,
    };
    assert_eq!(zaga.cdf(0.0, &zaga_theta), 0.2);
    assert_eq!(zaga.quantile(0.1, &zaga_theta), 0.0);

    let beinf = BeinfMuSigmaNuTau::new();
    let beinf_theta = BeinfTheta {
        mu: 0.4,
        sigma: 0.2,
        nu: 0.2,
        tau: 0.3,
    };
    assert!(beinf.cdf(0.0, &beinf_theta) > 0.0);
    assert_eq!(beinf.quantile(0.01, &beinf_theta), 0.0);
    assert_eq!(beinf.quantile(0.99, &beinf_theta), 1.0);
}
