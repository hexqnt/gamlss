use gamlss_core::{Family, HasCdf, HasQuantile, ParameterParts};
use gamlss_family::{
    BeinfMuSigmaNuTau, BeinfTheta, GeneralizedGammaMuSigmaNu, GeneralizedGammaTheta,
    GevMuSigmaShape, GevTheta, JohnsonSuMuSigmaNuTau, JohnsonSuTheta, NormalMuSigma, NormalTheta,
    PowerExponentialMuSigmaNu, PowerExponentialTheta, ShashMuSigmaNuTau, ShashTheta,
    SkewNormalMuSigmaNu, SkewNormalTheta, SkewStudentTMuSigmaNuTau, SkewStudentTTheta,
    TweedieMeanDispersionPower, TweedieTheta, ZagaMeanSigmaZeroProbability, ZagaTheta,
    ZinbMeanSizeZeroProbability, ZinbTheta, ZipMeanZeroProbability, ZipTheta,
};

const FD_REL_TOL: f64 = 8.0e-4;
const FD_ABS_TOL: f64 = 8.0e-4;

fn assert_close(actual: f64, expected: f64, rel_tol: f64, abs_tol: f64) {
    let diff = (actual - expected).abs();
    let scale = actual.abs().max(expected.abs()).max(1.0);
    assert!(
        diff <= abs_tol.max(rel_tol * scale),
        "actual {actual:?} differs from expected {expected:?}; diff={diff:?}"
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
        let epsilon = 1.0e-5 * eta[index].abs().max(1.0);
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
            "gradient component {index} is {actual:?}"
        );
        assert!(
            finite_difference.is_finite(),
            "finite-difference component {index} is {finite_difference:?}"
        );
        assert_close(actual, finite_difference, FD_REL_TOL, FD_ABS_TOL);
    }
}

fn assert_inverse<F>(family: &F, p: f64, theta: F::Theta, tolerance: f64)
where
    F: for<'obs> Family<Observation<'obs> = f64> + HasCdf + HasQuantile,
    F::Theta: Copy,
{
    let q = family.quantile(p, theta);
    assert!(q.is_finite(), "quantile({p}) returned {q:?}");
    assert_close(family.cdf(q, theta), p, 0.0, tolerance);
}

#[test]
fn new_continuous_family_gradients_match_finite_differences() {
    assert_gradient_matches_finite_difference::<_, 3>(
        &SkewNormalMuSigmaNu::new(),
        0.4,
        [0.1, -0.2, 0.3],
    );
    assert_gradient_matches_finite_difference::<_, 3>(
        &PowerExponentialMuSigmaNu::new(),
        0.4,
        [0.1, -0.2, 2.0_f64.ln()],
    );
    assert_gradient_matches_finite_difference::<_, 4>(
        &SkewStudentTMuSigmaNuTau::new(),
        0.4,
        [0.1, -0.2, 0.3, 5.0_f64.ln()],
    );
    assert_gradient_matches_finite_difference::<_, 4>(
        &ShashMuSigmaNuTau::new(),
        0.4,
        [0.1, -0.2, 0.5_f64.ln(), 0.8_f64.ln()],
    );
    assert_gradient_matches_finite_difference::<_, 4>(
        &JohnsonSuMuSigmaNuTau::new(),
        0.4,
        [0.1, -0.2, 0.3, 1.2_f64.ln()],
    );
    assert_gradient_matches_finite_difference::<_, 3>(
        &GeneralizedGammaMuSigmaNu::new(),
        1.4,
        [0.1, -0.2, 0.5],
    );
    assert_gradient_matches_finite_difference::<_, 3>(
        &GevMuSigmaShape::new(),
        0.4,
        [0.1, -0.2, 0.1],
    );
    assert_gradient_matches_finite_difference::<_, 3>(
        &TweedieMeanDispersionPower::new(),
        1.4,
        [0.2, -0.3, 0.0],
    );
}

#[test]
fn new_mixed_family_gradients_match_finite_differences() {
    let zip = ZipMeanZeroProbability::new();
    assert_gradient_matches_finite_difference::<_, 2>(&zip, 0.0, [1.2_f64.ln(), -1.0]);
    assert_gradient_matches_finite_difference::<_, 2>(&zip, 3.0, [1.2_f64.ln(), -1.0]);

    let zinb = ZinbMeanSizeZeroProbability::new();
    assert_gradient_matches_finite_difference::<_, 3>(
        &zinb,
        0.0,
        [1.2_f64.ln(), 2.0_f64.ln(), -1.0],
    );
    assert_gradient_matches_finite_difference::<_, 3>(
        &zinb,
        3.0,
        [1.2_f64.ln(), 2.0_f64.ln(), -1.0],
    );

    let zaga = ZagaMeanSigmaZeroProbability::new();
    assert_gradient_matches_finite_difference::<_, 3>(
        &zaga,
        0.0,
        [1.2_f64.ln(), 0.8_f64.ln(), -1.0],
    );
    assert_gradient_matches_finite_difference::<_, 3>(
        &zaga,
        1.4,
        [1.2_f64.ln(), 0.8_f64.ln(), -1.0],
    );

    let beinf = BeinfMuSigmaNuTau::new();
    assert_gradient_matches_finite_difference::<_, 4>(&beinf, 0.0, [0.1, -1.0, -2.0, -2.2]);
    assert_gradient_matches_finite_difference::<_, 4>(&beinf, 0.4, [0.1, -1.0, -2.0, -2.2]);
    assert_gradient_matches_finite_difference::<_, 4>(&beinf, 1.0, [0.1, -1.0, -2.0, -2.2]);
}

#[test]
fn cdf_quantile_roundtrips_for_new_families() {
    assert_inverse(
        &SkewNormalMuSigmaNu::new(),
        0.4,
        SkewNormalTheta {
            mu: 0.1,
            sigma: 1.2,
            nu: 0.5,
        },
        3.0e-5,
    );
    assert_inverse(
        &PowerExponentialMuSigmaNu::new(),
        0.4,
        PowerExponentialTheta {
            mu: 0.1,
            sigma: 1.2,
            nu: 1.5,
        },
        2.0e-7,
    );
    assert_inverse(
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
    assert_inverse(
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
    assert_inverse(
        &GeneralizedGammaMuSigmaNu::new(),
        0.4,
        GeneralizedGammaTheta {
            mu: 1.2,
            sigma: 0.6,
            nu: 0.8,
        },
        2.0e-7,
    );
    assert_inverse(
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
fn mixed_family_cdfs_and_quantiles_handle_atoms() {
    let zip = ZipMeanZeroProbability::new();
    let zip_theta = ZipTheta {
        mu: 2.0,
        sigma: 0.3,
    };
    assert!(zip.cdf(0.0, zip_theta) > 0.3);
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

#[test]
fn invalid_domains_return_non_finite_likelihoods_for_new_families() {
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
        TweedieMeanDispersionPower::new()
            .nll(
                -1.0,
                TweedieTheta {
                    mu: 1.0,
                    sigma: 1.0,
                    nu: 1.5,
                },
            )
            .is_infinite()
    );
}

#[test]
fn new_families_match_expected_symmetric_special_cases() {
    let normal = NormalMuSigma::new();
    let normal_theta = NormalTheta {
        mu: 0.2,
        sigma: 1.3,
    };

    let skew_normal = SkewNormalMuSigmaNu::new();
    let skew_normal_theta = SkewNormalTheta {
        mu: normal_theta.mu,
        sigma: normal_theta.sigma,
        nu: 0.0,
    };
    assert_close(
        skew_normal.nll(0.7, skew_normal_theta),
        normal.nll(0.7, normal_theta),
        1.0e-9,
        1.0e-9,
    );
    assert_close(
        skew_normal.cdf(0.7, skew_normal_theta),
        normal.cdf(0.7, normal_theta),
        0.0,
        2.0e-7,
    );

    let power_exponential = PowerExponentialMuSigmaNu::new();
    assert_close(
        power_exponential.nll(
            0.7,
            PowerExponentialTheta {
                mu: normal_theta.mu,
                sigma: normal_theta.sigma,
                nu: 2.0,
            },
        ),
        normal.nll(0.7, normal_theta),
        1.0e-12,
        1.0e-12,
    );

    let shash = ShashMuSigmaNuTau::new();
    let shash_theta = ShashTheta {
        mu: normal_theta.mu,
        sigma: normal_theta.sigma,
        nu: 1.0,
        tau: 1.0,
    };
    assert_close(
        shash.nll(0.7, shash_theta),
        normal.nll(0.7, normal_theta),
        1.0e-12,
        1.0e-12,
    );
    assert_close(
        shash.cdf(0.7, shash_theta),
        normal.cdf(0.7, normal_theta),
        0.0,
        2.0e-7,
    );
}
