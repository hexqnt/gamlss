use gamlss_core::{Family, HasCdf, HasQuantile, ParameterParts};
use gamlss_family::{
    BeinfMuSigmaNuTau, BeinfTheta, GeneralizedGammaMuSigmaNu, GeneralizedGammaTheta,
    GevMuSigmaShape, GevTheta, JohnsonSuMuSigmaNuTau, JohnsonSuTheta, NormalMuSigma, NormalTheta,
    PowerExponentialMuSigmaNu, PowerExponentialTheta, ShashMuSigmaNuTau, ShashTheta,
    SkewNormalMuSigmaNu, SkewNormalTheta, SkewStudentTMuSigmaNuTau, SkewStudentTTheta,
    StudentTMuSigma, StudentTMuSigmaTau, StudentTMuSigmaTauTheta, StudentTTheta,
    TweedieMeanCvPower, TweedieMeanCvPowerTheta, TweedieMeanDispersionPower, TweedieTheta,
    ZagaMeanSigmaZeroProbability, ZagaTheta, ZagaTotalMeanCvZeroProbability,
    ZagaTotalMeanCvZeroProbabilityTheta, ZinbMeanSizeZeroProbability, ZinbTheta,
    ZinbTotalMeanSizeZeroProbability, ZinbTotalMeanSizeZeroProbabilityTheta,
    ZipMeanZeroProbability, ZipTheta, ZipTotalMeanZeroProbability,
    ZipTotalMeanZeroProbabilityTheta,
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
    assert_gradient_matches_finite_difference::<_, 3>(
        &TweedieMeanCvPower::new(),
        1.4,
        [0.2, -0.3, 0.0],
    );
    assert_gradient_matches_finite_difference::<_, 3>(
        &StudentTMuSigmaTau::new(),
        0.4,
        [0.1, -0.2, 3.0_f64.ln()],
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

    let zinb_total = ZinbTotalMeanSizeZeroProbability::new();
    assert_gradient_matches_finite_difference::<_, 3>(
        &zinb_total,
        0.0,
        [0.84_f64.ln(), 2.0_f64.ln(), -1.0],
    );
    assert_gradient_matches_finite_difference::<_, 3>(
        &zinb_total,
        3.0,
        [0.84_f64.ln(), 2.0_f64.ln(), -1.0],
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

    let zaga_total = ZagaTotalMeanCvZeroProbability::new();
    assert_gradient_matches_finite_difference::<_, 3>(
        &zaga_total,
        0.0,
        [0.84_f64.ln(), 0.8_f64.ln(), -1.0],
    );
    assert_gradient_matches_finite_difference::<_, 3>(
        &zaga_total,
        1.4,
        [0.84_f64.ln(), 0.8_f64.ln(), -1.0],
    );

    let beinf = BeinfMuSigmaNuTau::new();
    assert_gradient_matches_finite_difference::<_, 4>(&beinf, 0.0, [0.1, -1.0, -2.0, -2.2]);
    assert_gradient_matches_finite_difference::<_, 4>(&beinf, 0.4, [0.1, -1.0, -2.0, -2.2]);
    assert_gradient_matches_finite_difference::<_, 4>(&beinf, 1.0, [0.1, -1.0, -2.0, -2.2]);
}

#[test]
fn total_mean_parameterizations_match_component_equivalents() {
    let zip_component = ZipMeanZeroProbability::new();
    let zip_total = ZipTotalMeanZeroProbability::new();
    let zip_component_theta = ZipTheta {
        mu: 2.0,
        sigma: 0.3,
    };
    let zip_total_theta = ZipTotalMeanZeroProbabilityTheta {
        total_mean: (1.0 - zip_component_theta.sigma) * zip_component_theta.mu,
        zero_probability: zip_component_theta.sigma,
    };
    assert_close(
        zip_total.nll(3.0, zip_total_theta),
        zip_component.nll(3.0, zip_component_theta),
        0.0,
        1.0e-12,
    );
    assert_close(
        zip_total.cdf(3.0, zip_total_theta),
        zip_component.cdf(3.0, zip_component_theta),
        0.0,
        1.0e-12,
    );

    let zaga_component = ZagaMeanSigmaZeroProbability::new();
    let zaga_total = ZagaTotalMeanCvZeroProbability::new();
    let zaga_component_theta = ZagaTheta {
        mu: 2.0,
        sigma: 0.6,
        nu: 0.25,
    };
    let zaga_total_theta = ZagaTotalMeanCvZeroProbabilityTheta {
        total_mean: (1.0 - zaga_component_theta.nu) * zaga_component_theta.mu,
        cv: zaga_component_theta.sigma,
        zero_probability: zaga_component_theta.nu,
    };
    assert_close(
        zaga_total.nll(1.4, zaga_total_theta),
        zaga_component.nll(1.4, zaga_component_theta),
        0.0,
        1.0e-12,
    );
    assert_close(
        zaga_total.cdf(1.4, zaga_total_theta),
        zaga_component.cdf(1.4, zaga_component_theta),
        0.0,
        1.0e-12,
    );

    let zinb_component = ZinbMeanSizeZeroProbability::new();
    let zinb_total = ZinbTotalMeanSizeZeroProbability::new();
    let zinb_component_theta = ZinbTheta {
        mu: 2.0,
        shape: 1.5,
        nu: 0.25,
    };
    let zinb_total_theta = ZinbTotalMeanSizeZeroProbabilityTheta {
        total_mean: (1.0 - zinb_component_theta.nu) * zinb_component_theta.mu,
        size: zinb_component_theta.shape,
        zero_probability: zinb_component_theta.nu,
    };
    assert_close(
        zinb_total.nll(3.0, zinb_total_theta),
        zinb_component.nll(3.0, zinb_component_theta),
        0.0,
        1.0e-12,
    );
    assert_close(
        zinb_total.cdf(3.0, zinb_total_theta),
        zinb_component.cdf(3.0, zinb_component_theta),
        0.0,
        1.0e-12,
    );
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
                    mean: 1.0,
                    dispersion: 1.0,
                    power: 1.5,
                },
            )
            .is_infinite()
    );
    assert!(
        TweedieMeanCvPower::new()
            .nll(
                1.0,
                TweedieMeanCvPowerTheta {
                    mean: 1.0,
                    cv: 0.0,
                    power: 1.5,
                },
            )
            .is_infinite()
    );
    assert!(
        TweedieMeanDispersionPower::new()
            .nll(
                1.0,
                TweedieTheta {
                    mean: 1.0,
                    dispersion: 1.0,
                    power: 2.0,
                },
            )
            .is_infinite()
    );
    assert!(
        StudentTMuSigmaTau::new()
            .nll(
                1.0,
                StudentTMuSigmaTauTheta {
                    mu: 0.0,
                    sigma: 1.0,
                    tau: 2.0,
                },
            )
            .is_infinite()
    );
    assert!(
        StudentTMuSigmaTau::new()
            .cdf(
                1.0,
                StudentTMuSigmaTauTheta {
                    mu: 0.0,
                    sigma: 0.0,
                    tau: 5.0,
                },
            )
            .is_nan()
    );
}

#[test]
fn tweedie_mean_cv_matches_mean_dispersion_equivalent() {
    let mean_cv = TweedieMeanCvPower::new();
    let mean_dispersion = TweedieMeanDispersionPower::new();
    let theta = TweedieMeanCvPowerTheta {
        mean: 1.4,
        cv: 0.7,
        power: 1.5,
    };
    let canonical: TweedieTheta = theta.into();

    assert_close(
        mean_cv.nll(1.1, theta),
        mean_dispersion.nll(1.1, canonical),
        0.0,
        1.0e-12,
    );
    assert_close(
        mean_cv.cdf(1.1, theta),
        mean_dispersion.cdf(1.1, canonical),
        0.0,
        1.0e-12,
    );
    assert_close(
        mean_cv.quantile(0.7, theta),
        mean_dispersion.quantile(0.7, canonical),
        0.0,
        1.0e-10,
    );
}

#[test]
fn student_t_dynamic_matches_fixed_df_equivalent() {
    let dynamic = StudentTMuSigmaTau::new();
    let fixed = StudentTMuSigma::try_new(5.0).unwrap();
    let dynamic_theta = StudentTMuSigmaTauTheta {
        mu: 0.2,
        sigma: 1.3,
        tau: 5.0,
    };
    let fixed_theta = StudentTTheta {
        mu: dynamic_theta.mu,
        sigma: dynamic_theta.sigma,
    };

    assert_close(
        dynamic.nll(0.7, dynamic_theta),
        fixed.nll(0.7, fixed_theta),
        0.0,
        1.0e-12,
    );
    assert_close(
        dynamic.cdf(0.7, dynamic_theta),
        fixed.cdf(0.7, fixed_theta),
        0.0,
        1.0e-12,
    );
    assert_close(
        dynamic.quantile(0.7, dynamic_theta),
        fixed.quantile(0.7, fixed_theta),
        0.0,
        1.0e-10,
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
