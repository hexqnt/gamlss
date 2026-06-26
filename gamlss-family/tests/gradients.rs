use gamlss_family::*;
use proptest::prelude::*;

use common::{
    assert_gradient_matches_finite_difference,
    assert_new_family_gradient_matches_finite_difference, proptest_config,
};

#[path = "common/helpers.rs"]
mod common;

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
}

#[test]
fn higher_parameter_continuous_family_gradients_match_finite_differences() {
    assert_new_family_gradient_matches_finite_difference::<_, 3>(
        &SkewNormalMuSigmaNu::new(),
        0.4,
        [0.1, -0.2, 0.3],
    );
    assert_new_family_gradient_matches_finite_difference::<_, 3>(
        &SkewNormalMeanSdNu::new(),
        0.4,
        [0.1, -0.2, 0.3],
    );
    assert_new_family_gradient_matches_finite_difference::<_, 3>(
        &PowerExponentialMuSigmaNu::new(),
        0.4,
        [0.1, -0.2, 2.0_f64.ln()],
    );
    assert_new_family_gradient_matches_finite_difference::<_, 4>(
        &SkewStudentTMuSigmaNuTau::new(),
        0.4,
        [0.1, -0.2, 0.3, 5.0_f64.ln()],
    );
    assert_new_family_gradient_matches_finite_difference::<_, 4>(
        &SkewStudentTMeanSdNuTau::new(),
        0.4,
        [0.1, -0.2, 0.3, 5.0_f64.ln()],
    );
    assert_new_family_gradient_matches_finite_difference::<_, 4>(
        &ShashMuSigmaNuTau::new(),
        0.4,
        [0.1, -0.2, 0.5_f64.ln(), 0.8_f64.ln()],
    );
    assert_new_family_gradient_matches_finite_difference::<_, 4>(
        &JohnsonSuMuSigmaNuTau::new(),
        0.4,
        [0.1, -0.2, 0.3, 1.2_f64.ln()],
    );
    assert_new_family_gradient_matches_finite_difference::<_, 3>(
        &GeneralizedGammaScaleSigmaNu::new(),
        1.4,
        [0.1, -0.2, 0.5],
    );
    assert_new_family_gradient_matches_finite_difference::<_, 3>(
        &GeneralizedGammaScaleSigmaNu::new(),
        1.4,
        [0.1, -0.2, 0.0],
    );
    assert_new_family_gradient_matches_finite_difference::<_, 3>(
        &GevMuSigmaShape::new(),
        0.4,
        [0.1, -0.2, 0.1],
    );
    assert_new_family_gradient_matches_finite_difference::<_, 3>(
        &GevMuSigmaShape::new(),
        0.4,
        [0.1, -0.2, 0.0],
    );
    assert_new_family_gradient_matches_finite_difference::<_, 3>(
        &TweedieMeanDispersionPower::new(),
        1.4,
        [0.2, -0.3, 0.0],
    );
    assert_new_family_gradient_matches_finite_difference::<_, 3>(
        &TweedieMeanCvPower::new(),
        1.4,
        [0.2, -0.3, 0.0],
    );
    assert_new_family_gradient_matches_finite_difference::<_, 3>(
        &StudentTMuSigmaTau::new(),
        0.4,
        [0.1, -0.2, 3.0_f64.ln()],
    );
}

#[test]
fn gradient_contracts_hold_near_representative_boundaries() {
    for eta in [[-6.0], [-2.0], [0.0], [2.0], [6.0]] {
        assert_gradient_matches_finite_difference::<_, 1>(&BernoulliProbability::new(), 0.0, eta);
        assert_gradient_matches_finite_difference::<_, 1>(&BernoulliProbability::new(), 1.0, eta);
        assert_gradient_matches_finite_difference::<_, 1>(&PoissonMean::new(), 0.0, eta);
        assert_gradient_matches_finite_difference::<_, 1>(&ExponentialRate::new(), 1.0e-6, eta);
        assert_gradient_matches_finite_difference::<_, 1>(&ExponentialRate::new(), 30.0, eta);
    }

    for eta in [
        [-2.0, -6.0],
        [2.0, -6.0],
        [-2.0, 4.0],
        [2.0, 4.0],
        [0.0, 0.0],
    ] {
        assert_gradient_matches_finite_difference::<_, 2>(&NormalMuSigma::new(), eta[0], eta);
        assert_gradient_matches_finite_difference::<_, 2>(&GumbelMuSigma::new(), eta[0], eta);
        assert_gradient_matches_finite_difference::<_, 2>(&LaplaceMuSigma::new(), eta[0], eta);
        assert_gradient_matches_finite_difference::<_, 2>(&LogisticMuSigma::new(), eta[0], eta);
        assert_gradient_matches_finite_difference::<_, 2>(&StudentTMuSigma::default(), eta[0], eta);
    }

    for eta in [
        [-6.0, -2.0],
        [6.0, -2.0],
        [-2.0, 4.0],
        [2.0, 4.0],
        [0.0, 0.0],
    ] {
        assert_gradient_matches_finite_difference::<_, 2>(&BetaMeanPrecision::new(), 0.5, eta);
    }

    for eta in [[-4.0, -4.0], [0.0, 0.0], [4.0, 4.0]] {
        assert_gradient_matches_finite_difference::<_, 2>(&GammaShapeRate::new(), 1.0e-6, eta);
        assert_gradient_matches_finite_difference::<_, 2>(&GammaShapeRate::new(), 30.0, eta);
        assert_gradient_matches_finite_difference::<_, 2>(
            &NegativeBinomialMeanSize::new(),
            0.0,
            eta,
        );
        assert_gradient_matches_finite_difference::<_, 2>(
            &NegativeBinomialMeanSize::new(),
            40.0,
            eta,
        );
    }
}
