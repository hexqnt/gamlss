use gamlss_family::*;

use common::assert_nll_eta_matches_theta;

#[path = "common/helpers.rs"]
mod common;

#[test]
fn nll_eta_matches_nll_after_theta_transform_for_builtin_families() {
    assert_nll_eta_matches_theta::<_, 1>(&BernoulliProbability::new(), 1.0, [0.4]);
    assert_nll_eta_matches_theta::<_, 1>(&PoissonMean::new(), 3.0, [1.2]);
    assert_nll_eta_matches_theta::<_, 1>(&ExponentialRate::new(), 1.4, [0.3]);
    assert_nll_eta_matches_theta::<_, 1>(&ExponentialMean::new(), 1.4, [0.3]);

    assert_nll_eta_matches_theta::<_, 2>(&NormalMuSigma::new(), 0.4, [0.2, -0.3]);
    assert_nll_eta_matches_theta::<_, 2>(&GumbelMuSigma::new(), 0.4, [0.2, -0.3]);
    assert_nll_eta_matches_theta::<_, 2>(&LaplaceMuSigma::new(), 0.4, [0.2, -0.3]);
    assert_nll_eta_matches_theta::<_, 2>(&LogisticMuSigma::new(), 0.4, [0.2, -0.3]);
    assert_nll_eta_matches_theta::<_, 2>(&StudentTMuSigma::default(), 0.4, [0.2, -0.3]);

    assert_nll_eta_matches_theta::<_, 2>(&BetaMeanPrecision::new(), 0.4, [0.2, 1.1]);
    assert_nll_eta_matches_theta::<_, 2>(&GammaShapeRate::new(), 1.4, [0.2, -0.3]);
    assert_nll_eta_matches_theta::<_, 2>(&GammaMeanShape::new(), 1.4, [0.2, -0.3]);
    assert_nll_eta_matches_theta::<_, 2>(&GammaMeanCv::new(), 1.4, [0.2, -0.3]);
    assert_nll_eta_matches_theta::<_, 2>(&InverseGaussianMuShape::new(), 1.4, [0.2, -0.3]);
    assert_nll_eta_matches_theta::<_, 2>(&InverseGaussianMeanCv::new(), 1.4, [0.2, -0.3]);
    assert_nll_eta_matches_theta::<_, 2>(&LogNormalLogLocationLogSd::new(), 1.4, [0.2, -0.3]);
    assert_nll_eta_matches_theta::<_, 2>(&LogNormalMeanLogSd::new(), 1.4, [0.2, -0.3]);
    assert_nll_eta_matches_theta::<_, 2>(&LogNormalMeanCv::new(), 1.4, [0.2, -0.3]);
    assert_nll_eta_matches_theta::<_, 2>(&LogNormalMedianLogSd::new(), 1.4, [0.2, -0.3]);
    assert_nll_eta_matches_theta::<_, 2>(&LomaxShapeScale::new(), 1.4, [0.2, -0.3]);
    assert_nll_eta_matches_theta::<_, 2>(&NegativeBinomialMeanSize::new(), 3.0, [1.2, 0.4]);
    assert_nll_eta_matches_theta::<_, 2>(&NegativeBinomialMeanDispersion::new(), 3.0, [1.2, -1.0]);
    assert_nll_eta_matches_theta::<_, 2>(&WeibullScaleShape::new(), 1.4, [0.2, -0.3]);
    assert_nll_eta_matches_theta::<_, 2>(&WeibullMeanShape::new(), 1.4, [0.2, -0.3]);
    assert_nll_eta_matches_theta::<_, 2>(&ZipMeanZeroProbability::new(), 3.0, [1.2, -1.0]);
    assert_nll_eta_matches_theta::<_, 2>(&ZipTotalMeanZeroProbability::new(), 3.0, [1.2, -1.0]);

    assert_nll_eta_matches_theta::<_, 3>(
        &GeneralizedGammaScaleSigmaNu::new(),
        1.4,
        [0.2, -0.3, 0.1],
    );
    assert_nll_eta_matches_theta::<_, 3>(&GevMuSigmaShape::new(), 0.4, [0.2, -0.3, 0.1]);
    assert_nll_eta_matches_theta::<_, 3>(&PowerExponentialMuSigmaNu::new(), 0.4, [0.2, -0.3, 0.4]);
    assert_nll_eta_matches_theta::<_, 3>(&SkewNormalMuSigmaNu::new(), 0.4, [0.2, -0.3, 0.1]);
    assert_nll_eta_matches_theta::<_, 3>(&SkewNormalMeanSdNu::new(), 0.4, [0.2, -0.3, 0.1]);
    assert_nll_eta_matches_theta::<_, 3>(&StudentTMuSigmaTau::new(), 0.4, [0.2, -0.3, 1.2]);
    assert_nll_eta_matches_theta::<_, 3>(&StudentTMuSdTau::new(), 0.4, [0.2, -0.3, 1.2]);
    assert_nll_eta_matches_theta::<_, 3>(&TweedieMeanDispersionPower::new(), 1.4, [0.2, -0.3, 0.0]);
    assert_nll_eta_matches_theta::<_, 3>(&TweedieMeanCvPower::new(), 1.4, [0.2, -0.3, 0.0]);
    assert_nll_eta_matches_theta::<_, 3>(
        &ZagaMeanSigmaZeroProbability::new(),
        1.4,
        [0.2, -0.3, -1.0],
    );
    assert_nll_eta_matches_theta::<_, 3>(
        &ZagaTotalMeanCvZeroProbability::new(),
        1.4,
        [0.2, -0.3, -1.0],
    );
    assert_nll_eta_matches_theta::<_, 3>(
        &ZinbMeanSizeZeroProbability::new(),
        3.0,
        [1.2, 0.4, -1.0],
    );
    assert_nll_eta_matches_theta::<_, 3>(
        &ZinbTotalMeanSizeZeroProbability::new(),
        3.0,
        [1.2, 0.4, -1.0],
    );

    assert_nll_eta_matches_theta::<_, 4>(&BeinfMuSigmaNuTau::new(), 0.4, [0.2, -0.3, 0.1, -0.2]);
    assert_nll_eta_matches_theta::<_, 4>(&JohnsonSuMuSigmaNuTau::new(), 0.4, [0.2, -0.3, 0.1, 0.2]);
    assert_nll_eta_matches_theta::<_, 4>(&ShashMuSigmaNuTau::new(), 0.4, [0.2, -0.3, 0.1, 0.2]);
    assert_nll_eta_matches_theta::<_, 4>(
        &SkewStudentTMuSigmaNuTau::new(),
        0.4,
        [0.2, -0.3, 0.1, 1.2],
    );
    assert_nll_eta_matches_theta::<_, 4>(
        &SkewStudentTMeanSdNuTau::new(),
        0.4,
        [0.2, -0.3, 0.1, 1.2],
    );
}
