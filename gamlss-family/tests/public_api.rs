use gamlss_core::Family;
use gamlss_family::{
    BetaMeanPrecision, ExponentialMean, ExponentialMeanTheta, ExponentialRate,
    ExponentialRateTheta, GammaMeanCv, GammaMeanCvTheta, GammaMeanShape, GammaMeanShapeTheta,
    GammaShapeRate, GammaShapeRateTheta, GeneralizedGammaScaleSigmaNu, GeneralizedGammaTheta,
    InverseGaussianMeanCv, InverseGaussianMeanCvTheta, LogNormalLogLocationLogSd,
    LogNormalLogLocationLogSdTheta, LogNormalMeanCv, LogNormalMeanCvTheta, LogNormalMeanLogSd,
    LogNormalMeanLogSdTheta, LogNormalMedianLogSd, LogNormalMedianLogSdTheta,
    NegativeBinomialMeanDispersion, NegativeBinomialMeanDispersionTheta, NegativeBinomialMeanSize,
    NegativeBinomialTheta, SkewNormalMeanSdNu, SkewNormalMeanSdTheta, SkewStudentTMeanSdNuTau,
    SkewStudentTMeanSdTheta, StudentTMuSdTau, StudentTMuSdTauTheta, StudentTMuSigmaTau,
    StudentTMuSigmaTauTheta, TweedieMeanCvPower, TweedieMeanCvPowerTheta,
    TweedieMeanDispersionPower, TweedieTheta, WeibullMeanShape, WeibullMeanShapeTheta,
    WeibullScaleShape, WeibullScaleShapeTheta, ZagaTotalMeanCvZeroProbability,
    ZagaTotalMeanCvZeroProbabilityTheta, ZinbTotalMeanSizeZeroProbability,
    ZinbTotalMeanSizeZeroProbabilityTheta, ZipTotalMeanZeroProbability,
    ZipTotalMeanZeroProbabilityTheta,
};

#[cfg(feature = "multivariate")]
use gamlss_family::{
    DirichletMeanPrecision, DirichletMeanPrecisionEta, DirichletMeanPrecisionTheta,
    FixedLowerTriangular, FixedPartialCorrelations, MvNormalMeanStdPartialCorrDefault,
    MvNormalMeanStdPartialCorrEta, MvStudentTCholeskyDefault, MvStudentTCholeskyEta,
    MvStudentTCholeskyTheta,
};

#[allow(clippy::needless_pass_by_value)]
fn finite_nll<F>(family: F, y: f64, theta: F::Theta) -> bool
where
    F: for<'obs> Family<Observation<'obs> = f64>,
{
    let mut workspace = family.workspace();
    family.nll(y, &theta, &mut workspace).is_finite()
}

#[test]
fn mixture_prelude_exposes_read_only_gradient_carriers() {
    use gamlss_family::prelude::{Mixture, MixtureEta, NormalEta, NormalMuSigma};

    let family = Mixture::<_, 2>::try_new(NormalMuSigma::new()).unwrap();
    let eta = MixtureEta::new(
        [0.2, 0.0],
        [
            NormalEta {
                mu: -0.5,
                sigma: 0.0,
            },
            NormalEta {
                mu: 0.5,
                sigma: 0.0,
            },
        ],
    );
    let (_, gradient) = family.nll_and_gradient_eta(0.1, &eta, &mut family.workspace());

    assert_eq!(gradient.logits().len(), 2);
    assert_eq!(gradient.components().len(), 2);
    assert!(gradient.components()[0].responsibility().is_finite());
    assert!(
        gradient.components()[0]
            .conditional_gradient()
            .mu
            .is_finite()
    );
}

#[test]
fn semantic_parameterization_aliases_construct_without_type_annotations() {
    assert!(finite_nll(
        GammaMeanCv::new(),
        1.2,
        GammaMeanCvTheta { mean: 1.0, cv: 0.5 }
    ));
    assert!(finite_nll(
        GammaMeanShape::new(),
        1.2,
        GammaMeanShapeTheta {
            mean: 1.0,
            shape: 4.0,
        }
    ));
    assert!(finite_nll(
        GammaShapeRate::new(),
        1.2,
        GammaShapeRateTheta {
            shape: 4.0,
            rate: 4.0,
        }
    ));

    assert!(finite_nll(
        ExponentialMean::new(),
        1.2,
        ExponentialMeanTheta { mean: 1.0 }
    ));
    assert!(finite_nll(
        ExponentialRate::new(),
        1.2,
        ExponentialRateTheta { rate: 1.0 }
    ));

    assert!(finite_nll(
        LogNormalMeanLogSd::new(),
        1.2,
        LogNormalMeanLogSdTheta {
            mean: 1.0,
            log_sd: 0.5,
        }
    ));
    assert!(finite_nll(
        LogNormalMeanCv::new(),
        1.2,
        LogNormalMeanCvTheta { mean: 1.0, cv: 0.5 },
    ));
    assert!(finite_nll(
        LogNormalMedianLogSd::new(),
        1.2,
        LogNormalMedianLogSdTheta {
            median: 1.0,
            log_sd: 0.5,
        }
    ));
    assert!(finite_nll(
        LogNormalLogLocationLogSd::new(),
        1.2,
        LogNormalLogLocationLogSdTheta {
            log_location: 0.0,
            log_sd: 0.5,
        }
    ));

    assert!(finite_nll(
        WeibullMeanShape::new(),
        1.2,
        WeibullMeanShapeTheta {
            mean: 1.0,
            shape: 1.5,
        }
    ));
    assert!(finite_nll(
        WeibullScaleShape::new(),
        1.2,
        WeibullScaleShapeTheta {
            scale: 1.0,
            shape: 1.5,
        }
    ));
    assert!(finite_nll(
        TweedieMeanDispersionPower::new(),
        1.2,
        TweedieTheta {
            mean: 1.0,
            dispersion: 0.5,
            power: 1.5,
        }
    ));
    assert!(finite_nll(
        TweedieMeanCvPower::new(),
        1.2,
        TweedieMeanCvPowerTheta {
            mean: 1.0,
            cv: 0.5,
            power: 1.5,
        }
    ));
    assert!(finite_nll(
        GeneralizedGammaScaleSigmaNu::new(),
        1.2,
        GeneralizedGammaTheta {
            scale: 1.0,
            sigma: 0.5,
            nu: 1.0,
        }
    ));
    assert!(finite_nll(
        StudentTMuSigmaTau::new(),
        1.2,
        StudentTMuSigmaTauTheta {
            mu: 0.0,
            sigma: 1.0,
            tau: 5.0,
        }
    ));
    assert!(finite_nll(
        StudentTMuSdTau::new(),
        1.2,
        StudentTMuSdTauTheta {
            mu: 0.0,
            sigma: 1.0,
            tau: 5.0,
        }
    ));
    assert!(finite_nll(
        SkewNormalMeanSdNu::new(),
        1.2,
        SkewNormalMeanSdTheta {
            mean: 0.0,
            sigma: 1.0,
            nu: 0.5,
        }
    ));
    assert!(finite_nll(
        SkewStudentTMeanSdNuTau::new(),
        1.2,
        SkewStudentTMeanSdTheta {
            mean: 0.0,
            sigma: 1.0,
            nu: 0.5,
            tau: 5.0,
        }
    ));
}

#[test]
fn existing_mean_style_aliases_construct_without_type_annotations() {
    assert!(finite_nll(
        BetaMeanPrecision::new(),
        0.4,
        gamlss_family::BetaTheta {
            mu: 0.4,
            precision: 5.0,
        }
    ));
    assert!(finite_nll(
        NegativeBinomialMeanSize::new(),
        2.0,
        NegativeBinomialTheta {
            mu: 3.0,
            shape: 4.0,
        }
    ));
    assert!(finite_nll(
        NegativeBinomialMeanDispersion::new(),
        2.0,
        NegativeBinomialMeanDispersionTheta {
            mean: 3.0,
            dispersion: 0.25,
        }
    ));
    assert!(finite_nll(
        InverseGaussianMeanCv::new(),
        1.2,
        InverseGaussianMeanCvTheta { mean: 1.0, cv: 0.5 }
    ));
    assert!(finite_nll(
        ZipTotalMeanZeroProbability::new(),
        2.0,
        ZipTotalMeanZeroProbabilityTheta {
            total_mean: 2.0,
            zero_probability: 0.2,
        }
    ));
    assert!(finite_nll(
        ZagaTotalMeanCvZeroProbability::new(),
        1.2,
        ZagaTotalMeanCvZeroProbabilityTheta {
            total_mean: 1.0,
            cv: 0.5,
            zero_probability: 0.2,
        }
    ));
    assert!(finite_nll(
        ZinbTotalMeanSizeZeroProbability::new(),
        2.0,
        ZinbTotalMeanSizeZeroProbabilityTheta {
            total_mean: 2.0,
            size: 4.0,
            zero_probability: 0.2,
        }
    ));
}

#[test]
fn univariate_namespace_exposes_distribution_modules() {
    assert!(finite_nll(
        gamlss_family::univariate::normal::NormalMuSigma::new(),
        0.25,
        gamlss_family::univariate::normal::NormalTheta {
            mu: 0.0,
            sigma: 1.0,
        }
    ));
    assert!(finite_nll(
        gamlss_family::univariate::gamma::GammaMeanCv::new(),
        1.2,
        gamlss_family::univariate::gamma::GammaMeanCvTheta { mean: 1.0, cv: 0.5 }
    ));
}

#[cfg(feature = "multivariate")]
#[test]
fn multivariate_generic_aliases_construct_without_dimension_specific_types() {
    let dirichlet = DirichletMeanPrecision::<3>::new();
    let dirichlet_eta = DirichletMeanPrecisionEta::new([0.0, 0.5, 0.0], 2.0_f64.ln());
    let mut workspace = ();
    let theta = dirichlet.theta(&dirichlet_eta, &mut workspace);
    assert!(
        dirichlet
            .nll([0.2, 0.3, 0.5], &theta, &mut workspace)
            .is_finite()
    );
    assert!(
        dirichlet
            .nll(
                [0.2, 0.3, 0.5],
                &DirichletMeanPrecisionTheta::try_new([0.2, 0.3, 0.5], 5.0).unwrap(),
                &mut workspace
            )
            .is_finite()
    );

    let student = MvStudentTCholeskyDefault::<3>::new();
    let cholesky =
        FixedLowerTriangular::from_lower_rows([[1.0, 0.0, 0.0], [0.2, 1.1, 0.0], [0.1, -0.3, 0.9]]);
    let eta = MvStudentTCholeskyEta::new([0.0, 0.0, 0.0], cholesky, 1.0);
    let mut workspace = ();
    let theta = student.theta(&eta, &mut workspace);
    assert!(theta.mu().iter().all(|value| value.abs() <= f64::EPSILON));
    assert_eq!(theta.cholesky().get(1, 0), Some(0.2));
    assert!(theta.tau() > 2.0);
    assert!(
        student
            .nll([0.1, -0.2, 0.3], &theta, &mut workspace)
            .is_finite()
    );
    assert!(
        student
            .nll(
                [0.1, -0.2, 0.3],
                &MvStudentTCholeskyTheta::try_new([0.0, 0.0, 0.0], cholesky, 5.0).unwrap(),
                &mut workspace
            )
            .is_finite()
    );

    let drd = MvNormalMeanStdPartialCorrDefault::<3>::new();
    let eta = MvNormalMeanStdPartialCorrEta::new(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        FixedPartialCorrelations::try_new(vec![0.1, -0.2, 0.3]).unwrap(),
    );
    let mut workspace = ();
    let theta = drd.theta(&eta, &mut workspace);
    assert!(
        drd.nll([0.1, -0.2, 0.3], &theta, &mut workspace)
            .is_finite()
    );
}

#[cfg(feature = "multivariate")]
#[test]
fn multivariate_prelude_exposes_new_generic_families() {
    use gamlss_family::prelude::*;

    let _ = DirichletMeanPrecision::<4>::new();
    let _ = MvNormalMeanStdPartialCorrDefault::<4>::new();
    let _ = MvStudentTCholeskyDefault::<4>::new();
    let _ = FixedPartialCorrelations::<4>::zeros();
}

#[cfg(feature = "multivariate")]
#[test]
fn multivariate_matrix_module_owns_triangular_storage() {
    use gamlss_family::multivariate::matrix::{FixedLowerTriangular, PackedLowerTriangular};

    let _ = FixedLowerTriangular::<2>::zeros();
    let packed = PackedLowerTriangular::try_new(2, vec![1.0, 0.0, 1.0]).unwrap();
    assert_eq!(packed.dimension(), 2);
}
