use gamlss_core::Family;
use gamlss_family::{
    BetaMeanPrecision, ExponentialMean, ExponentialMeanTheta, ExponentialRate,
    ExponentialRateTheta, GammaMeanCv, GammaMeanCvTheta, GammaMeanShape, GammaMeanShapeTheta,
    GammaShapeRate, GammaShapeRateTheta, GeneralizedGammaScaleSigmaNu, GeneralizedGammaTheta,
    InverseGaussianMeanCv, InverseGaussianMeanCvTheta, LogNormalLogLocationLogSd,
    LogNormalLogLocationLogSdTheta, LogNormalMeanCv, LogNormalMeanCvTheta, LogNormalMeanLogSd,
    LogNormalMeanLogSdTheta, LogNormalMedianLogSd, LogNormalMedianLogSdTheta,
    NegativeBinomialMeanDispersion, NegativeBinomialMeanDispersionTheta, NegativeBinomialMeanSize,
    NegativeBinomialTheta, ShashEta, ShashTheta, SinhArcsinhEta, SinhArcsinhMuSigmaNuTau,
    SinhArcsinhTheta, SkewNormalMeanSdNu, SkewNormalMeanSdTheta, SkewStudentTMeanSdNuTau,
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
    DirichletMultinomialFixedTrials, DirichletMultinomialMeanPrecisionTheta,
    DirichletMultinomialVaryingTrials, FixedLogRatioCholesky, FixedLowerTriangular,
    FixedPartialCorrelations, LogisticNormalAlrCholeskyDefault, LogisticNormalAlrCholeskyTheta,
    MultinomialEta, MultinomialFixedTrials, MultinomialTheta, MultinomialVaryingTrials,
    MvLogNormalCholeskyDefault, MvLogNormalCholeskyTheta, MvNormalMeanStdPartialCorrDefault,
    MvNormalMeanStdPartialCorrEta, MvPowerExponentialCholeskyDefault,
    MvPowerExponentialCholeskyTheta, MvShashMuSigmaNuTauPartialCorrDefault,
    MvShashMuSigmaNuTauPartialCorrTheta, MvSinhArcsinhMuSigmaNuTauPartialCorrEta,
    MvSinhArcsinhMuSigmaNuTauPartialCorrTheta, MvSkewNormalCholeskyDefault,
    MvSkewNormalCholeskyTheta, MvStudentTCholeskyDefault, MvStudentTCholeskyEta,
    MvStudentTCholeskyTheta, MvStudentTMeanStdPartialCorrDefault,
    MvStudentTMeanStdPartialCorrTheta,
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
    let eta: SinhArcsinhEta = ShashEta {
        mu: 0.0,
        sigma: 0.0,
        nu: 0.0,
        tau: 0.0,
    };
    let theta: SinhArcsinhTheta = ShashTheta {
        mu: 0.0,
        sigma: 1.0,
        nu: 1.0,
        tau: 1.0,
    };
    assert!(eta.nu.is_finite());
    assert!(finite_nll(SinhArcsinhMuSigmaNuTau::new(), 0.25, theta));

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
fn new_univariate_families_are_root_and_prelude_reexports() {
    use gamlss_family::prelude::*;

    assert!(finite_nll(
        GeometricMean::new(),
        2.0,
        GeometricTheta { mean: 1.5 }
    ));
    assert!(finite_nll(
        RayleighScale::new(),
        1.2,
        RayleighTheta { scale: 1.0 }
    ));
    assert!(finite_nll(
        LogLogisticScaleShape::new(),
        1.2,
        LogLogisticTheta {
            scale: 1.0,
            shape: 2.0,
        }
    ));
    assert!(finite_nll(
        ChiDegreesOfFreedom::new(),
        1.2,
        ChiTheta {
            degrees_of_freedom: 3.0,
        }
    ));
    assert!(finite_nll(
        ChiSquaredDegreesOfFreedom::new(),
        1.2,
        ChiSquaredTheta {
            degrees_of_freedom: 3.0,
        }
    ));
    assert!(finite_nll(
        GeneralizedParetoScaleShape::new(),
        1.2,
        GeneralizedParetoTheta {
            scale: 1.0,
            shape: 0.1,
        }
    ));

    let fixed = BinomialFixedTrialsProbability::try_new(10).unwrap();
    assert!(
        fixed
            .nll(4.0, &BinomialTheta { probability: 0.4 }, &mut ())
            .is_finite()
    );
    assert!(
        BinomialVaryingTrialsProbability::new()
            .nll([4.0, 10.0], &BinomialTheta { probability: 0.4 }, &mut (),)
            .is_finite()
    );
    assert!(
        BetaBinomialMeanPrecision::new()
            .nll(
                [4.0, 10.0],
                &BetaBinomialTheta {
                    probability: 0.4,
                    precision: 8.0,
                },
                &mut (),
            )
            .is_finite()
    );
    let categorical = Categorical::<3>::new();
    assert!(
        categorical
            .nll(
                1.0,
                &CategoricalTheta {
                    probabilities: [0.2, 0.3, 0.5],
                },
                &mut (),
            )
            .is_finite()
    );
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
    let _ = gamlss_family::univariate::geometric::GeometricMean::new();
    let _ = gamlss_family::univariate::rayleigh::RayleighScale::new();
    let _ = gamlss_family::univariate::binomial::BinomialVaryingTrialsProbability::new();
    let _ = gamlss_family::univariate::categorical::Categorical::<3>::new();
    let _ = gamlss_family::univariate::log_logistic::LogLogisticScaleShape::new();
    let _ = gamlss_family::univariate::chi_squared::ChiSquaredDegreesOfFreedom::new();
    let _ = gamlss_family::univariate::chi::ChiDegreesOfFreedom::new();
    let _ = gamlss_family::univariate::beta_binomial::BetaBinomialMeanPrecision::new();
    let _ = gamlss_family::univariate::generalized_pareto::GeneralizedParetoScaleShape::new();
    let _ = gamlss_family::univariate::sinh_arcsinh::SinhArcsinhMuSigmaNuTau::new();
}

#[cfg(feature = "multivariate")]
#[test]
fn multivariate_generic_aliases_construct_without_dimension_specific_types() {
    let multinomial = MultinomialFixedTrials::<3>::new(10);
    let multinomial_eta = MultinomialEta::new([0.2, -0.1, 0.0]);
    let multinomial_theta = multinomial.theta(&multinomial_eta, &mut ());
    assert!(
        multinomial
            .nll([2.0, 3.0, 5.0], &multinomial_theta, &mut ())
            .is_finite()
    );
    assert!(
        MultinomialVaryingTrials::<3>::new()
            .nll(
                [2.0, 3.0, 5.0],
                &MultinomialTheta::try_new([0.2, 0.3, 0.5]).unwrap(),
                &mut ()
            )
            .is_finite()
    );
    let dirichlet_multinomial_theta =
        DirichletMultinomialMeanPrecisionTheta::try_new([0.2, 0.3, 0.5], 8.0).unwrap();
    assert!(
        DirichletMultinomialFixedTrials::<3>::new(10)
            .nll([2.0, 3.0, 5.0], &dirichlet_multinomial_theta, &mut ())
            .is_finite()
    );
    assert!(
        DirichletMultinomialVaryingTrials::<3>::new()
            .nll([1.0, 2.0, 3.0], &dirichlet_multinomial_theta, &mut ())
            .is_finite()
    );

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
    let student_drd = MvStudentTMeanStdPartialCorrDefault::<3>::new();
    let student_drd_theta = MvStudentTMeanStdPartialCorrTheta::try_new(
        [0.0; 3],
        [1.0; 3],
        FixedPartialCorrelations::try_new(vec![0.1, -0.2, 0.3]).unwrap(),
        8.0,
    )
    .unwrap();
    assert!(
        student_drd
            .nll([0.1, -0.2, 0.3], &student_drd_theta, &mut ())
            .is_finite()
    );

    let log_normal = MvLogNormalCholeskyDefault::<2>::new();
    let log_normal_theta = MvLogNormalCholeskyTheta::try_new(
        [0.0; 2],
        FixedLowerTriangular::from_lower_rows([[1.0, 0.0], [0.2, 0.8]]),
    )
    .unwrap();
    assert!(
        log_normal
            .nll([1.0, 2.0], &log_normal_theta, &mut ())
            .is_finite()
    );

    let logistic_normal = LogisticNormalAlrCholeskyDefault::<3>::new();
    let logistic_normal_theta = LogisticNormalAlrCholeskyTheta::try_new(
        [0.0; 3],
        FixedLogRatioCholesky::try_from_packed(&[1.0, 0.2, 0.8]).unwrap(),
    )
    .unwrap();
    assert!(
        logistic_normal
            .nll([0.2, 0.3, 0.5], &logistic_normal_theta, &mut ())
            .is_finite()
    );

    let power_exponential = MvPowerExponentialCholeskyDefault::<2>::new();
    let power_exponential_theta = MvPowerExponentialCholeskyTheta::try_new(
        [0.0; 2],
        FixedLowerTriangular::from_lower_rows([[1.0, 0.0], [0.2, 0.8]]),
        1.4,
    )
    .unwrap();
    assert!(
        power_exponential
            .nll([0.1, -0.2], &power_exponential_theta, &mut ())
            .is_finite()
    );

    let shash = MvShashMuSigmaNuTauPartialCorrDefault::<2>::new();
    let shash_theta = MvShashMuSigmaNuTauPartialCorrTheta::try_new(
        [0.0; 2],
        [1.0; 2],
        [1.2, 0.8],
        [0.9, 1.1],
        FixedPartialCorrelations::zeros(),
    )
    .unwrap();
    let _: MvSinhArcsinhMuSigmaNuTauPartialCorrTheta<2> = shash_theta;
    let _: MvSinhArcsinhMuSigmaNuTauPartialCorrEta<2> =
        gamlss_family::MvShashMuSigmaNuTauPartialCorrEta::new(
            [0.0; 2],
            [0.0; 2],
            [0.0; 2],
            [0.0; 2],
            FixedPartialCorrelations::zeros(),
        );
    assert!(shash.nll([0.1, -0.2], &shash_theta, &mut ()).is_finite());

    let skew_normal = MvSkewNormalCholeskyDefault::<2>::new();
    let skew_normal_theta = MvSkewNormalCholeskyTheta::try_new(
        [0.0; 2],
        FixedLowerTriangular::from_lower_rows([[1.0, 0.0], [0.2, 0.8]]),
        [0.5, -0.3],
    )
    .unwrap();
    assert!(
        skew_normal
            .nll([0.1, -0.2], &skew_normal_theta, &mut ())
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
    let _ = DirichletMultinomialFixedTrials::<4>::new(10);
    let _ = DirichletMultinomialVaryingTrials::<4>::new();
    let _ = LogisticNormalAlrCholeskyDefault::<4>::new();
    let _ = MultinomialFixedTrials::<4>::new(10);
    let _ = MultinomialVaryingTrials::<4>::new();
    let _ = MvLogNormalCholeskyDefault::<4>::new();
    let _ = MvNormalMeanStdPartialCorrDefault::<4>::new();
    let _ = MvPoissonCommonShockDefault::<4>::new();
    let _ = MvPowerExponentialCholeskyDefault::<4>::new();
    let _ = MvPowerExponentialMeanStdPartialCorrDefault::<4>::new();
    let _ = MvShashMuSigmaNuTauPartialCorrDefault::<4>::new();
    let _ = MvSkewNormalCholeskyDefault::<4>::new();
    let _ = MvSkewNormalLocationKernelStdPartialCorrDefault::<4>::new();
    let _ = MvSkewStudentTFixedTauCholeskyDefault::<4>::new(5.0);
    let _ = MvStudentTCholeskyDefault::<4>::new();
    let _ = MvStudentTMeanStdPartialCorrDefault::<4>::new();
    let _ = FixedPartialCorrelations::<4>::zeros();
    let _ = FixedLogRatioCholesky::<4>::zeros();
}

#[cfg(feature = "multivariate")]
#[test]
fn multivariate_matrix_module_owns_triangular_storage() {
    use gamlss_family::multivariate::matrix::{FixedLowerTriangular, PackedLowerTriangular};

    let _ = FixedLowerTriangular::<2>::zeros();
    let packed = PackedLowerTriangular::try_new(2, vec![1.0, 0.0, 1.0]).unwrap();
    assert_eq!(packed.dimension(), 2);
}

#[cfg(feature = "multivariate")]
#[test]
fn multivariate_multinomial_module_exposes_count_families() {
    use gamlss_family::multivariate::multinomial::{
        MultinomialFixedTrials, MultinomialVaryingTrials,
    };

    let _ = MultinomialFixedTrials::<3>::new(5);
    let _ = MultinomialVaryingTrials::<3>::new();
}

#[cfg(feature = "multivariate")]
#[test]
fn new_multivariate_modules_expose_parameterized_families() {
    let _ = gamlss_family::multivariate::dirichlet_multinomial::DirichletMultinomialFixedTrials::<
        3,
    >::new(5);
    let _ = gamlss_family::multivariate::log_normal::MvLogNormalCholeskyDefault::<2>::new();
    let _ =
        gamlss_family::multivariate::logistic_normal::LogisticNormalAlrCholeskyDefault::<3>::new();
    let _ =
        gamlss_family::multivariate::power_exponential::MvPowerExponentialCholeskyDefault::<2>::new(
        );
    let _ =
        gamlss_family::multivariate::power_exponential::MvPowerExponentialMeanStdPartialCorrDefault::<
            2,
        >::new();
    let _ =
        gamlss_family::multivariate::poisson_common_shock::MvPoissonCommonShockDefault::<2>::new();
    let _ =
        gamlss_family::multivariate::shash::MvSinhArcsinhMuSigmaNuTauPartialCorrDefault::<2>::new();
    let _ =
        gamlss_family::multivariate::sinh_arcsinh::MvShashMuSigmaNuTauPartialCorrDefault::<2>::new(
        );
    let _ = gamlss_family::multivariate::skew_normal::MvSkewNormalCholeskyDefault::<2>::new();
    let _ =
        gamlss_family::multivariate::skew_normal::MvSkewNormalLocationKernelStdPartialCorrDefault::<
            2,
        >::new();
    let _ = gamlss_family::multivariate::skew_student_t::MvSkewStudentTFixedTauCholeskyDefault::<2>::new(5.0);
    let _ = gamlss_family::multivariate::student_t::MvStudentTMeanStdPartialCorrDefault::<2>::new();
}
