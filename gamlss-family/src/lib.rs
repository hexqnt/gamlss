#![forbid(unsafe_code)]
//! Distributions, likelihood and NLL gradients for GAMLSS.

#[cfg(feature = "multivariate")]
pub use multivariate::IndependentVec;
#[cfg(feature = "multivariate")]
pub use multivariate::normal::{
    DynMvNormalCholesky, DynMvNormalCholeskyDefault, DynMvNormalCholeskyEta,
    DynMvNormalCholeskyTheta, MvNormalCholesky, MvNormalCholeskyDefault, MvNormalCholeskyEta,
    MvNormalCholeskyTheta, MvNormalMeanStdPartialCorr, MvNormalMeanStdPartialCorrDefault,
    MvNormalMeanStdPartialCorrEta, MvNormalMeanStdPartialCorrTheta,
};
#[cfg(feature = "multivariate")]
pub use multivariate::student_t::{
    MvStudentTCholesky, MvStudentTCholeskyDefault, MvStudentTCholeskyEta, MvStudentTCholeskyTheta,
    MvStudentTMeanStdPartialCorr, MvStudentTMeanStdPartialCorrDefault,
    MvStudentTMeanStdPartialCorrEta, MvStudentTMeanStdPartialCorrTheta,
};
#[cfg(feature = "multivariate")]
pub use multivariate::{
    DirichletMeanPrecision, DirichletMeanPrecisionEta, DirichletMeanPrecisionTheta,
    DirichletMultinomialFixedTrials, DirichletMultinomialMeanPrecisionEta,
    DirichletMultinomialMeanPrecisionTheta, DirichletMultinomialVaryingTrials,
    FixedLogRatioCholesky, FixedLowerTriangular, FixedPartialCorrelations,
    LogisticNormalAlrCholesky, LogisticNormalAlrCholeskyDefault, LogisticNormalAlrCholeskyEta,
    LogisticNormalAlrCholeskyTheta, MultinomialEta, MultinomialFixedTrials, MultinomialTheta,
    MultinomialVaryingTrials, MvLogNormalCholesky, MvLogNormalCholeskyDefault,
    MvLogNormalCholeskyEta, MvLogNormalCholeskyTheta, MvPoissonCommonShock,
    MvPoissonCommonShockDefault, MvPoissonCommonShockEta, MvPoissonCommonShockTheta,
    MvPowerExponentialCholesky, MvPowerExponentialCholeskyDefault, MvPowerExponentialCholeskyEta,
    MvPowerExponentialCholeskyTheta, MvPowerExponentialMeanStdPartialCorr,
    MvPowerExponentialMeanStdPartialCorrDefault, MvPowerExponentialMeanStdPartialCorrEta,
    MvPowerExponentialMeanStdPartialCorrTheta, MvShashMuSigmaNuTauPartialCorr,
    MvShashMuSigmaNuTauPartialCorrDefault, MvShashMuSigmaNuTauPartialCorrEta,
    MvShashMuSigmaNuTauPartialCorrTheta, MvSinhArcsinhMuSigmaNuTauPartialCorr,
    MvSinhArcsinhMuSigmaNuTauPartialCorrDefault, MvSinhArcsinhMuSigmaNuTauPartialCorrEta,
    MvSinhArcsinhMuSigmaNuTauPartialCorrTheta, MvSkewNormalCholesky, MvSkewNormalCholeskyDefault,
    MvSkewNormalCholeskyEta, MvSkewNormalCholeskyTheta, MvSkewNormalLocationKernelStdPartialCorr,
    MvSkewNormalLocationKernelStdPartialCorrDefault, MvSkewNormalLocationKernelStdPartialCorrEta,
    MvSkewNormalLocationKernelStdPartialCorrTheta, MvSkewStudentTFixedTauCholesky,
    MvSkewStudentTFixedTauCholeskyDefault, MvSkewStudentTFixedTauCholeskyEta,
    MvSkewStudentTFixedTauCholeskyTheta, PackedLowerTriangular,
};
pub use univariate::{
    Beinf, BeinfEta, BeinfMuSigmaNuTau, BeinfTheta, Bernoulli, BernoulliEta, BernoulliProbability,
    BernoulliTheta, Beta, BetaBinomial, BetaBinomialEta, BetaBinomialMeanPrecision,
    BetaBinomialTheta, BetaEta, BetaMeanPrecision, BetaTheta, BinomialEta, BinomialFixedTrials,
    BinomialFixedTrialsProbability, BinomialTheta, BinomialVaryingTrials,
    BinomialVaryingTrialsProbability, Categorical, CategoricalEta, CategoricalTheta, Chi,
    ChiDegreesOfFreedom, ChiEta, ChiSquared, ChiSquaredDegreesOfFreedom, ChiSquaredEta,
    ChiSquaredTheta, ChiTheta, Exponential, ExponentialMean, ExponentialMeanEta,
    ExponentialMeanTheta, ExponentialRate, ExponentialRateEta, ExponentialRateTheta, Gamma,
    GammaMeanCv, GammaMeanCvEta, GammaMeanCvTheta, GammaMeanShape, GammaMeanShapeEta,
    GammaMeanShapeTheta, GammaShapeRate, GammaShapeRateEta, GammaShapeRateTheta, Ged, GedMuSigmaNu,
    GeneralizedGamma, GeneralizedGammaEta, GeneralizedGammaScaleSigmaNu, GeneralizedGammaTheta,
    GeneralizedPareto, GeneralizedParetoEta, GeneralizedParetoScaleShape, GeneralizedParetoTheta,
    Geometric, GeometricEta, GeometricMean, GeometricTheta, Gev, GevEta, GevMuSigmaShape, GevTheta,
    Gumbel, GumbelEta, GumbelMuSigma, GumbelTheta, InverseGaussian, InverseGaussianEta,
    InverseGaussianMeanCv, InverseGaussianMeanCvEta, InverseGaussianMeanCvTheta,
    InverseGaussianMeanShape, InverseGaussianMuShape, InverseGaussianTheta, JohnsonSu,
    JohnsonSuEta, JohnsonSuMuSigmaNuTau, JohnsonSuTheta, Laplace, LaplaceEta, LaplaceMuSigma,
    LaplaceTheta, LogLogistic, LogLogisticEta, LogLogisticScaleShape, LogLogisticTheta, LogNormal,
    LogNormalLogLocationLogSd, LogNormalLogLocationLogSdEta, LogNormalLogLocationLogSdTheta,
    LogNormalMeanCv, LogNormalMeanCvEta, LogNormalMeanCvTheta, LogNormalMeanLogSd,
    LogNormalMeanLogSdEta, LogNormalMeanLogSdTheta, LogNormalMedianLogSd, LogNormalMedianLogSdEta,
    LogNormalMedianLogSdTheta, Logistic, LogisticEta, LogisticMuSigma, LogisticTheta, Lomax,
    LomaxEta, LomaxShapeScale, LomaxTheta, NegativeBinomial, NegativeBinomialEta,
    NegativeBinomialMeanDispersion, NegativeBinomialMeanDispersionEta,
    NegativeBinomialMeanDispersionTheta, NegativeBinomialMeanSize, NegativeBinomialTheta, Normal,
    NormalEta, NormalGamlss, NormalMuSigma, NormalTheta, Poisson, PoissonEta, PoissonMean,
    PoissonTheta, PowerExponential, PowerExponentialEta, PowerExponentialMuSigmaNu,
    PowerExponentialTheta, Rayleigh, RayleighEta, RayleighScale, RayleighTheta, Shash, ShashEta,
    ShashMuSigmaNuTau, ShashTheta, SinhArcsinh, SinhArcsinhEta, SinhArcsinhMuSigmaNuTau,
    SinhArcsinhTheta, SkewNormal, SkewNormalEta, SkewNormalMeanSd, SkewNormalMeanSdEta,
    SkewNormalMeanSdNu, SkewNormalMeanSdTheta, SkewNormalMuSigmaNu, SkewNormalTheta,
    SkewPowerExponential, SkewPowerExponentialEta, SkewPowerExponentialMeanSd,
    SkewPowerExponentialMeanSdEta, SkewPowerExponentialMeanSdSkewPower,
    SkewPowerExponentialMeanSdTheta, SkewPowerExponentialMuSigmaSkewPower,
    SkewPowerExponentialTheta, SkewStudentT, SkewStudentTEta, SkewStudentTMeanSd,
    SkewStudentTMeanSdEta, SkewStudentTMeanSdNuTau, SkewStudentTMeanSdTheta,
    SkewStudentTMuSigmaNuTau, SkewStudentTTheta, StudentT, StudentTDynamic, StudentTEta,
    StudentTMuSdTau, StudentTMuSdTauEta, StudentTMuSdTauTheta, StudentTMuSigma, StudentTMuSigmaTau,
    StudentTMuSigmaTauEta, StudentTMuSigmaTauTheta, StudentTStdDev, StudentTTheta, Tweedie,
    TweedieCv, TweedieEta, TweedieMeanCvPower, TweedieMeanCvPowerEta, TweedieMeanCvPowerTheta,
    TweedieMeanDispersionPower, TweedieTheta, Weibull, WeibullMeanShape, WeibullMeanShapeEta,
    WeibullMeanShapeTheta, WeibullScaleShape, WeibullScaleShapeEta, WeibullScaleShapeTheta, Zaga,
    ZagaComponentMeanCvZeroProbability, ZagaComponentMeanCvZeroProbabilityEta,
    ZagaComponentMeanCvZeroProbabilityTheta, ZagaTotalMeanCvZeroProbability,
    ZagaTotalMeanCvZeroProbabilityEta, ZagaTotalMeanCvZeroProbabilityTheta, Zinb,
    ZinbComponentMeanSizeZeroProbability, ZinbComponentMeanSizeZeroProbabilityEta,
    ZinbComponentMeanSizeZeroProbabilityTheta, ZinbTotalMeanSizeZeroProbability,
    ZinbTotalMeanSizeZeroProbabilityEta, ZinbTotalMeanSizeZeroProbabilityTheta, Zip,
    ZipComponentMeanZeroProbability, ZipComponentMeanZeroProbabilityEta,
    ZipComponentMeanZeroProbabilityTheta, ZipTotalMeanZeroProbability,
    ZipTotalMeanZeroProbabilityEta, ZipTotalMeanZeroProbabilityTheta, normal_gamlss,
};

pub use domain::ScalarObservationDomain;
pub use mixture::{
    FactorizedComponentGradient, Mixture, MixtureEta, MixtureGradient, MixtureTheta,
    MixtureWorkspace,
};

mod constants;
mod crps;
mod domain;
mod initial;
mod link;
/// Homogeneous finite-mixture distributions.
pub mod mixture;
/// Multivariate distributions.
#[cfg(feature = "multivariate")]
pub mod multivariate;
mod numeric;
mod shash_kernel;
#[cfg(feature = "rand")]
mod simulation;
/// Univariate distributions.
pub mod univariate;

/// Most commonly used imports from `gamlss-family`.
pub mod prelude {
    #[cfg(feature = "multivariate")]
    pub use crate::{
        DirichletMeanPrecision, DirichletMeanPrecisionEta, DirichletMeanPrecisionTheta,
        DirichletMultinomialFixedTrials, DirichletMultinomialMeanPrecisionEta,
        DirichletMultinomialMeanPrecisionTheta, DirichletMultinomialVaryingTrials,
        DynMvNormalCholesky, DynMvNormalCholeskyDefault, DynMvNormalCholeskyEta,
        DynMvNormalCholeskyTheta, FixedLogRatioCholesky, FixedLowerTriangular,
        FixedPartialCorrelations, IndependentVec, LogisticNormalAlrCholesky,
        LogisticNormalAlrCholeskyDefault, LogisticNormalAlrCholeskyEta,
        LogisticNormalAlrCholeskyTheta, MultinomialEta, MultinomialFixedTrials, MultinomialTheta,
        MultinomialVaryingTrials, MvLogNormalCholesky, MvLogNormalCholeskyDefault,
        MvLogNormalCholeskyEta, MvLogNormalCholeskyTheta, MvNormalCholesky,
        MvNormalCholeskyDefault, MvNormalCholeskyEta, MvNormalCholeskyTheta,
        MvNormalMeanStdPartialCorr, MvNormalMeanStdPartialCorrDefault,
        MvNormalMeanStdPartialCorrEta, MvNormalMeanStdPartialCorrTheta, MvPoissonCommonShock,
        MvPoissonCommonShockDefault, MvPoissonCommonShockEta, MvPoissonCommonShockTheta,
        MvPowerExponentialCholesky, MvPowerExponentialCholeskyDefault,
        MvPowerExponentialCholeskyEta, MvPowerExponentialCholeskyTheta,
        MvPowerExponentialMeanStdPartialCorr, MvPowerExponentialMeanStdPartialCorrDefault,
        MvPowerExponentialMeanStdPartialCorrEta, MvPowerExponentialMeanStdPartialCorrTheta,
        MvShashMuSigmaNuTauPartialCorr, MvShashMuSigmaNuTauPartialCorrDefault,
        MvShashMuSigmaNuTauPartialCorrEta, MvShashMuSigmaNuTauPartialCorrTheta,
        MvSinhArcsinhMuSigmaNuTauPartialCorr, MvSinhArcsinhMuSigmaNuTauPartialCorrDefault,
        MvSinhArcsinhMuSigmaNuTauPartialCorrEta, MvSinhArcsinhMuSigmaNuTauPartialCorrTheta,
        MvSkewNormalCholesky, MvSkewNormalCholeskyDefault, MvSkewNormalCholeskyEta,
        MvSkewNormalCholeskyTheta, MvSkewNormalLocationKernelStdPartialCorr,
        MvSkewNormalLocationKernelStdPartialCorrDefault,
        MvSkewNormalLocationKernelStdPartialCorrEta, MvSkewNormalLocationKernelStdPartialCorrTheta,
        MvSkewStudentTFixedTauCholesky, MvSkewStudentTFixedTauCholeskyDefault,
        MvSkewStudentTFixedTauCholeskyEta, MvSkewStudentTFixedTauCholeskyTheta, MvStudentTCholesky,
        MvStudentTCholeskyDefault, MvStudentTCholeskyEta, MvStudentTCholeskyTheta,
        MvStudentTMeanStdPartialCorr, MvStudentTMeanStdPartialCorrDefault,
        MvStudentTMeanStdPartialCorrEta, MvStudentTMeanStdPartialCorrTheta, PackedLowerTriangular,
    };

    pub use crate::{
        Beinf, BeinfEta, BeinfMuSigmaNuTau, BeinfTheta, Bernoulli, BernoulliEta,
        BernoulliProbability, BernoulliTheta, Beta, BetaBinomial, BetaBinomialEta,
        BetaBinomialMeanPrecision, BetaBinomialTheta, BetaEta, BetaMeanPrecision, BetaTheta,
        BinomialEta, BinomialFixedTrials, BinomialFixedTrialsProbability, BinomialTheta,
        BinomialVaryingTrials, BinomialVaryingTrialsProbability, Categorical, CategoricalEta,
        CategoricalTheta, Chi, ChiDegreesOfFreedom, ChiEta, ChiSquared, ChiSquaredDegreesOfFreedom,
        ChiSquaredEta, ChiSquaredTheta, ChiTheta, Exponential, ExponentialMean, ExponentialMeanEta,
        ExponentialMeanTheta, ExponentialRate, ExponentialRateEta, ExponentialRateTheta,
        FactorizedComponentGradient, Gamma, GammaMeanCv, GammaMeanCvEta, GammaMeanCvTheta,
        GammaMeanShape, GammaMeanShapeEta, GammaMeanShapeTheta, GammaShapeRate, GammaShapeRateEta,
        GammaShapeRateTheta, Ged, GedMuSigmaNu, GeneralizedGamma, GeneralizedGammaEta,
        GeneralizedGammaScaleSigmaNu, GeneralizedGammaTheta, GeneralizedPareto,
        GeneralizedParetoEta, GeneralizedParetoScaleShape, GeneralizedParetoTheta, Geometric,
        GeometricEta, GeometricMean, GeometricTheta, Gev, GevEta, GevMuSigmaShape, GevTheta,
        Gumbel, GumbelEta, GumbelMuSigma, GumbelTheta, InverseGaussian, InverseGaussianEta,
        InverseGaussianMeanCv, InverseGaussianMeanCvEta, InverseGaussianMeanCvTheta,
        InverseGaussianMeanShape, InverseGaussianMuShape, InverseGaussianTheta, JohnsonSu,
        JohnsonSuEta, JohnsonSuMuSigmaNuTau, JohnsonSuTheta, Laplace, LaplaceEta, LaplaceMuSigma,
        LaplaceTheta, LogLogistic, LogLogisticEta, LogLogisticScaleShape, LogLogisticTheta,
        LogNormal, LogNormalLogLocationLogSd, LogNormalLogLocationLogSdEta,
        LogNormalLogLocationLogSdTheta, LogNormalMeanCv, LogNormalMeanCvEta, LogNormalMeanCvTheta,
        LogNormalMeanLogSd, LogNormalMeanLogSdEta, LogNormalMeanLogSdTheta, LogNormalMedianLogSd,
        LogNormalMedianLogSdEta, LogNormalMedianLogSdTheta, Logistic, LogisticEta, LogisticMuSigma,
        LogisticTheta, Lomax, LomaxEta, LomaxShapeScale, LomaxTheta, Mixture, MixtureEta,
        MixtureGradient, MixtureTheta, NegativeBinomial, NegativeBinomialEta,
        NegativeBinomialMeanDispersion, NegativeBinomialMeanDispersionEta,
        NegativeBinomialMeanDispersionTheta, NegativeBinomialMeanSize, NegativeBinomialTheta,
        Normal, NormalEta, NormalGamlss, NormalMuSigma, NormalTheta, Poisson, PoissonEta,
        PoissonMean, PoissonTheta, PowerExponential, PowerExponentialEta,
        PowerExponentialMuSigmaNu, PowerExponentialTheta, Rayleigh, RayleighEta, RayleighScale,
        RayleighTheta, Shash, ShashEta, ShashMuSigmaNuTau, ShashTheta, SinhArcsinh, SinhArcsinhEta,
        SinhArcsinhMuSigmaNuTau, SinhArcsinhTheta, SkewNormal, SkewNormalEta, SkewNormalMeanSd,
        SkewNormalMeanSdEta, SkewNormalMeanSdNu, SkewNormalMeanSdTheta, SkewNormalMuSigmaNu,
        SkewNormalTheta, SkewPowerExponential, SkewPowerExponentialEta, SkewPowerExponentialMeanSd,
        SkewPowerExponentialMeanSdEta, SkewPowerExponentialMeanSdSkewPower,
        SkewPowerExponentialMeanSdTheta, SkewPowerExponentialMuSigmaSkewPower,
        SkewPowerExponentialTheta, SkewStudentT, SkewStudentTEta, SkewStudentTMeanSd,
        SkewStudentTMeanSdEta, SkewStudentTMeanSdNuTau, SkewStudentTMeanSdTheta,
        SkewStudentTMuSigmaNuTau, SkewStudentTTheta, StudentT, StudentTDynamic, StudentTEta,
        StudentTMuSdTau, StudentTMuSdTauEta, StudentTMuSdTauTheta, StudentTMuSigma,
        StudentTMuSigmaTau, StudentTMuSigmaTauEta, StudentTMuSigmaTauTheta, StudentTStdDev,
        StudentTTheta, Tweedie, TweedieCv, TweedieEta, TweedieMeanCvPower, TweedieMeanCvPowerEta,
        TweedieMeanCvPowerTheta, TweedieMeanDispersionPower, TweedieTheta, Weibull,
        WeibullMeanShape, WeibullMeanShapeEta, WeibullMeanShapeTheta, WeibullScaleShape,
        WeibullScaleShapeEta, WeibullScaleShapeTheta, Zaga, ZagaComponentMeanCvZeroProbability,
        ZagaComponentMeanCvZeroProbabilityEta, ZagaComponentMeanCvZeroProbabilityTheta,
        ZagaTotalMeanCvZeroProbability, ZagaTotalMeanCvZeroProbabilityEta,
        ZagaTotalMeanCvZeroProbabilityTheta, Zinb, ZinbComponentMeanSizeZeroProbability,
        ZinbComponentMeanSizeZeroProbabilityEta, ZinbComponentMeanSizeZeroProbabilityTheta,
        ZinbTotalMeanSizeZeroProbability, ZinbTotalMeanSizeZeroProbabilityEta,
        ZinbTotalMeanSizeZeroProbabilityTheta, Zip, ZipComponentMeanZeroProbability,
        ZipComponentMeanZeroProbabilityEta, ZipComponentMeanZeroProbabilityTheta,
        ZipTotalMeanZeroProbability, ZipTotalMeanZeroProbabilityEta,
        ZipTotalMeanZeroProbabilityTheta, normal_gamlss,
    };
}

#[cfg(test)]
pub(crate) mod test_support {
    use approx::assert_relative_eq;
    use gamlss_core::{Family, ParameterParts};

    const DEFAULT_EPSILON: f64 = 1.0e-6;
    const DEFAULT_TOLERANCE: f64 = 1.0e-6;

    pub fn assert_gradient_matches_finite_difference<F, const K: usize>(
        family: &F,
        y: f64,
        eta: [f64; K],
    ) where
        F: for<'obs> Family<Observation<'obs> = f64>,
        F::Eta: ParameterParts<K>,
        F::GradientEta: ParameterParts<K>,
    {
        assert_gradient_matches_finite_difference_with_tolerance::<F, K>(
            family,
            y,
            eta,
            DEFAULT_EPSILON,
            DEFAULT_TOLERANCE,
        );
    }

    pub fn assert_gradient_matches_finite_difference_with_tolerance<F, const K: usize>(
        family: &F,
        y: f64,
        eta: [f64; K],
        epsilon: f64,
        tolerance: f64,
    ) where
        F: for<'obs> Family<Observation<'obs> = f64>,
        F::Eta: ParameterParts<K>,
        F::GradientEta: ParameterParts<K>,
    {
        let (_, gradient) =
            family.nll_and_gradient_eta(y, &F::Eta::from_array(eta), &mut family.workspace());

        for index in 0..K {
            let mut plus = eta;
            plus[index] += epsilon;
            let mut minus = eta;
            minus[index] -= epsilon;

            let finite_difference =
                (family.nll_eta(y, &F::Eta::from_array(plus), &mut family.workspace())
                    - family.nll_eta(y, &F::Eta::from_array(minus), &mut family.workspace()))
                    / (2.0 * epsilon);
            let actual = gradient.part(index);

            assert!(
                actual.is_finite(),
                "gradient component {index} is not finite: {actual}"
            );
            assert!(
                finite_difference.is_finite(),
                "finite-difference gradient component {index} is not finite: {finite_difference}"
            );
            assert_relative_eq!(actual, finite_difference, epsilon = tolerance);
        }
    }

    #[allow(clippy::while_float)]
    pub fn statrs_discrete_quantile<F>(p: f64, mut cdf: F) -> u64
    where
        F: FnMut(u64) -> f64,
    {
        let mut high = 1_u64;
        while cdf(high) < p {
            high *= 2;
        }
        let mut low = 0_u64;
        while low < high {
            let mid = low + (high - low) / 2;
            if cdf(mid) < p {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        low
    }
}

#[cfg(test)]
mod initializer_tests {
    use gamlss_core::{Family, InitialEtaFromObservations, ParameterParts};

    use crate::{
        BeinfMuSigmaNuTau, BernoulliProbability, BetaMeanPrecision, ChiDegreesOfFreedom,
        ChiSquaredDegreesOfFreedom, GammaShapeRate, GeneralizedGammaScaleSigmaNu,
        GeneralizedParetoScaleShape, GeometricMean, GevMuSigmaShape, GumbelMuSigma,
        InverseGaussianMuShape, JohnsonSuMuSigmaNuTau, LaplaceMuSigma, LogLogisticScaleShape,
        LogNormalLogLocationLogSd, LogisticMuSigma, LomaxShapeScale, NegativeBinomialMeanSize,
        NormalMuSigma, PoissonMean, PowerExponentialMuSigmaNu, RayleighScale, ShashMuSigmaNuTau,
        SkewNormalMuSigmaNu, SkewPowerExponentialMeanSdSkewPower,
        SkewPowerExponentialMuSigmaSkewPower, SkewStudentTMuSigmaNuTau, StudentTMuSigma,
        TweedieMeanDispersionPower, WeibullScaleShape, ZagaComponentMeanCvZeroProbability,
        ZinbComponentMeanSizeZeroProbability, ZipComponentMeanZeroProbability,
    };

    #[allow(clippy::needless_pass_by_value)]
    fn assert_finite_initial_eta<F, const K: usize>(family: F, data: &[f64], probe: f64)
    where
        F: for<'obs> Family<Observation<'obs> = f64> + InitialEtaFromObservations<K>,
        F::Eta: Copy + ParameterParts<K>,
        F::GradientEta: ParameterParts<K>,
    {
        let obs: &[f64] = data;
        let eta = family.initial_eta_from_observations(&obs);
        for index in 0..K {
            assert!(
                eta.part(index).is_finite(),
                "initializer component {index} is not finite"
            );
        }
        assert!(
            family
                .nll_eta(probe, &eta, &mut family.workspace())
                .is_finite(),
            "initialized eta should produce finite nll"
        );
    }

    #[test]
    fn built_in_initializers_return_finite_eta_and_likelihood() {
        assert_finite_initial_eta::<_, 1>(BernoulliProbability::new(), &[0.0, 1.0, 1.0], 1.0);
        assert_finite_initial_eta::<_, 4>(BeinfMuSigmaNuTau::new(), &[0.0, 0.2, 0.8, 1.0], 0.5);
        assert_finite_initial_eta::<_, 2>(BetaMeanPrecision::new(), &[0.1, 0.5, 0.9], 0.5);
        assert_finite_initial_eta::<_, 1>(GeometricMean::new(), &[0.0, 1.0, 3.0], 1.0);
        assert_finite_initial_eta::<_, 1>(RayleighScale::new(), &[0.5, 1.0, 2.0], 1.0);
        assert_finite_initial_eta::<_, 2>(LogLogisticScaleShape::new(), &[0.5, 1.0, 2.0], 1.0);
        assert_finite_initial_eta::<_, 1>(ChiDegreesOfFreedom::new(), &[0.5, 1.0, 2.0], 1.0);
        assert_finite_initial_eta::<_, 1>(ChiSquaredDegreesOfFreedom::new(), &[0.5, 1.0, 2.0], 1.0);
        assert_finite_initial_eta::<_, 2>(
            GeneralizedParetoScaleShape::new(),
            &[0.0, 1.0, 2.0],
            1.0,
        );
        assert_finite_initial_eta::<_, 2>(GammaShapeRate::new(), &[0.2, 1.0, 2.0], 1.0);
        assert_finite_initial_eta::<_, 3>(
            GeneralizedGammaScaleSigmaNu::new(),
            &[0.2, 1.0, 2.0],
            1.0,
        );
        assert_finite_initial_eta::<_, 3>(GevMuSigmaShape::new(), &[-1.0, 0.0, 3.0], 0.0);
        assert_finite_initial_eta::<_, 2>(GumbelMuSigma::new(), &[-1.0, 0.0, 3.0], 0.0);
        assert_finite_initial_eta::<_, 2>(InverseGaussianMuShape::new(), &[0.5, 1.0, 2.0], 1.0);
        assert_finite_initial_eta::<_, 4>(JohnsonSuMuSigmaNuTau::new(), &[-1.0, 0.0, 3.0], 0.0);
        assert_finite_initial_eta::<_, 2>(LaplaceMuSigma::new(), &[-1.0, 0.0, 3.0], 0.0);
        assert_finite_initial_eta::<_, 2>(LogNormalLogLocationLogSd::new(), &[0.5, 1.0, 4.0], 1.0);
        assert_finite_initial_eta::<_, 2>(LogisticMuSigma::new(), &[-1.0, 0.0, 3.0], 0.0);
        assert_finite_initial_eta::<_, 2>(LomaxShapeScale::new(), &[0.0, 1.0, 3.0, 8.0], 1.0);
        assert_finite_initial_eta::<_, 2>(
            NegativeBinomialMeanSize::new(),
            &[0.0, 1.0, 4.0, 6.0],
            1.0,
        );
        assert_finite_initial_eta::<_, 2>(NormalMuSigma::new(), &[-1.0, 0.0, 3.0], 0.0);
        assert_finite_initial_eta::<_, 1>(PoissonMean::new(), &[0.0, 2.0, 4.0], 2.0);
        assert_finite_initial_eta::<_, 3>(PowerExponentialMuSigmaNu::new(), &[-1.0, 0.0, 3.0], 0.0);
        assert_finite_initial_eta::<_, 4>(ShashMuSigmaNuTau::new(), &[-1.0, 0.0, 3.0], 0.0);
        assert_finite_initial_eta::<_, 3>(SkewNormalMuSigmaNu::new(), &[-1.0, 0.0, 3.0], 0.0);
        assert_finite_initial_eta::<_, 4>(
            SkewPowerExponentialMuSigmaSkewPower::new(),
            &[-1.0, 0.0, 3.0],
            0.0,
        );
        assert_finite_initial_eta::<_, 4>(
            SkewPowerExponentialMeanSdSkewPower::new(),
            &[-1.0, 0.0, 3.0],
            0.0,
        );
        assert_finite_initial_eta::<_, 4>(SkewStudentTMuSigmaNuTau::new(), &[-1.0, 0.0, 3.0], 0.0);
        assert_finite_initial_eta::<_, 2>(StudentTMuSigma::default(), &[-1.0, 0.0, 3.0], 0.0);
        assert_finite_initial_eta::<_, 3>(TweedieMeanDispersionPower::new(), &[0.0, 1.0, 3.0], 1.0);
        assert_finite_initial_eta::<_, 2>(WeibullScaleShape::new(), &[0.5, 1.5, 3.0], 1.5);
        assert_finite_initial_eta::<_, 3>(
            ZagaComponentMeanCvZeroProbability::new(),
            &[0.0, 1.0, 3.0],
            1.0,
        );
        assert_finite_initial_eta::<_, 3>(
            ZinbComponentMeanSizeZeroProbability::new(),
            &[0.0, 1.0, 4.0],
            1.0,
        );
        assert_finite_initial_eta::<_, 2>(
            ZipComponentMeanZeroProbability::new(),
            &[0.0, 1.0, 4.0],
            1.0,
        );
    }

    #[test]
    fn initializers_handle_edge_samples_without_nonfinite_eta() {
        assert_finite_initial_eta::<_, 2>(BetaMeanPrecision::new(), &[0.0, 1.0, f64::NAN], 0.5);
        assert_finite_initial_eta::<_, 2>(GammaShapeRate::new(), &[2.0, 2.0, f64::INFINITY], 2.0);
        assert_finite_initial_eta::<_, 2>(NegativeBinomialMeanSize::new(), &[0.0, 0.0, 0.0], 0.0);
        assert_finite_initial_eta::<_, 1>(PoissonMean::new(), &[0.0, 0.0, 0.0], 0.0);
    }
}
