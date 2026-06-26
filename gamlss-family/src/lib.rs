#![forbid(unsafe_code)]
//! Distributions, likelihood and NLL gradients for GAMLSS.

pub use beinf::{Beinf, BeinfEta, BeinfMuSigmaNuTau, BeinfTheta};
pub use bernoulli::{Bernoulli, BernoulliEta, BernoulliProbability, BernoulliTheta};
pub use beta::{Beta, BetaEta, BetaMeanPrecision, BetaTheta};
pub use exponential::Exponential;
pub use exponential::{
    ExponentialMean, ExponentialMeanEta, ExponentialMeanTheta, ExponentialRate, ExponentialRateEta,
    ExponentialRateTheta,
};
pub use gamma::{
    Gamma, GammaEta, GammaMeanCv, GammaMeanCvEta, GammaMeanCvTheta, GammaMeanShape,
    GammaMeanShapeEta, GammaMeanShapeTheta, GammaShapeRate, GammaShapeRateEta, GammaShapeRateTheta,
    GammaTheta,
};
pub use generalized_gamma::{
    GeneralizedGamma, GeneralizedGammaEta, GeneralizedGammaScaleSigmaNu, GeneralizedGammaTheta,
};
pub use gev::{Gev, GevEta, GevMuSigmaShape, GevTheta};
pub use gumbel::{Gumbel, GumbelEta, GumbelMuSigma, GumbelTheta};
pub use inverse_gaussian::{
    InverseGaussian, InverseGaussianEta, InverseGaussianMeanCv, InverseGaussianMeanCvEta,
    InverseGaussianMeanCvTheta, InverseGaussianMeanShape, InverseGaussianMuShape,
    InverseGaussianTheta,
};
pub use johnson_su::{JohnsonSu, JohnsonSuEta, JohnsonSuMuSigmaNuTau, JohnsonSuTheta};
pub use laplace::{Laplace, LaplaceEta, LaplaceMuSigma, LaplaceTheta};
pub use log_normal::{
    LogNormal, LogNormalEta, LogNormalLogLocationLogSd, LogNormalLogLocationLogSdEta,
    LogNormalLogLocationLogSdTheta, LogNormalMeanCv, LogNormalMeanCvEta, LogNormalMeanCvTheta,
    LogNormalMeanLogSd, LogNormalMeanLogSdEta, LogNormalMeanLogSdTheta, LogNormalMedianLogSd,
    LogNormalMedianLogSdEta, LogNormalMedianLogSdTheta, LogNormalTheta,
};
pub use logistic::{Logistic, LogisticEta, LogisticMuSigma, LogisticTheta};
pub use lomax::{Lomax, LomaxEta, LomaxShapeScale, LomaxTheta};
pub use negative_binomial::{
    NegativeBinomial, NegativeBinomialEta, NegativeBinomialMeanDispersion,
    NegativeBinomialMeanDispersionEta, NegativeBinomialMeanDispersionTheta,
    NegativeBinomialMeanSize, NegativeBinomialTheta,
};
pub use normal::{Normal, NormalEta, NormalGamlss, NormalMuSigma, NormalTheta, normal_gamlss};
pub use poisson::{Poisson, PoissonEta, PoissonMean, PoissonTheta};
pub use power_exponential::{
    Ged, GedMuSigmaNu, PowerExponential, PowerExponentialEta, PowerExponentialMuSigmaNu,
    PowerExponentialTheta,
};
pub use shash::{Shash, ShashEta, ShashMuSigmaNuTau, ShashTheta};
pub use skew_normal::{
    SkewNormal, SkewNormalEta, SkewNormalMeanSd, SkewNormalMeanSdEta, SkewNormalMeanSdNu,
    SkewNormalMeanSdTheta, SkewNormalMuSigmaNu, SkewNormalTheta,
};
pub use skew_student_t::{
    SkewStudentT, SkewStudentTEta, SkewStudentTMeanSd, SkewStudentTMeanSdEta,
    SkewStudentTMeanSdNuTau, SkewStudentTMeanSdTheta, SkewStudentTMuSigmaNuTau, SkewStudentTTheta,
};
pub use student_t::{
    StudentT, StudentTDynamic, StudentTEta, StudentTMuSdTau, StudentTMuSdTauEta,
    StudentTMuSdTauTheta, StudentTMuSigma, StudentTMuSigmaTau, StudentTMuSigmaTauEta,
    StudentTMuSigmaTauTheta, StudentTStdDev, StudentTTheta,
};
pub use tweedie::{
    Tweedie, TweedieCv, TweedieEta, TweedieMeanCvPower, TweedieMeanCvPowerEta,
    TweedieMeanCvPowerTheta, TweedieMeanDispersionPower, TweedieTheta,
};
pub use weibull::{
    Weibull, WeibullEta, WeibullMeanShape, WeibullMeanShapeEta, WeibullMeanShapeTheta,
    WeibullScaleShape, WeibullScaleShapeEta, WeibullScaleShapeTheta, WeibullTheta,
};
pub use zaga::{
    Zaga, ZagaComponentMeanCvZeroProbability, ZagaEta, ZagaMeanSigmaZeroProbability, ZagaTheta,
    ZagaTotalMeanCvZeroProbability, ZagaTotalMeanCvZeroProbabilityEta,
    ZagaTotalMeanCvZeroProbabilityTheta,
};
pub use zinb::{
    Zinb, ZinbComponentMeanSizeZeroProbability, ZinbEta, ZinbMeanSizeZeroProbability, ZinbTheta,
    ZinbTotalMeanSizeZeroProbability, ZinbTotalMeanSizeZeroProbabilityEta,
    ZinbTotalMeanSizeZeroProbabilityTheta,
};
pub use zip::{
    Zip, ZipComponentMeanZeroProbability, ZipEta, ZipMeanZeroProbability, ZipTheta,
    ZipTotalMeanZeroProbability, ZipTotalMeanZeroProbabilityTheta,
};

/// Beta inflated at zero and one distribution.
pub mod beinf;
/// Bernoulli distribution.
pub mod bernoulli;
/// Beta distribution.
pub mod beta;
mod domain;
/// Exponential distribution.
pub mod exponential;
/// Gamma distribution.
pub mod gamma;
/// Generalized gamma distribution.
pub mod generalized_gamma;
/// Generalized extreme value distribution.
pub mod gev;
/// Maximum-type Gumbel distribution.
pub mod gumbel;
mod initial;
/// Inverse Gaussian distribution.
pub mod inverse_gaussian;
/// Johnson SU distribution.
pub mod johnson_su;
/// Laplace distribution.
pub mod laplace;
/// Log-normal distribution.
pub mod log_normal;
/// Logistic distribution.
pub mod logistic;
/// Lomax distribution.
pub mod lomax;
/// Negative binomial distribution.
pub mod negative_binomial;
/// Normal distribution.
pub mod normal;
mod numeric;
/// Poisson distribution.
pub mod poisson;
/// Power exponential / generalized error distribution.
pub mod power_exponential;
/// Sinh-arcsinh distribution.
pub mod shash;
/// Skew-normal distribution.
pub mod skew_normal;
/// Skew Student-t distribution.
pub mod skew_student_t;
/// Student distribution with a fixed number of degrees of freedom.
pub mod student_t;
/// Tweedie compound Poisson-gamma distribution.
pub mod tweedie;
/// Weibull distribution.
pub mod weibull;
/// Zero-adjusted gamma distribution.
pub mod zaga;
/// Zero-inflated negative binomial distribution.
pub mod zinb;
/// Zero-inflated Poisson distribution.
pub mod zip;

/// Most commonly used imports from `gamlss-family`.
pub mod prelude {
    pub use crate::{
        Beinf, BeinfEta, BeinfMuSigmaNuTau, BeinfTheta, Bernoulli, BernoulliEta,
        BernoulliProbability, BernoulliTheta, Beta, BetaEta, BetaMeanPrecision, BetaTheta,
        Exponential, ExponentialMean, ExponentialMeanEta, ExponentialMeanTheta, ExponentialRate,
        ExponentialRateEta, ExponentialRateTheta, Gamma, GammaEta, GammaMeanCv, GammaMeanCvEta,
        GammaMeanCvTheta, GammaMeanShape, GammaMeanShapeEta, GammaMeanShapeTheta, GammaShapeRate,
        GammaShapeRateEta, GammaShapeRateTheta, GammaTheta, Ged, GedMuSigmaNu, GeneralizedGamma,
        GeneralizedGammaEta, GeneralizedGammaScaleSigmaNu, GeneralizedGammaTheta, Gev, GevEta,
        GevMuSigmaShape, GevTheta, Gumbel, GumbelEta, GumbelMuSigma, GumbelTheta, InverseGaussian,
        InverseGaussianEta, InverseGaussianMeanCv, InverseGaussianMeanCvEta,
        InverseGaussianMeanCvTheta, InverseGaussianMeanShape, InverseGaussianMuShape,
        InverseGaussianTheta, JohnsonSu, JohnsonSuEta, JohnsonSuMuSigmaNuTau, JohnsonSuTheta,
        Laplace, LaplaceEta, LaplaceMuSigma, LaplaceTheta, LogNormal, LogNormalEta,
        LogNormalLogLocationLogSd, LogNormalLogLocationLogSdEta, LogNormalLogLocationLogSdTheta,
        LogNormalMeanCv, LogNormalMeanCvEta, LogNormalMeanCvTheta, LogNormalMeanLogSd,
        LogNormalMeanLogSdEta, LogNormalMeanLogSdTheta, LogNormalMedianLogSd,
        LogNormalMedianLogSdEta, LogNormalMedianLogSdTheta, LogNormalTheta, Logistic, LogisticEta,
        LogisticMuSigma, LogisticTheta, Lomax, LomaxEta, LomaxShapeScale, LomaxTheta,
        NegativeBinomial, NegativeBinomialEta, NegativeBinomialMeanDispersion,
        NegativeBinomialMeanDispersionEta, NegativeBinomialMeanDispersionTheta,
        NegativeBinomialMeanSize, NegativeBinomialTheta, Normal, NormalEta, NormalGamlss,
        NormalMuSigma, NormalTheta, Poisson, PoissonEta, PoissonMean, PoissonTheta,
        PowerExponential, PowerExponentialEta, PowerExponentialMuSigmaNu, PowerExponentialTheta,
        Shash, ShashEta, ShashMuSigmaNuTau, ShashTheta, SkewNormal, SkewNormalEta,
        SkewNormalMeanSd, SkewNormalMeanSdEta, SkewNormalMeanSdNu, SkewNormalMeanSdTheta,
        SkewNormalMuSigmaNu, SkewNormalTheta, SkewStudentT, SkewStudentTEta, SkewStudentTMeanSd,
        SkewStudentTMeanSdEta, SkewStudentTMeanSdNuTau, SkewStudentTMeanSdTheta,
        SkewStudentTMuSigmaNuTau, SkewStudentTTheta, StudentT, StudentTDynamic, StudentTEta,
        StudentTMuSdTau, StudentTMuSdTauEta, StudentTMuSdTauTheta, StudentTMuSigma,
        StudentTMuSigmaTau, StudentTMuSigmaTauEta, StudentTMuSigmaTauTheta, StudentTStdDev,
        StudentTTheta, Tweedie, TweedieCv, TweedieEta, TweedieMeanCvPower, TweedieMeanCvPowerEta,
        TweedieMeanCvPowerTheta, TweedieMeanDispersionPower, TweedieTheta, Weibull, WeibullEta,
        WeibullMeanShape, WeibullMeanShapeEta, WeibullMeanShapeTheta, WeibullScaleShape,
        WeibullScaleShapeEta, WeibullScaleShapeTheta, WeibullTheta, Zaga,
        ZagaComponentMeanCvZeroProbability, ZagaEta, ZagaMeanSigmaZeroProbability, ZagaTheta,
        ZagaTotalMeanCvZeroProbability, ZagaTotalMeanCvZeroProbabilityEta,
        ZagaTotalMeanCvZeroProbabilityTheta, Zinb, ZinbComponentMeanSizeZeroProbability, ZinbEta,
        ZinbMeanSizeZeroProbability, ZinbTheta, ZinbTotalMeanSizeZeroProbability,
        ZinbTotalMeanSizeZeroProbabilityEta, ZinbTotalMeanSizeZeroProbabilityTheta, Zip,
        ZipComponentMeanZeroProbability, ZipEta, ZipMeanZeroProbability, ZipTheta,
        ZipTotalMeanZeroProbability, ZipTotalMeanZeroProbabilityTheta, normal_gamlss,
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
        F::NllGradientEta: ParameterParts<K>,
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
        F::NllGradientEta: ParameterParts<K>,
    {
        let (_, gradient) = family.nll_and_gradient_eta(y, F::Eta::from_array(eta));

        for index in 0..K {
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
    use gamlss_core::{Family, ParameterParts, ParameterizedFamily};

    use crate::{
        BeinfMuSigmaNuTau, BernoulliProbability, BetaMeanPrecision, GammaShapeRate,
        GeneralizedGammaScaleSigmaNu, GevMuSigmaShape, GumbelMuSigma, InverseGaussianMuShape,
        JohnsonSuMuSigmaNuTau, LaplaceMuSigma, LogNormalLogLocationLogSd, LogisticMuSigma,
        LomaxShapeScale, NegativeBinomialMeanSize, NormalMuSigma, PoissonMean,
        PowerExponentialMuSigmaNu, ShashMuSigmaNuTau, SkewNormalMuSigmaNu,
        SkewStudentTMuSigmaNuTau, StudentTMuSigma, TweedieMeanDispersionPower, WeibullScaleShape,
        ZagaMeanSigmaZeroProbability, ZinbMeanSizeZeroProbability, ZipMeanZeroProbability,
    };

    #[allow(clippy::needless_pass_by_value)]
    fn assert_finite_initial_eta<F, const K: usize>(family: F, data: &[f64], probe: f64)
    where
        F: for<'obs> Family<Observation<'obs> = f64> + ParameterizedFamily<K>,
        F::Eta: Copy + ParameterParts<K>,
        F::NllGradientEta: ParameterParts<K>,
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
            family.nll_eta(probe, eta).is_finite(),
            "initialized eta should produce finite nll"
        );
    }

    #[test]
    fn built_in_initializers_return_finite_eta_and_likelihood() {
        assert_finite_initial_eta::<_, 1>(BernoulliProbability::new(), &[0.0, 1.0, 1.0], 1.0);
        assert_finite_initial_eta::<_, 4>(BeinfMuSigmaNuTau::new(), &[0.0, 0.2, 0.8, 1.0], 0.5);
        assert_finite_initial_eta::<_, 2>(BetaMeanPrecision::new(), &[0.1, 0.5, 0.9], 0.5);
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
        assert_finite_initial_eta::<_, 4>(SkewStudentTMuSigmaNuTau::new(), &[-1.0, 0.0, 3.0], 0.0);
        assert_finite_initial_eta::<_, 2>(StudentTMuSigma::default(), &[-1.0, 0.0, 3.0], 0.0);
        assert_finite_initial_eta::<_, 3>(TweedieMeanDispersionPower::new(), &[0.0, 1.0, 3.0], 1.0);
        assert_finite_initial_eta::<_, 2>(WeibullScaleShape::new(), &[0.5, 1.5, 3.0], 1.5);
        assert_finite_initial_eta::<_, 3>(
            ZagaMeanSigmaZeroProbability::new(),
            &[0.0, 1.0, 3.0],
            1.0,
        );
        assert_finite_initial_eta::<_, 3>(
            ZinbMeanSizeZeroProbability::new(),
            &[0.0, 1.0, 4.0],
            1.0,
        );
        assert_finite_initial_eta::<_, 2>(ZipMeanZeroProbability::new(), &[0.0, 1.0, 4.0], 1.0);
    }

    #[test]
    fn initializers_handle_edge_samples_without_nonfinite_eta() {
        assert_finite_initial_eta::<_, 2>(BetaMeanPrecision::new(), &[0.0, 1.0, f64::NAN], 0.5);
        assert_finite_initial_eta::<_, 2>(GammaShapeRate::new(), &[2.0, 2.0, f64::INFINITY], 2.0);
        assert_finite_initial_eta::<_, 2>(NegativeBinomialMeanSize::new(), &[0.0, 0.0, 0.0], 0.0);
        assert_finite_initial_eta::<_, 1>(PoissonMean::new(), &[0.0, 0.0, 0.0], 0.0);
    }
}
