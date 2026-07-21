//! Univariate distribution families.

#[macro_use]
mod macros;

pub use beinf::{Beinf, BeinfEta, BeinfMuSigmaNuTau, BeinfTheta};
pub use bernoulli::{Bernoulli, BernoulliEta, BernoulliProbability, BernoulliTheta};
pub use beta::{Beta, BetaEta, BetaMeanPrecision, BetaTheta};
pub use beta_binomial::{
    BetaBinomial, BetaBinomialEta, BetaBinomialMeanPrecision, BetaBinomialTheta,
};
pub use binomial::{
    BinomialEta, BinomialFixedTrials, BinomialFixedTrialsProbability, BinomialTheta,
    BinomialVaryingTrials, BinomialVaryingTrialsProbability,
};
pub use categorical::{Categorical, CategoricalEta, CategoricalTheta};
pub use chi::{Chi, ChiDegreesOfFreedom, ChiEta, ChiTheta};
pub use chi_squared::{ChiSquared, ChiSquaredDegreesOfFreedom, ChiSquaredEta, ChiSquaredTheta};
pub use exponential::{
    Exponential, ExponentialMean, ExponentialMeanEta, ExponentialMeanTheta, ExponentialRate,
    ExponentialRateEta, ExponentialRateTheta,
};
pub use gamma::{
    Gamma, GammaMeanCv, GammaMeanCvEta, GammaMeanCvTheta, GammaMeanShape, GammaMeanShapeEta,
    GammaMeanShapeTheta, GammaShapeRate, GammaShapeRateEta, GammaShapeRateTheta,
};
pub use generalized_gamma::{
    GeneralizedGamma, GeneralizedGammaEta, GeneralizedGammaScaleSigmaNu, GeneralizedGammaTheta,
};
pub use generalized_pareto::{
    GeneralizedPareto, GeneralizedParetoEta, GeneralizedParetoScaleShape, GeneralizedParetoTheta,
};
pub use geometric::{Geometric, GeometricEta, GeometricMean, GeometricTheta};
pub use gev::{Gev, GevEta, GevMuSigmaShape, GevTheta};
pub use gumbel::{Gumbel, GumbelEta, GumbelMuSigma, GumbelTheta};
pub use inverse_gaussian::{
    InverseGaussian, InverseGaussianEta, InverseGaussianMeanCv, InverseGaussianMeanCvEta,
    InverseGaussianMeanCvTheta, InverseGaussianMeanShape, InverseGaussianMuShape,
    InverseGaussianTheta,
};
pub use johnson_su::{JohnsonSu, JohnsonSuEta, JohnsonSuMuSigmaNuTau, JohnsonSuTheta};
pub use laplace::{Laplace, LaplaceEta, LaplaceMuSigma, LaplaceTheta};
pub use log_logistic::{LogLogistic, LogLogisticEta, LogLogisticScaleShape, LogLogisticTheta};
pub use log_normal::{
    LogNormal, LogNormalLogLocationLogSd, LogNormalLogLocationLogSdEta,
    LogNormalLogLocationLogSdTheta, LogNormalMeanCv, LogNormalMeanCvEta, LogNormalMeanCvTheta,
    LogNormalMeanLogSd, LogNormalMeanLogSdEta, LogNormalMeanLogSdTheta, LogNormalMedianLogSd,
    LogNormalMedianLogSdEta, LogNormalMedianLogSdTheta,
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
pub use rayleigh::{Rayleigh, RayleighEta, RayleighScale, RayleighTheta};
pub use shash::{
    Shash, ShashEta, ShashMuSigmaNuTau, ShashTheta, SinhArcsinh, SinhArcsinhEta,
    SinhArcsinhMuSigmaNuTau, SinhArcsinhTheta,
};
pub use skew_normal::{
    SkewNormal, SkewNormalEta, SkewNormalMeanSd, SkewNormalMeanSdEta, SkewNormalMeanSdNu,
    SkewNormalMeanSdTheta, SkewNormalMuSigmaNu, SkewNormalTheta,
};
pub use skew_power_exponential::{
    SkewPowerExponential, SkewPowerExponentialEta, SkewPowerExponentialMeanSd,
    SkewPowerExponentialMeanSdEta, SkewPowerExponentialMeanSdSkewPower,
    SkewPowerExponentialMeanSdTheta, SkewPowerExponentialMuSigmaSkewPower,
    SkewPowerExponentialTheta,
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
    Weibull, WeibullMeanShape, WeibullMeanShapeEta, WeibullMeanShapeTheta, WeibullScaleShape,
    WeibullScaleShapeEta, WeibullScaleShapeTheta,
};
pub use zaga::{
    Zaga, ZagaComponentMeanCvZeroProbability, ZagaComponentMeanCvZeroProbabilityEta,
    ZagaComponentMeanCvZeroProbabilityTheta, ZagaTotalMeanCvZeroProbability,
    ZagaTotalMeanCvZeroProbabilityEta, ZagaTotalMeanCvZeroProbabilityTheta,
};
pub use zinb::{
    Zinb, ZinbComponentMeanSizeZeroProbability, ZinbComponentMeanSizeZeroProbabilityEta,
    ZinbComponentMeanSizeZeroProbabilityTheta, ZinbTotalMeanSizeZeroProbability,
    ZinbTotalMeanSizeZeroProbabilityEta, ZinbTotalMeanSizeZeroProbabilityTheta,
};
pub use zip::{
    Zip, ZipComponentMeanZeroProbability, ZipComponentMeanZeroProbabilityEta,
    ZipComponentMeanZeroProbabilityTheta, ZipTotalMeanZeroProbability,
    ZipTotalMeanZeroProbabilityEta, ZipTotalMeanZeroProbabilityTheta,
};

/// Beta inflated at zero and one distribution.
pub mod beinf;
/// Bernoulli distribution.
pub mod bernoulli;
/// Beta distribution.
pub mod beta;
/// Beta-binomial distribution.
pub mod beta_binomial;
/// Binomial distributions with fixed or observation-specific trial counts.
pub mod binomial;
/// Categorical distribution with compile-time category count.
pub mod categorical;
/// Chi distribution.
pub mod chi;
/// Chi-squared distribution.
pub mod chi_squared;
/// Exponential distribution.
pub mod exponential;
/// Gamma distribution.
pub mod gamma;
/// Generalized gamma distribution.
pub mod generalized_gamma;
/// Generalized Pareto excess distribution.
pub mod generalized_pareto;
/// Geometric distribution.
pub mod geometric;
/// Generalized extreme value distribution.
pub mod gev;
/// Maximum-type Gumbel distribution.
pub mod gumbel;
/// Inverse Gaussian distribution.
pub mod inverse_gaussian;
/// Johnson SU distribution.
pub mod johnson_su;
/// Laplace distribution.
pub mod laplace;
/// Log-logistic distribution.
pub mod log_logistic;
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
/// Poisson distribution.
pub mod poisson;
/// Power exponential / generalized error distribution.
pub mod power_exponential;
/// Rayleigh distribution.
pub mod rayleigh;
/// Sinh-arcsinh distribution.
pub mod shash;
/// Full-name module alias for the sinh-arcsinh / SHASH distribution.
pub use shash as sinh_arcsinh;
/// Skew-normal distribution.
pub mod skew_normal;
/// Fernández--Steel two-piece skew power-exponential / generalized-error distribution.
pub mod skew_power_exponential;
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
