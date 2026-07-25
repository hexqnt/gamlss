//! Multivariate distribution families.
//!
//! Fixed-dimensional families use array-based observation and parameter
//! carriers and are intended for dimensions known at compile time, especially
//! small and medium `D`. Runtime-dimensional families use borrowed observation
//! slices and packed owned parameter storage. At recoverable API boundaries,
//! prefer each fixed family's checked `try_new`; `new` remains a convenience for
//! compile-time dimensions that are programmer-controlled and documents its
//! panic condition.

pub use correlation::FixedPartialCorrelations;
pub use dirichlet_multinomial::{
    DirichletMultinomialFixedTrials, DirichletMultinomialMeanPrecisionEta,
    DirichletMultinomialMeanPrecisionTheta, DirichletMultinomialVaryingTrials,
};
pub use independent::IndependentVec;
pub use log_normal::{
    MvLogNormalCholesky, MvLogNormalCholeskyDefault, MvLogNormalCholeskyEta,
    MvLogNormalCholeskyTheta,
};
pub use logistic_normal::{
    FixedLogRatioCholesky, LogisticNormalAlrCholesky, LogisticNormalAlrCholeskyDefault,
    LogisticNormalAlrCholeskyEta, LogisticNormalAlrCholeskyTheta,
};
pub use matrix::{FixedLowerTriangular, PackedLowerTriangular};
pub use multinomial::{
    MultinomialEta, MultinomialFixedTrials, MultinomialTheta, MultinomialVaryingTrials,
};
pub use poisson_common_shock::{
    MvPoissonCommonShock, MvPoissonCommonShockDefault, MvPoissonCommonShockEta,
    MvPoissonCommonShockTheta,
};
pub use power_exponential::{
    MvPowerExponentialCholesky, MvPowerExponentialCholeskyDefault, MvPowerExponentialCholeskyEta,
    MvPowerExponentialCholeskyTheta, MvPowerExponentialMeanStdPartialCorr,
    MvPowerExponentialMeanStdPartialCorrDefault, MvPowerExponentialMeanStdPartialCorrEta,
    MvPowerExponentialMeanStdPartialCorrTheta,
};
pub use shash::{
    MvShashMuSigmaNuTauPartialCorr, MvShashMuSigmaNuTauPartialCorrDefault,
    MvShashMuSigmaNuTauPartialCorrEta, MvShashMuSigmaNuTauPartialCorrTheta,
    MvSinhArcsinhMuSigmaNuTauPartialCorr, MvSinhArcsinhMuSigmaNuTauPartialCorrDefault,
    MvSinhArcsinhMuSigmaNuTauPartialCorrEta, MvSinhArcsinhMuSigmaNuTauPartialCorrTheta,
};
pub use simplex::{DirichletMeanPrecision, DirichletMeanPrecisionEta, DirichletMeanPrecisionTheta};
pub use skew_normal::{
    MvSkewNormalCholesky, MvSkewNormalCholeskyDefault, MvSkewNormalCholeskyEta,
    MvSkewNormalCholeskyTheta, MvSkewNormalLocationKernelStdPartialCorr,
    MvSkewNormalLocationKernelStdPartialCorrDefault, MvSkewNormalLocationKernelStdPartialCorrEta,
    MvSkewNormalLocationKernelStdPartialCorrTheta,
};
pub use skew_student_t::{
    MvSkewStudentTFixedTauCholesky, MvSkewStudentTFixedTauCholeskyDefault,
    MvSkewStudentTFixedTauCholeskyEta, MvSkewStudentTFixedTauCholeskyTheta,
};

/// Full-name module alias for the multivariate sinh-arcsinh / SHASH distribution.
pub use shash as sinh_arcsinh;

/// Cholesky helpers.
pub mod cholesky;
mod correlation;
mod count;
/// Overdispersed multinomial count distributions.
pub mod dirichlet_multinomial;
mod elliptical;
/// Independent product construction.
pub mod independent;
mod initial;
/// Multivariate log-normal distributions.
pub mod log_normal;
/// Logistic-normal simplex distributions.
pub mod logistic_normal;
/// Matrix storage primitives.
pub mod matrix;
/// Multinomial count distributions.
pub mod multinomial;
/// Multivariate normal distributions.
pub mod normal;
/// Common-shock multivariate Poisson distributions.
pub mod poisson_common_shock;
/// Multivariate power-exponential distributions.
pub mod power_exponential;
/// Multivariate sinh-arcsinh-normal / SHASH distribution.
pub mod shash;
/// Simplex and compositional distributions.
pub mod simplex;
/// Multivariate skew-normal distributions.
pub mod skew_normal;
/// Multivariate skew-Student-t distributions.
pub mod skew_student_t;
/// Multivariate Student-t distributions.
pub mod student_t;
