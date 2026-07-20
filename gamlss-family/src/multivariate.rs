//! Multivariate distribution families.
//!
//! Fixed-dimensional families use array-based observation and parameter
//! carriers and are intended for dimensions known at compile time, especially
//! small and medium `D`. Runtime-dimensional families use borrowed observation
//! slices and packed owned parameter storage. At recoverable API boundaries,
//! prefer each fixed family's checked `try_new`; `new` remains a convenience for
//! compile-time dimensions that are programmer-controlled and documents its
//! panic condition.

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
pub use simplex::{DirichletMeanPrecision, DirichletMeanPrecisionEta, DirichletMeanPrecisionTheta};

/// Cholesky helpers.
pub mod cholesky;
mod count;
/// Overdispersed multinomial count distributions.
pub mod dirichlet_multinomial;
mod elliptical;
/// Independent product construction.
pub mod independent;
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
/// Simplex and compositional distributions.
pub mod simplex;
/// Multivariate Student-t distributions.
pub mod student_t;
