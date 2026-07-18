//! Multivariate distribution families.
//!
//! Fixed-dimensional families use array-based observation and parameter
//! carriers and are intended for dimensions known at compile time, especially
//! small and medium `D`. Runtime-dimensional families use borrowed observation
//! slices and packed owned parameter storage. At recoverable API boundaries,
//! prefer each fixed family's checked `try_new`; `new` remains a convenience for
//! compile-time dimensions that are programmer-controlled and documents its
//! panic condition.

pub use independent::IndependentVec;
pub use matrix::{FixedLowerTriangular, PackedLowerTriangular};
pub use simplex::{DirichletMeanPrecision, DirichletMeanPrecisionEta, DirichletMeanPrecisionTheta};

/// Cholesky helpers.
pub mod cholesky;
mod elliptical;
/// Independent product construction.
pub mod independent;
/// Matrix storage primitives.
pub mod matrix;
/// Multivariate normal distributions.
pub mod normal;
/// Simplex and compositional distributions.
pub mod simplex;
/// Multivariate Student-t distributions.
pub mod student_t;
