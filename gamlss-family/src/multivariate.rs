//! Multivariate distribution families.

pub use independent::IndependentVec;
pub use matrix::{FixedLowerTriangular, PackedLowerTriangular};
pub use simplex::{DirichletMeanPrecision, DirichletMeanPrecisionEta, DirichletMeanPrecisionTheta};

/// Cholesky helpers.
pub mod cholesky;
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
