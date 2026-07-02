//! Multivariate distribution families.

pub use independent::IndependentVec;

/// Cholesky helpers.
pub mod cholesky;
/// Independent product construction.
pub mod independent;
/// Matrix storage primitives.
pub mod matrix;
/// Multivariate normal distributions.
pub mod normal;
