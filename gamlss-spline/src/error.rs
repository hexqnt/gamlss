use gamlss_core::ModelError;
use thiserror::Error;

/// Errors for spline basis and spline design matrix construction.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SplineError {
    /// Input vector is empty.
    #[error("spline input must contain at least one value")]
    EmptyInput,

    /// Input vector contains `NaN` or infinity.
    #[error("spline input contains a non-finite value")]
    NonFiniteValue,

    /// Data range does not have two distinct finite boundaries and a finite span.
    #[error("spline range must have distinct finite boundaries and a finite span")]
    InvalidRange,

    /// Number of basis functions is insufficient for the spline degree.
    #[error("B-spline basis count {n_basis} must be greater than degree {degree}")]
    NotEnoughBasis {
        /// Requested number of basis functions.
        n_basis: usize,
        /// B-spline degree.
        degree: usize,
    },

    /// Knot vector contains non-finite values or is decreasing.
    #[error("knot vector must be finite and nondecreasing")]
    InvalidKnots,

    /// Knot vector does not contain enough knots.
    #[error("spline knot vector must contain at least {min} knots")]
    NotEnoughKnots {
        /// Minimum number of knots.
        min: usize,
    },

    /// Period must be a finite positive number.
    #[error("spline period must be finite and positive")]
    InvalidPeriod,

    /// Predictor blocks have different row counts.
    #[error("spline row count mismatch: expected {expected}, got {actual}")]
    RowMismatch {
        /// Expected row count.
        expected: usize,
        /// Actual row count.
        actual: usize,
    },

    /// Parameter count overflowed `usize`.
    #[error("spline parameter count overflowed")]
    ParameterOverflow,

    /// The spline degree is not supported by this compact predictor.
    #[error("spline degree {degree} is not supported")]
    UnsupportedDegree {
        /// Requested degree.
        degree: usize,
    },

    /// Core design matrix error.
    #[error(transparent)]
    Model(#[from] ModelError),
}

/// Errors for Fourier basis and Fourier predictor construction.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FourierError {
    /// Input vector contains `NaN` or infinity.
    #[error("Fourier input contains a non-finite value")]
    NonFiniteValue,

    /// Period must be positive and both it and its angular frequency must be finite.
    #[error("Fourier period must be positive with finite period and angular frequency")]
    InvalidPeriod,

    /// Number of harmonics must be positive.
    #[error("Fourier order must be greater than zero")]
    InvalidOrder,

    /// Coefficient count overflowed `usize`.
    #[error("Fourier coefficient count overflowed")]
    CoefficientOverflow,
}
