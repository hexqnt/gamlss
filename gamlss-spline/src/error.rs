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

    /// Requested derivative order is not supported by the basis.
    #[error("spline derivative order {requested} exceeds supported order {max}")]
    UnsupportedDerivativeOrder {
        /// Requested derivative order.
        requested: usize,
        /// Largest supported derivative order.
        max: usize,
    },

    /// Duchon smoothness parameters do not define a continuous spline in the requested dimension.
    #[error(
        "invalid Duchon smoothness for dimension {dimension}: m={derivative_order}, 2s={twice_s}; require m > 0, -d < 2s < d, and 2m + 2s > d"
    )]
    InvalidDuchonSmoothness {
        /// Coordinate-space dimension.
        dimension: usize,
        /// Integer derivative order `m`.
        derivative_order: usize,
        /// Exact half-step representation of `s`.
        twice_s: i32,
    },

    /// Requested Duchon regression rank cannot represent its polynomial null space.
    #[error("Duchon basis rank {rank} must be in {min}..={max}")]
    InvalidDuchonRank {
        /// Requested total basis rank.
        rank: usize,
        /// Minimum usable rank.
        min: usize,
        /// Number of supplied centers and maximum usable rank.
        max: usize,
    },

    /// Duchon centers do not identify the required polynomial null space.
    #[error("Duchon centers are duplicated or polynomially rank deficient")]
    DegenerateDuchonCenters,

    /// A numerical spline operation overflowed or failed to reach its required invariant.
    #[error("numerical spline operation failed while computing {context}")]
    NumericalFailure {
        /// Operation that failed.
        context: &'static str,
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
