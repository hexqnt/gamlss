use gamlss_core::ModelError;
use thiserror::Error;

/// Ошибки построения spline basis и spline design matrix.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SplineError {
    /// Входной вектор пуст.
    #[error("spline input must contain at least one value")]
    EmptyInput,

    /// Входной вектор содержит `NaN` или infinity.
    #[error("spline input contains a non-finite value")]
    NonFiniteValue,

    /// Диапазон данных не имеет двух различных конечных границ.
    #[error("spline range must have distinct finite boundaries")]
    InvalidRange,

    /// Число basis-функций недостаточно для степени spline.
    #[error("B-spline basis count {n_basis} must be greater than degree {degree}")]
    NotEnoughBasis {
        /// Запрошенное число basis-функций.
        n_basis: usize,
        /// Степень B-spline.
        degree: usize,
    },

    /// Knot vector содержит не-finite значения или убывает.
    #[error("knot vector must be finite and nondecreasing")]
    InvalidKnots,

    /// Knot vector содержит недостаточно узлов.
    #[error("spline knot vector must contain at least {min} knots")]
    NotEnoughKnots {
        /// Минимальное число узлов.
        min: usize,
    },

    /// Период должен быть конечным положительным числом.
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

    /// Число параметров переполнило `usize`.
    #[error("spline parameter count overflowed")]
    ParameterOverflow,

    /// Степень сплайна не поддерживается данным compact predictor-ом.
    #[error("spline degree {degree} is not supported")]
    UnsupportedDegree {
        /// Requested degree.
        degree: usize,
    },

    /// Ошибка core design matrix.
    #[error(transparent)]
    Model(#[from] ModelError),
}

/// Ошибки построения Fourier basis и Fourier predictor.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum FourierError {
    /// Входной вектор содержит `NaN` или infinity.
    #[error("Fourier input contains a non-finite value")]
    NonFiniteValue,

    /// Период должен быть конечным положительным числом.
    #[error("Fourier period must be finite and positive")]
    InvalidPeriod,

    /// Число гармоник должно быть положительным.
    #[error("Fourier order must be greater than zero")]
    InvalidOrder,

    /// Число коэффициентов переполнило `usize`.
    #[error("Fourier coefficient count overflowed")]
    CoefficientOverflow,
}
