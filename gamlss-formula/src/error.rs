use gamlss_core::ModelError;
use gamlss_spline::{FourierError, SplineError};
use thiserror::Error;

/// Errors returned by the typed formula/builder layer.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FormulaError {
    /// Requested column is not available in the data view.
    #[error("unknown column `{0}`")]
    UnknownColumn(String),

    /// A column returned by [`crate::DataView`] has a length different from
    /// [`crate::DataView::nrows`].
    #[error("column `{name}` has {actual} rows, expected {expected}")]
    ColumnLength {
        /// Column name.
        name: String,
        /// Expected number of rows.
        expected: usize,
        /// Actual number of rows.
        actual: usize,
    },

    /// Requested column type is not supported by this data view.
    #[error("column `{name}` does not support requested type `{requested}`")]
    UnsupportedColumnType {
        /// Column name.
        name: String,
        /// Requested logical type.
        requested: &'static str,
    },

    /// Numeric input contains a non-finite value.
    #[error("column `{name}` contains non-finite value at row {row}")]
    NonFiniteValue {
        /// Column name.
        name: String,
        /// Row index.
        row: usize,
    },

    /// Observation weight is not finite or is negative.
    #[error("weights column `{name}` has invalid value at row {row}")]
    InvalidWeight {
        /// Column name.
        name: String,
        /// Row index.
        row: usize,
    },

    /// Response value is outside the family domain.
    #[error("response column `{name}` has invalid {family} value at row {row}")]
    InvalidResponseDomain {
        /// Column name.
        name: String,
        /// Family name.
        family: &'static str,
        /// Row index.
        row: usize,
    },

    /// Prediction data contains a categorical level not seen during training.
    #[error("column `{name}` has unknown category `{level}` at row {row}")]
    UnknownCategoryLevel {
        /// Column name.
        name: String,
        /// Unknown level.
        level: String,
        /// Row index.
        row: usize,
    },

    /// The model spec did not include a response column.
    #[error("model spec is missing a response column")]
    MissingResponse,

    /// The data view has no rows.
    #[error("data view must contain at least one row")]
    EmptyData,

    /// A model parameter was configured more than once.
    #[error("parameter `{0}` terms were specified more than once")]
    DuplicateParameter(&'static str),

    /// Spline term construction failed.
    #[error(transparent)]
    Spline(#[from] SplineError),

    /// Fourier term construction failed.
    #[error(transparent)]
    Fourier(#[from] FourierError),

    /// Compiled core model validation failed.
    #[error(transparent)]
    Model(#[from] ModelError),
}
