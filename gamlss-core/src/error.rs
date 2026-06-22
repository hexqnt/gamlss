use thiserror::Error;

use crate::model::ParameterLayout;

/// Errors for GAMLSS model construction and validation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ModelError {
    /// Response vector is empty.
    #[error("response vector must contain at least one observation")]
    EmptyResponse,

    /// A scalar model parameter has an invalid value.
    #[error("{parameter} must be {expected}")]
    InvalidParameter {
        /// Parameter name.
        parameter: &'static str,
        /// Expected invariant.
        expected: &'static str,
    },

    /// Dense matrix received an incorrect number of row-major values.
    ///
    /// The provided `actual_values` count does not match `nrows * ncols`.
    #[error("design matrix has {actual_values} values, expected {expected_values}")]
    DesignSize {
        /// Expected number of values.
        expected_values: usize,
        /// Actual number of values.
        actual_values: usize,
    },

    /// Design matrix dimensions do not fit in `usize`.
    #[error("arithmetic overflow while computing {context}")]
    ArithmeticOverflow {
        /// Description of the computed size.
        context: &'static str,
    },

    /// Design matrix row count does not match the response length.
    #[error(
        "{parameter} design has {actual_rows} rows, expected {expected_rows} rows from response"
    )]
    DesignRowMismatch {
        /// Name or role of the parameter being checked.
        parameter: &'static str,
        /// Expected number of rows.
        expected_rows: usize,
        /// Actual number of rows.
        actual_rows: usize,
    },

    /// Response length does not match the expected length.
    #[error("response length is {actual}, expected {expected}")]
    ResponseLength {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },

    /// Observation weights length does not match the response length.
    #[error("weights length is {actual}, expected {expected}")]
    WeightLength {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },

    /// An observation weight has an invalid value.
    #[error("weight at index {index} must be finite and >= 0")]
    InvalidWeight {
        /// Index of the invalid weight.
        index: usize,
    },

    /// A scalar observation has a non-finite value.
    #[error("scalar observation at index {index} must be finite")]
    InvalidObservation {
        /// Index of the invalid observation.
        index: usize,
    },

    /// A dense design matrix entry has a non-finite value.
    #[error("design matrix value at row-major index {index} must be finite")]
    InvalidDesignValue {
        /// Row-major index of the invalid value.
        index: usize,
    },

    /// A product-block multiplier has a non-finite value.
    #[error("product multiplier at index {index} must be finite")]
    InvalidMultiplier {
        /// Index of the invalid multiplier.
        index: usize,
    },

    /// Beta vector length does not match the model coefficient count.
    #[error("beta length is {actual}, expected {expected}")]
    BetaLength {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },

    /// Gradient vector length does not match the model coefficient count.
    #[error("gradient length is {actual}, expected {expected}")]
    GradientLength {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },

    /// Predictor row index is outside the model's observation range.
    #[error("row index {row} is out of bounds for {nrows} rows")]
    RowOutOfBounds {
        /// Requested row index.
        row: usize,
        /// Number of rows in the model.
        nrows: usize,
    },

    /// Prediction blocks have a different coefficient layout.
    #[error(
        "prediction blocks have incompatible parameter layout: expected {expected:?}, got {got:?}"
    )]
    PredictionLayoutMismatch {
        /// Layout of the training model.
        expected: ParameterLayout,
        /// Layout of the provided prediction blocks.
        got: ParameterLayout,
    },

    /// Two parameter blocks use overlapping beta ranges.
    #[error("{first} parameter block overlaps with {second} parameter block")]
    BlockOverlap {
        /// First overlapping block.
        first: &'static str,
        /// Second overlapping block.
        second: &'static str,
    },

    /// Parameter block coefficient range does not fit in `usize`.
    #[error("{parameter} parameter block range overflows: offset {offset}, len {len}")]
    BlockRangeOverflow {
        /// Parameter name.
        parameter: &'static str,
        /// Block start position.
        offset: usize,
        /// Block length.
        len: usize,
    },

    /// Model does not contain a parameter block with the given name.
    ///
    /// Raised when attempting to create a `BlockObjective` for a parameter
    /// that is not present in the model.
    #[error("model has no parameter block named {name:?}")]
    UnknownParameter {
        /// Name of the requested parameter.
        name: &'static str,
    },

    /// A penalty references a coefficient index outside the model vector.
    #[error("penalty coefficient index {index} is out of bounds for dimension {dim}")]
    PenaltyIndexOutOfBounds {
        /// Referenced coefficient index.
        index: usize,
        /// Full coefficient dimension.
        dim: usize,
    },

    /// A segment penalty range is outside the coefficient vector.
    #[error("penalty range {start}..{end} is out of bounds for dimension {dim}")]
    PenaltyRangeOutOfBounds {
        /// Start of the requested range.
        start: usize,
        /// End of the requested range.
        end: usize,
        /// Full coefficient dimension.
        dim: usize,
    },
}
