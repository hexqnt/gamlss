use thiserror::Error;

use crate::{
    DynamicLayoutKey,
    model::{ParameterDescriptor, ParameterLayout},
};

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

    /// A flat dense observation buffer is not divisible into equal-width rows.
    #[error(
        "dense observation buffer has {actual_values} values, which is not divisible by row width {row_width}"
    )]
    DenseObservationSize {
        /// Number of values in the flat buffer.
        actual_values: usize,
        /// Requested number of values per row.
        row_width: usize,
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

    /// Prediction blocks have the same coarse slices but a different full identity.
    #[error(
        "prediction blocks have incompatible structured parameter identity: expected key {expected_key:?} and descriptors {expected_descriptors:?}, got key {got_key:?} and descriptors {got_descriptors:?}"
    )]
    PredictionLayoutIdentityMismatch {
        /// Runtime topology key of the training blocks, when applicable.
        expected_key: Option<DynamicLayoutKey>,
        /// Runtime topology key of the prediction blocks, when applicable.
        got_key: Option<DynamicLayoutKey>,
        /// Ordered structured descriptors of the training blocks.
        expected_descriptors: Vec<ParameterDescriptor>,
        /// Ordered structured descriptors of the prediction blocks.
        got_descriptors: Vec<ParameterDescriptor>,
    },

    /// Runtime blocks were built for another configuration of the same family type.
    #[error("dynamic blocks have layout key {got:?}, expected {expected:?}")]
    DynamicLayoutMismatch {
        /// Runtime topology required by the model family instance.
        expected: DynamicLayoutKey,
        /// Runtime topology captured when the blocks were built.
        got: DynamicLayoutKey,
    },

    /// Runtime coordinate metadata differs from the model family instance.
    #[error("dynamic coordinate {index} has descriptor {got:?}, expected {expected:?}")]
    DynamicCoordinateMismatch {
        /// Zero-based runtime coordinate index.
        index: usize,
        /// Descriptor required by the model family instance.
        expected: ParameterDescriptor,
        /// Descriptor captured when the blocks were built.
        got: ParameterDescriptor,
    },

    /// A runtime family initializer returned the wrong coordinate count.
    #[error("dynamic initializer returned {actual} eta values, expected {expected}")]
    DynamicInitialValueCount {
        /// Runtime coordinate count required by the family layout.
        expected: usize,
        /// Number of values returned by the family initializer.
        actual: usize,
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

    /// Parameter block coefficient range is outside the model vector.
    #[error(
        "{parameter} parameter block range {start}..{end} is out of bounds for dimension {dim}"
    )]
    BlockRangeOutOfBounds {
        /// Parameter name.
        parameter: &'static str,
        /// Start of the requested range.
        start: usize,
        /// End of the requested range.
        end: usize,
        /// Full coefficient dimension.
        dim: usize,
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

    /// A semantic parameter query matched more than one layout entry.
    #[error("parameter {name:?} is ambiguous: matched {matches} layout entries")]
    AmbiguousParameter {
        /// Parameter role used by the query.
        name: String,
        /// Number of matching blocks or descriptors.
        matches: usize,
    },

    /// A descriptor index is outside the model's ordered descriptor layout.
    #[error("parameter descriptor index {index} is out of bounds for {count} descriptors")]
    ParameterDescriptorIndexOutOfBounds {
        /// Requested stable index in this model layout.
        index: usize,
        /// Number of descriptors in the model layout.
        count: usize,
    },

    /// A full descriptor does not belong to the model layout.
    #[error("model has no parameter descriptor {descriptor:?}")]
    UnknownParameterDescriptor {
        /// Descriptor requested by the caller.
        descriptor: ParameterDescriptor,
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
