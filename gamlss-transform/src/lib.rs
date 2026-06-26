#![forbid(unsafe_code)]
//! Target transforms for GAMLSS modeling.

use thiserror::Error;

pub use transforms::{
    AsinhScale, AsinhScaleState, IdentityPositive, IdentityPositiveState, Log, Log1pShift,
    Log1pShiftState, LogState, Standardize, StandardizeState,
};

pub mod transforms;

/// Most commonly used imports from `gamlss-transform`.
pub mod prelude {
    pub use crate::{
        AsinhScale, AsinhScaleState, IdentityPositive, IdentityPositiveState, Log, Log1pShift,
        Log1pShiftState, LogState, Standardize, StandardizeState, TargetTransform, TransformError,
    };
}
/// Errors for building and applying target transforms.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TransformError {
    /// Target vector is empty.
    #[error("target vector must contain at least one value")]
    EmptyInput,

    /// Target contains `NaN` or infinity.
    #[error("target contains a non-finite value")]
    NonFiniteValue,

    /// Transform requires strictly positive targets.
    #[error("target value must be finite and > 0")]
    NonPositiveValue,

    /// Transform received a value below the fitted lower bound.
    #[error("target value is below the fitted lower bound")]
    BelowLowerBound,

    /// Standardize transform received zero variance.
    #[error("target scale must be positive")]
    ZeroScale,

    /// Output buffer length does not match the input length.
    #[error("output length is {actual}, expected {expected}")]
    LengthMismatch {
        /// Expected output length.
        expected: usize,
        /// Actual output length.
        actual: usize,
    },
}

/// Transform of the target variable with state estimated on the training target.
pub trait TargetTransform {
    /// State of the transform, persisted alongside the fitted model.
    type State;

    /// Estimates the transform state from the training target.
    ///
    /// # Errors
    ///
    /// Returns [`TransformError`] if the target is empty, contains non-finite
    /// values, or violates domain invariants of the specific transform.
    fn fit(y: &[f64]) -> Result<Self::State, TransformError>;
    /// Transforms a single target value.
    fn transform(state: &Self::State, y: f64) -> f64;
    /// Returns a value to the original scale.
    fn inverse(state: &Self::State, value: f64) -> f64;

    /// Estimates the state and transforms the entire target.
    ///
    /// # Errors
    ///
    /// Returns an error from [`Self::fit`] or [`Self::transform_slice`].
    #[inline]
    fn fit_transform(y: &[f64]) -> Result<(Self::State, Vec<f64>), TransformError> {
        let state = Self::fit(y)?;
        let transformed = Self::transform_slice(&state, y)?;
        Ok((state, transformed))
    }

    /// Transforms a target slice into a new `Vec`.
    ///
    /// # Errors
    ///
    /// Returns [`TransformError::NonFiniteValue`] if the input contains `NaN`
    /// or infinity. Specific transforms may strengthen the domain check, e.g.
    /// require strictly positive values.
    #[inline]
    fn transform_slice(state: &Self::State, y: &[f64]) -> Result<Vec<f64>, TransformError> {
        let mut out = vec![0.0; y.len()];
        Self::transform_into(state, y, &mut out)?;
        Ok(out)
    }

    /// Transforms a target slice into a caller-provided output buffer.
    ///
    /// # Errors
    ///
    /// Returns [`TransformError::LengthMismatch`] if the `out` length does not
    /// match the input length. Returns [`TransformError::NonFiniteValue`] if
    /// the input contains `NaN` or infinity. Specific transforms may
    /// strengthen the domain check.
    #[inline]
    fn transform_into(
        state: &Self::State,
        y: &[f64],
        out: &mut [f64],
    ) -> Result<(), TransformError> {
        map_slice_into(y, out, validate_finite, |value| {
            Self::transform(state, value)
        })
    }

    /// Returns a slice of values to the original scale into a new `Vec`.
    ///
    /// # Errors
    ///
    /// Returns [`TransformError::NonFiniteValue`] if the transform-scale values
    /// contain `NaN` or infinity.
    #[inline]
    fn inverse_slice(state: &Self::State, values: &[f64]) -> Result<Vec<f64>, TransformError> {
        let mut out = vec![0.0; values.len()];
        Self::inverse_into(state, values, &mut out)?;
        Ok(out)
    }

    /// Returns transform-scale values to the original scale into a caller-provided
    /// buffer.
    ///
    /// # Errors
    ///
    /// Returns [`TransformError::LengthMismatch`] if the `out` length does not
    /// match the input length. Returns [`TransformError::NonFiniteValue`] if
    /// the input contains `NaN` or infinity.
    #[inline]
    fn inverse_into(
        state: &Self::State,
        values: &[f64],
        out: &mut [f64],
    ) -> Result<(), TransformError> {
        map_slice_into(values, out, validate_finite, |value| {
            Self::inverse(state, value)
        })
    }
}

pub(crate) fn map_slice_into(
    values: &[f64],
    out: &mut [f64],
    validate: impl FnOnce(&[f64]) -> Result<(), TransformError>,
    mut map: impl FnMut(f64) -> f64,
) -> Result<(), TransformError> {
    validate_output_len(values.len(), out.len())?;
    validate(values)?;
    for (out, value) in out.iter_mut().zip(values.iter().copied()) {
        *out = map(value);
    }
    Ok(())
}

pub(crate) const fn validate_output_len(
    expected: usize,
    actual: usize,
) -> Result<(), TransformError> {
    if actual == expected {
        Ok(())
    } else {
        Err(TransformError::LengthMismatch { expected, actual })
    }
}

pub(crate) fn validate_non_empty_finite(values: &[f64]) -> Result<(), TransformError> {
    if values.is_empty() {
        return Err(TransformError::EmptyInput);
    }
    validate_finite(values)
}

pub(crate) fn validate_finite(values: &[f64]) -> Result<(), TransformError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(TransformError::NonFiniteValue)
    }
}

pub(crate) fn validate_positive(values: &[f64]) -> Result<(), TransformError> {
    validate_non_empty_finite(values)?;
    if values.iter().all(|value| *value > 0.0) {
        Ok(())
    } else {
        Err(TransformError::NonPositiveValue)
    }
}

pub(crate) fn validate_shifted_non_negative(
    values: &[f64],
    shift: f64,
) -> Result<(), TransformError> {
    validate_finite(values)?;
    if values.iter().all(|value| *value + shift >= 0.0) {
        Ok(())
    } else {
        Err(TransformError::BelowLowerBound)
    }
}

pub(crate) const fn median_sorted(values: &[f64]) -> Option<f64> {
    match values.len() {
        0 => None,
        len if len % 2 == 1 => Some(values[len / 2]),
        len => {
            let upper = len / 2;
            Some(values[upper - 1].midpoint(values[upper]))
        }
    }
}
