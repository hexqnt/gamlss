//! Built-in target transforms.

use std::marker::PhantomData;

use thiserror::Error;

pub use asinh_scale::{AsinhScale, AsinhScaleState};
pub use box_cox::{BoxCox, BoxCoxFixed, BoxCoxState};
pub use identity_positive::{IdentityPositive, IdentityPositiveState};
pub use log::{Log, LogState};
pub use log1p_shift::{Log1pShift, Log1pShiftState};
pub use max_abs_scale::{MaxAbsScale, MaxAbsScaleState};
pub use min_max_scale::{MinMaxScale, MinMaxScaleState};
pub use quantile::{QuantileNormal, QuantileState, QuantileUniform};
pub use robust_standardize::{RobustStandardize, RobustStandardizeState};
pub use standardize::{Standardize, StandardizeState};
pub use yeo_johnson::{YeoJohnson, YeoJohnsonFixed, YeoJohnsonState};

pub mod asinh_scale;
pub mod box_cox;
pub mod identity_positive;
pub mod log;
pub mod log1p_shift;
pub mod max_abs_scale;
pub mod min_max_scale;
mod power;
pub mod quantile;
pub mod robust_standardize;
pub mod standardize;
pub mod yeo_johnson;

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

    /// Transform received a value above the fitted upper bound.
    #[error("target value is above the fitted upper bound")]
    AboveUpperBound,

    /// Fitted or configured scale is zero or non-positive.
    #[error("target scale must be positive")]
    ZeroScale,

    /// Transform parameter is invalid.
    #[error("invalid transform parameter: {name}")]
    InvalidParameter {
        /// Parameter name.
        name: &'static str,
    },

    /// Output buffer length does not match the input length.
    #[error("output length is {actual}, expected {expected}")]
    LengthMismatch {
        /// Expected output length.
        expected: usize,
        /// Actual output length.
        actual: usize,
    },
}

/// Static composition of two target transforms.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Then<First, Second>(PhantomData<(First, Second)>);

impl<First, Second> TargetTransform for Then<First, Second>
where
    First: TargetTransform,
    Second: TargetTransform,
{
    type State = ThenState<First::State, Second::State>;

    fn fit(y: &[f64]) -> Result<Self::State, TransformError> {
        let (first, transformed) = First::fit_transform(y)?;
        let second = Second::fit(&transformed)?;
        Ok(ThenState { first, second })
    }

    #[inline]
    fn transform(state: &Self::State, y: f64) -> f64 {
        Second::transform(&state.second, First::transform(&state.first, y))
    }

    #[inline]
    fn inverse(state: &Self::State, value: f64) -> f64 {
        First::inverse(&state.first, Second::inverse(&state.second, value))
    }

    #[inline]
    fn checked_transform(state: &Self::State, y: f64) -> Result<f64, TransformError> {
        let first = First::checked_transform(&state.first, y)?;
        Second::checked_transform(&state.second, first)
    }

    #[inline]
    fn checked_inverse(state: &Self::State, value: f64) -> Result<f64, TransformError> {
        let second = Second::checked_inverse(&state.second, value)?;
        First::checked_inverse(&state.first, second)
    }
}

/// State for [`Then`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThenState<FirstState, SecondState> {
    /// State of the first transform.
    pub first: FirstState,
    /// State of the second transform.
    pub second: SecondState,
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
    /// Transforms a single target value without validation.
    fn transform(state: &Self::State, y: f64) -> f64;
    /// Returns a value to the original scale without validation.
    fn inverse(state: &Self::State, value: f64) -> f64;

    /// Validates a single original-scale value before transforming it.
    ///
    /// # Errors
    ///
    /// Returns [`TransformError::NonFiniteValue`] by default for non-finite
    /// values. Specific transforms may strengthen the domain check.
    #[inline]
    fn validate_transform_value(_state: &Self::State, y: f64) -> Result<(), TransformError> {
        validate_finite_value(y)
    }

    /// Validates a single transform-scale value before inverting it.
    ///
    /// # Errors
    ///
    /// Returns [`TransformError::NonFiniteValue`] for non-finite values.
    #[inline]
    fn validate_inverse_value(_state: &Self::State, value: f64) -> Result<(), TransformError> {
        validate_finite_value(value)
    }

    /// Validates and transforms a single target value.
    ///
    /// # Errors
    ///
    /// Returns [`TransformError`] if the value violates the transform domain.
    #[inline]
    fn checked_transform(state: &Self::State, y: f64) -> Result<f64, TransformError> {
        Self::validate_transform_value(state, y)?;
        Ok(Self::transform(state, y))
    }

    /// Validates and returns a value to the original scale.
    ///
    /// # Errors
    ///
    /// Returns [`TransformError`] if the transform-scale value is invalid.
    #[inline]
    fn checked_inverse(state: &Self::State, value: f64) -> Result<f64, TransformError> {
        Self::validate_inverse_value(state, value)?;
        Ok(Self::inverse(state, value))
    }

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
    /// or infinity. Specific transforms may strengthen the domain check.
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
        map_slice_into(y, out, |value| Self::checked_transform(state, value))
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
        map_slice_into(values, out, |value| Self::checked_inverse(state, value))
    }
}

pub(crate) fn map_slice_into(
    values: &[f64],
    out: &mut [f64],
    mut map: impl FnMut(f64) -> Result<f64, TransformError>,
) -> Result<(), TransformError> {
    validate_output_len(values.len(), out.len())?;
    for (out, value) in out.iter_mut().zip(values.iter().copied()) {
        *out = map(value)?;
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
    if values.iter().copied().all(f64::is_finite) {
        Ok(())
    } else {
        Err(TransformError::NonFiniteValue)
    }
}

pub(crate) const fn validate_finite_value(value: f64) -> Result<(), TransformError> {
    if value.is_finite() {
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

pub(crate) fn validate_positive_value(value: f64) -> Result<(), TransformError> {
    validate_finite_value(value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(TransformError::NonPositiveValue)
    }
}

pub(crate) fn validate_shifted_non_negative_value(
    value: f64,
    shift: f64,
) -> Result<(), TransformError> {
    validate_finite_value(value)?;
    if value + shift >= 0.0 {
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

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub(crate) fn quantile_sorted(values: &[f64], probability: f64) -> Option<f64> {
    if values.is_empty() || !(0.0..=1.0).contains(&probability) || !probability.is_finite() {
        return None;
    }
    if values.len() == 1 {
        return Some(values[0]);
    }

    let index = probability * (values.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    if lower == upper {
        Some(values[lower])
    } else {
        let weight = index - lower as f64;
        Some(values[lower] + weight * (values[upper] - values[lower]))
    }
}

pub(crate) const fn lambda_from_ratio(
    numerator: i32,
    denominator: i32,
) -> Result<f64, TransformError> {
    if denominator == 0 {
        Err(TransformError::InvalidParameter {
            name: "denominator",
        })
    } else {
        Ok(numerator as f64 / denominator as f64)
    }
}
