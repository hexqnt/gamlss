#![forbid(unsafe_code)]
//! Target transforms for GAMLSS modeling.

use thiserror::Error;

pub mod transforms;

pub use transforms::{
    AsinhScale, AsinhScaleState, IdentityPositive, IdentityPositiveState, Log, Log1pShift,
    Log1pShiftState, LogState, Standardize, StandardizeState,
};

/// Ошибки построения и применения target transforms.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TransformError {
    /// Target vector пуст.
    #[error("target vector must contain at least one value")]
    EmptyInput,

    /// Target содержит `NaN` или infinity.
    #[error("target contains a non-finite value")]
    NonFiniteValue,

    /// Transform требует строго положительный target.
    #[error("target value must be finite and > 0")]
    NonPositiveValue,

    /// Transform получил значение ниже обученной нижней границы.
    #[error("target value is below the fitted lower bound")]
    BelowLowerBound,

    /// Standardize transform получил нулевую дисперсию.
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

/// Transform целевой переменной с состоянием, оцениваемым на обучающем target.
pub trait TargetTransform {
    /// Состояние transform-а, сохраняемое вместе с обученной моделью.
    type State;

    /// Оценивает состояние transform-а по обучающему target.
    ///
    /// # Errors
    ///
    /// Возвращает [`TransformError`], если target пуст, содержит не-finite
    /// значения или нарушает domain-инварианты конкретного transform-а.
    fn fit(y: &[f64]) -> Result<Self::State, TransformError>;
    /// Преобразует одно значение target.
    fn transform(state: &Self::State, y: f64) -> f64;
    /// Возвращает значение на исходную шкалу.
    fn inverse(state: &Self::State, value: f64) -> f64;

    /// Оценивает состояние и преобразует весь target.
    ///
    /// # Errors
    ///
    /// Возвращает ошибку из [`Self::fit`] или [`Self::transform_slice`].
    #[inline]
    fn fit_transform(y: &[f64]) -> Result<(Self::State, Vec<f64>), TransformError> {
        let state = Self::fit(y)?;
        let transformed = Self::transform_slice(&state, y)?;
        Ok((state, transformed))
    }

    /// Преобразует срез target в новый `Vec`.
    ///
    /// # Errors
    ///
    /// Возвращает [`TransformError::NonFiniteValue`], если вход содержит
    /// `NaN` или infinity. Конкретные transform-ы могут усиливать проверку
    /// domain-а, например требовать строго положительные значения.
    #[inline]
    fn transform_slice(state: &Self::State, y: &[f64]) -> Result<Vec<f64>, TransformError> {
        let mut out = vec![0.0; y.len()];
        Self::transform_into(state, y, &mut out)?;
        Ok(out)
    }

    /// Преобразует срез target в caller-provided output buffer.
    ///
    /// # Errors
    ///
    /// Возвращает [`TransformError::LengthMismatch`], если длина `out` не
    /// совпадает с длиной входа. Возвращает [`TransformError::NonFiniteValue`],
    /// если вход содержит `NaN` или infinity. Конкретные transform-ы могут
    /// усиливать проверку domain-а.
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

    /// Возвращает срез значений на исходную шкалу в новый `Vec`.
    ///
    /// # Errors
    ///
    /// Возвращает [`TransformError::NonFiniteValue`], если значения на
    /// transform-шкале содержат `NaN` или infinity.
    #[inline]
    fn inverse_slice(state: &Self::State, values: &[f64]) -> Result<Vec<f64>, TransformError> {
        let mut out = vec![0.0; values.len()];
        Self::inverse_into(state, values, &mut out)?;
        Ok(out)
    }

    /// Возвращает transform-scale values на исходную шкалу в caller-provided buffer.
    ///
    /// # Errors
    ///
    /// Возвращает [`TransformError::LengthMismatch`], если длина `out` не
    /// совпадает с длиной входа. Возвращает [`TransformError::NonFiniteValue`],
    /// если вход содержит `NaN` или infinity.
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

pub(crate) fn validate_output_len(expected: usize, actual: usize) -> Result<(), TransformError> {
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

pub(crate) fn median_sorted(values: &[f64]) -> Option<f64> {
    match values.len() {
        0 => None,
        len if len % 2 == 1 => Some(values[len / 2]),
        len => {
            let upper = len / 2;
            Some((values[upper - 1] + values[upper]) / 2.0)
        }
    }
}

/// Наиболее часто используемые импорты из `gamlss-transform`.
pub mod prelude {
    pub use crate::{
        AsinhScale, AsinhScaleState, IdentityPositive, IdentityPositiveState, Log, Log1pShift,
        Log1pShiftState, LogState, Standardize, StandardizeState, TargetTransform, TransformError,
    };
}
