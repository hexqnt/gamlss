#![forbid(unsafe_code)]
//! Target transforms for GAMLSS modeling.

use thiserror::Error;

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

    /// Standardize transform получил нулевую дисперсию.
    #[error("target scale must be positive")]
    ZeroScale,
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
    fn transform_slice(state: &Self::State, y: &[f64]) -> Result<Vec<f64>, TransformError> {
        map_slice(y, validate_finite, |value| Self::transform(state, value))
    }

    /// Возвращает срез значений на исходную шкалу в новый `Vec`.
    ///
    /// # Errors
    ///
    /// Возвращает [`TransformError::NonFiniteValue`], если значения на
    /// transform-шкале содержат `NaN` или infinity.
    fn inverse_slice(state: &Self::State, values: &[f64]) -> Result<Vec<f64>, TransformError> {
        map_slice(values, validate_finite, |value| Self::inverse(state, value))
    }
}

/// Standardization transform: `(y - center) / scale`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Standardize;

/// State for [`Standardize`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StandardizeState {
    /// Training target mean.
    pub center: f64,
    /// Training target root-mean-square deviation.
    pub scale: f64,
}

impl TargetTransform for Standardize {
    type State = StandardizeState;

    #[allow(clippy::cast_precision_loss)]
    fn fit(y: &[f64]) -> Result<Self::State, TransformError> {
        validate_non_empty_finite(y)?;

        let center = y.iter().sum::<f64>() / y.len() as f64;
        let variance = y
            .iter()
            .map(|value| {
                let diff = value - center;
                diff * diff
            })
            .sum::<f64>()
            / y.len() as f64;
        let scale = variance.sqrt();
        if !scale.is_finite() || scale <= 0.0 {
            return Err(TransformError::ZeroScale);
        }

        Ok(StandardizeState { center, scale })
    }

    fn transform(state: &Self::State, y: f64) -> f64 {
        (y - state.center) / state.scale
    }

    fn inverse(state: &Self::State, value: f64) -> f64 {
        value.mul_add(state.scale, state.center)
    }
}

/// Log transform for strictly positive targets.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Log;

/// State for [`Log`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LogState;

impl TargetTransform for Log {
    type State = LogState;

    fn fit(y: &[f64]) -> Result<Self::State, TransformError> {
        validate_positive(y)?;
        Ok(LogState)
    }

    fn transform(_: &Self::State, y: f64) -> f64 {
        y.ln()
    }

    fn inverse(_: &Self::State, value: f64) -> f64 {
        value.exp()
    }

    fn transform_slice(state: &Self::State, y: &[f64]) -> Result<Vec<f64>, TransformError> {
        map_slice(y, validate_positive, |value| Self::transform(state, value))
    }
}

/// Identity transform that validates strictly positive targets.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IdentityPositive;

/// State for [`IdentityPositive`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IdentityPositiveState;

impl TargetTransform for IdentityPositive {
    type State = IdentityPositiveState;

    fn fit(y: &[f64]) -> Result<Self::State, TransformError> {
        validate_positive(y)?;
        Ok(IdentityPositiveState)
    }

    fn transform(_: &Self::State, y: f64) -> f64 {
        y
    }

    fn inverse(_: &Self::State, value: f64) -> f64 {
        value
    }

    fn transform_slice(state: &Self::State, y: &[f64]) -> Result<Vec<f64>, TransformError> {
        map_slice(y, validate_positive, |value| Self::transform(state, value))
    }
}

fn map_slice(
    values: &[f64],
    validate: impl FnOnce(&[f64]) -> Result<(), TransformError>,
    map: impl FnMut(f64) -> f64,
) -> Result<Vec<f64>, TransformError> {
    validate(values)?;
    Ok(values.iter().copied().map(map).collect())
}

fn validate_non_empty_finite(values: &[f64]) -> Result<(), TransformError> {
    if values.is_empty() {
        return Err(TransformError::EmptyInput);
    }
    validate_finite(values)
}

fn validate_finite(values: &[f64]) -> Result<(), TransformError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(TransformError::NonFiniteValue)
    }
}

fn validate_positive(values: &[f64]) -> Result<(), TransformError> {
    validate_non_empty_finite(values)?;
    if values.iter().all(|value| *value > 0.0) {
        Ok(())
    } else {
        Err(TransformError::NonPositiveValue)
    }
}

/// Наиболее часто используемые импорты из `gamlss-transform`.
pub mod prelude {
    pub use crate::{
        IdentityPositive, IdentityPositiveState, Log, LogState, Standardize, StandardizeState,
        TargetTransform, TransformError,
    };
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::{IdentityPositive, Log, Standardize, TargetTransform, TransformError};

    #[test]
    fn standardize_round_trips_values() {
        let y = [1.0, 2.0, 4.0];
        let (state, transformed) = Standardize::fit_transform(&y).unwrap();
        let restored = Standardize::inverse_slice(&state, &transformed).unwrap();

        for (actual, expected) in restored.iter().zip(y) {
            assert_relative_eq!(*actual, expected);
        }
    }

    #[test]
    fn positive_transforms_reject_non_positive_values() {
        assert_eq!(
            Log::fit(&[1.0, 0.0]).unwrap_err(),
            TransformError::NonPositiveValue
        );
        assert_eq!(
            IdentityPositive::fit(&[-1.0]).unwrap_err(),
            TransformError::NonPositiveValue
        );
    }

    #[test]
    fn transforms_reject_empty_and_non_finite_values() {
        assert_eq!(
            Standardize::fit(&[]).unwrap_err(),
            TransformError::EmptyInput
        );
        assert_eq!(
            Standardize::fit(&[1.0, f64::NAN]).unwrap_err(),
            TransformError::NonFiniteValue
        );
        assert_eq!(
            Log::fit(&[f64::INFINITY]).unwrap_err(),
            TransformError::NonFiniteValue
        );
    }

    #[test]
    fn standardize_rejects_zero_scale() {
        assert_eq!(
            Standardize::fit(&[2.0, 2.0]).unwrap_err(),
            TransformError::ZeroScale
        );
    }

    #[test]
    fn log_round_trips_positive_values() {
        let y = [1.0, 2.5];
        let (state, transformed) = Log::fit_transform(&y).unwrap();
        let restored = Log::inverse_slice(&state, &transformed).unwrap();

        for (actual, expected) in restored.iter().zip(y) {
            assert_relative_eq!(*actual, expected);
        }
    }
}
