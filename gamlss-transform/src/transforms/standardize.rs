use crate::{TargetTransform, TransformError, validate_non_empty_finite};

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

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{Standardize, TargetTransform, TransformError};

    #[test]
    fn round_trips_values() {
        let y = [1.0, 2.0, 4.0];
        let (state, transformed) = Standardize::fit_transform(&y).unwrap();
        let restored = Standardize::inverse_slice(&state, &transformed).unwrap();

        for (actual, expected) in restored.iter().zip(y) {
            assert_relative_eq!(*actual, expected);
        }
    }

    #[test]
    fn rejects_empty_and_non_finite_values() {
        assert_eq!(
            Standardize::fit(&[]).unwrap_err(),
            TransformError::EmptyInput
        );
        assert_eq!(
            Standardize::fit(&[1.0, f64::NAN]).unwrap_err(),
            TransformError::NonFiniteValue
        );
    }

    #[test]
    fn rejects_zero_scale() {
        assert_eq!(
            Standardize::fit(&[2.0, 2.0]).unwrap_err(),
            TransformError::ZeroScale
        );
    }
}
