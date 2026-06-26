use crate::transforms::{TargetTransform, TransformError, validate_non_empty_finite};

/// Min-max transform: `(y - min) / (max - min)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MinMaxScale;

impl TargetTransform for MinMaxScale {
    type State = MinMaxScaleState;

    fn fit(y: &[f64]) -> Result<Self::State, TransformError> {
        validate_non_empty_finite(y)?;

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for value in y.iter().copied() {
            min = min.min(value);
            max = max.max(value);
        }

        let scale = max - min;
        if !scale.is_finite() || scale <= 0.0 {
            return Err(TransformError::ZeroScale);
        }

        Ok(MinMaxScaleState { min, scale })
    }

    #[inline]
    fn transform(state: &Self::State, y: f64) -> f64 {
        (y - state.min) / state.scale
    }

    #[inline]
    fn inverse(state: &Self::State, value: f64) -> f64 {
        value.mul_add(state.scale, state.min)
    }
}

/// State for [`MinMaxScale`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinMaxScaleState {
    /// Training target minimum.
    pub min: f64,
    /// Training target range.
    pub scale: f64,
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{MinMaxScale, TargetTransform, TransformError};

    #[test]
    fn round_trips_values_without_clipping() {
        let y = [-1.0, 1.0, 3.0];
        let (state, transformed) = MinMaxScale::fit_transform(&y).unwrap();
        let restored = MinMaxScale::inverse_slice(&state, &transformed).unwrap();

        assert_eq!(transformed, vec![0.0, 0.5, 1.0]);
        assert_relative_eq!(MinMaxScale::transform(&state, 5.0), 1.5);
        for (actual, expected) in restored.iter().zip(y) {
            assert_relative_eq!(*actual, expected);
        }
    }

    #[test]
    fn rejects_constant_target() {
        assert_eq!(
            MinMaxScale::fit(&[2.0, 2.0]).unwrap_err(),
            TransformError::ZeroScale
        );
    }
}
