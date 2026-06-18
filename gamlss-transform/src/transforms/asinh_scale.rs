use crate::{TargetTransform, TransformError, median_sorted, validate_non_empty_finite};

/// Signed inverse-hyperbolic-sine transform with a fitted robust scale.
///
/// `transform(y) = asinh(y / scale)`. Unlike a log transform, this is defined
/// for negative, zero and positive finite targets while still compressing large
/// magnitudes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AsinhScale;

/// State for [`AsinhScale`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AsinhScaleState {
    /// Positive scale used before `asinh`.
    pub scale: f64,
}

impl TargetTransform for AsinhScale {
    type State = AsinhScaleState;

    fn fit(y: &[f64]) -> Result<Self::State, TransformError> {
        validate_non_empty_finite(y)?;

        let mut abs_values: Vec<_> = y.iter().map(|value| value.abs()).collect();
        abs_values.sort_by(f64::total_cmp);
        let scale = median_sorted(&abs_values)
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(1.0);

        Ok(AsinhScaleState { scale })
    }

    #[inline(always)]
    fn transform(state: &Self::State, y: f64) -> f64 {
        (y / state.scale).asinh()
    }

    #[inline(always)]
    fn inverse(state: &Self::State, value: f64) -> f64 {
        value.sinh() * state.scale
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{AsinhScale, TargetTransform};

    #[test]
    fn round_trips_signed_values() {
        let y = [-100.0, -1.0, 0.0, 2.0, 50.0];
        let (state, transformed) = AsinhScale::fit_transform(&y).unwrap();
        let restored = AsinhScale::inverse_slice(&state, &transformed).unwrap();

        assert!(state.scale > 0.0);
        for (actual, expected) in restored.iter().zip(y) {
            assert_relative_eq!(*actual, expected, epsilon = 1.0e-10);
        }
    }

    #[test]
    fn uses_unit_scale_for_all_zero_targets() {
        let state = AsinhScale::fit(&[0.0, 0.0]).unwrap();

        assert_relative_eq!(state.scale, 1.0);
        assert_relative_eq!(AsinhScale::transform(&state, 0.0), 0.0);
    }
}
