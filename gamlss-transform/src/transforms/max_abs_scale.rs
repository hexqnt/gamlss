use crate::transforms::{TargetTransform, TransformError, validate_non_empty_finite};

/// Max-absolute transform fitted from the training target.
///
/// With $s=\max_i|y_i|$,
///
/// $$
/// T(y)=\frac{y}{s},
/// \qquad
/// T^{-1}(z)=sz.
/// $$
///
/// [`MaxAbsScaleState::scale`] stores $s$ and $z=T(y)$ is the transform-scale value. All-zero training targets use $s=1$ so that the transform remains invertible.
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MaxAbsScale;

impl TargetTransform for MaxAbsScale {
    type State = MaxAbsScaleState;

    fn fit(y: &[f64]) -> Result<Self::State, TransformError> {
        validate_non_empty_finite(y)?;

        let max_abs = y.iter().copied().map(f64::abs).fold(0.0, f64::max);
        let scale = if max_abs > 0.0 { max_abs } else { 1.0 };
        Ok(MaxAbsScaleState { scale })
    }

    #[inline]
    fn transform(state: &Self::State, y: f64) -> f64 {
        y / state.scale
    }

    #[inline]
    fn inverse(state: &Self::State, value: f64) -> f64 {
        value * state.scale
    }
}

/// State for [`MaxAbsScale`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaxAbsScaleState {
    /// Maximum absolute training target value, or one for all-zero targets.
    pub scale: f64,
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{MaxAbsScale, TargetTransform};

    #[test]
    fn round_trips_values() {
        let y = [-2.0, 0.0, 4.0];
        let (state, transformed) = MaxAbsScale::fit_transform(&y).unwrap();
        let restored = MaxAbsScale::inverse_slice(&state, &transformed).unwrap();

        assert_relative_eq!(state.scale, 4.0);
        assert_eq!(transformed, vec![-0.5, 0.0, 1.0]);
        for (actual, expected) in restored.iter().zip(y) {
            assert_relative_eq!(*actual, expected);
        }
    }

    #[test]
    fn uses_unit_scale_for_all_zero_targets() {
        let (state, transformed) = MaxAbsScale::fit_transform(&[0.0, 0.0]).unwrap();

        assert_relative_eq!(state.scale, 1.0);
        assert_eq!(transformed, vec![0.0, 0.0]);
    }
}
