use crate::transforms::{TargetTransform, TransformError, validate_non_empty_finite};

/// Standardization by the training mean and root-mean-square deviation.
///
/// For $n$ training targets,
///
/// $$
/// c=\frac{1}{n}\sum_{i=1}^{n}y_i,
/// \qquad
/// s=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-c)^2},
/// $$
///
/// and the persisted transform is
///
/// $$
/// T(y)=\frac{y-c}{s},
/// \qquad
/// T^{-1}(z)=sz+c.
/// $$
///
/// The fitted state stores $c$ in [`StandardizeState::center`] and $s$ in [`StandardizeState::scale`]; $z=T(y)$ is the transform-scale value. The scale uses the population denominator $n$, not the sample denominator $n-1$, and fitting rejects $s=0$.
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Standardize;

impl TargetTransform for Standardize {
    type State = StandardizeState;

    #[allow(clippy::cast_precision_loss)]
    fn fit(y: &[f64]) -> Result<Self::State, TransformError> {
        validate_non_empty_finite(y)?;

        let mut count = 0.0;
        let mut center = 0.0;
        let mut sum_squares = 0.0;
        for value in y.iter().copied() {
            count += 1.0;
            let delta = value - center;
            center += delta / count;
            let centered = value - center;
            sum_squares += delta * centered;
        }

        let variance = sum_squares / count;
        let scale = variance.sqrt();
        if !scale.is_finite() || scale <= 0.0 {
            return Err(TransformError::ZeroScale);
        }

        Ok(StandardizeState { center, scale })
    }

    #[inline]
    fn transform(state: &Self::State, y: f64) -> f64 {
        (y - state.center) / state.scale
    }

    #[inline]
    fn inverse(state: &Self::State, value: f64) -> f64 {
        value.mul_add(state.scale, state.center)
    }
}

/// State for [`Standardize`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StandardizeState {
    /// Training target mean.
    pub center: f64,
    /// Training target root-mean-square deviation.
    pub scale: f64,
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

    #[test]
    fn handles_large_offset_values_stably() {
        let y = [1.0e12, 1.0e12 + 2.0, 1.0e12 + 4.0];
        let (state, transformed) = Standardize::fit_transform(&y).unwrap();
        let restored = Standardize::inverse_slice(&state, &transformed).unwrap();

        assert!(state.center.is_finite());
        assert!(state.scale.is_finite());
        assert!(state.scale > 0.0);
        for (actual, expected) in restored.iter().zip(y) {
            assert_relative_eq!(*actual, expected, epsilon = 1.0e-6);
        }
    }
}
