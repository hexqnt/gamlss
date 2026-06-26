use crate::transforms::{
    TargetTransform, TransformError, median_sorted, quantile_sorted, validate_non_empty_finite,
};

/// Robust standardization transform: `(y - median) / IQR`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RobustStandardize;

impl TargetTransform for RobustStandardize {
    type State = RobustStandardizeState;

    fn fit(y: &[f64]) -> Result<Self::State, TransformError> {
        validate_non_empty_finite(y)?;

        let mut sorted = y.to_vec();
        sorted.sort_by(f64::total_cmp);

        let center = median_sorted(&sorted).expect("non-empty sorted target has a median");
        let q1 = quantile_sorted(&sorted, 0.25).expect("non-empty sorted target has q1");
        let q3 = quantile_sorted(&sorted, 0.75).expect("non-empty sorted target has q3");
        let scale = q3 - q1;
        if !scale.is_finite() || scale <= 0.0 {
            return Err(TransformError::ZeroScale);
        }

        Ok(RobustStandardizeState { center, scale })
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

/// State for [`RobustStandardize`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RobustStandardizeState {
    /// Training target median.
    pub center: f64,
    /// Training target interquartile range.
    pub scale: f64,
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{RobustStandardize, TargetTransform, TransformError};

    #[test]
    fn round_trips_values() {
        let y = [-10.0, -1.0, 0.0, 2.0, 100.0];
        let (state, transformed) = RobustStandardize::fit_transform(&y).unwrap();
        let restored = RobustStandardize::inverse_slice(&state, &transformed).unwrap();

        assert_relative_eq!(state.center, 0.0);
        assert_relative_eq!(state.scale, 3.0);
        for (actual, expected) in restored.iter().zip(y) {
            assert_relative_eq!(*actual, expected);
        }
    }

    #[test]
    fn rejects_zero_iqr() {
        assert_eq!(
            RobustStandardize::fit(&[2.0, 2.0, 2.0]).unwrap_err(),
            TransformError::ZeroScale
        );
    }
}
