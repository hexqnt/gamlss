use crate::transforms::{
    TargetTransform, TransformError, validate_positive, validate_positive_value,
};

/// Log transform for strictly positive targets.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Log;

impl TargetTransform for Log {
    type State = LogState;

    fn fit(y: &[f64]) -> Result<Self::State, TransformError> {
        validate_positive(y)?;
        Ok(LogState)
    }

    #[inline]
    fn transform(_: &Self::State, y: f64) -> f64 {
        y.ln()
    }

    #[inline]
    fn inverse(_: &Self::State, value: f64) -> f64 {
        value.exp()
    }

    #[inline]
    fn validate_transform_value(_: &Self::State, y: f64) -> Result<(), TransformError> {
        validate_positive_value(y)
    }
}

/// State for [`Log`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LogState;

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{Log, TargetTransform, TransformError};

    #[test]
    fn round_trips_positive_values() {
        let y = [1.0, 2.5];
        let (state, transformed) = Log::fit_transform(&y).unwrap();
        let restored = Log::inverse_slice(&state, &transformed).unwrap();

        for (actual, expected) in restored.iter().zip(y) {
            assert_relative_eq!(*actual, expected);
        }
    }

    #[test]
    fn rejects_invalid_domain() {
        assert_eq!(
            Log::fit(&[1.0, 0.0]).unwrap_err(),
            TransformError::NonPositiveValue
        );
        assert_eq!(
            Log::fit(&[f64::INFINITY]).unwrap_err(),
            TransformError::NonFiniteValue
        );
    }
}
