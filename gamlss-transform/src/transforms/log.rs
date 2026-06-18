use crate::{TargetTransform, TransformError, map_slice_into, validate_positive};

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
        let mut out = vec![0.0; y.len()];
        Self::transform_into(state, y, &mut out)?;
        Ok(out)
    }

    fn transform_into(
        state: &Self::State,
        y: &[f64],
        out: &mut [f64],
    ) -> Result<(), TransformError> {
        map_slice_into(y, out, validate_positive, |value| {
            Self::transform(state, value)
        })
    }
}

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
