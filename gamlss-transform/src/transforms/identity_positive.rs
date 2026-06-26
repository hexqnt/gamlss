use crate::{TargetTransform, TransformError, map_slice_into, validate_positive};

/// Identity transform that validates strictly positive targets.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IdentityPositive;

impl TargetTransform for IdentityPositive {
    type State = IdentityPositiveState;

    fn fit(y: &[f64]) -> Result<Self::State, TransformError> {
        validate_positive(y)?;
        Ok(IdentityPositiveState)
    }

    #[inline]
    fn transform(_: &Self::State, y: f64) -> f64 {
        y
    }

    #[inline]
    fn inverse(_: &Self::State, value: f64) -> f64 {
        value
    }

    #[inline]
    fn transform_slice(state: &Self::State, y: &[f64]) -> Result<Vec<f64>, TransformError> {
        let mut out = vec![0.0; y.len()];
        Self::transform_into(state, y, &mut out)?;
        Ok(out)
    }

    #[inline]
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

/// State for [`IdentityPositive`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IdentityPositiveState;

#[cfg(test)]
mod tests {
    use crate::{IdentityPositive, TargetTransform, TransformError};

    #[test]
    fn rejects_non_positive_values() {
        assert_eq!(
            IdentityPositive::fit(&[-1.0]).unwrap_err(),
            TransformError::NonPositiveValue
        );
    }

    #[test]
    fn leaves_positive_values_unchanged() {
        let y = [1.0, 2.0];
        let (state, transformed) = IdentityPositive::fit_transform(&y).unwrap();

        assert_eq!(transformed, y);
        assert_eq!(
            IdentityPositive::inverse_slice(&state, &transformed).unwrap(),
            y
        );
    }
}
