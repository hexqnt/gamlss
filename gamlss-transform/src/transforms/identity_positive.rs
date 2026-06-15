use crate::{TargetTransform, TransformError, map_slice, validate_positive};

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
