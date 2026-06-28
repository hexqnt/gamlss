use std::marker::PhantomData;

use crate::transforms::{
    TargetTransform, TransformError, lambda_from_ratio, validate_finite_value, validate_positive,
    validate_positive_value,
};

use super::power::{ProfileAccumulator, fit_lambda};

const LAMBDA_EPSILON: f64 = 1.0e-12;

/// Box-Cox transform with a fitted lambda parameter for strictly positive targets.
///
/// The lambda parameter is fitted by deterministic profile-likelihood search on
/// `[-5, 5]`. [`TargetTransform::checked_inverse`] rejects transform-scale
/// values outside the inverse domain for non-zero lambdas.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BoxCox;

impl TargetTransform for BoxCox {
    type State = BoxCoxState;

    fn fit(y: &[f64]) -> Result<Self::State, TransformError> {
        validate_positive(y)?;
        let lambda = fit_lambda(y, box_cox_profile_log_likelihood)?;
        BoxCoxState::new(lambda)
    }

    #[inline]
    fn transform(state: &Self::State, y: f64) -> f64 {
        transform_value(state.lambda, y)
    }

    #[inline]
    fn inverse(state: &Self::State, value: f64) -> f64 {
        inverse_value(state.lambda, value)
    }

    #[inline]
    fn validate_transform_value(_: &Self::State, y: f64) -> Result<(), TransformError> {
        validate_positive_value(y)
    }

    #[inline]
    fn validate_inverse_value(state: &Self::State, value: f64) -> Result<(), TransformError> {
        validate_box_cox_inverse_value(state.lambda, value)
    }
}

/// Box-Cox transform with a lambda fixed at `NUMERATOR / DENOMINATOR`.
///
/// Targets must be strictly positive. [`TargetTransform::checked_inverse`]
/// rejects transform-scale values outside the inverse domain for non-zero
/// lambdas.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BoxCoxFixed<const NUMERATOR: i32, const DENOMINATOR: i32 = 1>(PhantomData<()>);

impl<const NUMERATOR: i32, const DENOMINATOR: i32> TargetTransform
    for BoxCoxFixed<NUMERATOR, DENOMINATOR>
{
    type State = BoxCoxState;

    fn fit(y: &[f64]) -> Result<Self::State, TransformError> {
        validate_positive(y)?;
        BoxCoxState::new(lambda_from_ratio(NUMERATOR, DENOMINATOR)?)
    }

    #[inline]
    fn transform(state: &Self::State, y: f64) -> f64 {
        transform_value(state.lambda, y)
    }

    #[inline]
    fn inverse(state: &Self::State, value: f64) -> f64 {
        inverse_value(state.lambda, value)
    }

    #[inline]
    fn validate_transform_value(_: &Self::State, y: f64) -> Result<(), TransformError> {
        validate_positive_value(y)
    }

    #[inline]
    fn validate_inverse_value(state: &Self::State, value: f64) -> Result<(), TransformError> {
        validate_box_cox_inverse_value(state.lambda, value)
    }
}

/// State for [`BoxCox`] and [`BoxCoxFixed`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxCoxState {
    /// Box-Cox power parameter.
    pub lambda: f64,
}

impl BoxCoxState {
    /// Creates a Box-Cox transform state from a finite lambda.
    ///
    /// # Errors
    ///
    /// Returns [`TransformError::InvalidParameter`] if `lambda` is not finite.
    pub const fn new(lambda: f64) -> Result<Self, TransformError> {
        if lambda.is_finite() {
            Ok(Self { lambda })
        } else {
            Err(TransformError::InvalidParameter { name: "lambda" })
        }
    }
}

#[inline]
pub(crate) fn transform_value(lambda: f64, y: f64) -> f64 {
    if lambda.abs() <= LAMBDA_EPSILON {
        y.ln()
    } else {
        (y.powf(lambda) - 1.0) / lambda
    }
}

#[inline]
pub(crate) fn inverse_value(lambda: f64, value: f64) -> f64 {
    if lambda.abs() <= LAMBDA_EPSILON {
        value.exp()
    } else {
        lambda.mul_add(value, 1.0).powf(1.0 / lambda)
    }
}

#[inline]
fn validate_box_cox_inverse_value(lambda: f64, value: f64) -> Result<(), TransformError> {
    validate_finite_value(value)?;
    if lambda.abs() <= LAMBDA_EPSILON || lambda.mul_add(value, 1.0) > 0.0 {
        Ok(())
    } else if lambda > 0.0 {
        Err(TransformError::BelowLowerBound)
    } else {
        Err(TransformError::AboveUpperBound)
    }
}

fn box_cox_profile_log_likelihood(y: &[f64], lambda: f64) -> Option<f64> {
    let mut accumulator = ProfileAccumulator::default();
    let mut log_jacobian = 0.0;
    for value in y.iter().copied() {
        accumulator.push(transform_value(lambda, value))?;
        log_jacobian += value.ln();
    }
    accumulator
        .log_likelihood()
        .map(|profile| (lambda - 1.0).mul_add(log_jacobian, profile))
}

#[cfg(test)]
pub(crate) fn fixed_score(y: &[f64], lambda: f64) -> Option<f64> {
    box_cox_profile_log_likelihood(y, lambda)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{BoxCox, BoxCoxFixed, TargetTransform, TransformError};

    #[test]
    fn fixed_lambda_zero_is_log_transform() {
        let y = [1.0, 2.0, 4.0];
        let (state, transformed) = BoxCoxFixed::<0>::fit_transform(&y).unwrap();
        let restored = BoxCoxFixed::<0>::inverse_slice(&state, &transformed).unwrap();

        assert_relative_eq!(state.lambda, 0.0);
        assert_relative_eq!(transformed[2], 4.0_f64.ln());
        for (actual, expected) in restored.iter().zip(y) {
            assert_relative_eq!(*actual, expected);
        }
    }

    #[test]
    fn fixed_lambda_one_is_shifted_identity() {
        let y = [1.0, 2.0, 4.0];
        let (_, transformed) = BoxCoxFixed::<1>::fit_transform(&y).unwrap();

        assert_eq!(transformed, vec![0.0, 1.0, 3.0]);
    }

    #[test]
    fn fitted_lambda_is_finite_and_beats_simple_fixed_reference() {
        let y = [0.2, 0.4, 1.0, 3.0, 10.0, 30.0];
        let state = BoxCox::fit(&y).unwrap();
        let fitted_score = super::fixed_score(&y, state.lambda).unwrap();
        let log_score = super::fixed_score(&y, 0.0).unwrap();
        let identity_score = super::fixed_score(&y, 1.0).unwrap();

        assert!(state.lambda.is_finite());
        assert!(fitted_score >= log_score.min(identity_score));
    }

    #[test]
    fn fitted_lambda_matches_hardcoded_regression_values() {
        let cases = [
            (
                &[0.2, 0.4, 1.0, 3.0, 10.0, 30.0][..],
                -0.078_822_802_215_665,
            ),
            (
                &[1.0, 1.5, 2.0, 3.0, 5.0, 8.0][..],
                -0.181_468_776_105_318_5,
            ),
            (&[1.0, 2.0, 3.0, 4.0, 5.0][..], 0.690_296_542_789_852_4),
        ];

        for (y, expected_lambda) in cases {
            let state = BoxCox::fit(y).unwrap();
            assert_relative_eq!(state.lambda, expected_lambda, epsilon = 1.0e-8);
        }
    }

    #[test]
    fn checked_inverse_rejects_values_outside_inverse_domain() {
        let positive_state = BoxCoxFixed::<1>::fit(&[1.0, 2.0]).unwrap();
        assert_eq!(
            BoxCoxFixed::<1>::checked_inverse(&positive_state, -1.0).unwrap_err(),
            TransformError::BelowLowerBound
        );

        let negative_state = BoxCoxFixed::<-1>::fit(&[1.0, 2.0]).unwrap();
        assert_eq!(
            BoxCoxFixed::<-1>::checked_inverse(&negative_state, 1.0).unwrap_err(),
            TransformError::AboveUpperBound
        );
    }

    #[test]
    fn rejects_non_positive_values_and_invalid_denominator() {
        assert_eq!(
            BoxCox::fit(&[1.0, 0.0]).unwrap_err(),
            TransformError::NonPositiveValue
        );
        assert_eq!(
            BoxCoxFixed::<1, 0>::fit(&[1.0]).unwrap_err(),
            TransformError::InvalidParameter {
                name: "denominator"
            }
        );
    }

    #[test]
    fn state_constructor_rejects_non_finite_lambda() {
        assert_eq!(
            super::BoxCoxState::new(f64::NAN).unwrap_err(),
            TransformError::InvalidParameter { name: "lambda" }
        );
    }
}
