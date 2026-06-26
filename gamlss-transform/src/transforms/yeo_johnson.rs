use std::marker::PhantomData;

use crate::transforms::{
    TargetTransform, TransformError, lambda_from_ratio, validate_finite_value,
    validate_non_empty_finite,
};

use super::power::{ProfileAccumulator, fit_lambda};

const LAMBDA_EPSILON: f64 = 1.0e-12;

/// Yeo-Johnson transform with a fitted lambda parameter for finite targets.
///
/// The lambda parameter is fitted by deterministic profile-likelihood search on
/// `[-5, 5]`. [`TargetTransform::checked_inverse`] rejects transform-scale
/// values outside the lambda-specific inverse domain.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct YeoJohnson;

impl TargetTransform for YeoJohnson {
    type State = YeoJohnsonState;

    fn fit(y: &[f64]) -> Result<Self::State, TransformError> {
        validate_non_empty_finite(y)?;
        let lambda = fit_lambda(y, yeo_johnson_profile_log_likelihood)?;
        YeoJohnsonState::new(lambda)
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
    fn validate_inverse_value(state: &Self::State, value: f64) -> Result<(), TransformError> {
        validate_yeo_johnson_inverse_value(state.lambda, value)
    }
}

/// Yeo-Johnson transform with a lambda fixed at `NUMERATOR / DENOMINATOR`.
///
/// The transform accepts all finite targets. [`TargetTransform::checked_inverse`]
/// rejects transform-scale values outside the lambda-specific inverse domain.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct YeoJohnsonFixed<const NUMERATOR: i32, const DENOMINATOR: i32 = 1>(PhantomData<()>);

impl<const NUMERATOR: i32, const DENOMINATOR: i32> TargetTransform
    for YeoJohnsonFixed<NUMERATOR, DENOMINATOR>
{
    type State = YeoJohnsonState;

    fn fit(y: &[f64]) -> Result<Self::State, TransformError> {
        validate_non_empty_finite(y)?;
        YeoJohnsonState::new(lambda_from_ratio(NUMERATOR, DENOMINATOR)?)
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
    fn validate_inverse_value(state: &Self::State, value: f64) -> Result<(), TransformError> {
        validate_yeo_johnson_inverse_value(state.lambda, value)
    }
}

/// State for [`YeoJohnson`] and [`YeoJohnsonFixed`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YeoJohnsonState {
    /// Yeo-Johnson power parameter.
    pub lambda: f64,
}

impl YeoJohnsonState {
    /// Creates a Yeo-Johnson transform state from a finite lambda.
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
    if y >= 0.0 {
        if lambda.abs() <= LAMBDA_EPSILON {
            y.ln_1p()
        } else {
            ((y + 1.0).powf(lambda) - 1.0) / lambda
        }
    } else if (lambda - 2.0).abs() <= LAMBDA_EPSILON {
        -(-y).ln_1p()
    } else {
        -(((1.0 - y).powf(2.0 - lambda) - 1.0) / (2.0 - lambda))
    }
}

#[inline]
pub(crate) fn inverse_value(lambda: f64, value: f64) -> f64 {
    if value >= 0.0 {
        if lambda.abs() <= LAMBDA_EPSILON {
            value.exp_m1()
        } else {
            lambda.mul_add(value, 1.0).powf(1.0 / lambda) - 1.0
        }
    } else if (lambda - 2.0).abs() <= LAMBDA_EPSILON {
        1.0 - (-value).exp()
    } else {
        1.0 - (2.0 - lambda)
            .mul_add(-value, 1.0)
            .powf(1.0 / (2.0 - lambda))
    }
}

#[inline]
fn validate_yeo_johnson_inverse_value(lambda: f64, value: f64) -> Result<(), TransformError> {
    validate_finite_value(value)?;
    if value >= 0.0 {
        let base = lambda.mul_add(value, 1.0);
        if lambda < -LAMBDA_EPSILON && base <= 0.0 {
            Err(TransformError::AboveUpperBound)
        } else {
            Ok(())
        }
    } else {
        let base = (2.0 - lambda).mul_add(-value, 1.0);
        if lambda > 2.0 + LAMBDA_EPSILON && base <= 0.0 {
            Err(TransformError::BelowLowerBound)
        } else {
            Ok(())
        }
    }
}

fn yeo_johnson_profile_log_likelihood(y: &[f64], lambda: f64) -> Option<f64> {
    let mut accumulator = ProfileAccumulator::default();
    let mut log_jacobian = 0.0;
    for value in y.iter().copied() {
        accumulator.push(transform_value(lambda, value))?;
        log_jacobian += if value >= 0.0 {
            (lambda - 1.0) * value.ln_1p()
        } else {
            (1.0 - lambda) * (-value).ln_1p()
        };
    }
    accumulator
        .log_likelihood()
        .map(|profile| profile + log_jacobian)
}

#[cfg(test)]
pub(crate) fn fixed_score(y: &[f64], lambda: f64) -> Option<f64> {
    if crate::transforms::validate_finite(y).is_err() {
        return None;
    }
    yeo_johnson_profile_log_likelihood(y, lambda)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{TargetTransform, TransformError, YeoJohnson, YeoJohnsonFixed};

    #[test]
    fn fixed_lambda_zero_uses_log_branch_for_non_negative_values() {
        let y = [0.0, 1.0, 3.0];
        let (state, transformed) = YeoJohnsonFixed::<0>::fit_transform(&y).unwrap();
        let restored = YeoJohnsonFixed::<0>::inverse_slice(&state, &transformed).unwrap();

        assert_relative_eq!(transformed[2], 4.0_f64.ln());
        for (actual, expected) in restored.iter().zip(y) {
            assert_relative_eq!(*actual, expected);
        }
    }

    #[test]
    fn fixed_lambda_one_is_identity() {
        let y = [-2.0, 0.0, 3.0];
        let (_, transformed) = YeoJohnsonFixed::<1>::fit_transform(&y).unwrap();

        for (actual, expected) in transformed.iter().zip(y) {
            assert_relative_eq!(*actual, expected);
        }
    }

    #[test]
    fn fixed_lambda_two_uses_log_branch_for_negative_values() {
        let y = [-3.0, -1.0, 0.0];
        let (state, transformed) = YeoJohnsonFixed::<2>::fit_transform(&y).unwrap();
        let restored = YeoJohnsonFixed::<2>::inverse_slice(&state, &transformed).unwrap();

        assert_relative_eq!(transformed[0], -4.0_f64.ln());
        for (actual, expected) in restored.iter().zip(y) {
            assert_relative_eq!(*actual, expected);
        }
    }

    #[test]
    fn fitted_lambda_is_finite_and_beats_simple_fixed_reference() {
        let y = [-5.0, -2.0, -1.0, 0.0, 1.0, 4.0, 12.0];
        let state = YeoJohnson::fit(&y).unwrap();
        let fitted_score = super::fixed_score(&y, state.lambda).unwrap();
        let identity_score = super::fixed_score(&y, 1.0).unwrap();

        assert!(state.lambda.is_finite());
        assert!(fitted_score >= identity_score);
    }

    #[test]
    fn fitted_lambda_matches_hardcoded_regression_values() {
        let cases = [
            (
                &[-5.0, -2.0, -1.0, 0.0, 1.0, 4.0, 12.0][..],
                0.689_444_039_598_926_5,
            ),
            (
                &[-2.0, -0.5, 0.0, 0.5, 2.0, 6.0][..],
                0.526_487_107_662_148_7,
            ),
            (&[-3.0, -1.0, 0.0, 1.0, 3.0][..], 0.999_999_991_302_686_2),
        ];

        for (y, expected_lambda) in cases {
            let state = YeoJohnson::fit(y).unwrap();
            assert_relative_eq!(state.lambda, expected_lambda, epsilon = 1.0e-10);
        }
    }

    #[test]
    fn checked_inverse_rejects_values_outside_inverse_domain() {
        let upper_state = YeoJohnsonFixed::<-1>::fit(&[-1.0, 1.0]).unwrap();
        assert_eq!(
            YeoJohnsonFixed::<-1>::checked_inverse(&upper_state, 1.0).unwrap_err(),
            TransformError::AboveUpperBound
        );

        let lower_state = YeoJohnsonFixed::<3>::fit(&[-1.0, 1.0]).unwrap();
        assert_eq!(
            YeoJohnsonFixed::<3>::checked_inverse(&lower_state, -1.0).unwrap_err(),
            TransformError::BelowLowerBound
        );
    }

    #[test]
    fn rejects_non_finite_values_and_invalid_denominator() {
        assert_eq!(
            YeoJohnson::fit(&[1.0, f64::NAN]).unwrap_err(),
            TransformError::NonFiniteValue
        );
        assert_eq!(
            YeoJohnsonFixed::<1, 0>::fit(&[1.0]).unwrap_err(),
            TransformError::InvalidParameter {
                name: "denominator"
            }
        );
    }

    #[test]
    fn state_constructor_rejects_non_finite_lambda() {
        assert_eq!(
            super::YeoJohnsonState::new(f64::INFINITY).unwrap_err(),
            TransformError::InvalidParameter { name: "lambda" }
        );
    }
}
