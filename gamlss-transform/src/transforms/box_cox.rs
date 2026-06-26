use std::marker::PhantomData;

use crate::transforms::{
    TargetTransform, TransformError, lambda_from_ratio, validate_finite_value, validate_positive,
    validate_positive_value,
};

const LAMBDA_EPSILON: f64 = 1.0e-12;
const INV_PHI: f64 = 0.618_033_988_749_894_9;
const LAMBDA_SEARCH_LOWER: f64 = -5.0;
const LAMBDA_SEARCH_UPPER: f64 = 5.0;
const LAMBDA_SEARCH_RADIUS: f64 = 2.0;
const LAMBDA_SEARCH_ITERATIONS: usize = 80;
const LAMBDA_SEARCH_GRID: [f64; 7] = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];

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
        Ok(BoxCoxState {
            lambda: lambda_from_ratio(NUMERATOR, DENOMINATOR)?,
        })
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
    let mut transformed = Vec::with_capacity(y.len());
    let mut log_jacobian = 0.0;
    for value in y.iter().copied() {
        transformed.push(transform_value(lambda, value));
        log_jacobian += value.ln();
    }
    profile_log_likelihood(&transformed)
        .map(|profile| (lambda - 1.0).mul_add(log_jacobian, profile))
}

pub(crate) fn fit_lambda(
    y: &[f64],
    objective: impl Fn(&[f64], f64) -> Option<f64>,
) -> Result<f64, TransformError> {
    let mut best_lambda = 0.0;
    let mut best_score =
        objective(y, best_lambda).ok_or(TransformError::InvalidParameter { name: "lambda" })?;

    for candidate in LAMBDA_SEARCH_GRID {
        if let Some(score) = objective(y, candidate)
            && score > best_score
        {
            best_lambda = candidate;
            best_score = score;
        }
    }

    let mut left = (best_lambda - LAMBDA_SEARCH_RADIUS).max(LAMBDA_SEARCH_LOWER);
    let mut right = (best_lambda + LAMBDA_SEARCH_RADIUS).min(LAMBDA_SEARCH_UPPER);
    if (left - right).abs() <= f64::EPSILON {
        return Ok(best_lambda);
    }

    let mut c = INV_PHI.mul_add(-(right - left), right);
    let mut d = INV_PHI.mul_add(right - left, left);
    let mut c_score = objective(y, c).unwrap_or(f64::NEG_INFINITY);
    let mut d_score = objective(y, d).unwrap_or(f64::NEG_INFINITY);

    for _ in 0..LAMBDA_SEARCH_ITERATIONS {
        if c_score < d_score {
            left = c;
            c = d;
            c_score = d_score;
            d = INV_PHI.mul_add(right - left, left);
            d_score = objective(y, d).unwrap_or(f64::NEG_INFINITY);
        } else {
            right = d;
            d = c;
            d_score = c_score;
            c = INV_PHI.mul_add(-(right - left), right);
            c_score = objective(y, c).unwrap_or(f64::NEG_INFINITY);
        }
    }

    let lambda = f64::midpoint(left, right);
    if lambda.is_finite() {
        Ok(lambda)
    } else {
        Err(TransformError::InvalidParameter { name: "lambda" })
    }
}

pub(crate) fn profile_log_likelihood(values: &[f64]) -> Option<f64> {
    if values.len() < 2 || !values.iter().copied().all(f64::is_finite) {
        return None;
    }

    let mut count = 0.0;
    let mut mean = 0.0;
    let mut sum_squares = 0.0;
    for value in values.iter().copied() {
        count += 1.0;
        let delta = value - mean;
        mean += delta / count;
        sum_squares += delta * (value - mean);
    }
    let variance = sum_squares / count;
    if variance.is_finite() && variance > 0.0 {
        Some(-0.5 * count * variance.ln())
    } else {
        None
    }
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
            assert_relative_eq!(state.lambda, expected_lambda, epsilon = 1.0e-10);
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
