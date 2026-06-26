#![allow(clippy::float_cmp)]
use approx::assert_relative_eq;
use gamlss_transform::{
    BoxCox, BoxCoxFixed, Log, Log1pShift, MaxAbsScale, MinMaxScale, QuantileNormal,
    QuantileUniform, RobustStandardize, Standardize, TargetTransform, Then, ThenState,
    TransformError, YeoJohnson, YeoJohnsonFixed,
    prelude::{AsinhScale, IdentityPositive},
    transforms::{log1p_shift, quantile, standardize},
};

#[test]
fn root_and_module_reexports_are_available() {
    let _: Standardize = standardize::Standardize;
    let _: Log1pShift = log1p_shift::Log1pShift;
    let _: QuantileUniform = quantile::QuantileUniform;
    let _: RobustStandardize = RobustStandardize;
    let _: MinMaxScale = MinMaxScale;
    let _: MaxAbsScale = MaxAbsScale;
    let _: BoxCox = BoxCox;
    let _: BoxCoxFixed<0> = BoxCoxFixed::default();
    let _: YeoJohnson = YeoJohnson;
    let _: YeoJohnsonFixed<1> = YeoJohnsonFixed::default();
    let _: QuantileNormal = QuantileNormal;
}

#[test]
fn prelude_exposes_common_transforms() {
    let positive_state = IdentityPositive::fit(&[1.0]).unwrap();
    let asinh_state = AsinhScale::fit(&[-1.0, 0.0, 1.0]).unwrap();

    assert_eq!(IdentityPositive::transform(&positive_state, 2.0), 2.0);
    assert!(AsinhScale::transform(&asinh_state, 10.0).is_finite());
}

#[test]
fn public_error_variants_remain_matchable() {
    let error = Log1pShift::transform_slice(
        &log1p_shift::Log1pShiftState {
            shift: 1.0,
            margin: 0.0,
        },
        &[-2.0],
    )
    .unwrap_err();

    assert_eq!(error, TransformError::BelowLowerBound);

    assert_eq!(
        Standardize::transform_into(
            &standardize::StandardizeState {
                center: 0.0,
                scale: 1.0,
            },
            &[1.0, 2.0],
            &mut [0.0],
        )
        .unwrap_err(),
        TransformError::LengthMismatch {
            expected: 2,
            actual: 1
        }
    );

    assert_eq!(
        BoxCoxFixed::<1, 0>::fit(&[1.0]).unwrap_err(),
        TransformError::InvalidParameter {
            name: "denominator"
        }
    );

    assert_eq!(
        TransformError::AboveUpperBound,
        TransformError::AboveUpperBound
    );
}

#[test]
fn buffer_api_matches_allocating_slice_api() {
    let y = [1.0, 2.0, 4.0];
    let (state, transformed) = Standardize::fit_transform(&y).unwrap();
    let mut transformed_into = [0.0; 3];
    let mut restored_into = [0.0; 3];

    Standardize::transform_into(&state, &y, &mut transformed_into).unwrap();
    Standardize::inverse_into(&state, &transformed_into, &mut restored_into).unwrap();

    assert_eq!(transformed_into.as_slice(), transformed.as_slice());
    for (actual, expected) in restored_into.iter().zip(y) {
        assert_relative_eq!(*actual, expected);
    }
}

#[test]
fn checked_single_value_api_validates_transform_domains() {
    let log_state = Log::fit(&[1.0, 2.0]).unwrap();
    assert_eq!(
        Log::checked_transform(&log_state, 0.0).unwrap_err(),
        TransformError::NonPositiveValue
    );
    assert_relative_eq!(
        Log::checked_transform(&log_state, 2.0).unwrap(),
        2.0_f64.ln()
    );

    let shift_state = Log1pShift::fit(&[-2.0, 1.0]).unwrap();
    assert_eq!(
        Log1pShift::checked_transform(&shift_state, -3.0).unwrap_err(),
        TransformError::BelowLowerBound
    );
}

#[test]
fn static_composition_round_trips_values() {
    type Transform = Then<Log, Standardize>;
    let y = [1.0, 2.0, 4.0];
    let (state, transformed) = Transform::fit_transform(&y).unwrap();
    let _: ThenState<_, _> = state;
    let restored = Transform::inverse_slice(&state, &transformed).unwrap();

    for (actual, expected) in restored.iter().zip(y) {
        assert_relative_eq!(*actual, expected);
    }
}
