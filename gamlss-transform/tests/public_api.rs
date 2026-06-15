use gamlss_transform::{
    Log1pShift, Standardize, TargetTransform, TransformError,
    prelude::{AsinhScale, IdentityPositive},
    transforms::{log1p_shift, standardize},
};

#[test]
fn root_and_module_reexports_are_available() {
    let _: Standardize = standardize::Standardize;
    let _: Log1pShift = log1p_shift::Log1pShift;
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
}
