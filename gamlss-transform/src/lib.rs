#![forbid(unsafe_code)]
//! Target transforms for GAMLSS modeling.

pub use transforms::{
    AsinhScale, AsinhScaleState, BoxCox, BoxCoxFixed, BoxCoxState, IdentityPositive,
    IdentityPositiveState, Log, Log1pShift, Log1pShiftState, LogState, MaxAbsScale,
    MaxAbsScaleState, MinMaxScale, MinMaxScaleState, QuantileNormal, QuantileState,
    QuantileUniform, RobustStandardize, RobustStandardizeState, Standardize, StandardizeState,
    TargetTransform, Then, ThenState, TransformError, YeoJohnson, YeoJohnsonFixed, YeoJohnsonState,
};

pub mod transforms;

/// Most commonly used imports from `gamlss-transform`.
pub mod prelude {
    pub use crate::{
        AsinhScale, AsinhScaleState, BoxCox, BoxCoxFixed, BoxCoxState, IdentityPositive,
        IdentityPositiveState, Log, Log1pShift, Log1pShiftState, LogState, MaxAbsScale,
        MaxAbsScaleState, MinMaxScale, MinMaxScaleState, QuantileNormal, QuantileState,
        QuantileUniform, RobustStandardize, RobustStandardizeState, Standardize, StandardizeState,
        TargetTransform, Then, ThenState, TransformError, YeoJohnson, YeoJohnsonFixed,
        YeoJohnsonState,
    };
}
