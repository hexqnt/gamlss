#![forbid(unsafe_code)]
//! Типизированное ядро GAMLSS: link-функции, parameter blocks, objectives и compiled models.

/// Абстракции design matrix.
pub mod design;
/// Ошибки модели и валидации.
pub mod error;
/// Контракты distribution families.
pub mod family;
/// Link-функции.
pub mod link;
/// Скомпилированные модели.
pub mod model;
/// Абстракции objective.
pub mod objective;
/// Типизированные параметры и parameter blocks.
pub mod param;
/// Penalty traits и реализации.
pub mod penalty;
/// Predictor block traits и композиция predictor-а.
pub mod predictor;

pub use design::{DenseDesign, DesignMatrix};
pub use error::ModelError;
pub use family::{CanSimulate, Family, HasCdf, HasQuantile, ParameterParts, ParameterizedFamily};
pub use link::{ClampedLog, Identity, Link, Log, LogPlus, Logit, PositiveLink, Softplus};
pub use model::{
    CachedGamlss, Diagnostics, Gamlss, GamlssBlocks, GradientWorkspace, ParameterCoefficients,
    ParameterLayout, ParameterSlice, UnpackedTheta, WithGlobalPenalties,
};
pub use objective::{BlockObjective, Objective};
pub use param::{Mu, Nu, ParameterBlock, ParameterName, Precision, Rate, Scale, Shape, Sigma, Tau};
pub use penalty::{GlobalPenalty, NoPenalty, Penalty, RidgePenalty};
pub use predictor::{LinearPredictorBlock, PredictorBlock, SumBlock};

/// Наиболее часто используемые импорты из `gamlss-core`.
pub mod prelude {
    pub use crate::{
        BlockObjective, CachedGamlss, CanSimulate, ClampedLog, DenseDesign, DesignMatrix,
        Diagnostics, Family, Gamlss, GamlssBlocks, GlobalPenalty, GradientWorkspace, HasCdf,
        HasQuantile, Identity, LinearPredictorBlock, Link, Log, LogPlus, Logit, ModelError, Mu,
        NoPenalty, Nu, Objective, ParameterBlock, ParameterCoefficients, ParameterLayout,
        ParameterName, ParameterParts, ParameterSlice, ParameterizedFamily, Penalty, PositiveLink,
        Precision, PredictorBlock, Rate, RidgePenalty, Scale, Shape, Sigma, Softplus, SumBlock,
        Tau, UnpackedTheta, WithGlobalPenalties,
    };
}
