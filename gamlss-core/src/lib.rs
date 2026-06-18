#![forbid(unsafe_code)]
//! Типизированное ядро GAMLSS: link-функции, parameter blocks, objectives и compiled models.
//!
//! `gamlss-core` содержит минимальные abstractions, которые нужны
//! distributional-regression моделям, но не зависит от optimizers,
//! dataframe-библиотек и тяжёлых matrix backends.
//!
//! # Parameter blocks
//!
//! Модель собирается из typed [`ParameterBlock`] values. Тип `P` задаёт
//! parameter marker (`Mu`, `Sigma`, `Shape`, пользовательский marker и т.д.),
//! `L` задаёт link, predictor block `X` считает link-scale predictor, а
//! `Penalty` добавляет локальную регуляризацию.
//!
//! Используйте [`ParameterBlocks::new`] для обычной сборки tuple blocks: она
//! последовательно назначает offsets и убирает ручной расчёт диапазонов в
//! общем beta-векторе.
//!
//! # Observations and prediction
//!
//! [`Gamlss::try_new`] строит unweighted модель над borrowed response slice.
//! [`Gamlss::try_new_weighted`] дополнительно принимает finite non-negative
//! observation weights; нулевой вес исключает наблюдение из likelihood и
//! gradient.
//!
//! Prediction methods возвращают link-scale `Eta` или natural-scale `Theta`:
//! `predict_eta`, `predict_theta` используют training blocks, а методы
//! `*_with_blocks` принимают совместимый tuple prediction blocks для новых
//! строк.

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
pub use family::{
    CanSimulate, DenseInformation, Family, HasCdf, HasCrps, HasDeviance, HasDiagonalFisherInfo,
    HasExpectedInformation, HasInitialEta, HasQuantile, ParameterParts, ParameterizedFamily,
};
pub use link::{
    ClampedLog, Identity, Link, Log, LogPlus, Logit, PositiveLink, Softplus, UnitIntervalLink,
};
pub use model::{
    Gamlss, GamlssBlocks, GradientWorkspace, ObjectiveScale, ObservationView,
    ParameterCoefficients, ParameterLayout, ParameterSlice, TrainingDiagnostics, UnpackedTheta,
    WithGlobalPenalties, WorkspaceGamlss,
};
pub use objective::{BlockObjective, Objective};
pub use param::{
    AssignParameterOffsets, Mu, Nu, ParameterBlock, ParameterBlocks, ParameterName, Precision,
    Rate, Scale, Shape, Sigma, Tau,
};
pub use penalty::{
    AbsoluteLimitPenalty, GlobalPenalty, HingeQuadraticPenalty, LinearForm, LinearFormBuilder,
    LinearTerm, MatrixPenalty, NoPenalty, Penalty, RidgePenalty, SegmentPenalty,
};
pub use predictor::{
    CoefficientTransform, FloorSoftplusScalar, HasDesignMatrix, LinearPredictorBlock,
    NegativeSoftplusScalar, NegativeSoftplusTransform, OffsetBlock, PredictorBlock, ProductBlock,
    SoftplusScalar, SoftplusTransform, SumBlock, TransformedScalar,
};

/// Наиболее часто используемые импорты из `gamlss-core`.
pub mod prelude {
    pub use crate::{
        AbsoluteLimitPenalty, AssignParameterOffsets, BlockObjective, CanSimulate, ClampedLog,
        CoefficientTransform, DenseDesign, DenseInformation, DesignMatrix, Family, Gamlss,
        GamlssBlocks, GlobalPenalty, GradientWorkspace, HasCdf, HasCrps, HasDesignMatrix,
        HasDeviance, HasDiagonalFisherInfo, HasExpectedInformation, HasInitialEta, HasQuantile,
        HingeQuadraticPenalty, Identity, LinearForm, LinearFormBuilder, LinearPredictorBlock,
        LinearTerm, Link, Log, LogPlus, Logit, MatrixPenalty, ModelError, Mu, NoPenalty, Nu,
        Objective, ObjectiveScale, ObservationView, OffsetBlock, ParameterBlock, ParameterBlocks,
        ParameterCoefficients, ParameterLayout, ParameterName, ParameterParts, ParameterSlice,
        ParameterizedFamily, Penalty, PositiveLink, Precision, PredictorBlock, ProductBlock, Rate,
        RidgePenalty, Scale, SegmentPenalty, Shape, Sigma, Softplus, SumBlock, Tau,
        TrainingDiagnostics, TransformedScalar, UnitIntervalLink, UnpackedTheta,
        WithGlobalPenalties, WorkspaceGamlss,
    };
}
