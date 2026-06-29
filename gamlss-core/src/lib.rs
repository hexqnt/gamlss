#![forbid(unsafe_code)]
//! Typed GAMLSS core: link functions, parameter blocks, objectives and compiled models.
//!
//! `gamlss-core` contains the minimal abstractions needed by
//! distributional-regression models, but does not depend on optimizers,
//! dataframe libraries or heavy matrix backends.
//!
//! # Parameter blocks
//!
//! The model is assembled from typed [`ParameterBlock`] values. The `P` type
//! specifies the parameter marker (`Mu`, `Sigma`, `Shape`, custom marker, etc.),
//! `L` specifies the link, predictor block `X` computes the link-scale predictor,
//! and `Penalty` adds local regularization.
//!
//! Use [`ParameterBlocks::new`] for ordinary tuple block assembly: it
//! sequentially assigns offsets and removes the need for manual range calculation
//! in the common beta vector.
//!
//! # Observations and prediction
//!
//! [`Gamlss::try_new`] builds an unweighted model over a borrowed response slice.
//! [`Gamlss::try_new_weighted`] additionally accepts finite non-negative
//! observation weights; a zero weight excludes the observation from the
//! likelihood and gradient. Scalar responses are otherwise validated by the
//! family/domain layer; use [`Gamlss::try_new_strict`] or
//! [`FiniteScalarObservations`] to reject non-finite scalar responses at model
//! construction.
//!
//! Prediction methods return link-scale `Eta` or natural-scale `Theta`:
//! `predict_eta`, `predict_theta` use training blocks, while the
//! `*_with_blocks` methods accept a compatible tuple of prediction blocks for new
//! rows.

pub use design::{DenseDesign, DesignMatrix, RowMultiplier};
pub use error::ModelError;
pub use family::{
    CanSimulate, DenseInformation, Family, FixedDimensionalFamily, HasCdf, HasConditionalCdf,
    HasCrps, HasDensity, HasDeviance, HasDiagonalFisherInfo, HasExpectedInformation, HasInitialEta,
    HasLogDensity, HasMarginalCdf, HasQuantile, HasRosenblattTransform, ParameterParts,
    ParameterizedFamily, VectorLowerTriangularFamily,
};
pub use link::{
    ClampedLog, Identity, InitialEtaFromTheta, Link, Log, LogPlus, Logit, PositiveLink, Softplus,
    UnitIntervalLink,
};
pub use model::{
    FiniteScalarObservations, Gamlss, GamlssBlocks, GradientWorkspace, ObjectiveScale,
    ObservationView, ParameterCoefficients, ParameterLayout, ParameterSlice, PredictionView,
    TrainingDiagnostics, UnpackedParameters, WithGlobalPenalties, WorkspaceGamlss,
};
pub use objective::{BlockObjective, Objective};
pub use param::{
    AssignParameterOffsets, CholeskyScale, ComponentMean, Cv, Dispersion, LogLocation, LogSd,
    LowerTriangularParameterBlock, Mean, Median, Mu, Nu, OneProbability, ParameterBlock,
    ParameterBlocks, ParameterName, Power, Precision, Probability, Rate, Scale, Shape, Sigma, Size,
    Tau, TotalMean, TryAssignParameterOffsets, VectorParameterBlock, ZeroProbability,
};
pub use penalty::{
    AbsoluteLimitPenalty, GlobalPenalty, HingeQuadraticPenalty, LinearForm, LinearFormBuilder,
    LinearTerm, MatrixPenalty, NoPenalty, Penalty, RidgePenalty, SegmentPenalty,
};
pub use predictor::{
    CoefficientTransform, FloorSoftplusScalar, HasDesignMatrix, LinearPredictorBlock,
    LinearPredictorGeometry, NegativeSoftplusScalar, NegativeSoftplusTransform, OffsetBlock,
    PredictorBlock, ProductBlock, SoftplusScalar, SoftplusTransform, SumBlock, TransformedScalar,
};

/// Design matrix abstractions.
pub mod design;
/// Model and validation errors.
pub mod error;
/// Distribution family contracts.
pub mod family;
/// Link functions.
pub mod link;
/// Compiled models.
pub mod model;
/// Objective abstractions.
pub mod objective;
/// Typed parameters and parameter blocks.
pub mod param;
/// Penalty traits and implementations.
pub mod penalty;
/// Predictor block traits and predictor composition.
pub mod predictor;

/// Most commonly used imports from `gamlss-core`.
pub mod prelude {
    pub use crate::{
        AbsoluteLimitPenalty, AssignParameterOffsets, BlockObjective, CanSimulate, CholeskyScale,
        ClampedLog, CoefficientTransform, ComponentMean, Cv, DenseDesign, DenseInformation,
        DesignMatrix, Dispersion, Family, FiniteScalarObservations, FixedDimensionalFamily, Gamlss,
        GamlssBlocks, GlobalPenalty, GradientWorkspace, HasCdf, HasConditionalCdf, HasCrps,
        HasDensity, HasDesignMatrix, HasDeviance, HasDiagonalFisherInfo, HasExpectedInformation,
        HasInitialEta, HasLogDensity, HasMarginalCdf, HasQuantile, HasRosenblattTransform,
        HingeQuadraticPenalty, Identity, InitialEtaFromTheta, LinearForm, LinearFormBuilder,
        LinearPredictorBlock, LinearPredictorGeometry, LinearTerm, Link, Log, LogLocation, LogPlus,
        LogSd, Logit, LowerTriangularParameterBlock, MatrixPenalty, Mean, Median, ModelError, Mu,
        NoPenalty, Nu, Objective, ObjectiveScale, ObservationView, OffsetBlock, OneProbability,
        ParameterBlock, ParameterBlocks, ParameterCoefficients, ParameterLayout, ParameterName,
        ParameterParts, ParameterSlice, ParameterizedFamily, Penalty, PositiveLink, Power,
        Precision, PredictionView, PredictorBlock, Probability, ProductBlock, Rate, RidgePenalty,
        RowMultiplier, Scale, SegmentPenalty, Shape, Sigma, Size, Softplus, SumBlock, Tau,
        TotalMean, TrainingDiagnostics, TransformedScalar, TryAssignParameterOffsets,
        UnitIntervalLink, UnpackedParameters, VectorLowerTriangularFamily, VectorParameterBlock,
        WithGlobalPenalties, WorkspaceGamlss, ZeroProbability,
    };
}
