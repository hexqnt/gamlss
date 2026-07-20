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
//! predictor block `X` computes the link-scale predictor, and `Penalty` adds
//! local regularization. Links are owned solely by the distribution family.
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
//! Runtime-dimensional multivariate responses can use [`DenseRows`] to borrow
//! equal-width rows from one contiguous flat buffer with optional weights.
//!
//! Prediction methods return link-scale `Eta` or natural-scale `Theta`:
//! `predict_eta`, `predict_theta` use training blocks, while the
//! `*_with_blocks` methods accept a compatible tuple of prediction blocks for new
//! rows.

pub use design::{DenseDesign, DesignMatrix, RowMultiplier};
pub use error::ModelError;
pub use family::{
    CompilableFamily, DenseInformation, DynamicLayoutKey, DynamicallyCompilableFamily, Family,
    FixedDimensionalFamily, HasCdf, HasConditionalCdf, HasCrps, HasDensity, HasDeviance,
    HasDiagonalFisherInfo, HasExpectedInformation, HasInitialEta, HasLogDensity, HasMarginalCdf,
    HasObservationDimension, HasQuantile, HasRosenblattTransform, InitialEtaFromObservations,
    ParameterParts, SimulationError, TrySimulate,
};
pub use link::{
    AboveTwoLink, ClampedLog, Identity, InitialEtaFromTheta, Link, Log, LogPlus, Logit,
    PositiveLink, Softplus, UnitIntervalLink,
};
pub use model::{
    DenseRows, DynamicParameterBlocks, FiniteScalarObservations, Gamlss, GamlssBlocks,
    GradientWorkspace, ModelWorkspace, ObjectiveScale, ObservationView, ParameterAxis,
    ParameterCoefficients, ParameterDescriptor, ParameterLayout, ParameterPath, ParameterSlice,
    PredictionView, ScoreTilePolicy, TrainingDiagnostics, UnpackedParameters, WithGlobalPenalties,
    WorkspaceGamlss,
};
pub use objective::{BlockObjective, Objective};
pub use param::{
    AssignParameterOffsets, CholeskyScale, ComponentMean, Cv, DegreesOfFreedom, Dispersion,
    IdiosyncraticRate, KernelSigma, LogLocation, LogSd, LowerTriangularParameterBlock, Mean,
    Median, MixtureWeight, Mu, Nu, OneProbability, ParameterBlock, ParameterBlocks, ParameterName,
    PartialCorrelation, Power, Precision, Probability, Rate, Scale, Shape, SharedRate, Sigma,
    SimplexLogitParameterBlock, Size, StrictLowerTriangularParameterBlock, Tau, TotalMean,
    TryAssignParameterOffsets, VectorParameterBlock, ZeroProbability,
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
pub use shape::{
    Broadcast, LocationCholesky, Lower, ParameterShape, Product, Repeated, Scalar, ScalarTuple,
    ShapeValues, Simplex, StrictLower, Vector,
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
/// Static parameter-shape algebra for compiled families.
pub mod shape;

/// Most commonly used imports from `gamlss-core`.
pub mod prelude {
    pub use crate::{
        AboveTwoLink, AbsoluteLimitPenalty, AssignParameterOffsets, BlockObjective, CholeskyScale,
        ClampedLog, CoefficientTransform, ComponentMean, Cv, DegreesOfFreedom, DenseDesign,
        DenseInformation, DenseRows, DesignMatrix, Dispersion, DynamicLayoutKey,
        DynamicParameterBlocks, DynamicallyCompilableFamily, Family, FiniteScalarObservations,
        FixedDimensionalFamily, Gamlss, GamlssBlocks, GlobalPenalty, GradientWorkspace, HasCdf,
        HasConditionalCdf, HasCrps, HasDensity, HasDesignMatrix, HasDeviance,
        HasDiagonalFisherInfo, HasExpectedInformation, HasInitialEta, HasLogDensity,
        HasMarginalCdf, HasObservationDimension, HasQuantile, HasRosenblattTransform,
        HingeQuadraticPenalty, Identity, IdiosyncraticRate, InitialEtaFromObservations,
        InitialEtaFromTheta, KernelSigma, LinearForm, LinearFormBuilder, LinearPredictorBlock,
        LinearPredictorGeometry, LinearTerm, Link, LocationCholesky, Log, LogLocation, LogPlus,
        LogSd, Logit, Lower, LowerTriangularParameterBlock, MatrixPenalty, Mean, Median,
        MixtureWeight, ModelError, ModelWorkspace, Mu, NoPenalty, Nu, Objective, ObjectiveScale,
        ObservationView, OffsetBlock, OneProbability, ParameterAxis, ParameterBlock,
        ParameterBlocks, ParameterCoefficients, ParameterDescriptor, ParameterLayout,
        ParameterName, ParameterParts, ParameterPath, ParameterShape, ParameterSlice,
        PartialCorrelation, Penalty, PositiveLink, Power, Precision, PredictionView,
        PredictorBlock, Probability, Product, ProductBlock, Rate, Repeated, RidgePenalty,
        RowMultiplier, Scalar, ScalarTuple, Scale, ScoreTilePolicy, SegmentPenalty, Shape,
        ShapeValues, SharedRate, Sigma, Simplex, SimplexLogitParameterBlock, SimulationError, Size,
        Softplus, StrictLower, StrictLowerTriangularParameterBlock, SumBlock, Tau, TotalMean,
        TrainingDiagnostics, TransformedScalar, TryAssignParameterOffsets, TrySimulate,
        UnitIntervalLink, UnpackedParameters, Vector, VectorParameterBlock, WithGlobalPenalties,
        WorkspaceGamlss, ZeroProbability,
    };
}
