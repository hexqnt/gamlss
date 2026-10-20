#![forbid(unsafe_code)]
//! Spline bases, constrained and unconstrained predictor designs, and penalties.
//!
//! Reusable `*Basis` and `*Spec` metadata can be applied to new coordinates.
//! Named prepared designs cache compact row geometry where that improves
//! repeated model passes; [`OnDemandSplineDesign`] retains only coordinates
//! and metadata. Linear designs share [`SplineRowBasis`],
//! [`gamlss_core::PredictorBlock`], and
//! [`gamlss_core::LinearPredictorGeometry`] contracts. Univariate derivative
//! rows share [`DifferentiableSplineBasis1d`]; fixed-dimensional isotropic
//! bases use [`SplineBasisNd`]. Penalty geometry can be retained independently
//! from smoothing scales through [`PenaltyKernel`] and [`ScaledPenalty`].

pub use adaptive::{
    AdaptiveDifferencePenalty, WeightedDifferenceKernel, WeightedDifferencePenalty,
};
pub use basis::SplineBasis1d;
pub use basis_nd::SplineBasisNd;
pub use bspline::{BSplineBasis, BSplineDesign, pspline_design};
pub use constraint::{CenteredSplineDesign, HelmertContrastDesign};
pub use cspline::{CSplineBasis, ConvexCSplineDesign, CurvatureDirection};
pub use cyclic::{CyclicSplineDesign, CyclicSplineSpec};
pub use derivative::{DifferentiableSplineBasis1d, DifferentiableSplineRowBasis};
pub use duchon::{DuchonSmoothness, DuchonSplineBasis, DuchonSplineDesign};
pub use error::{FourierError, SplineError};
pub use fourier::{FourierBasis, FourierDesign};
pub use ispline::{ISplineBasis, ISplineDesign};
pub use kernel::{
    DensePenaltyKernel, DiagonalPenaltyKernel, DifferencePenaltyKernel, PenaltyKernel,
    ScaledPenalty,
};
pub use knots::{KnotPlacement, OpenKnotVector};
pub use monotone::{MonotoneDirection, MonotoneISplineDesign};
pub use mspline::{MSplineBasis, MSplineDesign};
pub use natural::{NaturalCubicSplineBasis, NaturalCubicSplineDesign};
pub use on_demand::OnDemandSplineDesign;
pub use open_uniform::{OpenUniformSplineBasis, OpenUniformSplineDesign};
pub use order::SplineOrder;
pub use penalty::{
    CyclicDifferencePenalty, DifferencePenalty, EdgeMonotonicPenalty,
    PreparedCyclicDifferencePenalty, PreparedDifferencePenalty, SlopeLimitPenalty,
};
pub use periodic::{PeriodicSplineDesign, PeriodicSplineSpec};
pub use roughness::{
    BSplineDerivativePenalty, FourierRoughnessPenalty, NaturalCubicRoughnessPenalty,
    NullSpacePenalty,
};
pub use row_basis::{CsrParts, SplineRowBasis, SplineRowBasisExt, TripletParts};
pub use tensor::{TensorInteractionDesign, TensorSplineDesign};
pub use tensor_penalty::{HelmertContrast, HelmertContrastPenalty, TensorProductPenalty};
pub use truncated_power::{TruncatedPowerBasis, TruncatedPowerDesign};

pub mod adaptive;
pub mod basis;
pub mod basis_nd;
pub mod bspline;
pub mod constraint;
pub mod cspline;
pub mod cyclic;
pub mod derivative;
pub mod duchon;
pub mod error;
pub mod fourier;
mod geometry;
pub mod ispline;
pub mod kernel;
pub mod knots;
mod local;
pub mod monotone;
pub mod mspline;
pub mod natural;
mod numeric;
pub mod on_demand;
pub mod open_uniform;
pub mod order;
pub mod penalty;
pub mod periodic;
mod prepared;
pub mod roughness;
pub mod row_basis;
pub mod tensor;
pub mod tensor_penalty;
pub mod truncated_power;
mod validation;

/// Most commonly used imports from `gamlss-spline`.
pub mod prelude {
    pub use crate::{
        AdaptiveDifferencePenalty, BSplineBasis, BSplineDerivativePenalty, BSplineDesign,
        CSplineBasis, CenteredSplineDesign, ConvexCSplineDesign, CsrParts, CurvatureDirection,
        CyclicDifferencePenalty, CyclicSplineDesign, CyclicSplineSpec, DensePenaltyKernel,
        DiagonalPenaltyKernel, DifferencePenalty, DifferencePenaltyKernel,
        DifferentiableSplineBasis1d, DifferentiableSplineRowBasis, DuchonSmoothness,
        DuchonSplineBasis, DuchonSplineDesign, EdgeMonotonicPenalty, FourierBasis, FourierDesign,
        FourierError, FourierRoughnessPenalty, HelmertContrast, HelmertContrastDesign,
        HelmertContrastPenalty, ISplineBasis, ISplineDesign, KnotPlacement, MSplineBasis,
        MSplineDesign, MonotoneDirection, MonotoneISplineDesign, NaturalCubicRoughnessPenalty,
        NaturalCubicSplineBasis, NaturalCubicSplineDesign, NullSpacePenalty, OnDemandSplineDesign,
        OpenKnotVector, OpenUniformSplineBasis, OpenUniformSplineDesign, PenaltyKernel,
        PeriodicSplineDesign, PeriodicSplineSpec, PreparedCyclicDifferencePenalty,
        PreparedDifferencePenalty, ScaledPenalty, SlopeLimitPenalty, SplineBasis1d, SplineBasisNd,
        SplineError, SplineOrder, SplineRowBasis, SplineRowBasisExt, TensorInteractionDesign,
        TensorProductPenalty, TensorSplineDesign, TripletParts, TruncatedPowerBasis,
        TruncatedPowerDesign, WeightedDifferenceKernel, WeightedDifferencePenalty, pspline_design,
    };
}

#[cfg(test)]
mod tests;
