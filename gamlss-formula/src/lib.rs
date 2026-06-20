#![forbid(unsafe_code)]
//! Experimental optional typed formula/builder layer for compiling user model
//! specifications into typed `gamlss-core` models.
//!
//! `gamlss-formula` is a boundary layer: it reads user data through
//! [`DataView`], fits term metadata once, materializes predictor designs, and
//! returns a [`BuiltModel`] artifact. Optimizers, fitting loops, diagnostics,
//! dataframe ownership, and string formula parsing are intentionally outside
//! this crate.
//!
//! This crate is not the primary API of the project. The current primary path is
//! the lower-level typed API in `gamlss-core`, `gamlss-family`, `gamlss-spline`,
//! and `gamlss-transform`. `gamlss-formula` is an experimental convenience layer
//! for curated high-level workflows and is not expected to cover every
//! distribution, link, or parameterization available in the lower-level crates.
//!
//! # Example
//!
//! ```
//! use gamlss_core::Objective;
//! use gamlss_formula::prelude::*;
//!
//! # #[derive(Debug)]
//! # struct TestData {
//! #     y: Vec<f64>,
//! #     x: Vec<f64>,
//! # }
//! # impl DataView for TestData {
//! #     fn nrows(&self) -> usize { self.y.len() }
//! #     fn f64_col(&self, col: &Col<f64>) -> Result<NumericCol<'_>, FormulaError> {
//! #         match col.name() {
//! #             "y" => Ok(NumericCol::Borrowed(&self.y)),
//! #             "x" => Ok(NumericCol::Borrowed(&self.x)),
//! #             name => Err(FormulaError::UnknownColumn(name.to_owned())),
//! #         }
//! #     }
//! # }
//! let data = TestData {
//!     y: vec![0.0, 1.0, 2.0],
//!     x: vec![0.0, 1.0, 2.0],
//! };
//!
//! let y = col::<f64>("y");
//! let x = col::<f64>("x");
//! let mut built = normal()
//!     .response(y)
//!     .mu(intercept() + linear(x))
//!     .sigma(intercept())
//!     .build(&data)?;
//!
//! let theta = vec![0.0, 0.5, -0.2];
//! assert!(built.model_mut().value(&theta)?.is_finite());
//! # Ok::<_, Box<dyn std::error::Error>>(())
//! ```

pub use data::{BoolCol, CatCol, Category, Col, DataView, NumericCol, NumericResponse, col};
pub use error::FormulaError;
pub use penalty::FormulaPenalty;
pub use predictor::FormulaPredictorBlock;
pub use schema::{BuiltModel, ModelSchema, ParameterTerms, PredictionDesign, ResponseSchema};
pub use spec::{
    BetaBlocks, BetaSpec, BuiltBeta, BuiltGamma, BuiltInverseGaussian, BuiltLogNormal, BuiltNormal,
    BuiltWeibull, CompiledBeta, CompiledGamma, CompiledInverseGaussian, CompiledLogNormal,
    CompiledNormal, CompiledWeibull, FormulaBlock, GammaBlocks, GammaSpec, InverseGaussianBlocks,
    InverseGaussianSpec, LogNormalBlocks, LogNormalSpec, ModelSpec, NormalBlocks, NormalSpec,
    WeibullBlocks, WeibullSpec, beta, gamma, inverse_gaussian, log_normal, normal, weibull,
};
pub use terms::{
    CyclicPSplineTerm, FittedTerm, FourierTerm, MonotoneTerm, PSplineTerm, TensorPSplineTerm,
    TermExpr, TermSpec, cyclic_pspline, factor, fourier, indicator, interaction, intercept, linear,
    monotone, no_intercept, offset, pspline, tensor_pspline,
};

mod compile;
mod data;
mod error;
mod penalty;
mod predictor;
mod schema;
mod spec;
mod terms;

/// Common imports for the typed formula layer.
pub mod prelude {
    pub use crate::{
        BetaSpec, BoolCol, BuiltBeta, BuiltGamma, BuiltInverseGaussian, BuiltLogNormal, BuiltModel,
        BuiltNormal, BuiltWeibull, CatCol, Category, Col, CompiledBeta, CompiledGamma,
        CompiledInverseGaussian, CompiledLogNormal, CompiledNormal, CompiledWeibull,
        CyclicPSplineTerm, DataView, FittedTerm, FormulaError, FormulaPenalty,
        FormulaPredictorBlock, FourierTerm, GammaSpec, InverseGaussianSpec, LogNormalSpec,
        ModelSchema, ModelSpec, MonotoneTerm, NormalSpec, NumericCol, NumericResponse, PSplineTerm,
        ParameterTerms, PredictionDesign, ResponseSchema, TensorPSplineTerm, TermExpr, TermSpec,
        WeibullSpec, beta, col, cyclic_pspline, factor, fourier, gamma, indicator, interaction,
        intercept, inverse_gaussian, linear, log_normal, monotone, no_intercept, normal, offset,
        pspline, tensor_pspline, weibull,
    };
}

#[cfg(test)]
mod tests;
