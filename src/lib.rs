#![forbid(unsafe_code)]
//! High-level crate for Rust-native GAMLSS.
//!
//! `gamlss` re-exports the typed core, ready-made distribution families,
//! spline/predictor building blocks, special functions and target transforms.
//! The primary approach is a low-level typed API through [`core`], [`family`],
//! [`spline`] and [`transform`].
//!
//! When the `formula` feature is enabled, the [`formula`] namespace is also
//! available. This layer is an experimental optional convenience API: it compiles
//! curated high-level builder specifications into typed core models, but is not
//! the primary API and does not promise to cover all distributions, links and
//! parameterizations from the low-level crates.
//!
//! The `rand` feature enables the sampling API in [`family`] and corresponds to
//! the `gamlss-family/rand` feature.
//!
//! # Key features
//!
//! - typed [`core::ParameterBlock`] for each distribution parameter;
//! - [`core::ParameterBlocks`] for automatic layout offsets in the common
//!   beta vector;
//! - unweighted and weighted models via [`core::Gamlss::try_new`] and
//!   [`core::Gamlss::try_new_weighted`];
//! - prediction API for training rows and compatible prediction blocks;
//! - post-fit diagnostics namespace via [`diagnostics`];
//! - experimental formula builders via [`formula::ModelSpec`] when the
//!   `formula` feature is enabled.
//!
//! # Example
//!
//! ```
//! use gamlss::prelude::*;
//!
//! let y = [0.0, 1.0, 2.0];
//! let weights = [1.0, 0.5, 1.0];
//!
//! let blocks = ParameterBlocks::new((
//!     ParameterBlock::<Mu, Identity, _, _>::linear(
//!         DenseDesign::from_rows(&[[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]),
//!         NoPenalty,
//!         0,
//!     ),
//!     ParameterBlock::<Sigma, Log, _, _>::linear(
//!         DenseDesign::intercept(y.len()),
//!         NoPenalty,
//!         0,
//!     ),
//! ));
//!
//! let model = Gamlss::try_new_weighted(
//!     gamlss::family::NormalMuSigma::new(),
//!     blocks,
//!     &y,
//!     &weights,
//! )?;
//!
//! let parameters = model.initial_parameters()?;
//! let fitted_theta = model.predict_theta(&parameters)?;
//! assert_eq!(fitted_theta.len(), y.len());
//! # Ok::<_, gamlss::core::ModelError>(())
//! ```
//!
#![doc = include_str!("../docs/project-structure.md")]

/// Typed core abstractions.
pub use gamlss_core as core;
/// Post-fit diagnostics utilities.
pub use gamlss_diagnostics as diagnostics;
/// Distributions and likelihood implementations.
pub use gamlss_family as family;
/// Special functions and numerical helpers.
pub use gamlss_special as special;
/// Spline bases and penalties.
pub use gamlss_spline as spline;
/// Transform layer for response/target preprocessing.
pub use gamlss_transform as transform;

#[cfg(feature = "formula")]
/// Experimental optional formula/builder layer.
pub use gamlss_formula as formula;

/// Most commonly used imports.
pub mod prelude {
    pub use gamlss_core::prelude::*;
    pub use gamlss_diagnostics::prelude::*;
    pub use gamlss_family::prelude::*;
    pub use gamlss_spline::prelude::*;
    pub use gamlss_transform::{
        AsinhScale, AsinhScaleState, IdentityPositive, IdentityPositiveState, Log as TargetLog,
        Log1pShift, Log1pShiftState, LogState, Standardize, StandardizeState, TargetTransform,
        TransformError,
    };

    #[cfg(feature = "formula")]
    pub use gamlss_formula::prelude::*;
}
