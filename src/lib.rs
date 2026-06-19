#![forbid(unsafe_code)]
//! Высокоуровневый crate для Rust-native GAMLSS.
//!
//! `gamlss` реэкспортирует typed core, готовые distribution families,
//! spline/predictor building blocks и target transforms. Основной подход сейчас
//! — низкоуровневое typed API через [`core`], [`family`], [`spline`] и
//! [`transform`].
//!
//! При включённой feature `formula` также доступен [`formula`] namespace. Этот
//! слой является экспериментальным optional convenience API: он компилирует
//! curated high-level builder specifications в typed core models, но не является
//! основным API и не обещает покрывать все distributions, links и
//! parameterizations из низкоуровневых crate-ов.
//!
//! Feature `rand` включает sampling API в [`family`] и соответствует feature
//! `gamlss-family/rand`.
//!
//! # Основные возможности
//!
//! - typed [`core::ParameterBlock`] для каждого параметра распределения;
//! - [`core::ParameterBlocks`] для автоматического layout offsets в общем
//!   beta-векторе;
//! - unweighted и weighted модели через [`core::Gamlss::try_new`] и
//!   [`core::Gamlss::try_new_weighted`];
//! - prediction API для training rows и совместимых prediction blocks;
//! - post-fit diagnostics namespace через [`diagnostics`];
//! - experimental formula builders через [`formula::ModelSpec`] при включённой
//!   feature `formula`.
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
//!     gamlss::family::DefaultNormal::new(),
//!     blocks,
//!     &y,
//!     &weights,
//! )?;
//!
//! let theta = vec![0.0, 0.5, -0.2];
//! let fitted = model.predict_theta(&theta)?;
//! assert_eq!(fitted.len(), y.len());
//! # Ok::<_, gamlss::core::ModelError>(())
//! ```
//!
#![doc = include_str!("../docs/project-structure.md")]

/// Типизированные базовые абстракции.
pub use gamlss_core as core;
/// Post-fit diagnostics utilities.
pub use gamlss_diagnostics as diagnostics;
/// Распределения и реализации likelihood.
pub use gamlss_family as family;
/// Spline-базисы и штрафы.
pub use gamlss_spline as spline;
/// Transform-слой для response/target preprocessing.
pub use gamlss_transform as transform;

#[cfg(feature = "formula")]
/// Экспериментальный optional formula/builder слой.
pub use gamlss_formula as formula;

/// Наиболее часто используемые импорты.
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
