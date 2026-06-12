#![forbid(unsafe_code)]
//! Высокоуровневый crate для Rust-native GAMLSS.

/// Типизированные базовые абстракции.
pub use gamlss_core as core;
/// Распределения и реализации likelihood.
pub use gamlss_family as family;
/// Spline-базисы и штрафы.
pub use gamlss_spline as spline;

#[cfg(feature = "formula")]
/// Динамический formula/builder слой.
pub use gamlss_formula as formula;

/// Наиболее часто используемые импорты.
pub mod prelude {
    pub use gamlss_core::prelude::*;
    pub use gamlss_family::prelude::*;
    pub use gamlss_spline::prelude::*;

    #[cfg(feature = "formula")]
    pub use gamlss_formula::prelude::*;
}
