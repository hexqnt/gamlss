#![forbid(unsafe_code)]
//! Bayesian potential-energy wrappers kept separate from distribution families.
//!
//! This crate deliberately starts with one small, statistically explicit
//! vertical slice: normalized Gaussian priors on predictor coefficients plus
//! the summed, unpenalized compiled likelihood. It contains no sampler,
//! autodiff backend, chain storage, or implicit conversion from fitting
//! penalties to priors.

pub use potential::{PosteriorPotential, WorkspacePosteriorPotential};
pub use prior::{CoefficientPrior, GaussianCoefficientPrior};

mod potential;
mod prior;

/// Common imports for the Bayesian vertical slice.
pub mod prelude {
    pub use crate::{
        CoefficientPrior, GaussianCoefficientPrior, PosteriorPotential, WorkspacePosteriorPotential,
    };
}

const HALF_LOG_TWO_PI: f64 = 0.918_938_533_204_672_7;

#[cfg(test)]
mod tests;
