#![forbid(unsafe_code)]
//! Post-fit diagnostics utilities for Rust-native GAMLSS models.
//!
//! This crate is the intended home for diagnostics that are important for
//! GAMLSS parity but should not be part of the core [`gamlss_core::Family`]
//! contract.
//!
//! Current utilities include:
//!
//! - PIT/CDF residuals;
//! - ordered multivariate Rosenblatt residuals;
//! - CRPS summaries;
//! - reusable diagnostics views for prediction rows.
//!
//! Planned utilities include:
//!
//! - randomized quantile residuals;
//! - worm plot data;
//! - centile curve data;
//! - fitted parameter extraction helpers;
//! - distribution-level prediction helpers.
//!
//! The public surface intentionally grows through small extension APIs.
//! Distribution-specific capabilities should be expressed through extension
//! traits such as [`gamlss_core::HasCdf`] and [`gamlss_core::HasQuantile`],
//! while this crate provides post-fit utilities built on top of compiled model
//! predictions.
//!
//! ```
//! # use gamlss_core::{DenseDesign, Gamlss, Identity, Log, Mu, NoPenalty, ParameterBlock, ParameterBlocks, Sigma};
//! # use gamlss_diagnostics::prelude::*;
//! # use gamlss_family::Normal;
//! # let y = [0.0, 1.0, -1.0];
//! # let blocks = ParameterBlocks::new((
//! #     ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 0),
//! #     ParameterBlock::<Sigma, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 0),
//! # ));
//! # let model = Gamlss::try_new(Normal::<Identity, Log>::new(), blocks, &y)?;
//! let parameters = [0.0, 0.0];
//! let pit = model.pit_values(&parameters)?;
//! let residuals = model.quantile_residuals(&parameters)?;
//! # assert_eq!(pit.len(), residuals.len());
//! # Ok::<_, gamlss_core::ModelError>(())
//! ```

use gamlss_core::{Family, Gamlss, GamlssBlocks, ModelError, ObservationView};

use gamlss_special::unit_normal_quantile;

mod cdf;
mod crps;
mod multivariate;
mod prediction;

pub use cdf::CdfDiagnosticsExt;
pub use crps::CrpsDiagnosticsExt;
pub use multivariate::{
    MarginalCdfDiagnosticsExt, PredictionRosenblattDiagnosticsExt, RosenblattDiagnosticsExt,
};
pub use prediction::{PredictionDiagnosticsExt, PredictionDiagnosticsView};

/// Common diagnostics imports.
pub mod prelude {
    pub use crate::{
        CdfDiagnosticsExt, CrpsDiagnosticsExt, MarginalCdfDiagnosticsExt, PredictionDiagnosticsExt,
        PredictionDiagnosticsView, PredictionRosenblattDiagnosticsExt, RosenblattDiagnosticsExt,
    };
}

fn diagnostic_values_into<F, Blocks, Obs>(
    model: &Gamlss<F, Blocks, Obs>,
    parameters: &[f64],
    out: &mut [f64],
    mut evaluate: impl FnMut(&F, f64, &F::Theta) -> f64,
) -> Result<(), ModelError>
where
    F: for<'row> Family<Observation<'row> = f64>,
    Blocks: GamlssBlocks<F>,
    for<'row> Obs: ObservationView<'row, Observation = f64>,
{
    validate_output_len(model.nobs(), out.len())?;
    let family = model.family();
    let obs = model.obs();
    model.for_each_theta(parameters, |row, theta| {
        out[row] = evaluate(family, obs.observation_at(row), &theta);
    })
}

fn validate_prediction_observations<'obs, Obs>(
    expected: usize,
    obs: &'obs Obs,
) -> Result<(), ModelError>
where
    Obs: ObservationView<'obs> + 'obs,
{
    validate_output_len(expected, obs.len())?;
    obs.validate()
}

const fn validate_output_len(expected: usize, actual: usize) -> Result<(), ModelError> {
    if actual == expected {
        Ok(())
    } else {
        Err(ModelError::ResponseLength { expected, actual })
    }
}

fn checked_rosenblatt_len(nrows: usize, dimension: usize) -> Result<usize, ModelError> {
    nrows
        .checked_mul(dimension)
        .ok_or(ModelError::ArithmeticOverflow {
            context: "row-major Rosenblatt diagnostics length",
        })
}

fn normalize_pit_values(mut values: Vec<f64>) -> Vec<f64> {
    for value in &mut values {
        *value = unit_normal_quantile(*value);
    }
    values
}

#[cfg(test)]
mod tests;
