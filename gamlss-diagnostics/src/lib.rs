#![forbid(unsafe_code)]
//! Post-fit diagnostics utilities for Rust-native GAMLSS models.
//!
//! This crate is the intended home for diagnostics that are important for
//! GAMLSS parity but should not be part of the core [`gamlss_core::Family`]
//! contract.
//!
//! Planned utilities include:
//!
//! - randomized quantile residuals;
//! - PIT/CDF residuals;
//! - CRPS summaries;
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
//! #     ParameterBlock::<Mu, Identity, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 0),
//! #     ParameterBlock::<Sigma, Log, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 0),
//! # ));
//! # let model = Gamlss::try_new(Normal::<Identity, Log>::new(), blocks, &y)?;
//! let theta = [0.0, 0.0];
//! let pit = model.pit_values(&theta)?;
//! let residuals = model.quantile_residuals(&theta)?;
//! # assert_eq!(pit.len(), residuals.len());
//! # Ok::<_, gamlss_core::ModelError>(())
//! ```

use gamlss_core::{Family, Gamlss, GamlssBlocks, HasCdf, HasCrps, ModelError, ObservationView};

/// CDF-based diagnostics for fitted GAMLSS models.
pub trait CdfDiagnosticsExt {
    /// Returns probability integral transform values for training rows.
    ///
    /// The returned vector has one value per training observation, in row
    /// order. Invalid observation or parameter domains are represented by the
    /// family CDF result, usually `NaN`, rather than an additional diagnostics
    /// error.
    fn pit_values(&self, theta: &[f64]) -> Result<Vec<f64>, ModelError>;

    /// Returns normalized quantile residuals for training rows.
    ///
    /// For continuous distributions this is `Phi^-1(F(y_i; theta_i))`, where
    /// `Phi^-1` is the inverse standard-normal CDF and `F` is the family CDF.
    /// Non-finite PIT values propagate as `NaN`; PIT values at or beyond the
    /// unit interval boundaries map to infinities.
    fn quantile_residuals(&self, theta: &[f64]) -> Result<Vec<f64>, ModelError> {
        self.pit_values(theta)
            .map(|values| values.into_iter().map(inverse_unit_normal_cdf).collect())
    }
}

impl<F, Blocks, Obs> CdfDiagnosticsExt for Gamlss<F, Blocks, Obs>
where
    F: HasCdf + for<'row> Family<Observation<'row> = f64>,
    Blocks: GamlssBlocks<F>,
    for<'row> Obs: ObservationView<'row, Observation = f64>,
{
    fn pit_values(&self, theta: &[f64]) -> Result<Vec<f64>, ModelError> {
        let parameters = self.predict_theta(theta)?;
        Ok(parameters
            .into_iter()
            .enumerate()
            .map(|(row, parameters)| {
                let observation = self.obs.observation_at(row);
                self.family.cdf(observation, parameters)
            })
            .collect())
    }
}

/// CRPS-based diagnostics for fitted GAMLSS models.
pub trait CrpsDiagnosticsExt {
    /// Returns CRPS values for training rows.
    ///
    /// The returned vector has one value per training observation, in row
    /// order. Invalid observation or parameter domains are represented by the
    /// family CRPS result, usually `NaN`.
    fn crps_values(&self, theta: &[f64]) -> Result<Vec<f64>, ModelError>;

    /// Returns the arithmetic mean of [`Self::crps_values`].
    fn mean_crps(&self, theta: &[f64]) -> Result<f64, ModelError> {
        let values = self.crps_values(theta)?;
        Ok(values.iter().sum::<f64>() / values.len() as f64)
    }
}

impl<F, Blocks, Obs> CrpsDiagnosticsExt for Gamlss<F, Blocks, Obs>
where
    F: HasCrps + for<'row> Family<Observation<'row> = f64>,
    Blocks: GamlssBlocks<F>,
    for<'row> Obs: ObservationView<'row, Observation = f64>,
{
    fn crps_values(&self, theta: &[f64]) -> Result<Vec<f64>, ModelError> {
        let parameters = self.predict_theta(theta)?;
        Ok(parameters
            .into_iter()
            .enumerate()
            .map(|(row, parameters)| {
                let observation = self.obs.observation_at(row);
                self.family.crps(observation, parameters)
            })
            .collect())
    }
}

fn inverse_unit_normal_cdf(probability: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const LOW: f64 = 0.024_25;
    const HIGH: f64 = 1.0 - LOW;

    if probability.is_nan() {
        return f64::NAN;
    }
    if probability <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if probability >= 1.0 {
        return f64::INFINITY;
    }

    if probability < LOW {
        let q = (-2.0 * probability.ln()).sqrt();
        return (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }

    if probability <= HIGH {
        let q = probability - 0.5;
        let r = q * q;
        return (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0);
    }

    let q = (-2.0 * (1.0 - probability).ln()).sqrt();
    -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
        / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
}

/// Common diagnostics imports.
pub mod prelude {
    pub use crate::{CdfDiagnosticsExt, CrpsDiagnosticsExt};
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{
        DenseDesign, Gamlss, Identity, LinearPredictorBlock, Log, ModelError, Mu, NoPenalty,
        ParameterBlock, ParameterBlocks, Sigma,
    };
    use gamlss_family::Normal;

    use super::{CdfDiagnosticsExt, CrpsDiagnosticsExt, inverse_unit_normal_cdf};

    type TestModel<'a> = Gamlss<
        Normal<Identity, Log>,
        (
            ParameterBlock<Mu, Identity, LinearPredictorBlock<DenseDesign>, NoPenalty>,
            ParameterBlock<Sigma, Log, LinearPredictorBlock<DenseDesign>, NoPenalty>,
        ),
        &'a [f64],
    >;

    fn normal_intercept_model(y: &[f64]) -> TestModel<'_> {
        let blocks = ParameterBlocks::new((
            ParameterBlock::<Mu, Identity, _, _>::linear(
                DenseDesign::intercept(y.len()),
                NoPenalty,
                0,
            ),
            ParameterBlock::<Sigma, Log, _, _>::linear(
                DenseDesign::intercept(y.len()),
                NoPenalty,
                0,
            ),
        ));

        Gamlss::try_new(Normal::<Identity, Log>::new(), blocks, y).expect("valid normal model")
    }

    #[test]
    fn pit_values_use_training_observations_and_fitted_parameters() {
        let y = [0.0, 1.0, -1.0];
        let model = normal_intercept_model(&y);

        let pit = model.pit_values(&[0.0, 0.0]).expect("valid theta");

        assert_relative_eq!(pit[0], 0.5, epsilon = 1.0e-7);
        assert_relative_eq!(pit[1], 0.841_344_746, epsilon = 1.0e-7);
        assert_relative_eq!(pit[2], 0.158_655_254, epsilon = 1.0e-7);
    }

    #[test]
    fn quantile_residuals_transform_pit_values_to_standard_normal_scale() {
        let y = [-1.0, 0.0, 1.0];
        let model = normal_intercept_model(&y);

        let residuals = model.quantile_residuals(&[0.0, 0.0]).expect("valid theta");

        assert_relative_eq!(residuals[0], -1.0, epsilon = 1.0e-6);
        assert_relative_eq!(residuals[1], 0.0, epsilon = 1.0e-6);
        assert_relative_eq!(residuals[2], 1.0, epsilon = 1.0e-6);
    }

    #[test]
    fn quantile_residuals_map_invalid_and_boundary_pit_values() {
        assert!(inverse_unit_normal_cdf(f64::NAN).is_nan());
        assert!(inverse_unit_normal_cdf(0.0).is_infinite());
        assert!(inverse_unit_normal_cdf(0.0).is_sign_negative());
        assert!(inverse_unit_normal_cdf(1.0).is_infinite());
        assert!(inverse_unit_normal_cdf(1.0).is_sign_positive());
    }

    #[test]
    fn pit_values_reject_wrong_theta_length() {
        let y = [0.0];
        let model = normal_intercept_model(&y);

        assert_eq!(
            model.pit_values(&[]).unwrap_err(),
            ModelError::BetaLength {
                expected: 2,
                actual: 0
            }
        );
    }

    #[test]
    fn crps_values_use_training_observations_and_fitted_parameters() {
        let y = [0.0, 1.0];
        let model = normal_intercept_model(&y);

        let crps = model.crps_values(&[0.0, 0.0]).expect("valid theta");

        assert_relative_eq!(crps[0], 0.233_694_977_255_109_13, epsilon = 1.0e-12);
        assert_relative_eq!(crps[1], 0.602_441_337_825_803, epsilon = 1.0e-12);
        assert_relative_eq!(
            model.mean_crps(&[0.0, 0.0]).expect("valid theta"),
            (crps[0] + crps[1]) / 2.0,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn crps_values_reject_wrong_theta_length() {
        let y = [0.0];
        let model = normal_intercept_model(&y);

        assert_eq!(
            model.crps_values(&[]).unwrap_err(),
            ModelError::BetaLength {
                expected: 2,
                actual: 0
            }
        );
    }

    #[test]
    fn diagnostics_prelude_exposes_extension_traits() {
        use crate::prelude::*;

        let y = [0.0];
        let model = normal_intercept_model(&y);

        assert_relative_eq!(
            model.pit_values(&[0.0, 0.0]).unwrap()[0],
            0.5,
            epsilon = 1.0e-7
        );
        assert_relative_eq!(
            model.quantile_residuals(&[0.0, 0.0]).unwrap()[0],
            0.0,
            epsilon = 1.0e-6
        );
        assert_relative_eq!(
            model.mean_crps(&[0.0, 0.0]).unwrap(),
            0.233_694_977_255_109_13,
            epsilon = 1.0e-12
        );
    }
}
