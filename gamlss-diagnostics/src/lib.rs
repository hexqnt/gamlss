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
//! let parameters = [0.0, 0.0];
//! let pit = model.pit_values(&parameters)?;
//! let residuals = model.quantile_residuals(&parameters)?;
//! # assert_eq!(pit.len(), residuals.len());
//! # Ok::<_, gamlss_core::ModelError>(())
//! ```

use gamlss_core::{Family, Gamlss, GamlssBlocks, HasCdf, HasCrps, ModelError, ObservationView};
use gamlss_special::unit_normal_quantile;

/// Common diagnostics imports.
pub mod prelude {
    pub use crate::{CdfDiagnosticsExt, CrpsDiagnosticsExt};
}

/// CDF-based diagnostics for fitted GAMLSS models.
pub trait CdfDiagnosticsExt<F, Blocks>
where
    F: HasCdf + for<'row> Family<Observation<'row> = f64>,
    Blocks: GamlssBlocks<F>,
{
    /// Returns probability integral transform values for training rows.
    ///
    /// The returned vector has one value per training observation, in row
    /// order. Invalid observation or parameter domains are represented by the
    /// family CDF result, usually `NaN`, rather than an additional diagnostics
    /// error.
    fn pit_values(&self, parameters: &[f64]) -> Result<Vec<f64>, ModelError>;

    /// Returns PIT values for supplied compatible prediction blocks and observations.
    fn pit_values_with_blocks<'obs, PBlocks, PObs>(
        &self,
        parameters: &[f64],
        blocks: &PBlocks,
        obs: &'obs PObs,
    ) -> Result<Vec<f64>, ModelError>
    where
        PBlocks: GamlssBlocks<F>,
        PObs: ObservationView<'obs, Observation = f64> + 'obs;

    /// Returns normalized quantile residuals for training rows.
    ///
    /// For continuous distributions this is `Phi^-1(F(y_i; theta_i))`, where
    /// `Phi^-1` is the inverse standard-normal CDF and `F` is the family CDF.
    /// Non-finite and out-of-range PIT values propagate as `NaN`; PIT values at
    /// unit interval boundaries map to infinities.
    fn quantile_residuals(&self, parameters: &[f64]) -> Result<Vec<f64>, ModelError> {
        self.pit_values(parameters).map(normalize_pit_values)
    }

    /// Returns normalized quantile residuals for supplied prediction rows.
    fn quantile_residuals_with_blocks<'obs, PBlocks, PObs>(
        &self,
        parameters: &[f64],
        blocks: &PBlocks,
        obs: &'obs PObs,
    ) -> Result<Vec<f64>, ModelError>
    where
        PBlocks: GamlssBlocks<F>,
        PObs: ObservationView<'obs, Observation = f64> + 'obs,
    {
        self.pit_values_with_blocks(parameters, blocks, obs)
            .map(normalize_pit_values)
    }
}

impl<F, Blocks, Obs> CdfDiagnosticsExt<F, Blocks> for Gamlss<F, Blocks, Obs>
where
    F: HasCdf + for<'row> Family<Observation<'row> = f64>,
    Blocks: GamlssBlocks<F>,
    for<'row> Obs: ObservationView<'row, Observation = f64>,
{
    fn pit_values(&self, parameters: &[f64]) -> Result<Vec<f64>, ModelError> {
        let theta = self.predict_theta(parameters)?;
        Ok(map_diagnostic_values(
            theta,
            self.obs(),
            |observation, theta| self.family().cdf(observation, theta),
        ))
    }

    fn pit_values_with_blocks<'obs, PBlocks, PObs>(
        &self,
        parameters: &[f64],
        blocks: &PBlocks,
        obs: &'obs PObs,
    ) -> Result<Vec<f64>, ModelError>
    where
        PBlocks: GamlssBlocks<F>,
        PObs: ObservationView<'obs, Observation = f64> + 'obs,
    {
        validate_prediction_observations(blocks.nrows(), obs)?;
        let theta = self.predict_theta_with_blocks(parameters, blocks)?;
        Ok(map_diagnostic_values(theta, obs, |observation, theta| {
            self.family().cdf(observation, theta)
        }))
    }
}

/// CRPS-based diagnostics for fitted GAMLSS models.
pub trait CrpsDiagnosticsExt<F, Blocks>
where
    F: HasCrps + for<'row> Family<Observation<'row> = f64>,
    Blocks: GamlssBlocks<F>,
{
    /// Returns CRPS values for training rows.
    ///
    /// The returned vector has one value per training observation, in row
    /// order. Invalid observation or parameter domains are represented by the
    /// family CRPS result, usually `NaN`.
    fn crps_values(&self, parameters: &[f64]) -> Result<Vec<f64>, ModelError>;

    /// Returns CRPS values for supplied compatible prediction blocks and observations.
    fn crps_values_with_blocks<'obs, PBlocks, PObs>(
        &self,
        parameters: &[f64],
        blocks: &PBlocks,
        obs: &'obs PObs,
    ) -> Result<Vec<f64>, ModelError>
    where
        PBlocks: GamlssBlocks<F>,
        PObs: ObservationView<'obs, Observation = f64> + 'obs;

    /// Returns the arithmetic mean of [`Self::crps_values`].
    fn mean_crps(&self, parameters: &[f64]) -> Result<f64, ModelError> {
        let values = self.crps_values(parameters)?;
        Ok(values.iter().sum::<f64>() / values.len() as f64)
    }

    /// Returns the observation-weighted mean CRPS for training rows.
    ///
    /// If all observation weights are zero, returns `NaN`.
    fn weighted_mean_crps(&self, parameters: &[f64]) -> Result<f64, ModelError>;
}

impl<F, Blocks, Obs> CrpsDiagnosticsExt<F, Blocks> for Gamlss<F, Blocks, Obs>
where
    F: HasCrps + for<'row> Family<Observation<'row> = f64>,
    Blocks: GamlssBlocks<F>,
    for<'row> Obs: ObservationView<'row, Observation = f64>,
{
    fn crps_values(&self, parameters: &[f64]) -> Result<Vec<f64>, ModelError> {
        let theta = self.predict_theta(parameters)?;
        Ok(map_diagnostic_values(
            theta,
            self.obs(),
            |observation, theta| self.family().crps(observation, theta),
        ))
    }

    fn crps_values_with_blocks<'obs, PBlocks, PObs>(
        &self,
        parameters: &[f64],
        blocks: &PBlocks,
        obs: &'obs PObs,
    ) -> Result<Vec<f64>, ModelError>
    where
        PBlocks: GamlssBlocks<F>,
        PObs: ObservationView<'obs, Observation = f64> + 'obs,
    {
        validate_prediction_observations(blocks.nrows(), obs)?;
        let theta = self.predict_theta_with_blocks(parameters, blocks)?;
        Ok(map_diagnostic_values(theta, obs, |observation, theta| {
            self.family().crps(observation, theta)
        }))
    }

    fn weighted_mean_crps(&self, parameters: &[f64]) -> Result<f64, ModelError> {
        let theta = self.predict_theta(parameters)?;
        let family = self.family();
        let obs = self.obs();
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for (row, theta) in theta.into_iter().enumerate() {
            let weight = obs.weight_at(row);
            if weight == 0.0 {
                continue;
            }

            let value = family.crps(obs.observation_at(row), theta);
            weighted_sum += weight * value;
            weight_sum += weight;
        }

        Ok(weighted_sum / weight_sum)
    }
}

fn map_diagnostic_values<'obs, Theta, Obs>(
    parameters: Vec<Theta>,
    obs: &'obs Obs,
    mut evaluate: impl FnMut(f64, Theta) -> f64,
) -> Vec<f64>
where
    Obs: ObservationView<'obs, Observation = f64> + 'obs,
{
    parameters
        .into_iter()
        .enumerate()
        .map(|(row, parameters)| evaluate(obs.observation_at(row), parameters))
        .collect()
}

fn validate_prediction_observations<'obs, Obs>(
    expected: usize,
    obs: &'obs Obs,
) -> Result<(), ModelError>
where
    Obs: ObservationView<'obs, Observation = f64> + 'obs,
{
    let actual = obs.len();
    if actual != expected {
        return Err(ModelError::ResponseLength { expected, actual });
    }
    obs.validate()
}

fn normalize_pit_values(values: Vec<f64>) -> Vec<f64> {
    values.into_iter().map(unit_normal_quantile).collect()
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{
        DenseDesign, Gamlss, Identity, LinearPredictorBlock, Log, ModelError, Mu, NoPenalty,
        ParameterBlock, ParameterBlocks, Sigma,
    };
    use gamlss_family::Normal;

    use super::{CdfDiagnosticsExt, CrpsDiagnosticsExt, normalize_pit_values};

    type TestModel<'a> = Gamlss<
        Normal<Identity, Log>,
        (
            ParameterBlock<Mu, Identity, LinearPredictorBlock<DenseDesign>, NoPenalty>,
            ParameterBlock<Sigma, Log, LinearPredictorBlock<DenseDesign>, NoPenalty>,
        ),
        &'a [f64],
    >;

    type WeightedTestModel<'a> = Gamlss<
        Normal<Identity, Log>,
        (
            ParameterBlock<Mu, Identity, LinearPredictorBlock<DenseDesign>, NoPenalty>,
            ParameterBlock<Sigma, Log, LinearPredictorBlock<DenseDesign>, NoPenalty>,
        ),
        (&'a [f64], &'a [f64]),
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

    fn weighted_normal_intercept_model<'a>(
        y: &'a [f64],
        weights: &'a [f64],
    ) -> WeightedTestModel<'a> {
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

        Gamlss::try_new_weighted(Normal::<Identity, Log>::new(), blocks, y, weights)
            .expect("valid weighted normal model")
    }

    #[test]
    fn pit_values_use_training_observations_and_fitted_parameters() {
        let y = [0.0, 1.0, -1.0];
        let model = normal_intercept_model(&y);

        let pit = model.pit_values(&[0.0, 0.0]).expect("valid parameters");

        assert_relative_eq!(pit[0], 0.5, epsilon = 1.0e-7);
        assert_relative_eq!(pit[1], 0.841_344_746, epsilon = 1.0e-7);
        assert_relative_eq!(pit[2], 0.158_655_254, epsilon = 1.0e-7);
    }

    #[test]
    fn quantile_residuals_transform_pit_values_to_standard_normal_scale() {
        let y = [-1.0, 0.0, 1.0];
        let model = normal_intercept_model(&y);

        let residuals = model
            .quantile_residuals(&[0.0, 0.0])
            .expect("valid parameters");

        assert_relative_eq!(residuals[0], -1.0, epsilon = 1.0e-6);
        assert_relative_eq!(residuals[1], 0.0, epsilon = 1.0e-6);
        assert_relative_eq!(residuals[2], 1.0, epsilon = 1.0e-6);
    }

    #[test]
    fn quantile_residuals_map_invalid_and_boundary_pit_values() {
        let residuals = normalize_pit_values(vec![f64::NAN, -0.1, 0.0, 1.0, 1.1]);

        assert!(residuals[0].is_nan());
        assert!(residuals[1].is_nan());
        assert!(residuals[2].is_infinite());
        assert!(residuals[2].is_sign_negative());
        assert!(residuals[3].is_infinite());
        assert!(residuals[3].is_sign_positive());
        assert!(residuals[4].is_nan());
    }

    #[test]
    fn pit_values_reject_wrong_parameter_length() {
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

        let crps = model.crps_values(&[0.0, 0.0]).expect("valid parameters");

        assert_relative_eq!(crps[0], 0.233_694_977_255_109_13, epsilon = 1.0e-12);
        assert_relative_eq!(crps[1], 0.602_441_357_627_616_5, epsilon = 1.0e-12);
        assert_relative_eq!(
            model.mean_crps(&[0.0, 0.0]).expect("valid parameters"),
            (crps[0] + crps[1]) / 2.0,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn diagnostics_with_blocks_match_training_helpers_for_training_blocks() {
        let y = [-1.0, 0.0, 1.0];
        let model = normal_intercept_model(&y);
        let parameters = [0.0, 0.0];
        let obs = &y[..];

        assert_eq!(
            model
                .pit_values_with_blocks(&parameters, model.blocks(), &obs)
                .unwrap(),
            model.pit_values(&parameters).unwrap()
        );
        assert_eq!(
            model
                .quantile_residuals_with_blocks(&parameters, model.blocks(), &obs)
                .unwrap(),
            model.quantile_residuals(&parameters).unwrap()
        );
        assert_eq!(
            model
                .crps_values_with_blocks(&parameters, model.blocks(), &obs)
                .unwrap(),
            model.crps_values(&parameters).unwrap()
        );
    }

    #[test]
    fn diagnostics_with_blocks_reject_observation_row_mismatch() {
        let y = [0.0, 1.0];
        let model = normal_intercept_model(&y);
        let obs = &[0.0][..];

        assert_eq!(
            model
                .pit_values_with_blocks(&[0.0, 0.0], model.blocks(), &obs)
                .unwrap_err(),
            ModelError::ResponseLength {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn weighted_mean_crps_uses_observation_weights() {
        let y = [0.0, 1.0];
        let weights = [0.0, 2.0];
        let model = weighted_normal_intercept_model(&y, &weights);
        let crps = model.crps_values(&[0.0, 0.0]).unwrap();

        assert_relative_eq!(
            model.weighted_mean_crps(&[0.0, 0.0]).unwrap(),
            crps[1],
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn weighted_mean_crps_skips_zero_weighted_observations_before_crps() {
        let y = [f64::NAN, 1.0];
        let weights = [0.0, 2.0];
        let model = weighted_normal_intercept_model(&y, &weights);

        assert_relative_eq!(
            model.weighted_mean_crps(&[0.0, 0.0]).unwrap(),
            0.602_441_357_627_616_5,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn weighted_mean_crps_returns_nan_when_all_weights_are_zero() {
        let y = [0.0, 1.0];
        let weights = [0.0, 0.0];
        let model = weighted_normal_intercept_model(&y, &weights);

        assert!(model.weighted_mean_crps(&[0.0, 0.0]).unwrap().is_nan());
    }

    #[test]
    fn crps_values_reject_wrong_parameter_length() {
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
