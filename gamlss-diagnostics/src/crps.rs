use gamlss_core::{Family, Gamlss, GamlssBlocks, ModelError, ObservationView};

use gamlss_core::HasCrps;

use crate::{PredictionDiagnosticsExt, diagnostic_values_into, validate_output_len};

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
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `parameters` has the wrong length.
    fn crps_values(&self, parameters: &[f64]) -> Result<Vec<f64>, ModelError>;

    /// Writes CRPS values for training rows into an existing output slice.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `parameters` or `out` have the wrong length.
    fn crps_values_into(&self, parameters: &[f64], out: &mut [f64]) -> Result<(), ModelError>;

    /// Returns CRPS values for supplied compatible prediction blocks and observations.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if parameters, prediction blocks, or observations
    /// are incompatible with the fitted model.
    fn crps_values_with_blocks<'obs, PBlocks, PObs>(
        &self,
        parameters: &[f64],
        blocks: &PBlocks,
        obs: &'obs PObs,
    ) -> Result<Vec<f64>, ModelError>
    where
        PBlocks: GamlssBlocks<F>,
        PObs: ObservationView<'obs, Observation = f64> + 'obs;

    /// Writes CRPS values for supplied prediction rows into an existing output slice.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if parameters, prediction blocks, observations,
    /// or `out` are incompatible with the fitted model.
    fn crps_values_with_blocks_into<'obs, PBlocks, PObs>(
        &self,
        parameters: &[f64],
        blocks: &PBlocks,
        obs: &'obs PObs,
        out: &mut [f64],
    ) -> Result<(), ModelError>
    where
        PBlocks: GamlssBlocks<F>,
        PObs: ObservationView<'obs, Observation = f64> + 'obs;

    /// Returns the arithmetic mean of [`Self::crps_values`].
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `parameters` has the wrong length.
    #[allow(clippy::cast_precision_loss)]
    fn mean_crps(&self, parameters: &[f64]) -> Result<f64, ModelError> {
        let values = self.crps_values(parameters)?;
        Ok(values.iter().sum::<f64>() / values.len() as f64)
    }

    /// Returns the observation-weighted mean CRPS for training rows.
    ///
    /// If all observation weights are zero, returns `NaN`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `parameters` has the wrong length.
    fn weighted_mean_crps(&self, parameters: &[f64]) -> Result<f64, ModelError>;
}

impl<F, Blocks, Obs> CrpsDiagnosticsExt<F, Blocks> for Gamlss<F, Blocks, Obs>
where
    F: HasCrps + for<'row> Family<Observation<'row> = f64>,
    Blocks: GamlssBlocks<F>,
    for<'row> Obs: ObservationView<'row, Observation = f64>,
{
    fn crps_values(&self, parameters: &[f64]) -> Result<Vec<f64>, ModelError> {
        let mut out = vec![0.0; self.nobs()];
        self.crps_values_into(parameters, &mut out)?;
        Ok(out)
    }

    fn crps_values_into(&self, parameters: &[f64], out: &mut [f64]) -> Result<(), ModelError> {
        diagnostic_values_into(self, parameters, out, |family, observation, theta| {
            family.crps(observation, theta)
        })
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
        let prediction = self.prediction_view(blocks)?;
        prediction.with_observations(obs)?.crps_values(parameters)
    }

    fn crps_values_with_blocks_into<'obs, PBlocks, PObs>(
        &self,
        parameters: &[f64],
        blocks: &PBlocks,
        obs: &'obs PObs,
        out: &mut [f64],
    ) -> Result<(), ModelError>
    where
        PBlocks: GamlssBlocks<F>,
        PObs: ObservationView<'obs, Observation = f64> + 'obs,
    {
        let prediction = self.prediction_view(blocks)?;
        prediction
            .with_observations(obs)?
            .crps_values_into(parameters, out)
    }

    #[allow(clippy::cast_precision_loss)]
    fn mean_crps(&self, parameters: &[f64]) -> Result<f64, ModelError> {
        let count = self.nobs();
        validate_output_len(self.nobs(), count)?;
        // Match Iterator::sum's left-to-right addition, including signed zero,
        // without retaining one CRPS value per observation.
        let mut sum = -0.0;
        self.for_each_theta(parameters, |row, theta| {
            sum += self.family().crps(self.obs().observation_at(row), &theta);
        })?;
        Ok(sum / count as f64)
    }

    fn weighted_mean_crps(&self, parameters: &[f64]) -> Result<f64, ModelError> {
        let family = self.family();
        let obs = self.obs();
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        self.for_each_theta(parameters, |row, theta| {
            let weight = obs.weight_at(row);
            if weight == 0.0 {
                return;
            }

            let value = family.crps(obs.observation_at(row), &theta);
            weighted_sum = weight.mul_add(value, weighted_sum);
            weight_sum += weight;
        })?;

        Ok(weighted_sum / weight_sum)
    }
}
