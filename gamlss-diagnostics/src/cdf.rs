use gamlss_core::{Family, Gamlss, GamlssBlocks, ModelError, ObservationView};

use gamlss_core::HasCdf;

use crate::{PredictionDiagnosticsExt, diagnostic_values_into, normalize_pit_values};

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
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `parameters` has the wrong length.
    fn pit_values(&self, parameters: &[f64]) -> Result<Vec<f64>, ModelError>;

    /// Writes PIT values for training rows into an existing output slice.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `parameters` or `out` have the wrong length.
    fn pit_values_into(&self, parameters: &[f64], out: &mut [f64]) -> Result<(), ModelError>;

    /// Returns PIT values for supplied compatible prediction blocks and observations.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if parameters, prediction blocks, or observations
    /// are incompatible with the fitted model.
    fn pit_values_with_blocks<'obs, PBlocks, PObs>(
        &self,
        parameters: &[f64],
        blocks: &PBlocks,
        obs: &'obs PObs,
    ) -> Result<Vec<f64>, ModelError>
    where
        PBlocks: GamlssBlocks<F>,
        PObs: ObservationView<'obs, Observation = f64> + 'obs;

    /// Writes PIT values for supplied prediction rows into an existing output slice.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if parameters, prediction blocks, observations,
    /// or `out` are incompatible with the fitted model.
    fn pit_values_with_blocks_into<'obs, PBlocks, PObs>(
        &self,
        parameters: &[f64],
        blocks: &PBlocks,
        obs: &'obs PObs,
        out: &mut [f64],
    ) -> Result<(), ModelError>
    where
        PBlocks: GamlssBlocks<F>,
        PObs: ObservationView<'obs, Observation = f64> + 'obs;

    /// Returns normalized quantile residuals for training rows.
    ///
    /// For continuous distributions this is `Phi^-1(F(y_i; theta_i))`, where
    /// `Phi^-1` is the inverse standard-normal CDF and `F` is the family CDF.
    /// Non-finite and out-of-range PIT values propagate as `NaN`; PIT values at
    /// unit interval boundaries map to infinities.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `parameters` has the wrong length.
    fn quantile_residuals(&self, parameters: &[f64]) -> Result<Vec<f64>, ModelError> {
        self.pit_values(parameters).map(normalize_pit_values)
    }

    /// Returns normalized quantile residuals for supplied prediction rows.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if parameters, prediction blocks, or observations
    /// are incompatible with the fitted model.
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
        let mut out = vec![0.0; self.nobs()];
        self.pit_values_into(parameters, &mut out)?;
        Ok(out)
    }

    fn pit_values_into(&self, parameters: &[f64], out: &mut [f64]) -> Result<(), ModelError> {
        diagnostic_values_into(self, parameters, out, |family, observation, theta| {
            family.cdf(observation, theta)
        })
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
        let prediction = self.prediction_view(blocks)?;
        prediction.with_observations(obs)?.pit_values(parameters)
    }

    fn pit_values_with_blocks_into<'obs, PBlocks, PObs>(
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
            .pit_values_into(parameters, out)
    }
}
