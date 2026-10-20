use gamlss_core::{
    Family, GamlssBlocks, HasCdf, HasCrps, ModelError, ObservationView, PredictionView,
};

use crate::{normalize_pit_values, validate_output_len, validate_prediction_observations};

/// Validated diagnostics view over prediction rows and observations.
///
/// This view is built from a core [`PredictionView`] plus an observation view,
/// validating the prediction observation length and observation invariants once.
/// Reuse it when computing multiple diagnostics over the same prediction rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PredictionDiagnosticsView<'a, 'obs, F, PBlocks, PObs> {
    prediction: PredictionView<'a, F, PBlocks>,
    obs: &'obs PObs,
}

impl<'a, 'obs, F, PBlocks, PObs> PredictionDiagnosticsView<'a, 'obs, F, PBlocks, PObs>
where
    F: for<'row> Family<Observation<'row> = f64>,
    PBlocks: GamlssBlocks<F>,
    PObs: ObservationView<'obs, Observation = f64> + 'obs,
{
    fn new(
        prediction: &PredictionView<'a, F, PBlocks>,
        obs: &'obs PObs,
    ) -> Result<Self, ModelError> {
        validate_prediction_observations(prediction.nrows(), obs)?;
        Ok(Self {
            prediction: *prediction,
            obs,
        })
    }

    /// Number of prediction rows.
    #[must_use]
    #[inline]
    pub const fn nrows(&self) -> usize {
        self.prediction.nrows()
    }

    /// Returns PIT values for the validated prediction rows.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `parameters` has the wrong length.
    pub fn pit_values(&self, parameters: &[f64]) -> Result<Vec<f64>, ModelError>
    where
        F: HasCdf,
    {
        let mut out = vec![0.0; self.nrows()];
        self.pit_values_into(parameters, &mut out)?;
        Ok(out)
    }

    /// Writes PIT values for the validated prediction rows into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `parameters` or `out` have the wrong length.
    pub fn pit_values_into(&self, parameters: &[f64], out: &mut [f64]) -> Result<(), ModelError>
    where
        F: HasCdf,
    {
        self.diagnostic_values_into(parameters, out, |family, observation, theta| {
            family.cdf(observation, theta)
        })
    }

    /// Returns normalized quantile residuals for the validated prediction rows.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `parameters` has the wrong length.
    pub fn quantile_residuals(&self, parameters: &[f64]) -> Result<Vec<f64>, ModelError>
    where
        F: HasCdf,
    {
        self.pit_values(parameters).map(normalize_pit_values)
    }

    /// Returns CRPS values for the validated prediction rows.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `parameters` has the wrong length.
    pub fn crps_values(&self, parameters: &[f64]) -> Result<Vec<f64>, ModelError>
    where
        F: HasCrps,
    {
        let mut out = vec![0.0; self.nrows()];
        self.crps_values_into(parameters, &mut out)?;
        Ok(out)
    }

    /// Writes CRPS values for the validated prediction rows into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `parameters` or `out` have the wrong length.
    pub fn crps_values_into(&self, parameters: &[f64], out: &mut [f64]) -> Result<(), ModelError>
    where
        F: HasCrps,
    {
        self.diagnostic_values_into(parameters, out, |family, observation, theta| {
            family.crps(observation, theta)
        })
    }

    fn diagnostic_values_into(
        &self,
        parameters: &[f64],
        out: &mut [f64],
        mut evaluate: impl FnMut(&F, f64, &F::Theta) -> f64,
    ) -> Result<(), ModelError> {
        validate_output_len(self.nrows(), out.len())?;
        self.prediction.for_each_theta(parameters, |row, theta| {
            out[row] = evaluate(
                self.prediction.family(),
                self.obs.observation_at(row),
                &theta,
            );
        })
    }
}

/// Diagnostics extension for a validated core [`PredictionView`].
pub trait PredictionDiagnosticsExt<F, PBlocks>
where
    F: for<'row> Family<Observation<'row> = f64>,
    PBlocks: GamlssBlocks<F>,
{
    /// Adds validated prediction observations to this prediction view.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `obs` has the wrong length or invalid
    /// observation/weight values.
    fn with_observations<'obs, PObs>(
        &self,
        obs: &'obs PObs,
    ) -> Result<PredictionDiagnosticsView<'_, 'obs, F, PBlocks, PObs>, ModelError>
    where
        PObs: ObservationView<'obs, Observation = f64> + 'obs;
}

impl<F, PBlocks> PredictionDiagnosticsExt<F, PBlocks> for PredictionView<'_, F, PBlocks>
where
    F: for<'row> Family<Observation<'row> = f64>,
    PBlocks: GamlssBlocks<F>,
{
    fn with_observations<'obs, PObs>(
        &self,
        obs: &'obs PObs,
    ) -> Result<PredictionDiagnosticsView<'_, 'obs, F, PBlocks, PObs>, ModelError>
    where
        PObs: ObservationView<'obs, Observation = f64> + 'obs,
    {
        PredictionDiagnosticsView::new(self, obs)
    }
}
