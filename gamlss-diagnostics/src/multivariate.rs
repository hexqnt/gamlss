use gamlss_core::{
    Gamlss, GamlssBlocks, HasMarginalCdf, HasObservationDimension, HasRosenblattTransform,
    ModelError, ObservationView, PredictionView,
};

use crate::{checked_rosenblatt_len, validate_output_len, validate_prediction_observations};

/// Explicit component-wise marginal PIT diagnostics for multivariate families.
pub trait MarginalCdfDiagnosticsExt<F, Blocks> {
    /// Returns marginal PIT values for one response coordinate.
    ///
    /// `observations` contains that coordinate in model row order. The explicit
    /// scalar slice prevents a joint multivariate CDF from being mistaken for a
    /// scalar PIT.
    fn marginal_pit_values(
        &self,
        parameters: &[f64],
        component: usize,
        observations: &[f64],
    ) -> Result<Vec<f64>, ModelError>;

    /// Writes marginal PIT values into `out`.
    fn marginal_pit_values_into(
        &self,
        parameters: &[f64],
        component: usize,
        observations: &[f64],
        out: &mut [f64],
    ) -> Result<(), ModelError>;
}

impl<F, Blocks, Obs> MarginalCdfDiagnosticsExt<F, Blocks> for Gamlss<F, Blocks, Obs>
where
    F: HasMarginalCdf + HasObservationDimension,
    Blocks: GamlssBlocks<F>,
    for<'obs> Obs: ObservationView<'obs, Observation = F::Observation<'obs>>,
{
    fn marginal_pit_values(
        &self,
        parameters: &[f64],
        component: usize,
        observations: &[f64],
    ) -> Result<Vec<f64>, ModelError> {
        let mut out = vec![0.0; self.nobs()];
        self.marginal_pit_values_into(parameters, component, observations, &mut out)?;
        Ok(out)
    }

    fn marginal_pit_values_into(
        &self,
        parameters: &[f64],
        component: usize,
        observations: &[f64],
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        if component >= self.family().observation_dimension() {
            return Err(ModelError::InvalidParameter {
                parameter: "marginal PIT component",
                expected: "inside the family observation dimension",
            });
        }
        validate_output_len(self.nobs(), observations.len())?;
        validate_output_len(self.nobs(), out.len())?;
        self.for_each_theta(parameters, |row, theta| {
            out[row] = self
                .family()
                .marginal_cdf(component, observations[row], &theta);
        })
    }
}

/// Ordered multivariate PIT diagnostics for a validated core [`PredictionView`].
pub trait PredictionRosenblattDiagnosticsExt<F, PBlocks>
where
    F: HasRosenblattTransform,
    PBlocks: GamlssBlocks<F>,
{
    /// Returns row-major Rosenblatt values with one contiguous coordinate block per row.
    fn rosenblatt_values<'obs, PObs>(
        &self,
        parameters: &[f64],
        obs: &'obs PObs,
    ) -> Result<Vec<f64>, ModelError>
    where
        PObs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs;

    /// Writes row-major Rosenblatt values into `out`.
    fn rosenblatt_values_into<'obs, PObs>(
        &self,
        parameters: &[f64],
        obs: &'obs PObs,
        out: &mut [f64],
    ) -> Result<(), ModelError>
    where
        PObs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs;
}

impl<F, PBlocks> PredictionRosenblattDiagnosticsExt<F, PBlocks> for PredictionView<'_, F, PBlocks>
where
    F: HasRosenblattTransform,
    PBlocks: GamlssBlocks<F>,
{
    fn rosenblatt_values<'obs, PObs>(
        &self,
        parameters: &[f64],
        obs: &'obs PObs,
    ) -> Result<Vec<f64>, ModelError>
    where
        PObs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        let len = checked_rosenblatt_len(self.nrows(), self.family().observation_dimension())?;
        let mut out = vec![0.0; len];
        self.rosenblatt_values_into(parameters, obs, &mut out)?;
        Ok(out)
    }

    fn rosenblatt_values_into<'obs, PObs>(
        &self,
        parameters: &[f64],
        obs: &'obs PObs,
        out: &mut [f64],
    ) -> Result<(), ModelError>
    where
        PObs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        let nrows = self.nrows();
        validate_prediction_observations(nrows, obs)?;
        let dimension = self.family().observation_dimension();
        validate_output_len(checked_rosenblatt_len(nrows, dimension)?, out.len())?;

        let family = self.family();
        let mut transform_error = None;
        self.for_each_theta(parameters, |row, theta| {
            if transform_error.is_some() {
                return;
            }
            let start = row * dimension;
            transform_error = family
                .rosenblatt_into(
                    obs.observation_at(row),
                    &theta,
                    &mut out[start..start + dimension],
                )
                .err();
        })?;
        transform_error.map_or(Ok(()), Err)
    }
}

/// Ordered multivariate PIT diagnostics for fitted GAMLSS models.
pub trait RosenblattDiagnosticsExt<F, Blocks>
where
    F: HasRosenblattTransform,
    Blocks: GamlssBlocks<F>,
{
    /// Returns row-major Rosenblatt values for the training observations.
    ///
    /// Coordinate order is the family observation order and therefore also the
    /// conditioning order of the transform.
    fn rosenblatt_values(&self, parameters: &[f64]) -> Result<Vec<f64>, ModelError>;

    /// Writes row-major Rosenblatt values for the training observations into `out`.
    fn rosenblatt_values_into(&self, parameters: &[f64], out: &mut [f64])
    -> Result<(), ModelError>;

    /// Returns row-major Rosenblatt values for compatible prediction rows.
    fn rosenblatt_values_with_blocks<'obs, PBlocks, PObs>(
        &self,
        parameters: &[f64],
        blocks: &PBlocks,
        obs: &'obs PObs,
    ) -> Result<Vec<f64>, ModelError>
    where
        PBlocks: GamlssBlocks<F>,
        PObs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs;

    /// Writes row-major Rosenblatt values for compatible prediction rows into `out`.
    fn rosenblatt_values_with_blocks_into<'obs, PBlocks, PObs>(
        &self,
        parameters: &[f64],
        blocks: &PBlocks,
        obs: &'obs PObs,
        out: &mut [f64],
    ) -> Result<(), ModelError>
    where
        PBlocks: GamlssBlocks<F>,
        PObs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs;
}

impl<F, Blocks, Obs> RosenblattDiagnosticsExt<F, Blocks> for Gamlss<F, Blocks, Obs>
where
    F: HasRosenblattTransform,
    Blocks: GamlssBlocks<F>,
    for<'obs> Obs: ObservationView<'obs, Observation = F::Observation<'obs>>,
{
    fn rosenblatt_values(&self, parameters: &[f64]) -> Result<Vec<f64>, ModelError> {
        self.prediction_view(self.blocks())?
            .rosenblatt_values(parameters, self.obs())
    }

    fn rosenblatt_values_into(
        &self,
        parameters: &[f64],
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        self.prediction_view(self.blocks())?
            .rosenblatt_values_into(parameters, self.obs(), out)
    }

    fn rosenblatt_values_with_blocks<'obs, PBlocks, PObs>(
        &self,
        parameters: &[f64],
        blocks: &PBlocks,
        obs: &'obs PObs,
    ) -> Result<Vec<f64>, ModelError>
    where
        PBlocks: GamlssBlocks<F>,
        PObs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        self.prediction_view(blocks)?
            .rosenblatt_values(parameters, obs)
    }

    fn rosenblatt_values_with_blocks_into<'obs, PBlocks, PObs>(
        &self,
        parameters: &[f64],
        blocks: &PBlocks,
        obs: &'obs PObs,
        out: &mut [f64],
    ) -> Result<(), ModelError>
    where
        PBlocks: GamlssBlocks<F>,
        PObs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        self.prediction_view(blocks)?
            .rosenblatt_values_into(parameters, obs, out)
    }
}
