use std::ops::Range;

use crate::{
    BlockObjective, Family, GlobalPenalty, LowerTriangularParameterBlock, ModelError, Objective,
    ParameterBlock, ParameterName, ParameterParts, Penalty, PredictorBlock,
    SimplexLogitParameterBlock, StrictLowerTriangularParameterBlock, VectorParameterBlock,
    family::{
        InitialEtaFromObservations, LocationCholeskyScalarSpec, LocationCholeskySpec,
        LocationScalePartialCorrSpec, MeanPrecisionSimplexSpec, RepeatedScalarParamSpec,
        ScalarParamSpec,
    },
};

pub use layout::{
    ParameterAxis, ParameterCoefficients, ParameterDescriptor, ParameterLayout, ParameterPath,
    ParameterSlice, TrainingDiagnostics, UnpackedParameters,
};
pub use observation::{FiniteScalarObservations, ObservationView};
pub use workspace::{GradientWorkspace, ModelWorkspace};

mod layout;
mod observation;
mod workspace;

/// Scaling convention for the likelihood part of a compiled objective.
///
/// Local and global penalties are not scaled. This keeps penalty weights on a
/// stable scale when switching between summed and mean likelihood objectives.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ObjectiveScale {
    /// Use the weighted likelihood sum.
    #[default]
    Sum,
    /// Use the weighted likelihood mean.
    ///
    /// For weighted observations the denominator is the sum of observation
    /// weights. If all weights are zero, the likelihood contribution is left
    /// unscaled at zero.
    Mean,
}

impl ObjectiveScale {
    #[inline]
    fn likelihood_multiplier(self, weight_sum: f64) -> f64 {
        match self {
            Self::Mean if weight_sum > 0.0 => 1.0 / weight_sum,
            Self::Sum | Self::Mean => 1.0,
        }
    }
}

/// Compiled typed GAMLSS model.
///
/// `F` specifies the response distribution, and `Blocks` provides one
/// predictor block for each family parameter.
///
/// The model owns the family and parameter blocks, and stores an observation
/// view supplied by the caller. This keeps `gamlss-core` independent of the
/// caller's storage backend while static dispatch preserves zero-cost hot-path
/// evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct Gamlss<F, Blocks, Obs> {
    family: F,
    blocks: Blocks,
    obs: Obs,
    objective_scale: ObjectiveScale,
    weight_sum: f64,
}

impl<F, Blocks, Obs> Gamlss<F, Blocks, Obs> {
    /// Response distribution family.
    #[must_use]
    #[inline]
    pub const fn family(&self) -> &F {
        &self.family
    }

    /// Typed parameter blocks.
    #[must_use]
    #[inline]
    pub const fn blocks(&self) -> &Blocks {
        &self.blocks
    }

    /// Observation view used for training objective evaluation.
    #[must_use]
    #[inline]
    pub const fn obs(&self) -> &Obs {
        &self.obs
    }

    /// Consumes the model and returns its family, blocks and observation view.
    #[must_use]
    #[inline]
    pub fn into_parts(self) -> (F, Blocks, Obs) {
        (self.family, self.blocks, self.obs)
    }

    /// Wraps the model with unchecked penalties evaluated on the full beta vector.
    ///
    /// Use [`Self::try_with_global_penalties`] when penalties are assembled
    /// from dynamic indices or ranges.
    #[must_use]
    #[inline]
    pub const fn with_global_penalties<GP>(self, penalties: GP) -> WithGlobalPenalties<Self, GP> {
        WithGlobalPenalties {
            objective: self,
            penalties,
        }
    }

    /// Wraps the model with dimension-validated full-vector penalties.
    ///
    /// This is the checked counterpart of [`Self::with_global_penalties`].
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::PenaltyIndexOutOfBounds`] or
    /// [`ModelError::PenaltyRangeOutOfBounds`] when a penalty implementation
    /// reports references outside the model parameter vector.
    #[inline]
    pub fn try_with_global_penalties<GP>(
        self,
        penalties: GP,
    ) -> Result<WithGlobalPenalties<Self, GP>, ModelError>
    where
        F: Family,
        Blocks: GamlssBlocks<F>,
        for<'row> Obs: ObservationView<'row, Observation = F::Observation<'row>>,
        GP: GlobalPenalty,
    {
        let dim = self.nparams();
        with_validated_global_penalties(self, penalties, dim)
    }
}

impl<F, Blocks, Obs> Gamlss<F, Blocks, Obs>
where
    F: Family,
    Blocks: GamlssBlocks<F>,
    for<'row> Obs: ObservationView<'row, Observation = F::Observation<'row>>,
{
    /// Creates a model after validating the observation view and blocks.
    ///
    /// This is the extension point for custom storage backends. The observation
    /// view is stored by value, so callers can pass lightweight borrowed views,
    /// owned adapters, or newtypes around external dataframe/columnar storage.
    pub fn try_new_with_observations(
        family: F,
        blocks: Blocks,
        obs: Obs,
    ) -> Result<Self, ModelError> {
        if obs.is_empty() {
            return Err(ModelError::EmptyResponse);
        }

        let nobs = obs.len();
        obs.validate()?;
        let weight_sum = observation_weight_sum(&obs);
        blocks.validate(nobs)?;
        blocks.try_len()?;
        Ok(Self {
            family,
            blocks,
            obs,
            objective_scale: ObjectiveScale::Sum,
            weight_sum,
        })
    }

    /// Number of observations.
    #[must_use]
    #[inline]
    pub fn nobs(&self) -> usize {
        self.obs.len()
    }

    /// Sum of observation weights used as the denominator for mean likelihood objectives.
    ///
    /// For unweighted observation views this is equal to [`Gamlss::nobs`] as `f64`.
    #[must_use]
    #[inline]
    pub const fn weight_sum(&self) -> f64 {
        self.weight_sum
    }

    /// Effective number of observations represented by the observation weights.
    ///
    /// This is currently the same value as [`Gamlss::weight_sum`].
    #[must_use]
    #[inline]
    pub const fn effective_nobs(&self) -> f64 {
        self.weight_sum()
    }

    /// Number of coefficients in the common beta vector.
    pub fn nparams(&self) -> usize {
        self.blocks.len()
    }

    /// Returns the likelihood scaling convention used by this objective.
    pub const fn objective_scale(&self) -> ObjectiveScale {
        self.objective_scale
    }

    /// Returns `self` with a different likelihood scaling convention.
    #[must_use]
    pub const fn with_objective_scale(mut self, objective_scale: ObjectiveScale) -> Self {
        self.objective_scale = objective_scale;
        self
    }

    /// Updates the likelihood scaling convention in place.
    pub const fn set_objective_scale(&mut self, objective_scale: ObjectiveScale) {
        self.objective_scale = objective_scale;
    }

    fn likelihood_multiplier(&self) -> f64 {
        self.objective_scale.likelihood_multiplier(self.weight_sum)
    }

    /// Zero-valued initial optimizer parameter vector of the right length.
    pub fn initial_zeros(&self) -> Vec<f64> {
        vec![0.0; self.nparams()]
    }

    /// Initial optimizer parameter vector for external optimizers.
    ///
    /// The returned vector is the flat predictor-coefficient vector, commonly
    /// denoted `beta`, laid out according to this model's parameter blocks. It
    /// is not the natural-scale distribution parameter `theta`.
    ///
    /// Currently this falls back to zero components for unsupported predictor
    /// blocks or non-finite family starts. Future projection-based
    /// initializers may return recoverable errors.
    pub fn initial_parameters(&self) -> Result<Vec<f64>, ModelError> {
        self.blocks.try_initial_parameters(&self.family, &self.obs)
    }

    /// Creates reusable objective buffers sized for this model.
    pub fn gradient_workspace(&self) -> ModelWorkspace<F>
    where
        F: Family,
    {
        ModelWorkspace::new(&self.family, self.obs.len(), |nobs| {
            self.blocks.gradient_workspace(nobs)
        })
    }

    /// Wraps the model as an objective with reusable gradient buffers.
    pub fn into_workspace_objective(self) -> WorkspaceGamlss<F, Blocks, Obs> {
        let workspace = self.gradient_workspace();
        WorkspaceGamlss {
            model: self,
            workspace,
        }
    }

    /// Coefficient block ranges within beta.
    pub fn block_ranges(&self) -> Vec<Range<usize>> {
        self.blocks.block_ranges()
    }

    /// Visits coefficient ranges for each parameter block in model order without allocating.
    pub fn visit_block_ranges<V>(&self, visit: V)
    where
        V: FnMut(usize, Range<usize>),
    {
        self.blocks.visit_block_ranges(visit);
    }

    /// Layout of named parameter blocks inside the flat optimizer-parameter vector.
    pub fn parameter_layout(&self) -> ParameterLayout {
        self.blocks.parameter_layout()
    }

    /// Structured coefficient descriptors inside the flat optimizer-parameter vector.
    ///
    /// Ordinary scalar-parameter models return one whole-parameter descriptor
    /// per block. Structured multivariate block implementations may return
    /// component- or matrix-entry-level descriptors, such as `mu[i]` or
    /// `cholesky[row, col]`.
    pub fn parameter_descriptors(&self) -> Vec<ParameterDescriptor> {
        self.blocks.parameter_descriptors()
    }

    /// Visits named parameter slices in model order without allocating.
    pub fn visit_parameter_slices<V>(&self, visit: V)
    where
        V: FnMut(usize, &'static str, Range<usize>),
    {
        self.blocks.visit_parameter_slices(visit);
    }

    /// Visits structured coefficient descriptors in model order without allocating.
    pub fn visit_parameter_descriptors<V>(&self, visit: V)
    where
        V: FnMut(usize, ParameterDescriptor),
    {
        self.blocks.visit_parameter_descriptors(visit);
    }

    /// Creates a [`BlockObjective`] projected to the coefficients of parameter `P`.
    ///
    /// This is the zero-cost building block for staged/block-wise fitting:
    /// optimise one distribution parameter (e.g. `Mu`) while keeping the
    /// remaining coefficients fixed at the values in `full_beta`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::UnknownParameter`] if the model blocks do not
    /// contain a parameter named `P::NAME`. When blocks are constructed through
    /// the typed [`ParameterBlock`] API this cannot happen in practice — the
    /// compiler guarantees that `ParameterBlock<Mu, …>` registers itself as
    /// `"mu"`.
    pub fn block_objective_for<P>(
        &mut self,
        full_beta: Vec<f64>,
    ) -> Result<BlockObjective<'_, Self>, ModelError>
    where
        P: ParameterName,
    {
        validate_len("parameters", full_beta.len(), self.nparams())?;
        let range = self
            .blocks
            .parameter_slice_of::<P>()
            .ok_or(ModelError::UnknownParameter { name: P::NAME })?;
        BlockObjective::try_new(
            self,
            full_beta,
            ParameterSlice {
                name: P::NAME,
                range,
            },
        )
    }

    /// Unpacks a flat optimizer-parameter vector into named coefficient blocks.
    pub fn unpack_parameters(&self, parameters: &[f64]) -> Result<UnpackedParameters, ModelError> {
        validate_len("parameters", parameters.len(), self.nparams())?;

        let blocks = self
            .parameter_layout()
            .slices()
            .iter()
            .map(|slice| ParameterCoefficients {
                name: slice.name,
                coefficients: parameters[slice.range.clone()].to_vec(),
            })
            .collect();

        Ok(UnpackedParameters { blocks })
    }

    /// Computes training diagnostics for a candidate optimizer-parameter vector.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::BetaLength`] if `parameters` does not match the
    /// model parameter dimension.
    pub fn training_diagnostics(
        &self,
        parameters: &[f64],
    ) -> Result<TrainingDiagnostics, ModelError> {
        let mut grad = vec![0.0; self.nparams()];
        self.training_diagnostics_into(parameters, &mut grad)
    }

    /// Computes training diagnostics using a caller-provided gradient buffer.
    ///
    /// The buffer is overwritten with the objective gradient and then reused to
    /// compute the reported gradient norm. This avoids allocating a temporary
    /// gradient vector when diagnostics are evaluated repeatedly.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::BetaLength`] if `parameters` does not match the
    /// model parameter dimension, or [`ModelError::GradientLength`] if `grad`
    /// has the wrong length.
    pub fn training_diagnostics_into(
        &self,
        parameters: &[f64],
        grad: &mut [f64],
    ) -> Result<TrainingDiagnostics, ModelError> {
        let mut workspace = self.gradient_workspace();
        self.training_diagnostics_into_workspace(parameters, grad, &mut workspace)
    }

    /// Computes training diagnostics using caller-provided gradient and workspace buffers.
    ///
    /// The reusable [`GradientWorkspace`] is used for internal per-parameter
    /// buffers, while `grad` receives the full objective gradient and is reused
    /// to compute the reported gradient norm. Prefer
    /// [`Gamlss::into_workspace_objective`] in training loops:
    ///
    /// ```ignore
    /// let mut objective = model.into_workspace_objective();
    /// objective.training_diagnostics_into(parameters, &mut grad)?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::BetaLength`] if `parameters` does not match the
    /// model parameter dimension, or [`ModelError::GradientLength`] if `grad`
    /// has the wrong length.
    pub fn training_diagnostics_into_workspace(
        &self,
        parameters: &[f64],
        grad: &mut [f64],
        workspace: &mut ModelWorkspace<F>,
    ) -> Result<TrainingDiagnostics, ModelError> {
        validate_beta_and_gradient_len(self.nparams(), parameters, grad)?;

        let likelihood_multiplier = self.likelihood_multiplier();
        let train_nll =
            likelihood_multiplier * self.blocks.train_nll(&self.family, &self.obs, parameters);
        let penalty = self.blocks.penalty_value(parameters);
        self.try_gradient_into_workspace(parameters, grad, workspace)?;
        Ok(training_diagnostics_from_gradient(train_nll, penalty, grad))
    }

    /// Predicts link-scale distribution predictors for one training row.
    pub fn predict_eta_row(&self, parameters: &[f64], row: usize) -> Result<F::Eta, ModelError>
    where
        F: Family,
    {
        validate_len("parameters", parameters.len(), self.nparams())?;
        validate_row(row, self.nobs())?;
        Ok(self.blocks.eta_row(parameters, row))
    }

    /// Predicts natural-scale distribution parameters for one training row.
    pub fn predict_theta_row(&self, parameters: &[f64], row: usize) -> Result<F::Theta, ModelError>
    where
        F: Family,
    {
        let eta = self.predict_eta_row(parameters, row)?;
        let mut workspace = self.family.workspace();
        Ok(self.family.theta(&eta, &mut workspace))
    }

    /// Predicts link-scale distribution predictors for all training rows.
    pub fn predict_eta(&self, parameters: &[f64]) -> Result<Vec<F::Eta>, ModelError>
    where
        F: Family,
    {
        validate_len("parameters", parameters.len(), self.nparams())?;
        Ok((0..self.nobs())
            .map(|row| self.blocks.eta_row(parameters, row))
            .collect())
    }

    /// Predicts natural-scale distribution parameters for all training rows.
    pub fn predict_theta(&self, parameters: &[f64]) -> Result<Vec<F::Theta>, ModelError>
    where
        F: Family,
    {
        validate_len("parameters", parameters.len(), self.nparams())?;
        let mut workspace = self.family.workspace();
        Ok((0..self.nobs())
            .map(|row| {
                let eta = self.blocks.eta_row(parameters, row);
                self.family.theta(&eta, &mut workspace)
            })
            .collect())
    }

    /// Predicts natural-scale distribution parameters into an existing slice.
    ///
    /// `out` must have one slot per training row.
    pub fn predict_theta_into(
        &self,
        parameters: &[f64],
        out: &mut [F::Theta],
    ) -> Result<(), ModelError>
    where
        F: Family,
    {
        validate_len("parameters", parameters.len(), self.nparams())?;
        validate_output_len(self.nobs(), out.len())?;
        let mut workspace = self.family.workspace();
        for (row, out) in out.iter_mut().enumerate() {
            let eta = self.blocks.eta_row(parameters, row);
            *out = self.family.theta(&eta, &mut workspace);
        }
        Ok(())
    }

    /// Streams natural-scale distribution parameters for each training row.
    pub fn for_each_theta(
        &self,
        parameters: &[f64],
        mut visit: impl FnMut(usize, F::Theta),
    ) -> Result<(), ModelError>
    where
        F: Family,
    {
        validate_len("parameters", parameters.len(), self.nparams())?;
        let mut workspace = self.family.workspace();
        for row in 0..self.nobs() {
            let eta = self.blocks.eta_row(parameters, row);
            visit(row, self.family.theta(&eta, &mut workspace));
        }
        Ok(())
    }

    /// Predicts link-scale distribution predictors for one row from compatible prediction blocks.
    ///
    /// # Errors
    ///
    /// Returns an error if `parameters` has the wrong length, if `blocks` do not
    /// match this model's parameter layout, or if `row` is out of bounds for
    /// the supplied prediction blocks.
    pub fn predict_eta_row_with_blocks<PBlocks>(
        &self,
        parameters: &[f64],
        blocks: &PBlocks,
        row: usize,
    ) -> Result<F::Eta, ModelError>
    where
        F: Family,
        PBlocks: GamlssBlocks<F>,
    {
        self.prediction_view(blocks)?
            .predict_eta_row(parameters, row)
    }

    /// Predicts natural-scale distribution parameters for one row from compatible prediction blocks.
    ///
    /// # Errors
    ///
    /// Returns an error if `parameters` has the wrong length, if `blocks` do not
    /// match this model's parameter layout, or if `row` is out of bounds for
    /// the supplied prediction blocks.
    pub fn predict_theta_row_with_blocks<PBlocks>(
        &self,
        parameters: &[f64],
        blocks: &PBlocks,
        row: usize,
    ) -> Result<F::Theta, ModelError>
    where
        F: Family,
        PBlocks: GamlssBlocks<F>,
    {
        self.prediction_view(blocks)?
            .predict_theta_row(parameters, row)
    }

    /// Predicts link-scale distribution predictors for all rows in compatible prediction blocks.
    ///
    /// # Errors
    ///
    /// Returns an error if `parameters` has the wrong length or if `blocks` do not
    /// match this model's parameter layout.
    pub fn predict_eta_with_blocks<PBlocks>(
        &self,
        parameters: &[f64],
        blocks: &PBlocks,
    ) -> Result<Vec<F::Eta>, ModelError>
    where
        F: Family,
        PBlocks: GamlssBlocks<F>,
    {
        self.prediction_view(blocks)?.predict_eta(parameters)
    }

    /// Predicts natural-scale distribution parameters for all rows in compatible prediction blocks.
    ///
    /// # Errors
    ///
    /// Returns an error if `parameters` has the wrong length or if `blocks` do not
    /// match this model's parameter layout.
    pub fn predict_theta_with_blocks<PBlocks>(
        &self,
        parameters: &[f64],
        blocks: &PBlocks,
    ) -> Result<Vec<F::Theta>, ModelError>
    where
        F: Family,
        PBlocks: GamlssBlocks<F>,
    {
        self.prediction_view(blocks)?.predict_theta(parameters)
    }

    /// Predicts natural-scale distribution parameters from prediction blocks into `out`.
    ///
    /// `out` must have one slot per row in `blocks`.
    pub fn predict_theta_with_blocks_into<PBlocks>(
        &self,
        parameters: &[f64],
        blocks: &PBlocks,
        out: &mut [F::Theta],
    ) -> Result<(), ModelError>
    where
        F: Family,
        PBlocks: GamlssBlocks<F>,
    {
        self.prediction_view(blocks)?
            .predict_theta_into(parameters, out)
    }

    /// Streams natural-scale parameters for each row in compatible prediction blocks.
    pub fn for_each_theta_with_blocks<PBlocks>(
        &self,
        parameters: &[f64],
        blocks: &PBlocks,
        visit: impl FnMut(usize, F::Theta),
    ) -> Result<(), ModelError>
    where
        F: Family,
        PBlocks: GamlssBlocks<F>,
    {
        self.prediction_view(blocks)?
            .for_each_theta(parameters, visit)
    }

    /// Creates a validated reusable prediction view over compatible blocks.
    ///
    /// The returned view caches the prediction row count and validated
    /// parameter layout compatibility, so repeated `predict_*` calls only check
    /// parameter and output slice lengths.
    pub fn prediction_view<'a, PBlocks>(
        &'a self,
        blocks: &'a PBlocks,
    ) -> Result<PredictionView<'a, F, PBlocks>, ModelError>
    where
        F: Family,
        PBlocks: GamlssBlocks<F>,
    {
        PredictionView::new(self, blocks)
    }

    /// Validates beta length and computes the objective.
    #[allow(clippy::suboptimal_flops)]
    pub fn try_value(&self, beta: &[f64]) -> Result<f64, ModelError> {
        validate_len("parameters", beta.len(), self.nparams())?;

        let train_nll = self.blocks.train_nll(&self.family, &self.obs, beta);
        let penalty = self.blocks.penalty_value(beta);
        Ok(self.likelihood_multiplier() * train_nll + penalty)
    }

    /// Validates beta/grad sizes and writes the gradient.
    pub fn try_gradient_into(&self, beta: &[f64], grad: &mut [f64]) -> Result<(), ModelError> {
        let mut workspace = self.gradient_workspace();
        self.try_gradient_into_workspace(beta, grad, &mut workspace)
    }

    /// Validates beta/grad sizes and writes the gradient, reusing a workspace.
    pub fn try_gradient_into_workspace(
        &self,
        beta: &[f64],
        grad: &mut [f64],
        workspace: &mut ModelWorkspace<F>,
    ) -> Result<(), ModelError> {
        validate_beta_and_gradient_len(self.nparams(), beta, grad)?;

        grad.fill(0.0);
        let (family_workspace, gradient_workspace) = workspace.parts_mut();
        let value = self.blocks.value_gradient_into_workspace(
            &self.family,
            &self.obs,
            beta,
            grad,
            family_workspace,
            gradient_workspace,
        );
        self.scale_value_gradient(beta, grad, workspace.gradient_mut(), value);
        Ok(())
    }

    /// Validates beta/grad sizes and computes value + gradient in one pass.
    pub fn try_value_gradient_into(
        &self,
        beta: &[f64],
        grad: &mut [f64],
    ) -> Result<f64, ModelError> {
        let mut workspace = self.gradient_workspace();
        self.try_value_gradient_into_workspace(beta, grad, &mut workspace)
    }

    /// Validates beta/grad sizes and computes fused value + gradient with a
    /// workspace.
    pub fn try_value_gradient_into_workspace(
        &self,
        beta: &[f64],
        grad: &mut [f64],
        workspace: &mut ModelWorkspace<F>,
    ) -> Result<f64, ModelError> {
        validate_beta_and_gradient_len(self.nparams(), beta, grad)?;

        grad.fill(0.0);
        let (family_workspace, gradient_workspace) = workspace.parts_mut();
        let value = self.blocks.value_gradient_into_workspace(
            &self.family,
            &self.obs,
            beta,
            grad,
            family_workspace,
            gradient_workspace,
        );
        Ok(self.scale_value_gradient(beta, grad, workspace.gradient_mut(), value))
    }

    #[allow(clippy::float_cmp)]
    fn scale_value_gradient(
        &self,
        beta: &[f64],
        grad: &mut [f64],
        workspace: &mut GradientWorkspace,
        unscaled_value: f64,
    ) -> f64 {
        let likelihood_multiplier = self.likelihood_multiplier();
        if likelihood_multiplier == 1.0 {
            return unscaled_value;
        }

        let penalty = self.blocks.penalty_value(beta);
        let penalty_grad = workspace.penalty_gradient_mut(self.nparams());
        self.blocks.add_penalty_gradient(beta, penalty_grad);

        for (grad_value, penalty_grad_value) in grad.iter_mut().zip(penalty_grad.iter().copied()) {
            *grad_value = (*grad_value - penalty_grad_value)
                .mul_add(likelihood_multiplier, penalty_grad_value);
        }

        (unscaled_value - penalty).mul_add(likelihood_multiplier, penalty)
    }
}

impl<'a, F, Blocks> Gamlss<F, Blocks, &'a [f64]>
where
    F: for<'obs> Family<Observation<'obs> = f64>,
    Blocks: GamlssBlocks<F>,
{
    /// Creates an unweighted model after validating the response and blocks.
    pub fn try_new(family: F, blocks: Blocks, y: &'a [f64]) -> Result<Self, ModelError> {
        Self::try_new_with_observations(family, blocks, y)
    }
}

impl<'a, F, Blocks> Gamlss<F, Blocks, FiniteScalarObservations<'a>>
where
    F: for<'obs> Family<Observation<'obs> = f64>,
    Blocks: GamlssBlocks<F>,
{
    /// Creates an unweighted model that rejects non-finite scalar responses.
    ///
    /// The ordinary [`Gamlss::try_new`] constructor intentionally leaves scalar
    /// response domain checks to the family and to weights-aware workflows. This
    /// strict constructor rejects `NaN`, `inf` and `-inf` before model
    /// construction.
    pub fn try_new_strict(family: F, blocks: Blocks, y: &'a [f64]) -> Result<Self, ModelError> {
        Self::try_new_with_observations(family, blocks, FiniteScalarObservations::new(y)?)
    }
}

impl<'a, F, Blocks> Gamlss<F, Blocks, (&'a [f64], &'a [f64])>
where
    F: for<'obs> Family<Observation<'obs> = f64>,
    Blocks: GamlssBlocks<F>,
{
    /// Creates a model with observation weights after validating the response,
    /// weights and blocks.
    ///
    /// Weights must have the same length as `y`; each weight must be finite and
    /// non-negative. Zero weights are accepted and exclude the corresponding
    /// observation from likelihood and gradient contributions.
    pub fn try_new_weighted(
        family: F,
        blocks: Blocks,
        y: &'a [f64],
        weights: &'a [f64],
    ) -> Result<Self, ModelError> {
        Self::try_new_with_observations(family, blocks, (y, weights))
    }
}

impl<F, Blocks, Obs> Objective for Gamlss<F, Blocks, Obs>
where
    F: Family,
    Blocks: GamlssBlocks<F>,
    for<'row> Obs: ObservationView<'row, Observation = F::Observation<'row>>,
{
    type Error = ModelError;

    fn dim(&self) -> usize {
        self.nparams()
    }

    fn value(&mut self, parameters: &[f64]) -> Result<f64, Self::Error> {
        self.try_value(parameters)
    }

    fn gradient(&mut self, parameters: &[f64], grad: &mut [f64]) -> Result<(), Self::Error> {
        self.try_value_gradient_into(parameters, grad).map(|_| ())
    }

    fn value_gradient(&mut self, parameters: &[f64], grad: &mut [f64]) -> Result<f64, Self::Error> {
        self.try_value_gradient_into(parameters, grad)
    }
}

/// Validated prediction view over compatible parameter blocks.
///
/// The view borrows the fitted family and prediction blocks after validating
/// that the prediction block layout matches the fitted model. Reusing it avoids
/// repeating prediction-block validation across batch inference calls.
#[derive(Debug, PartialEq, Eq)]
pub struct PredictionView<'a, F, PBlocks> {
    family: &'a F,
    blocks: &'a PBlocks,
    nrows: usize,
    nparams: usize,
}

impl<'a, F, PBlocks> PredictionView<'a, F, PBlocks>
where
    F: Family,
    PBlocks: GamlssBlocks<F>,
{
    fn new<Blocks, Obs>(
        model: &'a Gamlss<F, Blocks, Obs>,
        blocks: &'a PBlocks,
    ) -> Result<Self, ModelError>
    where
        Blocks: GamlssBlocks<F>,
    {
        validate_prediction_blocks(&model.blocks, blocks)?;
        Ok(Self {
            family: &model.family,
            blocks,
            nrows: blocks.nrows(),
            nparams: model.blocks.len(),
        })
    }

    /// Response distribution family.
    #[must_use]
    #[inline]
    pub const fn family(&self) -> &'a F {
        self.family
    }

    /// Typed prediction parameter blocks.
    #[must_use]
    #[inline]
    pub const fn blocks(&self) -> &'a PBlocks {
        self.blocks
    }

    /// Number of prediction rows.
    #[must_use]
    #[inline]
    pub const fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of coefficients expected in the flat parameter vector.
    #[must_use]
    #[inline]
    pub const fn nparams(&self) -> usize {
        self.nparams
    }

    /// Predicts link-scale distribution predictors for one prediction row.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `parameters` has the wrong length or `row` is
    /// out of bounds for the validated prediction blocks.
    pub fn predict_eta_row(&self, parameters: &[f64], row: usize) -> Result<F::Eta, ModelError> {
        validate_len("parameters", parameters.len(), self.nparams)?;
        validate_row(row, self.nrows)?;
        Ok(self.blocks.eta_row(parameters, row))
    }

    /// Predicts natural-scale distribution parameters for one prediction row.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `parameters` has the wrong length or `row` is
    /// out of bounds for the validated prediction blocks.
    pub fn predict_theta_row(
        &self,
        parameters: &[f64],
        row: usize,
    ) -> Result<F::Theta, ModelError> {
        let eta = self.predict_eta_row(parameters, row)?;
        let mut workspace = self.family.workspace();
        Ok(self.family.theta(&eta, &mut workspace))
    }

    /// Predicts link-scale distribution predictors for all prediction rows.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `parameters` has the wrong length.
    pub fn predict_eta(&self, parameters: &[f64]) -> Result<Vec<F::Eta>, ModelError> {
        validate_len("parameters", parameters.len(), self.nparams)?;
        Ok((0..self.nrows)
            .map(|row| self.blocks.eta_row(parameters, row))
            .collect())
    }

    /// Predicts natural-scale distribution parameters for all prediction rows.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `parameters` has the wrong length.
    pub fn predict_theta(&self, parameters: &[f64]) -> Result<Vec<F::Theta>, ModelError> {
        validate_len("parameters", parameters.len(), self.nparams)?;
        let mut workspace = self.family.workspace();
        Ok((0..self.nrows)
            .map(|row| {
                let eta = self.blocks.eta_row(parameters, row);
                self.family.theta(&eta, &mut workspace)
            })
            .collect())
    }

    /// Predicts natural-scale distribution parameters into an existing slice.
    ///
    /// `out` must have one slot per prediction row.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `parameters` or `out` have the wrong length.
    pub fn predict_theta_into(
        &self,
        parameters: &[f64],
        out: &mut [F::Theta],
    ) -> Result<(), ModelError> {
        validate_len("parameters", parameters.len(), self.nparams)?;
        validate_output_len(self.nrows, out.len())?;
        let mut workspace = self.family.workspace();
        for (row, out) in out.iter_mut().enumerate() {
            let eta = self.blocks.eta_row(parameters, row);
            *out = self.family.theta(&eta, &mut workspace);
        }
        Ok(())
    }

    /// Streams natural-scale parameters for each prediction row.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `parameters` has the wrong length.
    pub fn for_each_theta(
        &self,
        parameters: &[f64],
        mut visit: impl FnMut(usize, F::Theta),
    ) -> Result<(), ModelError> {
        validate_len("parameters", parameters.len(), self.nparams)?;
        let mut workspace = self.family.workspace();
        for row in 0..self.nrows {
            let eta = self.blocks.eta_row(parameters, row);
            visit(row, self.family.theta(&eta, &mut workspace));
        }
        Ok(())
    }
}

impl<F, PBlocks> Clone for PredictionView<'_, F, PBlocks> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, PBlocks> Copy for PredictionView<'_, F, PBlocks> {}

/// GAMLSS objective with reusable gradient buffers.
///
/// This wrapper is intended for optimizers that call `gradient` repeatedly.
/// It owns the compiled model and keeps a [`ModelWorkspace`] between calls,
/// avoiding per-call allocation of family and gradient scratch buffers.
pub struct WorkspaceGamlss<F: Family, Blocks, Obs> {
    model: Gamlss<F, Blocks, Obs>,
    workspace: ModelWorkspace<F>,
}

impl<F, Blocks, Obs> WorkspaceGamlss<F, Blocks, Obs>
where
    F: Family,
    Blocks: GamlssBlocks<F>,
    for<'row> Obs: ObservationView<'row, Observation = F::Observation<'row>>,
{
    /// Creates a workspace-backed objective from a compiled model.
    #[must_use]
    #[inline]
    pub fn new(model: Gamlss<F, Blocks, Obs>) -> Self {
        model.into_workspace_objective()
    }

    /// Returns the wrapped model.
    #[must_use]
    #[inline]
    pub const fn model(&self) -> &Gamlss<F, Blocks, Obs> {
        &self.model
    }

    /// Returns the wrapped model mutably.
    #[inline]
    pub const fn model_mut(&mut self) -> &mut Gamlss<F, Blocks, Obs> {
        &mut self.model
    }

    /// Returns the reusable model workspace.
    #[must_use]
    #[inline]
    pub const fn workspace(&self) -> &ModelWorkspace<F> {
        &self.workspace
    }

    /// Returns the reusable model workspace mutably.
    #[inline]
    pub const fn workspace_mut(&mut self) -> &mut ModelWorkspace<F> {
        &mut self.workspace
    }

    /// Consumes the objective and returns the wrapped model and workspace.
    #[must_use]
    #[inline]
    pub fn into_parts(self) -> (Gamlss<F, Blocks, Obs>, ModelWorkspace<F>) {
        (self.model, self.workspace)
    }

    /// Consumes the workspace-backed objective and returns the wrapped model.
    #[must_use]
    #[inline]
    pub fn into_model(self) -> Gamlss<F, Blocks, Obs> {
        self.model
    }

    /// Returns the likelihood scaling convention used by this objective.
    pub const fn objective_scale(&self) -> ObjectiveScale {
        self.model.objective_scale()
    }

    /// Returns `self` with a different likelihood scaling convention.
    #[must_use]
    pub const fn with_objective_scale(mut self, objective_scale: ObjectiveScale) -> Self {
        self.model.set_objective_scale(objective_scale);
        self
    }

    /// Updates the likelihood scaling convention in place.
    pub const fn set_objective_scale(&mut self, objective_scale: ObjectiveScale) {
        self.model.set_objective_scale(objective_scale);
    }

    /// Creates a [`BlockObjective`] projected to the coefficients of parameter `P`.
    ///
    /// Delegates to the inner model's [`Gamlss::block_objective_for`] through
    /// [`model_mut`](Self::model_mut), so the returned objective borrows the
    /// workspace-backed model and reuses its gradient buffers.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::UnknownParameter`] if the model blocks do not
    /// contain a parameter named `P::NAME`.
    pub fn block_objective_for<P>(
        &mut self,
        full_beta: Vec<f64>,
    ) -> Result<BlockObjective<'_, Self>, ModelError>
    where
        P: ParameterName,
    {
        validate_len("parameters", full_beta.len(), self.dim())?;
        let range = self
            .model
            .blocks
            .parameter_slice_of::<P>()
            .ok_or(ModelError::UnknownParameter { name: P::NAME })?;
        BlockObjective::try_new(
            self,
            full_beta,
            ParameterSlice {
                name: P::NAME,
                range,
            },
        )
    }

    /// Computes training diagnostics using caller-provided objective-gradient storage.
    ///
    /// The reusable [`GradientWorkspace`] is used for internal per-parameter
    /// buffers, while `grad` receives the full objective gradient and is reused
    /// to compute the reported gradient norm.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::BetaLength`] if `parameters` does not match the
    /// model parameter dimension, or [`ModelError::GradientLength`] if `grad`
    /// has the wrong length.
    pub fn training_diagnostics_into(
        &mut self,
        parameters: &[f64],
        grad: &mut [f64],
    ) -> Result<TrainingDiagnostics, ModelError> {
        self.model
            .training_diagnostics_into_workspace(parameters, grad, &mut self.workspace)
    }

    /// Wraps the workspace-backed objective with unchecked penalties evaluated on the full beta vector.
    ///
    /// Use [`Self::try_with_global_penalties`] when penalties are assembled
    /// from dynamic indices or ranges.
    #[must_use]
    #[inline]
    pub const fn with_global_penalties<GP>(self, penalties: GP) -> WithGlobalPenalties<Self, GP> {
        WithGlobalPenalties {
            objective: self,
            penalties,
        }
    }

    /// Wraps the workspace-backed objective with dimension-validated full-vector penalties.
    ///
    /// This is the checked counterpart of [`Self::with_global_penalties`].
    ///
    /// # Errors
    ///
    /// Returns the invariant or dimension validation error reported by
    /// [`GlobalPenalty::validate`].
    #[inline]
    pub fn try_with_global_penalties<GP>(
        self,
        penalties: GP,
    ) -> Result<WithGlobalPenalties<Self, GP>, ModelError>
    where
        GP: GlobalPenalty,
    {
        let dim = self.model.nparams();
        with_validated_global_penalties(self, penalties, dim)
    }
}

impl<F, Blocks, Obs> std::fmt::Debug for WorkspaceGamlss<F, Blocks, Obs>
where
    F: Family,
    Gamlss<F, Blocks, Obs>: std::fmt::Debug,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WorkspaceGamlss")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl<F, Blocks, Obs> Objective for WorkspaceGamlss<F, Blocks, Obs>
where
    F: Family,
    Blocks: GamlssBlocks<F>,
    for<'row> Obs: ObservationView<'row, Observation = F::Observation<'row>>,
{
    type Error = ModelError;

    fn dim(&self) -> usize {
        self.model.nparams()
    }

    fn value(&mut self, parameters: &[f64]) -> Result<f64, Self::Error> {
        self.model.try_value(parameters)
    }

    fn gradient(&mut self, parameters: &[f64], grad: &mut [f64]) -> Result<(), Self::Error> {
        self.model
            .try_value_gradient_into_workspace(parameters, grad, &mut self.workspace)
            .map(|_| ())
    }

    fn value_gradient(&mut self, parameters: &[f64], grad: &mut [f64]) -> Result<f64, Self::Error> {
        self.model
            .try_value_gradient_into_workspace(parameters, grad, &mut self.workspace)
    }
}

/// Objective wrapper that adds penalties depending on the full beta vector.
///
/// Unlike [`Penalty`], which acts locally on a single block,
/// [`GlobalPenalty`] allows coupling of several blocks (e.g., centering or
/// LASSO-like penalties).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WithGlobalPenalties<O, GP> {
    objective: O,
    penalties: GP,
}

impl<O, GP> WithGlobalPenalties<O, GP> {
    /// Wrapped objective.
    #[must_use]
    #[inline]
    pub const fn objective(&self) -> &O {
        &self.objective
    }

    /// Wrapped objective, mutably.
    #[inline]
    pub const fn objective_mut(&mut self) -> &mut O {
        &mut self.objective
    }

    /// Global penalties evaluated on the full parameter vector.
    #[must_use]
    #[inline]
    pub const fn penalties(&self) -> &GP {
        &self.penalties
    }

    /// Global penalties evaluated on the full parameter vector, mutably.
    #[inline]
    pub const fn penalties_mut(&mut self) -> &mut GP {
        &mut self.penalties
    }

    /// Consumes the wrapper and returns the wrapped objective and penalties.
    #[must_use]
    #[inline]
    pub fn into_parts(self) -> (O, GP) {
        (self.objective, self.penalties)
    }
}

impl<O, GP> Objective for WithGlobalPenalties<O, GP>
where
    O: Objective,
    GP: GlobalPenalty,
{
    type Error = O::Error;

    fn dim(&self) -> usize {
        self.objective.dim()
    }

    fn value(&mut self, parameters: &[f64]) -> Result<f64, Self::Error> {
        Ok(self.objective.value(parameters)? + self.penalties.value(parameters))
    }

    fn gradient(&mut self, parameters: &[f64], grad: &mut [f64]) -> Result<(), Self::Error> {
        self.value_gradient(parameters, grad).map(|_| ())
    }

    fn value_gradient(&mut self, parameters: &[f64], grad: &mut [f64]) -> Result<f64, Self::Error> {
        let mut value = self.objective.value_gradient(parameters, grad)?;
        value += self.penalties.value(parameters);
        self.penalties.add_gradient(parameters, grad);
        Ok(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParameterStream {
    index: usize,
    role: &'static str,
    path_axis: Option<ParameterAxis>,
    range: Range<usize>,
}

impl ParameterStream {
    fn into_descriptor(self) -> ParameterDescriptor {
        ParameterDescriptor::new(
            self.role,
            self.path_axis
                .map_or_else(ParameterPath::whole, ParameterPath::from_axis),
            self.range,
        )
    }
}

/// Tuple contract for a set of parameter blocks compatible with family `F`.
///
/// Implementations are generated for typed tuples of [`ParameterBlock`]. The
/// model validates observation count, predictor row counts and coefficient
/// ranges before hot-path evaluation; generated methods may then assume
/// compatible row counts, finite non-negative weights and non-overlapping block
/// ranges.
pub trait GamlssBlocks<F>
where
    F: Family,
{
    /// Number of observations in the blocks.
    fn nrows(&self) -> usize;
    /// Length of the common beta vector covering all blocks.
    fn len(&self) -> usize;
    /// Validates and returns the common beta-vector length.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::BlockRangeOverflow`] if any block end index does
    /// not fit in `usize`.
    fn try_len(&self) -> Result<usize, ModelError> {
        Ok(self.len())
    }

    /// `true` if the blocks require no coefficients.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Validates that the blocks are compatible with the observation count
    /// `nobs`.
    fn validate(&self, nobs: usize) -> Result<(), ModelError>;
    /// Weighted negative log-likelihood without penalties.
    ///
    /// `obs` has already been validated by the model constructor. Each scalar
    /// likelihood contribution is multiplied by the corresponding observation
    /// weight.
    fn train_nll<'obs, Obs>(&self, family: &F, obs: &'obs Obs, beta: &[f64]) -> f64
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs;
    /// Additive predictors on the link scale for one row.
    fn eta_row(&self, beta: &[f64], row: usize) -> F::Eta
    where
        F: Family;
    /// Penalty value depending on coefficient blocks.
    fn penalty_value(&self, beta: &[f64]) -> f64;
    /// Creates a flat optimizer-parameter start vector from family-level
    /// link-scale initial predictors.
    fn initial_parameters<'obs, Obs>(&self, family: &F, obs: &'obs Obs) -> Vec<f64>
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs;
    /// Fallible variant of [`Self::initial_parameters`] for layouts whose
    /// total coefficient length must be checked first.
    fn try_initial_parameters<'obs, Obs>(
        &self,
        family: &F,
        obs: &'obs Obs,
    ) -> Result<Vec<f64>, ModelError>
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        Ok(self.initial_parameters(family, obs))
    }
    /// Adds the local penalty gradient into an existing full gradient vector.
    ///
    /// Implementations with local penalties should override this method. It is
    /// used by objective scaling to rescale likelihood gradients without
    /// changing the meaning of penalty weights. The default implementation is
    /// correct only for block collections whose [`penalty_value`](Self::penalty_value)
    /// has zero gradient.
    fn add_penalty_gradient(&self, _beta: &[f64], _grad: &mut [f64]) {}
    /// Value of the weighted negative log-likelihood plus penalties.
    fn value<'obs, Obs>(&self, family: &F, obs: &'obs Obs, beta: &[f64]) -> f64
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        self.train_nll(family, obs, beta) + self.penalty_value(beta)
    }
    /// Creates reusable buffers for repeated gradient evaluations.
    fn gradient_workspace(&self, nobs: usize) -> GradientWorkspace {
        let mut workspace = GradientWorkspace::new();
        let ranges = self.block_ranges();
        workspace.prepare(ranges.len());
        for (index, range) in ranges.iter().enumerate() {
            workspace.prepare_row_gradient(index, nobs);
            let _ = workspace.local_gradient_mut(index, range.len());
        }
        workspace
    }
    /// Adds the weighted gradient, reusing temporary buffers from `workspace`.
    ///
    /// The default implementation uses the fused value-gradient path and
    /// discards the value.
    fn gradient_into_workspace<'obs, Obs>(
        &self,
        family: &F,
        obs: &'obs Obs,
        beta: &[f64],
        grad: &mut [f64],
        family_workspace: &mut F::Workspace,
        workspace: &mut GradientWorkspace,
    ) where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        let _ = self.value_gradient_into_workspace(
            family,
            obs,
            beta,
            grad,
            family_workspace,
            workspace,
        );
    }

    /// Computes weighted objective value and gradient in one observation pass.
    fn value_gradient_into_workspace<'obs, Obs>(
        &self,
        family: &F,
        obs: &'obs Obs,
        beta: &[f64],
        grad: &mut [f64],
        family_workspace: &mut F::Workspace,
        workspace: &mut GradientWorkspace,
    ) -> f64
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs;
    /// Coefficient ranges for each block in the common beta vector.
    fn block_ranges(&self) -> Vec<Range<usize>>;
    /// Returns the layout of the coefficient blocks within the flat beta vector.
    fn parameter_layout(&self) -> ParameterLayout;

    /// Returns structured coefficient descriptors within the flat beta vector.
    ///
    /// Predictor blocks own coefficient ranges and raw `eta` construction.
    /// Families own the statistical meaning of those `eta` values and any
    /// dependent transforms into valid natural-scale parameters. Descriptors
    /// expose predictor-layer structure for diagnostics and formula builders
    /// without moving distribution parameterization logic into predictors.
    fn parameter_descriptors(&self) -> Vec<ParameterDescriptor> {
        self.parameter_layout().block_descriptors()
    }

    /// Visits coefficient ranges for each parameter block in model order without allocating.
    fn visit_block_ranges<V>(&self, mut visit: V)
    where
        V: FnMut(usize, Range<usize>),
    {
        for (index, range) in self.block_ranges().into_iter().enumerate() {
            visit(index, range);
        }
    }

    fn parameter_slice_count(&self) -> usize {
        self.parameter_layout().slices().len()
    }

    #[doc(hidden)]
    fn parameter_slice_matches(
        &self,
        index: usize,
        name: &'static str,
        range: Range<usize>,
    ) -> bool {
        self.parameter_layout()
            .slices()
            .get(index)
            .is_some_and(|slice| slice.name == name && slice.range == range)
    }

    /// Visits named parameter slices in model order without allocating.
    fn visit_parameter_slices<V>(&self, mut visit: V)
    where
        V: FnMut(usize, &'static str, Range<usize>),
    {
        for (index, slice) in self.parameter_layout().slices().iter().enumerate() {
            visit(index, slice.name, slice.range.clone());
        }
    }

    /// Visits structured coefficient descriptors in model order without allocating.
    fn visit_parameter_descriptors<V>(&self, visit: V)
    where
        V: FnMut(usize, ParameterDescriptor),
    {
        self.parameter_layout().visit_block_descriptors(visit);
    }

    #[doc(hidden)]
    fn parameter_slice_of<P>(&self) -> Option<Range<usize>>
    where
        P: ParameterName,
    {
        let mut found = None;
        self.visit_parameter_slices(|_, name, range| {
            if name == P::NAME && found.is_none() {
                found = Some(range);
            }
        });
        found
    }

    #[doc(hidden)]
    fn has_same_parameter_layout<Other>(&self, other: &Other) -> bool
    where
        Other: GamlssBlocks<F>,
    {
        let mut got_count = 0;
        let mut matches = true;

        other.visit_parameter_slices(|index, name, range| {
            got_count += 1;
            matches &= self.parameter_slice_matches(index, name, range);
        });

        matches && got_count == self.parameter_slice_count()
    }
}

impl<F, const D: usize, PVector, PLower, XVector, XLower, PenVector, PenLower> GamlssBlocks<F>
    for (
        VectorParameterBlock<PVector, D, XVector, PenVector>,
        LowerTriangularParameterBlock<PLower, D, XLower, PenLower>,
    )
where
    F: Family,
    F::ParamSpec:
        LocationCholeskySpec<F, D, VectorParameter = PVector, LowerTriangularParameter = PLower>,
    PVector: ParameterName,
    PLower: ParameterName,
    XVector: PredictorBlock,
    XLower: PredictorBlock,
    PenVector: Penalty,
    PenLower: Penalty,
{
    fn nrows(&self) -> usize {
        self.0.component(0).map_or(0, PredictorBlock::nrows)
    }

    fn len(&self) -> usize {
        <Self as GamlssBlocks<F>>::try_len(self)
            .expect("validated structured parameter block layout must fit in usize")
    }

    fn try_len(&self) -> Result<usize, ModelError> {
        Ok(self.0.try_range()?.end.max(self.1.try_range()?.end))
    }

    fn validate(&self, nobs: usize) -> Result<(), ModelError> {
        if D == 0 {
            return Err(ModelError::InvalidParameter {
                parameter: "dimension",
                expected: "positive",
            });
        }

        validate_vector_parameter_block::<PVector, D, _, _>(&self.0, nobs)?;
        validate_lower_triangular_parameter_block::<PLower, D, _, _>(&self.1, nobs)?;
        self.0.penalty().validate_dim(self.0.len())?;
        self.1.penalty().validate_dim(self.1.len())?;

        let vector = self.0.try_range()?;
        let lower = self.1.try_range()?;
        if ranges_overlap(vector, lower) {
            return Err(ModelError::BlockOverlap {
                first: PVector::NAME,
                second: PLower::NAME,
            });
        }

        Ok(())
    }

    fn train_nll<'obs, Obs>(&self, family: &F, obs: &'obs Obs, beta: &[f64]) -> f64
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        let mut loss = 0.0;
        let mut family_workspace = family.workspace();
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            if weight == 0.0 {
                continue;
            }
            let observation = obs.observation_at(row);
            loss = weight.mul_add(
                {
                    let eta = structured_eta_row::<F, D, _, _, _, _, _, _>(self, beta, row);
                    family.nll_eta(observation, &eta, &mut family_workspace)
                },
                loss,
            );
        }
        loss
    }

    fn eta_row(&self, beta: &[f64], row: usize) -> F::Eta {
        structured_eta_row::<F, D, _, _, _, _, _, _>(self, beta, row)
    }

    fn penalty_value(&self, beta: &[f64]) -> f64 {
        self.0.penalty().value(&beta[self.0.range()])
            + self.1.penalty().value(&beta[self.1.range()])
    }

    fn initial_parameters<'obs, Obs>(&self, family: &F, obs: &'obs Obs) -> Vec<f64>
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        <Self as GamlssBlocks<F>>::try_initial_parameters(self, family, obs)
            .expect("validated structured parameter block layout must fit in usize")
    }

    fn try_initial_parameters<'obs, Obs>(
        &self,
        family: &F,
        obs: &'obs Obs,
    ) -> Result<Vec<f64>, ModelError>
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        let (vector_eta, lower_eta) =
            <F::ParamSpec as LocationCholeskySpec<F, D>>::initial_vector_lower_from_observations(
                family, obs,
            );
        let mut beta = vec![0.0; <Self as GamlssBlocks<F>>::try_len(self)?];

        for (component, value) in vector_eta.iter().copied().enumerate() {
            if value.is_finite() {
                let predictor = self
                    .0
                    .component(component)
                    .expect("validated vector block has D components");
                let range = self
                    .0
                    .component_range(component)
                    .expect("validated vector component has a coefficient range");
                predictor.set_constant_start(value, &mut beta[range]);
            }
        }

        for (row, row_values) in lower_eta.iter().enumerate() {
            for (col, value) in row_values.iter().copied().take(row + 1).enumerate() {
                if value.is_finite() {
                    let predictor = self
                        .1
                        .entry(row, col)
                        .expect("validated lower-triangular block has this entry");
                    let range = self
                        .1
                        .entry_range(row, col)
                        .expect("validated lower-triangular entry has a coefficient range");
                    predictor.set_constant_start(value, &mut beta[range]);
                }
            }
        }

        Ok(beta)
    }

    fn add_penalty_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        self.0
            .penalty()
            .add_gradient(&beta[self.0.range()], &mut grad[self.0.range()]);
        self.1
            .penalty()
            .add_gradient(&beta[self.1.range()], &mut grad[self.1.range()]);
    }

    fn gradient_workspace(&self, nobs: usize) -> GradientWorkspace {
        let scalar_count = structured_scalar_count::<PLower, D>();
        let mut workspace = GradientWorkspace::new();
        workspace.prepare(scalar_count);
        for index in 0..scalar_count {
            workspace.prepare_row_gradient(index, nobs);
        }

        visit_vector_parameter_streams(&self.0, 0, |stream| {
            let _ = workspace.local_gradient_mut(stream.index, stream.range.len());
        });
        visit_lower_triangular_parameter_streams(&self.1, D, |stream| {
            let _ = workspace.local_gradient_mut(stream.index, stream.range.len());
        });

        workspace
    }

    fn value_gradient_into_workspace<'obs, Obs>(
        &self,
        family: &F,
        obs: &'obs Obs,
        beta: &[f64],
        grad: &mut [f64],
        family_workspace: &mut F::Workspace,
        workspace: &mut GradientWorkspace,
    ) -> f64
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        let scalar_count = structured_scalar_count::<PLower, D>();
        workspace.prepare(scalar_count);
        for index in 0..scalar_count {
            workspace.prepare_row_gradient(index, obs.len());
        }

        let mut loss = 0.0;
        for row_index in 0..obs.len() {
            let weight = obs.weight_at(row_index);
            if weight == 0.0 {
                for index in 0..scalar_count {
                    workspace.set_row_gradient(index, row_index, 0.0);
                }
                continue;
            }

            let observation = obs.observation_at(row_index);
            let eta = structured_eta_row::<F, D, _, _, _, _, _, _>(self, beta, row_index);
            let (nll, gradient) = family.nll_and_gradient_eta(observation, &eta, family_workspace);
            loss = weight.mul_add(nll, loss);

            for component in 0..D {
                workspace.set_row_gradient(
                    component,
                    row_index,
                    weight
                        * <F::ParamSpec as LocationCholeskySpec<F, D>>::vector_gradient_part(
                            &gradient, component,
                        ),
                );
            }
            for row in 0..D {
                for col in 0..=row {
                    workspace.set_row_gradient(
                        lower_workspace_index::<PLower, D>(row, col),
                        row_index,
                        weight
                            * <F::ParamSpec as LocationCholeskySpec<F, D>>::lower_triangular_gradient_part(
                                &gradient, row, col,
                            ),
                    );
                }
            }
        }

        for component in 0..D {
            let predictor = self
                .0
                .component(component)
                .expect("validated vector block has D components");
            let range = self
                .0
                .component_range(component)
                .expect("validated vector component has a coefficient range");
            let (row_gradient, local_gradient) =
                workspace.row_gradient_and_local_gradient_mut(component, range.len());
            predictor.add_gradient(row_gradient, &beta[range.clone()], local_gradient);
            add_into(&mut grad[range], local_gradient);
        }

        for row in 0..D {
            for col in 0..=row {
                let predictor = self
                    .1
                    .entry(row, col)
                    .expect("validated lower-triangular block has this entry");
                let range = self
                    .1
                    .entry_range(row, col)
                    .expect("validated lower-triangular entry has a coefficient range");
                let (row_gradient, local_gradient) = workspace.row_gradient_and_local_gradient_mut(
                    lower_workspace_index::<PLower, D>(row, col),
                    range.len(),
                );
                predictor.add_gradient(row_gradient, &beta[range.clone()], local_gradient);
                add_into(&mut grad[range], local_gradient);
            }
        }

        loss += self.0.penalty().value(&beta[self.0.range()]);
        self.0
            .penalty()
            .add_gradient(&beta[self.0.range()], &mut grad[self.0.range()]);
        loss += self.1.penalty().value(&beta[self.1.range()]);
        self.1
            .penalty()
            .add_gradient(&beta[self.1.range()], &mut grad[self.1.range()]);

        loss
    }

    fn block_ranges(&self) -> Vec<Range<usize>> {
        vec![self.0.range(), self.1.range()]
    }

    fn parameter_layout(&self) -> ParameterLayout {
        ParameterLayout::new(vec![
            ParameterSlice {
                name: PVector::NAME,
                range: self.0.range(),
            },
            ParameterSlice {
                name: PLower::NAME,
                range: self.1.range(),
            },
        ])
    }

    fn parameter_descriptors(&self) -> Vec<ParameterDescriptor> {
        let mut descriptors = Vec::with_capacity(structured_scalar_count::<PLower, D>());
        <Self as GamlssBlocks<F>>::visit_parameter_descriptors(self, |_, descriptor| {
            descriptors.push(descriptor);
        });
        descriptors
    }

    fn parameter_slice_count(&self) -> usize {
        2
    }

    fn parameter_slice_matches(
        &self,
        index: usize,
        name: &'static str,
        range: Range<usize>,
    ) -> bool {
        match index {
            0 => name == PVector::NAME && range == self.0.range(),
            1 => name == PLower::NAME && range == self.1.range(),
            _ => false,
        }
    }

    fn visit_parameter_slices<V>(&self, mut visit: V)
    where
        V: FnMut(usize, &'static str, Range<usize>),
    {
        visit(0, PVector::NAME, self.0.range());
        visit(1, PLower::NAME, self.1.range());
    }

    fn visit_parameter_descriptors<V>(&self, mut visit: V)
    where
        V: FnMut(usize, ParameterDescriptor),
    {
        visit_vector_parameter_streams(&self.0, 0, |stream| {
            visit(stream.index, stream.into_descriptor());
        });
        visit_lower_triangular_parameter_streams(&self.1, D, |stream| {
            visit(stream.index, stream.into_descriptor());
        });
    }

    fn has_same_parameter_layout<Other>(&self, other: &Other) -> bool
    where
        Other: GamlssBlocks<F>,
    {
        other.parameter_slice_count() == 2
            && other.parameter_slice_matches(0, PVector::NAME, self.0.range())
            && other.parameter_slice_matches(1, PLower::NAME, self.1.range())
    }
}

impl<
    F,
    const D: usize,
    PLocation,
    PScale,
    PCorr,
    XLocation,
    XScale,
    XCorr,
    PenLocation,
    PenScale,
    PenCorr,
> GamlssBlocks<F>
    for (
        VectorParameterBlock<PLocation, D, XLocation, PenLocation>,
        VectorParameterBlock<PScale, D, XScale, PenScale>,
        StrictLowerTriangularParameterBlock<PCorr, D, XCorr, PenCorr>,
    )
where
    F: Family,
    F::ParamSpec: LocationScalePartialCorrSpec<
            F,
            D,
            LocationParameter = PLocation,
            ScaleParameter = PScale,
            PartialCorrelationParameter = PCorr,
        >,
    PLocation: ParameterName,
    PScale: ParameterName,
    PCorr: ParameterName,
    XLocation: PredictorBlock,
    XScale: PredictorBlock,
    XCorr: PredictorBlock,
    PenLocation: Penalty,
    PenScale: Penalty,
    PenCorr: Penalty,
{
    fn nrows(&self) -> usize {
        self.0.component(0).map_or(0, PredictorBlock::nrows)
    }

    fn len(&self) -> usize {
        <Self as GamlssBlocks<F>>::try_len(self)
            .expect("validated partial-correlation parameter block layout must fit in usize")
    }

    fn try_len(&self) -> Result<usize, ModelError> {
        Ok(self
            .0
            .try_range()?
            .end
            .max(self.1.try_range()?.end)
            .max(self.2.try_range()?.end))
    }

    fn validate(&self, nobs: usize) -> Result<(), ModelError> {
        if D == 0 {
            return Err(ModelError::InvalidParameter {
                parameter: "dimension",
                expected: "positive",
            });
        }

        validate_vector_parameter_block::<PLocation, D, _, _>(&self.0, nobs)?;
        validate_vector_parameter_block::<PScale, D, _, _>(&self.1, nobs)?;
        validate_strict_lower_triangular_parameter_block::<PCorr, D, _, _>(&self.2, nobs)?;
        self.0.penalty().validate_dim(self.0.len())?;
        self.1.penalty().validate_dim(self.1.len())?;
        self.2.penalty().validate_dim(self.2.len())?;

        let ranges = [
            (PLocation::NAME, self.0.try_range()?),
            (PScale::NAME, self.1.try_range()?),
            (PCorr::NAME, self.2.try_range()?),
        ];
        validate_non_overlapping_ranges(&ranges)?;

        Ok(())
    }

    fn train_nll<'obs, Obs>(&self, family: &F, obs: &'obs Obs, beta: &[f64]) -> f64
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        let mut loss = 0.0;
        let mut family_workspace = family.workspace();
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            if weight == 0.0 {
                continue;
            }
            let eta = location_scale_partial_corr_eta_row::<F, D, _, _, _, _, _, _, _, _, _>(
                self, beta, row,
            );
            loss = weight.mul_add(
                family.nll_eta(obs.observation_at(row), &eta, &mut family_workspace),
                loss,
            );
        }
        loss
    }

    fn eta_row(&self, beta: &[f64], row: usize) -> F::Eta {
        location_scale_partial_corr_eta_row::<F, D, _, _, _, _, _, _, _, _, _>(self, beta, row)
    }

    fn penalty_value(&self, beta: &[f64]) -> f64 {
        self.0.penalty().value(&beta[self.0.range()])
            + self.1.penalty().value(&beta[self.1.range()])
            + self.2.penalty().value(&beta[self.2.range()])
    }

    fn initial_parameters<'obs, Obs>(&self, _family: &F, _obs: &'obs Obs) -> Vec<f64>
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        vec![0.0; <Self as GamlssBlocks<F>>::len(self)]
    }

    fn try_initial_parameters<'obs, Obs>(
        &self,
        _family: &F,
        _obs: &'obs Obs,
    ) -> Result<Vec<f64>, ModelError>
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        Ok(vec![0.0; <Self as GamlssBlocks<F>>::try_len(self)?])
    }

    fn add_penalty_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        self.0
            .penalty()
            .add_gradient(&beta[self.0.range()], &mut grad[self.0.range()]);
        self.1
            .penalty()
            .add_gradient(&beta[self.1.range()], &mut grad[self.1.range()]);
        self.2
            .penalty()
            .add_gradient(&beta[self.2.range()], &mut grad[self.2.range()]);
    }

    fn gradient_workspace(&self, nobs: usize) -> GradientWorkspace {
        let scalar_count = D + D + strict_lower_workspace_count::<PCorr, D>();
        let mut workspace = GradientWorkspace::new();
        workspace.prepare(scalar_count);
        for index in 0..scalar_count {
            workspace.prepare_row_gradient(index, nobs);
        }
        visit_vector_parameter_streams(&self.0, 0, |stream| {
            let _ = workspace.local_gradient_mut(stream.index, stream.range.len());
        });
        visit_vector_parameter_streams(&self.1, D, |stream| {
            let _ = workspace.local_gradient_mut(stream.index, stream.range.len());
        });
        visit_strict_lower_triangular_parameter_streams(&self.2, D + D, |stream| {
            let _ = workspace.local_gradient_mut(stream.index, stream.range.len());
        });
        workspace
    }

    fn value_gradient_into_workspace<'obs, Obs>(
        &self,
        family: &F,
        obs: &'obs Obs,
        beta: &[f64],
        grad: &mut [f64],
        family_workspace: &mut F::Workspace,
        workspace: &mut GradientWorkspace,
    ) -> f64
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        let scalar_count = D + D + strict_lower_workspace_count::<PCorr, D>();
        workspace.prepare(scalar_count);
        for index in 0..scalar_count {
            workspace.prepare_row_gradient(index, obs.len());
        }

        let mut loss = 0.0;
        for row_index in 0..obs.len() {
            let weight = obs.weight_at(row_index);
            if weight == 0.0 {
                for index in 0..scalar_count {
                    workspace.set_row_gradient(index, row_index, 0.0);
                }
                continue;
            }

            let eta = location_scale_partial_corr_eta_row::<F, D, _, _, _, _, _, _, _, _, _>(
                self, beta, row_index,
            );
            let (nll, gradient) =
                family.nll_and_gradient_eta(obs.observation_at(row_index), &eta, family_workspace);
            loss = weight.mul_add(nll, loss);
            for component in 0..D {
                workspace.set_row_gradient(
                    component,
                    row_index,
                    weight
                        * <F::ParamSpec as LocationScalePartialCorrSpec<F, D>>::location_gradient_part(
                            &gradient, component,
                        ),
                );
                workspace.set_row_gradient(
                    D + component,
                    row_index,
                    weight
                        * <F::ParamSpec as LocationScalePartialCorrSpec<F, D>>::scale_gradient_part(
                            &gradient, component,
                        ),
                );
            }
            for row in 0..D {
                for col in 0..row {
                    workspace.set_row_gradient(
                        partial_corr_workspace_index::<PCorr, D>(row, col),
                        row_index,
                        weight
                            * <F::ParamSpec as LocationScalePartialCorrSpec<F, D>>::partial_corr_gradient_part(
                                &gradient, row, col,
                            ),
                    );
                }
            }
        }

        for component in 0..D {
            let predictor = self.0.component(component).unwrap();
            let range = self.0.component_range(component).unwrap();
            let (row_gradient, local_gradient) =
                workspace.row_gradient_and_local_gradient_mut(component, range.len());
            predictor.add_gradient(row_gradient, &beta[range.clone()], local_gradient);
            add_into(&mut grad[range], local_gradient);

            let predictor = self.1.component(component).unwrap();
            let range = self.1.component_range(component).unwrap();
            let (row_gradient, local_gradient) =
                workspace.row_gradient_and_local_gradient_mut(D + component, range.len());
            predictor.add_gradient(row_gradient, &beta[range.clone()], local_gradient);
            add_into(&mut grad[range], local_gradient);
        }
        for row in 0..D {
            for col in 0..row {
                let predictor = self.2.entry(row, col).unwrap();
                let range = self.2.entry_range(row, col).unwrap();
                let (row_gradient, local_gradient) = workspace.row_gradient_and_local_gradient_mut(
                    partial_corr_workspace_index::<PCorr, D>(row, col),
                    range.len(),
                );
                predictor.add_gradient(row_gradient, &beta[range.clone()], local_gradient);
                add_into(&mut grad[range], local_gradient);
            }
        }

        loss += self.0.penalty().value(&beta[self.0.range()]);
        self.0
            .penalty()
            .add_gradient(&beta[self.0.range()], &mut grad[self.0.range()]);
        loss += self.1.penalty().value(&beta[self.1.range()]);
        self.1
            .penalty()
            .add_gradient(&beta[self.1.range()], &mut grad[self.1.range()]);
        loss += self.2.penalty().value(&beta[self.2.range()]);
        self.2
            .penalty()
            .add_gradient(&beta[self.2.range()], &mut grad[self.2.range()]);

        loss
    }

    fn block_ranges(&self) -> Vec<Range<usize>> {
        vec![self.0.range(), self.1.range(), self.2.range()]
    }

    fn parameter_layout(&self) -> ParameterLayout {
        ParameterLayout::new(vec![
            ParameterSlice {
                name: PLocation::NAME,
                range: self.0.range(),
            },
            ParameterSlice {
                name: PScale::NAME,
                range: self.1.range(),
            },
            ParameterSlice {
                name: PCorr::NAME,
                range: self.2.range(),
            },
        ])
    }

    fn parameter_descriptors(&self) -> Vec<ParameterDescriptor> {
        let mut descriptors =
            Vec::with_capacity(D + D + strict_lower_workspace_count::<PCorr, D>());
        <Self as GamlssBlocks<F>>::visit_parameter_descriptors(self, |_, descriptor| {
            descriptors.push(descriptor);
        });
        descriptors
    }

    fn parameter_slice_count(&self) -> usize {
        3
    }

    fn parameter_slice_matches(
        &self,
        index: usize,
        name: &'static str,
        range: Range<usize>,
    ) -> bool {
        match index {
            0 => name == PLocation::NAME && range == self.0.range(),
            1 => name == PScale::NAME && range == self.1.range(),
            2 => name == PCorr::NAME && range == self.2.range(),
            _ => false,
        }
    }

    fn visit_parameter_slices<V>(&self, mut visit: V)
    where
        V: FnMut(usize, &'static str, Range<usize>),
    {
        visit(0, PLocation::NAME, self.0.range());
        visit(1, PScale::NAME, self.1.range());
        visit(2, PCorr::NAME, self.2.range());
    }

    fn visit_parameter_descriptors<V>(&self, mut visit: V)
    where
        V: FnMut(usize, ParameterDescriptor),
    {
        visit_vector_parameter_streams(&self.0, 0, |stream| {
            visit(stream.index, stream.into_descriptor());
        });
        visit_vector_parameter_streams(&self.1, D, |stream| {
            visit(stream.index, stream.into_descriptor());
        });
        visit_strict_lower_triangular_parameter_streams(&self.2, D + D, |stream| {
            visit(stream.index, stream.into_descriptor());
        });
    }

    fn has_same_parameter_layout<Other>(&self, other: &Other) -> bool
    where
        Other: GamlssBlocks<F>,
    {
        other.parameter_slice_count() == 3
            && other.parameter_slice_matches(0, PLocation::NAME, self.0.range())
            && other.parameter_slice_matches(1, PScale::NAME, self.1.range())
            && other.parameter_slice_matches(2, PCorr::NAME, self.2.range())
    }
}

impl<F, const D: usize, PMean, PPrecision, LPrecision, XMean, XPrecision, PenMean, PenPrecision>
    GamlssBlocks<F>
    for (
        SimplexLogitParameterBlock<PMean, D, XMean, PenMean>,
        ParameterBlock<PPrecision, LPrecision, XPrecision, PenPrecision>,
    )
where
    F: Family,
    F::ParamSpec: MeanPrecisionSimplexSpec<
            F,
            D,
            MeanParameter = PMean,
            PrecisionParameter = PPrecision,
            PrecisionLink = LPrecision,
        >,
    PMean: ParameterName,
    PPrecision: ParameterName,
    LPrecision: crate::Link<f64>,
    XMean: PredictorBlock,
    XPrecision: PredictorBlock,
    PenMean: Penalty,
    PenPrecision: Penalty,
{
    fn nrows(&self) -> usize {
        self.0
            .logit(0)
            .map_or_else(|| PredictorBlock::nrows(self.1.x()), PredictorBlock::nrows)
    }

    fn len(&self) -> usize {
        <Self as GamlssBlocks<F>>::try_len(self)
            .expect("validated simplex parameter block layout must fit in usize")
    }

    fn try_len(&self) -> Result<usize, ModelError> {
        Ok(self.0.try_range()?.end.max(self.1.try_range()?.end))
    }

    fn validate(&self, nobs: usize) -> Result<(), ModelError> {
        if D < 2 {
            return Err(ModelError::InvalidParameter {
                parameter: "dimension",
                expected: "at least two simplex components",
            });
        }

        validate_simplex_logit_parameter_block::<PMean, D, _, _>(&self.0, nobs)?;
        self.1.x().validate()?;
        validate_block_rows(PPrecision::NAME, self.1.x().nrows(), nobs)?;
        self.0.penalty().validate_dim(self.0.len())?;
        self.1.penalty().validate_dim(self.1.len())?;

        let ranges = [
            (PMean::NAME, self.0.try_range()?),
            (PPrecision::NAME, self.1.try_range()?),
        ];
        validate_non_overlapping_ranges(&ranges)?;
        Ok(())
    }

    fn train_nll<'obs, Obs>(&self, family: &F, obs: &'obs Obs, beta: &[f64]) -> f64
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        let mut loss = 0.0;
        let mut family_workspace = family.workspace();
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            if weight == 0.0 {
                continue;
            }
            let eta = simplex_precision_eta_row::<F, D, _, _, _, _, _, _, _>(self, beta, row);
            loss = weight.mul_add(
                family.nll_eta(obs.observation_at(row), &eta, &mut family_workspace),
                loss,
            );
        }
        loss
    }

    fn eta_row(&self, beta: &[f64], row: usize) -> F::Eta {
        simplex_precision_eta_row::<F, D, _, _, _, _, _, _, _>(self, beta, row)
    }

    fn penalty_value(&self, beta: &[f64]) -> f64 {
        self.0.penalty().value(&beta[self.0.range()])
            + self.1.penalty().value(&beta[self.1.range()])
    }

    fn initial_parameters<'obs, Obs>(&self, _family: &F, _obs: &'obs Obs) -> Vec<f64>
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        vec![0.0; <Self as GamlssBlocks<F>>::len(self)]
    }

    fn try_initial_parameters<'obs, Obs>(
        &self,
        _family: &F,
        _obs: &'obs Obs,
    ) -> Result<Vec<f64>, ModelError>
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        Ok(vec![0.0; <Self as GamlssBlocks<F>>::try_len(self)?])
    }

    fn add_penalty_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        self.0
            .penalty()
            .add_gradient(&beta[self.0.range()], &mut grad[self.0.range()]);
        self.1
            .penalty()
            .add_gradient(&beta[self.1.range()], &mut grad[self.1.range()]);
    }

    fn gradient_workspace(&self, nobs: usize) -> GradientWorkspace {
        let scalar_count = D;
        let mut workspace = GradientWorkspace::new();
        workspace.prepare(scalar_count);
        for index in 0..scalar_count {
            workspace.prepare_row_gradient(index, nobs);
        }
        visit_simplex_logit_parameter_streams(&self.0, 0, |stream| {
            let _ = workspace.local_gradient_mut(stream.index, stream.range.len());
        });
        visit_scalar_parameter_stream(&self.1, D - 1, |stream| {
            let _ = workspace.local_gradient_mut(stream.index, stream.range.len());
        });
        workspace
    }

    fn value_gradient_into_workspace<'obs, Obs>(
        &self,
        family: &F,
        obs: &'obs Obs,
        beta: &[f64],
        grad: &mut [f64],
        family_workspace: &mut F::Workspace,
        workspace: &mut GradientWorkspace,
    ) -> f64
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        let scalar_count = D;
        workspace.prepare(scalar_count);
        for index in 0..scalar_count {
            workspace.prepare_row_gradient(index, obs.len());
        }

        let mut loss = 0.0;
        for row_index in 0..obs.len() {
            let weight = obs.weight_at(row_index);
            if weight == 0.0 {
                for index in 0..scalar_count {
                    workspace.set_row_gradient(index, row_index, 0.0);
                }
                continue;
            }

            let eta = simplex_precision_eta_row::<F, D, _, _, _, _, _, _, _>(self, beta, row_index);
            let (nll, gradient) =
                family.nll_and_gradient_eta(obs.observation_at(row_index), &eta, family_workspace);
            loss = weight.mul_add(nll, loss);
            for component in 0..D.saturating_sub(1) {
                workspace.set_row_gradient(
                    component,
                    row_index,
                    weight
                        * <F::ParamSpec as MeanPrecisionSimplexSpec<F, D>>::simplex_logit_gradient_part(
                            &gradient, component,
                        ),
                );
            }
            workspace.set_row_gradient(
                D - 1,
                row_index,
                weight
                    * <F::ParamSpec as MeanPrecisionSimplexSpec<F, D>>::precision_gradient_part(
                        &gradient,
                    ),
            );
        }

        for component in 0..D.saturating_sub(1) {
            let predictor = self.0.logit(component).unwrap();
            let range = self.0.logit_range(component).unwrap();
            let (row_gradient, local_gradient) =
                workspace.row_gradient_and_local_gradient_mut(component, range.len());
            predictor.add_gradient(row_gradient, &beta[range.clone()], local_gradient);
            add_into(&mut grad[range], local_gradient);
        }
        let beta_block = &beta[self.1.range()];
        let (row_gradient, local_gradient) =
            workspace.row_gradient_and_local_gradient_mut(D - 1, self.1.len());
        self.1
            .x()
            .add_gradient(row_gradient, beta_block, local_gradient);
        add_into(&mut grad[self.1.range()], local_gradient);

        loss += self.0.penalty().value(&beta[self.0.range()]);
        self.0
            .penalty()
            .add_gradient(&beta[self.0.range()], &mut grad[self.0.range()]);
        loss += self.1.penalty().value(beta_block);
        self.1
            .penalty()
            .add_gradient(beta_block, &mut grad[self.1.range()]);

        loss
    }

    fn block_ranges(&self) -> Vec<Range<usize>> {
        vec![self.0.range(), self.1.range()]
    }

    fn parameter_layout(&self) -> ParameterLayout {
        ParameterLayout::new(vec![
            ParameterSlice {
                name: PMean::NAME,
                range: self.0.range(),
            },
            ParameterSlice {
                name: PPrecision::NAME,
                range: self.1.range(),
            },
        ])
    }

    fn parameter_descriptors(&self) -> Vec<ParameterDescriptor> {
        let mut descriptors = Vec::with_capacity(D);
        <Self as GamlssBlocks<F>>::visit_parameter_descriptors(self, |_, descriptor| {
            descriptors.push(descriptor);
        });
        descriptors
    }

    fn parameter_slice_count(&self) -> usize {
        2
    }

    fn parameter_slice_matches(
        &self,
        index: usize,
        name: &'static str,
        range: Range<usize>,
    ) -> bool {
        match index {
            0 => name == PMean::NAME && range == self.0.range(),
            1 => name == PPrecision::NAME && range == self.1.range(),
            _ => false,
        }
    }

    fn visit_parameter_slices<V>(&self, mut visit: V)
    where
        V: FnMut(usize, &'static str, Range<usize>),
    {
        visit(0, PMean::NAME, self.0.range());
        visit(1, PPrecision::NAME, self.1.range());
    }

    fn visit_parameter_descriptors<V>(&self, mut visit: V)
    where
        V: FnMut(usize, ParameterDescriptor),
    {
        visit_simplex_logit_parameter_streams(&self.0, 0, |stream| {
            visit(stream.index, stream.into_descriptor());
        });
        visit_scalar_parameter_stream(&self.1, D - 1, |stream| {
            visit(stream.index, stream.into_descriptor());
        });
    }

    fn has_same_parameter_layout<Other>(&self, other: &Other) -> bool
    where
        Other: GamlssBlocks<F>,
    {
        other.parameter_slice_count() == 2
            && other.parameter_slice_matches(0, PMean::NAME, self.0.range())
            && other.parameter_slice_matches(1, PPrecision::NAME, self.1.range())
    }
}

impl<
    F,
    const D: usize,
    PVector,
    PLower,
    PScalar,
    LScalar,
    XVector,
    XLower,
    XScalar,
    PenVector,
    PenLower,
    PenScalar,
> GamlssBlocks<F>
    for (
        VectorParameterBlock<PVector, D, XVector, PenVector>,
        LowerTriangularParameterBlock<PLower, D, XLower, PenLower>,
        ParameterBlock<PScalar, LScalar, XScalar, PenScalar>,
    )
where
    F: Family,
    F::ParamSpec: LocationCholeskyScalarSpec<
            F,
            D,
            VectorParameter = PVector,
            LowerTriangularParameter = PLower,
            ScalarParameter = PScalar,
            ScalarLink = LScalar,
        >,
    PVector: ParameterName,
    PLower: ParameterName,
    PScalar: ParameterName,
    LScalar: crate::Link<f64>,
    XVector: PredictorBlock,
    XLower: PredictorBlock,
    XScalar: PredictorBlock,
    PenVector: Penalty,
    PenLower: Penalty,
    PenScalar: Penalty,
{
    fn nrows(&self) -> usize {
        self.0.component(0).map_or(0, PredictorBlock::nrows)
    }

    fn len(&self) -> usize {
        <Self as GamlssBlocks<F>>::try_len(self)
            .expect("validated Cholesky-scalar parameter block layout must fit in usize")
    }

    fn try_len(&self) -> Result<usize, ModelError> {
        Ok(self
            .0
            .try_range()?
            .end
            .max(self.1.try_range()?.end)
            .max(self.2.try_range()?.end))
    }

    fn validate(&self, nobs: usize) -> Result<(), ModelError> {
        if D == 0 {
            return Err(ModelError::InvalidParameter {
                parameter: "dimension",
                expected: "positive",
            });
        }

        validate_vector_parameter_block::<PVector, D, _, _>(&self.0, nobs)?;
        validate_lower_triangular_parameter_block::<PLower, D, _, _>(&self.1, nobs)?;
        self.2.x().validate()?;
        validate_block_rows(PScalar::NAME, self.2.x().nrows(), nobs)?;
        self.0.penalty().validate_dim(self.0.len())?;
        self.1.penalty().validate_dim(self.1.len())?;
        self.2.penalty().validate_dim(self.2.len())?;

        let ranges = [
            (PVector::NAME, self.0.try_range()?),
            (PLower::NAME, self.1.try_range()?),
            (PScalar::NAME, self.2.try_range()?),
        ];
        validate_non_overlapping_ranges(&ranges)?;

        Ok(())
    }

    fn train_nll<'obs, Obs>(&self, family: &F, obs: &'obs Obs, beta: &[f64]) -> f64
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        let mut loss = 0.0;
        let mut family_workspace = family.workspace();
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            if weight == 0.0 {
                continue;
            }
            let eta = location_cholesky_scalar_eta_row::<F, D, _, _, _, _, _, _, _, _, _, _>(
                self, beta, row,
            );
            loss = weight.mul_add(
                family.nll_eta(obs.observation_at(row), &eta, &mut family_workspace),
                loss,
            );
        }
        loss
    }

    fn eta_row(&self, beta: &[f64], row: usize) -> F::Eta {
        location_cholesky_scalar_eta_row::<F, D, _, _, _, _, _, _, _, _, _, _>(self, beta, row)
    }

    fn penalty_value(&self, beta: &[f64]) -> f64 {
        self.0.penalty().value(&beta[self.0.range()])
            + self.1.penalty().value(&beta[self.1.range()])
            + self.2.penalty().value(&beta[self.2.range()])
    }

    fn initial_parameters<'obs, Obs>(&self, _family: &F, _obs: &'obs Obs) -> Vec<f64>
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        vec![0.0; <Self as GamlssBlocks<F>>::len(self)]
    }

    fn try_initial_parameters<'obs, Obs>(
        &self,
        _family: &F,
        _obs: &'obs Obs,
    ) -> Result<Vec<f64>, ModelError>
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        Ok(vec![0.0; <Self as GamlssBlocks<F>>::try_len(self)?])
    }

    fn add_penalty_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        self.0
            .penalty()
            .add_gradient(&beta[self.0.range()], &mut grad[self.0.range()]);
        self.1
            .penalty()
            .add_gradient(&beta[self.1.range()], &mut grad[self.1.range()]);
        self.2
            .penalty()
            .add_gradient(&beta[self.2.range()], &mut grad[self.2.range()]);
    }

    fn gradient_workspace(&self, nobs: usize) -> GradientWorkspace {
        let scalar_count = D
            + LowerTriangularParameterBlock::<PLower, D, (), ()>::packed_len()
                .expect("D * (D + 1) / 2 must fit")
            + 1;
        let mut workspace = GradientWorkspace::new();
        workspace.prepare(scalar_count);
        for index in 0..scalar_count {
            workspace.prepare_row_gradient(index, nobs);
        }
        visit_vector_parameter_streams(&self.0, 0, |stream| {
            let _ = workspace.local_gradient_mut(stream.index, stream.range.len());
        });
        visit_lower_triangular_parameter_streams(&self.1, D, |stream| {
            let _ = workspace.local_gradient_mut(stream.index, stream.range.len());
        });
        visit_scalar_parameter_stream(&self.2, scalar_count - 1, |stream| {
            let _ = workspace.local_gradient_mut(stream.index, stream.range.len());
        });
        workspace
    }

    fn value_gradient_into_workspace<'obs, Obs>(
        &self,
        family: &F,
        obs: &'obs Obs,
        beta: &[f64],
        grad: &mut [f64],
        family_workspace: &mut F::Workspace,
        workspace: &mut GradientWorkspace,
    ) -> f64
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        let scalar_index = D + LowerTriangularParameterBlock::<PLower, D, (), ()>::packed_len()
            .expect("D * (D + 1) / 2 must fit");
        let scalar_count = scalar_index + 1;
        workspace.prepare(scalar_count);
        for index in 0..scalar_count {
            workspace.prepare_row_gradient(index, obs.len());
        }

        let mut loss = 0.0;
        for row_index in 0..obs.len() {
            let weight = obs.weight_at(row_index);
            if weight == 0.0 {
                for index in 0..scalar_count {
                    workspace.set_row_gradient(index, row_index, 0.0);
                }
                continue;
            }
            let eta = location_cholesky_scalar_eta_row::<F, D, _, _, _, _, _, _, _, _, _, _>(
                self, beta, row_index,
            );
            let (nll, gradient) =
                family.nll_and_gradient_eta(obs.observation_at(row_index), &eta, family_workspace);
            loss = weight.mul_add(nll, loss);
            for component in 0..D {
                workspace.set_row_gradient(
                    component,
                    row_index,
                    weight
                        * <F::ParamSpec as LocationCholeskyScalarSpec<F, D>>::vector_gradient_part(
                            &gradient, component,
                        ),
                );
            }
            for row in 0..D {
                for col in 0..=row {
                    workspace.set_row_gradient(
                        lower_workspace_index::<PLower, D>(row, col),
                        row_index,
                        weight
                            * <F::ParamSpec as LocationCholeskyScalarSpec<F, D>>::lower_triangular_gradient_part(
                                &gradient, row, col,
                            ),
                    );
                }
            }
            workspace.set_row_gradient(
                scalar_index,
                row_index,
                weight
                    * <F::ParamSpec as LocationCholeskyScalarSpec<F, D>>::scalar_gradient_part(
                        &gradient,
                    ),
            );
        }

        for component in 0..D {
            let predictor = self.0.component(component).unwrap();
            let range = self.0.component_range(component).unwrap();
            let (row_gradient, local_gradient) =
                workspace.row_gradient_and_local_gradient_mut(component, range.len());
            predictor.add_gradient(row_gradient, &beta[range.clone()], local_gradient);
            add_into(&mut grad[range], local_gradient);
        }
        for row in 0..D {
            for col in 0..=row {
                let predictor = self.1.entry(row, col).unwrap();
                let range = self.1.entry_range(row, col).unwrap();
                let (row_gradient, local_gradient) = workspace.row_gradient_and_local_gradient_mut(
                    lower_workspace_index::<PLower, D>(row, col),
                    range.len(),
                );
                predictor.add_gradient(row_gradient, &beta[range.clone()], local_gradient);
                add_into(&mut grad[range], local_gradient);
            }
        }
        let beta_block = &beta[self.2.range()];
        let (row_gradient, local_gradient) =
            workspace.row_gradient_and_local_gradient_mut(scalar_index, self.2.len());
        self.2
            .x()
            .add_gradient(row_gradient, beta_block, local_gradient);
        add_into(&mut grad[self.2.range()], local_gradient);

        loss += self.0.penalty().value(&beta[self.0.range()]);
        self.0
            .penalty()
            .add_gradient(&beta[self.0.range()], &mut grad[self.0.range()]);
        loss += self.1.penalty().value(&beta[self.1.range()]);
        self.1
            .penalty()
            .add_gradient(&beta[self.1.range()], &mut grad[self.1.range()]);
        loss += self.2.penalty().value(beta_block);
        self.2
            .penalty()
            .add_gradient(beta_block, &mut grad[self.2.range()]);

        loss
    }

    fn block_ranges(&self) -> Vec<Range<usize>> {
        vec![self.0.range(), self.1.range(), self.2.range()]
    }

    fn parameter_layout(&self) -> ParameterLayout {
        ParameterLayout::new(vec![
            ParameterSlice {
                name: PVector::NAME,
                range: self.0.range(),
            },
            ParameterSlice {
                name: PLower::NAME,
                range: self.1.range(),
            },
            ParameterSlice {
                name: PScalar::NAME,
                range: self.2.range(),
            },
        ])
    }

    fn parameter_descriptors(&self) -> Vec<ParameterDescriptor> {
        let mut descriptors = Vec::new();
        <Self as GamlssBlocks<F>>::visit_parameter_descriptors(self, |_, descriptor| {
            descriptors.push(descriptor);
        });
        descriptors
    }

    fn parameter_slice_count(&self) -> usize {
        3
    }

    fn parameter_slice_matches(
        &self,
        index: usize,
        name: &'static str,
        range: Range<usize>,
    ) -> bool {
        match index {
            0 => name == PVector::NAME && range == self.0.range(),
            1 => name == PLower::NAME && range == self.1.range(),
            2 => name == PScalar::NAME && range == self.2.range(),
            _ => false,
        }
    }

    fn visit_parameter_slices<V>(&self, mut visit: V)
    where
        V: FnMut(usize, &'static str, Range<usize>),
    {
        visit(0, PVector::NAME, self.0.range());
        visit(1, PLower::NAME, self.1.range());
        visit(2, PScalar::NAME, self.2.range());
    }

    fn visit_parameter_descriptors<V>(&self, mut visit: V)
    where
        V: FnMut(usize, ParameterDescriptor),
    {
        let scalar_index = D + LowerTriangularParameterBlock::<PLower, D, (), ()>::packed_len()
            .expect("D * (D + 1) / 2 must fit");
        visit_vector_parameter_streams(&self.0, 0, |stream| {
            visit(stream.index, stream.into_descriptor());
        });
        visit_lower_triangular_parameter_streams(&self.1, D, |stream| {
            visit(stream.index, stream.into_descriptor());
        });
        visit_scalar_parameter_stream(&self.2, scalar_index, |stream| {
            visit(stream.index, stream.into_descriptor());
        });
    }

    fn has_same_parameter_layout<Other>(&self, other: &Other) -> bool
    where
        Other: GamlssBlocks<F>,
    {
        other.parameter_slice_count() == 3
            && other.parameter_slice_matches(0, PVector::NAME, self.0.range())
            && other.parameter_slice_matches(1, PLower::NAME, self.1.range())
            && other.parameter_slice_matches(2, PScalar::NAME, self.2.range())
    }
}

#[inline]
fn with_validated_global_penalties<O, GP>(
    objective: O,
    penalties: GP,
    dim: usize,
) -> Result<WithGlobalPenalties<O, GP>, ModelError>
where
    GP: GlobalPenalty,
{
    penalties.validate(dim)?;
    Ok(WithGlobalPenalties {
        objective,
        penalties,
    })
}

fn structured_eta_row<F, const D: usize, PVector, PLower, XVector, XLower, PenVector, PenLower>(
    blocks: &(
        VectorParameterBlock<PVector, D, XVector, PenVector>,
        LowerTriangularParameterBlock<PLower, D, XLower, PenLower>,
    ),
    beta: &[f64],
    row: usize,
) -> F::Eta
where
    F: Family,
    F::ParamSpec: LocationCholeskySpec<F, D>,
    XVector: PredictorBlock,
    XLower: PredictorBlock,
{
    let mut vector = [0.0; D];
    let mut lower = [[0.0; D]; D];

    for (component, value) in vector.iter_mut().enumerate() {
        let predictor = blocks
            .0
            .component(component)
            .expect("validated vector block has D components");
        let range = blocks
            .0
            .component_range(component)
            .expect("validated vector component has a coefficient range");
        *value = predictor.eta_row(row, &beta[range]);
    }

    for (matrix_row, row_values) in lower.iter_mut().enumerate() {
        for (matrix_col, value) in row_values.iter_mut().take(matrix_row + 1).enumerate() {
            let predictor = blocks
                .1
                .entry(matrix_row, matrix_col)
                .expect("validated lower-triangular block has this entry");
            let range = blocks
                .1
                .entry_range(matrix_row, matrix_col)
                .expect("validated lower-triangular entry has a coefficient range");
            *value = predictor.eta_row(row, &beta[range]);
        }
    }

    <F::ParamSpec as LocationCholeskySpec<F, D>>::eta_from_vector_lower(vector, lower)
}

#[allow(clippy::type_complexity)]
fn location_scale_partial_corr_eta_row<
    F,
    const D: usize,
    PLocation,
    PScale,
    PCorr,
    XLocation,
    XScale,
    XCorr,
    PenLocation,
    PenScale,
    PenCorr,
>(
    blocks: &(
        VectorParameterBlock<PLocation, D, XLocation, PenLocation>,
        VectorParameterBlock<PScale, D, XScale, PenScale>,
        StrictLowerTriangularParameterBlock<PCorr, D, XCorr, PenCorr>,
    ),
    beta: &[f64],
    row: usize,
) -> F::Eta
where
    F: Family,
    F::ParamSpec: LocationScalePartialCorrSpec<F, D>,
    XLocation: PredictorBlock,
    XScale: PredictorBlock,
    XCorr: PredictorBlock,
{
    let mut location = [0.0; D];
    let mut scale = [0.0; D];
    let mut partial_corr = [[0.0; D]; D];

    for component in 0..D {
        let predictor = blocks.0.component(component).unwrap();
        let range = blocks.0.component_range(component).unwrap();
        location[component] = predictor.eta_row(row, &beta[range]);

        let predictor = blocks.1.component(component).unwrap();
        let range = blocks.1.component_range(component).unwrap();
        scale[component] = predictor.eta_row(row, &beta[range]);
    }

    for (matrix_row, row_values) in partial_corr.iter_mut().enumerate() {
        for (matrix_col, value) in row_values.iter_mut().take(matrix_row).enumerate() {
            let predictor = blocks.2.entry(matrix_row, matrix_col).unwrap();
            let range = blocks.2.entry_range(matrix_row, matrix_col).unwrap();
            *value = predictor.eta_row(row, &beta[range]);
        }
    }

    <F::ParamSpec as LocationScalePartialCorrSpec<F, D>>::eta_from_location_scale_partial_corr(
        location,
        scale,
        partial_corr,
    )
}

#[allow(clippy::type_complexity)]
fn simplex_precision_eta_row<
    F,
    const D: usize,
    PMean,
    PPrecision,
    LPrecision,
    XMean,
    XPrecision,
    PenMean,
    PenPrecision,
>(
    blocks: &(
        SimplexLogitParameterBlock<PMean, D, XMean, PenMean>,
        ParameterBlock<PPrecision, LPrecision, XPrecision, PenPrecision>,
    ),
    beta: &[f64],
    row: usize,
) -> F::Eta
where
    F: Family,
    F::ParamSpec: MeanPrecisionSimplexSpec<F, D>,
    XMean: PredictorBlock,
    XPrecision: PredictorBlock,
{
    let mut logits = [0.0; D];
    for (component, value) in logits.iter_mut().take(D.saturating_sub(1)).enumerate() {
        let predictor = blocks.0.logit(component).unwrap();
        let range = blocks.0.logit_range(component).unwrap();
        *value = predictor.eta_row(row, &beta[range]);
    }
    let precision = blocks.1.x().eta_row(row, &beta[blocks.1.range()]);
    <F::ParamSpec as MeanPrecisionSimplexSpec<F, D>>::eta_from_simplex_logits_precision(
        logits, precision,
    )
}

#[allow(clippy::type_complexity)]
fn location_cholesky_scalar_eta_row<
    F,
    const D: usize,
    PVector,
    PLower,
    PScalar,
    LScalar,
    XVector,
    XLower,
    XScalar,
    PenVector,
    PenLower,
    PenScalar,
>(
    blocks: &(
        VectorParameterBlock<PVector, D, XVector, PenVector>,
        LowerTriangularParameterBlock<PLower, D, XLower, PenLower>,
        ParameterBlock<PScalar, LScalar, XScalar, PenScalar>,
    ),
    beta: &[f64],
    row: usize,
) -> F::Eta
where
    F: Family,
    F::ParamSpec: LocationCholeskyScalarSpec<F, D>,
    XVector: PredictorBlock,
    XLower: PredictorBlock,
    XScalar: PredictorBlock,
{
    let mut vector = [0.0; D];
    let mut lower = [[0.0; D]; D];

    for (component, value) in vector.iter_mut().enumerate() {
        let predictor = blocks.0.component(component).unwrap();
        let range = blocks.0.component_range(component).unwrap();
        *value = predictor.eta_row(row, &beta[range]);
    }

    for (matrix_row, row_values) in lower.iter_mut().enumerate() {
        for (matrix_col, value) in row_values.iter_mut().take(matrix_row + 1).enumerate() {
            let predictor = blocks.1.entry(matrix_row, matrix_col).unwrap();
            let range = blocks.1.entry_range(matrix_row, matrix_col).unwrap();
            *value = predictor.eta_row(row, &beta[range]);
        }
    }

    let scalar = blocks.2.x().eta_row(row, &beta[blocks.2.range()]);
    <F::ParamSpec as LocationCholeskyScalarSpec<F, D>>::eta_from_vector_lower_scalar(
        vector, lower, scalar,
    )
}

fn validate_vector_parameter_block<P, const D: usize, X, Penalty>(
    block: &VectorParameterBlock<P, D, X, Penalty>,
    nobs: usize,
) -> Result<(), ModelError>
where
    P: ParameterName,
    X: PredictorBlock,
{
    block.try_range()?;
    for component in 0..D {
        let predictor = block
            .component(component)
            .expect("vector block has D components");
        predictor.validate()?;
        validate_block_rows(P::NAME, predictor.nrows(), nobs)?;
    }
    Ok(())
}

fn validate_lower_triangular_parameter_block<P, const D: usize, X, Penalty>(
    block: &LowerTriangularParameterBlock<P, D, X, Penalty>,
    nobs: usize,
) -> Result<(), ModelError>
where
    P: ParameterName,
    X: PredictorBlock,
{
    block.try_range()?;
    let expected = LowerTriangularParameterBlock::<P, D, X, Penalty>::packed_len().ok_or(
        ModelError::ArithmeticOverflow {
            context: "lower-triangular predictor count",
        },
    )?;
    if block.entries().len() != expected {
        return Err(ModelError::InvalidParameter {
            parameter: P::NAME,
            expected: "D * (D + 1) / 2 predictor blocks",
        });
    }
    for predictor in block.entries() {
        predictor.validate()?;
        validate_block_rows(P::NAME, predictor.nrows(), nobs)?;
    }
    Ok(())
}

fn validate_strict_lower_triangular_parameter_block<P, const D: usize, X, Penalty>(
    block: &StrictLowerTriangularParameterBlock<P, D, X, Penalty>,
    nobs: usize,
) -> Result<(), ModelError>
where
    P: ParameterName,
    X: PredictorBlock,
{
    block.try_range()?;
    let expected = StrictLowerTriangularParameterBlock::<P, D, X, Penalty>::packed_len().ok_or(
        ModelError::ArithmeticOverflow {
            context: "strict-lower predictor count",
        },
    )?;
    if block.entries().len() != expected {
        return Err(ModelError::InvalidParameter {
            parameter: P::NAME,
            expected: "D * (D - 1) / 2 predictor blocks",
        });
    }
    for predictor in block.entries() {
        predictor.validate()?;
        validate_block_rows(P::NAME, predictor.nrows(), nobs)?;
    }
    Ok(())
}

fn validate_simplex_logit_parameter_block<P, const D: usize, X, Penalty>(
    block: &SimplexLogitParameterBlock<P, D, X, Penalty>,
    nobs: usize,
) -> Result<(), ModelError>
where
    P: ParameterName,
    X: PredictorBlock,
{
    block.try_range()?;
    let expected = SimplexLogitParameterBlock::<P, D, X, Penalty>::free_len().ok_or(
        ModelError::InvalidParameter {
            parameter: P::NAME,
            expected: "D >= 1",
        },
    )?;
    if block.logits().len() != expected {
        return Err(ModelError::InvalidParameter {
            parameter: P::NAME,
            expected: "D - 1 predictor blocks",
        });
    }
    for predictor in block.logits() {
        predictor.validate()?;
        validate_block_rows(P::NAME, predictor.nrows(), nobs)?;
    }
    Ok(())
}

fn visit_scalar_parameter_stream<P, L, X, Penalty, V>(
    block: &ParameterBlock<P, L, X, Penalty>,
    index: usize,
    mut visit: V,
) where
    P: ParameterName,
    V: FnMut(ParameterStream),
{
    visit(ParameterStream {
        index,
        role: P::NAME,
        path_axis: None,
        range: block.range(),
    });
}

fn visit_vector_parameter_streams<P, const D: usize, X, Penalty, V>(
    block: &VectorParameterBlock<P, D, X, Penalty>,
    start_index: usize,
    mut visit: V,
) where
    P: ParameterName,
    X: PredictorBlock,
    V: FnMut(ParameterStream),
{
    for component in 0..D {
        visit(ParameterStream {
            index: start_index + component,
            role: P::NAME,
            path_axis: Some(ParameterAxis::Vector { component }),
            range: block
                .component_range(component)
                .expect("validated vector component has a coefficient range"),
        });
    }
}

fn visit_lower_triangular_parameter_streams<P, const D: usize, X, Penalty, V>(
    block: &LowerTriangularParameterBlock<P, D, X, Penalty>,
    start_index: usize,
    mut visit: V,
) where
    P: ParameterName,
    X: PredictorBlock,
    V: FnMut(ParameterStream),
{
    for row in 0..D {
        for col in 0..=row {
            let packed_index =
                LowerTriangularParameterBlock::<P, D, (), ()>::packed_index(row, col)
                    .expect("row and col are valid lower-triangular indices");
            visit(ParameterStream {
                index: start_index + packed_index,
                role: P::NAME,
                path_axis: Some(ParameterAxis::Lower { row, col }),
                range: block
                    .entry_range(row, col)
                    .expect("validated lower-triangular entry has a coefficient range"),
            });
        }
    }
}

fn visit_strict_lower_triangular_parameter_streams<P, const D: usize, X, Penalty, V>(
    block: &StrictLowerTriangularParameterBlock<P, D, X, Penalty>,
    start_index: usize,
    mut visit: V,
) where
    P: ParameterName,
    X: PredictorBlock,
    V: FnMut(ParameterStream),
{
    for row in 0..D {
        for col in 0..row {
            let packed_index =
                StrictLowerTriangularParameterBlock::<P, D, (), ()>::packed_index(row, col)
                    .expect("row and col are valid strict-lower indices");
            visit(ParameterStream {
                index: start_index + packed_index,
                role: P::NAME,
                path_axis: Some(ParameterAxis::StrictLower { row, col }),
                range: block
                    .entry_range(row, col)
                    .expect("validated strict-lower entry has a coefficient range"),
            });
        }
    }
}

fn visit_simplex_logit_parameter_streams<P, const D: usize, X, Penalty, V>(
    block: &SimplexLogitParameterBlock<P, D, X, Penalty>,
    start_index: usize,
    mut visit: V,
) where
    P: ParameterName,
    X: PredictorBlock,
    V: FnMut(ParameterStream),
{
    for class in 0..D.saturating_sub(1) {
        visit(ParameterStream {
            index: start_index + class,
            role: P::NAME,
            path_axis: Some(ParameterAxis::SimplexLogit { class }),
            range: block
                .logit_range(class)
                .expect("validated simplex logit has a coefficient range"),
        });
    }
}

const fn structured_scalar_count<P, const D: usize>() -> usize {
    D + LowerTriangularParameterBlock::<P, D, (), ()>::packed_len()
        .expect("D * (D + 1) / 2 must fit")
}

fn lower_workspace_index<P, const D: usize>(row: usize, col: usize) -> usize {
    D + LowerTriangularParameterBlock::<P, D, (), ()>::packed_index(row, col)
        .expect("row and col are valid lower-triangular indices")
}

const fn strict_lower_workspace_count<P, const D: usize>() -> usize {
    StrictLowerTriangularParameterBlock::<P, D, (), ()>::packed_len()
        .expect("D * (D - 1) / 2 must fit")
}

fn partial_corr_workspace_index<P, const D: usize>(row: usize, col: usize) -> usize {
    D + D
        + StrictLowerTriangularParameterBlock::<P, D, (), ()>::packed_index(row, col)
            .expect("row and col are valid strict-lower indices")
}

const fn repeated_scalar_workspace_index(
    component: usize,
    parameter: usize,
    arity: usize,
) -> usize {
    component * arity + parameter
}

fn repeated_parameter_array_range<P, L, X, Penalty, const D: usize>(
    blocks: &[ParameterBlock<P, L, X, Penalty>; D],
) -> Range<usize> {
    match (blocks.first(), blocks.last()) {
        (Some(first), Some(last)) => first.range().start..last.range().end,
        _ => 0..0,
    }
}

fn validate_contiguous_repeated_parameter_ranges<P, L, X, Penalty, const D: usize>(
    parameter: &'static str,
    blocks: &[ParameterBlock<P, L, X, Penalty>; D],
) -> Result<(), ModelError>
where
    P: ParameterName,
{
    for pair in blocks.windows(2) {
        let previous = pair[0].try_range()?;
        let current = pair[1].try_range()?;
        if previous.end != current.start {
            return Err(ModelError::InvalidParameter {
                parameter,
                expected: "ordered, contiguous coefficient ranges for repeated components",
            });
        }
    }
    Ok(())
}

/// Macro that generates a [`GamlssBlocks`] implementation for tuple parameter
/// blocks.
///
/// Takes the arity `K`, lists of parameter types, link types, design types and
/// penalty types, plus internal variable names. Produces a zero-cost
/// implementation of `train_nll`, `value_gradient_into_workspace`,
/// `penalty_value` and helper methods without dynamic dispatch.
macro_rules! impl_gamlss_blocks {
    (
        $k:literal;
        params = ($($param:ident),+);
        links = ($($link:ident),+);
        designs = ($($design:ident),+);
        penalties = ($($penalty:ident),+);
        blocks = ($($block:ident),+);
        beta_blocks = ($($beta_block:ident),+);
        row_gradients = ($($row_gradient:ident),+);
        local_grads = ($($local_grad:ident),+);
        indices = ($($idx:tt),+)
    ) => {
        impl<F, $($param, $link, $design, $penalty,)+> GamlssBlocks<F>
            for ($(ParameterBlock<$param, $link, $design, $penalty>,)+)
        where
            F: InitialEtaFromObservations<$k>,
            F::ParamSpec: ScalarParamSpec<F, $k, Params = ($($param,)+), Links = ($($link,)+)>,
            F::Eta: ParameterParts<$k>,
            F::GradientEta: ParameterParts<$k>,
            $($param: ParameterName,)+
            $($link: crate::Link<f64>,)+
            $($design: PredictorBlock,)+
            $($penalty: Penalty,)+
        {
            fn nrows(&self) -> usize {
                PredictorBlock::nrows(self.0.x())
            }

            fn len(&self) -> usize {
                <Self as GamlssBlocks<F>>::try_len(self)
                    .expect("validated parameter block layout must fit in usize")
            }

            fn try_len(&self) -> Result<usize, ModelError> {
                let mut len = 0;
                $(
                    let end = self.$idx.offset().checked_add(self.$idx.len()).ok_or(
                        ModelError::BlockRangeOverflow {
                            parameter: <$param as ParameterName>::NAME,
                            offset: self.$idx.offset(),
                            len: self.$idx.len(),
                        },
                    )?;
                    len = len.max(end);
                )+
                Ok(len)
            }

            fn validate(&self, y_len: usize) -> Result<(), ModelError> {
                $(
                    self.$idx.x().validate()?;
                    validate_block_rows(
                        <$param as ParameterName>::NAME,
                        PredictorBlock::nrows(self.$idx.x()),
                        y_len,
                    )?;
                    self.$idx.penalty().validate_dim(self.$idx.len())?;
                )+

                let ranges = [$((
                    <$param as ParameterName>::NAME,
                    self.$idx.try_range()?,
                ),)+];
                validate_non_overlapping_ranges(&ranges)?;

                Ok(())
            }

            #[allow(clippy::suboptimal_flops)]
            fn train_nll<'obs, Obs>(
                &self,
                family: &F,
                obs: &'obs Obs,
                beta: &[f64],
            ) -> f64
            where
                Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
            {
                $(let $block = &self.$idx;)+
                $(let $beta_block = &beta[$block.range()];)+
                let mut loss = 0.0;
                let mut family_workspace = family.workspace();

                for row in 0..obs.len() {
                    let weight = obs.weight_at(row);
                    if weight == 0.0 {
                        continue;
                    }
                    let observation = obs.observation_at(row);
                    let eta = F::Eta::from_array([$($block.x().eta_row(row, $beta_block),)+]);
                    loss += weight * family.nll_eta(observation, &eta, &mut family_workspace);
                }

                loss
            }

            fn eta_row(&self, beta: &[f64], row: usize) -> F::Eta
            where
                F: Family,
            {
                $(let $block = &self.$idx;)+
                $(let $beta_block = &beta[$block.range()];)+
                F::Eta::from_array([$($block.x().eta_row(row, $beta_block),)+])
            }

            fn penalty_value(&self, beta: &[f64]) -> f64 {
                $(let $block = &self.$idx;)+
                $(let $beta_block = &beta[$block.range()];)+

                0.0 $(+ $block.penalty().value($beta_block))+
            }

            fn initial_parameters<'obs, Obs>(&self, family: &F, obs: &'obs Obs) -> Vec<f64>
            where
                Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
            {
                <Self as GamlssBlocks<F>>::try_initial_parameters(self, family, obs)
                    .expect("validated parameter block layout must fit in usize")
            }

            fn try_initial_parameters<'obs, Obs>(
                &self,
                family: &F,
                obs: &'obs Obs,
            ) -> Result<Vec<f64>, ModelError>
            where
                Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
            {
                let eta = family.initial_eta_from_observations(obs);
                let mut beta = vec![0.0; <Self as GamlssBlocks<F>>::try_len(self)?];
                $(
                    let $block = &self.$idx;
                    let value = eta.part($idx);
                    if value.is_finite() {
                        $block
                            .x()
                            .set_constant_start(value, &mut beta[$block.range()]);
                    }
                )+
                Ok(beta)
            }

            fn add_penalty_gradient(&self, beta: &[f64], grad: &mut [f64]) {
                $(let $block = &self.$idx;)+
                $(let $beta_block = &beta[$block.range()];)+

                $(
                    $block.penalty().add_gradient(
                        $beta_block,
                        &mut grad[$block.offset()..$block.offset() + $block.len()],
                    );
                )+
            }

            fn gradient_workspace(&self, y_len: usize) -> GradientWorkspace {
                let mut workspace = GradientWorkspace::new();
                workspace.prepare($k);
                $(
                    workspace.prepare_row_gradient($idx, y_len);
                    let _ = workspace.local_gradient_mut($idx, self.$idx.len());
                )+
                workspace
            }

            #[allow(clippy::suboptimal_flops)]
            fn value_gradient_into_workspace<'obs, Obs>(
                &self,
                family: &F,
                obs: &'obs Obs,
                beta: &[f64],
                grad: &mut [f64],
                family_workspace: &mut F::Workspace,
                workspace: &mut GradientWorkspace,
            ) -> f64
            where
                Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
            {
                $(let $block = &self.$idx;)+
                $(let $beta_block = &beta[$block.range()];)+
                workspace.prepare($k);
                $(workspace.prepare_row_gradient($idx, obs.len());)+
                let mut loss = 0.0;

                for row in 0..obs.len() {
                    let weight = obs.weight_at(row);
                    if weight == 0.0 {
                        $(workspace.set_row_gradient($idx, row, 0.0);)+
                        continue;
                    }
                    let observation = obs.observation_at(row);
                    let eta = F::Eta::from_array([$($block.x().eta_row(row, $beta_block),)+]);
                    let (nll, gradient) =
                        family.nll_and_gradient_eta(observation, &eta, family_workspace);
                    loss += weight * nll;
                    $(workspace.set_row_gradient($idx, row, weight * gradient.part($idx));)+
                }

                $(
                    loss += $block.penalty().value($beta_block);
                    let ($row_gradient, $local_grad) =
                        workspace.row_gradient_and_local_gradient_mut($idx, $block.len());
                    $block.x().add_gradient($row_gradient, $beta_block, $local_grad);
                    $block.penalty().add_gradient($beta_block, $local_grad);
                    add_into(&mut grad[$block.offset()..$block.offset() + $block.len()], $local_grad);
                )+

                loss
            }

            fn block_ranges(&self) -> Vec<Range<usize>> {
                vec![$(self.$idx.range(),)+]
            }

            fn visit_block_ranges<V>(&self, mut visit: V)
            where
                V: FnMut(usize, Range<usize>),
            {
                $(
                    visit($idx, self.$idx.range());
                )+
            }

            fn parameter_layout(&self) -> ParameterLayout {
                ParameterLayout::new(vec![$(
                    ParameterSlice {
                        name: <$param as ParameterName>::NAME,
                        range: self.$idx.range(),
                    },
                )+])
            }

            #[doc(hidden)]
            fn parameter_slice_count(&self) -> usize {
                $k
            }

            #[doc(hidden)]
            fn parameter_slice_matches(
                &self,
                index: usize,
                name: &'static str,
                range: Range<usize>,
            ) -> bool {
                match index {
                    $(
                        $idx => name == <$param as ParameterName>::NAME
                            && range == self.$idx.range(),
                    )+
                    _ => false,
                }
            }

            #[doc(hidden)]
            fn visit_parameter_slices<V>(&self, mut visit: V)
            where
                V: FnMut(usize, &'static str, Range<usize>),
            {
                $(
                    visit($idx, <$param as ParameterName>::NAME, self.$idx.range());
                )+
            }

            #[doc(hidden)]
            fn visit_parameter_descriptors<V>(&self, mut visit: V)
            where
                V: FnMut(usize, ParameterDescriptor),
            {
                $(
                    visit_scalar_parameter_stream(&self.$idx, $idx, |stream| {
                        visit(stream.index, stream.into_descriptor());
                    });
                )+
            }

            #[doc(hidden)]
            fn has_same_parameter_layout<Other>(&self, other: &Other) -> bool
            where
                Other: GamlssBlocks<F>,
            {
                other.parameter_slice_count() == $k
                    $(&& other.parameter_slice_matches(
                        $idx,
                        <$param as ParameterName>::NAME,
                        self.$idx.range(),
                    ))+
            }
        }
    };
}

impl_gamlss_blocks!(
    1;
    params = (P1);
    links = (L1);
    designs = (X1);
    penalties = (Pen1);
    blocks = (block1);
    beta_blocks = (beta1);
    row_gradients = (row_gradient1);
    local_grads = (grad1);
    indices = (0)
);

impl_gamlss_blocks!(
    2;
    params = (P1, P2);
    links = (L1, L2);
    designs = (X1, X2);
    penalties = (Pen1, Pen2);
    blocks = (block1, block2);
    beta_blocks = (beta1, beta2);
    row_gradients = (row_gradient1, row_gradient2);
    local_grads = (grad1, grad2);
    indices = (0, 1)
);

impl_gamlss_blocks!(
    3;
    params = (P1, P2, P3);
    links = (L1, L2, L3);
    designs = (X1, X2, X3);
    penalties = (Pen1, Pen2, Pen3);
    blocks = (block1, block2, block3);
    beta_blocks = (beta1, beta2, beta3);
    row_gradients = (row_gradient1, row_gradient2, row_gradient3);
    local_grads = (grad1, grad2, grad3);
    indices = (0, 1, 2)
);

impl_gamlss_blocks!(
    4;
    params = (P1, P2, P3, P4);
    links = (L1, L2, L3, L4);
    designs = (X1, X2, X3, X4);
    penalties = (Pen1, Pen2, Pen3, Pen4);
    blocks = (block1, block2, block3, block4);
    beta_blocks = (beta1, beta2, beta3, beta4);
    row_gradients = (row_gradient1, row_gradient2, row_gradient3, row_gradient4);
    local_grads = (grad1, grad2, grad3, grad4);
    indices = (0, 1, 2, 3)
);

impl_gamlss_blocks!(
    5;
    params = (P1, P2, P3, P4, P5);
    links = (L1, L2, L3, L4, L5);
    designs = (X1, X2, X3, X4, X5);
    penalties = (Pen1, Pen2, Pen3, Pen4, Pen5);
    blocks = (block1, block2, block3, block4, block5);
    beta_blocks = (beta1, beta2, beta3, beta4, beta5);
    row_gradients = (
        row_gradient1,
        row_gradient2,
        row_gradient3,
        row_gradient4,
        row_gradient5
    );
    local_grads = (grad1, grad2, grad3, grad4, grad5);
    indices = (0, 1, 2, 3, 4)
);

impl_gamlss_blocks!(
    6;
    params = (P1, P2, P3, P4, P5, P6);
    links = (L1, L2, L3, L4, L5, L6);
    designs = (X1, X2, X3, X4, X5, X6);
    penalties = (Pen1, Pen2, Pen3, Pen4, Pen5, Pen6);
    blocks = (block1, block2, block3, block4, block5, block6);
    beta_blocks = (beta1, beta2, beta3, beta4, beta5, beta6);
    row_gradients = (
        row_gradient1,
        row_gradient2,
        row_gradient3,
        row_gradient4,
        row_gradient5,
        row_gradient6
    );
    local_grads = (grad1, grad2, grad3, grad4, grad5, grad6);
    indices = (0, 1, 2, 3, 4, 5)
);

impl_gamlss_blocks!(
    7;
    params = (P1, P2, P3, P4, P5, P6, P7);
    links = (L1, L2, L3, L4, L5, L6, L7);
    designs = (X1, X2, X3, X4, X5, X6, X7);
    penalties = (Pen1, Pen2, Pen3, Pen4, Pen5, Pen6, Pen7);
    blocks = (block1, block2, block3, block4, block5, block6, block7);
    beta_blocks = (beta1, beta2, beta3, beta4, beta5, beta6, beta7);
    row_gradients = (
        row_gradient1,
        row_gradient2,
        row_gradient3,
        row_gradient4,
        row_gradient5,
        row_gradient6,
        row_gradient7
    );
    local_grads = (grad1, grad2, grad3, grad4, grad5, grad6, grad7);
    indices = (0, 1, 2, 3, 4, 5, 6)
);

impl_gamlss_blocks!(
    8;
    params = (P1, P2, P3, P4, P5, P6, P7, P8);
    links = (L1, L2, L3, L4, L5, L6, L7, L8);
    designs = (X1, X2, X3, X4, X5, X6, X7, X8);
    penalties = (Pen1, Pen2, Pen3, Pen4, Pen5, Pen6, Pen7, Pen8);
    blocks = (block1, block2, block3, block4, block5, block6, block7, block8);
    beta_blocks = (beta1, beta2, beta3, beta4, beta5, beta6, beta7, beta8);
    row_gradients = (
        row_gradient1,
        row_gradient2,
        row_gradient3,
        row_gradient4,
        row_gradient5,
        row_gradient6,
        row_gradient7,
        row_gradient8
    );
    local_grads = (grad1, grad2, grad3, grad4, grad5, grad6, grad7, grad8);
    indices = (0, 1, 2, 3, 4, 5, 6, 7)
);

macro_rules! impl_repeated_scalar_gamlss_blocks {
    (
        $k:literal;
        params = ($($param:ident),+);
        links = ($($link:ident),+);
        designs = ($($design:ident),+);
        penalties = ($($penalty:ident),+);
        blocks = ($($block:ident),+);
        beta_blocks = ($($beta_block:ident),+);
        row_gradients = ($($row_gradient:ident),+);
        local_grads = ($($local_grad:ident),+);
        indices = ($($idx:tt),+)
    ) => {
        impl<F, const D: usize, $($param, $link, $design, $penalty,)+> GamlssBlocks<F>
            for ($([ParameterBlock<$param, $link, $design, $penalty>; D],)+)
        where
            F: Family,
            F::ParamSpec: RepeatedScalarParamSpec<
                F,
                D,
                $k,
                Params = ($($param,)+),
                Links = ($($link,)+),
            >,
            <F::ParamSpec as RepeatedScalarParamSpec<F, D, $k>>::ComponentFamily:
                InitialEtaFromObservations<$k>,
            <<F::ParamSpec as RepeatedScalarParamSpec<F, D, $k>>::ComponentFamily as Family>::Eta:
                ParameterParts<$k>,
            <<F::ParamSpec as RepeatedScalarParamSpec<F, D, $k>>::ComponentFamily as Family>::GradientEta:
                ParameterParts<$k>,
            $($param: ParameterName,)+
            $($link: crate::Link<f64>,)+
            $($design: PredictorBlock,)+
            $($penalty: Penalty,)+
        {
            fn nrows(&self) -> usize {
                if D == 0 {
                    0
                } else {
                    PredictorBlock::nrows(self.0[0].x())
                }
            }

            fn len(&self) -> usize {
                <Self as GamlssBlocks<F>>::try_len(self)
                    .expect("validated repeated parameter block layout must fit in usize")
            }

            fn try_len(&self) -> Result<usize, ModelError> {
                let mut len = 0;
                $(
                    for block in &self.$idx {
                        let end = block.offset().checked_add(block.len()).ok_or(
                            ModelError::BlockRangeOverflow {
                                parameter: <$param as ParameterName>::NAME,
                                offset: block.offset(),
                                len: block.len(),
                            },
                        )?;
                        len = len.max(end);
                    }
                )+
                Ok(len)
            }

            fn validate(&self, nobs: usize) -> Result<(), ModelError> {
                if D == 0 {
                    return Err(ModelError::InvalidParameter {
                        parameter: "repeated components",
                        expected: "at least one component",
                    });
                }

                $(
                    for block in &self.$idx {
                        block.x().validate()?;
                        validate_block_rows(
                            <$param as ParameterName>::NAME,
                            PredictorBlock::nrows(block.x()),
                            nobs,
                        )?;
                        block.penalty().validate_dim(block.len())?;
                        block.try_range()?;
                    }
                )+

                let mut ranges = Vec::with_capacity(D * $k);
                $(
                    for block in &self.$idx {
                        ranges.push((<$param as ParameterName>::NAME, block.try_range()?));
                    }
                )+
                validate_non_overlapping_ranges(&ranges)?;
                $(
                    validate_contiguous_repeated_parameter_ranges(
                        <$param as ParameterName>::NAME,
                        &self.$idx,
                    )?;
                )+

                Ok(())
            }

            fn train_nll<'obs, Obs>(
                &self,
                family: &F,
                obs: &'obs Obs,
                beta: &[f64],
            ) -> f64
            where
                Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
            {
                let mut loss = 0.0;
                let mut family_workspace = family.workspace();
                for row in 0..obs.len() {
                    let weight = obs.weight_at(row);
                    if weight == 0.0 {
                        continue;
                    }
                    let components = std::array::from_fn(|component| {
                        <<F::ParamSpec as RepeatedScalarParamSpec<F, D, $k>>::ComponentFamily as Family>::Eta::from_array([
                            $(self.$idx[component].x().eta_row(row, &beta[self.$idx[component].range()]),)+
                        ])
                    });
                    let eta =
                        <F::ParamSpec as RepeatedScalarParamSpec<F, D, $k>>::eta_from_components(
                            components,
                        );
                    loss = weight.mul_add(
                        family.nll_eta(obs.observation_at(row), &eta, &mut family_workspace),
                        loss,
                    );
                }
                loss
            }

            fn eta_row(&self, beta: &[f64], row: usize) -> F::Eta {
                let components = std::array::from_fn(|component| {
                    <<F::ParamSpec as RepeatedScalarParamSpec<F, D, $k>>::ComponentFamily as Family>::Eta::from_array([
                        $(self.$idx[component].x().eta_row(row, &beta[self.$idx[component].range()]),)+
                    ])
                });
                <F::ParamSpec as RepeatedScalarParamSpec<F, D, $k>>::eta_from_components(components)
            }

            fn penalty_value(&self, beta: &[f64]) -> f64 {
                let mut value = 0.0;
                $(
                    for block in &self.$idx {
                        value += block.penalty().value(&beta[block.range()]);
                    }
                )+
                value
            }

            fn initial_parameters<'obs, Obs>(&self, family: &F, obs: &'obs Obs) -> Vec<f64>
            where
                Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
            {
                <Self as GamlssBlocks<F>>::try_initial_parameters(self, family, obs)
                    .expect("validated repeated parameter block layout must fit in usize")
            }

            fn try_initial_parameters<'obs, Obs>(
                &self,
                family: &F,
                obs: &'obs Obs,
            ) -> Result<Vec<f64>, ModelError>
            where
                Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
            {
                let mut beta = vec![0.0; <Self as GamlssBlocks<F>>::try_len(self)?];
                for component in 0..D {
                    let eta = <F::ParamSpec as RepeatedScalarParamSpec<
                        F,
                        D,
                        $k,
                    >>::initial_component_eta_from_observations(family, obs, component);
                    $(
                        let value = eta.part($idx);
                        if value.is_finite() {
                            let block = &self.$idx[component];
                            block
                                .x()
                                .set_constant_start(value, &mut beta[block.range()]);
                        }
                    )+
                }
                Ok(beta)
            }

            fn add_penalty_gradient(&self, beta: &[f64], grad: &mut [f64]) {
                $(
                    for block in &self.$idx {
                        block
                            .penalty()
                            .add_gradient(&beta[block.range()], &mut grad[block.range()]);
                    }
                )+
            }

            fn gradient_workspace(&self, nobs: usize) -> GradientWorkspace {
                let scalar_count = D * $k;
                let mut workspace = GradientWorkspace::new();
                workspace.prepare(scalar_count);
                for index in 0..scalar_count {
                    workspace.prepare_row_gradient(index, nobs);
                }
                for component in 0..D {
                    $(
                        let workspace_index = repeated_scalar_workspace_index(component, $idx, $k);
                        let _ = workspace.local_gradient_mut(workspace_index, self.$idx[component].len());
                    )+
                }
                workspace
            }

            fn value_gradient_into_workspace<'obs, Obs>(
                &self,
                family: &F,
                obs: &'obs Obs,
                beta: &[f64],
                grad: &mut [f64],
                family_workspace: &mut F::Workspace,
                workspace: &mut GradientWorkspace,
            ) -> f64
            where
                Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
            {
                let scalar_count = D * $k;
                workspace.prepare(scalar_count);
                for index in 0..scalar_count {
                    workspace.prepare_row_gradient(index, obs.len());
                }

                let mut loss = 0.0;
                for row_index in 0..obs.len() {
                    let weight = obs.weight_at(row_index);
                    if weight == 0.0 {
                        for index in 0..scalar_count {
                            workspace.set_row_gradient(index, row_index, 0.0);
                        }
                        continue;
                    }
                    let components = std::array::from_fn(|component| {
                        <<F::ParamSpec as RepeatedScalarParamSpec<F, D, $k>>::ComponentFamily as Family>::Eta::from_array([
                            $(self.$idx[component].x().eta_row(row_index, &beta[self.$idx[component].range()]),)+
                        ])
                    });
                    let eta =
                        <F::ParamSpec as RepeatedScalarParamSpec<F, D, $k>>::eta_from_components(
                            components,
                        );
                    let (nll, gradient) =
                        family.nll_and_gradient_eta(obs.observation_at(row_index), &eta, family_workspace);
                    loss = weight.mul_add(nll, loss);

                    for component in 0..D {
                        let component_gradient =
                            <F::ParamSpec as RepeatedScalarParamSpec<F, D, $k>>::gradient_component(
                                &gradient,
                                component,
                            );
                        $(
                            workspace.set_row_gradient(
                                repeated_scalar_workspace_index(component, $idx, $k),
                                row_index,
                                weight * component_gradient.part($idx),
                            );
                        )+
                    }
                }

                for component in 0..D {
                    $(
                        let block = &self.$idx[component];
                        let range = block.range();
                        let (row_gradient, local_gradient) = workspace.row_gradient_and_local_gradient_mut(
                            repeated_scalar_workspace_index(component, $idx, $k),
                            range.len(),
                        );
                        block.x().add_gradient(row_gradient, &beta[range.clone()], local_gradient);
                        block.penalty().add_gradient(&beta[range.clone()], local_gradient);
                        loss += block.penalty().value(&beta[range.clone()]);
                        add_into(&mut grad[range], local_gradient);
                    )+
                }

                loss
            }

            fn block_ranges(&self) -> Vec<Range<usize>> {
                let mut ranges = Vec::with_capacity(D * $k);
                $(
                    for block in &self.$idx {
                        ranges.push(block.range());
                    }
                )+
                ranges
            }

            fn parameter_layout(&self) -> ParameterLayout {
                ParameterLayout::new(vec![$(
                    ParameterSlice {
                        name: <$param as ParameterName>::NAME,
                        range: repeated_parameter_array_range(&self.$idx),
                    },
                )+])
            }

            fn parameter_descriptors(&self) -> Vec<ParameterDescriptor> {
                let mut descriptors = Vec::with_capacity(D * $k);
                <Self as GamlssBlocks<F>>::visit_parameter_descriptors(self, |_, descriptor| {
                    descriptors.push(descriptor);
                });
                descriptors
            }

            fn parameter_slice_count(&self) -> usize {
                $k
            }

            fn parameter_slice_matches(
                &self,
                index: usize,
                name: &'static str,
                range: Range<usize>,
            ) -> bool {
                match index {
                    $(
                        $idx => name == <$param as ParameterName>::NAME
                            && range == repeated_parameter_array_range(&self.$idx),
                    )+
                    _ => false,
                }
            }

            fn visit_parameter_slices<V>(&self, mut visit: V)
            where
                V: FnMut(usize, &'static str, Range<usize>),
            {
                $(
                    visit(
                        $idx,
                        <$param as ParameterName>::NAME,
                        repeated_parameter_array_range(&self.$idx),
                    );
                )+
            }

            fn visit_parameter_descriptors<V>(&self, mut visit: V)
            where
                V: FnMut(usize, ParameterDescriptor),
            {
                let mut index = 0;
                for component in 0..D {
                    $(
                        visit(
                            index,
                            ParameterDescriptor::component(
                                <$param as ParameterName>::NAME,
                                component,
                                self.$idx[component].range(),
                            ),
                        );
                        index += 1;
                    )+
                }
            }

            fn has_same_parameter_layout<Other>(&self, other: &Other) -> bool
            where
                Other: GamlssBlocks<F>,
            {
                other.parameter_slice_count() == $k
                    $(&& other.parameter_slice_matches(
                        $idx,
                        <$param as ParameterName>::NAME,
                        repeated_parameter_array_range(&self.$idx),
                    ))+
            }
        }
    };
}

impl_repeated_scalar_gamlss_blocks!(
    1;
    params = (P1);
    links = (L1);
    designs = (X1);
    penalties = (Pen1);
    blocks = (block1);
    beta_blocks = (beta1);
    row_gradients = (row_gradient1);
    local_grads = (grad1);
    indices = (0)
);

impl_repeated_scalar_gamlss_blocks!(
    2;
    params = (P1, P2);
    links = (L1, L2);
    designs = (X1, X2);
    penalties = (Pen1, Pen2);
    blocks = (block1, block2);
    beta_blocks = (beta1, beta2);
    row_gradients = (row_gradient1, row_gradient2);
    local_grads = (grad1, grad2);
    indices = (0, 1)
);

impl_repeated_scalar_gamlss_blocks!(
    3;
    params = (P1, P2, P3);
    links = (L1, L2, L3);
    designs = (X1, X2, X3);
    penalties = (Pen1, Pen2, Pen3);
    blocks = (block1, block2, block3);
    beta_blocks = (beta1, beta2, beta3);
    row_gradients = (row_gradient1, row_gradient2, row_gradient3);
    local_grads = (grad1, grad2, grad3);
    indices = (0, 1, 2)
);

impl_repeated_scalar_gamlss_blocks!(
    4;
    params = (P1, P2, P3, P4);
    links = (L1, L2, L3, L4);
    designs = (X1, X2, X3, X4);
    penalties = (Pen1, Pen2, Pen3, Pen4);
    blocks = (block1, block2, block3, block4);
    beta_blocks = (beta1, beta2, beta3, beta4);
    row_gradients = (row_gradient1, row_gradient2, row_gradient3, row_gradient4);
    local_grads = (grad1, grad2, grad3, grad4);
    indices = (0, 1, 2, 3)
);

/// Validates that the predictor row count matches the response length.
const fn validate_block_rows(
    parameter: &'static str,
    actual_rows: usize,
    expected_rows: usize,
) -> Result<(), ModelError> {
    if actual_rows == expected_rows {
        Ok(())
    } else {
        Err(ModelError::DesignRowMismatch {
            parameter,
            expected_rows,
            actual_rows,
        })
    }
}

/// Checks whether two ranges overlap (non-empty intersection).
const fn ranges_overlap(first: Range<usize>, second: Range<usize>) -> bool {
    first.start < second.end && second.start < first.end
}

fn validate_non_overlapping_ranges(
    ranges: &[(&'static str, Range<usize>)],
) -> Result<(), ModelError> {
    for (first_index, first) in ranges.iter().enumerate() {
        for second in ranges.iter().skip(first_index + 1) {
            if ranges_overlap(first.1.clone(), second.1.clone()) {
                return Err(ModelError::BlockOverlap {
                    first: first.0,
                    second: second.0,
                });
            }
        }
    }
    Ok(())
}

/// Element-wise adds `values` to `out`.
///
/// The caller must guarantee `out.len() == values.len()`.
fn add_into(out: &mut [f64], values: &[f64]) {
    debug_assert_eq!(out.len(), values.len());

    for (out_value, value) in out.iter_mut().zip(values) {
        *out_value += value;
    }
}

fn observation_weight_sum<Obs>(obs: &Obs) -> f64
where
    for<'row> Obs: ObservationView<'row>,
{
    (0..obs.len()).map(|row| obs.weight_at(row)).sum()
}

/// Validates the vector length (beta or gradient) and returns a typed error.
fn validate_len(name: &'static str, actual: usize, expected: usize) -> Result<(), ModelError> {
    if actual == expected {
        Ok(())
    } else if name == "gradient" {
        Err(ModelError::GradientLength { expected, actual })
    } else {
        Err(ModelError::BetaLength { expected, actual })
    }
}

fn validate_beta_and_gradient_len(
    expected: usize,
    beta: &[f64],
    grad: &[f64],
) -> Result<(), ModelError> {
    validate_len("parameters", beta.len(), expected)?;
    validate_len("gradient", grad.len(), expected)
}

const fn validate_row(row: usize, nrows: usize) -> Result<(), ModelError> {
    if row < nrows {
        Ok(())
    } else {
        Err(ModelError::RowOutOfBounds { row, nrows })
    }
}

const fn validate_output_len(expected: usize, actual: usize) -> Result<(), ModelError> {
    if actual == expected {
        Ok(())
    } else {
        Err(ModelError::ResponseLength { expected, actual })
    }
}

fn training_diagnostics_from_gradient(
    train_nll: f64,
    penalty: f64,
    grad: &[f64],
) -> TrainingDiagnostics {
    let (finite_gradient_sum_squares, nonfinite_gradient_count) =
        grad.iter().fold((0.0, 0), |(sum_squares, count), value| {
            if value.is_finite() {
                (sum_squares + value * value, count)
            } else {
                (sum_squares, count + 1)
            }
        });

    TrainingDiagnostics {
        objective: train_nll + penalty,
        train_nll,
        penalty,
        gradient_norm: finite_gradient_sum_squares.sqrt(),
        nonfinite_gradient_count,
    }
}

fn validate_prediction_blocks<F, Blocks, PBlocks>(
    expected_blocks: &Blocks,
    blocks: &PBlocks,
) -> Result<(), ModelError>
where
    F: Family,
    Blocks: GamlssBlocks<F>,
    PBlocks: GamlssBlocks<F>,
{
    blocks.validate(blocks.nrows())?;
    if expected_blocks.has_same_parameter_layout(blocks) {
        Ok(())
    } else {
        Err(ModelError::PredictionLayoutMismatch {
            expected: expected_blocks.parameter_layout(),
            got: blocks.parameter_layout(),
        })
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{
        CholeskyScale, DenseDesign, Family, Gamlss, GamlssBlocks, GlobalPenalty,
        HingeQuadraticPenalty, Identity, InitialEtaFromObservations, LinearFormBuilder,
        LinearPredictorBlock, LocationCholesky, LocationCholeskySpec,
        LowerTriangularParameterBlock, ModelError, Mu, NoPenalty, Nu, Objective, ObjectiveScale,
        ObservationView, OffsetBlock, ParameterAxis, ParameterBlock, ParameterBlocks,
        ParameterDescriptor, ParameterLayout, ParameterName, ParameterPath, ParameterSlice,
        PredictorBlock, RidgePenalty, ScalarParams, Sigma, SumBlock, Tau, VectorParameterBlock,
    };

    #[derive(Debug, Clone, Copy)]
    struct FixedSigmaNormal;

    impl Family for FixedSigmaNormal {
        type Eta = f64;
        type Theta = f64;
        type GradientEta = f64;
        type Observation<'obs> = f64;
        type Workspace = ();
        type ParamSpec = ScalarParams<(Mu,), (Identity,), 1>;
        #[inline]
        fn workspace(&self) -> Self::Workspace {}

        fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
            *eta
        }

        fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
            let residual = y - theta;
            0.5 * residual * residual
        }

        fn nll_and_gradient_eta(
            &self,
            y: f64,
            eta: &Self::Eta,
            _workspace: &mut Self::Workspace,
        ) -> (f64, Self::GradientEta) {
            let theta = self.theta(eta, _workspace);
            (self.nll(y, &theta, _workspace), eta - y)
        }
    }

    impl InitialEtaFromObservations<1> for FixedSigmaNormal {}

    #[derive(Debug, Clone, Copy)]
    struct TwoParameterMock;

    impl Family for TwoParameterMock {
        type Eta = (f64, f64);
        type Theta = (f64, f64);
        type GradientEta = (f64, f64);
        type Observation<'obs> = f64;
        type Workspace = ();
        type ParamSpec = ScalarParams<(Mu, Sigma), (Identity, Identity), 2>;
        #[inline]
        fn workspace(&self) -> Self::Workspace {}

        fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
            *eta
        }

        fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
            let first = theta.0 - y;
            let second = theta.1 - 1.0;
            f64::midpoint(first * first, second * second)
        }

        fn nll_and_gradient_eta(
            &self,
            y: f64,
            eta: &Self::Eta,
            _workspace: &mut Self::Workspace,
        ) -> (f64, Self::GradientEta) {
            let gradient = (eta.0 - y, eta.1 - 1.0);
            (self.nll(y, eta, _workspace), gradient)
        }
    }

    impl InitialEtaFromObservations<2> for TwoParameterMock {}

    #[derive(Debug, Clone, Copy)]
    struct StructuredMock;

    impl StructuredMock {
        #[allow(clippy::suboptimal_flops)]
        fn linear_prediction(eta: &([f64; 2], [[f64; 2]; 2])) -> f64 {
            eta.0[0] + 2.0 * eta.0[1] + 3.0 * eta.1[0][0] + 4.0 * eta.1[1][0] + 5.0 * eta.1[1][1]
        }
    }

    impl Family for StructuredMock {
        type Eta = ([f64; 2], [[f64; 2]; 2]);
        type Theta = Self::Eta;
        type GradientEta = Self::Eta;
        type Observation<'obs> = [f64; 2];
        type Workspace = ();
        type ParamSpec = LocationCholesky<Mu, CholeskyScale, 2>;
        #[inline]
        fn workspace(&self) -> Self::Workspace {}

        fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
            *eta
        }

        fn nll(
            &self,
            observation: Self::Observation<'_>,
            theta: &Self::Theta,
            _workspace: &mut Self::Workspace,
        ) -> f64 {
            let residual = Self::linear_prediction(theta) - observation[0];
            0.5 * residual * residual
        }

        fn nll_and_gradient_eta(
            &self,
            observation: Self::Observation<'_>,
            eta: &Self::Eta,
            _workspace: &mut Self::Workspace,
        ) -> (f64, Self::GradientEta) {
            let residual = Self::linear_prediction(eta) - observation[0];
            (
                self.nll(observation, eta, _workspace),
                (
                    [residual, 2.0 * residual],
                    [[3.0 * residual, 0.0], [4.0 * residual, 5.0 * residual]],
                ),
            )
        }
    }

    impl LocationCholeskySpec<StructuredMock, 2> for LocationCholesky<Mu, CholeskyScale, 2> {
        type VectorParameter = Mu;
        type LowerTriangularParameter = CholeskyScale;

        fn eta_from_vector_lower(
            vector: [f64; 2],
            lower: [[f64; 2]; 2],
        ) -> <StructuredMock as Family>::Eta {
            (vector, lower)
        }

        fn vector_gradient_part(
            gradient: &<StructuredMock as Family>::GradientEta,
            component: usize,
        ) -> f64 {
            gradient.0[component]
        }

        fn lower_triangular_gradient_part(
            gradient: &<StructuredMock as Family>::GradientEta,
            row: usize,
            col: usize,
        ) -> f64 {
            gradient.1[row][col]
        }

        fn initial_vector_lower_from_observations<'obs, Obs>(
            _family: &StructuredMock,
            _obs: &'obs Obs,
        ) -> ([f64; 2], [[f64; 2]; 2])
        where
            Obs: ObservationView<'obs, Observation = [f64; 2]> + 'obs,
        {
            ([1.0, 2.0], [[3.0, 0.0], [4.0, 5.0]])
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct InitializingLocation;

    impl Family for InitializingLocation {
        type Eta = f64;
        type Theta = f64;
        type GradientEta = f64;
        type Observation<'obs> = f64;
        type Workspace = ();
        type ParamSpec = ScalarParams<(Mu,), (Identity,), 1>;
        #[inline]
        fn workspace(&self) -> Self::Workspace {}

        fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
            *eta
        }

        fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
            0.5 * (theta - y) * (theta - y)
        }

        fn nll_and_gradient_eta(
            &self,
            y: f64,
            eta: &Self::Eta,
            _workspace: &mut Self::Workspace,
        ) -> (f64, Self::GradientEta) {
            (self.nll(y, eta, _workspace), *eta - y)
        }
    }

    impl InitialEtaFromObservations<1> for InitializingLocation {
        fn initial_eta_from_observations<'obs, Obs>(&self, _: &'obs Obs) -> Self::Eta
        where
            Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
        {
            2.0
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct NonFiniteInitializingLocation;

    impl Family for NonFiniteInitializingLocation {
        type Eta = f64;
        type Theta = f64;
        type GradientEta = f64;
        type Observation<'obs> = f64;
        type Workspace = ();
        type ParamSpec = ScalarParams<(Mu,), (Identity,), 1>;
        #[inline]
        fn workspace(&self) -> Self::Workspace {}

        fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
            *eta
        }

        fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
            0.5 * (theta - y) * (theta - y)
        }

        fn nll_and_gradient_eta(
            &self,
            y: f64,
            eta: &Self::Eta,
            _workspace: &mut Self::Workspace,
        ) -> (f64, Self::GradientEta) {
            (self.nll(y, eta, _workspace), *eta - y)
        }
    }

    impl InitialEtaFromObservations<1> for NonFiniteInitializingLocation {
        fn initial_eta_from_observations<'obs, Obs>(&self, _: &'obs Obs) -> Self::Eta
        where
            Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
        {
            f64::NAN
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct ShiftedObservations<'a> {
        y: &'a [f64],
        shift: f64,
        weight: f64,
    }

    impl<'row> ObservationView<'row> for ShiftedObservations<'_> {
        type Observation = f64;

        fn len(&self) -> usize {
            self.y.len()
        }

        fn observation_at(&'row self, row: usize) -> Self::Observation {
            self.y[row] + self.shift
        }

        fn weight_at(&self, _row: usize) -> f64 {
            self.weight
        }
    }

    #[test]
    fn custom_one_parameter_family_uses_generic_family_contract() {
        let y = vec![1.0, 2.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let mut model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();
        let beta = vec![1.5];
        let mut grad = vec![0.0];

        assert_relative_eq!(model.value(&beta).unwrap(), 0.25);

        model.gradient(&beta, &mut grad).unwrap();

        assert_relative_eq!(grad[0], 0.0);
    }

    #[test]
    fn structured_vector_lower_triangular_blocks_use_generic_contract() {
        let y = [[3.0, 0.0], [5.0, 0.0]];
        let n = y.len();
        let vector = VectorParameterBlock::<Mu, 2, _, _>::new(
            [
                LinearPredictorBlock::new(DenseDesign::intercept(n)),
                LinearPredictorBlock::new(DenseDesign::intercept(n)),
            ],
            RidgePenalty::new_unchecked(0.25),
            99,
        );
        let lower = LowerTriangularParameterBlock::<CholeskyScale, 2, _, _>::new(
            vec![
                LinearPredictorBlock::new(DenseDesign::intercept(n)),
                LinearPredictorBlock::new(DenseDesign::intercept(n)),
                LinearPredictorBlock::new(DenseDesign::intercept(n)),
            ],
            NoPenalty,
            99,
        );
        let blocks = ParameterBlocks::new((vector, lower));
        let model =
            Gamlss::try_new_with_observations(StructuredMock, blocks, y.as_slice()).unwrap();

        assert_eq!(model.nparams(), 5);
        assert_eq!(model.parameter_layout().slice("mu"), Some(0..2));
        assert_eq!(model.parameter_layout().slice("cholesky"), Some(2..5));
        assert_eq!(
            model.parameter_descriptors(),
            vec![
                ParameterDescriptor::vector_component("mu", 0, 0..1),
                ParameterDescriptor::vector_component("mu", 1, 1..2),
                ParameterDescriptor::lower_triangular_entry("cholesky", 0, 0, 2..3),
                ParameterDescriptor::lower_triangular_entry("cholesky", 1, 0, 3..4),
                ParameterDescriptor::lower_triangular_entry("cholesky", 1, 1, 4..5),
            ]
        );

        let mut parts = Vec::new();
        model.visit_parameter_descriptors(|index, descriptor| {
            parts.push((index, descriptor.role, descriptor.path, descriptor.range));
        });
        assert_eq!(
            parts,
            vec![
                (
                    0,
                    "mu",
                    ParameterPath::new(vec![ParameterAxis::Vector { component: 0 }]),
                    0..1
                ),
                (
                    1,
                    "mu",
                    ParameterPath::new(vec![ParameterAxis::Vector { component: 1 }]),
                    1..2
                ),
                (
                    2,
                    "cholesky",
                    ParameterPath::new(vec![ParameterAxis::Lower { row: 0, col: 0 }]),
                    2..3,
                ),
                (
                    3,
                    "cholesky",
                    ParameterPath::new(vec![ParameterAxis::Lower { row: 1, col: 0 }]),
                    3..4,
                ),
                (
                    4,
                    "cholesky",
                    ParameterPath::new(vec![ParameterAxis::Lower { row: 1, col: 1 }]),
                    4..5,
                ),
            ]
        );
        assert_eq!(
            model.initial_parameters().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0]
        );

        let beta = vec![0.2, -0.1, 0.3, 0.4, -0.2];
        let eta = model.predict_eta_row(&beta, 0).unwrap();
        assert_relative_eq!(eta.0[0], 0.2);
        assert_relative_eq!(eta.0[1], -0.1);
        assert_relative_eq!(eta.1[0][0], 0.3);
        assert_relative_eq!(eta.1[0][1], 0.0);
        assert_relative_eq!(eta.1[1][0], 0.4);
        assert_relative_eq!(eta.1[1][1], -0.2);

        let mut gradient = vec![0.0; model.nparams()];
        let value = model.try_value_gradient_into(&beta, &mut gradient).unwrap();
        assert!(value.is_finite());

        let epsilon = 1.0e-6;
        for index in 0..beta.len() {
            let mut plus = beta.clone();
            plus[index] += epsilon;
            let mut minus = beta.clone();
            minus[index] -= epsilon;
            let finite_difference = (model.try_value(&plus).unwrap()
                - model.try_value(&minus).unwrap())
                / (2.0 * epsilon);
            assert_relative_eq!(gradient[index], finite_difference, epsilon = 1.0e-6);
        }
    }

    #[test]
    fn initial_parameters_default_to_zero_for_custom_family() {
        let y = vec![1.0, 2.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();

        assert_eq!(model.initial_parameters().unwrap(), vec![0.0]);
    }

    #[test]
    fn initial_parameters_write_intercept_like_constant() {
        let y = vec![1.0, 2.0, 3.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let mut model = Gamlss::try_new(InitializingLocation, (mu,), &y).unwrap();
        let beta = model.initial_parameters().unwrap();

        assert_eq!(beta.len(), model.nparams());
        assert_eq!(beta, vec![2.0]);
        assert!(model.value(&beta).unwrap().is_finite());
    }

    #[test]
    fn initial_parameters_leave_no_intercept_design_zero() {
        let y = vec![1.0, 2.0, 3.0];
        let x = DenseDesign::from_rows(&[[0.0], [1.0], [2.0]]);
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new(InitializingLocation, (mu,), &y).unwrap();

        assert_eq!(model.initial_parameters().unwrap(), vec![0.0]);
    }

    #[test]
    fn initial_parameters_write_first_compatible_sum_term() {
        let y = vec![1.0, 2.0, 3.0];
        let first = LinearPredictorBlock::new(DenseDesign::from_rows(&[[0.0], [1.0], [2.0]]));
        let second = LinearPredictorBlock::new(DenseDesign::intercept(y.len()));
        let predictor = SumBlock::new((first, second));
        let mu = ParameterBlock::<Mu, Identity, _, _>::new(predictor, NoPenalty, 0);
        let model = Gamlss::try_new(InitializingLocation, (mu,), &y).unwrap();

        assert_eq!(model.initial_parameters().unwrap(), vec![0.0, 2.0]);
    }

    #[test]
    fn initial_parameters_account_for_sum_block_constant_baselines() {
        let y = vec![1.0, 2.0, 3.0];
        let offset = OffsetBlock::new(y.len(), 10.0);
        let intercept = LinearPredictorBlock::new(DenseDesign::intercept(y.len()));
        let predictor = SumBlock::new((offset, intercept));
        let mu = ParameterBlock::<Mu, Identity, _, _>::new(predictor, NoPenalty, 0);
        let model = Gamlss::try_new(InitializingLocation, (mu,), &y).unwrap();
        let beta = model.initial_parameters().unwrap();

        assert_eq!(beta, vec![-8.0]);
        assert_eq!(model.predict_eta(&beta).unwrap(), vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn initial_parameters_ignore_nonfinite_family_starts() {
        let y = vec![1.0, 2.0, 3.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new(NonFiniteInitializingLocation, (mu,), &y).unwrap();

        assert_eq!(model.initial_parameters().unwrap(), vec![0.0]);
    }

    #[test]
    fn model_accepts_user_defined_observation_view() {
        let y = vec![1.0, 2.0];
        let obs = ShiftedObservations {
            y: &y,
            shift: 1.0,
            weight: 0.5,
        };
        let x = DenseDesign::intercept(obs.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let mut model = Gamlss::try_new_with_observations(FixedSigmaNormal, (mu,), obs).unwrap();
        let beta = vec![2.5];
        let mut grad = vec![0.0];

        assert_relative_eq!(model.value(&beta).unwrap(), 0.125);

        model.gradient(&beta, &mut grad).unwrap();

        assert_relative_eq!(grad[0], 0.0);
    }

    #[test]
    fn custom_observation_view_rejects_invalid_weight() {
        let y = vec![1.0, 2.0];
        let obs = ShiftedObservations {
            y: &y,
            shift: 0.0,
            weight: f64::NAN,
        };
        let x = DenseDesign::intercept(obs.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);

        assert_eq!(
            Gamlss::try_new_with_observations(FixedSigmaNormal, (mu,), obs).unwrap_err(),
            ModelError::InvalidWeight { index: 0 }
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn model_borrows_response_without_copying() {
        let y = vec![1.0, 2.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();

        assert_eq!(model.obs.as_ptr(), y.as_ptr());
        assert_eq!(model.obs, y.as_slice());
        assert_eq!(model.obs.weight_at(0), 1.0);
    }

    #[test]
    fn unweighted_model_matches_unit_weights() {
        let y = vec![1.0, 2.0];
        let unit_weights = vec![1.0, 1.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x.clone(), NoPenalty, 0);
        let weighted_mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();
        let weighted =
            Gamlss::try_new_weighted(FixedSigmaNormal, (weighted_mu,), &y, &unit_weights).unwrap();
        let beta = vec![1.5];
        let mut grad = vec![0.0];
        let mut weighted_grad = vec![0.0];

        assert_relative_eq!(
            model.try_value(&beta).unwrap(),
            weighted.try_value(&beta).unwrap()
        );

        model.try_gradient_into(&beta, &mut grad).unwrap();
        weighted
            .try_gradient_into(&beta, &mut weighted_grad)
            .unwrap();

        assert_relative_eq!(grad[0], weighted_grad[0]);
    }

    #[test]
    fn model_exposes_weight_sum_and_effective_nobs() {
        let y = vec![1.0, 2.0, 3.0];
        let weights = vec![0.5, 0.0, 2.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x.clone(), NoPenalty, 0);
        let weighted_mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();
        let weighted =
            Gamlss::try_new_weighted(FixedSigmaNormal, (weighted_mu,), &y, &weights).unwrap();

        assert_relative_eq!(model.weight_sum(), 3.0);
        assert_relative_eq!(model.effective_nobs(), 3.0);
        assert_relative_eq!(weighted.weight_sum(), 2.5);
        assert_relative_eq!(weighted.effective_nobs(), 2.5);

        let mean_weighted = weighted.with_objective_scale(ObjectiveScale::Mean);
        let row0_nll = 0.5 * (1.0_f64 - 2.0).powi(2);
        let row1_nll = 0.5 * (2.0_f64 - 2.0).powi(2);
        let row2_nll = 0.5 * (3.0_f64 - 2.0).powi(2);
        let weighted_nll_sum = weights
            .iter()
            .copied()
            .zip([row0_nll, row1_nll, row2_nll])
            .map(|(weight, nll)| weight * nll)
            .sum::<f64>();

        assert_relative_eq!(
            mean_weighted.try_value(&[2.0]).unwrap(),
            weighted_nll_sum / mean_weighted.weight_sum()
        );
    }

    #[test]
    fn zero_weight_excludes_observation_from_value_and_gradient() {
        let y = vec![1.0, 10.0];
        let weights = vec![1.0, 0.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new_weighted(FixedSigmaNormal, (mu,), &y, &weights).unwrap();
        let beta = vec![1.0];
        let mut grad = vec![f64::NAN];

        assert_relative_eq!(model.try_value(&beta).unwrap(), 0.0);

        model.try_gradient_into(&beta, &mut grad).unwrap();

        assert_relative_eq!(grad[0], 0.0);
    }

    #[test]
    fn zero_weight_excludes_invalid_observation_from_value_and_gradient() {
        let y = vec![1.0, f64::NAN];
        let weights = vec![1.0, 0.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new_weighted(FixedSigmaNormal, (mu,), &y, &weights).unwrap();
        let beta = vec![1.0];
        let mut grad = vec![f64::NAN];

        assert_relative_eq!(model.try_value(&beta).unwrap(), 0.0);

        model.try_gradient_into(&beta, &mut grad).unwrap();

        assert_relative_eq!(grad[0], 0.0);
    }

    #[test]
    fn zero_weight_excludes_invalid_observation_and_design_row_from_value_and_gradient() {
        let y = vec![1.0, f64::NAN, 2.0];
        let weights = vec![1.0, 0.0, 1.0];
        let x = DenseDesign::from_row_major(3, 2, vec![1.0, 0.0, f64::NAN, f64::NAN, 1.0, 1.0])
            .unwrap();
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new_weighted(FixedSigmaNormal, (mu,), &y, &weights).unwrap();
        let beta = vec![1.0, 1.0];
        let mut grad = vec![f64::NAN, f64::NAN];

        assert_relative_eq!(model.try_value(&beta).unwrap(), 0.0);

        model.try_gradient_into(&beta, &mut grad).unwrap();

        assert!(grad.iter().all(|value| value.is_finite()));
        assert_relative_eq!(grad[0], 0.0);
        assert_relative_eq!(grad[1], 0.0);
    }

    #[test]
    fn weighted_model_rejects_invalid_weights() {
        let y = vec![1.0, 2.0];
        let short_weights = vec![1.0];
        let infinite_weights = vec![1.0, f64::INFINITY];
        let negative_weights = vec![1.0, -0.1];
        let mu =
            ParameterBlock::<Mu, Identity, _, _>::linear(DenseDesign::intercept(2), NoPenalty, 0);

        assert_eq!(
            Gamlss::try_new_weighted(FixedSigmaNormal, (mu.clone(),), &y, &short_weights,)
                .unwrap_err(),
            ModelError::WeightLength {
                expected: 2,
                actual: 1,
            }
        );
        assert_eq!(
            Gamlss::try_new_weighted(FixedSigmaNormal, (mu.clone(),), &y, &infinite_weights,)
                .unwrap_err(),
            ModelError::InvalidWeight { index: 1 }
        );
        assert_eq!(
            Gamlss::try_new_weighted(FixedSigmaNormal, (mu,), &y, &negative_weights).unwrap_err(),
            ModelError::InvalidWeight { index: 1 }
        );
    }

    #[test]
    fn scalar_response_is_permissive_but_strict_constructor_rejects_non_finite() {
        let y = vec![1.0, f64::NAN];
        let mu =
            ParameterBlock::<Mu, Identity, _, _>::linear(DenseDesign::intercept(2), NoPenalty, 0);

        Gamlss::try_new(FixedSigmaNormal, (mu.clone(),), &y).unwrap();

        assert_eq!(
            Gamlss::try_new_strict(FixedSigmaNormal, (mu,), &y).unwrap_err(),
            ModelError::InvalidObservation { index: 1 }
        );
    }

    #[test]
    fn finite_scalar_observation_adapter_rejects_non_finite() {
        let y = vec![1.0, f64::NEG_INFINITY];

        assert_eq!(
            crate::FiniteScalarObservations::new(&y).unwrap_err(),
            ModelError::InvalidObservation { index: 1 }
        );
    }

    #[test]
    fn prediction_api_returns_eta_and_theta_for_one_parameter_model() {
        let y = vec![1.0, 2.0];
        let x = DenseDesign::from_rows(&[[1.0, 0.0], [1.0, 2.0]]);
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();
        let beta = vec![0.5, 0.25];

        assert_relative_eq!(model.predict_eta_row(&beta, 1).unwrap(), 1.0);
        assert_relative_eq!(model.predict_theta_row(&beta, 1).unwrap(), 1.0);
        assert_eq!(model.predict_eta(&beta).unwrap(), vec![0.5, 1.0]);
        assert_eq!(model.predict_theta(&beta).unwrap(), vec![0.5, 1.0]);

        let mut theta = vec![f64::NAN; model.nobs()];
        model.predict_theta_into(&beta, &mut theta).unwrap();
        assert_eq!(theta, model.predict_theta(&beta).unwrap());

        let mut streamed = Vec::new();
        model
            .for_each_theta(&beta, |row, theta| streamed.push((row, theta)))
            .unwrap();
        assert_eq!(streamed, vec![(0, 0.5), (1, 1.0)]);
        assert_eq!(
            model.predict_theta_into(&beta, &mut [0.0]).unwrap_err(),
            ModelError::ResponseLength {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn prediction_api_rejects_invalid_parameter_length_and_row() {
        let y = vec![1.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();

        assert_eq!(
            model.predict_eta_row(&[], 0).unwrap_err(),
            ModelError::BetaLength {
                expected: 1,
                actual: 0,
            }
        );
        assert_eq!(
            model.predict_eta_row(&[0.0], 1).unwrap_err(),
            ModelError::RowOutOfBounds { row: 1, nrows: 1 }
        );
    }

    #[test]
    fn prediction_api_uses_compatible_blocks_for_new_rows() {
        let y = vec![1.0, 2.0];
        let train_x = DenseDesign::from_rows(&[[1.0, 0.0], [1.0, 1.0]]);
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(train_x, NoPenalty, 0);
        let model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();
        let prediction_x = DenseDesign::from_rows(&[[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]);
        let prediction_mu =
            ParameterBlock::<Mu, Identity, _, _>::linear(prediction_x, NoPenalty, 0);
        let prediction_blocks = (prediction_mu,);
        let beta = vec![0.5, 0.25];
        let prediction = model.prediction_view(&prediction_blocks).unwrap();

        assert_relative_eq!(
            model
                .predict_eta_row_with_blocks(&beta, &prediction_blocks, 1)
                .unwrap(),
            1.25
        );
        assert_eq!(prediction.nrows(), 3);
        assert_eq!(prediction.nparams(), 2);
        assert_relative_eq!(prediction.predict_eta_row(&beta, 1).unwrap(), 1.25);
        assert_relative_eq!(prediction.predict_theta_row(&beta, 1).unwrap(), 1.25);
        assert_eq!(
            model
                .predict_eta_with_blocks(&beta, &prediction_blocks)
                .unwrap(),
            vec![1.0, 1.25, 1.5]
        );
        assert_eq!(prediction.predict_eta(&beta).unwrap(), vec![1.0, 1.25, 1.5]);
        assert_eq!(
            model
                .predict_theta_with_blocks(&beta, &prediction_blocks)
                .unwrap(),
            vec![1.0, 1.25, 1.5]
        );
        assert_eq!(
            prediction.predict_theta(&beta).unwrap(),
            vec![1.0, 1.25, 1.5]
        );

        let mut theta = vec![f64::NAN; 3];
        model
            .predict_theta_with_blocks_into(&beta, &prediction_blocks, &mut theta)
            .unwrap();
        assert_eq!(theta, vec![1.0, 1.25, 1.5]);
        theta.fill(f64::NAN);
        prediction.predict_theta_into(&beta, &mut theta).unwrap();
        assert_eq!(theta, vec![1.0, 1.25, 1.5]);

        let mut streamed = Vec::new();
        model
            .for_each_theta_with_blocks(&beta, &prediction_blocks, |row, theta| {
                streamed.push((row, theta));
            })
            .unwrap();
        assert_eq!(streamed, vec![(0, 1.0), (1, 1.25), (2, 1.5)]);
        streamed.clear();
        prediction
            .for_each_theta(&beta, |row, theta| streamed.push((row, theta)))
            .unwrap();
        assert_eq!(streamed, vec![(0, 1.0), (1, 1.25), (2, 1.5)]);
    }

    #[test]
    fn prediction_api_rejects_incompatible_prediction_blocks() {
        let y = vec![1.0, 2.0];
        let train_x = DenseDesign::from_rows(&[[1.0, 0.0], [1.0, 1.0]]);
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(train_x, NoPenalty, 0);
        let model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();
        let prediction_x = DenseDesign::from_rows(&[[1.0, 2.0, 3.0]]);
        let prediction_mu =
            ParameterBlock::<Mu, Identity, _, _>::linear(prediction_x, NoPenalty, 0);
        let prediction_blocks = (prediction_mu,);

        assert_eq!(
            model
                .predict_eta_with_blocks(&[0.5, 0.25], &prediction_blocks)
                .unwrap_err(),
            ModelError::PredictionLayoutMismatch {
                expected: ParameterLayout::new(vec![ParameterSlice {
                    name: "mu",
                    range: 0..2,
                }]),
                got: ParameterLayout::new(vec![ParameterSlice {
                    name: "mu",
                    range: 0..3,
                }]),
            }
        );
        assert_eq!(
            model.prediction_view(&prediction_blocks).unwrap_err(),
            ModelError::PredictionLayoutMismatch {
                expected: ParameterLayout::new(vec![ParameterSlice {
                    name: "mu",
                    range: 0..2,
                }]),
                got: ParameterLayout::new(vec![ParameterSlice {
                    name: "mu",
                    range: 0..3,
                }]),
            }
        );
    }

    #[test]
    fn prediction_api_rejects_same_length_different_parameter_layout() {
        let y = vec![1.0, 2.0];
        let train_mu_x = DenseDesign::from_rows(&[[1.0, 0.0], [1.0, 1.0]]);
        let train_sigma_x = DenseDesign::intercept(y.len());
        let train_mu = ParameterBlock::<Mu, Identity, _, _>::linear(train_mu_x, NoPenalty, 0);
        let train_sigma =
            ParameterBlock::<Sigma, Identity, _, _>::linear(train_sigma_x, NoPenalty, 2);
        let model = Gamlss::try_new(TwoParameterMock, (train_mu, train_sigma), &y).unwrap();

        let prediction_mu_x = DenseDesign::intercept(1);
        let prediction_sigma_x = DenseDesign::from_rows(&[[1.0, 2.0]]);
        let prediction_mu =
            ParameterBlock::<Mu, Identity, _, _>::linear(prediction_mu_x, NoPenalty, 0);
        let prediction_sigma =
            ParameterBlock::<Sigma, Identity, _, _>::linear(prediction_sigma_x, NoPenalty, 1);
        let prediction_blocks = (prediction_mu, prediction_sigma);

        assert_eq!(
            model
                .predict_eta_with_blocks(&[0.5, 0.25, 0.75], &prediction_blocks)
                .unwrap_err(),
            ModelError::PredictionLayoutMismatch {
                expected: ParameterLayout::new(vec![
                    ParameterSlice {
                        name: "mu",
                        range: 0..2,
                    },
                    ParameterSlice {
                        name: "sigma",
                        range: 2..3,
                    },
                ]),
                got: ParameterLayout::new(vec![
                    ParameterSlice {
                        name: "mu",
                        range: 0..1,
                    },
                    ParameterSlice {
                        name: "sigma",
                        range: 1..3,
                    },
                ]),
            }
        );
    }

    #[test]
    fn rejects_overflowing_parameter_block_ranges() {
        let y = vec![1.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, usize::MAX);

        assert_eq!(
            Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap_err(),
            ModelError::BlockRangeOverflow {
                parameter: "mu",
                offset: usize::MAX,
                len: 1,
            }
        );
    }

    #[test]
    fn blocks_try_len_reports_overflowing_total_length() {
        let y = [1.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, usize::MAX);
        let blocks = (mu,);

        assert_eq!(
            GamlssBlocks::<FixedSigmaNormal>::try_len(&blocks).unwrap_err(),
            ModelError::BlockRangeOverflow {
                parameter: "mu",
                offset: usize::MAX,
                len: 1,
            }
        );
    }

    #[test]
    fn blocks_validate_rejects_invalid_local_penalty() {
        let y = vec![1.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(
            x,
            RidgePenalty::new_unchecked(f64::NAN),
            0,
        );

        assert_eq!(
            Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap_err(),
            ModelError::InvalidParameter {
                parameter: "ridge penalty lambda",
                expected: "finite and >= 0",
            }
        );
    }

    #[test]
    fn workspace_objective_matches_model_gradient_on_repeated_calls() {
        let y = vec![1.0, 2.0];
        let x = DenseDesign::from_rows(&[[1.0, 0.0], [1.0, 1.0]]);
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();
        let mut workspace_objective = model.clone().into_workspace_objective();

        for beta in [vec![1.0, 0.25], vec![1.5, -0.1]] {
            let mut expected_grad = vec![0.0; beta.len()];
            let mut workspace_grad = vec![0.0; beta.len()];

            model.try_gradient_into(&beta, &mut expected_grad).unwrap();
            let workspace_value = workspace_objective
                .value_gradient(&beta, &mut workspace_grad)
                .unwrap();

            assert_relative_eq!(workspace_value, model.try_value(&beta).unwrap());
            for (actual, expected) in workspace_grad.iter().zip(&expected_grad) {
                assert_relative_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn value_gradient_matches_separate_value_and_gradient() {
        let y = vec![1.0, 2.0];
        let weights = vec![0.5, 2.0];
        let x = DenseDesign::from_rows(&[[1.0, 0.0], [1.0, 1.0]]);
        let mu =
            ParameterBlock::<Mu, Identity, _, _>::linear(x, RidgePenalty::new_unchecked(0.25), 0);
        let model = Gamlss::try_new_weighted(FixedSigmaNormal, (mu,), &y, &weights).unwrap();
        let beta = vec![0.75, 0.5];
        let mut separate_grad = vec![0.0; beta.len()];
        let mut fused_grad = vec![f64::NAN; beta.len()];

        let separate_value = model.try_value(&beta).unwrap();
        model.try_gradient_into(&beta, &mut separate_grad).unwrap();
        let fused_value = model
            .try_value_gradient_into(&beta, &mut fused_grad)
            .unwrap();

        assert_relative_eq!(fused_value, separate_value);
        for (actual, expected) in fused_grad.iter().zip(&separate_grad) {
            assert_relative_eq!(actual, expected);
        }
    }

    #[test]
    fn mean_objective_scales_likelihood_but_not_penalties() {
        let y = vec![0.0, 3.0];
        let x = DenseDesign::intercept(y.len());
        let mu =
            ParameterBlock::<Mu, Identity, _, _>::linear(x, RidgePenalty::new_unchecked(0.5), 0);
        let model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y)
            .unwrap()
            .with_objective_scale(ObjectiveScale::Mean)
            .with_global_penalties(GlobalSquarePenalty { lambda: 2.0 });
        let beta = vec![1.0];
        let mut grad = vec![0.0];

        let value = model.clone().value(&beta).unwrap();
        let fused_value = model.clone().value_gradient(&beta, &mut grad).unwrap();

        assert_relative_eq!(value, 3.75);
        assert_relative_eq!(fused_value, value);
        assert_relative_eq!(grad[0], 4.5);
    }

    #[test]
    fn value_gradient_rejects_invalid_lengths() {
        let y = vec![1.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();
        let mut grad = vec![0.0];

        assert_eq!(
            model.try_value_gradient_into(&[], &mut grad).unwrap_err(),
            ModelError::BetaLength {
                expected: 1,
                actual: 0,
            }
        );

        assert_eq!(
            model.try_value_gradient_into(&[0.0], &mut []).unwrap_err(),
            ModelError::GradientLength {
                expected: 1,
                actual: 0,
            }
        );
    }

    #[derive(Debug, Clone, Copy)]
    struct SoftplusIntercept {
        nrows: usize,
    }

    impl PredictorBlock for SoftplusIntercept {
        fn nrows(&self) -> usize {
            self.nrows
        }

        fn nparams(&self) -> usize {
            1
        }

        fn eta_row(&self, _: usize, beta: &[f64]) -> f64 {
            softplus(beta[0])
        }

        #[allow(clippy::suboptimal_flops)]
        fn add_gradient(&self, scores: &[f64], beta: &[f64], grad: &mut [f64]) {
            debug_assert_eq!(grad.len(), 1);
            grad[0] += scores.iter().sum::<f64>() * sigmoid(beta[0]);
        }
    }

    #[test]
    fn sum_block_supports_user_defined_nonlinear_predictors() {
        let y = vec![1.0, 2.0];
        let linear = crate::LinearPredictorBlock::new(DenseDesign::intercept(y.len()));
        let nonlinear = SoftplusIntercept { nrows: y.len() };
        let predictor = SumBlock::new((linear, nonlinear));
        let mu = ParameterBlock::<Mu, Identity, _, _>::new(predictor, NoPenalty, 0);
        let mut model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();
        let beta = vec![0.4, -0.2];
        let eps = 1.0e-6;
        let mut grad = vec![0.0; beta.len()];

        model.gradient(&beta, &mut grad).unwrap();

        for index in 0..beta.len() {
            let mut plus = beta.clone();
            plus[index] += eps;
            let mut minus = beta.clone();
            minus[index] -= eps;
            let finite_difference =
                (model.value(&plus).unwrap() - model.value(&minus).unwrap()) / (2.0 * eps);

            assert_relative_eq!(grad[index], finite_difference, epsilon = 1.0e-6);
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct StatefulLocation {
        target_shift: f64,
    }

    impl Family for StatefulLocation {
        type Eta = f64;
        type Theta = f64;
        type GradientEta = f64;
        type Observation<'obs> = f64;
        type Workspace = ();
        type ParamSpec = ScalarParams<(Mu,), (Identity,), 1>;
        #[inline]
        fn workspace(&self) -> Self::Workspace {}

        fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
            *eta + self.target_shift
        }

        fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
            let residual = y - theta;
            0.5 * residual * residual
        }

        fn nll_and_gradient_eta(
            &self,
            y: f64,
            eta: &Self::Eta,
            _workspace: &mut Self::Workspace,
        ) -> (f64, Self::GradientEta) {
            let theta = self.theta(eta, _workspace);
            (self.nll(y, &theta, _workspace), theta - y)
        }
    }

    impl InitialEtaFromObservations<1> for StatefulLocation {}

    #[test]
    fn family_instance_state_participates_in_objective() {
        let y = vec![2.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let mut model = Gamlss::try_new(StatefulLocation { target_shift: 1.0 }, (mu,), &y).unwrap();
        let beta = vec![0.5];
        let mut grad = vec![0.0];

        assert_relative_eq!(model.value(&beta).unwrap(), 0.125);

        model.gradient(&beta, &mut grad).unwrap();

        assert_relative_eq!(grad[0], -0.5);
    }

    #[derive(Debug, Clone, Copy)]
    struct BivariateLocation;

    impl Family for BivariateLocation {
        type Eta = f64;
        type Theta = f64;
        type GradientEta = f64;
        type Observation<'obs> = [f64; 2];
        type Workspace = ();
        type ParamSpec = ScalarParams<(Mu,), (Identity,), 1>;
        #[inline]
        fn workspace(&self) -> Self::Workspace {}

        fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
            *eta
        }

        fn nll(
            &self,
            observation: Self::Observation<'_>,
            theta: &Self::Theta,
            _workspace: &mut Self::Workspace,
        ) -> f64 {
            let first = theta - observation[0];
            let second = theta - observation[1];
            f64::midpoint(first * first, second * second)
        }

        fn nll_and_gradient_eta(
            &self,
            observation: Self::Observation<'_>,
            eta: &Self::Eta,
            _workspace: &mut Self::Workspace,
        ) -> (f64, Self::GradientEta) {
            let gradient = (*eta - observation[0]) + (*eta - observation[1]);
            (self.nll(observation, eta, _workspace), gradient)
        }
    }

    impl InitialEtaFromObservations<1> for BivariateLocation {}

    #[test]
    fn model_accepts_multivariate_observation_rows() {
        let y = vec![[1.0, 3.0], [2.0, 4.0]];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let mut model =
            Gamlss::try_new_with_observations(BivariateLocation, (mu,), y.as_slice()).unwrap();
        let beta = vec![2.0];
        let mut grad = vec![0.0];

        assert_relative_eq!(model.value(&beta).unwrap(), 3.0);

        model.gradient(&beta, &mut grad).unwrap();

        assert_relative_eq!(grad[0], -2.0);
    }

    #[derive(Debug, Clone, PartialEq)]
    struct BorrowedRows {
        rows: Vec<Vec<f64>>,
    }

    impl<'row> ObservationView<'row> for BorrowedRows {
        type Observation = &'row [f64];

        fn len(&self) -> usize {
            self.rows.len()
        }

        fn observation_at(&'row self, row: usize) -> Self::Observation {
            self.rows[row].as_slice()
        }

        fn weight_at(&self, _row: usize) -> f64 {
            1.0
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct BorrowedRowMean;

    impl Family for BorrowedRowMean {
        type Eta = f64;
        type Theta = f64;
        type GradientEta = f64;
        type Observation<'obs> = &'obs [f64];
        type Workspace = ();
        type ParamSpec = ScalarParams<(Mu,), (Identity,), 1>;
        #[inline]
        fn workspace(&self) -> Self::Workspace {}

        fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
            *eta
        }

        #[allow(clippy::cast_precision_loss)]
        fn nll(
            &self,
            observation: Self::Observation<'_>,
            theta: &Self::Theta,
            _workspace: &mut Self::Workspace,
        ) -> f64 {
            let mean = observation.iter().sum::<f64>() / observation.len() as f64;
            let residual = theta - mean;
            0.5 * residual * residual
        }

        #[allow(clippy::cast_precision_loss)]
        fn nll_and_gradient_eta(
            &self,
            observation: Self::Observation<'_>,
            eta: &Self::Eta,
            _workspace: &mut Self::Workspace,
        ) -> (f64, Self::GradientEta) {
            let mean = observation.iter().sum::<f64>() / observation.len() as f64;
            (self.nll(observation, eta, _workspace), *eta - mean)
        }
    }

    impl InitialEtaFromObservations<1> for BorrowedRowMean {}

    #[test]
    fn model_accepts_borrowed_dynamic_observation_rows() {
        let obs = BorrowedRows {
            rows: vec![vec![1.0, 3.0], vec![2.0, 4.0]],
        };
        let x = DenseDesign::intercept(obs.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let mut model = Gamlss::try_new_with_observations(BorrowedRowMean, (mu,), obs).unwrap();
        let beta = vec![2.0];
        let mut grad = vec![0.0];

        assert_relative_eq!(model.value(&beta).unwrap(), 0.5);

        model.gradient(&beta, &mut grad).unwrap();

        assert_relative_eq!(grad[0], -1.0);
    }

    #[derive(Debug, Clone, Copy)]
    struct ThreeParameterMock;

    impl Family for ThreeParameterMock {
        type Eta = (f64, f64, f64);
        type Theta = (f64, f64, f64);
        type GradientEta = (f64, f64, f64);
        type Observation<'obs> = f64;
        type Workspace = ();
        type ParamSpec = ScalarParams<(Mu, Sigma, Nu), (Identity, Identity, Identity), 3>;
        #[inline]
        fn workspace(&self) -> Self::Workspace {}

        fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
            *eta
        }

        #[allow(clippy::suboptimal_flops)]
        fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
            let first = theta.0 - y;
            let second = theta.1 - 1.0;
            let third = theta.2 + 1.0;
            0.5 * (first * first + second * second + third * third)
        }

        fn nll_and_gradient_eta(
            &self,
            y: f64,
            eta: &Self::Eta,
            _workspace: &mut Self::Workspace,
        ) -> (f64, Self::GradientEta) {
            let gradient = (eta.0 - y, eta.1 - 1.0, eta.2 + 1.0);
            (self.nll(y, eta, _workspace), gradient)
        }
    }

    impl InitialEtaFromObservations<3> for ThreeParameterMock {}

    #[test]
    fn custom_three_parameter_family_uses_generic_blocks() {
        let y = vec![2.0];
        let first = ParameterBlock::<Mu, Identity, _, _>::linear(
            DenseDesign::intercept(y.len()),
            NoPenalty,
            0,
        );
        let second = ParameterBlock::<Sigma, Identity, _, _>::linear(
            DenseDesign::intercept(y.len()),
            NoPenalty,
            1,
        );
        let third = ParameterBlock::<Nu, Identity, _, _>::linear(
            DenseDesign::intercept(y.len()),
            NoPenalty,
            2,
        );
        let mut model = Gamlss::try_new(ThreeParameterMock, (first, second, third), &y).unwrap();
        let beta = vec![1.5, 0.5, -0.5];
        let mut grad = vec![0.0; 3];

        assert_relative_eq!(model.value(&beta).unwrap(), 0.375);

        model.gradient(&beta, &mut grad).unwrap();

        assert_relative_eq!(grad[0], -0.5);
        assert_relative_eq!(grad[1], -0.5);
        assert_relative_eq!(grad[2], 0.5);
    }

    #[derive(Debug, Clone, Copy)]
    struct FourParameterMock;

    impl Family for FourParameterMock {
        type Eta = (f64, f64, f64, f64);
        type Theta = (f64, f64, f64, f64);
        type GradientEta = (f64, f64, f64, f64);
        type Observation<'obs> = f64;
        type Workspace = ();
        type ParamSpec =
            ScalarParams<(Mu, Sigma, Nu, Tau), (Identity, Identity, Identity, Identity), 4>;
        #[inline]
        fn workspace(&self) -> Self::Workspace {}

        fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
            (eta.0 + 1.0, eta.1 + 2.0, eta.2 + 3.0, eta.3 + 4.0)
        }

        fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
            theta.0 + theta.1 + theta.2 + theta.3 + y
        }

        fn nll_and_gradient_eta(
            &self,
            y: f64,
            eta: &Self::Eta,
            _workspace: &mut Self::Workspace,
        ) -> (f64, Self::GradientEta) {
            (
                {
                    let theta = self.theta(eta, _workspace);
                    self.nll(y, &theta, _workspace)
                },
                (1.0, 1.0, 1.0, 1.0),
            )
        }
    }

    impl InitialEtaFromObservations<4> for FourParameterMock {}

    #[derive(Debug, Clone, Copy)]
    struct Fifth;

    impl ParameterName for Fifth {
        const NAME: &'static str = "fifth";
    }

    #[derive(Debug, Clone, Copy)]
    struct FiveParameterMock;

    impl Family for FiveParameterMock {
        type Eta = (f64, f64, f64, f64, f64);
        type Theta = (f64, f64, f64, f64, f64);
        type GradientEta = (f64, f64, f64, f64, f64);
        type Observation<'obs> = f64;
        type Workspace = ();
        type ParamSpec = ScalarParams<
            (Mu, Sigma, Nu, Tau, Fifth),
            (Identity, Identity, Identity, Identity, Identity),
            5,
        >;
        #[inline]
        fn workspace(&self) -> Self::Workspace {}

        fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
            *eta
        }

        fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
            let targets = [y, 1.0, 2.0, 3.0, 4.0];
            let values: [f64; 5] = (*theta).into();
            0.5 * values
                .iter()
                .zip(targets)
                .map(|(value, target)| {
                    let residual = value - target;
                    residual * residual
                })
                .sum::<f64>()
        }

        fn nll_and_gradient_eta(
            &self,
            y: f64,
            eta: &Self::Eta,
            _workspace: &mut Self::Workspace,
        ) -> (f64, Self::GradientEta) {
            (
                self.nll(y, eta, _workspace),
                (
                    eta.0 - y,
                    eta.1 - 1.0,
                    eta.2 - 2.0,
                    eta.3 - 3.0,
                    eta.4 - 4.0,
                ),
            )
        }
    }

    impl InitialEtaFromObservations<5> for FiveParameterMock {}

    #[test]
    fn prediction_api_returns_eta_and_theta_for_four_parameter_model() {
        let y = vec![2.0];
        let first = ParameterBlock::<Mu, Identity, _, _>::linear(
            DenseDesign::intercept(y.len()),
            NoPenalty,
            0,
        );
        let second = ParameterBlock::<Sigma, Identity, _, _>::linear(
            DenseDesign::intercept(y.len()),
            NoPenalty,
            1,
        );
        let third = ParameterBlock::<Nu, Identity, _, _>::linear(
            DenseDesign::intercept(y.len()),
            NoPenalty,
            2,
        );
        let fourth = ParameterBlock::<Tau, Identity, _, _>::linear(
            DenseDesign::intercept(y.len()),
            NoPenalty,
            3,
        );
        let model = Gamlss::try_new(FourParameterMock, (first, second, third, fourth), &y).unwrap();
        let beta = vec![0.5, 1.5, 2.5, 3.5];

        assert_eq!(
            model.predict_eta_row(&beta, 0).unwrap(),
            (0.5, 1.5, 2.5, 3.5)
        );
        assert_eq!(
            model.predict_theta_row(&beta, 0).unwrap(),
            (1.5, 3.5, 5.5, 7.5)
        );
    }

    #[test]
    fn custom_five_parameter_family_uses_generic_blocks() {
        let y = vec![2.0];
        let blocks = ParameterBlocks::new((
            intercept_block::<Mu>(y.len()),
            intercept_block::<Sigma>(y.len()),
            intercept_block::<Nu>(y.len()),
            intercept_block::<Tau>(y.len()),
            intercept_block::<Fifth>(y.len()),
        ));
        let mut model = Gamlss::try_new(FiveParameterMock, blocks, &y).unwrap();
        let beta = vec![1.5, 0.5, 1.5, 2.5, 3.5];
        let mut grad = vec![0.0; 5];

        assert_relative_eq!(model.value(&beta).unwrap(), 0.625);

        model.gradient(&beta, &mut grad).unwrap();

        assert_eq!(model.parameter_layout().slice("fifth").unwrap(), 4..5);
        assert_relative_eq!(grad[0], -0.5);
        assert_relative_eq!(grad[1], -0.5);
        assert_relative_eq!(grad[2], -0.5);
        assert_relative_eq!(grad[3], -0.5);
        assert_relative_eq!(grad[4], -0.5);
    }

    fn intercept_block<P>(
        nrows: usize,
    ) -> ParameterBlock<P, Identity, LinearPredictorBlock<DenseDesign>, NoPenalty> {
        ParameterBlock::linear(DenseDesign::intercept(nrows), NoPenalty, 99)
    }

    #[test]
    fn parameter_layout_and_unpack_use_distribution_parameter_names() {
        let y = vec![2.0];
        let first = ParameterBlock::<Mu, Identity, _, _>::linear(
            DenseDesign::intercept(y.len()),
            NoPenalty,
            0,
        );
        let second = ParameterBlock::<Sigma, Identity, _, _>::linear(
            DenseDesign::intercept(y.len()),
            NoPenalty,
            1,
        );
        let third = ParameterBlock::<Nu, Identity, _, _>::linear(
            DenseDesign::intercept(y.len()),
            NoPenalty,
            2,
        );
        let model = Gamlss::try_new(ThreeParameterMock, (first, second, third), &y).unwrap();
        let parameters = vec![1.5, 0.5, -0.5];
        let layout = model.parameter_layout();
        let unpacked = model.unpack_parameters(&parameters).unwrap();

        assert_eq!(layout.len(), 3);
        assert!(!layout.is_empty());
        assert_eq!(layout.ncoefficients(), parameters.len());
        assert_eq!(layout.slice("mu").unwrap(), 0..1);
        assert_eq!(layout.slice_of::<Mu>().unwrap(), 0..1);
        assert_eq!(layout.slice("sigma").unwrap(), 1..2);
        assert_eq!(layout.slice_of::<Sigma>().unwrap(), 1..2);
        assert_eq!(layout.slice("nu").unwrap(), 2..3);
        assert_eq!(layout.slice_of::<Nu>().unwrap(), 2..3);
        assert_eq!(unpacked.coefficients("mu").unwrap(), &[1.5]);
        assert_eq!(unpacked.coefficients_of::<Mu>().unwrap(), &[1.5]);
        assert_eq!(unpacked.block_of::<Mu>().unwrap().name, "mu");
        assert_eq!(unpacked.coefficients("sigma").unwrap(), &[0.5]);
        assert_eq!(unpacked.coefficients("nu").unwrap(), &[-0.5]);
    }

    #[test]
    fn visitor_apis_match_allocating_layout_helpers() {
        let y = vec![2.0];
        let first = ParameterBlock::<Mu, Identity, _, _>::linear(
            DenseDesign::intercept(y.len()),
            NoPenalty,
            0,
        );
        let second = ParameterBlock::<Sigma, Identity, _, _>::linear(
            DenseDesign::intercept(y.len()),
            NoPenalty,
            1,
        );
        let third = ParameterBlock::<Nu, Identity, _, _>::linear(
            DenseDesign::intercept(y.len()),
            NoPenalty,
            2,
        );
        let model = Gamlss::try_new(ThreeParameterMock, (first, second, third), &y).unwrap();

        let mut visited_ranges = Vec::new();
        model.visit_block_ranges(|index, range| visited_ranges.push((index, range)));
        assert_eq!(
            visited_ranges,
            model
                .block_ranges()
                .into_iter()
                .enumerate()
                .collect::<Vec<_>>()
        );

        let mut visited_slices = Vec::new();
        model.visit_parameter_slices(|index, name, range| {
            visited_slices.push((index, name, range));
        });
        let layout = model.parameter_layout();
        assert_eq!(
            visited_slices,
            layout
                .slices()
                .iter()
                .enumerate()
                .map(|(index, slice)| (index, slice.name, slice.range.clone()))
                .collect::<Vec<_>>()
        );

        let mut layout_visited = Vec::new();
        layout.visit_slices(|index, name, range| layout_visited.push((index, name, range)));
        assert_eq!(layout_visited, visited_slices);
    }

    #[test]
    fn training_diagnostics_report_train_nll_penalty_and_gradient_norm() {
        let y = vec![1.0, 2.0];
        let x = DenseDesign::intercept(y.len());
        let mu =
            ParameterBlock::<Mu, Identity, _, _>::linear(x, RidgePenalty::new_unchecked(0.5), 0);
        let model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();
        let parameters = vec![1.5];
        let mut grad = vec![f64::NAN; parameters.len()];
        let diagnostics = model.training_diagnostics(&parameters).unwrap();
        let diagnostics_into = model
            .training_diagnostics_into(&parameters, &mut grad)
            .unwrap();

        assert_relative_eq!(diagnostics.train_nll, 0.25);
        assert_relative_eq!(diagnostics.penalty, 1.125);
        assert_relative_eq!(diagnostics.objective, 1.375);
        assert_relative_eq!(diagnostics.gradient_norm, 1.5);
        assert_eq!(diagnostics.nonfinite_gradient_count, 0);
        assert_eq!(diagnostics_into, diagnostics);
        assert_relative_eq!(grad[0], 1.5);

        let mut workspace = model.gradient_workspace();
        grad.fill(f64::NAN);
        assert_eq!(
            model
                .training_diagnostics_into_workspace(&parameters, &mut grad, &mut workspace)
                .unwrap(),
            diagnostics
        );
        assert_relative_eq!(grad[0], 1.5);

        let mut workspace_model = model.into_workspace_objective();
        grad.fill(f64::NAN);
        assert_eq!(
            workspace_model
                .training_diagnostics_into(&parameters, &mut grad)
                .unwrap(),
            diagnostics
        );
        assert_relative_eq!(grad[0], 1.5);
    }

    #[derive(Debug, Clone, Copy)]
    struct DifferenceGlobalPenalty {
        lambda: f64,
    }

    impl GlobalPenalty for DifferenceGlobalPenalty {
        fn value(&self, beta: &[f64]) -> f64 {
            let diff = beta[0] - beta[1];
            self.lambda * diff * diff
        }

        fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
            let diff = beta[0] - beta[1];
            let slope = 2.0 * self.lambda * diff;
            grad[0] += slope;
            grad[1] -= slope;
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct GlobalSquarePenalty {
        lambda: f64,
    }

    impl GlobalPenalty for GlobalSquarePenalty {
        fn value(&self, beta: &[f64]) -> f64 {
            self.lambda * beta[0] * beta[0]
        }

        #[allow(clippy::suboptimal_flops)]
        fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
            grad[0] += 2.0 * self.lambda * beta[0];
        }
    }

    #[test]
    fn global_penalty_adds_value_and_gradient_to_full_objective() {
        let y = vec![0.0, 0.0];
        let x = DenseDesign::from_rows(&[[1.0, 0.0], [0.0, 1.0]]);
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let mut model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y)
            .unwrap()
            .with_global_penalties(DifferenceGlobalPenalty { lambda: 1.0 });
        let beta = vec![1.0, -1.0];
        let mut grad = vec![0.0; beta.len()];

        assert_relative_eq!(model.value(&beta).unwrap(), 5.0);

        assert_relative_eq!(model.value_gradient(&beta, &mut grad).unwrap(), 5.0);

        assert_relative_eq!(grad[0], 5.0);
        assert_relative_eq!(grad[1], -5.0);
    }

    #[test]
    fn try_with_global_penalties_validates_full_parameter_dimension() {
        let y = vec![0.0, 0.0];
        let x = DenseDesign::from_rows(&[[1.0, 0.0], [0.0, 1.0]]);
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();
        let penalty =
            HingeQuadraticPenalty::new(LinearFormBuilder::new().term(2, 1.0).build(), 1.0);

        assert_eq!(
            model.try_with_global_penalties(penalty).unwrap_err(),
            ModelError::PenaltyIndexOutOfBounds { index: 2, dim: 2 }
        );
    }

    #[test]
    fn try_with_global_penalties_validates_penalty_invariants() {
        let y = vec![0.0, 0.0];
        let x = DenseDesign::from_rows(&[[1.0, 0.0], [0.0, 1.0]]);
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();
        let penalty =
            HingeQuadraticPenalty::new(LinearFormBuilder::new().term(0, f64::NAN).build(), 1.0);

        assert_eq!(
            model.try_with_global_penalties(penalty).unwrap_err(),
            ModelError::InvalidParameter {
                parameter: "linear term weight",
                expected: "finite",
            }
        );
    }

    #[test]
    fn workspace_try_with_global_penalties_validates_full_parameter_dimension() {
        let y = vec![0.0, 0.0];
        let x = DenseDesign::from_rows(&[[1.0, 0.0], [0.0, 1.0]]);
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y)
            .unwrap()
            .into_workspace_objective();
        let penalty =
            HingeQuadraticPenalty::new(LinearFormBuilder::new().term(2, 1.0).build(), 1.0);

        assert_eq!(
            model.try_with_global_penalties(penalty).unwrap_err(),
            ModelError::PenaltyIndexOutOfBounds { index: 2, dim: 2 }
        );
    }

    #[test]
    fn block_objective_for_projects_mu_coefficients() {
        // Simple 2-param mock: identity link for both, NLL = 0.5 * sum of squares.
        #[derive(Debug, Clone, Copy)]
        struct TwoParamMock;

        impl Family for TwoParamMock {
            type Eta = (f64, f64);
            type Theta = (f64, f64);
            type GradientEta = (f64, f64);
            type Observation<'obs> = f64;
            type Workspace = ();
            type ParamSpec = ScalarParams<(Mu, Sigma), (Identity, Identity), 2>;
            #[inline]
            fn workspace(&self) -> Self::Workspace {}

            fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
                *eta
            }

            fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
                let first = theta.0 - y;
                let second = theta.1 - 1.0;
                f64::midpoint(first * first, second * second)
            }

            fn nll_and_gradient_eta(
                &self,
                y: f64,
                eta: &Self::Eta,
                _workspace: &mut Self::Workspace,
            ) -> (f64, Self::GradientEta) {
                let gradient = (eta.0 - y, eta.1 - 1.0);
                (self.nll(y, eta, _workspace), gradient)
            }
        }

        impl InitialEtaFromObservations<2> for TwoParamMock {}

        let y = vec![1.0, 2.0, 3.0];
        let x_mu = DenseDesign::from_rows(&[[1.0, 0.5], [1.0, 1.5], [1.0, 2.5]]);
        let x_sigma = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x_mu, NoPenalty, 0);
        let sigma = ParameterBlock::<Sigma, Identity, _, _>::linear(x_sigma, NoPenalty, 0);
        let (mu, sigma) = ParameterBlocks::new((mu, sigma));
        let mut model = Gamlss::try_new(TwoParamMock, (mu, sigma), &y).unwrap();

        let beta = vec![0.5, 0.2, 0.3];
        let nparams = model.nparams();

        // Scope the block objective to release the mutable borrow on model.
        let (mu_dim, mu_value, mu_grad) = {
            let mut mu_block = model.block_objective_for::<Mu>(beta.clone()).unwrap();
            let dim = mu_block.dim();
            let mut block_grad = vec![0.0; dim];
            let value = mu_block
                .value_gradient(&beta[..2], &mut block_grad)
                .unwrap();
            (dim, value, block_grad)
        };

        assert_eq!(mu_dim, 2);
        assert_relative_eq!(mu_value, model.value(&beta).unwrap());

        let mut full_grad = vec![0.0; nparams];
        model.gradient(&beta, &mut full_grad).unwrap();

        assert_relative_eq!(mu_grad[0], full_grad[0]);
        assert_relative_eq!(mu_grad[1], full_grad[1]);
    }

    #[test]
    fn block_objective_for_rejects_wrong_full_beta_length() {
        let y = vec![1.0, 2.0];
        let mu =
            ParameterBlock::<Mu, Identity, _, _>::linear(DenseDesign::intercept(2), NoPenalty, 0);
        let mut model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();

        assert_eq!(
            model.block_objective_for::<Mu>(Vec::new()).unwrap_err(),
            ModelError::BetaLength {
                expected: 1,
                actual: 0,
            }
        );
    }

    #[test]
    fn workspace_block_objective_for_rejects_wrong_full_beta_length() {
        let y = vec![1.0, 2.0];
        let mu =
            ParameterBlock::<Mu, Identity, _, _>::linear(DenseDesign::intercept(2), NoPenalty, 0);
        let model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();
        let mut objective = model.into_workspace_objective();

        assert_eq!(
            objective.block_objective_for::<Mu>(Vec::new()).unwrap_err(),
            ModelError::BetaLength {
                expected: 1,
                actual: 0,
            }
        );
    }

    fn softplus(value: f64) -> f64 {
        if value > 30.0 {
            value
        } else if value < -30.0 {
            value.exp()
        } else {
            value.exp().ln_1p()
        }
    }

    fn sigmoid(value: f64) -> f64 {
        if value >= 0.0 {
            1.0 / (1.0 + (-value).exp())
        } else {
            let exp_value = value.exp();
            exp_value / (1.0 + exp_value)
        }
    }
}
