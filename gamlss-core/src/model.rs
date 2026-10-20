use std::ops::Range;

use crate::{
    BlockObjective, CompilableFamily, DynamicLayoutKey, Family, GlobalPenalty, ModelError,
    Objective, ParameterBlocks, ParameterName,
};
use executor::ShapeBlocks;
use layout::UniqueParameterMatch;

pub use dynamic::DynamicParameterBlocks;
pub use layout::{
    ParameterAxis, ParameterCoefficients, ParameterDescriptor, ParameterLayout, ParameterPath,
    ParameterSlice, TrainingDiagnostics, UnpackedParameters,
};
pub use observation::{DenseRows, FiniteScalarObservations, ObservationView};
pub use prediction::PredictionView;
pub use workspace::{GradientWorkspace, ModelWorkspace, ScoreTilePolicy};

mod dynamic;
mod executor;
mod layout;
mod observation;
mod prediction;
mod workspace;

mod sealed {
    pub trait Sealed {}
}

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
///
/// The direct [`Objective`] implementation is a convenience that creates
/// scratch storage for gradient calls. Repeated optimizer evaluations should
/// use [`Self::into_workspace_objective`] so value and gradient paths reuse the
/// same [`ModelWorkspace`].
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
        blocks.validate_for(&family, nobs)?;
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
    #[must_use]
    pub fn nparams(&self) -> usize {
        self.blocks.len()
    }

    /// Returns the likelihood scaling convention used by this objective.
    #[must_use]
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
    #[must_use]
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

    /// Creates reusable objective buffers sized for this model using the
    /// default [`ScoreTilePolicy`].
    ///
    /// Use [`Self::gradient_workspace_with_policy`] when an explicit score
    /// memory or row budget is required.
    #[must_use]
    pub fn gradient_workspace(&self) -> ModelWorkspace<F>
    where
        F: Family,
    {
        self.gradient_workspace_with_policy(ScoreTilePolicy::default())
    }

    /// Creates reusable objective buffers with an explicit score-tile policy.
    ///
    /// Most callers should use [`Self::gradient_workspace`]. A custom policy is
    /// useful when profiling tile sizes or bounding worker-local score memory.
    #[must_use]
    pub fn gradient_workspace_with_policy(&self, policy: ScoreTilePolicy) -> ModelWorkspace<F>
    where
        F: Family,
    {
        ModelWorkspace::new(&self.family, self.obs.len(), |nobs| {
            self.blocks.gradient_workspace(nobs, policy)
        })
    }

    /// Wraps the model as an objective with reusable gradient buffers using the
    /// default [`ScoreTilePolicy`].
    ///
    /// Use [`Self::into_workspace_objective_with_policy`] to tune the score
    /// tile for a measured workload or worker-local memory budget.
    #[must_use]
    pub fn into_workspace_objective(self) -> WorkspaceGamlss<F, Blocks, Obs> {
        self.into_workspace_objective_with_policy(ScoreTilePolicy::default())
    }

    /// Wraps the model as an objective with reusable gradient buffers and an
    /// explicit score-tile policy.
    #[must_use]
    pub fn into_workspace_objective_with_policy(
        self,
        policy: ScoreTilePolicy,
    ) -> WorkspaceGamlss<F, Blocks, Obs> {
        let workspace = self.gradient_workspace_with_policy(policy);
        WorkspaceGamlss {
            model: self,
            workspace,
        }
    }

    /// Coefficient block ranges within beta.
    #[must_use]
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
    #[must_use]
    pub fn parameter_layout(&self) -> ParameterLayout {
        self.blocks.parameter_layout()
    }

    /// Structured coefficient descriptors inside the flat optimizer-parameter vector.
    ///
    /// Ordinary scalar-parameter models return one whole-parameter descriptor
    /// per block. Structured multivariate block implementations may return
    /// component- or matrix-entry-level descriptors, such as `mu[i]` or
    /// `cholesky[row, col]`.
    #[must_use]
    pub fn parameter_descriptors(&self) -> Vec<ParameterDescriptor> {
        self.blocks.parameter_descriptors()
    }

    /// Number of canonical scalar-leaf descriptors in this model layout.
    #[must_use]
    pub fn parameter_descriptor_count(&self) -> usize {
        let mut count = 0;
        self.visit_parameter_descriptors(|_, _| count += 1);
        count
    }

    /// Returns one canonical descriptor by its stable index in this model layout.
    #[must_use]
    pub fn parameter_descriptor(&self, descriptor_index: usize) -> Option<ParameterDescriptor> {
        let mut found = None;
        self.visit_parameter_descriptors(|index, descriptor| {
            if index == descriptor_index {
                found = Some(descriptor);
            }
        });
        found
    }

    /// Visits every canonical descriptor for typed parameter marker `P`.
    pub fn visit_parameter_descriptors_of<P>(
        &self,
        mut visit: impl FnMut(usize, ParameterDescriptor),
    ) where
        P: ParameterName,
    {
        self.visit_parameter_descriptors(|index, descriptor| {
            if descriptor.role == P::NAME {
                visit(index, descriptor);
            }
        });
    }

    /// Returns every canonical descriptor for typed parameter marker `P`.
    #[must_use]
    pub fn parameter_descriptors_of<P>(&self) -> Vec<(usize, ParameterDescriptor)>
    where
        P: ParameterName,
    {
        let mut descriptors = Vec::new();
        self.visit_parameter_descriptors_of::<P>(|index, descriptor| {
            descriptors.push((index, descriptor));
        });
        descriptors
    }

    /// Finds the unique descriptor for typed role `P` at `path`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::AmbiguousParameter`] when `(P::NAME, path)` is
    /// not unique. This can occur when identical roles appear in separate
    /// product branches whose paths do not introduce another axis.
    pub fn unique_parameter_descriptor_at_path<P>(
        &self,
        path: &ParameterPath,
    ) -> Result<Option<(usize, ParameterDescriptor)>, ModelError>
    where
        P: ParameterName,
    {
        let mut matched = UniqueParameterMatch::new();
        self.visit_parameter_descriptors_of::<P>(|index, descriptor| {
            if descriptor.path == *path {
                matched.record((index, descriptor));
            }
        });
        matched.resolve(P::NAME)
    }

    /// Finds the stable index of a full descriptor in this model layout.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::AmbiguousParameter`] if a malformed layout
    /// contains the same full descriptor more than once.
    pub fn parameter_descriptor_index(
        &self,
        selected: &ParameterDescriptor,
    ) -> Result<Option<usize>, ModelError> {
        let mut matched = UniqueParameterMatch::new();
        self.visit_parameter_descriptors(|index, descriptor| {
            if descriptor == *selected {
                matched.record(index);
            }
        });
        matched.resolve(selected.role)
    }

    fn unique_parameter_range(
        &self,
        name: &'static str,
    ) -> Result<Option<Range<usize>>, ModelError> {
        let mut matched = UniqueParameterMatch::new();
        self.visit_parameter_slices(|_, candidate, range| {
            if candidate == name {
                matched.record(range);
            }
        });
        matched.resolve(name)
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
    /// contain a parameter named `P::NAME`, or
    /// [`ModelError::AmbiguousParameter`] if the role occurs in more than one
    /// coarse block. When blocks are constructed through the typed
    /// [`crate::ParameterBlock`] API, absence cannot happen in practice for a
    /// marker that is part of the family shape.
    pub fn block_objective_for<P>(
        &mut self,
        full_beta: Vec<f64>,
    ) -> Result<BlockObjective<'_, Self>, ModelError>
    where
        P: ParameterName,
    {
        validate_len("parameters", full_beta.len(), self.nparams())?;
        let range = self
            .unique_parameter_range(P::NAME)?
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

    /// Creates a [`BlockObjective`] for one canonical descriptor index.
    ///
    /// Unlike [`Self::block_objective_for`], this selects one scalar predictor
    /// leaf and is unambiguous for repeated, mixture, vector, and matrix roles.
    pub fn block_objective_at(
        &mut self,
        descriptor_index: usize,
        full_beta: Vec<f64>,
    ) -> Result<BlockObjective<'_, Self>, ModelError> {
        validate_len("parameters", full_beta.len(), self.nparams())?;
        let descriptor = self.parameter_descriptor(descriptor_index).ok_or_else(|| {
            ModelError::ParameterDescriptorIndexOutOfBounds {
                index: descriptor_index,
                count: self.parameter_descriptor_count(),
            }
        })?;
        BlockObjective::try_new_for_descriptor(self, full_beta, descriptor)
    }

    /// Creates a [`BlockObjective`] for a full descriptor from this model.
    pub fn block_objective_for_descriptor(
        &mut self,
        descriptor: &ParameterDescriptor,
        full_beta: Vec<f64>,
    ) -> Result<BlockObjective<'_, Self>, ModelError> {
        validate_len("parameters", full_beta.len(), self.nparams())?;
        if self.parameter_descriptor_index(descriptor)?.is_none() {
            return Err(ModelError::UnknownParameterDescriptor {
                descriptor: descriptor.clone(),
            });
        }
        BlockObjective::try_new_for_descriptor(self, full_beta, descriptor.clone())
    }

    /// Unpacks a flat optimizer-parameter vector into descriptor-aware leaf blocks.
    pub fn unpack_parameters(&self, parameters: &[f64]) -> Result<UnpackedParameters, ModelError> {
        validate_len("parameters", parameters.len(), self.nparams())?;

        let mut blocks = Vec::with_capacity(self.parameter_descriptor_count());
        self.visit_parameter_descriptors(|descriptor_index, descriptor| {
            blocks.push(ParameterCoefficients {
                descriptor_index,
                coefficients: parameters[descriptor.range.clone()].to_vec(),
                descriptor,
            });
        });

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
        let train_nll = likelihood_multiplier * {
            let (family_workspace, gradient_workspace) = workspace.parts_mut();
            self.blocks.train_nll_into_workspace(
                &self.family,
                &self.obs,
                parameters,
                family_workspace,
                gradient_workspace,
            )
        };
        let penalty = self.blocks.penalty_value(parameters);
        self.try_gradient_into_workspace(parameters, grad, workspace)?;
        Ok(training_diagnostics_from_gradient(train_nll, penalty, grad))
    }

    fn training_prediction_view(
        &self,
        parameters: &[f64],
    ) -> Result<PredictionView<'_, F, Blocks>, ModelError> {
        validate_len("parameters", parameters.len(), self.nparams())?;
        Ok(PredictionView::from_parts(
            &self.family,
            &self.blocks,
            self.nobs(),
            parameters.len(),
        ))
    }

    /// Predicts link-scale distribution predictors for one training row.
    pub fn predict_eta_row(&self, parameters: &[f64], row: usize) -> Result<F::Eta, ModelError>
    where
        F: Family,
    {
        self.training_prediction_view(parameters)?
            .predict_eta_row(parameters, row)
    }

    /// Predicts natural-scale distribution parameters for one training row.
    pub fn predict_theta_row(&self, parameters: &[f64], row: usize) -> Result<F::Theta, ModelError>
    where
        F: Family,
    {
        self.training_prediction_view(parameters)?
            .predict_theta_row(parameters, row)
    }

    /// Predicts link-scale distribution predictors for all training rows.
    pub fn predict_eta(&self, parameters: &[f64]) -> Result<Vec<F::Eta>, ModelError>
    where
        F: Family,
    {
        self.training_prediction_view(parameters)?
            .predict_eta(parameters)
    }

    /// Predicts natural-scale distribution parameters for all training rows.
    pub fn predict_theta(&self, parameters: &[f64]) -> Result<Vec<F::Theta>, ModelError>
    where
        F: Family,
    {
        validate_len("parameters", parameters.len(), self.nparams())?;
        Ok(prediction::collect_theta(
            &self.family,
            &self.blocks,
            parameters,
            || self.nobs(),
        ))
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
        self.training_prediction_view(parameters)?
            .predict_theta_into(parameters, out)
    }

    /// Streams natural-scale distribution parameters for each training row.
    pub fn for_each_theta(
        &self,
        parameters: &[f64],
        visit: impl FnMut(usize, F::Theta),
    ) -> Result<(), ModelError>
    where
        F: Family,
    {
        validate_len("parameters", parameters.len(), self.nparams())?;
        prediction::visit_theta(
            &self.family,
            &self.blocks,
            parameters,
            || self.nobs(),
            visit,
        );
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
        Ok(self.likelihood_multiplier().mul_add(train_nll, penalty))
    }

    /// Computes the objective while reusing model and runtime-family buffers.
    pub fn try_value_into_workspace(
        &self,
        beta: &[f64],
        workspace: &mut ModelWorkspace<F>,
    ) -> Result<f64, ModelError> {
        validate_len("parameters", beta.len(), self.nparams())?;
        let train_nll = {
            let (family_workspace, gradient_workspace) = workspace.parts_mut();
            self.blocks.train_nll_into_workspace(
                &self.family,
                &self.obs,
                beta,
                family_workspace,
                gradient_workspace,
            )
        };
        let penalty = self.blocks.penalty_value(beta);
        Ok(self.likelihood_multiplier().mul_add(train_nll, penalty))
    }

    /// Returns the summed weighted negative log-likelihood without penalties.
    pub fn try_likelihood_value(&self, beta: &[f64]) -> Result<f64, ModelError> {
        validate_len("parameters", beta.len(), self.nparams())?;
        Ok(self.blocks.train_nll(&self.family, &self.obs, beta))
    }

    /// Workspace-reusing variant of [`Self::try_likelihood_value`].
    pub fn try_likelihood_value_into_workspace(
        &self,
        beta: &[f64],
        workspace: &mut ModelWorkspace<F>,
    ) -> Result<f64, ModelError> {
        validate_len("parameters", beta.len(), self.nparams())?;
        let (family_workspace, gradient_workspace) = workspace.parts_mut();
        Ok(self.blocks.train_nll_into_workspace(
            &self.family,
            &self.obs,
            beta,
            family_workspace,
            gradient_workspace,
        ))
    }

    /// Computes summed weighted NLL and its coefficient gradient without penalties.
    pub fn try_likelihood_value_gradient_into(
        &self,
        beta: &[f64],
        grad: &mut [f64],
    ) -> Result<f64, ModelError> {
        let mut workspace = self.gradient_workspace();
        self.try_likelihood_value_gradient_into_workspace(beta, grad, &mut workspace)
    }

    /// Workspace-reusing variant of [`Self::try_likelihood_value_gradient_into`].
    pub fn try_likelihood_value_gradient_into_workspace(
        &self,
        beta: &[f64],
        grad: &mut [f64],
        workspace: &mut ModelWorkspace<F>,
    ) -> Result<f64, ModelError> {
        validate_beta_and_gradient_len(self.nparams(), beta, grad)?;
        grad.fill(0.0);
        let value = {
            let (family_workspace, gradient_workspace) = workspace.parts_mut();
            self.blocks.value_gradient_into_workspace(
                &self.family,
                &self.obs,
                beta,
                grad,
                family_workspace,
                gradient_workspace,
            )
        };
        let penalty = self.blocks.penalty_value(beta);
        let penalty_grad = workspace
            .gradient_mut()
            .penalty_gradient_mut(self.nparams());
        self.blocks.add_penalty_gradient(beta, penalty_grad);
        for (value, penalty_value) in grad.iter_mut().zip(penalty_grad.iter().copied()) {
            *value -= penalty_value;
        }
        Ok(value - penalty)
    }

    /// Writes raw per-observation negative log-likelihood contributions.
    pub fn try_pointwise_nll_into(&self, beta: &[f64], out: &mut [f64]) -> Result<(), ModelError> {
        self.try_pointwise_nll_impl(beta, out, false)
    }

    /// Writes observation-weighted per-observation NLL contributions.
    pub fn try_weighted_pointwise_nll_into(
        &self,
        beta: &[f64],
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        self.try_pointwise_nll_impl(beta, out, true)
    }

    /// Workspace-reusing variant of [`Self::try_pointwise_nll_into`].
    pub fn try_pointwise_nll_into_workspace(
        &self,
        beta: &[f64],
        out: &mut [f64],
        workspace: &mut ModelWorkspace<F>,
    ) -> Result<(), ModelError> {
        self.try_pointwise_nll_impl_into_workspace(beta, out, false, workspace)
    }

    /// Workspace-reusing variant of [`Self::try_weighted_pointwise_nll_into`].
    pub fn try_weighted_pointwise_nll_into_workspace(
        &self,
        beta: &[f64],
        out: &mut [f64],
        workspace: &mut ModelWorkspace<F>,
    ) -> Result<(), ModelError> {
        self.try_pointwise_nll_impl_into_workspace(beta, out, true, workspace)
    }

    /// Writes raw per-observation log-likelihood contributions.
    pub fn try_pointwise_log_likelihood_into(
        &self,
        beta: &[f64],
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        self.try_pointwise_nll_impl(beta, out, false)?;
        for value in out.iter_mut() {
            *value = -*value;
        }
        Ok(())
    }

    /// Writes observation-weighted per-observation log-likelihood contributions.
    pub fn try_weighted_pointwise_log_likelihood_into(
        &self,
        beta: &[f64],
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        self.try_pointwise_nll_impl(beta, out, true)?;
        for value in out.iter_mut() {
            *value = -*value;
        }
        Ok(())
    }

    /// Workspace-reusing variant of [`Self::try_pointwise_log_likelihood_into`].
    pub fn try_pointwise_log_likelihood_into_workspace(
        &self,
        beta: &[f64],
        out: &mut [f64],
        workspace: &mut ModelWorkspace<F>,
    ) -> Result<(), ModelError> {
        self.try_pointwise_nll_impl_into_workspace(beta, out, false, workspace)?;
        for value in out.iter_mut() {
            *value = -*value;
        }
        Ok(())
    }

    /// Workspace-reusing variant of [`Self::try_weighted_pointwise_log_likelihood_into`].
    pub fn try_weighted_pointwise_log_likelihood_into_workspace(
        &self,
        beta: &[f64],
        out: &mut [f64],
        workspace: &mut ModelWorkspace<F>,
    ) -> Result<(), ModelError> {
        self.try_pointwise_nll_impl_into_workspace(beta, out, true, workspace)?;
        for value in out.iter_mut() {
            *value = -*value;
        }
        Ok(())
    }

    fn try_pointwise_nll_impl(
        &self,
        beta: &[f64],
        out: &mut [f64],
        weighted: bool,
    ) -> Result<(), ModelError> {
        validate_len("parameters", beta.len(), self.nparams())?;
        validate_output_len(self.nobs(), out.len())?;
        self.blocks
            .pointwise_nll_into(&self.family, &self.obs, beta, weighted, out);
        Ok(())
    }

    fn try_pointwise_nll_impl_into_workspace(
        &self,
        beta: &[f64],
        out: &mut [f64],
        weighted: bool,
        workspace: &mut ModelWorkspace<F>,
    ) -> Result<(), ModelError> {
        validate_len("parameters", beta.len(), self.nparams())?;
        validate_output_len(self.nobs(), out.len())?;
        self.blocks.pointwise_nll_into_workspace(
            &self.family,
            &self.obs,
            beta,
            weighted,
            out,
            workspace.parts_mut(),
        );
        Ok(())
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
        self.try_value_gradient_into_workspace(beta, grad, workspace)
            .map(|_| ())
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
    /// Creates a workspace-backed objective from a compiled model using the
    /// default [`ScoreTilePolicy`].
    ///
    /// Use [`Self::new_with_policy`] when an explicit score memory or row
    /// budget is required.
    #[must_use]
    #[inline]
    pub fn new(model: Gamlss<F, Blocks, Obs>) -> Self {
        model.into_workspace_objective()
    }

    /// Creates a workspace-backed objective with an explicit score-tile policy.
    #[must_use]
    #[inline]
    pub fn new_with_policy(model: Gamlss<F, Blocks, Obs>, policy: ScoreTilePolicy) -> Self {
        model.into_workspace_objective_with_policy(policy)
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
    #[must_use]
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
    /// contain a parameter named `P::NAME`, or
    /// [`ModelError::AmbiguousParameter`] if the role occurs in more than one
    /// coarse block.
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
            .unique_parameter_range(P::NAME)?
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

    /// Creates a [`BlockObjective`] for one canonical descriptor index.
    pub fn block_objective_at(
        &mut self,
        descriptor_index: usize,
        full_beta: Vec<f64>,
    ) -> Result<BlockObjective<'_, Self>, ModelError> {
        validate_len("parameters", full_beta.len(), self.dim())?;
        let descriptor = self
            .model
            .parameter_descriptor(descriptor_index)
            .ok_or_else(|| ModelError::ParameterDescriptorIndexOutOfBounds {
                index: descriptor_index,
                count: self.model.parameter_descriptor_count(),
            })?;
        BlockObjective::try_new_for_descriptor(self, full_beta, descriptor)
    }

    /// Creates a [`BlockObjective`] for a full descriptor from the inner model.
    pub fn block_objective_for_descriptor(
        &mut self,
        descriptor: &ParameterDescriptor,
        full_beta: Vec<f64>,
    ) -> Result<BlockObjective<'_, Self>, ModelError> {
        validate_len("parameters", full_beta.len(), self.dim())?;
        if self.model.parameter_descriptor_index(descriptor)?.is_none() {
            return Err(ModelError::UnknownParameterDescriptor {
                descriptor: descriptor.clone(),
            });
        }
        BlockObjective::try_new_for_descriptor(self, full_beta, descriptor.clone())
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
        self.model
            .try_value_into_workspace(parameters, &mut self.workspace)
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
/// Unlike [`crate::Penalty`], which acts locally on a single block,
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

/// Tuple contract for a set of parameter blocks compatible with family `F`.
///
/// Built-in implementations use [`ParameterBlocks`] around a static block tree.
/// The model validates observation count, predictor row counts and coefficient
/// ranges before hot-path evaluation.
pub trait GamlssBlocks<F>: sealed::Sealed
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
    /// Validates blocks together with runtime family configuration.
    fn validate_for(&self, _family: &F, nobs: usize) -> Result<(), ModelError> {
        self.validate(nobs)
    }
    /// Weighted negative log-likelihood without penalties.
    ///
    /// `obs` has already been validated by the model constructor. Each scalar
    /// likelihood contribution is multiplied by the corresponding observation
    /// weight.
    fn train_nll<'obs, Obs>(&self, family: &F, obs: &'obs Obs, beta: &[f64]) -> f64
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        let mut family_workspace = family.workspace();
        train_nll_rows(self, family, obs, beta, &mut family_workspace)
    }
    /// Workspace-reusing weighted negative log-likelihood without penalties.
    ///
    /// Runtime-dimensional block implementations should override this method
    /// to consume their flat eta coordinates directly. The default is suitable
    /// for static carriers and reuses the caller's family workspace.
    fn train_nll_into_workspace<'obs, Obs>(
        &self,
        family: &F,
        obs: &'obs Obs,
        beta: &[f64],
        family_workspace: &mut F::Workspace,
        _workspace: &mut GradientWorkspace,
    ) -> f64
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        train_nll_rows(self, family, obs, beta, family_workspace)
    }
    /// Writes raw or weighted per-observation NLL contributions.
    fn pointwise_nll_into<'obs, Obs>(
        &self,
        family: &F,
        obs: &'obs Obs,
        beta: &[f64],
        weighted: bool,
        out: &mut [f64],
    ) where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        let mut family_workspace = family.workspace();
        let mut workspace = GradientWorkspace::new();
        self.pointwise_nll_into_workspace(
            family,
            obs,
            beta,
            weighted,
            out,
            (&mut family_workspace, &mut workspace),
        );
    }
    /// Workspace-reusing variant of [`Self::pointwise_nll_into`].
    fn pointwise_nll_into_workspace<'obs, Obs>(
        &self,
        family: &F,
        obs: &'obs Obs,
        beta: &[f64],
        weighted: bool,
        out: &mut [f64],
        workspace: (&mut F::Workspace, &mut GradientWorkspace),
    ) where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        let (family_workspace, _) = workspace;
        for (row, value) in out.iter_mut().enumerate() {
            let weight = obs.weight_at(row);
            if weighted && weight == 0.0 {
                *value = 0.0;
                continue;
            }
            let eta = self.eta_row(family, beta, row);
            let nll = family.nll_eta(obs.observation_at(row), &eta, family_workspace);
            *value = if weighted { weight * nll } else { nll };
        }
    }
    /// Additive predictors on the link scale for one row.
    fn eta_row(&self, family: &F, beta: &[f64], row: usize) -> F::Eta
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
    /// coefficient length or runtime initializer shape must be checked first.
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
    fn gradient_workspace(&self, nobs: usize, policy: ScoreTilePolicy) -> GradientWorkspace {
        let mut workspace = GradientWorkspace::with_score_tile_policy(policy);
        let _ = workspace.prepare_score_tile(self.block_ranges().len(), nobs);
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

    /// Runtime topology key captured by these blocks, when applicable.
    fn dynamic_layout_key(&self) -> Option<&DynamicLayoutKey> {
        None
    }

    /// Visits coefficient ranges for each parameter block in model order.
    ///
    /// The default implementation materializes [`Self::block_ranges`]; compiled
    /// block implementations override it with an allocation-free traversal.
    fn visit_block_ranges<V>(&self, mut visit: V)
    where
        V: FnMut(usize, Range<usize>),
    {
        for (index, range) in self.block_ranges().into_iter().enumerate() {
            visit(index, range);
        }
    }

    /// Visits named parameter slices in model order.
    ///
    /// The default implementation materializes [`Self::parameter_layout`];
    /// compiled block implementations override it with an allocation-free
    /// traversal.
    fn visit_parameter_slices<V>(&self, mut visit: V)
    where
        V: FnMut(usize, &'static str, Range<usize>),
    {
        for (index, slice) in self.parameter_layout().slices().iter().enumerate() {
            visit(index, slice.name, slice.range.clone());
        }
    }

    /// Visits structured coefficient descriptors in model order.
    ///
    /// The default implementation materializes
    /// [`Self::parameter_descriptors`], ensuring that implementations which
    /// override the allocating descriptor API retain the same metadata here.
    /// Compiled block implementations override this visitor to avoid the
    /// allocation.
    fn visit_parameter_descriptors<V>(&self, mut visit: V)
    where
        V: FnMut(usize, ParameterDescriptor),
    {
        for (index, descriptor) in self.parameter_descriptors().into_iter().enumerate() {
            visit(index, descriptor);
        }
    }
}

impl<F, B> GamlssBlocks<F> for ParameterBlocks<B>
where
    F: CompilableFamily,
    B: ShapeBlocks<F::Shape>,
{
    fn nrows(&self) -> usize {
        self.as_inner().nrows().unwrap_or(0)
    }

    fn len(&self) -> usize {
        <Self as GamlssBlocks<F>>::try_len(self)
            .expect("validated static parameter tree length must fit in usize")
    }

    fn try_len(&self) -> Result<usize, ModelError> {
        self.as_inner().try_len()
    }

    fn validate(&self, nobs: usize) -> Result<(), ModelError> {
        self.as_inner().validate(nobs)?;
        let mut ranges = Vec::with_capacity(self.as_inner().leaf_count());
        <Self as GamlssBlocks<F>>::visit_parameter_descriptors(self, |_, descriptor| {
            ranges.push((descriptor.role, descriptor.range));
        });
        validate_non_overlapping_ranges(&ranges)
    }

    fn validate_for(&self, family: &F, nobs: usize) -> Result<(), ModelError> {
        family.validate_compiled()?;
        <Self as GamlssBlocks<F>>::validate(self, nobs)
    }

    fn train_nll_into_workspace<'obs, Obs>(
        &self,
        family: &F,
        obs: &'obs Obs,
        beta: &[f64],
        family_workspace: &mut F::Workspace,
        workspace: &mut GradientWorkspace,
    ) -> f64
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        let tile_rows = workspace.prepare_score_tile(self.as_inner().leaf_count(), obs.len());
        let mut loss = 0.0;
        let mut tile_start = 0;

        while tile_start < obs.len() {
            let tile_end = tile_start.saturating_add(tile_rows).min(obs.len());
            let rows = tile_start..tile_end;
            workspace.set_score_tile_len(rows.len());
            let use_prepared_values = rows.clone().all(|row| obs.weight_at(row) != 0.0);

            if use_prepared_values {
                let mut cursor = 0;
                self.as_inner()
                    .prepare_values(rows.clone(), beta, workspace, &mut cursor);
                debug_assert_eq!(cursor, self.as_inner().leaf_count());
            }

            for (tile_row, row) in rows.enumerate() {
                let weight = obs.weight_at(row);
                if weight == 0.0 {
                    continue;
                }
                let values = if use_prepared_values {
                    let mut cursor = 0;
                    let values =
                        self.as_inner()
                            .prepared_values_row(tile_row, workspace, &mut cursor);
                    debug_assert_eq!(cursor, self.as_inner().leaf_count());
                    values
                } else {
                    self.as_inner().values_row(beta, row)
                };
                let eta = F::eta_from_shape(values);
                loss = weight.mul_add(
                    family.nll_eta(obs.observation_at(row), &eta, family_workspace),
                    loss,
                );
            }

            tile_start = tile_end;
        }
        loss
    }

    fn eta_row(&self, _family: &F, beta: &[f64], row: usize) -> F::Eta {
        F::eta_from_shape(self.as_inner().values_row(beta, row))
    }

    fn penalty_value(&self, beta: &[f64]) -> f64 {
        self.as_inner().penalty_value(beta)
    }

    fn initial_parameters<'obs, Obs>(&self, family: &F, obs: &'obs Obs) -> Vec<f64>
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        self.try_initial_parameters(family, obs)
            .expect("validated static parameter tree length must fit in usize")
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
        let values = family.initial_shape(obs);
        self.as_inner().set_initial(&values, &mut beta);
        Ok(beta)
    }

    fn add_penalty_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        self.as_inner().add_penalty_gradient(beta, grad);
    }

    fn gradient_workspace(&self, nobs: usize, policy: ScoreTilePolicy) -> GradientWorkspace {
        let mut workspace = GradientWorkspace::with_score_tile_policy(policy);
        let _ = workspace.prepare_score_tile(self.as_inner().leaf_count(), nobs);
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
        let tile_rows = workspace.prepare_score_tile(self.as_inner().leaf_count(), obs.len());
        let mut loss = 0.0;

        let mut tile_start = 0;
        while tile_start < obs.len() {
            let tile_end = tile_start.saturating_add(tile_rows).min(obs.len());
            let rows = tile_start..tile_end;
            workspace.set_score_tile_len(rows.len());

            // Bulk forward paths may read every selected design row. Preserve
            // the contract that a zero-weight observation disables its row by
            // retaining the row-wise path for tiles containing a mask.
            let use_prepared_values = rows.clone().all(|row| obs.weight_at(row) != 0.0);
            if use_prepared_values {
                let mut cursor = 0;
                self.as_inner()
                    .prepare_values(rows.clone(), beta, workspace, &mut cursor);
                debug_assert_eq!(cursor, self.as_inner().leaf_count());
            }

            for (tile_row, row) in rows.clone().enumerate() {
                let weight = obs.weight_at(row);
                if weight == 0.0 {
                    workspace.fill_score_row(tile_row, 0.0);
                    continue;
                }
                let values = if use_prepared_values {
                    let mut cursor = 0;
                    let values =
                        self.as_inner()
                            .prepared_values_row(tile_row, workspace, &mut cursor);
                    debug_assert_eq!(cursor, self.as_inner().leaf_count());
                    values
                } else {
                    self.as_inner().values_row(beta, row)
                };
                let eta = F::eta_from_shape(values);
                let (nll, gradient) =
                    family.nll_and_gradient_eta(obs.observation_at(row), &eta, family_workspace);
                loss = weight.mul_add(nll, loss);
                let scores = F::gradient_to_shape(&gradient);
                let mut cursor = 0;
                self.as_inner()
                    .set_scores(&scores, tile_row, weight, workspace, &mut cursor);
                debug_assert_eq!(cursor, self.as_inner().leaf_count());
            }

            let mut cursor = 0;
            self.as_inner()
                .backprop(rows, beta, grad, workspace, &mut cursor);
            debug_assert_eq!(cursor, self.as_inner().leaf_count());
            tile_start = tile_end;
        }
        self.as_inner().add_penalty_gradient(beta, grad);
        loss + self.as_inner().penalty_value(beta)
    }

    fn block_ranges(&self) -> Vec<Range<usize>> {
        let mut ranges = Vec::with_capacity(self.as_inner().leaf_count());
        <Self as GamlssBlocks<F>>::visit_parameter_descriptors(self, |_, descriptor| {
            ranges.push(descriptor.range);
        });
        ranges
    }

    fn parameter_layout(&self) -> ParameterLayout {
        let mut slices = Vec::with_capacity(self.as_inner().leaf_count());
        <Self as GamlssBlocks<F>>::visit_parameter_slices(self, |_, name, range| {
            slices.push(ParameterSlice { name, range });
        });
        ParameterLayout::new(slices)
    }

    fn parameter_descriptors(&self) -> Vec<ParameterDescriptor> {
        let mut descriptors = Vec::with_capacity(self.as_inner().leaf_count());
        <Self as GamlssBlocks<F>>::visit_parameter_descriptors(self, |_, descriptor| {
            descriptors.push(descriptor);
        });
        descriptors
    }

    fn visit_block_ranges<V>(&self, mut visit: V)
    where
        V: FnMut(usize, Range<usize>),
    {
        <Self as GamlssBlocks<F>>::visit_parameter_descriptors(self, |index, descriptor| {
            visit(index, descriptor.range);
        });
    }

    fn visit_parameter_slices<V>(&self, mut visit: V)
    where
        V: FnMut(usize, &'static str, Range<usize>),
    {
        let mut cursor = 0;
        self.as_inner().visit_slices(&mut cursor, &mut visit);
    }

    fn visit_parameter_descriptors<V>(&self, mut visit: V)
    where
        V: FnMut(usize, ParameterDescriptor),
    {
        let mut cursor = 0;
        self.as_inner()
            .visit_descriptors(&ParameterPath::whole(), &mut cursor, &mut visit);
    }
}

impl<B> sealed::Sealed for ParameterBlocks<B> {}

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

fn train_nll_rows<'obs, F, Blocks, Obs>(
    blocks: &Blocks,
    family: &F,
    obs: &'obs Obs,
    beta: &[f64],
    family_workspace: &mut F::Workspace,
) -> f64
where
    F: Family,
    Blocks: GamlssBlocks<F> + ?Sized,
    Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
{
    let mut loss = 0.0;
    for row in 0..obs.len() {
        let weight = obs.weight_at(row);
        if weight == 0.0 {
            continue;
        }
        let eta = blocks.eta_row(family, beta, row);
        loss = weight.mul_add(
            family.nll_eta(obs.observation_at(row), &eta, family_workspace),
            loss,
        );
    }
    loss
}

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
    let expected_layout = expected_blocks.parameter_layout();
    let got_layout = blocks.parameter_layout();
    if expected_layout != got_layout {
        return Err(ModelError::PredictionLayoutMismatch {
            expected: expected_layout,
            got: got_layout,
        });
    }

    let expected_key = expected_blocks.dynamic_layout_key().cloned();
    let got_key = blocks.dynamic_layout_key().cloned();
    let expected_descriptors = expected_blocks.parameter_descriptors();
    let got_descriptors = blocks.parameter_descriptors();
    if expected_key == got_key && expected_descriptors == got_descriptors {
        Ok(())
    } else {
        Err(ModelError::PredictionLayoutIdentityMismatch {
            expected_key,
            got_key,
            expected_descriptors,
            got_descriptors,
        })
    }
}

#[cfg(test)]
mod tests;
