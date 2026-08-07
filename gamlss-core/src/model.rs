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
pub use workspace::{GradientWorkspace, ModelWorkspace, ScoreTilePolicy};

mod dynamic;
mod executor;
mod layout;
mod observation;
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

    /// Predicts link-scale distribution predictors for one training row.
    pub fn predict_eta_row(&self, parameters: &[f64], row: usize) -> Result<F::Eta, ModelError>
    where
        F: Family,
    {
        validate_len("parameters", parameters.len(), self.nparams())?;
        validate_row(row, self.nobs())?;
        Ok(self.blocks.eta_row(&self.family, parameters, row))
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
            .map(|row| self.blocks.eta_row(&self.family, parameters, row))
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
                let eta = self.blocks.eta_row(&self.family, parameters, row);
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
            let eta = self.blocks.eta_row(&self.family, parameters, row);
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
            let eta = self.blocks.eta_row(&self.family, parameters, row);
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
        Ok(self.blocks.eta_row(self.family, parameters, row))
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
            .map(|row| self.blocks.eta_row(self.family, parameters, row))
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
                let eta = self.blocks.eta_row(self.family, parameters, row);
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
            let eta = self.blocks.eta_row(self.family, parameters, row);
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
            let eta = self.blocks.eta_row(self.family, parameters, row);
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
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs;
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
        let mut loss = 0.0;
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            if weight == 0.0 {
                continue;
            }
            let eta = self.eta_row(family, beta, row);
            loss = weight.mul_add(
                family.nll_eta(obs.observation_at(row), &eta, family_workspace),
                loss,
            );
        }
        loss
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
            let values = self.as_inner().values_row(beta, row);
            let eta = F::eta_from_shape(values);
            loss = weight.mul_add(
                family.nll_eta(obs.observation_at(row), &eta, &mut family_workspace),
                loss,
            );
        }
        loss
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

            for (tile_row, row) in rows.clone().enumerate() {
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
mod tests {
    use std::ops::Range;

    use approx::assert_relative_eq;

    use crate::{
        Broadcast, CholeskyScale, CompilableFamily, DenseDesign, DenseRows, DynamicLayoutKey,
        DynamicParameterBlocks, DynamicallyCompilableFamily, Family, Gamlss, GamlssBlocks,
        GlobalPenalty, HingeQuadraticPenalty, InitialEtaFromObservations, LinearFormBuilder,
        LinearPredictorBlock, Lower, LowerTriangularParameterBlock, ModelError, Mu, NoPenalty, Nu,
        Objective, ObjectiveScale, ObservationView, OffsetBlock, ParameterAxis, ParameterBlock,
        ParameterBlocks, ParameterDescriptor, ParameterLayout, ParameterName, ParameterPath,
        ParameterSlice, PredictorBlock, Product, RidgePenalty, Scalar, ScoreTilePolicy, Sigma,
        SumBlock, Tau, Vector, VectorParameterBlock,
    };

    #[derive(Debug, Clone, Copy)]
    struct FixedSigmaNormal;

    crate::impl_scalar_compilable_family!(
        impl for FixedSigmaNormal;
        parameters = (Mu,);
        arity = 1;
    );

    impl Family for FixedSigmaNormal {
        type Eta = f64;
        type Theta = f64;
        type GradientEta = f64;
        type Observation<'obs> = f64;
        type Workspace = ();
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
    struct SharedMeanPair;

    impl Family for SharedMeanPair {
        type Eta = [f64; 2];
        type Theta = [f64; 2];
        type GradientEta = [f64; 2];
        type Observation<'obs> = [f64; 2];
        type Workspace = ();

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
            observation
                .iter()
                .zip(theta)
                .map(|(y, mean)| 0.5 * (mean - y).powi(2))
                .sum()
        }

        fn nll_and_gradient_eta(
            &self,
            observation: Self::Observation<'_>,
            eta: &Self::Eta,
            workspace: &mut Self::Workspace,
        ) -> (f64, Self::GradientEta) {
            (
                self.nll(observation, eta, workspace),
                std::array::from_fn(|index| eta[index] - observation[index]),
            )
        }
    }

    impl CompilableFamily for SharedMeanPair {
        type Shape = Broadcast<Scalar<Mu>, 2>;

        fn eta_from_shape(values: <Self::Shape as crate::ParameterShape>::Values) -> Self::Eta {
            values
        }

        fn gradient_to_shape(
            gradient: &Self::GradientEta,
        ) -> <Self::Shape as crate::ParameterShape>::Values {
            *gradient
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct TwoParameterMock;

    crate::impl_scalar_compilable_family!(
        impl for TwoParameterMock;
        parameters = (Mu, Sigma);
        arity = 2;
    );

    impl Family for TwoParameterMock {
        type Eta = (f64, f64);
        type Theta = (f64, f64);
        type GradientEta = (f64, f64);
        type Observation<'obs> = f64;
        type Workspace = ();
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

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct RuntimeLayoutMock {
        key: usize,
        reverse: bool,
        whole_paths: bool,
        initial_count: usize,
    }

    impl Family for RuntimeLayoutMock {
        type Eta = [f64; 2];
        type Theta = [f64; 2];
        type GradientEta = [f64; 2];
        type Observation<'obs> = f64;
        type Workspace = ();

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
            theta
                .iter()
                .map(|value| 0.5 * (value - observation).powi(2))
                .sum()
        }

        fn nll_and_gradient_eta(
            &self,
            observation: Self::Observation<'_>,
            eta: &Self::Eta,
            workspace: &mut Self::Workspace,
        ) -> (f64, Self::GradientEta) {
            (
                self.nll(observation, eta, workspace),
                [eta[0] - observation, eta[1] - observation],
            )
        }
    }

    impl DynamicallyCompilableFamily for RuntimeLayoutMock {
        fn dynamic_parameter_count(&self) -> usize {
            2
        }

        fn dynamic_layout_key(&self) -> DynamicLayoutKey {
            DynamicLayoutKey::new(vec![self.key, usize::from(self.whole_paths)])
        }

        fn eta_from_flat(&self, values: &[f64]) -> Self::Eta {
            assert_ne!(
                self.key,
                usize::MAX,
                "workspace value paths must not materialize dynamic eta"
            );
            let [first, second] = values else {
                return [f64::NAN; 2];
            };
            if self.reverse {
                [*second, *first]
            } else {
                [*first, *second]
            }
        }

        fn nll_eta_flat(
            &self,
            observation: Self::Observation<'_>,
            values: &[f64],
            _workspace: &mut Self::Workspace,
        ) -> f64 {
            let [first, second] = values else {
                return f64::INFINITY;
            };
            let (first, second) = if self.reverse {
                (*second, *first)
            } else {
                (*first, *second)
            };
            0.5_f64.mul_add(
                (first - observation).powi(2),
                0.5 * (second - observation).powi(2),
            )
        }

        fn nll_and_gradient_eta_flat(
            &self,
            observation: Self::Observation<'_>,
            values: &[f64],
            gradient: &mut [f64],
            workspace: &mut Self::Workspace,
        ) -> f64 {
            let [first, second] = values else {
                gradient.fill(f64::NAN);
                return f64::INFINITY;
            };
            let [first_gradient, second_gradient] = gradient else {
                gradient.fill(f64::NAN);
                return f64::INFINITY;
            };
            *first_gradient = *first - observation;
            *second_gradient = *second - observation;
            self.nll_eta_flat(observation, values, workspace)
        }

        fn gradient_to_flat(&self, gradient: &Self::GradientEta, out: &mut [f64]) {
            let [first, second] = out else {
                out.fill(f64::NAN);
                return;
            };
            if self.reverse {
                *first = gradient[1];
                *second = gradient[0];
            } else {
                *first = gradient[0];
                *second = gradient[1];
            }
        }

        fn dynamic_parameter_coordinate(&self, index: usize) -> (&'static str, ParameterPath) {
            if self.whole_paths {
                return (Mu::NAME, ParameterPath::whole());
            }
            let component = match (self.reverse, index) {
                (true, 0) => 1,
                (true, 1) => 0,
                _ => index,
            };
            (
                Mu::NAME,
                ParameterPath::from_axis(ParameterAxis::Vector { component }),
            )
        }

        fn initial_flat<'obs, Obs>(&self, _obs: &'obs Obs) -> Vec<f64>
        where
            Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
        {
            vec![0.0; self.initial_count]
        }
    }

    fn runtime_layout_blocks(
        family: &RuntimeLayoutMock,
        nrows: usize,
    ) -> DynamicParameterBlocks<LinearPredictorBlock<DenseDesign>, NoPenalty> {
        DynamicParameterBlocks::try_new(
            family,
            (0..2)
                .map(|_| {
                    (
                        LinearPredictorBlock::new(DenseDesign::intercept(nrows)),
                        NoPenalty,
                    )
                })
                .collect(),
        )
        .unwrap()
    }

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

    impl CompilableFamily for StructuredMock {
        type Shape = Product<Vector<Mu, 2>, Lower<CholeskyScale, 2>>;

        fn eta_from_shape(values: ([f64; 2], [[f64; 2]; 2])) -> Self::Eta {
            values
        }

        fn gradient_to_shape(gradient: &Self::GradientEta) -> ([f64; 2], [[f64; 2]; 2]) {
            *gradient
        }

        fn initial_shape<'obs, Obs>(&self, _obs: &'obs Obs) -> ([f64; 2], [[f64; 2]; 2])
        where
            Obs: ObservationView<'obs, Observation = [f64; 2]> + 'obs,
        {
            ([1.0, 2.0], [[3.0, 0.0], [4.0, 5.0]])
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct InitializingLocation;

    crate::impl_scalar_compilable_family!(
        impl for InitializingLocation;
        parameters = (Mu,);
        arity = 1;
    );

    impl Family for InitializingLocation {
        type Eta = f64;
        type Theta = f64;
        type GradientEta = f64;
        type Observation<'obs> = f64;
        type Workspace = ();
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

    crate::impl_scalar_compilable_family!(
        impl for NonFiniteInitializingLocation;
        parameters = (Mu,);
        arity = 1;
    );

    impl Family for NonFiniteInitializingLocation {
        type Eta = f64;
        type Theta = f64;
        type GradientEta = f64;
        type Observation<'obs> = f64;
        type Workspace = ();
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
        let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
        let mut model =
            Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
        let beta = vec![1.5];
        let mut grad = vec![0.0];

        assert_relative_eq!(model.value(&beta).unwrap(), 0.25);

        model.gradient(&beta, &mut grad).unwrap();

        assert_relative_eq!(grad[0], 0.0);
    }

    #[derive(Debug, Clone, Copy)]
    struct BulkOnlyPredictor {
        nrows: usize,
    }

    impl PredictorBlock for BulkOnlyPredictor {
        fn nrows(&self) -> usize {
            self.nrows
        }

        fn nparams(&self) -> usize {
            1
        }

        fn eta_row(&self, _: usize, _: &[f64]) -> f64 {
            panic!("unmasked workspace objective must use bulk forward")
        }

        fn eta_range(&self, rows: Range<usize>, beta: &[f64], out: &mut [f64]) {
            assert_eq!(out.len(), rows.len());
            out.fill(beta[0]);
        }

        fn add_gradient_range(
            &self,
            rows: Range<usize>,
            scores: &[f64],
            _: &[f64],
            gradient: &mut [f64],
        ) {
            assert_eq!(scores.len(), rows.len());
            gradient[0] += scores.iter().sum::<f64>();
        }
    }

    #[test]
    fn workspace_objective_uses_bulk_predictor_forward() {
        let y = [1.0, 2.0, 3.0];
        let mu =
            ParameterBlock::<Mu, _, _>::new(BulkOnlyPredictor { nrows: y.len() }, NoPenalty, 0);
        let model =
            Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
        let beta = [2.0];
        let mut gradient = [0.0];
        let mut workspace = model.gradient_workspace();

        let value = model
            .try_value_into_workspace(&beta, &mut workspace)
            .unwrap();
        let fused_value = model
            .try_value_gradient_into_workspace(&beta, &mut gradient, &mut workspace)
            .unwrap();

        assert_relative_eq!(value, 1.0);
        assert_relative_eq!(fused_value, value);
        assert_relative_eq!(gradient[0], 0.0);
    }

    #[test]
    fn broadcast_shape_owns_one_predictor_and_sums_consumer_scores() {
        let y = [[1.0, 3.0], [2.0, 4.0]];
        let block = ParameterBlock::<Mu, _, _>::linear(
            DenseDesign::intercept(y.len()),
            RidgePenalty::new_unchecked(0.5),
            0,
        );
        let model = Gamlss::try_new_with_observations(
            SharedMeanPair,
            ParameterBlocks::from_assigned(block),
            y.as_slice(),
        )
        .unwrap();
        let beta = [0.5];
        let mut likelihood_gradient = [0.0];
        let mut objective_gradient = [0.0];

        model
            .try_likelihood_value_gradient_into(&beta, &mut likelihood_gradient)
            .unwrap();
        model
            .try_value_gradient_into(&beta, &mut objective_gradient)
            .unwrap();

        assert_relative_eq!(likelihood_gradient[0], -8.0);
        assert_relative_eq!(objective_gradient[0], -7.5);
        assert_eq!(model.parameter_descriptors().len(), 1);
        assert_eq!(model.parameter_descriptors()[0].range, 0..1);
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
        assert_eq!(
            model.parameter_layout().unique_slice("mu").unwrap(),
            Some(0..2)
        );
        assert_eq!(
            model.parameter_layout().unique_slice("cholesky").unwrap(),
            Some(2..5)
        );
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
        let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
        let model =
            Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();

        assert_eq!(model.initial_parameters().unwrap(), vec![0.0]);
    }

    #[test]
    fn initial_parameters_write_intercept_like_constant() {
        let y = vec![1.0, 2.0, 3.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
        let mut model = Gamlss::try_new(
            InitializingLocation,
            ParameterBlocks::from_assigned((mu,)),
            &y,
        )
        .unwrap();
        let beta = model.initial_parameters().unwrap();

        assert_eq!(beta.len(), model.nparams());
        assert_eq!(beta, vec![2.0]);
        assert!(model.value(&beta).unwrap().is_finite());
    }

    #[test]
    fn initial_parameters_leave_no_intercept_design_zero() {
        let y = vec![1.0, 2.0, 3.0];
        let x = DenseDesign::from_rows(&[[0.0], [1.0], [2.0]]);
        let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new(
            InitializingLocation,
            ParameterBlocks::from_assigned((mu,)),
            &y,
        )
        .unwrap();

        assert_eq!(model.initial_parameters().unwrap(), vec![0.0]);
    }

    #[test]
    fn initial_parameters_write_first_compatible_sum_term() {
        let y = vec![1.0, 2.0, 3.0];
        let first = LinearPredictorBlock::new(DenseDesign::from_rows(&[[0.0], [1.0], [2.0]]));
        let second = LinearPredictorBlock::new(DenseDesign::intercept(y.len()));
        let predictor = SumBlock::new((first, second));
        let mu = ParameterBlock::<Mu, _, _>::new(predictor, NoPenalty, 0);
        let model = Gamlss::try_new(
            InitializingLocation,
            ParameterBlocks::from_assigned((mu,)),
            &y,
        )
        .unwrap();

        assert_eq!(model.initial_parameters().unwrap(), vec![0.0, 2.0]);
    }

    #[test]
    fn initial_parameters_account_for_sum_block_constant_baselines() {
        let y = vec![1.0, 2.0, 3.0];
        let offset = OffsetBlock::new(y.len(), 10.0);
        let intercept = LinearPredictorBlock::new(DenseDesign::intercept(y.len()));
        let predictor = SumBlock::new((offset, intercept));
        let mu = ParameterBlock::<Mu, _, _>::new(predictor, NoPenalty, 0);
        let model = Gamlss::try_new(
            InitializingLocation,
            ParameterBlocks::from_assigned((mu,)),
            &y,
        )
        .unwrap();
        let beta = model.initial_parameters().unwrap();

        assert_eq!(beta, vec![-8.0]);
        assert_eq!(model.predict_eta(&beta).unwrap(), vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn initial_parameters_ignore_nonfinite_family_starts() {
        let y = vec![1.0, 2.0, 3.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new(
            NonFiniteInitializingLocation,
            ParameterBlocks::from_assigned((mu,)),
            &y,
        )
        .unwrap();

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
        let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
        let mut model = Gamlss::try_new_with_observations(
            FixedSigmaNormal,
            ParameterBlocks::from_assigned((mu,)),
            obs,
        )
        .unwrap();
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
        let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);

        assert_eq!(
            Gamlss::try_new_with_observations(
                FixedSigmaNormal,
                ParameterBlocks::from_assigned((mu,)),
                obs
            )
            .unwrap_err(),
            ModelError::InvalidWeight { index: 0 }
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn model_borrows_response_without_copying() {
        let y = vec![1.0, 2.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
        let model =
            Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();

        assert_eq!(model.obs.as_ptr(), y.as_ptr());
        assert_eq!(model.obs, y.as_slice());
        assert_eq!(model.obs.weight_at(0), 1.0);
    }

    #[test]
    fn unweighted_model_matches_unit_weights() {
        let y = vec![1.0, 2.0];
        let unit_weights = vec![1.0, 1.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, _, _>::linear(x.clone(), NoPenalty, 0);
        let weighted_mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
        let model =
            Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
        let weighted = Gamlss::try_new_weighted(
            FixedSigmaNormal,
            ParameterBlocks::from_assigned((weighted_mu,)),
            &y,
            &unit_weights,
        )
        .unwrap();
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
        let mu = ParameterBlock::<Mu, _, _>::linear(x.clone(), NoPenalty, 0);
        let weighted_mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
        let model =
            Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
        let weighted = Gamlss::try_new_weighted(
            FixedSigmaNormal,
            ParameterBlocks::from_assigned((weighted_mu,)),
            &y,
            &weights,
        )
        .unwrap();

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
        let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new_weighted(
            FixedSigmaNormal,
            ParameterBlocks::from_assigned((mu,)),
            &y,
            &weights,
        )
        .unwrap();
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
        let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new_weighted(
            FixedSigmaNormal,
            ParameterBlocks::from_assigned((mu,)),
            &y,
            &weights,
        )
        .unwrap();
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
        let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new_weighted(
            FixedSigmaNormal,
            ParameterBlocks::from_assigned((mu,)),
            &y,
            &weights,
        )
        .unwrap();
        let beta = vec![1.0, 1.0];
        let mut grad = vec![f64::NAN, f64::NAN];

        assert_relative_eq!(model.try_value(&beta).unwrap(), 0.0);

        model.try_gradient_into(&beta, &mut grad).unwrap();

        assert!(grad.iter().all(|value| value.is_finite()));
        assert_relative_eq!(grad[0], 0.0);
        assert_relative_eq!(grad[1], 0.0);

        for tile_rows in [1, 2] {
            let policy = ScoreTilePolicy::try_max_rows(tile_rows).unwrap();
            let mut workspace = model.gradient_workspace_with_policy(policy);
            let workspace_value = model
                .try_value_into_workspace(&beta, &mut workspace)
                .unwrap();
            let workspace_likelihood = model
                .try_likelihood_value_into_workspace(&beta, &mut workspace)
                .unwrap();
            assert_relative_eq!(workspace_value, 0.0);
            assert_relative_eq!(workspace_likelihood, 0.0);

            grad.fill(f64::NAN);
            let value = model
                .try_value_gradient_into_workspace(&beta, &mut grad, &mut workspace)
                .unwrap();

            assert_relative_eq!(value, 0.0);
            assert!(grad.iter().all(|value| value.is_finite()));
            assert_relative_eq!(grad[0], 0.0);
            assert_relative_eq!(grad[1], 0.0);
        }
    }

    #[test]
    fn weighted_model_rejects_invalid_weights() {
        let y = vec![1.0, 2.0];
        let short_weights = vec![1.0];
        let infinite_weights = vec![1.0, f64::INFINITY];
        let negative_weights = vec![1.0, -0.1];
        let mu = ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(2), NoPenalty, 0);

        assert_eq!(
            Gamlss::try_new_weighted(
                FixedSigmaNormal,
                ParameterBlocks::from_assigned((mu.clone(),)),
                &y,
                &short_weights,
            )
            .unwrap_err(),
            ModelError::WeightLength {
                expected: 2,
                actual: 1,
            }
        );
        assert_eq!(
            Gamlss::try_new_weighted(
                FixedSigmaNormal,
                ParameterBlocks::from_assigned((mu.clone(),)),
                &y,
                &infinite_weights,
            )
            .unwrap_err(),
            ModelError::InvalidWeight { index: 1 }
        );
        assert_eq!(
            Gamlss::try_new_weighted(
                FixedSigmaNormal,
                ParameterBlocks::from_assigned((mu,)),
                &y,
                &negative_weights
            )
            .unwrap_err(),
            ModelError::InvalidWeight { index: 1 }
        );
    }

    #[test]
    fn scalar_response_is_permissive_but_strict_constructor_rejects_non_finite() {
        let y = vec![1.0, f64::NAN];
        let mu = ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(2), NoPenalty, 0);

        Gamlss::try_new(
            FixedSigmaNormal,
            ParameterBlocks::from_assigned((mu.clone(),)),
            &y,
        )
        .unwrap();

        assert_eq!(
            Gamlss::try_new_strict(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y)
                .unwrap_err(),
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
        let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
        let model =
            Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
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
        let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
        let model =
            Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();

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
    fn dynamic_layout_identity_is_checked_for_models_and_prediction() {
        let canonical = RuntimeLayoutMock {
            key: 7,
            reverse: false,
            whole_paths: false,
            initial_count: 2,
        };
        let different_key = RuntimeLayoutMock {
            key: 8,
            reverse: false,
            whole_paths: false,
            initial_count: 2,
        };
        let different_coordinates = RuntimeLayoutMock {
            key: 7,
            reverse: true,
            whole_paths: false,
            initial_count: 2,
        };
        let canonical_eta = canonical.eta_from_flat(&[1.0, 2.0]);
        assert_relative_eq!(canonical_eta[0], 1.0);
        assert_relative_eq!(canonical_eta[1], 2.0);
        let reversed_eta = different_coordinates.eta_from_flat(&[1.0, 2.0]);
        assert_relative_eq!(reversed_eta[0], 2.0);
        assert_relative_eq!(reversed_eta[1], 1.0);
        let mut flat_gradient = [0.0; 2];
        different_coordinates.gradient_to_flat(&[3.0, 4.0], &mut flat_gradient);
        assert_relative_eq!(flat_gradient[0], 4.0);
        assert_relative_eq!(flat_gradient[1], 3.0);

        let y = [0.0];
        assert_eq!(
            Gamlss::try_new(
                different_key,
                runtime_layout_blocks(&canonical, y.len()),
                &y,
            )
            .unwrap_err(),
            ModelError::DynamicLayoutMismatch {
                expected: DynamicLayoutKey::new(vec![8, 0]),
                got: DynamicLayoutKey::new(vec![7, 0]),
            }
        );
        assert!(matches!(
            Gamlss::try_new(
                different_coordinates,
                runtime_layout_blocks(&canonical, y.len()),
                &y,
            )
            .unwrap_err(),
            ModelError::DynamicCoordinateMismatch { index: 0, .. }
        ));

        let model =
            Gamlss::try_new(canonical, runtime_layout_blocks(&canonical, y.len()), &y).unwrap();
        let prediction_blocks = runtime_layout_blocks(&different_key, 2);
        match model.prediction_view(&prediction_blocks).unwrap_err() {
            ModelError::PredictionLayoutIdentityMismatch {
                expected_key,
                got_key,
                expected_descriptors,
                got_descriptors,
            } => {
                assert_eq!(expected_key, Some(DynamicLayoutKey::new(vec![7, 0])));
                assert_eq!(got_key, Some(DynamicLayoutKey::new(vec![8, 0])));
                assert_eq!(expected_descriptors, got_descriptors);
            }
            error => panic!("unexpected prediction validation error: {error:?}"),
        }

        let prediction_blocks = runtime_layout_blocks(&different_coordinates, 2);
        assert!(matches!(
            model.prediction_view(&prediction_blocks).unwrap_err(),
            ModelError::PredictionLayoutIdentityMismatch {
                expected_key: Some(_),
                got_key: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn dynamic_initializer_coordinate_count_is_checked() {
        let family = RuntimeLayoutMock {
            key: 10,
            reverse: false,
            whole_paths: false,
            initial_count: 1,
        };
        let y = [0.0];
        let model = Gamlss::try_new(family, runtime_layout_blocks(&family, y.len()), &y).unwrap();

        assert_eq!(
            model.initial_parameters().unwrap_err(),
            ModelError::DynamicInitialValueCount {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn descriptor_selectors_reject_ambiguous_roles_and_paths() {
        let family = RuntimeLayoutMock {
            key: 9,
            reverse: false,
            whole_paths: true,
            initial_count: 2,
        };
        let y = [0.0];
        let mut model =
            Gamlss::try_new(family, runtime_layout_blocks(&family, y.len()), &y).unwrap();
        let beta = [0.25, -0.5];
        let layout = model.parameter_layout();

        assert_eq!(layout.ranges_of::<Mu>(), vec![0..1, 1..2]);
        assert_eq!(
            layout.unique_slice_of::<Mu>().unwrap_err(),
            ModelError::AmbiguousParameter {
                name: "mu".to_owned(),
                matches: 2,
            }
        );
        assert_eq!(
            model
                .unique_parameter_descriptor_at_path::<Mu>(&ParameterPath::whole())
                .unwrap_err(),
            ModelError::AmbiguousParameter {
                name: "mu".to_owned(),
                matches: 2,
            }
        );
        assert_eq!(
            model.block_objective_for::<Mu>(beta.to_vec()).unwrap_err(),
            ModelError::AmbiguousParameter {
                name: "mu".to_owned(),
                matches: 2,
            }
        );

        let descriptors = model.parameter_descriptors_of::<Mu>();
        assert_eq!(descriptors.len(), 2);
        assert_eq!(descriptors[0].0, 0);
        assert_eq!(descriptors[1].0, 1);
        assert_eq!(descriptors[0].1.path, ParameterPath::whole());
        assert_eq!(descriptors[1].1.path, ParameterPath::whole());
        assert_eq!(
            model.parameter_descriptor_index(&descriptors[1].1).unwrap(),
            Some(1)
        );

        let unpacked = model.unpack_parameters(&beta).unwrap();
        assert_eq!(unpacked.blocks_of::<Mu>().count(), 2);
        assert_eq!(unpacked.block_at(1).unwrap().descriptor, descriptors[1].1);
        assert_eq!(unpacked.block_at(1).unwrap().coefficients, vec![-0.5]);
        assert_eq!(
            unpacked.unique_coefficients_of::<Mu>().unwrap_err(),
            ModelError::AmbiguousParameter {
                name: "mu".to_owned(),
                matches: 2,
            }
        );

        {
            let mut second = model.block_objective_at(1, beta.to_vec()).unwrap();
            assert_eq!(second.descriptor(), &descriptors[1].1);
            let mut gradient = [0.0];
            let value = second.value_gradient(&[-0.5], &mut gradient).unwrap();
            assert_relative_eq!(value, 0.15625);
            assert_relative_eq!(gradient[0], -0.5);
        }
        {
            let mut second = model
                .block_objective_for_descriptor(&descriptors[1].1, beta.to_vec())
                .unwrap();
            assert_eq!(second.dim(), 1);
            assert_relative_eq!(second.value(&[-0.5]).unwrap(), 0.15625);
        }
        assert_eq!(
            model.block_objective_at(2, beta.to_vec()).unwrap_err(),
            ModelError::ParameterDescriptorIndexOutOfBounds { index: 2, count: 2 }
        );
        let unknown = ParameterDescriptor::whole("mu", 2..3);
        assert_eq!(
            model
                .block_objective_for_descriptor(&unknown, beta.to_vec())
                .unwrap_err(),
            ModelError::UnknownParameterDescriptor {
                descriptor: unknown,
            }
        );

        let selected = descriptors[1].1.clone();
        let mut workspace_objective = model.into_workspace_objective();
        assert_eq!(
            workspace_objective
                .block_objective_for::<Mu>(beta.to_vec())
                .unwrap_err(),
            ModelError::AmbiguousParameter {
                name: "mu".to_owned(),
                matches: 2,
            }
        );
        {
            let descriptor_objective = workspace_objective
                .block_objective_for_descriptor(&selected, beta.to_vec())
                .unwrap();
            assert_eq!(descriptor_objective.descriptor(), &selected);
        }
        {
            let indexed_objective = workspace_objective
                .block_objective_at(1, beta.to_vec())
                .unwrap();
            assert_eq!(indexed_objective.descriptor(), &selected);
        }
    }

    #[test]
    fn prediction_api_uses_compatible_blocks_for_new_rows() {
        let y = vec![1.0, 2.0];
        let train_x = DenseDesign::from_rows(&[[1.0, 0.0], [1.0, 1.0]]);
        let mu = ParameterBlock::<Mu, _, _>::linear(train_x, NoPenalty, 0);
        let model =
            Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
        let prediction_x = DenseDesign::from_rows(&[[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]);
        let prediction_mu = ParameterBlock::<Mu, _, _>::linear(prediction_x, NoPenalty, 0);
        let prediction_blocks = ParameterBlocks::from_assigned((prediction_mu,));
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
        let mu = ParameterBlock::<Mu, _, _>::linear(train_x, NoPenalty, 0);
        let model =
            Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
        let prediction_x = DenseDesign::from_rows(&[[1.0, 2.0, 3.0]]);
        let prediction_mu = ParameterBlock::<Mu, _, _>::linear(prediction_x, NoPenalty, 0);
        let prediction_blocks = ParameterBlocks::from_assigned((prediction_mu,));

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
        let train_mu = ParameterBlock::<Mu, _, _>::linear(train_mu_x, NoPenalty, 0);
        let train_sigma = ParameterBlock::<Sigma, _, _>::linear(train_sigma_x, NoPenalty, 2);
        let model = Gamlss::try_new(
            TwoParameterMock,
            ParameterBlocks::from_assigned((train_mu, train_sigma)),
            &y,
        )
        .unwrap();

        let prediction_mu_x = DenseDesign::intercept(1);
        let prediction_sigma_x = DenseDesign::from_rows(&[[1.0, 2.0]]);
        let prediction_mu = ParameterBlock::<Mu, _, _>::linear(prediction_mu_x, NoPenalty, 0);
        let prediction_sigma =
            ParameterBlock::<Sigma, _, _>::linear(prediction_sigma_x, NoPenalty, 1);
        let prediction_blocks = ParameterBlocks::from_assigned((prediction_mu, prediction_sigma));

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
        let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, usize::MAX);

        assert_eq!(
            Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y)
                .unwrap_err(),
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
        let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, usize::MAX);
        let blocks = ParameterBlocks::from_assigned((mu,));

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
        let mu = ParameterBlock::<Mu, _, _>::linear(x, RidgePenalty::new_unchecked(f64::NAN), 0);

        assert_eq!(
            Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y)
                .unwrap_err(),
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
        let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
        let model =
            Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
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
    fn score_tile_policy_flows_into_workspace_and_objective() {
        let y = vec![1.0; 7];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
        let model =
            Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
        let policy = ScoreTilePolicy::try_max_bytes(2 * size_of::<f64>()).unwrap();

        let default_workspace = model.gradient_workspace();
        assert_eq!(
            default_workspace.gradient().score_tile_policy(),
            ScoreTilePolicy::automatic()
        );
        assert_eq!(default_workspace.gradient().score_tile_rows(), y.len());

        let custom_workspace = model.gradient_workspace_with_policy(policy);
        assert_eq!(custom_workspace.gradient().score_tile_policy(), policy);
        assert_eq!(custom_workspace.gradient().score_tile_rows(), 2);
        assert_eq!(
            custom_workspace.gradient().score_tile_bytes(),
            2 * size_of::<f64>()
        );

        let objective = model.into_workspace_objective_with_policy(policy);
        assert_eq!(objective.workspace().gradient().score_tile_policy(), policy);
        assert_eq!(objective.workspace().gradient().score_tile_rows(), 2);
    }

    #[test]
    fn static_executor_is_additive_across_score_tile_sizes() {
        let y = vec![1.0, -0.5, 2.0, 0.25, 3.0, -1.0, 1.5];
        let weights = vec![0.5, 0.0, 2.0, 1.5, 0.25, 3.0, 0.75];
        let x = DenseDesign::from_rows(&[
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0],
            [1.0, 1.5],
            [1.0, 2.0],
            [1.0, 3.0],
        ]);
        let mu = ParameterBlock::<Mu, _, _>::linear(x, RidgePenalty::new_unchecked(0.2), 0);
        let model = Gamlss::try_new_weighted(
            FixedSigmaNormal,
            ParameterBlocks::from_assigned((mu,)),
            &y,
            &weights,
        )
        .unwrap();
        let beta = [0.4, -0.3];
        let mut expected_gradient = [0.0; 2];
        let expected_value = model
            .try_value_gradient_into(&beta, &mut expected_gradient)
            .unwrap();

        for tile_rows in [1, 2, 3, 5, 20] {
            let policy = ScoreTilePolicy::try_max_rows(tile_rows).unwrap();
            let mut workspace = model.gradient_workspace_with_policy(policy);
            for _ in 0..2 {
                let mut gradient = [f64::NAN; 2];
                let value = model
                    .try_value_gradient_into_workspace(&beta, &mut gradient, &mut workspace)
                    .unwrap();

                assert_relative_eq!(value, expected_value, epsilon = 1.0e-12);
                for (actual, expected) in gradient.iter().zip(expected_gradient) {
                    assert_relative_eq!(*actual, expected, epsilon = 1.0e-12);
                }
                assert_eq!(
                    workspace.gradient().score_tile_rows(),
                    tile_rows.min(y.len())
                );
                assert_eq!(
                    workspace.gradient().score_tile_value_count(),
                    tile_rows.min(y.len())
                );
            }
        }
    }

    #[test]
    fn dynamic_executor_is_additive_across_score_tile_sizes() {
        let family = RuntimeLayoutMock {
            key: 41,
            reverse: false,
            whole_paths: false,
            initial_count: 2,
        };
        let y = [1.0, -0.5, 2.0, 0.25, 3.0, -1.0, 1.5];
        let weights = [0.5, 0.0, 2.0, 1.5, 0.25, 3.0, 0.75];
        let model = Gamlss::try_new_weighted(
            family,
            runtime_layout_blocks(&family, y.len()),
            &y,
            &weights,
        )
        .unwrap();
        let beta = [0.4, -0.3];
        let mut expected_gradient = [0.0; 2];
        let expected_value = model
            .try_value_gradient_into(&beta, &mut expected_gradient)
            .unwrap();

        for tile_rows in [1, 2, 3, 5, 20] {
            let policy = ScoreTilePolicy::try_max_rows(tile_rows).unwrap();
            let mut workspace = model.gradient_workspace_with_policy(policy);
            let mut gradient = [f64::NAN; 2];
            let value = model
                .try_value_gradient_into_workspace(&beta, &mut gradient, &mut workspace)
                .unwrap();

            assert_relative_eq!(value, expected_value, epsilon = 1.0e-12);
            for (actual, expected) in gradient.iter().zip(expected_gradient) {
                assert_relative_eq!(*actual, expected, epsilon = 1.0e-12);
            }
            assert_eq!(
                workspace.gradient().score_tile_rows(),
                tile_rows.min(y.len())
            );
            assert_eq!(
                workspace.gradient().score_tile_value_count(),
                2 * tile_rows.min(y.len())
            );
        }
    }

    #[test]
    fn dynamic_workspace_value_and_pointwise_paths_stay_flat() {
        let family = RuntimeLayoutMock {
            key: usize::MAX,
            reverse: false,
            whole_paths: false,
            initial_count: 2,
        };
        let y = [0.0, 1.0];
        let weights = [2.0, 0.5];
        let model = Gamlss::try_new_weighted(
            family,
            runtime_layout_blocks(&family, y.len()),
            &y,
            &weights,
        )
        .unwrap();
        let beta = [0.5, -0.25];
        let expected_raw = [0.15625, 0.90625];
        let expected_value = 2.0_f64.mul_add(expected_raw[0], 0.5 * expected_raw[1]);
        let mut workspace = model.gradient_workspace();

        assert_relative_eq!(
            model
                .try_value_into_workspace(&beta, &mut workspace)
                .unwrap(),
            expected_value
        );
        assert_relative_eq!(
            model
                .try_likelihood_value_into_workspace(&beta, &mut workspace)
                .unwrap(),
            expected_value
        );

        let mut raw = [f64::NAN; 2];
        model.try_pointwise_nll_into(&beta, &mut raw).unwrap();
        for (actual, expected) in raw.iter().zip(expected_raw) {
            assert_relative_eq!(*actual, expected);
        }

        let mut weighted = [f64::NAN; 2];
        model
            .try_weighted_pointwise_log_likelihood_into_workspace(
                &beta,
                &mut weighted,
                &mut workspace,
            )
            .unwrap();
        for (actual, expected) in weighted
            .iter()
            .zip([-2.0 * expected_raw[0], -0.5 * expected_raw[1]])
        {
            assert_relative_eq!(*actual, expected);
        }

        let mut objective = model.into_workspace_objective();
        assert_relative_eq!(objective.value(&beta).unwrap(), expected_value);
        assert_relative_eq!(objective.value(&beta).unwrap(), expected_value);
    }

    #[test]
    fn value_gradient_matches_separate_value_and_gradient() {
        let y = vec![1.0, 2.0];
        let weights = vec![0.5, 2.0];
        let x = DenseDesign::from_rows(&[[1.0, 0.0], [1.0, 1.0]]);
        let mu = ParameterBlock::<Mu, _, _>::linear(x, RidgePenalty::new_unchecked(0.25), 0);
        let model = Gamlss::try_new_weighted(
            FixedSigmaNormal,
            ParameterBlocks::from_assigned((mu,)),
            &y,
            &weights,
        )
        .unwrap();
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
    fn likelihood_and_pointwise_apis_exclude_penalties_and_preserve_weights() {
        let y = vec![1.0, 2.0];
        let weights = vec![2.0, 0.5];
        let x = DenseDesign::from_rows(&[[1.0, 0.0], [1.0, 1.0]]);
        let mu = ParameterBlock::<Mu, _, _>::linear(x, RidgePenalty::new_unchecked(0.25), 0);
        let model = Gamlss::try_new_weighted(
            FixedSigmaNormal,
            ParameterBlocks::from_assigned((mu,)),
            &y,
            &weights,
        )
        .unwrap()
        .with_objective_scale(ObjectiveScale::Mean);
        let beta = vec![0.5, 0.25];
        let mut raw = vec![0.0; y.len()];
        let mut weighted = vec![0.0; y.len()];
        let mut log_likelihood = vec![0.0; y.len()];
        let mut likelihood_gradient = vec![0.0; beta.len()];

        model.try_pointwise_nll_into(&beta, &mut raw).unwrap();
        model
            .try_weighted_pointwise_nll_into(&beta, &mut weighted)
            .unwrap();
        model
            .try_pointwise_log_likelihood_into(&beta, &mut log_likelihood)
            .unwrap();
        let likelihood = model
            .try_likelihood_value_gradient_into(&beta, &mut likelihood_gradient)
            .unwrap();

        assert_relative_eq!(raw[0], 0.125);
        assert_relative_eq!(raw[1], 0.78125);
        assert_relative_eq!(weighted[0], weights[0] * raw[0]);
        assert_relative_eq!(weighted[1], weights[1] * raw[1]);
        assert_relative_eq!(log_likelihood[0], -raw[0]);
        assert_relative_eq!(log_likelihood[1], -raw[1]);
        assert_relative_eq!(likelihood, weighted.iter().sum::<f64>());
        assert_relative_eq!(model.try_likelihood_value(&beta).unwrap(), likelihood);
        assert_relative_eq!(likelihood_gradient[0], -1.625);
        assert_relative_eq!(likelihood_gradient[1], -0.625);

        // Sum likelihood is intentionally independent of optimizer mean scaling.
        assert_relative_eq!(likelihood, 0.640_625);
        assert_relative_eq!(
            model.try_value(&beta).unwrap(),
            likelihood / weights.iter().sum::<f64>() + 0.078_125
        );
    }

    #[test]
    fn mean_objective_scales_likelihood_but_not_penalties() {
        let y = vec![0.0, 3.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, _, _>::linear(x, RidgePenalty::new_unchecked(0.5), 0);
        let model = Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y)
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
        let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
        let model =
            Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
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
        fn add_gradient_range(
            &self,
            _: Range<usize>,
            scores: &[f64],
            beta: &[f64],
            grad: &mut [f64],
        ) {
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
        let mu = ParameterBlock::<Mu, _, _>::new(predictor, NoPenalty, 0);
        let mut model =
            Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
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

    crate::impl_scalar_compilable_family!(
        impl for StatefulLocation;
        parameters = (Mu,);
        arity = 1;
    );

    impl Family for StatefulLocation {
        type Eta = f64;
        type Theta = f64;
        type GradientEta = f64;
        type Observation<'obs> = f64;
        type Workspace = ();
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
        let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
        let mut model = Gamlss::try_new(
            StatefulLocation { target_shift: 1.0 },
            ParameterBlocks::from_assigned((mu,)),
            &y,
        )
        .unwrap();
        let beta = vec![0.5];
        let mut grad = vec![0.0];

        assert_relative_eq!(model.value(&beta).unwrap(), 0.125);

        model.gradient(&beta, &mut grad).unwrap();

        assert_relative_eq!(grad[0], -0.5);
    }

    #[derive(Debug, Clone, Copy)]
    struct BivariateLocation;

    crate::impl_scalar_compilable_family!(
        impl for BivariateLocation;
        parameters = (Mu,);
        arity = 1;
    );

    impl Family for BivariateLocation {
        type Eta = f64;
        type Theta = f64;
        type GradientEta = f64;
        type Observation<'obs> = [f64; 2];
        type Workspace = ();
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
        let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
        let mut model = Gamlss::try_new_with_observations(
            BivariateLocation,
            ParameterBlocks::from_assigned((mu,)),
            y.as_slice(),
        )
        .unwrap();
        let beta = vec![2.0];
        let mut grad = vec![0.0];

        assert_relative_eq!(model.value(&beta).unwrap(), 3.0);

        model.gradient(&beta, &mut grad).unwrap();

        assert_relative_eq!(grad[0], -2.0);
    }

    #[derive(Debug, Clone, Copy)]
    struct BorrowedRowMean;

    crate::impl_scalar_compilable_family!(
        impl for BorrowedRowMean;
        parameters = (Mu,);
        arity = 1;
    );

    impl Family for BorrowedRowMean {
        type Eta = f64;
        type Theta = f64;
        type GradientEta = f64;
        type Observation<'obs> = &'obs [f64];
        type Workspace = ();
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
        let values = [1.0, 3.0, 2.0, 4.0];
        let obs = DenseRows::try_new(&values, 2).unwrap();
        let x = DenseDesign::intercept(obs.len());
        let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
        let mut model = Gamlss::try_new_with_observations(
            BorrowedRowMean,
            ParameterBlocks::from_assigned((mu,)),
            obs,
        )
        .unwrap();
        let beta = vec![2.0];
        let mut grad = vec![0.0];

        assert_relative_eq!(model.value(&beta).unwrap(), 0.5);

        model.gradient(&beta, &mut grad).unwrap();

        assert_relative_eq!(grad[0], -1.0);
    }

    #[test]
    fn dense_rows_validate_geometry_and_weights_once() {
        let values = [1.0, 3.0, 2.0, 4.0];
        let weights = [2.0, 0.5];
        let rows = DenseRows::try_new_weighted(&values, 2, &weights).unwrap();

        assert_eq!(rows.values(), values);
        assert_eq!(rows.width(), 2);
        assert_eq!(rows.nrows(), 2);
        assert_eq!(rows.weights(), Some(weights.as_slice()));
        assert_eq!(rows.observation_at(1), &[2.0, 4.0]);
        assert_relative_eq!(rows.weight_at(1), 0.5);

        assert_eq!(
            DenseRows::try_new(&values, 0).unwrap_err(),
            ModelError::InvalidParameter {
                parameter: "dense observation row width",
                expected: "positive",
            }
        );
        assert_eq!(
            DenseRows::try_new(&values[..3], 2).unwrap_err(),
            ModelError::DenseObservationSize {
                actual_values: 3,
                row_width: 2,
            }
        );
        assert_eq!(
            DenseRows::try_new_weighted(&values, 2, &[1.0]).unwrap_err(),
            ModelError::WeightLength {
                expected: 2,
                actual: 1,
            }
        );
        assert_eq!(
            DenseRows::try_new_weighted(&values, 2, &[1.0, f64::NAN]).unwrap_err(),
            ModelError::InvalidWeight { index: 1 }
        );
    }

    #[derive(Debug, Clone, Copy)]
    struct ThreeParameterMock;

    crate::impl_scalar_compilable_family!(
        impl for ThreeParameterMock;
        parameters = (Mu, Sigma, Nu);
        arity = 3;
    );

    impl Family for ThreeParameterMock {
        type Eta = (f64, f64, f64);
        type Theta = (f64, f64, f64);
        type GradientEta = (f64, f64, f64);
        type Observation<'obs> = f64;
        type Workspace = ();
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
        let first =
            ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 0);
        let second =
            ParameterBlock::<Sigma, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 1);
        let third =
            ParameterBlock::<Nu, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 2);
        let mut model = Gamlss::try_new(
            ThreeParameterMock,
            ParameterBlocks::from_assigned((first, second, third)),
            &y,
        )
        .unwrap();
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

    crate::impl_scalar_compilable_family!(
        impl for FourParameterMock;
        parameters = (Mu, Sigma, Nu, Tau);
        arity = 4;
    );

    impl Family for FourParameterMock {
        type Eta = (f64, f64, f64, f64);
        type Theta = (f64, f64, f64, f64);
        type GradientEta = (f64, f64, f64, f64);
        type Observation<'obs> = f64;
        type Workspace = ();
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

    crate::impl_scalar_compilable_family!(
        impl for FiveParameterMock;
        parameters = (Mu, Sigma, Nu, Tau, Fifth);
        arity = 5;
    );

    impl Family for FiveParameterMock {
        type Eta = (f64, f64, f64, f64, f64);
        type Theta = (f64, f64, f64, f64, f64);
        type GradientEta = (f64, f64, f64, f64, f64);
        type Observation<'obs> = f64;
        type Workspace = ();
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
        let first =
            ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 0);
        let second =
            ParameterBlock::<Sigma, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 1);
        let third =
            ParameterBlock::<Nu, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 2);
        let fourth =
            ParameterBlock::<Tau, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 3);
        let model = Gamlss::try_new(
            FourParameterMock,
            ParameterBlocks::from_assigned((first, second, third, fourth)),
            &y,
        )
        .unwrap();
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

        assert_eq!(
            model.parameter_layout().unique_slice("fifth").unwrap(),
            Some(4..5)
        );
        assert_relative_eq!(grad[0], -0.5);
        assert_relative_eq!(grad[1], -0.5);
        assert_relative_eq!(grad[2], -0.5);
        assert_relative_eq!(grad[3], -0.5);
        assert_relative_eq!(grad[4], -0.5);
    }

    fn intercept_block<P>(
        nrows: usize,
    ) -> ParameterBlock<P, LinearPredictorBlock<DenseDesign>, NoPenalty> {
        ParameterBlock::linear(DenseDesign::intercept(nrows), NoPenalty, 99)
    }

    #[test]
    fn parameter_layout_and_unpack_use_distribution_parameter_names() {
        let y = vec![2.0];
        let first =
            ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 0);
        let second =
            ParameterBlock::<Sigma, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 1);
        let third =
            ParameterBlock::<Nu, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 2);
        let model = Gamlss::try_new(
            ThreeParameterMock,
            ParameterBlocks::from_assigned((first, second, third)),
            &y,
        )
        .unwrap();
        let parameters = vec![1.5, 0.5, -0.5];
        let layout = model.parameter_layout();
        let unpacked = model.unpack_parameters(&parameters).unwrap();

        assert_eq!(layout.len(), 3);
        assert!(!layout.is_empty());
        assert_eq!(layout.ncoefficients(), parameters.len());
        assert_eq!(layout.unique_slice("mu").unwrap(), Some(0..1));
        assert_eq!(layout.unique_slice_of::<Mu>().unwrap(), Some(0..1));
        assert_eq!(layout.unique_slice("sigma").unwrap(), Some(1..2));
        assert_eq!(layout.unique_slice_of::<Sigma>().unwrap(), Some(1..2));
        assert_eq!(layout.unique_slice("nu").unwrap(), Some(2..3));
        assert_eq!(layout.unique_slice_of::<Nu>().unwrap(), Some(2..3));
        assert_eq!(unpacked.unique_coefficients("mu").unwrap().unwrap(), &[1.5]);
        assert_eq!(
            unpacked.unique_coefficients_of::<Mu>().unwrap().unwrap(),
            &[1.5]
        );
        assert_eq!(
            unpacked.unique_block_of::<Mu>().unwrap().unwrap().name(),
            "mu"
        );
        assert_eq!(unpacked.block_at(0).unwrap().descriptor_index, 0);
        assert_eq!(
            unpacked.block_at(0).unwrap().descriptor,
            ParameterDescriptor::whole("mu", 0..1)
        );
        assert_eq!(
            unpacked
                .block_for_descriptor(&ParameterDescriptor::whole("sigma", 1..2))
                .unwrap()
                .coefficients,
            vec![0.5]
        );
        assert_eq!(
            unpacked.unique_coefficients("sigma").unwrap().unwrap(),
            &[0.5]
        );
        assert_eq!(
            unpacked.unique_coefficients("nu").unwrap().unwrap(),
            &[-0.5]
        );
    }

    #[test]
    fn visitor_apis_match_allocating_layout_helpers() {
        let y = vec![2.0];
        let first =
            ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 0);
        let second =
            ParameterBlock::<Sigma, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 1);
        let third =
            ParameterBlock::<Nu, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 2);
        let model = Gamlss::try_new(
            ThreeParameterMock,
            ParameterBlocks::from_assigned((first, second, third)),
            &y,
        )
        .unwrap();

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

        let mut visited_descriptors = Vec::new();
        model.visit_parameter_descriptors(|index, descriptor| {
            visited_descriptors.push((index, descriptor));
        });
        assert_eq!(
            visited_descriptors,
            model
                .parameter_descriptors()
                .into_iter()
                .enumerate()
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn training_diagnostics_report_train_nll_penalty_and_gradient_norm() {
        let y = vec![1.0, 2.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, _, _>::linear(x, RidgePenalty::new_unchecked(0.5), 0);
        let model =
            Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
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
        let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
        let mut model =
            Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y)
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
        let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
        let model =
            Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
        let penalty =
            HingeQuadraticPenalty::try_new(LinearFormBuilder::new().term(2, 1.0).build(), 1.0)
                .unwrap();

        assert_eq!(
            model.try_with_global_penalties(penalty).unwrap_err(),
            ModelError::PenaltyIndexOutOfBounds { index: 2, dim: 2 }
        );
    }

    #[test]
    fn workspace_try_with_global_penalties_validates_full_parameter_dimension() {
        let y = vec![0.0, 0.0];
        let x = DenseDesign::from_rows(&[[1.0, 0.0], [0.0, 1.0]]);
        let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y)
            .unwrap()
            .into_workspace_objective();
        let penalty =
            HingeQuadraticPenalty::try_new(LinearFormBuilder::new().term(2, 1.0).build(), 1.0)
                .unwrap();

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

        crate::impl_scalar_compilable_family!(
            impl for TwoParamMock;
            parameters = (Mu, Sigma);
            arity = 2;
        );

        impl Family for TwoParamMock {
            type Eta = (f64, f64);
            type Theta = (f64, f64);
            type GradientEta = (f64, f64);
            type Observation<'obs> = f64;
            type Workspace = ();
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
        let mu = ParameterBlock::<Mu, _, _>::linear(x_mu, NoPenalty, 0);
        let sigma = ParameterBlock::<Sigma, _, _>::linear(x_sigma, NoPenalty, 0);
        let (mu, sigma) = ParameterBlocks::new((mu, sigma)).into_inner();
        let mut model = Gamlss::try_new(
            TwoParamMock,
            ParameterBlocks::from_assigned((mu, sigma)),
            &y,
        )
        .unwrap();

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
        let mu = ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(2), NoPenalty, 0);
        let mut model =
            Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();

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
        let mu = ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(2), NoPenalty, 0);
        let model =
            Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
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
