use gamlss_core::{
    Family, Gamlss, GamlssBlocks, ModelError, ModelWorkspace, Objective, ObservationView,
};

use crate::CoefficientPrior;

/// Potential energy `summed weighted NLL - log p(beta)`.
///
/// Local fitting penalties and [`gamlss_core::ObjectiveScale`] are ignored:
/// the wrapper always calls the model's summed unpenalized likelihood API.
/// Non-unit observation weights therefore have explicit power-likelihood
/// semantics. The [`Objective`] implementation is an allocating convenience;
/// repeated sampler or optimizer calls should use
/// [`Self::into_workspace_objective`].
#[derive(Debug, Clone)]
pub struct PosteriorPotential<F, Blocks, Obs, Prior> {
    model: Gamlss<F, Blocks, Obs>,
    prior: Prior,
}

impl<F, Blocks, Obs, Prior> PosteriorPotential<F, Blocks, Obs, Prior>
where
    F: Family,
    Blocks: GamlssBlocks<F>,
    for<'obs> Obs: ObservationView<'obs, Observation = F::Observation<'obs>>,
    Prior: CoefficientPrior,
{
    /// Creates a posterior potential whose prior covers the full beta vector.
    pub fn try_new(model: Gamlss<F, Blocks, Obs>, prior: Prior) -> Result<Self, ModelError> {
        if prior.dimension() != model.nparams() {
            return Err(ModelError::InvalidParameter {
                parameter: "coefficient prior dimension",
                expected: "equal to compiled model parameter dimension",
            });
        }
        Ok(Self { model, prior })
    }

    /// Borrow the compiled likelihood model.
    #[must_use]
    pub const fn model(&self) -> &Gamlss<F, Blocks, Obs> {
        &self.model
    }

    /// Borrow the normalized coefficient prior.
    #[must_use]
    pub const fn prior(&self) -> &Prior {
        &self.prior
    }

    /// Consumes the wrapper and returns its model and prior.
    #[must_use]
    pub fn into_parts(self) -> (Gamlss<F, Blocks, Obs>, Prior) {
        (self.model, self.prior)
    }

    /// Converts this potential into an objective with reusable model buffers.
    #[must_use]
    pub fn into_workspace_objective(self) -> WorkspacePosteriorPotential<F, Blocks, Obs, Prior> {
        let workspace = self.model.gradient_workspace();
        WorkspacePosteriorPotential {
            potential: self,
            workspace,
        }
    }

    /// Evaluates posterior potential energy for the full coefficient vector.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] when the coefficient length is incompatible with
    /// the compiled model.
    pub fn try_value(&self, coefficients: &[f64]) -> Result<f64, ModelError> {
        Ok(self.model.try_likelihood_value(coefficients)?
            + self.prior.negative_log_density(coefficients))
    }

    /// Workspace-reusing variant of [`Self::try_value`].
    pub fn try_value_into_workspace(
        &self,
        coefficients: &[f64],
        workspace: &mut ModelWorkspace<F>,
    ) -> Result<f64, ModelError> {
        Ok(self
            .model
            .try_likelihood_value_into_workspace(coefficients, workspace)?
            + self.prior.negative_log_density(coefficients))
    }

    /// Evaluates posterior potential energy and writes its analytical gradient.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] when the coefficient or gradient length is
    /// incompatible with the compiled model.
    pub fn try_value_gradient_into(
        &self,
        coefficients: &[f64],
        gradient: &mut [f64],
    ) -> Result<f64, ModelError> {
        let likelihood = self
            .model
            .try_likelihood_value_gradient_into(coefficients, gradient)?;
        Ok(likelihood
            + self
                .prior
                .negative_log_density_and_add_gradient(coefficients, gradient))
    }

    /// Evaluates posterior potential energy and gradient with reusable model buffers.
    pub fn try_value_gradient_into_workspace(
        &self,
        coefficients: &[f64],
        gradient: &mut [f64],
        workspace: &mut ModelWorkspace<F>,
    ) -> Result<f64, ModelError> {
        let likelihood = self.model.try_likelihood_value_gradient_into_workspace(
            coefficients,
            gradient,
            workspace,
        )?;
        Ok(likelihood
            + self
                .prior
                .negative_log_density_and_add_gradient(coefficients, gradient))
    }

    /// Writes raw pointwise log-likelihood values for WAIC/LOO-style tooling.
    ///
    /// Observation weights are deliberately excluded. This is the usual input
    /// when rows remain the pointwise units of the predictive diagnostic.
    pub fn raw_pointwise_log_likelihood_into(
        &self,
        coefficients: &[f64],
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        self.model
            .try_pointwise_log_likelihood_into(coefficients, out)
    }

    /// Writes power-likelihood contributions including observation weights.
    ///
    /// Use this variant only when the chosen diagnostic intentionally treats
    /// weights as likelihood powers or replication counts.
    pub fn weighted_pointwise_log_likelihood_into(
        &self,
        coefficients: &[f64],
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        self.model
            .try_weighted_pointwise_log_likelihood_into(coefficients, out)
    }

    /// Alias for [`Self::raw_pointwise_log_likelihood_into`].
    pub fn pointwise_log_likelihood_into(
        &self,
        coefficients: &[f64],
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        self.raw_pointwise_log_likelihood_into(coefficients, out)
    }
}

impl<F, Blocks, Obs, Prior> Objective for PosteriorPotential<F, Blocks, Obs, Prior>
where
    F: Family,
    Blocks: GamlssBlocks<F>,
    for<'obs> Obs: ObservationView<'obs, Observation = F::Observation<'obs>>,
    Prior: CoefficientPrior,
{
    type Error = ModelError;

    fn dim(&self) -> usize {
        self.model.nparams()
    }

    fn value(&mut self, coefficients: &[f64]) -> Result<f64, Self::Error> {
        self.try_value(coefficients)
    }

    fn value_gradient(
        &mut self,
        coefficients: &[f64],
        gradient: &mut [f64],
    ) -> Result<f64, Self::Error> {
        self.try_value_gradient_into(coefficients, gradient)
    }
}

/// Posterior potential with reusable likelihood buffers.
///
/// This is the primary objective adapter for repeated HMC, VI, or optimizer
/// evaluations. It preserves the simple beta-only posterior semantics of
/// [`PosteriorPotential`] while avoiding a fresh model workspace per call.
pub struct WorkspacePosteriorPotential<F: Family, Blocks, Obs, Prior> {
    potential: PosteriorPotential<F, Blocks, Obs, Prior>,
    workspace: ModelWorkspace<F>,
}

impl<F, Blocks, Obs, Prior> WorkspacePosteriorPotential<F, Blocks, Obs, Prior>
where
    F: Family,
    Blocks: GamlssBlocks<F>,
    for<'obs> Obs: ObservationView<'obs, Observation = F::Observation<'obs>>,
    Prior: CoefficientPrior,
{
    /// Borrows the underlying posterior definition.
    #[must_use]
    pub const fn potential(&self) -> &PosteriorPotential<F, Blocks, Obs, Prior> {
        &self.potential
    }

    /// Borrows the reusable likelihood workspace.
    #[must_use]
    pub const fn workspace(&self) -> &ModelWorkspace<F> {
        &self.workspace
    }

    /// Borrows the reusable likelihood workspace mutably.
    pub const fn workspace_mut(&mut self) -> &mut ModelWorkspace<F> {
        &mut self.workspace
    }

    /// Consumes the wrapper and returns its posterior definition and workspace.
    #[must_use]
    pub fn into_parts(self) -> (PosteriorPotential<F, Blocks, Obs, Prior>, ModelWorkspace<F>) {
        (self.potential, self.workspace)
    }

    /// Writes raw pointwise log-likelihood values while reusing flat buffers.
    pub fn raw_pointwise_log_likelihood_into(
        &mut self,
        coefficients: &[f64],
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        self.potential
            .model
            .try_pointwise_log_likelihood_into_workspace(coefficients, out, &mut self.workspace)
    }

    /// Writes observation-weighted power-likelihood contributions while reusing buffers.
    pub fn weighted_pointwise_log_likelihood_into(
        &mut self,
        coefficients: &[f64],
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        self.potential
            .model
            .try_weighted_pointwise_log_likelihood_into_workspace(
                coefficients,
                out,
                &mut self.workspace,
            )
    }
}

impl<F, Blocks, Obs, Prior> Objective for WorkspacePosteriorPotential<F, Blocks, Obs, Prior>
where
    F: Family,
    Blocks: GamlssBlocks<F>,
    for<'obs> Obs: ObservationView<'obs, Observation = F::Observation<'obs>>,
    Prior: CoefficientPrior,
{
    type Error = ModelError;

    fn dim(&self) -> usize {
        self.potential.model.nparams()
    }

    fn value(&mut self, coefficients: &[f64]) -> Result<f64, Self::Error> {
        self.potential
            .try_value_into_workspace(coefficients, &mut self.workspace)
    }

    fn gradient(&mut self, coefficients: &[f64], gradient: &mut [f64]) -> Result<(), Self::Error> {
        self.value_gradient(coefficients, gradient).map(|_| ())
    }

    fn value_gradient(
        &mut self,
        coefficients: &[f64],
        gradient: &mut [f64],
    ) -> Result<f64, Self::Error> {
        self.potential.try_value_gradient_into_workspace(
            coefficients,
            gradient,
            &mut self.workspace,
        )
    }
}
