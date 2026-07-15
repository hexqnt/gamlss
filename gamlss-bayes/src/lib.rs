#![forbid(unsafe_code)]
//! Bayesian potential-energy wrappers kept separate from distribution families.
//!
//! This crate deliberately starts with one small, statistically explicit
//! vertical slice: normalized Gaussian priors on predictor coefficients plus
//! the summed, unpenalized compiled likelihood. It contains no sampler,
//! autodiff backend, chain storage, or implicit conversion from fitting
//! penalties to priors.

use gamlss_core::{
    Family, Gamlss, GamlssBlocks, ModelError, ModelWorkspace, Objective, ObservationView,
};

/// Common imports for the Bayesian vertical slice.
pub mod prelude {
    pub use crate::{
        CoefficientPrior, GaussianCoefficientPrior, PosteriorPotential, WorkspacePosteriorPotential,
    };
}

const HALF_LOG_TWO_PI: f64 = 0.918_938_533_204_672_7;

/// Independent normalized Gaussian priors on predictor coefficients.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianCoefficientPrior {
    means: Vec<f64>,
    standard_deviations: Vec<f64>,
}

impl GaussianCoefficientPrior {
    /// Creates a diagonal Gaussian prior after validating its parameters.
    pub fn try_new(means: Vec<f64>, standard_deviations: Vec<f64>) -> Result<Self, ModelError> {
        if means.len() != standard_deviations.len() {
            return Err(ModelError::InvalidParameter {
                parameter: "Gaussian prior",
                expected: "equal mean and standard-deviation lengths",
            });
        }
        if means.iter().any(|value| !value.is_finite()) {
            return Err(ModelError::InvalidParameter {
                parameter: "Gaussian prior mean",
                expected: "finite",
            });
        }
        if standard_deviations
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(ModelError::InvalidParameter {
                parameter: "Gaussian prior standard deviation",
                expected: "finite and positive",
            });
        }
        Ok(Self {
            means,
            standard_deviations,
        })
    }

    /// Creates an isotropic normalized Gaussian prior.
    pub fn isotropic(
        dimension: usize,
        mean: f64,
        standard_deviation: f64,
    ) -> Result<Self, ModelError> {
        Self::try_new(vec![mean; dimension], vec![standard_deviation; dimension])
    }

    /// Prior means in coefficient order.
    #[must_use]
    pub fn means(&self) -> &[f64] {
        &self.means
    }

    /// Prior standard deviations in coefficient order.
    #[must_use]
    pub fn standard_deviations(&self) -> &[f64] {
        &self.standard_deviations
    }
}

impl CoefficientPrior for GaussianCoefficientPrior {
    fn dimension(&self) -> usize {
        self.means.len()
    }

    fn negative_log_density(&self, coefficients: &[f64]) -> f64 {
        debug_assert_eq!(coefficients.len(), self.dimension());
        coefficients
            .iter()
            .zip(&self.means)
            .zip(&self.standard_deviations)
            .map(|((&coefficient, &mean), &standard_deviation)| {
                let standardized = (coefficient - mean) / standard_deviation;
                (0.5 * standardized).mul_add(standardized, standard_deviation.ln())
                    + HALF_LOG_TWO_PI
            })
            .sum()
    }

    fn add_gradient(&self, coefficients: &[f64], gradient: &mut [f64]) {
        debug_assert_eq!(coefficients.len(), self.dimension());
        debug_assert_eq!(gradient.len(), self.dimension());
        for (((gradient, coefficient), mean), standard_deviation) in gradient
            .iter_mut()
            .zip(coefficients)
            .zip(&self.means)
            .zip(&self.standard_deviations)
        {
            *gradient += (coefficient - mean) / standard_deviation.powi(2);
        }
    }

    fn negative_log_density_and_add_gradient(
        &self,
        coefficients: &[f64],
        gradient: &mut [f64],
    ) -> f64 {
        debug_assert_eq!(coefficients.len(), self.dimension());
        debug_assert_eq!(gradient.len(), self.dimension());
        let mut value = 0.0;
        for (((gradient, coefficient), mean), standard_deviation) in gradient
            .iter_mut()
            .zip(coefficients)
            .zip(&self.means)
            .zip(&self.standard_deviations)
        {
            let standardized = (coefficient - mean) / standard_deviation;
            value += (0.5 * standardized).mul_add(standardized, standard_deviation.ln())
                + HALF_LOG_TWO_PI;
            *gradient += standardized / standard_deviation;
        }
        value
    }
}

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

/// A normalized prior defined with respect to the flat coefficient vector.
pub trait CoefficientPrior {
    /// Number of coefficients covered by this prior.
    fn dimension(&self) -> usize;

    /// Negative normalized log-density at `coefficients`.
    ///
    /// Callers must pass exactly [`Self::dimension`] coefficients.
    fn negative_log_density(&self, coefficients: &[f64]) -> f64;

    /// Adds the negative-log-density gradient to `gradient`.
    ///
    /// Both slices must have length [`Self::dimension`]. Existing gradient
    /// entries are preserved and incremented rather than overwritten.
    fn add_gradient(&self, coefficients: &[f64], gradient: &mut [f64]);

    /// Computes the negative log-density and adds its gradient in one operation.
    ///
    /// The default preserves compatibility for custom priors. Implementations
    /// can override it when value and gradient share intermediate work.
    fn negative_log_density_and_add_gradient(
        &self,
        coefficients: &[f64],
        gradient: &mut [f64],
    ) -> f64 {
        let value = self.negative_log_density(coefficients);
        self.add_gradient(coefficients, gradient);
        value
    }

    /// Checked negative log-density evaluation for direct prior use.
    fn try_negative_log_density(&self, coefficients: &[f64]) -> Result<f64, ModelError> {
        if coefficients.len() != self.dimension() {
            return Err(ModelError::BetaLength {
                expected: self.dimension(),
                actual: coefficients.len(),
            });
        }
        Ok(self.negative_log_density(coefficients))
    }

    /// Checked gradient addition for direct prior use.
    fn try_add_gradient(
        &self,
        coefficients: &[f64],
        gradient: &mut [f64],
    ) -> Result<(), ModelError> {
        if coefficients.len() != self.dimension() {
            return Err(ModelError::BetaLength {
                expected: self.dimension(),
                actual: coefficients.len(),
            });
        }
        if gradient.len() != self.dimension() {
            return Err(ModelError::GradientLength {
                expected: self.dimension(),
                actual: gradient.len(),
            });
        }
        self.add_gradient(coefficients, gradient);
        Ok(())
    }

    /// Checked fused value/gradient operation for direct prior use.
    fn try_negative_log_density_and_add_gradient(
        &self,
        coefficients: &[f64],
        gradient: &mut [f64],
    ) -> Result<f64, ModelError> {
        if coefficients.len() != self.dimension() {
            return Err(ModelError::BetaLength {
                expected: self.dimension(),
                actual: coefficients.len(),
            });
        }
        if gradient.len() != self.dimension() {
            return Err(ModelError::GradientLength {
                expected: self.dimension(),
                actual: gradient.len(),
            });
        }
        Ok(self.negative_log_density_and_add_gradient(coefficients, gradient))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{
        DenseDesign, Gamlss, ModelError, Mu, NoPenalty, Objective, ParameterBlock, ParameterBlocks,
        Sigma,
    };
    use gamlss_family::NormalMuSigma;

    use super::{CoefficientPrior, GaussianCoefficientPrior, PosteriorPotential};

    #[test]
    fn normalized_gaussian_prior_includes_constant_and_gradient() {
        let prior = GaussianCoefficientPrior::try_new(vec![0.0, 1.0], vec![1.0, 2.0]).unwrap();
        let coefficients = [0.5, -1.0];
        let mut gradient = [0.0; 2];
        prior.add_gradient(&coefficients, &mut gradient);
        let mut fused_gradient = [0.0; 2];
        let fused_value = prior
            .try_negative_log_density_and_add_gradient(&coefficients, &mut fused_gradient)
            .unwrap();

        assert_relative_eq!(gradient[0], 0.5);
        assert_relative_eq!(gradient[1], -0.5);
        for (actual, expected) in fused_gradient.iter().zip(gradient) {
            assert_relative_eq!(*actual, expected);
        }
        let expected_value = 2.0_f64.mul_add(super::HALF_LOG_TWO_PI, 0.125 + 0.5) + 2.0_f64.ln();
        assert_relative_eq!(prior.negative_log_density(&coefficients), expected_value);
        assert_relative_eq!(fused_value, expected_value);

        assert_eq!(
            prior.try_negative_log_density(&[0.0]).unwrap_err(),
            ModelError::BetaLength {
                expected: 2,
                actual: 1,
            }
        );
        assert_eq!(
            prior
                .try_add_gradient(&coefficients, &mut [0.0])
                .unwrap_err(),
            ModelError::GradientLength {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn posterior_uses_summed_unpenalized_likelihood_and_matches_finite_difference() {
        let y = [0.0, 1.0, 2.0];
        let blocks = ParameterBlocks::new((
            ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 0),
            ParameterBlock::<Sigma, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 0),
        ));
        let model = Gamlss::try_new(NormalMuSigma::new(), blocks, &y).unwrap();
        let prior = GaussianCoefficientPrior::isotropic(model.nparams(), 0.0, 2.0).unwrap();
        let mut potential = PosteriorPotential::try_new(model, prior).unwrap();
        let coefficients = [0.2, -0.1];
        let mut gradient = [0.0; 2];
        let value = potential
            .value_gradient(&coefficients, &mut gradient)
            .unwrap();

        for index in 0..coefficients.len() {
            let mut plus = coefficients;
            let mut minus = coefficients;
            plus[index] += 1.0e-6;
            minus[index] -= 1.0e-6;
            let finite_difference =
                (potential.value(&plus).unwrap() - potential.value(&minus).unwrap()) / 2.0e-6;
            assert_relative_eq!(gradient[index], finite_difference, epsilon = 1.0e-6);
        }

        let mut pointwise = [0.0; 3];
        potential
            .pointwise_log_likelihood_into(&coefficients, &mut pointwise)
            .unwrap();
        let likelihood_nll = -pointwise.iter().sum::<f64>();
        assert_relative_eq!(
            value,
            likelihood_nll + potential.prior().negative_log_density(&coefficients)
        );
    }

    #[test]
    fn workspace_posterior_reuses_buffers_and_exposes_weighted_pointwise_values() {
        let y = [0.0, 1.0, 2.0];
        let weights = [2.0, 0.5, 0.0];
        let blocks = ParameterBlocks::new((
            ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 0),
            ParameterBlock::<Sigma, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 0),
        ));
        let model = Gamlss::try_new_weighted(NormalMuSigma::new(), blocks, &y, &weights).unwrap();
        let prior = GaussianCoefficientPrior::isotropic(model.nparams(), 0.0, 2.0).unwrap();
        let potential = PosteriorPotential::try_new(model, prior).unwrap();
        let coefficients = [0.2, -0.1];
        let mut expected_gradient = [0.0; 2];
        let expected_value = potential
            .try_value_gradient_into(&coefficients, &mut expected_gradient)
            .unwrap();

        let mut raw = [f64::NAN; 3];
        let mut weighted = [f64::NAN; 3];
        potential
            .raw_pointwise_log_likelihood_into(&coefficients, &mut raw)
            .unwrap();
        potential
            .weighted_pointwise_log_likelihood_into(&coefficients, &mut weighted)
            .unwrap();
        for ((weighted, raw), weight) in weighted.iter().zip(raw).zip(weights) {
            assert_relative_eq!(*weighted, weight * raw);
        }

        let mut workspace_potential = potential.into_workspace_objective();
        for _ in 0..2 {
            let mut gradient = [f64::NAN; 2];
            assert_relative_eq!(
                workspace_potential
                    .value_gradient(&coefficients, &mut gradient)
                    .unwrap(),
                expected_value
            );
            for (actual, expected) in gradient.iter().zip(expected_gradient) {
                assert_relative_eq!(*actual, expected);
            }
            assert_relative_eq!(
                workspace_potential.value(&coefficients).unwrap(),
                expected_value
            );
        }

        let mut workspace_weighted = [f64::NAN; 3];
        workspace_potential
            .weighted_pointwise_log_likelihood_into(&coefficients, &mut workspace_weighted)
            .unwrap();
        for (actual, expected) in workspace_weighted.iter().zip(weighted) {
            assert_relative_eq!(*actual, expected);
        }
        assert_relative_eq!(
            -weighted.iter().sum::<f64>()
                + workspace_potential
                    .potential()
                    .prior()
                    .negative_log_density(&coefficients),
            expected_value
        );
    }
}
