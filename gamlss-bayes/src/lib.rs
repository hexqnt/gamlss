#![forbid(unsafe_code)]
//! Bayesian potential-energy wrappers kept separate from distribution families.
//!
//! This crate deliberately starts with one small, statistically explicit
//! vertical slice: normalized Gaussian priors on predictor coefficients plus
//! the summed, unpenalized compiled likelihood. It contains no sampler,
//! autodiff backend, chain storage, or implicit conversion from fitting
//! penalties to priors.

use gamlss_core::{Family, Gamlss, GamlssBlocks, ModelError, Objective, ObservationView};

/// Common imports for the Bayesian vertical slice.
pub mod prelude {
    pub use crate::{CoefficientPrior, GaussianCoefficientPrior, PosteriorPotential};
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
}

/// Potential energy `summed weighted NLL - log p(beta)`.
///
/// Local fitting penalties and [`gamlss_core::ObjectiveScale`] are ignored:
/// the wrapper always calls the model's summed unpenalized likelihood API.
/// Non-unit observation weights therefore have explicit power-likelihood
/// semantics.
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
        self.prior.add_gradient(coefficients, gradient);
        Ok(likelihood + self.prior.negative_log_density(coefficients))
    }

    /// Writes raw pointwise log-likelihood values for WAIC/LOO-style tooling.
    pub fn pointwise_log_likelihood_into(
        &self,
        coefficients: &[f64],
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        self.model
            .try_pointwise_log_likelihood_into(coefficients, out)
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
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{
        DenseDesign, Gamlss, Mu, NoPenalty, Objective, ParameterBlock, ParameterBlocks, Sigma,
    };
    use gamlss_family::NormalMuSigma;

    use super::{CoefficientPrior, GaussianCoefficientPrior, PosteriorPotential};

    #[test]
    fn normalized_gaussian_prior_includes_constant_and_gradient() {
        let prior = GaussianCoefficientPrior::try_new(vec![0.0, 1.0], vec![1.0, 2.0]).unwrap();
        let coefficients = [0.5, -1.0];
        let mut gradient = [0.0; 2];
        prior.add_gradient(&coefficients, &mut gradient);

        assert_relative_eq!(gradient[0], 0.5);
        assert_relative_eq!(gradient[1], -0.5);
        assert_relative_eq!(
            prior.negative_log_density(&coefficients),
            2.0_f64.mul_add(super::HALF_LOG_TWO_PI, 0.125 + 0.5) + 2.0_f64.ln()
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
}
