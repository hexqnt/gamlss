use gamlss_core::ModelError;

use crate::HALF_LOG_TWO_PI;

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
    /// The default composes the separate value and gradient operations.
    /// Implementations can override it when both share intermediate work.
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
        validate_coefficients(self, coefficients)?;
        Ok(self.negative_log_density(coefficients))
    }

    /// Checked gradient addition for direct prior use.
    fn try_add_gradient(
        &self,
        coefficients: &[f64],
        gradient: &mut [f64],
    ) -> Result<(), ModelError> {
        validate_prior_inputs(self, coefficients, gradient)?;
        self.add_gradient(coefficients, gradient);
        Ok(())
    }

    /// Checked fused value/gradient operation for direct prior use.
    fn try_negative_log_density_and_add_gradient(
        &self,
        coefficients: &[f64],
        gradient: &mut [f64],
    ) -> Result<f64, ModelError> {
        validate_prior_inputs(self, coefficients, gradient)?;
        Ok(self.negative_log_density_and_add_gradient(coefficients, gradient))
    }
}

fn validate_coefficients<P: CoefficientPrior + ?Sized>(
    prior: &P,
    coefficients: &[f64],
) -> Result<(), ModelError> {
    if coefficients.len() != prior.dimension() {
        return Err(ModelError::BetaLength {
            expected: prior.dimension(),
            actual: coefficients.len(),
        });
    }
    Ok(())
}

fn validate_prior_inputs<P: CoefficientPrior + ?Sized>(
    prior: &P,
    coefficients: &[f64],
    gradient: &[f64],
) -> Result<(), ModelError> {
    validate_coefficients(prior, coefficients)?;
    if gradient.len() != prior.dimension() {
        return Err(ModelError::GradientLength {
            expected: prior.dimension(),
            actual: gradient.len(),
        });
    }
    Ok(())
}
