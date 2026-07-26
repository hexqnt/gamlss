use gamlss_core::{MatrixPenalty, ModelError, Penalty};

use crate::kernel::{DifferenceOperator, delegate_scaled_penalty};
use crate::{PenaltyKernel, ScaledPenalty};

const EXPECTED_DIFFERENCE_WEIGHTS: &str = "finite and >= 0 with at least one positive value";

/// Normalized finite-difference kernel with one weight per difference row.
///
/// For weights `w_i` and `q = dim - order`, the quadratic form is
/// `sum_i w_i (Delta^order beta_i)^2 / q`. Unit weights therefore reproduce
/// [`crate::DifferencePenaltyKernel`] exactly, while spatially varying weights
/// support adaptive P-spline smoothing without changing the B-spline basis.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedDifferenceKernel {
    operator: DifferenceOperator,
    rank: usize,
    weights: Box<[f64]>,
}

impl WeightedDifferenceKernel {
    /// Creates a weighted difference kernel.
    ///
    /// `weights.len()` is the number of difference rows, so the coefficient
    /// dimension is `weights.len() + order`.
    ///
    /// # Errors
    ///
    /// Returns an error when `order` is zero, weights are empty, negative or
    /// non-finite, every weight is zero, or dimensions overflow.
    pub fn try_new(order: usize, weights: Vec<f64>) -> Result<Self, ModelError> {
        if weights.is_empty()
            || weights
                .iter()
                .any(|weight| !weight.is_finite() || *weight < 0.0)
            || !weights.iter().any(|weight| *weight > 0.0)
        {
            return Err(ModelError::InvalidParameter {
                parameter: "weighted difference row weights",
                expected: EXPECTED_DIFFERENCE_WEIGHTS,
            });
        }
        let dim = weights
            .len()
            .checked_add(order)
            .ok_or(ModelError::ArithmeticOverflow {
                context: "weighted difference coefficient dimension",
            })?;
        let operator = DifferenceOperator::try_new(dim, order)?;
        let rank = weights.iter().filter(|weight| **weight > 0.0).count();
        Ok(Self {
            operator,
            rank,
            weights: weights.into_boxed_slice(),
        })
    }

    /// Difference order.
    #[must_use]
    pub const fn order(&self) -> usize {
        self.operator.order()
    }

    /// Relative difference-row weights.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Cached finite-difference stencil.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        self.operator.coefficients()
    }
}

impl PenaltyKernel for WeightedDifferenceKernel {
    fn dim(&self) -> usize {
        self.operator.dim()
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn bandwidth(&self) -> Option<usize> {
        Some(self.order())
    }

    fn quadratic_form(&self, beta: &[f64]) -> f64 {
        self.operator
            .quadratic_form_by(beta, |row| self.weights[row])
    }

    fn add_product(&self, beta: &[f64], out: &mut [f64]) {
        self.add_scaled_product(1.0, beta, out);
    }

    fn add_scaled_product(&self, scale: f64, beta: &[f64], out: &mut [f64]) {
        self.operator
            .add_scaled_product_by(scale, beta, out, |row| self.weights[row]);
    }

    fn for_each_matrix_entry(&self, mut f: impl FnMut(usize, usize, f64)) {
        self.operator
            .for_each_matrix_entry_by(|row| self.weights[row], &mut f);
    }
}

/// Convenient scaled [`WeightedDifferenceKernel`] penalty.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedDifferencePenalty {
    inner: ScaledPenalty<WeightedDifferenceKernel>,
}

impl WeightedDifferencePenalty {
    /// Creates a spatially weighted difference penalty.
    pub fn try_new(lambda: f64, order: usize, weights: Vec<f64>) -> Result<Self, ModelError> {
        Ok(Self {
            inner: ScaledPenalty::try_new(
                lambda,
                WeightedDifferenceKernel::try_new(order, weights)?,
            )?,
        })
    }

    /// Smoothing scale.
    #[must_use]
    pub const fn lambda(&self) -> f64 {
        self.inner.lambda()
    }

    /// Unscaled weighted kernel.
    #[must_use]
    pub const fn kernel(&self) -> &WeightedDifferenceKernel {
        self.inner.kernel()
    }
}

delegate_scaled_penalty!(WeightedDifferencePenalty, inner);

/// Multiple spatial difference components with independent smoothing scales.
///
/// Each component supplies non-negative weights over the same difference
/// rows. This is a low-rank representation of a varying local smoothing
/// strength and exposes exact component derivatives with respect to each
/// `log(lambda_j)`.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveDifferencePenalty {
    components: Box<[ScaledPenalty<WeightedDifferenceKernel>]>,
}

impl AdaptiveDifferencePenalty {
    /// Creates an adaptive penalty from component scales and row-weight vectors.
    ///
    /// # Errors
    ///
    /// Returns an error for no components, mismatched scale/component counts,
    /// inconsistent row counts, or any invalid component.
    pub fn try_new(
        order: usize,
        lambdas: &[f64],
        component_weights: Vec<Vec<f64>>,
    ) -> Result<Self, ModelError> {
        if lambdas.is_empty() || lambdas.len() != component_weights.len() {
            return Err(ModelError::DesignSize {
                expected_values: component_weights.len().max(1),
                actual_values: lambdas.len(),
            });
        }
        let difference_count = component_weights[0].len();
        if component_weights
            .iter()
            .any(|weights| weights.len() != difference_count)
        {
            return Err(ModelError::InvalidParameter {
                parameter: "adaptive difference component lengths",
                expected: "equal and non-empty",
            });
        }
        let components = lambdas
            .iter()
            .copied()
            .zip(component_weights)
            .map(|(lambda, weights)| {
                ScaledPenalty::try_new(lambda, WeightedDifferenceKernel::try_new(order, weights)?)
            })
            .collect::<Result<Box<[_]>, ModelError>>()?;
        Ok(Self { components })
    }

    /// Coefficient dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.components[0].kernel().dim()
    }

    /// Difference order.
    #[must_use]
    pub fn order(&self) -> usize {
        self.components[0].kernel().order()
    }

    /// Number of independently scaled spatial components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.components.len()
    }

    /// One scaled component.
    #[must_use]
    pub fn component(&self, index: usize) -> Option<&ScaledPenalty<WeightedDifferenceKernel>> {
        self.components.get(index)
    }

    /// Returns a copy with new component scales and shared cloned kernels.
    pub fn with_lambdas(&self, lambdas: &[f64]) -> Result<Self, ModelError> {
        if lambdas.len() != self.components.len() {
            return Err(ModelError::DesignSize {
                expected_values: self.components.len(),
                actual_values: lambdas.len(),
            });
        }
        let components = self
            .components
            .iter()
            .zip(lambdas.iter().copied())
            .map(|(component, lambda)| ScaledPenalty::try_new(lambda, component.kernel().clone()))
            .collect::<Result<Box<[_]>, ModelError>>()?;
        Ok(Self { components })
    }

    /// Derivative of the total penalty value with respect to one
    /// `log(lambda_j)`.
    #[must_use]
    pub fn log_lambda_value_derivative(&self, component: usize, beta: &[f64]) -> Option<f64> {
        self.components
            .get(component)
            .map(|penalty| penalty.log_lambda_value_derivative(beta))
    }

    /// Adds the curvature derivative for one `log(lambda_j)`.
    pub fn add_log_lambda_matrix_derivative(
        &self,
        component: usize,
        dim: usize,
        out: &mut [f64],
    ) -> bool {
        let Some(component) = self.components.get(component) else {
            return false;
        };
        component.add_log_lambda_matrix_derivative(dim, out);
        true
    }
}

impl Penalty for AdaptiveDifferencePenalty {
    fn value(&self, beta: &[f64]) -> f64 {
        self.components
            .iter()
            .map(|component| component.value(beta))
            .sum()
    }

    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        for component in &self.components {
            component.add_gradient(beta, grad);
        }
    }

    fn validate_dim(&self, dim: usize) -> Result<(), ModelError> {
        for component in &self.components {
            component.validate_dim(dim)?;
        }
        Ok(())
    }
}

impl MatrixPenalty for AdaptiveDifferencePenalty {
    fn add_penalty_matrix(&self, dim: usize, gram: &mut [f64]) {
        for component in &self.components {
            component.add_penalty_matrix(dim, gram);
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{MatrixPenalty, Penalty};

    use super::{AdaptiveDifferencePenalty, WeightedDifferencePenalty};
    use crate::PreparedDifferencePenalty;

    #[test]
    fn unit_weighted_penalty_matches_standard_difference_penalty() {
        let weighted = WeightedDifferencePenalty::try_new(0.7, 2, vec![1.0; 4]).unwrap();
        let standard = PreparedDifferencePenalty::try_new(0.7, 2).unwrap();
        let beta = [0.2, -0.4, 0.8, 0.1, 0.9, -0.3];
        assert_relative_eq!(
            weighted.value(&beta),
            standard.value(&beta),
            epsilon = 1.0e-14
        );
        let mut weighted_gradient = [0.0; 6];
        let mut standard_gradient = [0.0; 6];
        weighted.add_gradient(&beta, &mut weighted_gradient);
        standard.add_gradient(&beta, &mut standard_gradient);
        for (weighted, standard) in weighted_gradient.into_iter().zip(standard_gradient) {
            assert_relative_eq!(weighted, standard, epsilon = 1.0e-14);
        }
    }

    #[test]
    fn adaptive_component_derivatives_match_log_scale_finite_differences() {
        let penalty = AdaptiveDifferencePenalty::try_new(
            1,
            &[0.4, 1.7],
            vec![vec![1.0, 0.7, 0.2, 0.0], vec![0.0, 0.3, 0.8, 1.0]],
        )
        .unwrap();
        let beta = [0.2, -0.4, 0.8, 0.1, -0.3];
        let step = 1.0e-6_f64;
        for component in 0..2 {
            let mut lower_lambdas = [0.4_f64, 1.7];
            let mut upper_lambdas = lower_lambdas;
            lower_lambdas[component] *= (-step).exp();
            upper_lambdas[component] *= step.exp();
            let lower = penalty.with_lambdas(&lower_lambdas).unwrap().value(&beta);
            let upper = penalty.with_lambdas(&upper_lambdas).unwrap().value(&beta);
            let actual = penalty
                .log_lambda_value_derivative(component, &beta)
                .unwrap();
            assert_relative_eq!(actual, (upper - lower) / (2.0 * step), epsilon = 2.0e-10);

            let mut derivative = [0.0; 25];
            assert!(penalty.add_log_lambda_matrix_derivative(component, 5, &mut derivative));
            let mut expected = [0.0; 25];
            penalty
                .component(component)
                .unwrap()
                .add_penalty_matrix(5, &mut expected);
            for (derivative, expected) in derivative.into_iter().zip(expected) {
                assert_relative_eq!(derivative, expected, epsilon = 1.0e-14);
            }
        }
    }
}
