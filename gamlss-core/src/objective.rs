use std::ops::Range;

use crate::ModelError;

/// Optimizer-independent oracle over a flat parameter vector.
///
/// Methods accept `&mut self` so implementations can reuse temporary buffers
/// without exposing optimizer-specific state in `gamlss-core`.
///
/// Implementations validate input lengths and return recoverable errors for
/// shape mismatches. `value` and `gradient` are allowed to reuse internal
/// buffers; callers should not assume they are pure with respect to internal
/// cache state. `value_gradient` is the primary first-order hot path; the
/// default `gradient` wrapper exists for optimizer traits that request a
/// gradient-only callback.
pub trait Objective {
    /// Recoverable error returned by objective evaluation.
    type Error;

    /// Dimension of the flat parameter vector accepted by this objective.
    fn dim(&self) -> usize;

    /// Objective value at `parameters`.
    fn value(&mut self, parameters: &[f64]) -> Result<f64, Self::Error>;

    /// Computes objective value and gradient at `parameters`.
    ///
    /// Implementations overwrite the full gradient buffer after validating
    /// `grad.len() == dim()`. They should return an error instead of panicking
    /// for ordinary caller mistakes such as wrong vector length.
    fn value_gradient(&mut self, parameters: &[f64], grad: &mut [f64]) -> Result<f64, Self::Error>;

    /// Writes the gradient at `parameters` into preallocated `grad`.
    fn gradient(&mut self, parameters: &[f64], grad: &mut [f64]) -> Result<(), Self::Error> {
        self.value_gradient(parameters, grad).map(|_| ())
    }
}

/// Convenience adapter for optimizing one coefficient block at a time.
///
/// This is not part of the fundamental model representation. It exists for
/// block-wise fitting and optimizer integration code that wants to expose a
/// projected view of a full [`Objective`].
///
/// Wraps a full objective and projects calls onto the range of a single
/// parameter block, reusing working buffers across calls.
#[derive(Debug)]
pub struct BlockObjective<'a, O> {
    /// Full objective.
    pub full_objective: &'a mut O,
    /// Working full beta vector.
    pub working_beta: Vec<f64>,
    /// Working full gradient vector.
    pub full_grad: Vec<f64>,
    /// Range of the block being optimized.
    pub block: Range<usize>,
}

impl<'a, O> BlockObjective<'a, O>
where
    O: Objective,
{
    /// Creates a block objective over a full objective.
    ///
    /// This is a low-level constructor. Callers must pass a full parameter
    /// vector whose length equals `full_objective.dim()` and a block range
    /// contained in that vector. Prefer typed model helpers such as
    /// [`crate::Gamlss::block_objective_for`], which validate these invariants
    /// and return recoverable errors for ordinary caller mistakes.
    pub fn new(full_objective: &'a mut O, full_beta: Vec<f64>, block: Range<usize>) -> Self {
        let full_grad = vec![0.0; full_objective.dim()];
        debug_assert!(block.end <= full_beta.len());
        debug_assert!(block.end <= full_grad.len());
        debug_assert_eq!(full_beta.len(), full_grad.len());
        Self {
            full_objective,
            working_beta: full_beta,
            full_grad,
            block,
        }
    }

    fn update_block_beta(&mut self, block_beta: &[f64]) {
        self.working_beta[self.block.start..self.block.end].copy_from_slice(block_beta);
    }
}

impl<O> Objective for BlockObjective<'_, O>
where
    O: Objective,
    O::Error: From<ModelError>,
{
    type Error = O::Error;

    fn dim(&self) -> usize {
        self.block.len()
    }

    fn value(&mut self, block_beta: &[f64]) -> Result<f64, Self::Error> {
        validate_block_len("parameters", block_beta.len(), self.block.len())?;

        self.update_block_beta(block_beta);
        self.full_objective.value(&self.working_beta)
    }

    fn gradient(&mut self, block_beta: &[f64], grad: &mut [f64]) -> Result<(), Self::Error> {
        self.value_gradient(block_beta, grad).map(|_| ())
    }

    fn value_gradient(&mut self, block_beta: &[f64], grad: &mut [f64]) -> Result<f64, Self::Error> {
        validate_block_len("parameters", block_beta.len(), self.block.len())?;
        validate_block_len("gradient", grad.len(), self.block.len())?;

        self.update_block_beta(block_beta);
        let value = self
            .full_objective
            .value_gradient(&self.working_beta, &mut self.full_grad)?;
        grad.copy_from_slice(&self.full_grad[self.block.start..self.block.end]);
        Ok(value)
    }
}

fn validate_block_len(
    name: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), ModelError> {
    if actual == expected {
        Ok(())
    } else if name == "gradient" {
        Err(ModelError::GradientLength { expected, actual })
    } else {
        Err(ModelError::BetaLength { expected, actual })
    }
}

#[cfg(test)]
mod tests {
    use super::{BlockObjective, Objective};
    use crate::ModelError;

    #[derive(Debug)]
    struct QuadraticObjective {
        dim: usize,
    }

    impl Objective for QuadraticObjective {
        type Error = ModelError;

        fn dim(&self) -> usize {
            self.dim
        }

        fn value(&mut self, parameters: &[f64]) -> Result<f64, Self::Error> {
            Ok(0.5 * parameters.iter().map(|value| value * value).sum::<f64>())
        }

        fn value_gradient(
            &mut self,
            parameters: &[f64],
            grad: &mut [f64],
        ) -> Result<f64, Self::Error> {
            grad.copy_from_slice(parameters);
            self.value(parameters)
        }
    }

    #[test]
    fn block_objective_reuses_working_buffers_on_repeated_calls() {
        let mut full = QuadraticObjective { dim: 3 };
        let mut objective = BlockObjective::new(&mut full, vec![1.0, 2.0, 3.0], 1..3);
        let beta_capacity = objective.working_beta.capacity();
        let grad_capacity = objective.full_grad.capacity();
        let mut grad = vec![0.0; objective.dim()];

        assert_eq!(objective.dim(), 2);
        assert_eq!(objective.value(&[4.0, 5.0]).unwrap(), 21.0);

        assert_eq!(
            objective.value_gradient(&[6.0, 7.0], &mut grad).unwrap(),
            43.0
        );

        assert_eq!(grad, vec![6.0, 7.0]);
        assert_eq!(objective.working_beta, vec![1.0, 6.0, 7.0]);
        assert_eq!(objective.working_beta.capacity(), beta_capacity);
        assert_eq!(objective.full_grad.capacity(), grad_capacity);

        assert_eq!(objective.value(&[8.0, 9.0]).unwrap(), 73.0);
        assert_eq!(objective.working_beta.capacity(), beta_capacity);
        assert_eq!(objective.full_grad.capacity(), grad_capacity);
    }
}
