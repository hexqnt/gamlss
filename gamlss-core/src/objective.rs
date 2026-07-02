use crate::{ModelError, ParameterSlice};

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
    full_objective: &'a mut O,
    /// Full beta vector with the current block values patched in before each
    /// objective call.
    full_beta: Vec<f64>,
    /// Working full gradient vector.
    full_grad: Vec<f64>,
    /// Slice of the block being optimized.
    block: ParameterSlice,
}

impl<'a, O> BlockObjective<'a, O>
where
    O: Objective,
{
    /// Creates a block objective after validating all shape invariants.
    ///
    /// Prefer typed model helpers such as
    /// [`crate::Gamlss::block_objective_for`] when working with compiled
    /// models; they select the [`ParameterSlice`] from the model layout.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::BetaLength`] when `full_beta.len()` does not match
    /// `full_objective.dim()`. Returns [`ModelError::BlockRangeOutOfBounds`]
    /// when `block.range` is not contained in the full beta vector.
    pub fn try_new(
        full_objective: &'a mut O,
        full_beta: Vec<f64>,
        block: ParameterSlice,
    ) -> Result<Self, ModelError> {
        let full_grad = vec![0.0; full_objective.dim()];

        validate_block_range(&block, full_beta.len())?;
        if full_grad.len() != full_beta.len() {
            return Err(ModelError::BetaLength {
                expected: full_grad.len(),
                actual: full_beta.len(),
            });
        }

        Ok(Self {
            full_objective,
            full_beta,
            full_grad,
            block,
        })
    }

    fn update_block_beta(&mut self, block_beta: &[f64]) {
        self.full_beta[self.block.range.clone()].copy_from_slice(block_beta);
    }
}

impl<O> Objective for BlockObjective<'_, O>
where
    O: Objective,
    O::Error: From<ModelError>,
{
    type Error = O::Error;

    fn dim(&self) -> usize {
        self.block.range.len()
    }

    fn value(&mut self, block_beta: &[f64]) -> Result<f64, Self::Error> {
        validate_block_len("parameters", block_beta.len(), self.block.range.len())?;

        self.update_block_beta(block_beta);
        self.full_objective.value(&self.full_beta)
    }

    fn gradient(&mut self, block_beta: &[f64], grad: &mut [f64]) -> Result<(), Self::Error> {
        self.value_gradient(block_beta, grad).map(|_| ())
    }

    fn value_gradient(&mut self, block_beta: &[f64], grad: &mut [f64]) -> Result<f64, Self::Error> {
        validate_block_len("parameters", block_beta.len(), self.block.range.len())?;
        validate_block_len("gradient", grad.len(), self.block.range.len())?;

        self.update_block_beta(block_beta);
        let value = self
            .full_objective
            .value_gradient(&self.full_beta, &mut self.full_grad)?;
        grad.copy_from_slice(&self.full_grad[self.block.range.clone()]);
        Ok(value)
    }
}

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

const fn validate_block_range(block: &ParameterSlice, dim: usize) -> Result<(), ModelError> {
    if block.range.start <= block.range.end && block.range.end <= dim {
        Ok(())
    } else {
        Err(ModelError::BlockRangeOutOfBounds {
            parameter: block.name,
            start: block.range.start,
            end: block.range.end,
            dim,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{BlockObjective, Objective};
    use crate::{ModelError, ParameterSlice};

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
    #[allow(clippy::float_cmp)]
    fn block_objective_reuses_working_buffers_on_repeated_calls() {
        let mut full = QuadraticObjective { dim: 3 };
        let mut objective = BlockObjective::try_new(
            &mut full,
            vec![1.0, 2.0, 3.0],
            ParameterSlice {
                name: "sigma",
                range: 1..3,
            },
        )
        .unwrap();
        let full_beta_capacity = objective.full_beta.capacity();
        let grad_capacity = objective.full_grad.capacity();
        let mut grad = vec![0.0; objective.dim()];

        assert_eq!(objective.dim(), 2);
        assert_eq!(objective.value(&[4.0, 5.0]).unwrap(), 21.0);

        assert_eq!(
            objective.value_gradient(&[6.0, 7.0], &mut grad).unwrap(),
            43.0
        );

        assert_eq!(grad, vec![6.0, 7.0]);
        assert_eq!(objective.full_beta, vec![1.0, 6.0, 7.0]);
        assert_eq!(objective.full_beta.capacity(), full_beta_capacity);
        assert_eq!(objective.full_grad.capacity(), grad_capacity);

        assert_eq!(objective.value(&[8.0, 9.0]).unwrap(), 73.0);
        assert_eq!(objective.full_beta.capacity(), full_beta_capacity);
        assert_eq!(objective.full_grad.capacity(), grad_capacity);
    }

    #[test]
    fn try_new_rejects_wrong_full_beta_length() {
        let mut full = QuadraticObjective { dim: 3 };

        assert_eq!(
            BlockObjective::try_new(
                &mut full,
                vec![1.0, 2.0],
                ParameterSlice {
                    name: "mu",
                    range: 0..2,
                },
            )
            .unwrap_err(),
            ModelError::BetaLength {
                expected: 3,
                actual: 2,
            }
        );
    }

    #[test]
    fn try_new_rejects_block_outside_full_beta() {
        let mut full = QuadraticObjective { dim: 3 };

        assert_eq!(
            BlockObjective::try_new(
                &mut full,
                vec![1.0, 2.0, 3.0],
                ParameterSlice {
                    name: "mu",
                    range: 2..4,
                },
            )
            .unwrap_err(),
            ModelError::BlockRangeOutOfBounds {
                parameter: "mu",
                start: 2,
                end: 4,
                dim: 3,
            }
        );
    }
}
