use std::ops::Range;

use crate::ModelError;

/// Независимый от оптимизатора оракул над плоским вектором параметров.
///
/// Методы принимают `&mut self`, чтобы реализации могли переиспользовать
/// временные буферы, не раскрывая состояние, специфичное для оптимизатора,
/// в `gamlss-core`.
///
/// Implementations validate input lengths and return recoverable errors for
/// shape mismatches. `value` and `gradient` are allowed to reuse internal
/// buffers; callers should not assume they are pure with respect to internal
/// cache state.
pub trait Objective {
    /// Recoverable error returned by objective evaluation.
    type Error;

    /// Dimension of the flat parameter vector accepted by this objective.
    fn dim(&self) -> usize;

    /// Objective value at `theta`.
    fn value(&mut self, theta: &[f64]) -> Result<f64, Self::Error>;

    /// Writes the gradient at `theta` into preallocated `grad`.
    ///
    /// Implementations overwrite the full gradient buffer after validating
    /// `grad.len() == dim()`. They should return an error instead of panicking
    /// for ordinary caller mistakes such as wrong vector length.
    fn gradient(&mut self, theta: &[f64], grad: &mut [f64]) -> Result<(), Self::Error>;

    /// Computes objective value and gradient at `theta`.
    fn value_gradient(&mut self, theta: &[f64], grad: &mut [f64]) -> Result<f64, Self::Error> {
        let value = self.value(theta)?;
        self.gradient(theta, grad)?;
        Ok(value)
    }
}

/// Convenience adapter for optimizing one coefficient block at a time.
///
/// This is not part of the fundamental model representation. It exists for
/// block-wise fitting and optimizer integration code that wants to expose a
/// projected view of a full [`Objective`].
///
/// Оборачивает полный objective и проецирует вызовы на диапазон одного
/// параметрического блока, переиспользуя рабочие буферы между вызовами.
#[derive(Debug)]
pub struct BlockObjective<'a, O> {
    /// Полный objective.
    pub full_objective: &'a mut O,
    /// Рабочий полный beta-вектор.
    pub working_beta: Vec<f64>,
    /// Рабочий полный gradient-вектор.
    pub full_grad: Vec<f64>,
    /// Диапазон оптимизируемого блока.
    pub block: Range<usize>,
}

impl<'a, O> BlockObjective<'a, O>
where
    O: Objective,
{
    /// Создаёт block objective поверх полного objective.
    pub fn new(full_objective: &'a mut O, full_beta: Vec<f64>, block: Range<usize>) -> Self {
        let full_grad = vec![0.0; full_objective.dim()];
        debug_assert!(block.end <= full_beta.len());
        debug_assert!(block.end <= full_grad.len());
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
        validate_block_len("theta", block_beta.len(), self.block.len())?;

        self.update_block_beta(block_beta);
        self.full_objective.value(&self.working_beta)
    }

    fn gradient(&mut self, block_beta: &[f64], grad: &mut [f64]) -> Result<(), Self::Error> {
        validate_block_len("theta", block_beta.len(), self.block.len())?;
        validate_block_len("gradient", grad.len(), self.block.len())?;

        self.update_block_beta(block_beta);
        self.full_objective
            .gradient(&self.working_beta, &mut self.full_grad)?;
        grad.copy_from_slice(&self.full_grad[self.block.start..self.block.end]);
        Ok(())
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

        fn value(&mut self, theta: &[f64]) -> Result<f64, Self::Error> {
            Ok(0.5 * theta.iter().map(|value| value * value).sum::<f64>())
        }

        fn gradient(&mut self, theta: &[f64], grad: &mut [f64]) -> Result<(), Self::Error> {
            grad.copy_from_slice(theta);
            Ok(())
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

        objective.gradient(&[6.0, 7.0], &mut grad).unwrap();

        assert_eq!(grad, vec![6.0, 7.0]);
        assert_eq!(objective.working_beta, vec![1.0, 6.0, 7.0]);
        assert_eq!(objective.working_beta.capacity(), beta_capacity);
        assert_eq!(objective.full_grad.capacity(), grad_capacity);

        assert_eq!(objective.value(&[8.0, 9.0]).unwrap(), 73.0);
        assert_eq!(objective.working_beta.capacity(), beta_capacity);
        assert_eq!(objective.full_grad.capacity(), grad_capacity);
    }
}
