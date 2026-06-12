use std::ops::Range;

use crate::ModelError;

/// Независимый от оптимизатора оракул над плоским вектором параметров.
///
/// Методы принимают `&mut self`, чтобы реализации могли переиспользовать
/// временные буферы, не раскрывая состояние, специфичное для оптимизатора,
/// в `gamlss-core`.
pub trait Objective {
    /// Recoverable error returned by objective evaluation.
    type Error;

    /// Dimension of the flat parameter vector accepted by this objective.
    fn dim(&self) -> usize;

    /// Objective value at `theta`.
    fn value(&mut self, theta: &[f64]) -> Result<f64, Self::Error>;

    /// Writes the gradient at `theta` into preallocated `grad`.
    fn gradient(&mut self, theta: &[f64], grad: &mut [f64]) -> Result<(), Self::Error>;

    /// Computes objective value and gradient at `theta`.
    fn value_gradient(&mut self, theta: &[f64], grad: &mut [f64]) -> Result<f64, Self::Error> {
        let value = self.value(theta)?;
        self.gradient(theta, grad)?;
        Ok(value)
    }
}

/// Objective по одному блоку коэффициентов при фиксированных остальных блоках.
///
/// Оборачивает полный objective и проецирует вызовы на диапазон одного
/// параметрического блока, копируя коэффициенты блока в общий `full_beta`
/// перед вычислением.
#[derive(Debug)]
pub struct BlockObjective<'a, O> {
    /// Полный objective.
    pub full_objective: &'a mut O,
    /// Текущий полный beta-вектор.
    pub full_beta: Vec<f64>,
    /// Диапазон оптимизируемого блока.
    pub block: Range<usize>,
}

impl<'a, O> BlockObjective<'a, O> {
    /// Создаёт block objective поверх полного objective.
    pub fn new(full_objective: &'a mut O, full_beta: Vec<f64>, block: Range<usize>) -> Self {
        Self {
            full_objective,
            full_beta,
            block,
        }
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

        let mut beta = self.full_beta.clone();
        beta[self.block.clone()].copy_from_slice(block_beta);
        self.full_objective.value(&beta)
    }

    fn gradient(&mut self, block_beta: &[f64], grad: &mut [f64]) -> Result<(), Self::Error> {
        validate_block_len("theta", block_beta.len(), self.block.len())?;
        validate_block_len("gradient", grad.len(), self.block.len())?;

        let mut beta = self.full_beta.clone();
        beta[self.block.clone()].copy_from_slice(block_beta);

        let mut full_grad = vec![0.0; self.full_objective.dim()];
        self.full_objective.gradient(&beta, &mut full_grad)?;
        grad.copy_from_slice(&full_grad[self.block.clone()]);
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
