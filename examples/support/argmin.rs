//! Shared adapter used by the optimizer examples.

use std::cell::RefCell;

use argmin::core::{CostFunction, Error, Gradient};
use gamlss::core::Objective;

// ANCHOR: adapter
/// Adapts a mutable, buffer-reusing objective to argmin's shared callbacks.
#[derive(Debug)]
pub struct ArgminObjective<O> {
    objective: RefCell<O>,
}

impl<O> ArgminObjective<O> {
    pub(super) const fn new(objective: O) -> Self {
        Self {
            objective: RefCell::new(objective),
        }
    }

    pub(super) fn dim(&self) -> usize
    where
        O: Objective,
    {
        self.objective.borrow().dim()
    }

    #[allow(
        dead_code,
        reason = "only the gas_like example uses the fused callback"
    )]
    pub(super) fn value_gradient(
        &mut self,
        parameters: &[f64],
        gradient: &mut [f64],
    ) -> Result<f64, O::Error>
    where
        O: Objective,
    {
        self.objective
            .get_mut()
            .value_gradient(parameters, gradient)
    }
}

impl<O> CostFunction for ArgminObjective<O>
where
    O: Objective,
    O::Error: std::error::Error + Send + Sync + 'static,
{
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, parameters: &Self::Param) -> Result<Self::Output, Error> {
        self.objective
            .borrow_mut()
            .value(parameters)
            .map_err(Error::new)
    }
}

impl<O> Gradient for ArgminObjective<O>
where
    O: Objective,
    O::Error: std::error::Error + Send + Sync + 'static,
{
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, parameters: &Self::Param) -> Result<Self::Gradient, Error> {
        let mut gradient = vec![0.0; self.dim()];
        self.objective
            .borrow_mut()
            .gradient(parameters, &mut gradient)
            .map_err(Error::new)?;
        Ok(gradient)
    }
}
// ANCHOR_END: adapter
