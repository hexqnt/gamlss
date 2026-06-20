//! "Gas-like" model example: fitting a normal GAMLSS (`y_i ~ N(μ_i, σ_i)`)
//! via L-BFGS from the `argmin` library.
//!
//! Demonstrates the crate's key concepts:
//! - Predictor composition via `SumBlock` (intercept + spline + linear effects).
//! - Link functions at the type level: `Identity` for μ (unconstrained),
//!   `ClampedLog<-12,12>` for σ (strict positivity + numerical protection).
//! - Penalties as part of `ParameterBlock`: `PreparedCyclicDifferencePenalty` on second
//!   differences smooths the seasonal volatility profile.
//! - The `ArgminObjective<O>` adapter: `RefCell` resolves the `&self`/`&mut self`
//!   incompatibility between argmin and gamlss-core without copying buffers on each call.

#![allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]

use std::cell::RefCell;

use argmin::{
    core::{CostFunction, Error, Executor, Gradient, State},
    solver::{linesearch::MoreThuenteLineSearch, quasinewton::LBFGS},
};
use gamlss::core::Objective;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use gamlss::core::{
        ClampedLog, DenseDesign, Gamlss, Identity, LinearPredictorBlock, Mu, NoPenalty,
        ParameterBlock, Sigma, SumBlock,
    };
    use gamlss::family::Normal;
    use gamlss::spline::{
        CyclicSplineDesign, OpenUniformSplineDesign, PreparedCyclicDifferencePenalty, SplineOrder,
    };

    // Synthetic data: the true relationship is quadratic in temperature
    // (U-shaped consumption profile), minus the weekend effect, plus a trend.
    let data = generate_surrogate_data(90);
    let n = data.y.len();

    // ── μ predictor: intercept + cubic B-spline of temperature (8 functions) +
    //                  weekend (binary) + linear trend.
    // Spline knots automatically placed at quantiles — no empty intervals.
    let mu_predictor = SumBlock::new((
        LinearPredictorBlock::new(DenseDesign::intercept(n)),
        OpenUniformSplineDesign::from_data(&data.temperature, 8, SplineOrder::Cubic)?,
        LinearPredictorBlock::new(DenseDesign::column(&data.weekend)),
        LinearPredictorBlock::new(DenseDesign::column(&data.time_x)),
    ));

    // ── σ predictor: cyclic cubic spline on year phase phi ∈ [0,1].
    // Cyclic condition: the December–January junction is smooth, without a gap.
    let sigma_predictor = CyclicSplineDesign::new(&data.phi, 8, SplineOrder::Cubic)?;

    // ParameterBlock links the predictor, link and penalty. The offset specifies
    // the block's position in the flat parameter vector θ — compile-time concatenation.
    let mu = ParameterBlock::<Mu, Identity, _, _>::new(mu_predictor, NoPenalty, 0);
    let sigma = ParameterBlock::<Sigma, ClampedLog<-12, 12>, _, _>::new(
        sigma_predictor,
        PreparedCyclicDifferencePenalty::new(0.05, 2), // λ=0.05, d=2 — smoothing second differences
        mu.len(),                                      // σ parameters follow μ parameters in θ
    );

    // try_new checks dimensional consistency (design matrix rows = y length,
    // offsets do not overlap). The model type is fully inferred by the compiler.
    let model = Gamlss::try_new(
        Normal::<Identity, ClampedLog<-12, 12>>::new(),
        (mu, sigma),
        &data.y,
    )?;

    let initial_parameters = model.initial_parameters()?;

    // into_workspace_objective reuses gradient buffers across optimizer calls.
    let problem = ArgminObjective::new(model.into_workspace_objective());

    // L-BFGS: quasi-Newton method, m=7 of the latest (s_k, y_k) pairs.
    // The More-Thuente line search guarantees the Wolfe conditions at every step.
    let solver = LBFGS::new(MoreThuenteLineSearch::new().with_c(1.0e-4, 0.9)?, 7)
        .with_tolerance_grad(1.0e-5)?
        .with_tolerance_cost(1.0e-10)?;

    let result = Executor::new(problem, solver)
        .configure(|state| state.param(initial_parameters).max_iters(100))
        .run()?;

    // Extract the final optimizer parameters and compute the gradient norm to check optimality.
    let parameters = result
        .state
        .get_best_param()
        .or_else(|| result.state.get_param())
        .expect("argmin returns a final parameter vector");
    let mut objective = result
        .problem
        .problem
        .expect("argmin returns the original problem");
    let mut grad = vec![0.0; objective.dim()];
    let loss = objective.value_gradient(parameters, &mut grad)?;

    println!(
        "gas_like argmin l-bfgs: loss={loss:.6}, iterations={}, grad_norm={:.6}",
        result.state.get_iter(),
        grad.iter().map(|v| v * v).sum::<f64>().sqrt()
    );
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// ARGMIN ADAPTER: RefCell resolves the &self (argmin) vs &mut self (Objective) conflict
// ═══════════════════════════════════════════════════════════════════════════
//
// Objective requires &mut self to reuse internal buffers (predictor vectors,
// gradient vectors). Allocating them anew on every call is unacceptable.
// argmin requires &self. RefCell moves the borrow check to runtime:
// since argmin calls cost/gradient strictly sequentially, there will be no panic.

#[derive(Debug)]
struct ArgminObjective<O> {
    objective: RefCell<O>,
}

impl<O> ArgminObjective<O> {
    const fn new(objective: O) -> Self {
        Self {
            objective: RefCell::new(objective),
        }
    }

    fn dim(&self) -> usize
    where
        O: Objective,
    {
        self.objective.borrow().dim()
    }

    fn value_gradient(&mut self, parameters: &[f64], grad: &mut [f64]) -> Result<f64, O::Error>
    where
        O: Objective,
    {
        self.objective.get_mut().value_gradient(parameters, grad)
    }
}

impl<O> CostFunction for ArgminObjective<O>
where
    O: Objective,
    O::Error: std::error::Error + Send + Sync + 'static,
{
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        self.objective.borrow_mut().value(param).map_err(Error::new)
    }
}

impl<O> Gradient for ArgminObjective<O>
where
    O: Objective,
    O::Error: std::error::Error + Send + Sync + 'static,
{
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
        let mut grad = vec![0.0; self.dim()];
        self.objective
            .borrow_mut()
            .gradient(param, &mut grad)
            .map_err(Error::new)?;
        Ok(grad)
    }
}

// Synthetic data: y = 3 - 0.08·T + 0.004·T² - 0.35·wknd + 0.2·trnd + noise.
struct SurrogateData {
    y: Vec<f64>,
    temperature: Vec<f64>,
    phi: Vec<f64>,
    weekend: Vec<f64>,
    time_x: Vec<f64>,
}

/// Generates surrogate data mimicking seasonal + weekend + trend
/// structure of gas consumption.
fn generate_surrogate_data(n: usize) -> SurrogateData {
    let mut y = Vec::with_capacity(n);
    let mut temperature = Vec::with_capacity(n);
    let mut phi = Vec::with_capacity(n);
    let mut weekend = Vec::with_capacity(n);
    let mut time_x = Vec::with_capacity(n);

    for day in 0..n {
        let day_f = day as f64;
        let seasonal = (std::f64::consts::TAU * day_f / n as f64).sin();
        let temp = 5.0 + 12.0 * seasonal;
        let weekend_value = f64::from(day % 7 >= 5);
        let trend = (day_f - (n - 1) as f64 / 2.0) / n as f64;
        let noise = 0.25 * (1.7 * day_f).sin();
        let mean = 3.0 - 0.08 * temp + 0.004 * temp * temp - 0.35 * weekend_value + 0.2 * trend;

        y.push(mean + noise);
        temperature.push(temp);
        phi.push(day_f / n as f64);
        weekend.push(weekend_value);
        time_x.push(trend);
    }

    SurrogateData {
        y,
        temperature,
        phi,
        weekend,
        time_x,
    }
}
