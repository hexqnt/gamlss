//! Fit the built-in `a1` data with a spline-based skew-normal GAMLSS.
//!
//! The conditional mean, standard deviation, and skewness parameter each get
//! their own cubic B-spline predictor. L-BFGS from `argmin` minimizes the
//! penalized mean negative log-likelihood using the analytic GAMLSS gradient.
//! Pass `--csv` to print the observations, fitted parameters, and 90% interval
//! as CSV instead of the compact fit summary.

#![allow(clippy::cast_precision_loss)]

use std::{cell::RefCell, io, ops::Range};

use argmin::{
    core::{CostFunction, Error, Executor, Gradient, State},
    solver::{linesearch::MoreThuenteLineSearch, quasinewton::LBFGS},
};
use gamlss::core::Objective;

const MEAN_BASIS: usize = 20;
const SIGMA_BASIS: usize = 14;
const NU_BASIS: usize = 12;
const MEAN_SMOOTHING: f64 = 0.05;
const SIGMA_SMOOTHING: f64 = 0.10;
const NU_SMOOTHING: f64 = 0.01;
const NU_STARTS: [f64; 2] = [-5.0, 5.0];
const LBFGS_MEMORY: usize = 10;
const MAX_ITERATIONS: u64 = 750;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputMode {
    Summary,
    Csv,
}

impl OutputMode {
    fn from_args() -> Result<Self, io::Error> {
        let arguments = std::env::args().skip(1).collect::<Vec<_>>();
        match arguments.as_slice() {
            [] => Ok(Self::Summary),
            [argument] if argument == "--csv" => Ok(Self::Csv),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "usage: cargo run --release --example a1_skew_normal [-- --csv]",
            )),
        }
    }
}

#[derive(Debug)]
struct Fit {
    parameters: Vec<f64>,
    objective: f64,
    iterations: u64,
    termination: String,
    nu_start: f64,
}

// ANCHOR: adapter
/// Single-threaded adapter from the mutable, buffer-reusing GAMLSS objective to
/// the shared-reference callbacks expected by argmin.
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
        let mut gradient = vec![0.0; self.dim()];
        self.objective
            .borrow_mut()
            .gradient(param, &mut gradient)
            .map_err(Error::new)?;
        Ok(gradient)
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use gamlss::core::{
        ClampedLog, Gamlss, HasQuantile, Identity, Mean, Nu, ObjectiveScale, ParameterBlock,
        ParameterBlocks, Sigma,
    };
    use gamlss::diagnostics::CdfDiagnosticsExt;
    use gamlss::family::SkewNormalMeanSd;
    use gamlss::spline::{OpenUniformSplineDesign, PreparedDifferencePenalty, SplineOrder};

    let output_mode = OutputMode::from_args()?;
    // ANCHOR: data
    let data = gamlss_datasets::a1();
    let (x, y) = (data.x, data.y);
    let n = y.len();
    // ANCHOR_END: data

    // ANCHOR: model
    // Each spline contains its own constant direction because the B-spline
    // rows sum to one. Second-difference penalties suppress local wiggles while
    // leaving constant and linear trends unpenalized.
    let mean_design = OpenUniformSplineDesign::from_data(x, MEAN_BASIS, SplineOrder::Cubic)?;
    let sigma_design = OpenUniformSplineDesign::from_data(x, SIGMA_BASIS, SplineOrder::Cubic)?;
    let nu_design = OpenUniformSplineDesign::from_data(x, NU_BASIS, SplineOrder::Cubic)?;

    let mean = ParameterBlock::<Mean, _, _>::new(
        mean_design,
        PreparedDifferencePenalty::try_new(MEAN_SMOOTHING, 2)?,
        0,
    );
    let sigma_offset = mean.len();
    let sigma = ParameterBlock::<Sigma, _, _>::new(
        sigma_design,
        PreparedDifferencePenalty::try_new(SIGMA_SMOOTHING, 2)?,
        sigma_offset,
    );
    let nu_offset = sigma_offset + sigma.len();
    let nu = ParameterBlock::<Nu, _, _>::new(
        nu_design,
        PreparedDifferencePenalty::try_new(NU_SMOOTHING, 2)?,
        nu_offset,
    );

    // The mean/SD parameterization keeps the first two fitted curves directly
    // interpretable even when the skewness parameter changes with x.
    let model = Gamlss::try_new(
        SkewNormalMeanSd::<Identity, ClampedLog<-8, 2>, Identity>::new(),
        ParameterBlocks::new((mean, sigma, nu)),
        y,
    )?
    .with_objective_scale(ObjectiveScale::Mean);
    // ANCHOR_END: model

    // ANCHOR: argmin
    // A constant start on the natural response scale is more useful than an
    // all-zero spline start. Equal B-spline coefficients produce a constant.
    let (sample_mean, sample_sd) = mean_sd(y);
    let mut constant_start = model.initial_parameters()?;
    constant_start[..sigma_offset].fill(sample_mean);
    constant_start[sigma_offset..nu_offset].fill(sample_sd.ln());

    // The adapter retains the reusable GAMLSS workspace behind RefCell because
    // argmin asks for `&self`, whereas Objective deliberately uses `&mut self`.
    // nu=0 is a stationary symmetric submodel in the mean/SD
    // parameterization. We therefore fit both non-zero signs and keep the
    // solution with the lower penalized objective.
    let nu_range = nu_offset..model.nparams();
    let candidate_fits = NU_STARTS
        .into_iter()
        .map(|nu_start| {
            fit_once(
                model.clone().into_workspace_objective(),
                &constant_start,
                nu_range.clone(),
                nu_start,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    // ANCHOR_END: argmin

    let fit = candidate_fits
        .into_iter()
        .min_by(|left, right| left.objective.total_cmp(&right.objective))
        .expect("NU_STARTS is non-empty");
    let parameters = fit.parameters;

    // ANCHOR: diagnostics
    let fitted = model.predict_theta(&parameters)?;
    let intervals_90 = fitted
        .iter()
        .map(|theta| {
            (
                model.family().quantile(0.05, theta),
                model.family().quantile(0.95, theta),
            )
        })
        .collect::<Vec<_>>();
    if output_mode == OutputMode::Csv {
        print_prediction_csv(x, y, &fitted, &intervals_90);
        return Ok(());
    }

    let diagnostics = model.training_diagnostics(&parameters)?;
    let pit = model.pit_values(&parameters)?;
    let residuals = model.quantile_residuals(&parameters)?;
    let coverage_90 = intervals_90
        .iter()
        .zip(y)
        .filter(|(interval, observation)| (interval.0..=interval.1).contains(observation))
        .count() as f64
        / n as f64;

    let (mean_min, mean_max) = range(fitted.iter().map(|theta| theta.mean));
    let (sigma_min, sigma_max) = range(fitted.iter().map(|theta| theta.sigma));
    let (nu_min, nu_max) = range(fitted.iter().map(|theta| theta.nu));
    let (pit_mean, pit_sd) = mean_sd(&pit);
    let (residual_mean, residual_sd) = mean_sd(&residuals);
    // ANCHOR_END: diagnostics

    println!(
        "a1 skew-normal spline fit: n={n}, parameters={}, starts={}, selected_nu_start={:.1}",
        parameters.len(),
        NU_STARTS.len(),
        fit.nu_start,
    );
    println!(
        "iterations={}, termination={}",
        fit.iterations, fit.termination,
    );
    println!(
        "objective={:.6}, mean_nll={:.6}, penalty={:.6}, grad_norm={:.3e}",
        diagnostics.objective,
        diagnostics.train_nll,
        diagnostics.penalty,
        diagnostics.gradient_norm,
    );
    println!(
        "fitted mean=[{mean_min:.4}, {mean_max:.4}], sigma=[{sigma_min:.4}, {sigma_max:.4}], nu=[{nu_min:.4}, {nu_max:.4}]",
    );
    println!(
        "PIT mean={pit_mean:.4}, PIT sd={pit_sd:.4}, qres mean={residual_mean:.4}, qres sd={residual_sd:.4}, 90% coverage={coverage_90:.3}",
    );
    Ok(())
}

fn mean_sd(values: &[f64]) -> (f64, f64) {
    debug_assert!(!values.is_empty());
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let variance = values
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>()
        / count;
    (mean, variance.sqrt())
}

fn range(mut values: impl Iterator<Item = f64>) -> (f64, f64) {
    let first = values.next().expect("fitted values are non-empty");
    values.fold((first, first), |(min, max), value| {
        (min.min(value), max.max(value))
    })
}

// ANCHOR: optimizer
fn fit_once<O>(
    objective: O,
    constant_start: &[f64],
    nu_range: Range<usize>,
    nu_start: f64,
) -> Result<Fit, Error>
where
    O: Objective,
    O::Error: std::error::Error + Send + Sync + 'static,
{
    let mut initial_parameters = constant_start.to_vec();
    initial_parameters[nu_range].fill(nu_start);

    let problem = ArgminObjective::new(objective);
    let line_search = MoreThuenteLineSearch::new().with_c(1.0e-4, 0.9)?;
    let solver = LBFGS::new(line_search, LBFGS_MEMORY)
        .with_tolerance_grad(1.0e-5)?
        .with_tolerance_cost(1.0e-8)?;
    let result = Executor::new(problem, solver)
        .configure(|state| state.param(initial_parameters).max_iters(MAX_ITERATIONS))
        .run()?;

    let parameters = result
        .state
        .get_best_param()
        .or_else(|| result.state.get_param())
        .cloned()
        .ok_or_else(|| Error::msg("argmin finished without a parameter vector"))?;

    Ok(Fit {
        parameters,
        objective: result.state.get_best_cost(),
        iterations: result.state.get_iter(),
        termination: result
            .state
            .get_termination_reason()
            .map_or_else(|| "unknown".to_owned(), ToString::to_string),
        nu_start,
    })
}
// ANCHOR_END: optimizer

fn print_prediction_csv(
    x: &[f64],
    y: &[f64],
    fitted: &[gamlss::family::SkewNormalMeanSdTheta],
    intervals_90: &[(f64, f64)],
) {
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), fitted.len());
    debug_assert_eq!(x.len(), intervals_90.len());
    println!("x,y,mean,sigma,nu,q05,q95");
    for (((x, observation), theta), &(q05, q95)) in x.iter().zip(y).zip(fitted).zip(intervals_90) {
        println!(
            "{x:.8},{observation:.8},{:.8},{:.8},{:.8},{q05:.8},{q95:.8}",
            theta.mean, theta.sigma, theta.nu,
        );
    }
}

// ANCHOR_END: adapter
