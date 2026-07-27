//! Profile an implicit inflation target from monthly Bank of Russia data.
//!
//! The response is the monthly key-rate change from 2017 onward, after the
//! announced disinflation transition had ended. Exact zero changes get their
//! own point mass;
//! non-zero changes follow a Student-t distribution with a constant scale. Its
//! location uses a sign-constrained piecewise-linear reaction anchored to zero
//! at each candidate target, so the candidate cannot be absorbed by a free
//! intercept or rescued by a flexible target-specific curve. Target-independent
//! inflation-momentum and rate-inertia terms absorb part of the omitted dynamics.
//! Rolling-origin validation compares the lag and target. Rayon runs
//! independent outer folds in parallel while each target path remains ordered
//! for warm starts. The official 4% target is printed only after model selection
//! and is never used as a fitting input. The historical policy reference is the
//! [Bank of Russia inflation page](https://www.cbr.ru/hd_base/infl/).
//!
//! This is intentionally a substantial reproducible analysis rather than a
//! quick-start example. The two observed series cannot identify a causal
//! monetary-policy rule, and the reported one-standard-error range is a
//! predictive-stability heuristic rather than a confidence interval. If the
//! target-gap slopes collapse, the threshold is unidentified rather than
//! precisely estimated; this is the nuisance-parameter problem discussed by
//! [Hansen (1996)](https://users.ssc.wisc.edu/~behansen/papers/ecnmt_96.html).
//! The target-free comparison is deliberately labelled optimistic because the
//! best target is chosen on the same outer scores: failure to beat the null is
//! strong negative evidence, while success would require a nested confirmation.
//!
//! Run the optimized example with
//! `cargo run --release --example cbr_inflation_target`.

#![allow(clippy::cast_precision_loss)]

use std::{
    cell::RefCell,
    error::Error as StdError,
    fmt,
    ops::Range,
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};

use argmin::{
    core::{CostFunction, Error as ArgminError, Executor, Gradient, State, TerminationReason},
    solver::{linesearch::MoreThuenteLineSearch, quasinewton::LBFGS},
};
use gamlss::{
    core::{
        DenseDesign, Family, Gamlss, HasCdf, HasQuantile, LinearPredictorBlock, Mu,
        NegativeSoftplusScalar, NoPenalty, Objective, ObjectiveScale, ParameterBlock,
        ParameterBlocks, ProductBlock, Sigma, SoftplusScalar, SumBlock, ZeroProbability,
    },
    family::{ZeroAdjustedStudentTMuSigma, ZeroAdjustedStudentTTheta},
};
use gamlss_datasets::Date;
use rayon::prelude::*;

const REACTION_LAGS: [usize; 6] = [1, 2, 3, 6, 9, 12];
const MAX_REACTION_LAG: usize = 12;
const TARGET_MIN: f64 = 0.0;
const TARGET_MAX: f64 = 10.0;
const OFFICIAL_TARGET: f64 = 4.0;
const STUDENT_T_DF: f64 = 5.0;
const ANALYSIS_START_YEAR: i32 = 2017;
const SUPPORT_SENSITIVITY_COUNTS: [usize; 3] = [1, 3, 5];
const MIN_NONZERO_DECISIONS_PER_SIDE: usize = SUPPORT_SENSITIVITY_COUNTS[0];

// Annual validation blocks reduce dependence between scores and usually contain
// several non-zero policy decisions. The 54-month prefix plus five blocks uses
// the complete post-2017 sample.
const OUTER_INITIAL_TRAIN: usize = 54;
const OUTER_FOLDS: usize = 5;
const OUTER_FOLD_MONTHS: usize = 12;

const LBFGS_MEMORY: usize = 10;
const PRIMARY_MAX_ITERATIONS: u64 = 200;
const RETRY_MAX_ITERATIONS: u64 = 10_000;
const GRADIENT_TOLERANCE: f64 = 1.0e-5;
const COST_TOLERANCE: f64 = 1.0e-8;

type ExampleResult<T> = Result<T, Box<dyn StdError + Send + Sync>>;
type PolicyFamily = ZeroAdjustedStudentTMuSigma;
type PolicyTheta = ZeroAdjustedStudentTTheta;
type LinearTerm<'a> = LinearPredictorBlock<&'a DenseDesign>;
type AboveTargetTerm = ProductBlock<SoftplusScalar>;
type BelowTargetTerm = ProductBlock<NegativeSoftplusScalar>;
type MeanPredictor<'a> = SumBlock<(
    AboveTargetTerm,
    BelowTargetTerm,
    LinearTerm<'a>,
    LinearTerm<'a>,
)>;
type PolicyBlocks<'a> = ParameterBlocks<(
    ParameterBlock<Mu, MeanPredictor<'a>, NoPenalty>,
    ParameterBlock<Sigma, LinearTerm<'a>, NoPenalty>,
    ParameterBlock<ZeroProbability, LinearTerm<'a>, NoPenalty>,
)>;
type NullMeanPredictor<'a> = SumBlock<(LinearTerm<'a>, LinearTerm<'a>)>;
type NullPolicyBlocks<'a> = ParameterBlocks<(
    ParameterBlock<Mu, NullMeanPredictor<'a>, NoPenalty>,
    ParameterBlock<Sigma, LinearTerm<'a>, NoPenalty>,
    ParameterBlock<ZeroProbability, LinearTerm<'a>, NoPenalty>,
)>;

fn l2_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn main() -> ExampleResult<()> {
    let started = Instant::now();
    let data = PreparedData::load()?;
    let config = AnalysisConfig::full();
    let analysis = run_analysis(&data, &config)?;
    print_report(&data, &config, &analysis, started.elapsed());
    Ok(())
}

#[derive(Debug)]
struct ExampleError(String);

impl ExampleError {
    fn boxed(message: impl Into<String>) -> Box<dyn StdError + Send + Sync> {
        Box::new(Self(message.into()))
    }
}

impl fmt::Display for ExampleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl StdError for ExampleError {}

fn target_grid() -> Vec<f64> {
    (0..=100).map(|tenths| f64::from(tenths) / 10.0).collect()
}

#[derive(Clone, Debug)]
struct AnalysisConfig {
    lags: Vec<usize>,
    targets: Vec<f64>,
    initial_train: usize,
    outer_folds: usize,
    outer_fold_months: usize,
    show_progress: bool,
}

impl AnalysisConfig {
    fn full() -> Self {
        Self {
            lags: REACTION_LAGS.to_vec(),
            targets: target_grid(),
            initial_train: OUTER_INITIAL_TRAIN,
            outer_folds: OUTER_FOLDS,
            outer_fold_months: OUTER_FOLD_MONTHS,
            show_progress: true,
        }
    }

    fn analysis_len(&self) -> Option<usize> {
        self.outer_folds
            .checked_mul(self.outer_fold_months)
            .and_then(|validation| self.initial_train.checked_add(validation))
    }

    fn validate(&self, data: &PreparedData) -> ExampleResult<()> {
        if self.lags.is_empty()
            || self.targets.is_empty()
            || self.outer_folds == 0
            || self.outer_fold_months == 0
        {
            return Err(ExampleError::boxed(
                "analysis grids and fold counts must be non-empty",
            ));
        }
        let analysis_len = self
            .analysis_len()
            .ok_or_else(|| ExampleError::boxed("analysis row count overflowed"))?;
        if analysis_len > data.len() {
            return Err(ExampleError::boxed(format!(
                "analysis needs {analysis_len} rows but the prepared dataset has {}",
                data.len()
            )));
        }
        if self.lags.iter().any(|lag| !REACTION_LAGS.contains(lag)) {
            return Err(ExampleError::boxed(format!(
                "lags must be selected from {REACTION_LAGS:?}"
            )));
        }
        if self.lags.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(ExampleError::boxed("lags must be strictly increasing"));
        }
        if self
            .targets
            .iter()
            .any(|target| !target.is_finite() || !(TARGET_MIN..=TARGET_MAX).contains(target))
        {
            return Err(ExampleError::boxed(format!(
                "targets must be finite and inside [{TARGET_MIN}, {TARGET_MAX}]"
            )));
        }
        if self.targets.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(ExampleError::boxed("targets must be strictly increasing"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct PreparedData {
    source_start: Date,
    source_end: Date,
    dates: Vec<Date>,
    delta_rate: Vec<f64>,
    previous_delta_rate: Vec<f64>,
    lagged_inflation: Vec<Vec<f64>>,
    lagged_inflation_change: Vec<Vec<f64>>,
}

impl PreparedData {
    fn load() -> ExampleResult<Self> {
        let dataset = gamlss_datasets::cbr_inflation_and_interest_rate();
        let mut raw = dataset
            .x
            .iter()
            .copied()
            .zip(dataset.y.iter().copied())
            .collect::<Vec<_>>();
        raw.sort_by_key(|(date, _)| *date);

        if raw.len() <= MAX_REACTION_LAG {
            return Err(ExampleError::boxed(format!(
                "CBR dataset needs more than {MAX_REACTION_LAG} rows"
            )));
        }
        if raw.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
            return Err(ExampleError::boxed(
                "CBR dates must be unique and strictly increasing",
            ));
        }
        let source_start = raw[0].0;
        let source_end = raw[raw.len() - 1].0;

        let capacity = raw.len() - MAX_REACTION_LAG;
        let mut dates = Vec::with_capacity(capacity);
        let mut delta_rate = Vec::with_capacity(capacity);
        let mut previous_delta_rate = Vec::with_capacity(capacity);
        let mut lagged_inflation = REACTION_LAGS
            .iter()
            .map(|_| Vec::with_capacity(capacity))
            .collect::<Vec<_>>();
        let mut lagged_inflation_change = REACTION_LAGS
            .iter()
            .map(|_| Vec::with_capacity(capacity))
            .collect::<Vec<_>>();

        for current in MAX_REACTION_LAG..raw.len() {
            if raw[current].0.year() < ANALYSIS_START_YEAR {
                continue;
            }
            let [rate, _inflation] = raw[current].1;
            let [previous_rate, _previous_inflation] = raw[current - 1].1;
            let [twice_previous_rate, _twice_previous_inflation] = raw[current - 2].1;

            dates.push(raw[current].0);
            delta_rate.push(rate - previous_rate);
            previous_delta_rate.push(previous_rate - twice_previous_rate);
            for ((level_column, change_column), lag) in lagged_inflation
                .iter_mut()
                .zip(&mut lagged_inflation_change)
                .zip(REACTION_LAGS)
            {
                let lagged = raw[current - lag].1[1];
                let preceding = raw[current - lag - 1].1[1];
                level_column.push(lagged);
                change_column.push(lagged - preceding);
            }
        }

        Ok(Self {
            source_start,
            source_end,
            dates,
            delta_rate,
            previous_delta_rate,
            lagged_inflation,
            lagged_inflation_change,
        })
    }

    const fn len(&self) -> usize {
        self.dates.len()
    }

    fn lag_index(lag: usize) -> Option<usize> {
        REACTION_LAGS.iter().position(|candidate| *candidate == lag)
    }

    fn rows(&self, lag: usize, range: Range<usize>) -> ExampleResult<FeatureRows<'_>> {
        if range.start > range.end || range.end > self.len() {
            return Err(ExampleError::boxed(format!(
                "row range {range:?} is outside 0..{}",
                self.len()
            )));
        }
        let lag_index = Self::lag_index(lag)
            .ok_or_else(|| ExampleError::boxed(format!("unsupported reaction lag {lag}")))?;
        Ok(FeatureRows {
            response: &self.delta_rate[range.clone()],
            previous_delta_rate: &self.previous_delta_rate[range.clone()],
            inflation: &self.lagged_inflation[lag_index][range.clone()],
            inflation_change: &self.lagged_inflation_change[lag_index][range],
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct FeatureRows<'a> {
    response: &'a [f64],
    previous_delta_rate: &'a [f64],
    inflation: &'a [f64],
    inflation_change: &'a [f64],
}

impl FeatureRows<'_> {
    const fn len(self) -> usize {
        self.response.len()
    }

    const fn is_empty(self) -> bool {
        self.response.is_empty()
    }
}

fn outer_ranges(
    config: &AnalysisConfig,
    outer_fold: usize,
) -> ExampleResult<(Range<usize>, Range<usize>)> {
    if outer_fold >= config.outer_folds {
        return Err(ExampleError::boxed(format!(
            "outer fold {outer_fold} is outside 0..{}",
            config.outer_folds
        )));
    }
    let validation_start = outer_fold
        .checked_mul(config.outer_fold_months)
        .and_then(|offset| config.initial_train.checked_add(offset))
        .ok_or_else(|| ExampleError::boxed("outer fold boundary overflowed"))?;
    let validation_end = validation_start
        .checked_add(config.outer_fold_months)
        .ok_or_else(|| ExampleError::boxed("outer validation boundary overflowed"))?;
    Ok((0..validation_start, validation_start..validation_end))
}

fn common_supported_targets(
    data: &PreparedData,
    config: &AnalysisConfig,
    minimum_per_side: usize,
) -> ExampleResult<Vec<f64>> {
    if minimum_per_side == 0 {
        return Err(ExampleError::boxed(
            "target support requires at least one non-zero decision per side",
        ));
    }
    let (initial_train, _) = outer_ranges(config, 0)?;
    let mut supported = Vec::new();
    'target: for target in config.targets.iter().copied() {
        for lag in config.lags.iter().copied() {
            let rows = data.rows(lag, initial_train.clone())?;
            let mut below = 0usize;
            let mut above = 0usize;
            for (response, inflation) in rows.response.iter().zip(rows.inflation) {
                if *response == 0.0 {
                    continue;
                }
                below += usize::from(*inflation < target);
                above += usize::from(*inflation > target);
            }
            if below < minimum_per_side || above < minimum_per_side {
                continue 'target;
            }
        }
        supported.push(target);
    }

    if supported.is_empty() {
        return Err(ExampleError::boxed(format!(
            "no target has at least {minimum_per_side} initial-training non-zero decisions on each side for every lag"
        )));
    }
    Ok(supported)
}

#[derive(Clone, Debug)]
struct PolicyDesigns {
    intercept: DenseDesign,
    above: Vec<f64>,
    below: Vec<f64>,
    inflation_change: DenseDesign,
    previous_delta_rate: DenseDesign,
}

impl PolicyDesigns {
    fn new(rows: FeatureRows<'_>, target: f64) -> Self {
        let above = rows
            .inflation
            .iter()
            .map(|inflation| (inflation - target).max(0.0))
            .collect::<Vec<_>>();
        let below = rows
            .inflation
            .iter()
            .map(|inflation| (target - inflation).max(0.0))
            .collect::<Vec<_>>();
        Self {
            intercept: DenseDesign::intercept(rows.len()),
            above,
            below,
            inflation_change: DenseDesign::column(rows.inflation_change),
            previous_delta_rate: DenseDesign::column(rows.previous_delta_rate),
        }
    }

    fn blocks(&self) -> ExampleResult<PolicyBlocks<'_>> {
        let rows = self.above.len();
        let mean_predictor = SumBlock::new((
            ProductBlock::try_new(self.above.clone(), SoftplusScalar::new(rows))?,
            ProductBlock::try_new(self.below.clone(), NegativeSoftplusScalar::new(rows))?,
            LinearPredictorBlock::new(&self.inflation_change),
            LinearPredictorBlock::new(&self.previous_delta_rate),
        ));
        let mean = ParameterBlock::<Mu, _, _>::new(mean_predictor, NoPenalty, 0);

        let scale = ParameterBlock::<Sigma, _, _>::new(
            LinearPredictorBlock::new(&self.intercept),
            NoPenalty,
            0,
        );
        let zero_probability = ParameterBlock::<ZeroProbability, _, _>::new(
            LinearPredictorBlock::new(&self.intercept),
            NoPenalty,
            0,
        );

        Ok(ParameterBlocks::new((mean, scale, zero_probability)))
    }
}

#[derive(Clone, Debug)]
struct NullPolicyDesigns {
    intercept: DenseDesign,
    inflation_change: DenseDesign,
    previous_delta_rate: DenseDesign,
}

impl NullPolicyDesigns {
    fn new(rows: FeatureRows<'_>) -> Self {
        Self {
            intercept: DenseDesign::intercept(rows.len()),
            inflation_change: DenseDesign::column(rows.inflation_change),
            previous_delta_rate: DenseDesign::column(rows.previous_delta_rate),
        }
    }

    fn blocks(&self) -> NullPolicyBlocks<'_> {
        let mean_predictor = SumBlock::new((
            LinearPredictorBlock::new(&self.inflation_change),
            LinearPredictorBlock::new(&self.previous_delta_rate),
        ));
        let mean = ParameterBlock::<Mu, _, _>::new(mean_predictor, NoPenalty, 0);
        let scale = ParameterBlock::<Sigma, _, _>::new(
            LinearPredictorBlock::new(&self.intercept),
            NoPenalty,
            0,
        );
        let zero_probability = ParameterBlock::<ZeroProbability, _, _>::new(
            LinearPredictorBlock::new(&self.intercept),
            NoPenalty,
            0,
        );
        ParameterBlocks::new((mean, scale, zero_probability))
    }
}

fn policy_family() -> ExampleResult<PolicyFamily> {
    Ok(PolicyFamily::try_new(STUDENT_T_DF)?)
}

#[derive(Clone, Copy, Debug)]
struct SplitContext<'a> {
    train: FeatureRows<'a>,
    validation: FeatureRows<'a>,
}

impl<'a> SplitContext<'a> {
    fn for_outer_fold(
        data: &'a PreparedData,
        config: &AnalysisConfig,
        lag: usize,
        outer_fold: usize,
    ) -> ExampleResult<Self> {
        let (outer_train, outer_validation) = outer_ranges(config, outer_fold)?;
        Ok(Self {
            train: data.rows(lag, outer_train)?,
            validation: data.rows(lag, outer_validation)?,
        })
    }
}

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
    O::Error: StdError + Send + Sync + 'static,
{
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, parameters: &Self::Param) -> Result<Self::Output, ArgminError> {
        self.objective
            .borrow_mut()
            .value(parameters)
            .map_err(ArgminError::new)
    }
}

impl<O> Gradient for ArgminObjective<O>
where
    O: Objective,
    O::Error: StdError + Send + Sync + 'static,
{
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, parameters: &Self::Param) -> Result<Self::Gradient, ArgminError> {
        let mut gradient = vec![0.0; self.dim()];
        self.objective
            .borrow_mut()
            .gradient(parameters, &mut gradient)
            .map_err(ArgminError::new)?;
        Ok(gradient)
    }
}

#[derive(Clone, Debug)]
struct OptimizedFit {
    parameters: Vec<f64>,
    objective: f64,
    gradient_norm: f64,
    iterations: u64,
    termination: TerminationReason,
}

impl OptimizedFit {
    fn converged(&self) -> bool {
        (matches!(
            self.termination,
            TerminationReason::SolverConverged | TerminationReason::TargetCostReached
        ) || self.gradient_norm <= GRADIENT_TOLERANCE)
            && self.objective.is_finite()
            && self.gradient_norm.is_finite()
            && self.parameters.iter().all(|value| value.is_finite())
    }

    fn max_abs_parameter(&self) -> f64 {
        self.parameters
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct OptimizationStats {
    solver_runs: usize,
    retries: usize,
    iterations: u64,
}

impl OptimizationStats {
    const fn add(&mut self, other: Self) {
        self.solver_runs += other.solver_runs;
        self.retries += other.retries;
        self.iterations += other.iterations;
    }
}

fn run_lbfgs<O>(
    objective: O,
    initial_parameters: Vec<f64>,
    max_iterations: u64,
) -> Result<OptimizedFit, ArgminError>
where
    O: Objective,
    O::Error: StdError + Send + Sync + 'static,
{
    let problem = ArgminObjective::new(objective);
    let line_search = MoreThuenteLineSearch::new().with_c(1.0e-4, 0.9)?;
    let solver = LBFGS::new(line_search, LBFGS_MEMORY)
        .with_tolerance_grad(GRADIENT_TOLERANCE)?
        .with_tolerance_cost(COST_TOLERANCE)?;
    let result = Executor::new(problem, solver)
        .configure(|state| state.param(initial_parameters).max_iters(max_iterations))
        .run()?;
    let parameters = result
        .state
        .get_best_param()
        .or_else(|| result.state.get_param())
        .cloned()
        .ok_or_else(|| ArgminError::msg("argmin finished without a parameter vector"))?;
    let termination = result
        .state
        .get_termination_reason()
        .cloned()
        .unwrap_or_default();
    let gradient_norm = result
        .state
        .get_gradient()
        .map_or(f64::INFINITY, |gradient| l2_norm(gradient));

    Ok(OptimizedFit {
        parameters,
        objective: result.state.get_best_cost(),
        gradient_norm,
        iterations: result.state.get_iter(),
        termination,
    })
}

#[derive(Clone, Copy, Debug)]
struct ObjectiveEvaluation {
    value: f64,
    gradient_norm: f64,
}

fn evaluate_objective<O>(
    mut objective: O,
    parameters: &[f64],
) -> Result<ObjectiveEvaluation, ArgminError>
where
    O: Objective,
    O::Error: StdError + Send + Sync + 'static,
{
    let mut gradient = vec![0.0; objective.dim()];
    let value = objective
        .value_gradient(parameters, &mut gradient)
        .map_err(ArgminError::new)?;
    let gradient_norm = l2_norm(&gradient);
    Ok(ObjectiveEvaluation {
        value,
        gradient_norm,
    })
}

fn assess_fit<O>(fit: &mut OptimizedFit, objective: O) -> Result<(), ArgminError>
where
    O: Objective,
    O::Error: StdError + Send + Sync + 'static,
{
    let evaluation = evaluate_objective(objective, &fit.parameters)?;
    fit.objective = evaluation.value;
    fit.gradient_norm = evaluation.gradient_norm;
    Ok(())
}

fn optimize_with_retry<O, MakeObjective>(
    make_objective: MakeObjective,
    warm_start: Option<&[f64]>,
    data_start: &[f64],
    stats: &mut OptimizationStats,
) -> Result<OptimizedFit, ArgminError>
where
    O: Objective,
    O::Error: StdError + Send + Sync + 'static,
    MakeObjective: Fn() -> O,
{
    stats.solver_runs += 1;
    let primary = run_lbfgs(
        make_objective(),
        warm_start.unwrap_or(data_start).to_vec(),
        PRIMARY_MAX_ITERATIONS,
    )
    .and_then(|mut fit| {
        assess_fit(&mut fit, make_objective())?;
        Ok(fit)
    });
    if let Ok(fit) = primary.as_ref() {
        stats.iterations += fit.iterations;
        if fit.converged() {
            return primary;
        }
    }

    let primary_failure = match primary {
        Ok(fit) => format!(
            "terminated with {} after {} iterations (objective {:.6}, gradient norm {:.3e}, max |beta| {:.3e})",
            fit.termination,
            fit.iterations,
            fit.objective,
            fit.gradient_norm,
            fit.max_abs_parameter(),
        ),
        Err(error) => error.to_string(),
    };
    stats.solver_runs += 1;
    stats.retries += 1;
    let retry = run_lbfgs(make_objective(), data_start.to_vec(), RETRY_MAX_ITERATIONS)
        .and_then(|mut fit| {
            assess_fit(&mut fit, make_objective())?;
            Ok(fit)
        })
        .map_err(|error| {
            ArgminError::msg(format!(
                "primary fit failed ({primary_failure}); data-derived retry errored: {error}"
            ))
        })?;
    stats.iterations += retry.iterations;
    if retry.converged() {
        Ok(retry)
    } else {
        Err(ArgminError::msg(format!(
            "primary fit failed ({primary_failure}); data-derived retry terminated with {} after {} iterations (objective {:.6}, gradient norm {:.3e}, max |beta| {:.3e})",
            retry.termination,
            retry.iterations,
            retry.objective,
            retry.gradient_norm,
            retry.max_abs_parameter(),
        )))
    }
}

#[derive(Clone, Debug)]
struct ScoredFit {
    fit: OptimizedFit,
    validation_mean_nll: f64,
}

#[derive(Clone, Copy, Debug)]
struct DesignedSplit<'design, 'data> {
    train: FeatureRows<'data>,
    validation: FeatureRows<'data>,
    train_designs: &'design PolicyDesigns,
    validation_designs: &'design PolicyDesigns,
}

fn fit_and_score(
    family: PolicyFamily,
    split: DesignedSplit<'_, '_>,
    warm_start: Option<&[f64]>,
    optimization: &mut OptimizationStats,
) -> ExampleResult<ScoredFit> {
    if split.validation.is_empty() {
        return Err(ExampleError::boxed(
            "validation window must contain observations",
        ));
    }
    let blocks = split.train_designs.blocks()?;
    let model = Gamlss::try_new(family, blocks, split.train.response)?
        .with_objective_scale(ObjectiveScale::Mean);
    let cold_start = model.initial_parameters()?;
    let fit = optimize_with_retry(
        || model.clone().into_workspace_objective(),
        warm_start,
        &cold_start,
        optimization,
    )?;

    let validation_blocks = split.validation_designs.blocks()?;
    let fitted = model.predict_theta_with_blocks(&fit.parameters, &validation_blocks)?;
    let validation_mean_nll = mean_validation_nll(family, split.validation.response, &fitted)?;

    Ok(ScoredFit {
        fit,
        validation_mean_nll,
    })
}

#[derive(Clone, Copy, Debug)]
struct FoldEvaluation {
    validation_mean_nll: f64,
}

fn evaluate_target(
    family: PolicyFamily,
    context: &SplitContext<'_>,
    target: f64,
    warm_start: &mut Option<Vec<f64>>,
    stats: &mut OptimizationStats,
) -> ExampleResult<FoldEvaluation> {
    let train_designs = PolicyDesigns::new(context.train, target);
    let validation_designs = PolicyDesigns::new(context.validation, target);
    let scored = fit_and_score(
        family,
        DesignedSplit {
            train: context.train,
            validation: context.validation,
            train_designs: &train_designs,
            validation_designs: &validation_designs,
        },
        warm_start.as_deref(),
        stats,
    )
    .map_err(|error| ExampleError::boxed(format!("rolling refit: {error}")))?;
    *warm_start = Some(scored.fit.parameters);

    Ok(FoldEvaluation {
        validation_mean_nll: scored.validation_mean_nll,
    })
}

fn evaluate_null(
    family: PolicyFamily,
    context: &SplitContext<'_>,
    stats: &mut OptimizationStats,
) -> ExampleResult<f64> {
    if context.validation.is_empty() {
        return Err(ExampleError::boxed(
            "validation window must contain observations",
        ));
    }
    let train_designs = NullPolicyDesigns::new(context.train);
    let model = Gamlss::try_new(family, train_designs.blocks(), context.train.response)?
        .with_objective_scale(ObjectiveScale::Mean);
    let cold_start = model.initial_parameters()?;
    let fit = optimize_with_retry(
        || model.clone().into_workspace_objective(),
        None,
        &cold_start,
        stats,
    )?;

    let validation_designs = NullPolicyDesigns::new(context.validation);
    let fitted = model.predict_theta_with_blocks(&fit.parameters, &validation_designs.blocks())?;
    mean_validation_nll(family, context.validation.response, &fitted)
}

fn mean_validation_nll(
    family: PolicyFamily,
    observations: &[f64],
    fitted: &[PolicyTheta],
) -> ExampleResult<f64> {
    if observations.is_empty() || observations.len() != fitted.len() {
        return Err(ExampleError::boxed(
            "validation observations and predictions must have equal non-zero lengths",
        ));
    }
    let mean_nll = observations
        .iter()
        .copied()
        .zip(fitted)
        .map(|(observation, theta)| family.nll(observation, theta, &mut ()))
        .sum::<f64>()
        / observations.len() as f64;
    if !mean_nll.is_finite() {
        return Err(ExampleError::boxed(
            "validation negative log-likelihood is non-finite",
        ));
    }
    Ok(mean_nll)
}

fn central_target_index(targets: &[f64]) -> usize {
    let midpoint = f64::midpoint(targets[0], targets[targets.len() - 1]);
    targets
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            (*left - midpoint)
                .abs()
                .total_cmp(&(*right - midpoint).abs())
        })
        .map_or(0, |(index, _)| index)
}

#[derive(Clone, Debug)]
struct ProfilePoint {
    lag: usize,
    target: f64,
    fold_nll: Vec<f64>,
}

impl ProfilePoint {
    fn mean_se(&self) -> (f64, f64) {
        mean_and_standard_error(&self.fold_nll)
    }

    fn mean_excluding(&self, excluded_fold: Option<usize>) -> f64 {
        let mut count = 0usize;
        let sum = self
            .fold_nll
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, score)| (Some(index) != excluded_fold).then_some(score))
            .inspect(|_| count += 1)
            .sum::<f64>();
        debug_assert!(count > 0);
        sum / count as f64
    }
}

fn select_profile_index(profile: &[ProfilePoint], excluded_fold: Option<usize>) -> usize {
    (0..profile.len())
        .min_by(|left, right| {
            profile[*left]
                .mean_excluding(excluded_fold)
                .total_cmp(&profile[*right].mean_excluding(excluded_fold))
                .then_with(|| profile[*left].lag.cmp(&profile[*right].lag))
                .then_with(|| profile[*left].target.total_cmp(&profile[*right].target))
        })
        .expect("validated profile is non-empty")
}

fn lag_averaged_target_index(
    profile: &[ProfilePoint],
    lag_count: usize,
    target_count: usize,
    excluded_fold: Option<usize>,
) -> usize {
    debug_assert_eq!(profile.len(), lag_count * target_count);
    (0..target_count)
        .min_by(|left, right| {
            let mean = |target_index: usize| {
                (0..lag_count)
                    .map(|lag_index| {
                        profile[profile_index(target_count, lag_index, target_index)]
                            .mean_excluding(excluded_fold)
                    })
                    .sum::<f64>()
                    / lag_count as f64
            };
            mean(*left)
                .total_cmp(&mean(*right))
                .then_with(|| left.cmp(right))
        })
        .expect("validated target grid is non-empty")
}

#[derive(Clone, Copy, Debug)]
struct LagSummary {
    lag: usize,
    target: f64,
    mean_nll: f64,
    standard_error: f64,
    target_minus_null_mean: f64,
    target_minus_null_standard_error: f64,
}

#[derive(Clone, Copy, Debug)]
struct SupportSensitivity {
    minimum_per_side: usize,
    target_range: (f64, f64),
    target_count: usize,
    selected_lag: usize,
    selected_target: f64,
    lag_averaged_target: f64,
}

#[derive(Clone, Debug)]
struct FinalFitReport {
    parameters: usize,
    iterations: u64,
    termination: TerminationReason,
    objective: f64,
    train_nll: f64,
    penalty: f64,
    gradient_norm: f64,
    nonfinite_gradients: usize,
    mu_range: (f64, f64),
    sigma_range: (f64, f64),
    zero_probability_range: (f64, f64),
    structural_response_one_point_below: f64,
    structural_response_one_point_above: f64,
    pit_mean_sd: (f64, f64),
    residual_mean_sd: (f64, f64),
    coverage_90: f64,
}

#[derive(Clone, Debug)]
struct AnalysisResult {
    candidate_target_range: (f64, f64),
    candidate_target_count: usize,
    lag_summaries: Vec<LagSummary>,
    selected_lag: usize,
    selected_target: f64,
    selected_mean_nll: f64,
    selected_standard_error: f64,
    paired_one_se_range: (f64, f64),
    lag_averaged_target: f64,
    selected_lag_nll_span: f64,
    lag_averaged_nll_span: f64,
    selected_target_minus_null_mean: f64,
    selected_target_minus_null_standard_error: f64,
    support_sensitivity: Vec<SupportSensitivity>,
    leave_one_fold_out_selections: Vec<(usize, f64)>,
    final_fit: FinalFitReport,
    optimization: OptimizationStats,
}

const fn profile_index(target_count: usize, lag_index: usize, target_index: usize) -> usize {
    lag_index * target_count + target_index
}

struct TargetScanner<'scan, 'data> {
    family: PolicyFamily,
    context: &'scan SplitContext<'data>,
    config: &'scan AnalysisConfig,
    outer_fold: usize,
    lag: usize,
    lag_index: usize,
    scores: &'scan mut [f64],
    optimization: &'scan mut OptimizationStats,
}

impl TargetScanner<'_, '_> {
    fn evaluate<I>(
        &mut self,
        target_indices: I,
        warm_start: &mut Option<Vec<f64>>,
    ) -> ExampleResult<()>
    where
        I: IntoIterator<Item = usize>,
    {
        let target_count = self.config.targets.len();
        for target_index in target_indices {
            let target = self.config.targets[target_index];
            let evaluation = evaluate_target(
                self.family,
                self.context,
                target,
                warm_start,
                self.optimization,
            )
            .map_err(|error| {
                ExampleError::boxed(format!(
                    "outer fold {}, lag {}, target {target:.1}: {error}",
                    self.outer_fold + 1,
                    self.lag,
                ))
            })?;
            self.scores[profile_index(target_count, self.lag_index, target_index)] =
                evaluation.validation_mean_nll;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct OuterFoldResult {
    scores: Vec<f64>,
    null_scores: Vec<f64>,
    optimization: OptimizationStats,
}

fn evaluate_outer_fold(
    data: &PreparedData,
    config: &AnalysisConfig,
    family: PolicyFamily,
    outer_fold: usize,
) -> ExampleResult<OuterFoldResult> {
    let targets_len = config.targets.len();
    let mut scores = vec![f64::NAN; config.lags.len() * targets_len];
    let mut null_scores = vec![f64::NAN; config.lags.len()];
    let mut optimization = OptimizationStats::default();

    for (lag_index, lag) in config.lags.iter().copied().enumerate() {
        let context = SplitContext::for_outer_fold(data, config, lag, outer_fold)?;
        null_scores[lag_index] =
            evaluate_null(family, &context, &mut optimization).map_err(|error| {
                ExampleError::boxed(format!(
                    "outer fold {}, lag {lag}, target-free model: {error}",
                    outer_fold + 1,
                ))
            })?;
        let center = central_target_index(&config.targets);
        let mut center_warm = None;
        let mut scanner = TargetScanner {
            family,
            context: &context,
            config,
            outer_fold,
            lag,
            lag_index,
            scores: &mut scores,
            optimization: &mut optimization,
        };
        scanner.evaluate(std::iter::once(center), &mut center_warm)?;

        let mut lower_warm = center_warm.clone();
        scanner.evaluate((0..center).rev(), &mut lower_warm)?;

        let mut upper_warm = center_warm;
        scanner.evaluate(center + 1..targets_len, &mut upper_warm)?;
    }

    if scores
        .iter()
        .chain(&null_scores)
        .any(|score| !score.is_finite())
    {
        return Err(ExampleError::boxed(format!(
            "outer fold {} produced an incomplete or non-finite profile",
            outer_fold + 1
        )));
    }

    Ok(OuterFoldResult {
        scores,
        null_scores,
        optimization,
    })
}

fn run_analysis(data: &PreparedData, config: &AnalysisConfig) -> ExampleResult<AnalysisResult> {
    config.validate(data)?;
    let support_grids = SUPPORT_SENSITIVITY_COUNTS
        .into_iter()
        .map(|minimum| {
            common_supported_targets(data, config, minimum).map(|targets| (minimum, targets))
        })
        .collect::<ExampleResult<Vec<_>>>()?;
    let supported_targets = support_grids[0].1.clone();
    let effective_config = AnalysisConfig {
        targets: supported_targets,
        ..config.clone()
    };
    let config = &effective_config;
    let family = policy_family()?;
    let targets_len = config.targets.len();
    let mut profile = config
        .lags
        .iter()
        .copied()
        .flat_map(|lag| {
            config
                .targets
                .iter()
                .copied()
                .map(move |target| ProfilePoint {
                    lag,
                    target,
                    fold_nll: Vec::with_capacity(config.outer_folds),
                })
        })
        .collect::<Vec<_>>();
    let mut optimization = OptimizationStats::default();
    let mut null_fold_nll = config
        .lags
        .iter()
        .map(|_| Vec::with_capacity(config.outer_folds))
        .collect::<Vec<_>>();
    let analysis_started = Instant::now();

    let completed_folds = AtomicUsize::new(0);
    let fold_results = (0..config.outer_folds)
        .into_par_iter()
        .map(|outer_fold| {
            let result = evaluate_outer_fold(data, config, family, outer_fold);
            if let Ok(fold) = &result
                && config.show_progress
            {
                let (_, validation_range) =
                    outer_ranges(config, outer_fold).expect("validated outer fold");
                let completed = completed_folds.fetch_add(1, Ordering::Relaxed) + 1;
                eprintln!(
                    "outer fold {}/{} complete ({}/{}): validation {}..{}, solver_runs={}, retries={}, elapsed={:.1}s",
                    outer_fold + 1,
                    config.outer_folds,
                    completed,
                    config.outer_folds,
                    data.dates[validation_range.start],
                    data.dates[validation_range.end - 1],
                    fold.optimization.solver_runs,
                    fold.optimization.retries,
                    analysis_started.elapsed().as_secs_f64(),
                );
            }
            result
        })
        .collect::<ExampleResult<Vec<_>>>()?;

    // Collecting an indexed Rayon iterator preserves chronological fold order.
    for fold in fold_results {
        debug_assert_eq!(fold.scores.len(), profile.len());
        debug_assert_eq!(fold.null_scores.len(), config.lags.len());
        optimization.add(fold.optimization);
        for (point, score) in profile.iter_mut().zip(fold.scores) {
            point.fold_nll.push(score);
        }
        for (scores, score) in null_fold_nll.iter_mut().zip(fold.null_scores) {
            scores.push(score);
        }
    }

    let (lag_summaries, selected_index) =
        summarize_profile(&profile, &config.lags, targets_len, &null_fold_nll)?;
    let selected_lag = profile[selected_index].lag;
    let selected_target = profile[selected_index].target;
    let (selected_mean_nll, selected_standard_error) = profile[selected_index].mean_se();
    let selected_lag_summary = *lag_summaries
        .iter()
        .find(|summary| summary.lag == selected_lag)
        .expect("every selected lag has a summary");
    let selected_target_minus_null_mean = selected_lag_summary.target_minus_null_mean;
    let selected_target_minus_null_standard_error =
        selected_lag_summary.target_minus_null_standard_error;
    let paired_one_se_range = contiguous_paired_one_se_range(&profile, selected_index, targets_len);
    let lag_averaged_target =
        config.targets[lag_averaged_target_index(&profile, config.lags.len(), targets_len, None)];
    let selected_lag_start = selected_index / targets_len * targets_len;
    let selected_lag_range = finite_range(
        profile[selected_lag_start..selected_lag_start + targets_len]
            .iter()
            .map(|point| point.mean_se().0),
    )
    .expect("validated profile scores are finite");
    let lag_averaged_scores = (0..targets_len).map(|target_index| {
        (0..config.lags.len())
            .map(|lag_index| {
                profile[profile_index(targets_len, lag_index, target_index)].mean_excluding(None)
            })
            .sum::<f64>()
            / config.lags.len() as f64
    });
    let lag_averaged_range =
        finite_range(lag_averaged_scores).expect("lag-averaged profile scores are finite");
    let support_sensitivity =
        summarize_support_sensitivity(&profile, &config.lags, &config.targets, &support_grids)?;
    let leave_one_fold_out_selections = if config.outer_folds > 1 {
        (0..config.outer_folds)
            .map(|excluded_fold| {
                let index = select_profile_index(&profile, Some(excluded_fold));
                (profile[index].lag, profile[index].target)
            })
            .collect()
    } else {
        Vec::new()
    };
    let analysis_len = config
        .analysis_len()
        .expect("validated analysis length must fit");
    let final_fit = fit_final_model(
        data,
        family,
        analysis_len,
        selected_lag,
        selected_target,
        &mut optimization,
    )?;

    Ok(AnalysisResult {
        candidate_target_range: (config.targets[0], config.targets[config.targets.len() - 1]),
        candidate_target_count: config.targets.len(),
        lag_summaries,
        selected_lag,
        selected_target,
        selected_mean_nll,
        selected_standard_error,
        paired_one_se_range,
        lag_averaged_target,
        selected_lag_nll_span: selected_lag_range.1 - selected_lag_range.0,
        lag_averaged_nll_span: lag_averaged_range.1 - lag_averaged_range.0,
        selected_target_minus_null_mean,
        selected_target_minus_null_standard_error,
        support_sensitivity,
        leave_one_fold_out_selections,
        final_fit,
        optimization,
    })
}

fn summarize_profile(
    profile: &[ProfilePoint],
    lags: &[usize],
    target_count: usize,
    null_fold_nll: &[Vec<f64>],
) -> ExampleResult<(Vec<LagSummary>, usize)> {
    if profile.len() != lags.len() * target_count
        || target_count == 0
        || null_fold_nll.len() != lags.len()
    {
        return Err(ExampleError::boxed(
            "profile dimensions do not match lag, target, and null-score grids",
        ));
    }

    let mut lag_summaries = Vec::with_capacity(lags.len());
    let mut selected_index = 0;
    let mut selected_mean = f64::INFINITY;

    for (lag_index, lag) in lags.iter().copied().enumerate() {
        let start = lag_index * target_count;
        let end = start + target_count;
        let best = (start..end)
            .min_by(|left, right| {
                let left_mean = profile[*left].mean_se().0;
                let right_mean = profile[*right].mean_se().0;
                left_mean
                    .total_cmp(&right_mean)
                    .then_with(|| profile[*left].target.total_cmp(&profile[*right].target))
            })
            .expect("each lag has at least one target");
        let (mean_nll, standard_error) = profile[best].mean_se();
        if profile[best].fold_nll.len() != null_fold_nll[lag_index].len()
            || null_fold_nll[lag_index].is_empty()
        {
            return Err(ExampleError::boxed(format!(
                "lag {lag} target and null scores have incompatible fold counts"
            )));
        }
        let target_minus_null = profile[best]
            .fold_nll
            .iter()
            .zip(&null_fold_nll[lag_index])
            .map(|(target, null)| target - null)
            .collect::<Vec<_>>();
        let (target_minus_null_mean, target_minus_null_standard_error) =
            mean_and_standard_error(&target_minus_null);
        lag_summaries.push(LagSummary {
            lag,
            target: profile[best].target,
            mean_nll,
            standard_error,
            target_minus_null_mean,
            target_minus_null_standard_error,
        });

        let score_order = mean_nll.total_cmp(&selected_mean);
        let candidate_is_better = score_order.is_lt()
            || (score_order.is_eq()
                && (lag < profile[selected_index].lag
                    || (lag == profile[selected_index].lag
                        && profile[best].target < profile[selected_index].target)));
        if candidate_is_better {
            selected_index = best;
            selected_mean = mean_nll;
        }
    }

    Ok((lag_summaries, selected_index))
}

fn summarize_support_sensitivity(
    profile: &[ProfilePoint],
    lags: &[usize],
    targets: &[f64],
    support_grids: &[(usize, Vec<f64>)],
) -> ExampleResult<Vec<SupportSensitivity>> {
    if lags.is_empty()
        || targets.is_empty()
        || profile.len() != lags.len() * targets.len()
        || support_grids.is_empty()
    {
        return Err(ExampleError::boxed(
            "support sensitivity dimensions do not match the fitted profile",
        ));
    }

    support_grids
        .iter()
        .map(|(minimum_per_side, grid)| {
            if grid.is_empty() {
                return Err(ExampleError::boxed(format!(
                    "support grid for minimum {minimum_per_side} is empty"
                )));
            }
            let target_indices = grid
                .iter()
                .map(|target| {
                    targets
                        .binary_search_by(|candidate| candidate.total_cmp(target))
                        .map_err(|_| {
                            ExampleError::boxed(format!(
                                "support target {target} is outside the fitted profile"
                            ))
                        })
                })
                .collect::<ExampleResult<Vec<_>>>()?;
            let selected_index = (0..lags.len())
                .flat_map(|lag_index| {
                    target_indices.iter().copied().map(move |target_index| {
                        profile_index(targets.len(), lag_index, target_index)
                    })
                })
                .min_by(|left, right| {
                    profile[*left]
                        .mean_excluding(None)
                        .total_cmp(&profile[*right].mean_excluding(None))
                        .then_with(|| profile[*left].lag.cmp(&profile[*right].lag))
                        .then_with(|| profile[*left].target.total_cmp(&profile[*right].target))
                })
                .expect("validated support grid is non-empty");
            let lag_averaged_index = *target_indices
                .iter()
                .min_by(|left, right| {
                    let mean = |target_index: usize| {
                        (0..lags.len())
                            .map(|lag_index| {
                                profile[profile_index(targets.len(), lag_index, target_index)]
                                    .mean_excluding(None)
                            })
                            .sum::<f64>()
                            / lags.len() as f64
                    };
                    mean(**left)
                        .total_cmp(&mean(**right))
                        .then_with(|| left.cmp(right))
                })
                .expect("validated support grid is non-empty");

            Ok(SupportSensitivity {
                minimum_per_side: *minimum_per_side,
                target_range: (grid[0], grid[grid.len() - 1]),
                target_count: grid.len(),
                selected_lag: profile[selected_index].lag,
                selected_target: profile[selected_index].target,
                lag_averaged_target: targets[lag_averaged_index],
            })
        })
        .collect()
}

fn contiguous_paired_one_se_range(
    profile: &[ProfilePoint],
    selected_index: usize,
    target_count: usize,
) -> (f64, f64) {
    let lag_start = selected_index / target_count * target_count;
    let lag_end = lag_start + target_count;
    let selected = &profile[selected_index];
    let is_competitive = |candidate: &ProfilePoint| {
        let differences = candidate
            .fold_nll
            .iter()
            .zip(&selected.fold_nll)
            .map(|(candidate, selected)| candidate - selected)
            .collect::<Vec<_>>();
        let (mean_difference, standard_error) = mean_and_standard_error(&differences);
        mean_difference <= standard_error
    };
    let mut left = selected_index;
    while left > lag_start && is_competitive(&profile[left - 1]) {
        left -= 1;
    }
    let mut right = selected_index;
    while right + 1 < lag_end && is_competitive(&profile[right + 1]) {
        right += 1;
    }
    (profile[left].target, profile[right].target)
}

fn fit_final_model(
    data: &PreparedData,
    family: PolicyFamily,
    analysis_len: usize,
    lag: usize,
    target: f64,
    stats: &mut OptimizationStats,
) -> ExampleResult<FinalFitReport> {
    let full_rows = data.rows(lag, 0..analysis_len)?;
    let designs = PolicyDesigns::new(full_rows, target);
    let blocks = designs.blocks()?;
    let model = Gamlss::try_new(family, blocks, full_rows.response)?
        .with_objective_scale(ObjectiveScale::Mean);
    let cold_start = model.initial_parameters()?;
    let fit = optimize_with_retry(
        || model.clone().into_workspace_objective(),
        None,
        &cold_start,
        stats,
    )?;
    let diagnostics = model.training_diagnostics(&fit.parameters)?;
    let fitted = model.predict_theta(&fit.parameters)?;
    let pit = fitted
        .iter()
        .zip(full_rows.response)
        .filter(|(_, observation)| **observation != 0.0)
        .map(|(theta, observation)| {
            model
                .family()
                .component()
                .cdf(*observation, &theta.component())
        })
        .collect::<Vec<_>>();
    let residuals = pit
        .iter()
        .copied()
        .map(gamlss::special::unit_normal_quantile)
        .collect::<Vec<_>>();
    let mu_range = finite_range(fitted.iter().map(|theta| theta.mu))
        .ok_or_else(|| ExampleError::boxed("fitted locations are non-finite"))?;
    let sigma_range = finite_range(fitted.iter().map(|theta| theta.sigma))
        .ok_or_else(|| ExampleError::boxed("fitted scales are non-finite"))?;
    let zero_probability_range = finite_range(fitted.iter().map(|theta| theta.zero_probability))
        .ok_or_else(|| ExampleError::boxed("fitted zero probabilities are non-finite"))?;
    let pit_mean_sd = finite_mean_sd(&pit)
        .ok_or_else(|| ExampleError::boxed("PIT diagnostics are non-finite"))?;
    let residual_mean_sd = finite_mean_sd(&residuals)
        .ok_or_else(|| ExampleError::boxed("quantile residuals are non-finite"))?;
    let coverage_90 = interval_coverage_90(*model.family(), &fitted, full_rows.response);
    let structural_response = structural_target_response(&model, &fit.parameters, target)?;

    Ok(FinalFitReport {
        parameters: fit.parameters.len(),
        iterations: fit.iterations,
        termination: fit.termination,
        objective: diagnostics.objective,
        train_nll: diagnostics.train_nll,
        penalty: diagnostics.penalty,
        gradient_norm: diagnostics.gradient_norm,
        nonfinite_gradients: diagnostics.nonfinite_gradient_count,
        mu_range,
        sigma_range,
        zero_probability_range,
        structural_response_one_point_below: structural_response.0,
        structural_response_one_point_above: structural_response.1,
        pit_mean_sd,
        residual_mean_sd,
        coverage_90,
    })
}

fn structural_target_response(
    model: &Gamlss<PolicyFamily, PolicyBlocks<'_>, &[f64]>,
    parameters: &[f64],
    target: f64,
) -> ExampleResult<(f64, f64)> {
    let response = [0.0; 3];
    let zeros = [0.0; 3];
    let inflation = [target - 1.0, target, target + 1.0];
    let rows = FeatureRows {
        response: &response,
        previous_delta_rate: &zeros,
        inflation: &inflation,
        inflation_change: &zeros,
    };
    let designs = PolicyDesigns::new(rows, target);
    let blocks = designs.blocks()?;
    let fitted = model.predict_theta_with_blocks(parameters, &blocks)?;
    if fitted.len() != 3 || fitted[1].mu.abs() > 1.0e-12 {
        return Err(ExampleError::boxed(
            "structural target response lost its zero anchor",
        ));
    }
    Ok((fitted[0].mu, fitted[2].mu))
}

fn interval_coverage_90(family: PolicyFamily, fitted: &[PolicyTheta], observations: &[f64]) -> f64 {
    let covered = fitted
        .iter()
        .zip(observations)
        .filter(|(theta, observation)| {
            let lower = family.quantile(0.05, theta);
            let upper = family.quantile(0.95, theta);
            (lower..=upper).contains(observation)
        })
        .count();
    covered as f64 / observations.len() as f64
}

fn mean_and_standard_error(values: &[f64]) -> (f64, f64) {
    debug_assert!(!values.is_empty());
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    if values.len() == 1 {
        return (mean, 0.0);
    }
    let sum_squares = values
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>();
    let sample_variance = sum_squares / (count - 1.0);
    (mean, (sample_variance / count).sqrt())
}

fn finite_mean_sd(values: &[f64]) -> Option<(f64, f64)> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
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
    Some((mean, variance.sqrt()))
}

fn finite_range(mut values: impl Iterator<Item = f64>) -> Option<(f64, f64)> {
    let first = values.next()?;
    if !first.is_finite() {
        return None;
    }
    values.try_fold((first, first), |(minimum, maximum), value| {
        value
            .is_finite()
            .then_some((minimum.min(value), maximum.max(value)))
    })
}

fn print_report(
    data: &PreparedData,
    config: &AnalysisConfig,
    analysis: &AnalysisResult,
    elapsed: Duration,
) {
    let analysis_len = config
        .analysis_len()
        .expect("validated analysis length must fit");
    let zero_changes = data.delta_rate[..analysis_len]
        .iter()
        .filter(|change| **change == 0.0)
        .count();
    println!("CBR implicit inflation-target profile");
    println!(
        "source history={}..{}, comparable responses={} ({}..{}), zero changes={} ({:.1}%)",
        data.source_start,
        data.source_end,
        analysis_len,
        data.dates[0],
        data.dates[analysis_len - 1],
        zero_changes,
        100.0 * zero_changes as f64 / analysis_len as f64,
    );
    println!(
        "model=zero-adjusted Student-t(df={STUDENT_T_DF:.0}), sign-constrained piecewise-linear target response, inflation momentum, rate inertia, constant scale/zero mass, lags={:?}",
        config.lags,
    );
    println!(
        "searched targets={TARGET_MIN:.1}..={TARGET_MAX:.1} by 0.1; common two-sided-support grid={:.1}..={:.1} ({} points, at least {MIN_NONZERO_DECISIONS_PER_SIDE} initial non-zero decisions per side and lag)",
        analysis.candidate_target_range.0,
        analysis.candidate_target_range.1,
        analysis.candidate_target_count,
    );
    println!("support-guard sensitivity (all rows reuse the same fitted profile):");
    println!("  minimum_per_side  supported_grid  points  best_spec  lag_averaged_target");
    for sensitivity in &analysis.support_sensitivity {
        println!(
            "  {:>16}  {:>5.1}..={:<5.1}  {:>6}  {:>4}m/{:.1}%  {:>18.1}%",
            sensitivity.minimum_per_side,
            sensitivity.target_range.0,
            sensitivity.target_range.1,
            sensitivity.target_count,
            sensitivity.selected_lag,
            sensitivity.selected_target,
            sensitivity.lag_averaged_target,
        );
    }
    println!(
        "rolling_folds={}x{}m, rayon_threads={}",
        config.outer_folds,
        config.outer_fold_months,
        rayon::current_num_threads(),
    );
    println!();
    println!(
        "lag_months  best_target  mean_validation_nll  standard_error  target-minus-null  paired_SE"
    );
    for summary in &analysis.lag_summaries {
        println!(
            "{:>10}  {:>11.1}  {:>19.6}  {:>14.6}  {:>17.6}  {:>9.6}",
            summary.lag,
            summary.target,
            summary.mean_nll,
            summary.standard_error,
            summary.target_minus_null_mean,
            summary.target_minus_null_standard_error,
        );
    }
    println!();
    println!(
        "best single specification: lag={} months, target={:.1}%, validation mean NLL={:.6}, SE={:.6}",
        analysis.selected_lag,
        analysis.selected_target,
        analysis.selected_mean_nll,
        analysis.selected_standard_error,
    );
    println!(
        "paired one-SE target range at that lag=[{:.1}%, {:.1}%] (heuristic; not a confidence interval)",
        analysis.paired_one_se_range.0, analysis.paired_one_se_range.1,
    );
    println!(
        "equal-weight lag-averaged sensitivity target={:.1}%",
        analysis.lag_averaged_target,
    );
    println!(
        "NLL profile span across supported targets: selected lag={:.6}, lag-averaged={:.6}",
        analysis.selected_lag_nll_span, analysis.lag_averaged_nll_span,
    );
    let target_signal_supported = analysis.selected_target_minus_null_mean
        + analysis.selected_target_minus_null_standard_error
        < 0.0;
    println!(
        "optimistic target-signal screen: selected-target minus target-free NLL={:+.6} (paired SE={:.6}, negative is better), one-SE support={target_signal_supported}",
        analysis.selected_target_minus_null_mean,
        analysis.selected_target_minus_null_standard_error,
    );
    if !analysis.leave_one_fold_out_selections.is_empty() {
        let selections = analysis
            .leave_one_fold_out_selections
            .iter()
            .map(|(lag, target)| format!("{lag}m/{target:.1}%"))
            .collect::<Vec<_>>()
            .join(", ");
        println!("delete-one-validation-block joint selections: {selections}");

        let (minimum, maximum) = finite_range(
            analysis
                .leave_one_fold_out_selections
                .iter()
                .map(|(_, target)| *target),
        )
        .expect("leave-one-fold-out selections are finite");
        let lower = analysis.candidate_target_range.0;
        let upper = analysis.candidate_target_range.1;
        let boundary_count = analysis
            .leave_one_fold_out_selections
            .iter()
            .filter(|(_, target)| *target <= lower || *target >= upper)
            .count();
        let fitted_target_response = analysis
            .final_fit
            .structural_response_one_point_below
            .abs()
            .max(analysis.final_fit.structural_response_one_point_above.abs());
        let (verdict, interpretation) = if target_signal_supported {
            (
                "provisional only",
                "the optimistic screen would still need nested confirmation",
            )
        } else {
            (
                "not identified",
                "the two observed series do not robustly identify a target",
            )
        };
        println!(
            "robustness verdict={verdict}: delete-one-block target span=[{minimum:.1}%, {maximum:.1}%], boundary selections={boundary_count}/{}, fitted max |1 pp target-gap response|={fitted_target_response:.3e}; {interpretation}",
            analysis.leave_one_fold_out_selections.len(),
        );
    }
    let official_inside = (analysis.paired_one_se_range.0..=analysis.paired_one_se_range.1)
        .contains(&OFFICIAL_TARGET);
    println!(
        "official benchmark revealed after all modeling choices: {OFFICIAL_TARGET:.1}%, numerical-minimum-minus-benchmark={:+.1} pp, inside fixed-lag paired range={official_inside}",
        analysis.selected_target - OFFICIAL_TARGET,
    );
    println!();
    println!(
        "full fit: parameters={}, iterations={}, termination={}, objective={:.6}, mean_nll={:.6}, penalty={:.6}, grad_norm={:.3e}, nonfinite_gradients={}",
        analysis.final_fit.parameters,
        analysis.final_fit.iterations,
        analysis.final_fit.termination,
        analysis.final_fit.objective,
        analysis.final_fit.train_nll,
        analysis.final_fit.penalty,
        analysis.final_fit.gradient_norm,
        analysis.final_fit.nonfinite_gradients,
    );
    println!(
        "fitted mu=[{:.4}, {:.4}], sigma=[{:.4}, {:.4}], zero_probability=[{:.4}, {:.4}]",
        analysis.final_fit.mu_range.0,
        analysis.final_fit.mu_range.1,
        analysis.final_fit.sigma_range.0,
        analysis.final_fit.sigma_range.1,
        analysis.final_fit.zero_probability_range.0,
        analysis.final_fit.zero_probability_range.1,
    );
    println!(
        "structural mean response at target -/+ 1 pp (zero momentum/inertia)={:.3e}/{:.3e}",
        analysis.final_fit.structural_response_one_point_below,
        analysis.final_fit.structural_response_one_point_above,
    );
    println!(
        "non-zero component PIT mean/sd={:.4}/{:.4}, qres mean/sd={:.4}/{:.4}, full-mixture in-sample 90% coverage={:.3}",
        analysis.final_fit.pit_mean_sd.0,
        analysis.final_fit.pit_mean_sd.1,
        analysis.final_fit.residual_mean_sd.0,
        analysis.final_fit.residual_mean_sd.1,
        analysis.final_fit.coverage_90,
    );
    println!(
        "optimization: solver_runs={}, retries={}, total_iterations={}, elapsed={:.1}s",
        analysis.optimization.solver_runs,
        analysis.optimization.retries,
        analysis.optimization.iterations,
        elapsed.as_secs_f64(),
    );
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use approx::assert_abs_diff_eq;
    use gamlss_datasets::Month;

    use super::*;

    #[test]
    fn prepared_data_has_common_lag_alignment() {
        let data = PreparedData::load().unwrap();
        let dataset = gamlss_datasets::cbr_inflation_and_interest_rate();
        let mut raw = dataset
            .x
            .iter()
            .copied()
            .zip(dataset.y.iter().copied())
            .collect::<Vec<_>>();
        raw.sort_by_key(|(date, _)| *date);

        let analysis_start = raw
            .iter()
            .position(|(date, _)| date.year() >= ANALYSIS_START_YEAR)
            .unwrap();

        assert_eq!(data.len(), 114);
        assert_eq!(data.source_start, raw[0].0);
        assert_eq!(data.source_end, raw[raw.len() - 1].0);
        assert_eq!(
            data.dates[0],
            Date::from_calendar_date(2017, Month::January, 1).unwrap()
        );
        assert_eq!(
            data.dates[113],
            Date::from_calendar_date(2026, Month::June, 1).unwrap()
        );
        assert_eq!(
            data.delta_rate
                .iter()
                .filter(|change| **change == 0.0)
                .count(),
            65
        );
        for lag in REACTION_LAGS {
            let rows = data.rows(lag, 0..data.len()).unwrap();
            assert_eq!(rows.len(), data.len());
        }
        assert_abs_diff_eq!(
            data.delta_rate[0],
            raw[analysis_start].1[0] - raw[analysis_start - 1].1[0],
        );
        assert_abs_diff_eq!(
            data.previous_delta_rate[0],
            raw[analysis_start - 1].1[0] - raw[analysis_start - 2].1[0],
        );
        for (lag_index, lag) in REACTION_LAGS.into_iter().enumerate() {
            assert_abs_diff_eq!(
                data.lagged_inflation[lag_index][0],
                raw[analysis_start - lag].1[1],
            );
            assert_abs_diff_eq!(
                data.lagged_inflation_change[lag_index][0],
                raw[analysis_start - lag].1[1] - raw[analysis_start - lag - 1].1[1],
            );
        }
    }

    #[test]
    fn full_rolling_folds_are_ordered_and_disjoint() {
        let data = PreparedData::load().unwrap();
        let config = AnalysisConfig::full();
        config.validate(&data).unwrap();

        let (first_train, first_validation) = outer_ranges(&config, 0).unwrap();
        assert_eq!(first_train, 0..54);
        assert_eq!(first_validation, 54..66);
        let (last_train, last_validation) = outer_ranges(&config, 4).unwrap();
        assert_eq!(last_train, 0..102);
        assert_eq!(last_validation, 102..114);

        for outer_fold in 0..config.outer_folds {
            let (outer_train, outer_validation) = outer_ranges(&config, outer_fold).unwrap();
            assert_eq!(outer_train.start, 0);
            assert!(outer_train.end <= outer_validation.start);
            assert_eq!(outer_validation.len(), OUTER_FOLD_MONTHS);
        }
    }

    #[test]
    fn analysis_grids_cover_requested_targets_and_lags() {
        let targets = target_grid();
        assert_eq!(targets.len(), 101);
        assert_eq!(targets[0], 0.0);
        assert_eq!(targets[40], 4.0);
        assert_eq!(targets[100], 10.0);
        assert_eq!(REACTION_LAGS, [1, 2, 3, 6, 9, 12]);
        assert!(!REACTION_LAGS.contains(&0));
    }

    #[test]
    fn common_support_grid_excludes_one_sided_candidates() {
        let data = PreparedData::load().unwrap();
        let config = AnalysisConfig::full();
        let supported =
            common_supported_targets(&data, &config, MIN_NONZERO_DECISIONS_PER_SIDE).unwrap();

        assert_eq!(supported.first(), Some(&2.5));
        assert_eq!(supported.last(), Some(&5.6));
        assert!(supported.windows(2).all(|pair| pair[0] < pair[1]));

        let conservative = common_supported_targets(&data, &config, 5).unwrap();
        assert_eq!(conservative.first(), Some(&3.5));
        assert_eq!(conservative.last(), Some(&4.6));
    }

    #[test]
    fn target_reaction_is_anchored_and_sign_constrained() {
        let response = [0.0; 3];
        let zeros = [0.0; 3];
        let inflation = [4.0, 6.0, 2.0];
        let rows = FeatureRows {
            response: &response,
            previous_delta_rate: &zeros,
            inflation: &inflation,
            inflation_change: &zeros,
        };
        let designs = PolicyDesigns::new(rows, 4.0);
        let model = Gamlss::try_new(
            policy_family().unwrap(),
            designs.blocks().unwrap(),
            &response,
        )
        .unwrap();
        let fitted = model.predict_theta(&[0.0; 6]).unwrap();

        assert_abs_diff_eq!(fitted[0].mu, 0.0, epsilon = 1.0e-15);
        assert!(fitted[1].mu > 0.0);
        assert!(fitted[2].mu < 0.0);
    }

    #[test]
    fn paired_one_se_range_keeps_only_the_contiguous_component() {
        let scores = [
            (2.0, vec![1.10, 1.10]),
            (2.1, vec![0.80, 1.10]),
            (2.2, vec![0.90, 0.90]),
            (2.3, vec![0.90, 1.00]),
            (2.4, vec![1.20, 1.20]),
            (2.5, vec![0.90, 0.90]),
        ];
        let profile = scores
            .into_iter()
            .map(|(target, fold_nll)| ProfilePoint {
                lag: 1,
                target,
                fold_nll,
            })
            .collect::<Vec<_>>();

        assert_eq!(
            contiguous_paired_one_se_range(&profile, 2, profile.len()),
            (2.1, 2.3)
        );
    }

    #[test]
    fn reduced_rolling_pipeline_has_finite_scores_and_gradient() {
        let data = PreparedData::load().unwrap();
        let config = AnalysisConfig {
            lags: vec![9],
            targets: vec![5.0],
            initial_train: OUTER_INITIAL_TRAIN,
            outer_folds: 1,
            outer_fold_months: OUTER_FOLD_MONTHS,
            show_progress: false,
        };

        let result = run_analysis(&data, &config).unwrap();

        assert!(result.selected_mean_nll.is_finite());
        assert!(result.final_fit.objective.is_finite());
        assert!(result.final_fit.gradient_norm.is_finite());
        assert_eq!(result.final_fit.nonfinite_gradients, 0);
    }

    #[test]
    fn target_scanner_fills_both_warm_start_branches() {
        let data = PreparedData::load().unwrap();
        let config = AnalysisConfig {
            lags: vec![9],
            targets: vec![4.9, 5.0, 5.1],
            initial_train: OUTER_INITIAL_TRAIN,
            outer_folds: 1,
            outer_fold_months: OUTER_FOLD_MONTHS,
            show_progress: false,
        };

        let fold = evaluate_outer_fold(&data, &config, policy_family().unwrap(), 0).unwrap();

        assert_eq!(fold.scores.len(), 3);
        assert!(fold.scores.iter().all(|score| score.is_finite()));
        assert_eq!(fold.null_scores.len(), 1);
        assert!(fold.null_scores.iter().all(|score| score.is_finite()));
    }

    #[test]
    fn profile_summary_selects_each_lag_minimum_and_global_minimum() {
        let profile = [
            ProfilePoint {
                lag: 1,
                target: 2.0,
                fold_nll: vec![1.1, 1.1],
            },
            ProfilePoint {
                lag: 1,
                target: 2.1,
                fold_nll: vec![1.0, 1.0],
            },
            ProfilePoint {
                lag: 3,
                target: 2.0,
                fold_nll: vec![0.8, 0.8],
            },
            ProfilePoint {
                lag: 3,
                target: 2.1,
                fold_nll: vec![0.9, 0.9],
            },
        ];

        let null_fold_nll = [vec![1.2, 1.2], vec![1.0, 1.0]];
        let (summaries, selected) =
            summarize_profile(&profile, &[1, 3], 2, &null_fold_nll).unwrap();

        assert_eq!(summaries[0].target, 2.1);
        assert_eq!(summaries[1].target, 2.0);
        assert_abs_diff_eq!(summaries[0].target_minus_null_mean, -0.2);
        assert_abs_diff_eq!(summaries[1].target_minus_null_mean, -0.2);
        assert_eq!(selected, 2);
    }
}
