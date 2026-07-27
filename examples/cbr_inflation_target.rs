//! Profile an implicit inflation target from monthly Bank of Russia data.
//!
//! The response is the monthly key-rate change from 2017 onward, when the 4%
//! target was already in force. Exact zero changes get their own point mass;
//! non-zero changes follow a Student-t distribution with a constant scale. Its
//! location is anchored to zero at each candidate target, so the candidate has
//! the intended interpretation instead of being absorbed by a free intercept.
//! Strict nested rolling-origin validation selects the mean-spline penalty,
//! while outer rolling-origin validation selects the lag and target. Rayon runs
//! independent outer folds in parallel while each target path remains ordered
//! for warm starts. The official 4% target is printed only after model selection
//! and is never used as a fitting input. The historical policy reference is the
//! [Bank of Russia inflation page](https://www.cbr.ru/hd_base/infl/).
//!
//! This is intentionally a substantial reproducible analysis rather than a
//! quick-start example. The two observed series cannot identify a causal
//! monetary-policy rule, and the reported one-standard-error range is a
//! predictive-stability heuristic rather than a confidence interval.
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
        DenseDesign, Family, Gamlss, HasCdf, HasQuantile, LinearPredictorBlock, Mu, NoPenalty,
        Objective, ObjectiveScale, ParameterBlock, ParameterBlocks, SegmentPenalty, Sigma,
        SumBlock, ZeroProbability,
    },
    family::{ZeroAdjustedStudentTMuSigma, ZeroAdjustedStudentTTheta},
    spline::{
        HelmertContrastDesign, HelmertContrastPenalty, OpenUniformSplineBasis,
        PreparedDifferencePenalty, SplineOrder,
    },
};
use gamlss_datasets::Date;
use rayon::prelude::*;

const REACTION_LAGS: [usize; 6] = [1, 2, 3, 6, 9, 12];
const MAX_REACTION_LAG: usize = 12;
const TARGET_MIN: f64 = 2.0;
const TARGET_MAX: f64 = 8.0;
const OFFICIAL_TARGET: f64 = 4.0;
const STUDENT_T_DF: f64 = 5.0;
const ANALYSIS_START_YEAR: i32 = 2017;

const MEAN_BASIS: usize = 6;
const DIFFERENCE_ORDER: usize = 2;

const OUTER_INITIAL_TRAIN: usize = 48;
const OUTER_FOLDS: usize = 11;
const INNER_FOLDS: usize = 3;
const FOLD_MONTHS: usize = 6;
const SMOOTHING_LEVELS_DESC: [f64; 3] = [1.0, 0.1, 0.01];

const LBFGS_MEMORY: usize = 10;
const PRIMARY_MAX_ITERATIONS: u64 = 200;
const RETRY_MAX_ITERATIONS: u64 = 10_000;
const GRADIENT_TOLERANCE: f64 = 1.0e-5;
const COST_TOLERANCE: f64 = 1.0e-8;
const RETRY_GRADIENT_RATIO_LIMIT: f64 = 10.0;

type ExampleResult<T> = Result<T, Box<dyn StdError + Send + Sync>>;
type PolicyFamily = ZeroAdjustedStudentTMuSigma;
type PolicyTheta = ZeroAdjustedStudentTTheta;
type LinearTerm<'a> = LinearPredictorBlock<&'a DenseDesign>;
type MeanPredictor<'a> = SumBlock<(LinearTerm<'a>, LinearTerm<'a>, LinearTerm<'a>)>;
type SmoothPenalty = HelmertContrastPenalty<PreparedDifferencePenalty>;
type MeanPenalty = (SegmentPenalty<SmoothPenalty>, SegmentPenalty<SmoothPenalty>);
type PolicyBlocks<'a> = ParameterBlocks<(
    ParameterBlock<Mu, MeanPredictor<'a>, MeanPenalty>,
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

#[derive(Clone, Copy, Debug, PartialEq)]
struct Smoothing {
    mean: f64,
}

impl Smoothing {
    const fn new(mean: f64) -> Self {
        Self { mean }
    }
}

fn smoothing_grid() -> Vec<Smoothing> {
    // Stronger smoothing comes first so the first cold-started fit is the most
    // regularized. Selection itself uses an explicit score/tie comparison.
    SMOOTHING_LEVELS_DESC
        .into_iter()
        .map(Smoothing::new)
        .collect()
}

fn target_grid() -> Vec<f64> {
    (20..=80).map(|tenths| f64::from(tenths) / 10.0).collect()
}

#[derive(Clone, Debug)]
struct AnalysisConfig {
    lags: Vec<usize>,
    targets: Vec<f64>,
    smoothings: Vec<Smoothing>,
    initial_train: usize,
    outer_folds: usize,
    inner_folds: usize,
    fold_months: usize,
    show_progress: bool,
}

impl AnalysisConfig {
    fn full() -> Self {
        Self {
            lags: REACTION_LAGS.to_vec(),
            targets: target_grid(),
            smoothings: smoothing_grid(),
            initial_train: OUTER_INITIAL_TRAIN,
            outer_folds: OUTER_FOLDS,
            inner_folds: INNER_FOLDS,
            fold_months: FOLD_MONTHS,
            show_progress: true,
        }
    }

    fn analysis_len(&self) -> Option<usize> {
        self.outer_folds
            .checked_mul(self.fold_months)
            .and_then(|validation| self.initial_train.checked_add(validation))
    }

    fn validate(&self, data: &PreparedData) -> ExampleResult<()> {
        if self.lags.is_empty()
            || self.targets.is_empty()
            || self.smoothings.is_empty()
            || self.outer_folds == 0
            || self.inner_folds == 0
            || self.fold_months == 0
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
        let inner_validation = self
            .inner_folds
            .checked_mul(self.fold_months)
            .ok_or_else(|| ExampleError::boxed("inner validation row count overflowed"))?;
        if self.initial_train <= inner_validation {
            return Err(ExampleError::boxed(
                "the initial outer training window must precede all inner validation windows",
            ));
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
        if self
            .smoothings
            .iter()
            .any(|smoothing| !smoothing.mean.is_finite() || smoothing.mean <= 0.0)
        {
            return Err(ExampleError::boxed(
                "smoothing values must be finite and positive",
            ));
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
            for (column, lag) in lagged_inflation.iter_mut().zip(REACTION_LAGS) {
                column.push(raw[current - lag].1[1]);
            }
        }

        Ok(Self {
            source_start,
            source_end,
            dates,
            delta_rate,
            previous_delta_rate,
            lagged_inflation,
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
            inflation: &self.lagged_inflation[lag_index][range],
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct FeatureRows<'a> {
    response: &'a [f64],
    previous_delta_rate: &'a [f64],
    inflation: &'a [f64],
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
        .checked_mul(config.fold_months)
        .and_then(|offset| config.initial_train.checked_add(offset))
        .ok_or_else(|| ExampleError::boxed("outer fold boundary overflowed"))?;
    let validation_end = validation_start
        .checked_add(config.fold_months)
        .ok_or_else(|| ExampleError::boxed("outer validation boundary overflowed"))?;
    Ok((0..validation_start, validation_start..validation_end))
}

fn inner_ranges(
    train_end: usize,
    inner_folds: usize,
    fold_months: usize,
) -> ExampleResult<Vec<(Range<usize>, Range<usize>)>> {
    let validation_rows = inner_folds
        .checked_mul(fold_months)
        .ok_or_else(|| ExampleError::boxed("inner validation row count overflowed"))?;
    let first_validation = train_end.checked_sub(validation_rows).ok_or_else(|| {
        ExampleError::boxed(format!(
            "training endpoint {train_end} is too short for {inner_folds} inner folds"
        ))
    })?;
    if first_validation == 0 {
        return Err(ExampleError::boxed(
            "inner rolling-origin validation needs a non-empty initial training window",
        ));
    }

    (0..inner_folds)
        .map(|inner_fold| {
            let validation_start = inner_fold
                .checked_mul(fold_months)
                .and_then(|offset| first_validation.checked_add(offset))
                .ok_or_else(|| ExampleError::boxed("inner fold boundary overflowed"))?;
            let validation_end = validation_start
                .checked_add(fold_months)
                .ok_or_else(|| ExampleError::boxed("inner validation boundary overflowed"))?;
            Ok((0..validation_start, validation_start..validation_end))
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct SplineBases {
    above: OpenUniformSplineBasis,
    below: OpenUniformSplineBasis,
}

impl SplineBases {
    fn from_training(rows: FeatureRows<'_>) -> ExampleResult<Self> {
        if rows.is_empty() {
            return Err(ExampleError::boxed(
                "cannot construct spline bases from an empty training window",
            ));
        }
        let (minimum, maximum) = finite_range(rows.inflation.iter().copied())
            .ok_or_else(|| ExampleError::boxed("training inflation has no finite values"))?;
        let above_max = maximum - TARGET_MIN;
        let below_max = TARGET_MAX - minimum;
        if above_max <= 0.0 || below_max <= 0.0 {
            return Err(ExampleError::boxed(format!(
                "training inflation range [{minimum}, {maximum}] does not cover the target profile domain"
            )));
        }

        Ok(Self {
            above: OpenUniformSplineBasis::new(0.0, above_max, MEAN_BASIS, SplineOrder::Cubic)?,
            below: OpenUniformSplineBasis::new(0.0, below_max, MEAN_BASIS, SplineOrder::Cubic)?,
        })
    }
}

#[derive(Clone, Debug)]
struct PolicyDesigns {
    intercept: DenseDesign,
    above: DenseDesign,
    below: DenseDesign,
    previous_delta_rate: DenseDesign,
}

impl PolicyDesigns {
    fn try_new(rows: FeatureRows<'_>, bases: SplineBases, target: f64) -> ExampleResult<Self> {
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
        Ok(Self {
            intercept: DenseDesign::intercept(rows.len()),
            above: anchored_spline_design(bases.above, &above)?,
            below: anchored_spline_design(bases.below, &below)?,
            previous_delta_rate: DenseDesign::column(rows.previous_delta_rate),
        })
    }

    fn blocks(&self, smoothing: Smoothing) -> ExampleResult<PolicyBlocks<'_>> {
        let mean_contrasts = MEAN_BASIS - 1;
        let above_start = 0;
        let below_start = above_start + mean_contrasts;

        let mean_predictor = SumBlock::new((
            LinearPredictorBlock::new(&self.above),
            LinearPredictorBlock::new(&self.below),
            LinearPredictorBlock::new(&self.previous_delta_rate),
        ));
        let mean_penalty = (
            SegmentPenalty::new(
                above_start..below_start,
                smooth_penalty(MEAN_BASIS, smoothing.mean)?,
            ),
            SegmentPenalty::new(
                below_start..below_start + mean_contrasts,
                smooth_penalty(MEAN_BASIS, smoothing.mean)?,
            ),
        );
        let mean = ParameterBlock::<Mu, _, _>::new(mean_predictor, mean_penalty, 0);

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

fn anchored_spline_design(
    basis: OpenUniformSplineBasis,
    coordinates: &[f64],
) -> ExampleResult<DenseDesign> {
    let transformed = HelmertContrastDesign::try_new(basis.design(coordinates)?)?;
    let anchor = HelmertContrastDesign::try_new(basis.design(&[0.0])?)?;
    let ncols = transformed.contrast().target_dim();
    let anchor_values = anchor.values();
    let mut values = transformed.values().to_vec();

    for row in values.chunks_exact_mut(ncols) {
        for (value, anchor_value) in row.iter_mut().zip(anchor_values) {
            *value -= anchor_value;
        }
    }

    Ok(DenseDesign::from_row_major_strict(
        coordinates.len(),
        ncols,
        values,
    )?)
}

fn smooth_penalty(source_dim: usize, lambda: f64) -> ExampleResult<SmoothPenalty> {
    Ok(HelmertContrastPenalty::try_new(
        source_dim,
        PreparedDifferencePenalty::try_new(lambda, DIFFERENCE_ORDER)?,
    )?)
}

fn policy_family() -> ExampleResult<PolicyFamily> {
    Ok(PolicyFamily::try_new(STUDENT_T_DF)?)
}

#[derive(Clone, Copy, Debug)]
struct SplitContext<'a> {
    train: FeatureRows<'a>,
    validation: FeatureRows<'a>,
    bases: SplineBases,
}

impl<'a> SplitContext<'a> {
    fn new(train: FeatureRows<'a>, validation: FeatureRows<'a>) -> ExampleResult<Self> {
        Ok(Self {
            train,
            validation,
            bases: SplineBases::from_training(train)?,
        })
    }
}

#[derive(Clone, Debug)]
struct NestedContext<'a> {
    inner: Vec<SplitContext<'a>>,
    outer: SplitContext<'a>,
}

impl<'a> NestedContext<'a> {
    fn for_outer_fold(
        data: &'a PreparedData,
        config: &AnalysisConfig,
        lag: usize,
        outer_fold: usize,
    ) -> ExampleResult<Self> {
        let (outer_train, outer_validation) = outer_ranges(config, outer_fold)?;
        let inner = inner_ranges(outer_train.end, config.inner_folds, config.fold_months)?
            .into_iter()
            .map(|(train, validation)| {
                SplitContext::new(data.rows(lag, train)?, data.rows(lag, validation)?)
            })
            .collect::<ExampleResult<Vec<_>>>()?;
        let outer = SplitContext::new(
            data.rows(lag, outer_train)?,
            data.rows(lag, outer_validation)?,
        )?;
        Ok(Self { inner, outer })
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

#[derive(Clone, Copy, Debug, Default)]
struct FitStarts<'a> {
    primary: Option<&'a [f64]>,
    retry: Option<&'a [f64]>,
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
    smoothing: Smoothing,
    starts: FitStarts<'_>,
    optimization: &mut OptimizationStats,
) -> ExampleResult<ScoredFit> {
    if split.validation.is_empty() {
        return Err(ExampleError::boxed(
            "validation window must contain observations",
        ));
    }
    let blocks = split.train_designs.blocks(smoothing)?;
    let model = Gamlss::try_new(family, blocks, split.train.response)?
        .with_objective_scale(ObjectiveScale::Mean);
    let cold_start = model.initial_parameters()?;
    // Coefficients from the last inner window are usually the strongest
    // data-derived retry, but its training basis has different knots. Reject
    // that transfer when the new objective exposes an explosive gradient.
    let retry_start = if let Some(candidate) = starts.retry {
        let candidate_evaluation =
            evaluate_objective(model.clone().into_workspace_objective(), candidate)?;
        let cold_evaluation =
            evaluate_objective(model.clone().into_workspace_objective(), &cold_start)?;
        if candidate_evaluation.value.is_finite()
            && candidate_evaluation.gradient_norm.is_finite()
            && candidate_evaluation.value <= cold_evaluation.value
            && candidate_evaluation.gradient_norm
                <= RETRY_GRADIENT_RATIO_LIMIT * cold_evaluation.gradient_norm.max(1.0)
        {
            candidate
        } else {
            &cold_start
        }
    } else {
        &cold_start
    };
    let fit = optimize_with_retry(
        || model.clone().into_workspace_objective(),
        starts.primary,
        retry_start,
        optimization,
    )?;

    let validation_blocks = split.validation_designs.blocks(smoothing)?;
    let fitted = model.predict_theta_with_blocks(&fit.parameters, &validation_blocks)?;
    let validation_nll = split
        .validation
        .response
        .iter()
        .copied()
        .zip(&fitted)
        .map(|(observation, theta)| model.family().nll(observation, theta, &mut ()))
        .sum::<f64>();
    let validation_mean_nll = validation_nll / split.validation.len() as f64;
    if !validation_mean_nll.is_finite() {
        return Err(ExampleError::boxed(
            "validation negative log-likelihood is non-finite",
        ));
    }

    Ok(ScoredFit {
        fit,
        validation_mean_nll,
    })
}

#[derive(Clone, Debug)]
struct NestedWarmStarts {
    inner: Vec<Option<Vec<f64>>>,
    smoothing_count: usize,
    outer: Option<Vec<f64>>,
}

impl NestedWarmStarts {
    fn new(inner_folds: usize, smoothing_count: usize) -> Self {
        Self {
            inner: vec![None; inner_folds * smoothing_count],
            smoothing_count,
            outer: None,
        }
    }

    fn inner_index(&self, inner_fold: usize, smoothing: usize) -> usize {
        debug_assert!(smoothing < self.smoothing_count);
        let index = inner_fold * self.smoothing_count + smoothing;
        debug_assert!(index < self.inner.len());
        index
    }
}

#[derive(Clone, Debug)]
struct SmoothingSelection {
    smoothing: Smoothing,
    inner_mean_nll: f64,
    refit_start: Vec<f64>,
}

fn smoothing_is_better(
    candidate_score: f64,
    candidate: Smoothing,
    best_score: f64,
    best: Smoothing,
) -> bool {
    match candidate_score.total_cmp(&best_score) {
        std::cmp::Ordering::Less => true,
        std::cmp::Ordering::Greater => false,
        std::cmp::Ordering::Equal => candidate.mean > best.mean,
    }
}

fn select_smoothing(
    family: PolicyFamily,
    inner: &[SplitContext<'_>],
    target: f64,
    smoothings: &[Smoothing],
    warm_starts: &mut NestedWarmStarts,
    stats: &mut OptimizationStats,
) -> ExampleResult<SmoothingSelection> {
    let mut score_sums = vec![0.0; smoothings.len()];

    for (inner_index, split) in inner.iter().enumerate() {
        let train_designs = PolicyDesigns::try_new(split.train, split.bases, target)?;
        let validation_designs = PolicyDesigns::try_new(split.validation, split.bases, target)?;
        let mut preceding_smoothing_fit: Option<Vec<f64>> = None;

        for (smoothing_index, smoothing) in smoothings.iter().copied().enumerate() {
            let cache_index = warm_starts.inner_index(inner_index, smoothing_index);
            let cached = warm_starts.inner[cache_index].as_deref();
            let start = cached.or(preceding_smoothing_fit.as_deref());
            let scored = fit_and_score(
                family,
                DesignedSplit {
                    train: split.train,
                    validation: split.validation,
                    train_designs: &train_designs,
                    validation_designs: &validation_designs,
                },
                smoothing,
                FitStarts {
                    primary: start,
                    retry: None,
                },
                stats,
            )
            .map_err(|error| {
                ExampleError::boxed(format!(
                    "inner fold {}, lambda_mu={}: {error}",
                    inner_index + 1,
                    smoothing.mean,
                ))
            })?;
            score_sums[smoothing_index] += scored.validation_mean_nll;
            preceding_smoothing_fit = Some(scored.fit.parameters.clone());
            warm_starts.inner[cache_index] = Some(scored.fit.parameters);
        }
    }

    let fold_count = inner.len() as f64;
    let mut best_index = 0;
    let mut best_score = score_sums[0] / fold_count;
    for index in 1..smoothings.len() {
        let score = score_sums[index] / fold_count;
        if smoothing_is_better(score, smoothings[index], best_score, smoothings[best_index]) {
            best_index = index;
            best_score = score;
        }
    }
    let last_inner = inner.len() - 1;
    let refit_index = warm_starts.inner_index(last_inner, best_index);
    let refit_start = warm_starts.inner[refit_index]
        .clone()
        .ok_or_else(|| ExampleError::boxed("selected smoothing has no fitted parameters"))?;

    Ok(SmoothingSelection {
        smoothing: smoothings[best_index],
        inner_mean_nll: best_score,
        refit_start,
    })
}

#[derive(Clone, Copy, Debug)]
struct FoldEvaluation {
    validation_mean_nll: f64,
}

fn evaluate_target(
    family: PolicyFamily,
    context: &NestedContext<'_>,
    target: f64,
    smoothings: &[Smoothing],
    warm_starts: &mut NestedWarmStarts,
    stats: &mut OptimizationStats,
) -> ExampleResult<FoldEvaluation> {
    let selection = select_smoothing(
        family,
        &context.inner,
        target,
        smoothings,
        warm_starts,
        stats,
    )?;
    let train_designs = PolicyDesigns::try_new(context.outer.train, context.outer.bases, target)?;
    let validation_designs =
        PolicyDesigns::try_new(context.outer.validation, context.outer.bases, target)?;
    let neighboring_target_start = warm_starts.outer.as_deref();
    let start = neighboring_target_start.unwrap_or(&selection.refit_start);
    let scored = fit_and_score(
        family,
        DesignedSplit {
            train: context.outer.train,
            validation: context.outer.validation,
            train_designs: &train_designs,
            validation_designs: &validation_designs,
        },
        selection.smoothing,
        FitStarts {
            primary: Some(start),
            retry: Some(&selection.refit_start),
        },
        stats,
    )
    .map_err(|error| {
        ExampleError::boxed(format!(
            "outer refit, lambda_mu={}: {error}",
            selection.smoothing.mean,
        ))
    })?;
    warm_starts.outer = Some(scored.fit.parameters);

    Ok(FoldEvaluation {
        validation_mean_nll: scored.validation_mean_nll,
    })
}

fn central_target_index(targets: &[f64]) -> usize {
    let midpoint = f64::midpoint(TARGET_MIN, TARGET_MAX);
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
}

#[derive(Clone, Copy, Debug)]
struct LagSummary {
    lag: usize,
    target: f64,
    mean_nll: f64,
    standard_error: f64,
}

#[derive(Clone, Debug)]
struct FinalFitReport {
    smoothing: Smoothing,
    inner_mean_nll: f64,
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
    pit_mean_sd: (f64, f64),
    residual_mean_sd: (f64, f64),
    coverage_90: f64,
}

#[derive(Clone, Debug)]
struct AnalysisResult {
    lag_summaries: Vec<LagSummary>,
    selected_lag: usize,
    selected_target: f64,
    selected_mean_nll: f64,
    selected_standard_error: f64,
    one_se_range: (f64, f64),
    final_fit: FinalFitReport,
    optimization: OptimizationStats,
}

const fn profile_index(target_count: usize, lag_index: usize, target_index: usize) -> usize {
    lag_index * target_count + target_index
}

struct TargetScanner<'scan, 'data> {
    family: PolicyFamily,
    context: &'scan NestedContext<'data>,
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
        warm_starts: &mut NestedWarmStarts,
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
                &self.config.smoothings,
                warm_starts,
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
    let mut optimization = OptimizationStats::default();

    for (lag_index, lag) in config.lags.iter().copied().enumerate() {
        let context = NestedContext::for_outer_fold(data, config, lag, outer_fold)?;
        let center = central_target_index(&config.targets);
        let mut center_warm = NestedWarmStarts::new(config.inner_folds, config.smoothings.len());
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

    if scores.iter().any(|score| !score.is_finite()) {
        return Err(ExampleError::boxed(format!(
            "outer fold {} produced an incomplete or non-finite profile",
            outer_fold + 1
        )));
    }

    Ok(OuterFoldResult {
        scores,
        optimization,
    })
}

fn run_analysis(data: &PreparedData, config: &AnalysisConfig) -> ExampleResult<AnalysisResult> {
    config.validate(data)?;
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
        optimization.add(fold.optimization);
        for (point, score) in profile.iter_mut().zip(fold.scores) {
            point.fold_nll.push(score);
        }
    }

    let (lag_summaries, selected_index) = summarize_profile(&profile, &config.lags, targets_len)?;
    let selected_lag = profile[selected_index].lag;
    let selected_target = profile[selected_index].target;
    let (selected_mean_nll, selected_standard_error) = profile[selected_index].mean_se();
    let one_se_range = contiguous_one_se_range(
        &profile,
        selected_index,
        targets_len,
        selected_standard_error,
    );
    let analysis_len = config
        .analysis_len()
        .expect("validated analysis length must fit");
    let final_fit = fit_final_model(
        data,
        config,
        family,
        analysis_len,
        selected_lag,
        selected_target,
        &mut optimization,
    )?;

    Ok(AnalysisResult {
        lag_summaries,
        selected_lag,
        selected_target,
        selected_mean_nll,
        selected_standard_error,
        one_se_range,
        final_fit,
        optimization,
    })
}

fn summarize_profile(
    profile: &[ProfilePoint],
    lags: &[usize],
    target_count: usize,
) -> ExampleResult<(Vec<LagSummary>, usize)> {
    if profile.len() != lags.len() * target_count || target_count == 0 {
        return Err(ExampleError::boxed(
            "profile dimensions do not match lag and target grids",
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
        lag_summaries.push(LagSummary {
            lag,
            target: profile[best].target,
            mean_nll,
            standard_error,
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

fn contiguous_one_se_range(
    profile: &[ProfilePoint],
    selected_index: usize,
    target_count: usize,
    standard_error: f64,
) -> (f64, f64) {
    let lag_start = selected_index / target_count * target_count;
    let lag_end = lag_start + target_count;
    let threshold = profile[selected_index].mean_se().0 + standard_error;
    let mut left = selected_index;
    while left > lag_start && profile[left - 1].mean_se().0 <= threshold {
        left -= 1;
    }
    let mut right = selected_index;
    while right + 1 < lag_end && profile[right + 1].mean_se().0 <= threshold {
        right += 1;
    }
    (profile[left].target, profile[right].target)
}

fn fit_final_model(
    data: &PreparedData,
    config: &AnalysisConfig,
    family: PolicyFamily,
    analysis_len: usize,
    lag: usize,
    target: f64,
    stats: &mut OptimizationStats,
) -> ExampleResult<FinalFitReport> {
    let full_rows = data.rows(lag, 0..analysis_len)?;
    let inner = inner_ranges(analysis_len, config.inner_folds, config.fold_months)?
        .into_iter()
        .map(|(train, validation)| {
            SplitContext::new(data.rows(lag, train)?, data.rows(lag, validation)?)
        })
        .collect::<ExampleResult<Vec<_>>>()?;
    let mut warm_starts = NestedWarmStarts::new(config.inner_folds, config.smoothings.len());
    let selection = select_smoothing(
        family,
        &inner,
        target,
        &config.smoothings,
        &mut warm_starts,
        stats,
    )?;
    let bases = SplineBases::from_training(full_rows)?;
    let designs = PolicyDesigns::try_new(full_rows, bases, target)?;
    let blocks = designs.blocks(selection.smoothing)?;
    let model = Gamlss::try_new(family, blocks, full_rows.response)?
        .with_objective_scale(ObjectiveScale::Mean);
    let cold_start = model.initial_parameters()?;
    let fit = optimize_with_retry(
        || model.clone().into_workspace_objective(),
        Some(&selection.refit_start),
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

    Ok(FinalFitReport {
        smoothing: selection.smoothing,
        inner_mean_nll: selection.inner_mean_nll,
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
        pit_mean_sd,
        residual_mean_sd,
        coverage_90,
    })
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
        "model=zero-adjusted Student-t(df={STUDENT_T_DF:.0}), anchored mean without intercept, constant scale/zero mass, mean_bases={MEAN_BASIS}+{MEAN_BASIS}, lags={:?}, targets={TARGET_MIN:.1}..={TARGET_MAX:.1} by 0.1",
        config.lags,
    );
    println!(
        "outer_folds={}, inner_folds={}, fold_months={}, lambda_levels={SMOOTHING_LEVELS_DESC:?}, rayon_threads={}",
        config.outer_folds,
        config.inner_folds,
        config.fold_months,
        rayon::current_num_threads(),
    );
    println!();
    println!("lag_months  best_target  mean_validation_nll  standard_error");
    for summary in &analysis.lag_summaries {
        println!(
            "{:>10}  {:>11.1}  {:>19.6}  {:>14.6}",
            summary.lag, summary.target, summary.mean_nll, summary.standard_error,
        );
    }
    println!();
    println!(
        "selected lag={} months, target={:.1}%, validation mean NLL={:.6}, SE={:.6}",
        analysis.selected_lag,
        analysis.selected_target,
        analysis.selected_mean_nll,
        analysis.selected_standard_error,
    );
    println!(
        "one-SE predictive-stability range=[{:.1}%, {:.1}%] (heuristic; not a confidence interval)",
        analysis.one_se_range.0, analysis.one_se_range.1,
    );
    let official_inside =
        (analysis.one_se_range.0..=analysis.one_se_range.1).contains(&OFFICIAL_TARGET);
    println!(
        "official benchmark revealed after selection: {OFFICIAL_TARGET:.1}%, estimate-minus-benchmark={:+.1} pp, inside one-SE range={official_inside}",
        analysis.selected_target - OFFICIAL_TARGET,
    );
    println!();
    println!(
        "final nested-CV smoothing: lambda_mu={}, inner mean NLL={:.6}",
        analysis.final_fit.smoothing.mean, analysis.final_fit.inner_mean_nll,
    );
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
    use gamlss::core::DesignMatrix;
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
        }
    }

    #[test]
    fn full_outer_and_inner_folds_are_ordered_and_disjoint() {
        let data = PreparedData::load().unwrap();
        let config = AnalysisConfig::full();
        config.validate(&data).unwrap();

        let (first_train, first_validation) = outer_ranges(&config, 0).unwrap();
        assert_eq!(first_train, 0..48);
        assert_eq!(first_validation, 48..54);
        let first_inner =
            inner_ranges(first_train.end, config.inner_folds, config.fold_months).unwrap();
        assert_eq!(first_inner[0], (0..30, 30..36));
        assert_eq!(first_inner[2], (0..42, 42..48));

        let (last_train, last_validation) = outer_ranges(&config, 10).unwrap();
        assert_eq!(last_train, 0..108);
        assert_eq!(last_validation, 108..114);

        for outer_fold in 0..config.outer_folds {
            let (outer_train, outer_validation) = outer_ranges(&config, outer_fold).unwrap();
            assert_eq!(outer_train.start, 0);
            assert!(outer_train.end <= outer_validation.start);
            assert_eq!(outer_validation.len(), FOLD_MONTHS);

            let inner =
                inner_ranges(outer_train.end, config.inner_folds, config.fold_months).unwrap();
            for (train, validation) in inner {
                assert_eq!(train.start, 0);
                assert!(train.end <= validation.start);
                assert!(validation.end <= outer_train.end);
                assert_eq!(validation.len(), FOLD_MONTHS);
            }
        }
    }

    #[test]
    fn analysis_grids_cover_requested_targets_and_lags() {
        let targets = target_grid();
        assert_eq!(targets.len(), 61);
        assert_eq!(targets[0], 2.0);
        assert_eq!(targets[20], 4.0);
        assert_eq!(targets[60], 8.0);
        assert_eq!(REACTION_LAGS, [1, 2, 3, 6, 9, 12]);
        assert!(!REACTION_LAGS.contains(&0));
        let smoothings = smoothing_grid();
        assert_eq!(smoothings.len(), 3);
        for mean in [0.01, 0.1, 1.0] {
            assert!(smoothings.contains(&Smoothing::new(mean)));
        }
    }

    #[test]
    fn anchored_design_is_zero_at_the_target_and_reuses_basis() {
        let basis = OpenUniformSplineBasis::new(0.0, 10.0, MEAN_BASIS, SplineOrder::Cubic).unwrap();
        let training = anchored_spline_design(basis, &[0.0, 2.0, 7.0]).unwrap();
        let validation = anchored_spline_design(basis, &[0.0, 5.0]).unwrap();

        assert_eq!(training.ncols(), MEAN_BASIS - 1);
        assert_eq!(validation.ncols(), training.ncols());
        for value in &training.values()[..training.ncols()] {
            assert_abs_diff_eq!(*value, 0.0, epsilon = 1.0e-15);
        }
        for value in &validation.values()[..validation.ncols()] {
            assert_abs_diff_eq!(*value, 0.0, epsilon = 1.0e-15);
        }
    }

    #[test]
    fn smoothing_ties_prefer_stronger_mean_penalty() {
        assert!(smoothing_is_better(
            1.0,
            Smoothing::new(1.0),
            1.0,
            Smoothing::new(0.1),
        ));
        assert!(!smoothing_is_better(
            1.1,
            Smoothing::new(1.0),
            1.0,
            Smoothing::new(0.01),
        ));
    }

    #[test]
    fn one_se_range_keeps_only_the_contiguous_component() {
        let scores = [
            (2.0, vec![1.10, 1.10]),
            (2.1, vec![0.95, 0.95]),
            (2.2, vec![0.90, 0.90]),
            (2.3, vec![0.95, 0.95]),
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
        let standard_error = 0.1;

        assert_eq!(
            contiguous_one_se_range(&profile, 2, profile.len(), standard_error),
            (2.1, 2.3)
        );
    }

    #[test]
    fn reduced_nested_pipeline_has_finite_scores_and_gradient() {
        let data = PreparedData::load().unwrap();
        let config = AnalysisConfig {
            lags: vec![9],
            targets: vec![6.0],
            smoothings: vec![Smoothing::new(0.01)],
            initial_train: OUTER_INITIAL_TRAIN,
            outer_folds: 1,
            inner_folds: INNER_FOLDS,
            fold_months: FOLD_MONTHS,
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
            smoothings: vec![Smoothing::new(1.0)],
            initial_train: OUTER_INITIAL_TRAIN,
            outer_folds: 1,
            inner_folds: INNER_FOLDS,
            fold_months: FOLD_MONTHS,
            show_progress: false,
        };

        let fold = evaluate_outer_fold(&data, &config, policy_family().unwrap(), 0).unwrap();

        assert_eq!(fold.scores.len(), 3);
        assert!(fold.scores.iter().all(|score| score.is_finite()));
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

        let (summaries, selected) = summarize_profile(&profile, &[1, 3], 2).unwrap();

        assert_eq!(summaries[0].target, 2.1);
        assert_eq!(summaries[1].target, 2.0);
        assert_eq!(selected, 2);
    }
}
