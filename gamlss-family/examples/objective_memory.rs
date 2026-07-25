#![allow(clippy::cast_precision_loss)]

use std::{env, hint::black_box, process::ExitCode};

use gamlss_core::{
    DenseDesign, DenseRows, DynamicParameterBlocks, Family, Gamlss, GamlssBlocks,
    LinearPredictorBlock, ModelError, ModelWorkspace, NoPenalty, ObservationView,
};
use gamlss_family::DynMvNormalCholeskyDefault;

const USAGE: &str =
    "usage: objective_memory [dimension] [nobs] [ncols] [warm|cold|retained|run] [iterations]";

#[global_allocator]
static ALLOCATOR: dhat::Alloc = dhat::Alloc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Warm,
    Cold,
    Retained,
    Run,
}

impl Mode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "warm" => Ok(Self::Warm),
            "cold" => Ok(Self::Cold),
            "retained" => Ok(Self::Retained),
            "run" => Ok(Self::Run),
            _ => Err(format!(
                "invalid mode {value:?}; expected warm, cold, retained, or run"
            )),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Warm => "warm",
            Self::Cold => "cold",
            Self::Retained => "retained",
            Self::Run => "run",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Config {
    dimension: usize,
    nobs: usize,
    ncols: usize,
    mode: Mode,
    iterations: usize,
}

impl Config {
    fn parse() -> Result<Self, String> {
        let mut arguments = env::args().skip(1);
        let dimension = parse_positive("dimension", arguments.next().as_deref().unwrap_or("8"))?;
        let nobs = parse_positive("nobs", arguments.next().as_deref().unwrap_or("1000"))?;
        let ncols = parse_positive("ncols", arguments.next().as_deref().unwrap_or("8"))?;
        let mode = Mode::parse(arguments.next().as_deref().unwrap_or("warm"))?;
        let iterations = parse_positive("iterations", arguments.next().as_deref().unwrap_or("10"))?;
        if arguments.next().is_some() {
            return Err(String::from(USAGE));
        }
        Ok(Self {
            dimension,
            nobs,
            ncols,
            mode,
            iterations,
        })
    }
}

fn parse_positive(name: &str, value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("invalid {name} {value:?}: {error}"))?;
    if parsed == 0 {
        Err(format!("{name} must be positive"))
    } else {
        Ok(parsed)
    }
}

fn dense_design(nobs: usize, ncols: usize) -> DenseDesign {
    let values = (0..nobs)
        .flat_map(|row| {
            (0..ncols).map(move |col| {
                if col == 0 {
                    1.0
                } else {
                    let period = 17 + 2 * col;
                    2.0 * ((row * (col + 3)) % period) as f64 / (period - 1) as f64 - 1.0
                }
            })
        })
        .collect();
    DenseDesign::from_row_major(nobs, ncols, values).unwrap()
}

fn flat_observations(nobs: usize, dimension: usize) -> Vec<f64> {
    (0..nobs)
        .flat_map(|row| {
            (0..dimension).map(move |component| {
                let trend = 2.0 * (row % 101) as f64 / 100.0 - 1.0;
                let ripple = 2.0 * ((row * 17 + component * 7) % 29) as f64 / 28.0 - 1.0;
                0.08_f64.mul_add(
                    ripple,
                    0.025_f64
                        .mul_add(component as f64, 0.3)
                        .mul_add(trend, 0.15 * (component + 1) as f64),
                )
            })
        })
        .collect()
}

fn start_profiler(config: Config) -> dhat::Profiler {
    let file_name = format!(
        "dhat-objective-d{}-n{}-p{}-{}.json",
        config.dimension,
        config.nobs,
        config.ncols,
        config.mode.name()
    );
    dhat::Profiler::builder().file_name(file_name).build()
}

fn evaluate_once<F, Blocks, Obs>(
    model: &Gamlss<F, Blocks, Obs>,
    beta: &[f64],
    gradient: &mut [f64],
    workspace: &mut ModelWorkspace<F>,
) -> Result<f64, ModelError>
where
    F: Family,
    Blocks: GamlssBlocks<F>,
    for<'row> Obs: ObservationView<'row, Observation = F::Observation<'row>>,
{
    model.try_likelihood_value_gradient_into_workspace(
        black_box(beta),
        black_box(gradient),
        workspace,
    )
}

fn evaluate_repeated<F, Blocks, Obs>(
    model: &Gamlss<F, Blocks, Obs>,
    beta: &[f64],
    gradient: &mut [f64],
    workspace: &mut ModelWorkspace<F>,
    iterations: usize,
) -> Result<f64, ModelError>
where
    F: Family,
    Blocks: GamlssBlocks<F>,
    for<'row> Obs: ObservationView<'row, Observation = F::Observation<'row>>,
{
    let mut checksum = 0.0;
    for _ in 0..iterations {
        checksum += evaluate_once(model, beta, gradient, workspace)?;
    }
    black_box(&gradient);
    Ok(checksum)
}

fn run(config: Config) -> Result<f64, String> {
    let flat = flat_observations(config.nobs, config.dimension);
    let observations = DenseRows::try_new(&flat, config.dimension)
        .map_err(|error| format!("failed to build observations: {error}"))?;
    let shared_design = dense_design(config.nobs, config.ncols);
    let family = DynMvNormalCholeskyDefault::new(config.dimension)
        .map_err(|error| format!("failed to build family: {error}"))?;
    let coordinate_count = family.dimension()
        + DynMvNormalCholeskyDefault::cholesky_len_for_dimension(family.dimension())
            .ok_or_else(|| String::from("dynamic coordinate count overflow"))?;
    let predictors = (0..coordinate_count)
        .map(|_| (LinearPredictorBlock::new(&shared_design), NoPenalty))
        .collect();
    let blocks = DynamicParameterBlocks::try_new(&family, predictors)
        .map_err(|error| format!("failed to build parameter blocks: {error}"))?;
    let model = Gamlss::try_new_with_observations(family, blocks, observations)
        .map_err(|error| format!("failed to compile model: {error}"))?;
    let beta = vec![0.0; model.nparams()];
    let mut gradient = vec![0.0; model.nparams()];
    eprintln!(
        "objective_memory: d={}, n={}, p={}, k={}, beta={}, mode={}, iterations={}",
        config.dimension,
        config.nobs,
        config.ncols,
        coordinate_count,
        model.nparams(),
        config.mode.name(),
        config.iterations
    );

    let checksum = match config.mode {
        Mode::Warm => {
            let mut workspace = model.gradient_workspace();
            evaluate_once(&model, &beta, &mut gradient, &mut workspace)
                .map_err(|error| format!("warm-up failed: {error}"))?;
            let profiler = start_profiler(config);
            let checksum = evaluate_repeated(
                &model,
                &beta,
                &mut gradient,
                &mut workspace,
                config.iterations,
            )
            .map_err(|error| format!("evaluation failed: {error}"))?;
            drop(profiler);
            checksum
        }
        Mode::Cold => {
            let profiler = start_profiler(config);
            let mut checksum = 0.0;
            for _ in 0..config.iterations {
                let mut workspace = model.gradient_workspace();
                checksum += evaluate_once(&model, &beta, &mut gradient, &mut workspace)
                    .map_err(|error| format!("evaluation failed: {error}"))?;
                black_box(workspace);
            }
            black_box(&gradient);
            drop(profiler);
            checksum
        }
        Mode::Retained => {
            let profiler = start_profiler(config);
            let mut workspace = model.gradient_workspace();
            let checksum = evaluate_repeated(
                &model,
                &beta,
                &mut gradient,
                &mut workspace,
                config.iterations,
            )
            .map_err(|error| format!("evaluation failed: {error}"))?;
            drop(profiler);
            black_box(workspace);
            checksum
        }
        Mode::Run => {
            let mut workspace = model.gradient_workspace();
            let checksum = evaluate_repeated(
                &model,
                &beta,
                &mut gradient,
                &mut workspace,
                config.iterations,
            )
            .map_err(|error| format!("evaluation failed: {error}"))?;
            black_box(workspace);
            checksum
        }
    };
    Ok(checksum)
}

fn main() -> ExitCode {
    let config = match Config::parse() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    match run(config) {
        Ok(checksum) => {
            eprintln!("objective_memory: checksum={checksum:.12e}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
