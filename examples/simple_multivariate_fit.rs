//! Simple example: fitting a four-dimensional normal GAMLSS model to synthetic data.
//!
//! All four means have linear predictors, while the marginal standard deviations
//! and six partial correlations are intercept-only. The observations are generated
//! reproducibly through the same family implementation that is subsequently fitted.
#![allow(clippy::cast_precision_loss)]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::array;

    use gamlss::core::{
        DenseDesign, Gamlss, LinearPredictorBlock, Mu, NoPenalty, Objective, ObjectiveScale,
        ParameterBlocks, PartialCorrelation, Sigma, StrictLowerTriangularParameterBlock,
        TrySimulate, VectorParameterBlock,
    };
    use gamlss::diagnostics::MarginalCdfDiagnosticsExt;
    use gamlss::family::{
        FixedPartialCorrelations, MvNormalMeanStdPartialCorrDefault,
        MvNormalMeanStdPartialCorrTheta,
    };
    use rand::{SeedableRng, rngs::StdRng};

    // ANCHOR: data
    const D: usize = 4;
    const N: usize = 400;
    const PARTIAL_CORRELATION_COUNT: usize = D * (D - 1) / 2;

    let x: [f64; N] = array::from_fn(|index| {
        let fraction = index as f64 / (N - 1) as f64;
        2.0_f64.mul_add(fraction, -1.0)
    });
    let true_mu_intercepts: [f64; D] = [1.0, -0.5, 0.25, 1.5];
    let true_mu_slopes: [f64; D] = [0.8, -0.6, 0.4, -0.3];
    let true_sigma: [f64; D] = [0.45, 0.70, 0.55, 0.80];
    // Packed row-major order: (1,0), (2,0), (2,1), (3,0), (3,1), (3,2).
    let true_partial_correlation_values = [0.45, -0.25, 0.30, 0.15, -0.35, 0.20];
    let true_partial_correlation =
        FixedPartialCorrelations::<D>::try_new(true_partial_correlation_values.to_vec())?;

    let family = MvNormalMeanStdPartialCorrDefault::<D>::new();
    let generating_theta = x
        .iter()
        .map(|&x_i| {
            let mu = array::from_fn(|component| {
                true_mu_slopes[component].mul_add(x_i, true_mu_intercepts[component])
            });
            MvNormalMeanStdPartialCorrTheta::try_new(mu, true_sigma, true_partial_correlation)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut rng = StdRng::seed_from_u64(42);
    let mut y = vec![[0.0; D]; N];
    family.try_fill_varying(&mut rng, &generating_theta, &mut y)?;
    // ANCHOR_END: data

    // ANCHOR: model
    let mean_rows = x.iter().map(|&x_i| [1.0, x_i]).collect::<Vec<_>>();
    let mean_design = DenseDesign::from_rows(&mean_rows);
    let intercept = DenseDesign::intercept(N);
    let mu = VectorParameterBlock::<Mu, D, _, _>::new(
        array::from_fn(|_| LinearPredictorBlock::new(&mean_design)),
        NoPenalty,
        0,
    );
    let sigma = VectorParameterBlock::<Sigma, D, _, _>::new(
        array::from_fn(|_| LinearPredictorBlock::new(&intercept)),
        NoPenalty,
        mu.len(),
    );
    let partial_correlation =
        StrictLowerTriangularParameterBlock::<PartialCorrelation, D, _, _>::new(
            (0..PARTIAL_CORRELATION_COUNT)
                .map(|_| LinearPredictorBlock::new(&intercept))
                .collect(),
            NoPenalty,
            mu.len() + sigma.len(),
        );
    let blocks = ParameterBlocks::new((mu, sigma, partial_correlation));
    let mut model = Gamlss::try_new_with_observations(family, blocks, y.as_slice())?
        .with_objective_scale(ObjectiveScale::Mean);
    // ANCHOR_END: model

    // ANCHOR: fit
    let mut parameters = model.initial_parameters()?;
    let mut candidate = parameters.clone();
    let mut gradient = vec![0.0; model.nparams()];
    let mut step_size: f64 = 0.1;
    let mut iterations = 0;

    for iteration in 0..20_000 {
        model.gradient(&parameters, &mut gradient)?;
        let squared_gradient_norm = gradient.iter().map(|value| value * value).sum::<f64>();
        if squared_gradient_norm.sqrt() < 1.0e-6 {
            iterations = iteration;
            break;
        }

        let objective = model.try_value(&parameters)?;
        loop {
            for ((next, parameter), gradient_value) in
                candidate.iter_mut().zip(&parameters).zip(&gradient)
            {
                *next = step_size.mul_add(-gradient_value, *parameter);
            }
            let candidate_objective = model.try_value(&candidate)?;
            if candidate_objective.is_finite() && candidate_objective < objective {
                parameters.copy_from_slice(&candidate);
                step_size = (step_size * 1.05).min(0.5);
                break;
            }
            step_size *= 0.5;
            if step_size < 1.0e-12 {
                return Err(format!(
                    "gradient descent line search failed at iteration {iteration}: objective={objective}, gradient_norm={}",
                    squared_gradient_norm.sqrt()
                )
                .into());
            }
        }
        iterations = iteration + 1;
    }
    // ANCHOR_END: fit

    // ANCHOR: diagnostics
    let diagnostics = model.training_diagnostics(&parameters)?;
    let coefficients = model.unpack_parameters(&parameters)?;
    let mu_coefficients = coefficients
        .blocks_of::<Mu>()
        .map(|block| [block.coefficients[0], block.coefficients[1]])
        .collect::<Vec<_>>();
    let fitted_theta = model.predict_theta_row(&parameters, 0)?;

    println!(
        "simple_multivariate_fit: n={N}, dimension={D}, parameters={}, iterations={iterations}, objective={:.6}, grad_norm={:.3e}",
        model.nparams(),
        diagnostics.objective,
        diagnostics.gradient_norm,
    );
    for component in 0..D {
        let observed = y
            .iter()
            .map(|observation| observation[component])
            .collect::<Vec<_>>();
        let pit = model.marginal_pit_values(&parameters, component, &observed)?;
        let (pit_min, pit_max) = finite_range(&pit);
        let pit_mean = finite_mean(&pit);
        println!(
            "component {component}: mu=[{:.4}, {:.4}] (true [{:.4}, {:.4}]), sigma={:.4} (true {:.4}), pit_mean={pit_mean:.4}, pit=[{pit_min:.4}, {pit_max:.4}]",
            mu_coefficients[component][0],
            mu_coefficients[component][1],
            true_mu_intercepts[component],
            true_mu_slopes[component],
            fitted_theta.sigma()[component],
            true_sigma[component],
        );
    }

    let mut packed_index = 0;
    for row in 1..D {
        for col in 0..row {
            let fitted = fitted_theta
                .partial_corr()
                .get(row, col)
                .expect("loop visits a strict-lower coordinate");
            println!(
                "partial_corr({row},{col})={fitted:.4} (true {:.4})",
                true_partial_correlation_values[packed_index],
            );
            packed_index += 1;
        }
    }
    // ANCHOR_END: diagnostics

    Ok(())
}

fn finite_range(values: &[f64]) -> (f64, f64) {
    let range = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), value| {
            (min.min(value), max.max(value))
        });

    if range.0.is_finite() && range.1.is_finite() {
        range
    } else {
        (f64::NAN, f64::NAN)
    }
}

fn finite_mean(values: &[f64]) -> f64 {
    let (sum, count) = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold((0.0, 0_usize), |(sum, count), value| {
            (sum + value, count + 1)
        });
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}
