//! Simple example: fitting a normal GAMLSS model with a linear predictor for
//! `mu` and an intercept-only predictor for `sigma` using gradient descent.
//!
//! Demonstrates the minimal cycle of model assembly, gradient computation
//! and manual parameter updates. With the `rand` feature enabled, it also
//! generates one fitted-model simulation for every training row.
#![allow(clippy::cast_precision_loss)]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use gamlss::core::{
        DenseDesign, Gamlss, Identity, Log, Mu, NoPenalty, Objective, ParameterBlock,
        ParameterBlocks, Sigma,
    };
    use gamlss::diagnostics::CdfDiagnosticsExt;
    use gamlss::family::Normal;
    use gamlss_datasets::simulate::normal_linear;
    use rand::{SeedableRng, rngs::StdRng};

    // Generate synthetic training data.
    let x = [0.0, 1.0, 2.0, 3.0, 4.0];
    let mut rng = StdRng::seed_from_u64(42);
    let y = normal_linear(&x, 1.0, 0.4, 0.5, &mut rng)?;
    let n = y.len();
    let x_design = x.iter().map(|&x_i| [1.0, x_i]).collect::<Vec<_>>();

    // ANCHOR: model
    // 1. Assemble the typed model and its parameter blocks.
    let mu = ParameterBlock::<Mu, _, _>::linear(DenseDesign::from_rows(&x_design), NoPenalty, 0);
    let sigma =
        ParameterBlock::<Sigma, _, _>::linear(DenseDesign::intercept(n), NoPenalty, mu.len());
    let blocks = ParameterBlocks::new((mu, sigma));
    let mut model = Gamlss::try_new(Normal::<Identity, Log>::new(), blocks, &y)?;
    // ANCHOR_END: model

    // ANCHOR: fit
    // 2. Fit the model with manual gradient-descent updates.
    let mut parameters = model.initial_parameters()?;
    let mut grad = vec![0.0; model.dim()];

    for _ in 0..10_000 {
        model.gradient(&parameters, &mut grad)?;
        for (parameter, grad_value) in parameters.iter_mut().zip(&grad) {
            *parameter -= 0.002 * grad_value;
        }
    }
    // ANCHOR_END: fit

    // ANCHOR: prediction
    // 3. Predict distribution parameters and simulate fitted responses.
    let fitted_theta = model.predict_theta(&parameters)?;
    let fitted_rows = fitted_theta.len();

    #[cfg(feature = "rand")]
    let simulated = {
        use gamlss::core::TrySimulate;

        let mut simulation_rng = StdRng::seed_from_u64(43);
        let mut simulated = vec![f64::NAN; fitted_theta.len()];
        model
            .family()
            .try_fill_varying(&mut simulation_rng, &fitted_theta, &mut simulated)?;
        simulated
    };
    // ANCHOR_END: prediction

    // ANCHOR: diagnostics
    // 4. Inspect diagnostics for the fitted model.
    let diagnostics = model.training_diagnostics(&parameters)?;
    let pit = model.pit_values(&parameters)?;
    let residuals = model.quantile_residuals(&parameters)?;
    // ANCHOR_END: diagnostics

    // Unpack the fitted coefficients for the report.
    let coefficients = model.unpack_parameters(&parameters)?;
    let mu_coefficients = coefficients
        .unique_coefficients_of::<Mu>()?
        .expect("mu block is present");
    let mu_intercept = mu_coefficients[0];
    let mu_slope = mu_coefficients[1];
    let sigma_intercept = coefficients
        .unique_coefficients_of::<Sigma>()?
        .and_then(|values| values.first().copied())
        .expect("sigma block has an intercept coefficient");
    let sigma_hat = sigma_intercept.exp();

    // Report the fitted model, diagnostics, and simulation.
    let (pit_min, pit_max) = finite_range(&pit);
    let (residual_min, residual_max) = finite_range(&residuals);

    println!(
        "simple_fit: fitted_rows={fitted_rows}, objective={:.6}, grad_norm={:.6}, mu_intercept={mu_intercept:.4}, mu_slope={mu_slope:.4}, sigma_intercept={sigma_intercept:.4}, sigma={sigma_hat:.4}, pit=[{pit_min:.4}, {pit_max:.4}], qres=[{residual_min:.4}, {residual_max:.4}]",
        diagnostics.objective, diagnostics.gradient_norm,
    );

    #[cfg(feature = "rand")]
    println!("fitted-model simulation: {simulated:?}");

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
