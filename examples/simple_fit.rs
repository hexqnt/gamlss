//! Simple example: fitting a normal GAMLSS model with intercept-only
//! predictors for `mu` and `sigma` using gradient descent.
//!
//! Demonstrates the minimal cycle of model assembly, gradient computation
//! and manual parameter updates.
#![allow(clippy::cast_precision_loss)]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use gamlss::core::{
        DenseDesign, Gamlss, Identity, Log, Mu, NoPenalty, Objective, ParameterBlock, Sigma,
    };
    use gamlss::diagnostics::CdfDiagnosticsExt;
    use gamlss::family::Normal;

    let y = vec![1.0, 1.4, 1.8, 2.2, 2.6];
    let n = y.len();

    let mu = ParameterBlock::<Mu, Identity, _, _>::linear(DenseDesign::intercept(n), NoPenalty, 0);
    let sigma =
        ParameterBlock::<Sigma, Log, _, _>::linear(DenseDesign::intercept(n), NoPenalty, mu.len());
    let mut model = Gamlss::try_new(Normal::<Identity, Log>::new(), (mu, sigma), &y)?;

    let mut parameters = model.initial_parameters()?;
    let mut grad = vec![0.0; model.dim()];

    for _ in 0..2_000 {
        model.gradient(&parameters, &mut grad)?;
        for (parameter, grad_value) in parameters.iter_mut().zip(&grad) {
            *parameter -= 0.02 * grad_value;
        }
    }

    let diagnostics = model.training_diagnostics(&parameters)?;
    let pit = model.pit_values(&parameters)?;
    let residuals = model.quantile_residuals(&parameters)?;
    let coefficients = model.unpack_parameters(&parameters)?;
    let mu_hat = coefficients
        .coefficients_of::<Mu>()
        .and_then(|values| values.first().copied())
        .expect("mu block has an intercept coefficient");
    let sigma_hat = coefficients
        .coefficients_of::<Sigma>()
        .and_then(|values| values.first().copied())
        .expect("sigma block has an intercept coefficient")
        .exp();

    let (pit_min, pit_max) = finite_range(&pit);
    let (residual_min, residual_max) = finite_range(&residuals);

    println!(
        "simple_fit: objective={:.6}, grad_norm={:.6}, mu={mu_hat:.4}, sigma={sigma_hat:.4}, pit=[{pit_min:.4}, {pit_max:.4}], qres=[{residual_min:.4}, {residual_max:.4}]",
        diagnostics.objective, diagnostics.gradient_norm,
    );

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
