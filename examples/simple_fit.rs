//! Простой пример: подгонка нормальной GAMLSS-модели с intercept-only
//! predictor-ами для `mu` и `sigma` градиентным спуском.
//!
//! Демонстрирует минимальный цикл сборки модели, вычисления градиента
//! и ручного обновления параметров.
#![allow(clippy::cast_precision_loss)]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use gamlss::core::{
        DenseDesign, Gamlss, Identity, Log, Mu, NoPenalty, Objective, ParameterBlock, Sigma,
    };
    use gamlss::family::Normal;

    let y = vec![1.0, 1.4, 1.8, 2.2, 2.6];
    let n = y.len();

    let mu = ParameterBlock::<Mu, Identity, _, _>::linear(DenseDesign::intercept(n), NoPenalty, 0);
    let sigma =
        ParameterBlock::<Sigma, Log, _, _>::linear(DenseDesign::intercept(n), NoPenalty, mu.len());
    let mut model = Gamlss::try_new(Normal::<Identity, Log>::new(), (mu, sigma), y)?;

    let mut theta = model.initial_theta()?;
    let mut grad = vec![0.0; model.dim()];

    for _ in 0..2_000 {
        model.gradient(&theta, &mut grad)?;
        for (theta_value, grad_value) in theta.iter_mut().zip(&grad) {
            *theta_value -= 0.02 * grad_value;
        }
    }

    let diagnostics = model.diagnostics(&theta)?;
    let coefficients = model.unpack_theta(&theta)?;
    let mu_hat = coefficients.coefficients("mu").expect("mu block")[0];
    let sigma_hat = coefficients.coefficients("sigma").expect("sigma block")[0].exp();

    println!(
        "simple_fit: objective={:.6}, grad_norm={:.6}, mu={mu_hat:.4}, sigma={sigma_hat:.4}",
        diagnostics.objective, diagnostics.gradient_norm,
    );

    Ok(())
}
