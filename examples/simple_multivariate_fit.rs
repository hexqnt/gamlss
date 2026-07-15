//! Simple example: fitting a bivariate normal GAMLSS model to fifteen observations.
//!
//! Both means have a linear predictor, while both marginal standard deviations
//! and the correlation are intercept-only. The model is fitted with gradient
//! descent, mirroring the minimal workflow in `simple_fit.rs`.
#![allow(clippy::cast_precision_loss)]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::array;

    use gamlss::core::{
        DenseDesign, Gamlss, LinearPredictorBlock, Mu, NoPenalty, Objective, ParameterBlocks,
        PartialCorrelation, Sigma, StrictLowerTriangularParameterBlock, VectorParameterBlock,
    };
    use gamlss::diagnostics::MarginalCdfDiagnosticsExt;
    use gamlss::family::MvNormalMeanStdPartialCorrDefault;

    const D: usize = 2;

    let x = [
        -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4,
    ];
    let y = [
        [1.925, 3.220],
        [0.525, 2.510],
        [1.750, 3.425],
        [1.100, 2.090],
        [2.700, 3.755],
        [1.550, 2.420],
        [2.650, 3.335],
        [1.000, 2.125],
        [2.475, 3.540],
        [1.575, 2.705],
        [3.175, 3.870],
        [2.025, 2.535],
        [3.000, 3.450],
        [1.850, 2.740],
        [2.825, 3.405],
    ];
    let n = y.len();

    let mean_design = DenseDesign::from_rows(&x.map(|x_i| [1.0, x_i]));
    let intercept = DenseDesign::intercept(n);
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
            vec![LinearPredictorBlock::new(&intercept)],
            NoPenalty,
            mu.len() + sigma.len(),
        );
    let blocks = ParameterBlocks::new((mu, sigma, partial_correlation));
    let mut model = Gamlss::try_new_with_observations(
        MvNormalMeanStdPartialCorrDefault::<D>::new(),
        blocks,
        y.as_slice(),
    )?;

    let mut parameters = model.initial_parameters()?;
    let mut grad = vec![0.0; model.nparams()];

    for _ in 0..10_000 {
        model.gradient(&parameters, &mut grad)?;
        for (parameter, grad_value) in parameters.iter_mut().zip(&grad) {
            *parameter -= 0.002 * grad_value;
        }
    }

    let diagnostics = model.training_diagnostics(&parameters)?;
    let coefficients = model.unpack_parameters(&parameters)?;
    let mu_coefficients: Vec<_> = coefficients
        .blocks_of::<Mu>()
        .map(|block| block.coefficients.as_slice())
        .collect();
    let theta = model.predict_theta_row(&parameters, 0)?;
    let rho = theta
        .partial_corr()
        .get(1, 0)
        .expect("the bivariate model has one correlation");

    let y_0 = y.map(|observation| observation[0]);
    let y_1 = y.map(|observation| observation[1]);
    let pit_0 = model.marginal_pit_values(&parameters, 0, &y_0)?;
    let pit_1 = model.marginal_pit_values(&parameters, 1, &y_1)?;
    let pit_0_range = finite_range(&pit_0);
    let pit_1_range = finite_range(&pit_1);

    println!(
        "simple_multivariate_fit: objective={:.6}, grad_norm={:.6}, mu_0=[{:.4}, {:.4}], mu_1=[{:.4}, {:.4}], sigma=[{:.4}, {:.4}], rho={rho:.4}, pit_0=[{:.4}, {:.4}], pit_1=[{:.4}, {:.4}]",
        diagnostics.objective,
        diagnostics.gradient_norm,
        mu_coefficients[0][0],
        mu_coefficients[0][1],
        mu_coefficients[1][0],
        mu_coefficients[1][1],
        theta.sigma()[0],
        theta.sigma()[1],
        pit_0_range.0,
        pit_0_range.1,
        pit_1_range.0,
        pit_1_range.1,
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
