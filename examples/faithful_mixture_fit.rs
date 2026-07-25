//! Simple example: fitting a two-component mixture of bivariate normal
//! distributions to the Old Faithful eruption and waiting-time data.
//!
//! Every component has an intercept-only mean vector and Cholesky covariance
//! factor, while the mixture weights are also intercept-only. The initialization
//! deliberately separates short- and long-waiting observations so that the two
//! exchangeable mixture components do not start at the same solution.
#![allow(clippy::cast_precision_loss)]

#[derive(Clone, Copy)]
struct EmpiricalComponent {
    count: usize,
    mean: [f64; 2],
    cholesky: [f64; 3],
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::array;

    use gamlss::core::{
        CholeskyScale, DenseDesign, Gamlss, LinearPredictorBlock, LowerTriangularParameterBlock,
        MixtureWeight, Mu, NoPenalty, Objective, ObjectiveScale, ParameterBlocks,
        SimplexLogitParameterBlock, VectorParameterBlock,
    };
    use gamlss::family::{Mixture, MvNormalCholeskyDefault};

    // ANCHOR: data
    const D: usize = 2;
    const COMPONENTS: usize = 2;

    let data = gamlss_datasets::faithful();
    let y = data.y;
    let n = y.len();
    // ANCHOR_END: data

    // ANCHOR: model
    let intercept = DenseDesign::intercept(n);
    let weights = SimplexLogitParameterBlock::<MixtureWeight, COMPONENTS, _, _>::new(
        vec![LinearPredictorBlock::new(&intercept)],
        NoPenalty,
        0,
    );
    let component_blocks = array::from_fn(|component| {
        let offset = weights.len() + component * 5;
        let mu = VectorParameterBlock::<Mu, D, _, _>::new(
            array::from_fn(|_| LinearPredictorBlock::new(&intercept)),
            NoPenalty,
            offset,
        );
        let cholesky = LowerTriangularParameterBlock::<CholeskyScale, D, _, _>::new(
            (0..3)
                .map(|_| LinearPredictorBlock::new(&intercept))
                .collect(),
            NoPenalty,
            offset + mu.len(),
        );
        (mu, cholesky)
    });
    let blocks = ParameterBlocks::new((weights, component_blocks));
    let family = Mixture::<_, COMPONENTS>::try_new(MvNormalCholeskyDefault::<D>::new())?;
    let mut model = Gamlss::try_new_with_observations(family, blocks, y)?
        .with_objective_scale(ObjectiveScale::Mean);
    // ANCHOR_END: model

    // ANCHOR: fit
    let starts = [
        empirical_component(y.iter().copied().filter(|row| row[1] < 70.0))?,
        empirical_component(y.iter().copied().filter(|row| row[1] >= 70.0))?,
    ];
    let mut parameters = initial_parameters(n, &starts);
    let mut candidate = parameters.clone();
    let mut gradient = vec![0.0; model.nparams()];
    let mut step_size: f64 = 0.05;
    let mut iterations = 0;

    for iteration in 0..20_000 {
        model.gradient(&parameters, &mut gradient)?;
        let gradient_norm = gradient
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if gradient_norm < 1.0e-6 {
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
                step_size = (step_size * 1.05).min(0.25);
                break;
            }
            step_size *= 0.5;
            if step_size < 1.0e-12 {
                return Err(format!(
                    "gradient descent line search failed at iteration {iteration}: objective={objective}, gradient_norm={gradient_norm}"
                )
                .into());
            }
        }
        iterations = iteration + 1;
    }
    // ANCHOR_END: fit

    // ANCHOR: diagnostics
    let diagnostics = model.training_diagnostics(&parameters)?;
    let fitted = model.predict_theta_row(&parameters, 0)?;

    println!(
        "faithful_mixture_fit: n={n}, components={COMPONENTS}, parameters={}, iterations={iterations}, objective={:.6}, grad_norm={:.3e}",
        model.nparams(),
        diagnostics.objective,
        diagnostics.gradient_norm,
    );
    for (component, (&weight, theta)) in
        fitted.weights().iter().zip(fitted.components()).enumerate()
    {
        let variance_eruption = theta.covariance(0, 0).expect("valid covariance index");
        let covariance = theta.covariance(1, 0).expect("valid covariance index");
        let variance_waiting = theta.covariance(1, 1).expect("valid covariance index");
        let correlation = covariance / (variance_eruption * variance_waiting).sqrt();
        println!(
            "component {component}: weight={weight:.4}, mean=[{:.4}, {:.4}], covariance=[[{variance_eruption:.4}, {covariance:.4}], [{covariance:.4}, {variance_waiting:.4}]], correlation={correlation:.4}",
            theta.mu()[0],
            theta.mu()[1],
        );
    }
    // ANCHOR_END: diagnostics

    Ok(())
}

fn empirical_component(
    rows: impl Iterator<Item = [f64; 2]>,
) -> Result<EmpiricalComponent, Box<dyn std::error::Error>> {
    let rows = rows.collect::<Vec<_>>();
    if rows.len() < 2 {
        return Err("a mixture component needs at least two initial observations".into());
    }

    let count = rows.len();
    let mean = rows.iter().fold([0.0; 2], |mut sum, row| {
        sum[0] += row[0];
        sum[1] += row[1];
        sum
    });
    let mean = [mean[0] / count as f64, mean[1] / count as f64];
    let covariance = rows.iter().fold([0.0; 3], |mut sum, row| {
        let centered = [row[0] - mean[0], row[1] - mean[1]];
        sum[0] = centered[0].mul_add(centered[0], sum[0]);
        sum[1] = centered[1].mul_add(centered[0], sum[1]);
        sum[2] = centered[1].mul_add(centered[1], sum[2]);
        sum
    });
    let denominator = (count - 1) as f64;
    let covariance = covariance.map(|value| value / denominator);
    let l00 = covariance[0].sqrt();
    let l10 = covariance[1] / l00;
    let l11 = (covariance[2] - l10 * l10).sqrt();

    Ok(EmpiricalComponent {
        count,
        mean,
        cholesky: [l00, l10, l11],
    })
}

fn initial_parameters(n: usize, starts: &[EmpiricalComponent; 2]) -> Vec<f64> {
    let mut parameters = Vec::with_capacity(11);
    parameters.push((starts[0].count as f64 / starts[1].count as f64).ln());
    for start in starts {
        parameters.extend(start.mean);
        parameters.extend([
            start.cholesky[0].ln(),
            start.cholesky[1],
            start.cholesky[2].ln(),
        ]);
    }
    debug_assert_eq!(starts.iter().map(|start| start.count).sum::<usize>(), n);
    parameters
}
