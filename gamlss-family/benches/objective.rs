#![allow(clippy::cast_precision_loss)]

use std::{array, hint::black_box, time::Duration};

use criterion::{
    BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main, measurement::WallTime,
};
use gamlss_core::{
    CholeskyScale, DenseDesign, DenseRows, DynamicParameterBlocks, Family, Gamlss, GamlssBlocks,
    HasRosenblattTransform, LinearPredictorBlock, LowerTriangularParameterBlock, MixtureWeight, Mu,
    NoPenalty, Nu, ObservationView, ParameterBlock, ParameterBlocks, PartialCorrelation,
    Probability, Sigma, SimplexLogitParameterBlock, StrictLowerTriangularParameterBlock, Tau,
    VectorParameterBlock,
};
use gamlss_family::{
    BinomialFixedTrialsProbability, DynMvNormalCholeskyDefault, Mixture, MultinomialFixedTrials,
    MvNormalCholeskyDefault, MvNormalMeanStdPartialCorrDefault,
    MvSkewStudentTFixedTauCholeskyDefault, MvStudentTCholeskyDefault, NormalMuSigma,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Coverage {
    Full,
    WorkspaceHotPath,
}

fn benchmark_model<F, Blocks, Obs>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    model: &Gamlss<F, Blocks, Obs>,
    beta: &[f64],
    coverage: Coverage,
) where
    F: Family,
    Blocks: GamlssBlocks<F>,
    for<'row> Obs: ObservationView<'row, Observation = F::Observation<'row>>,
{
    group.throughput(Throughput::Elements(model.nobs() as u64));

    let mut value_workspace = model.gradient_workspace();
    group.bench_function("value_workspace", |bencher| {
        bencher.iter(|| {
            black_box(
                model
                    .try_value_into_workspace(black_box(beta), &mut value_workspace)
                    .unwrap(),
            )
        });
    });

    let mut workspace_gradient = vec![0.0; model.nparams()];
    let mut fused_workspace = model.gradient_workspace();
    group.bench_function("value_gradient_workspace", |bencher| {
        bencher.iter(|| {
            let value = model
                .try_value_gradient_into_workspace(
                    black_box(beta),
                    black_box(&mut workspace_gradient),
                    &mut fused_workspace,
                )
                .unwrap();
            black_box((value, &workspace_gradient));
        });
    });

    if coverage == Coverage::WorkspaceHotPath {
        return;
    }

    group.bench_function("value_ordinary", |bencher| {
        bencher.iter(|| black_box(model.try_value(black_box(beta)).unwrap()));
    });

    let mut ordinary_gradient = vec![0.0; model.nparams()];
    group.bench_function("value_gradient_ordinary", |bencher| {
        bencher.iter(|| {
            let value = model
                .try_value_gradient_into(black_box(beta), black_box(&mut ordinary_gradient))
                .unwrap();
            black_box((value, &ordinary_gradient));
        });
    });

    let mut ordinary_pointwise = vec![0.0; model.nobs()];
    group.bench_function("pointwise_log_likelihood_ordinary", |bencher| {
        bencher.iter(|| {
            model
                .try_pointwise_log_likelihood_into(
                    black_box(beta),
                    black_box(&mut ordinary_pointwise),
                )
                .unwrap();
            black_box(&ordinary_pointwise);
        });
    });

    let mut workspace_pointwise = vec![0.0; model.nobs()];
    let mut pointwise_workspace = model.gradient_workspace();
    group.bench_function("pointwise_log_likelihood_workspace", |bencher| {
        bencher.iter(|| {
            model
                .try_pointwise_log_likelihood_into_workspace(
                    black_box(beta),
                    black_box(&mut workspace_pointwise),
                    &mut pointwise_workspace,
                )
                .unwrap();
            black_box(&workspace_pointwise);
        });
    });

    let mut theta = model.predict_theta(beta).unwrap();
    group.bench_function("predict_theta_into", |bencher| {
        bencher.iter(|| {
            model
                .predict_theta_into(black_box(beta), black_box(&mut theta))
                .unwrap();
            black_box(&theta);
        });
    });

    group.bench_function("predict_eta", |bencher| {
        bencher.iter(|| black_box(model.predict_eta(black_box(beta)).unwrap()));
    });
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

fn response_value(row: usize, coordinate: usize) -> f64 {
    let trend = 2.0 * (row % 101) as f64 / 100.0 - 1.0;
    let ripple = 2.0 * ((row * 17 + coordinate * 7) % 29) as f64 / 28.0 - 1.0;
    let base = 0.15 * (coordinate + 1) as f64;
    let slope = 0.025_f64.mul_add(coordinate as f64, 0.3);
    0.08_f64.mul_add(ripple, slope.mul_add(trend, base))
}

fn scalar_observations(nobs: usize) -> Vec<f64> {
    (0..nobs).map(|row| response_value(row, 0)).collect()
}

fn fixed_observations<const D: usize>(nobs: usize) -> Vec<[f64; D]> {
    (0..nobs)
        .map(|row| array::from_fn(|coordinate| response_value(row, coordinate)))
        .collect()
}

fn flat_observations(nobs: usize, dimension: usize) -> Vec<f64> {
    (0..nobs)
        .flat_map(|row| (0..dimension).map(move |coordinate| response_value(row, coordinate)))
        .collect()
}

fn benchmark_scalar(criterion: &mut Criterion, nobs: usize, ncols: usize, coverage: Coverage) {
    let observations = scalar_observations(nobs);
    let mu_design = dense_design(nobs, ncols);
    let sigma_design = DenseDesign::intercept(nobs);
    let blocks = ParameterBlocks::new((
        ParameterBlock::<Mu, _, _>::linear(&mu_design, NoPenalty, 0),
        ParameterBlock::<Sigma, _, _>::linear(&sigma_design, NoPenalty, 0),
    ));
    let model = Gamlss::try_new(NormalMuSigma::new(), blocks, observations.as_slice()).unwrap();
    let mut beta = vec![0.0; model.nparams()];
    beta[0] = 0.15;
    if ncols > 1 {
        beta[1] = 0.3;
    }
    beta[ncols] = -0.1;

    let mut group = criterion.benchmark_group(format!("scalar_normal/n{nobs}/dense_p{ncols}"));
    benchmark_model(&mut group, &model, &beta, coverage);
    group.finish();
}

fn benchmark_fixed_binomial(criterion: &mut Criterion, nobs: usize, trials: u32) {
    assert!(trials > 1, "benchmark needs interior binomial counts");
    let interior_count = usize::try_from(trials - 1).expect("trial count fits usize");
    let observations = (0..nobs)
        .map(|row| 1.0 + (row % interior_count) as f64)
        .collect::<Vec<_>>();
    let probability_design = DenseDesign::intercept(nobs);
    let blocks = ParameterBlocks::new((ParameterBlock::<Probability, _, _>::linear(
        &probability_design,
        NoPenalty,
        0,
    ),));
    let model = Gamlss::try_new(
        BinomialFixedTrialsProbability::try_new(trials).unwrap(),
        blocks,
        observations.as_slice(),
    )
    .unwrap();
    let beta = [0.3];

    let mut group =
        criterion.benchmark_group(format!("binomial_fixed_trials/n{nobs}/trials{trials}"));
    benchmark_model(&mut group, &model, &beta, Coverage::WorkspaceHotPath);
    group.finish();
}

fn benchmark_fixed_multinomial(criterion: &mut Criterion, nobs: usize) {
    let observations = (0..nobs)
        .map(|row| {
            let first = 1 + row % 4;
            let second = 2 + (3 * row) % 5;
            let third = 3 + (5 * row) % 6;
            [
                first as f64,
                second as f64,
                third as f64,
                (20 - first - second - third) as f64,
            ]
        })
        .collect::<Vec<_>>();
    let shared_design = DenseDesign::intercept(nobs);
    let probabilities = SimplexLogitParameterBlock::<Probability, 4, _, _>::new(
        (0..3)
            .map(|_| LinearPredictorBlock::new(&shared_design))
            .collect(),
        NoPenalty,
        0,
    );
    let model = Gamlss::try_new_with_observations(
        MultinomialFixedTrials::<4>::new(20),
        ParameterBlocks::new(probabilities),
        observations.as_slice(),
    )
    .unwrap();
    let beta = [0.3, -0.2, 0.1];

    let mut group =
        criterion.benchmark_group(format!("multinomial_fixed_trials/k4/n{nobs}/trials20"));
    benchmark_model(&mut group, &model, &beta, Coverage::WorkspaceHotPath);
    group.finish();
}

fn benchmark_fixed_mvn<const D: usize>(
    criterion: &mut Criterion,
    nobs: usize,
    ncols: usize,
    coverage: Coverage,
) {
    let observations = fixed_observations::<D>(nobs);
    let shared_design = dense_design(nobs, ncols);
    let mu = VectorParameterBlock::<Mu, D, _, _>::new(
        array::from_fn(|_| LinearPredictorBlock::new(&shared_design)),
        NoPenalty,
        0,
    );
    let cholesky = LowerTriangularParameterBlock::<CholeskyScale, D, _, _>::new(
        (0..D * (D + 1) / 2)
            .map(|_| LinearPredictorBlock::new(&shared_design))
            .collect(),
        NoPenalty,
        0,
    );
    let model = Gamlss::try_new_with_observations(
        MvNormalCholeskyDefault::<D>::new(),
        ParameterBlocks::new((mu, cholesky)),
        observations.as_slice(),
    )
    .unwrap();
    let beta = vec![0.0; model.nparams()];

    let mut group = criterion.benchmark_group(format!(
        "mvn_cholesky_static/d{D}/n{nobs}/shared_dense_p{ncols}"
    ));
    benchmark_model(&mut group, &model, &beta, coverage);
    let theta = model.predict_theta(&beta).unwrap();
    let mut rosenblatt = [0.0; D];
    group.bench_function("rosenblatt", |bencher| {
        bencher.iter(|| {
            for (observation, theta) in observations.iter().zip(&theta) {
                model
                    .family()
                    .rosenblatt_into(
                        black_box(*observation),
                        black_box(theta),
                        black_box(&mut rosenblatt),
                    )
                    .unwrap();
            }
            black_box(&rosenblatt);
        });
    });
    group.finish();
}

fn benchmark_dynamic_mvn(
    criterion: &mut Criterion,
    dimension: usize,
    nobs: usize,
    ncols: usize,
    coverage: Coverage,
) {
    let flat = flat_observations(nobs, dimension);
    let observations = DenseRows::try_new(&flat, dimension).unwrap();
    let shared_design = dense_design(nobs, ncols);
    let family = DynMvNormalCholeskyDefault::new(dimension).unwrap();
    let coordinate_count = dimension + dimension * (dimension + 1) / 2;
    let predictors = (0..coordinate_count)
        .map(|_| (LinearPredictorBlock::new(&shared_design), NoPenalty))
        .collect();
    let blocks = DynamicParameterBlocks::try_new(&family, predictors).unwrap();
    let model = Gamlss::try_new_with_observations(family, blocks, observations).unwrap();
    let beta = vec![0.0; model.nparams()];

    let mut group = criterion.benchmark_group(format!(
        "mvn_cholesky_dynamic/d{dimension}/n{nobs}/shared_dense_p{ncols}"
    ));
    benchmark_model(&mut group, &model, &beta, coverage);
    let theta = model.predict_theta(&beta).unwrap();
    let mut rosenblatt = vec![0.0; dimension];
    group.bench_function("rosenblatt", |bencher| {
        bencher.iter(|| {
            for (observation, theta) in flat.chunks_exact(dimension).zip(&theta) {
                model
                    .family()
                    .rosenblatt_into(
                        black_box(observation),
                        black_box(theta),
                        black_box(&mut rosenblatt),
                    )
                    .unwrap();
            }
            black_box(&rosenblatt);
        });
    });
    group.finish();
}

fn benchmark_mean_std_partial<const D: usize>(criterion: &mut Criterion, nobs: usize) {
    let observations = fixed_observations::<D>(nobs);
    let shared_design = DenseDesign::intercept(nobs);
    let mu = VectorParameterBlock::<Mu, D, _, _>::new(
        array::from_fn(|_| LinearPredictorBlock::new(&shared_design)),
        NoPenalty,
        0,
    );
    let sigma = VectorParameterBlock::<Sigma, D, _, _>::new(
        array::from_fn(|_| LinearPredictorBlock::new(&shared_design)),
        NoPenalty,
        0,
    );
    let partial = StrictLowerTriangularParameterBlock::<PartialCorrelation, D, _, _>::new(
        (0..D * (D - 1) / 2)
            .map(|_| LinearPredictorBlock::new(&shared_design))
            .collect(),
        NoPenalty,
        0,
    );
    let model = Gamlss::try_new_with_observations(
        MvNormalMeanStdPartialCorrDefault::<D>::new(),
        ParameterBlocks::new((mu, sigma, partial)),
        observations.as_slice(),
    )
    .unwrap();
    let beta = vec![0.0; model.nparams()];

    let mut group = criterion.benchmark_group(format!(
        "mvn_mean_std_partial/d{D}/n{nobs}/shared_intercept"
    ));
    benchmark_model(&mut group, &model, &beta, Coverage::Full);
    group.finish();
}

fn benchmark_student_t<const D: usize>(criterion: &mut Criterion, nobs: usize) {
    let observations = fixed_observations::<D>(nobs);
    let shared_design = DenseDesign::intercept(nobs);
    let mu = VectorParameterBlock::<Mu, D, _, _>::new(
        array::from_fn(|_| LinearPredictorBlock::new(&shared_design)),
        NoPenalty,
        0,
    );
    let cholesky = LowerTriangularParameterBlock::<CholeskyScale, D, _, _>::new(
        (0..D * (D + 1) / 2)
            .map(|_| LinearPredictorBlock::new(&shared_design))
            .collect(),
        NoPenalty,
        0,
    );
    let tau = ParameterBlock::<Tau, _, _>::linear(&shared_design, NoPenalty, 0);
    let model = Gamlss::try_new_with_observations(
        MvStudentTCholeskyDefault::<D>::new(),
        ParameterBlocks::new((mu, cholesky, tau)),
        observations.as_slice(),
    )
    .unwrap();
    let beta = vec![0.0; model.nparams()];

    let mut group = criterion.benchmark_group(format!(
        "mv_student_t_cholesky/d{D}/n{nobs}/shared_intercept"
    ));
    benchmark_model(&mut group, &model, &beta, Coverage::Full);
    group.finish();
}

fn benchmark_fixed_skew_student_t<const D: usize>(criterion: &mut Criterion, nobs: usize) {
    let observations = fixed_observations::<D>(nobs);
    let shared_design = DenseDesign::intercept(nobs);
    let mu = VectorParameterBlock::<Mu, D, _, _>::new(
        array::from_fn(|_| LinearPredictorBlock::new(&shared_design)),
        NoPenalty,
        0,
    );
    let cholesky = LowerTriangularParameterBlock::<CholeskyScale, D, _, _>::new(
        (0..D * (D + 1) / 2)
            .map(|_| LinearPredictorBlock::new(&shared_design))
            .collect(),
        NoPenalty,
        0,
    );
    let shape = VectorParameterBlock::<Nu, D, _, _>::new(
        array::from_fn(|_| LinearPredictorBlock::new(&shared_design)),
        NoPenalty,
        0,
    );
    let model = Gamlss::try_new_with_observations(
        MvSkewStudentTFixedTauCholeskyDefault::<D>::new(5.0),
        ParameterBlocks::new((mu, cholesky, shape)),
        observations.as_slice(),
    )
    .unwrap();
    let mut beta = vec![0.0; model.nparams()];
    let shape_start = D + D * (D + 1) / 2;
    beta[shape_start..].fill(0.5);

    let mut group = criterion.benchmark_group(format!(
        "mv_skew_student_t_fixed_tau/d{D}/n{nobs}/shared_intercept"
    ));
    benchmark_model(&mut group, &model, &beta, Coverage::WorkspaceHotPath);
    group.finish();
}

fn benchmark_mixture<const C: usize>(criterion: &mut Criterion, nobs: usize) {
    let observations = scalar_observations(nobs);
    let shared_design = DenseDesign::intercept(nobs);
    let weights = SimplexLogitParameterBlock::<MixtureWeight, C, _, _>::new(
        (0..C - 1)
            .map(|_| LinearPredictorBlock::new(&shared_design))
            .collect(),
        NoPenalty,
        0,
    );
    let components = array::from_fn::<_, C, _>(|_| {
        (
            ParameterBlock::<Mu, _, _>::linear(&shared_design, NoPenalty, 0),
            ParameterBlock::<Sigma, _, _>::linear(&shared_design, NoPenalty, 0),
        )
    });
    let family = Mixture::<_, C>::try_new(NormalMuSigma::new()).unwrap();
    let model = Gamlss::try_new(
        family,
        ParameterBlocks::new((weights, components)),
        observations.as_slice(),
    )
    .unwrap();
    let mut beta = vec![0.0; model.nparams()];
    for component in 0..C {
        beta[C - 1 + 2 * component] = component as f64 - (C - 1) as f64 / 2.0;
        beta[C + 2 * component] = -0.1;
    }

    let mut group =
        criterion.benchmark_group(format!("normal_mixture/c{C}/n{nobs}/shared_intercept"));
    benchmark_model(&mut group, &model, &beta, Coverage::Full);
    group.finish();
}

fn objective_baseline(criterion: &mut Criterion) {
    benchmark_scalar(criterion, 1_000, 8, Coverage::Full);
    benchmark_scalar(criterion, 1_000, 64, Coverage::WorkspaceHotPath);
    benchmark_scalar(criterion, 100_000, 8, Coverage::WorkspaceHotPath);
    benchmark_fixed_binomial(criterion, 1_000, 20);
    benchmark_fixed_multinomial(criterion, 1_000);

    benchmark_fixed_mvn::<2>(criterion, 1_000, 1, Coverage::Full);
    benchmark_fixed_mvn::<8>(criterion, 1_000, 1, Coverage::Full);
    benchmark_dynamic_mvn(criterion, 2, 1_000, 1, Coverage::Full);
    benchmark_dynamic_mvn(criterion, 8, 1_000, 1, Coverage::Full);

    benchmark_fixed_mvn::<8>(criterion, 1_000, 8, Coverage::WorkspaceHotPath);
    benchmark_dynamic_mvn(criterion, 8, 1_000, 8, Coverage::WorkspaceHotPath);
    benchmark_fixed_mvn::<16>(criterion, 1_000, 1, Coverage::WorkspaceHotPath);
    benchmark_fixed_mvn::<32>(criterion, 1_000, 1, Coverage::WorkspaceHotPath);
    benchmark_dynamic_mvn(criterion, 16, 1_000, 1, Coverage::WorkspaceHotPath);
    benchmark_dynamic_mvn(criterion, 32, 1_000, 1, Coverage::WorkspaceHotPath);
    benchmark_dynamic_mvn(criterion, 32, 1_000, 32, Coverage::WorkspaceHotPath);
    benchmark_dynamic_mvn(criterion, 50, 1_000, 1, Coverage::WorkspaceHotPath);
    benchmark_fixed_mvn::<2>(criterion, 100_000, 1, Coverage::WorkspaceHotPath);
    benchmark_dynamic_mvn(criterion, 2, 100_000, 1, Coverage::WorkspaceHotPath);

    benchmark_mean_std_partial::<4>(criterion, 1_000);
    benchmark_student_t::<4>(criterion, 1_000);
    benchmark_fixed_skew_student_t::<4>(criterion, 1_000);
    benchmark_mixture::<2>(criterion, 1_000);
    benchmark_mixture::<4>(criterion, 1_000);
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    targets = objective_baseline
}
criterion_main!(benches);
