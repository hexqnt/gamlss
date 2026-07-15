#![allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]

use std::{hint::black_box, time::Duration};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use gamlss_core::{LinearPredictorBlock, LinearPredictorGeometry, Penalty, PredictorBlock};
use gamlss_spline::{
    BSplineBasis, DifferencePenalty, OpenUniformSplineDesign, PreparedDifferencePenalty,
    SplineOrder, SplineRowBasisExt,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Coverage {
    Full,
    HotPath,
}

fn coordinates(nobs: usize) -> Vec<f64> {
    (0..nobs)
        .map(|row| row as f64 / (nobs - 1) as f64)
        .collect()
}

fn coefficients(nparams: usize) -> Vec<f64> {
    (0..nparams)
        .map(|index| 2.0 * ((index * 17 + 5) % 101) as f64 / 100.0 - 1.0)
        .collect()
}

fn row_scores(nobs: usize) -> Vec<f64> {
    (0..nobs)
        .map(|row| 2.0 * ((row * 29 + 11) % 127) as f64 / 126.0 - 1.0)
        .collect()
}

fn row_weights(nobs: usize) -> Vec<f64> {
    (0..nobs)
        .map(|row| 0.25 + ((row * 13 + 3) % 97) as f64 / 96.0)
        .collect()
}

fn benchmark_design_case(
    criterion: &mut Criterion,
    nobs: usize,
    n_basis: usize,
    coverage: Coverage,
) {
    let x = coordinates(nobs);
    let local =
        OpenUniformSplineDesign::with_range(&x, 0.0, 1.0, n_basis, SplineOrder::Cubic).unwrap();
    let dense_design = local.to_dense_design().unwrap();
    let dense = LinearPredictorBlock::new(&dense_design);
    let beta = coefficients(n_basis);
    let scores = row_scores(nobs);
    let weights = row_weights(nobs);

    let mut group = criterion.benchmark_group(format!("open_uniform_cubic/n{nobs}/k{n_basis}"));
    group.throughput(Throughput::Elements(nobs as u64));

    group.bench_function("eta_rows_local", |bencher| {
        bencher.iter(|| {
            let beta = black_box(beta.as_slice());
            let mut checksum = 0.0;
            for row in 0..nobs {
                checksum += local.eta_row(row, beta);
            }
            black_box(checksum);
        });
    });

    group.bench_function("eta_rows_dense", |bencher| {
        bencher.iter(|| {
            let beta = black_box(beta.as_slice());
            let mut checksum = 0.0;
            for row in 0..nobs {
                checksum += dense.eta_row(row, beta);
            }
            black_box(checksum);
        });
    });

    let mut local_gradient = vec![0.0; n_basis];
    group.bench_function("vjp_local", |bencher| {
        bencher.iter(|| {
            local_gradient.fill(0.0);
            local.add_gradient(
                black_box(&scores),
                black_box(&beta),
                black_box(&mut local_gradient),
            );
            black_box(&local_gradient);
        });
    });

    let mut dense_gradient = vec![0.0; n_basis];
    group.bench_function("vjp_dense", |bencher| {
        bencher.iter(|| {
            dense_gradient.fill(0.0);
            dense.add_gradient(
                black_box(&scores),
                black_box(&beta),
                black_box(&mut dense_gradient),
            );
            black_box(&dense_gradient);
        });
    });

    if coverage == Coverage::Full {
        let mut local_gram = vec![0.0; n_basis * n_basis];
        group.bench_function("weighted_gram_local", |bencher| {
            bencher.iter(|| {
                local_gram.fill(0.0);
                local
                    .add_weighted_gram(black_box(&weights), black_box(&mut local_gram))
                    .unwrap();
                black_box(&local_gram);
            });
        });

        let mut dense_gram = vec![0.0; n_basis * n_basis];
        group.bench_function("weighted_gram_dense", |bencher| {
            bencher.iter(|| {
                dense_gram.fill(0.0);
                dense
                    .add_weighted_gram(black_box(&weights), black_box(&mut dense_gram))
                    .unwrap();
                black_box(&dense_gram);
            });
        });

        let local_basis = local.basis();
        group.bench_function("basis_rows_local_sparse", |bencher| {
            bencher.iter(|| {
                let mut checksum = 0.0;
                for value in black_box(&x) {
                    local_basis
                        .for_each_value_basis(*value, |_, weight| checksum += weight)
                        .unwrap();
                }
                black_box(checksum);
            });
        });

        let full_basis = BSplineBasis::open_uniform_from_data(&x, n_basis, 3).unwrap();
        let mut full_values = vec![0.0; n_basis];
        group.bench_function("basis_rows_full_buffer", |bencher| {
            bencher.iter(|| {
                let mut checksum = 0.0;
                for value in black_box(&x) {
                    full_basis.evaluate_into(*value, &mut full_values);
                    checksum += full_values.iter().sum::<f64>();
                }
                black_box((checksum, &full_values));
            });
        });
    }

    group.finish();
}

fn benchmark_penalty_case(criterion: &mut Criterion, n_basis: usize) {
    let beta = coefficients(n_basis);
    let unprepared = DifferencePenalty::new_unchecked(0.7, 2);
    let prepared = PreparedDifferencePenalty::new_unchecked(0.7, 2);
    let mut group = criterion.benchmark_group(format!("difference_penalty/k{n_basis}/order2"));
    group.throughput(Throughput::Elements(n_basis as u64));

    group.bench_function("value_unprepared", |bencher| {
        bencher.iter(|| black_box(unprepared.value(black_box(&beta))));
    });
    group.bench_function("value_prepared", |bencher| {
        bencher.iter(|| black_box(prepared.value(black_box(&beta))));
    });

    let mut unprepared_gradient = vec![0.0; n_basis];
    group.bench_function("gradient_unprepared", |bencher| {
        bencher.iter(|| {
            unprepared_gradient.fill(0.0);
            unprepared.add_gradient(black_box(&beta), black_box(&mut unprepared_gradient));
            black_box(&unprepared_gradient);
        });
    });

    let mut prepared_gradient = vec![0.0; n_basis];
    group.bench_function("gradient_prepared", |bencher| {
        bencher.iter(|| {
            prepared_gradient.fill(0.0);
            prepared.add_gradient(black_box(&beta), black_box(&mut prepared_gradient));
            black_box(&prepared_gradient);
        });
    });

    group.finish();
}

fn spline_hot_paths(criterion: &mut Criterion) {
    benchmark_design_case(criterion, 1_000, 16, Coverage::Full);
    benchmark_design_case(criterion, 1_000, 64, Coverage::Full);
    benchmark_design_case(criterion, 100_000, 16, Coverage::HotPath);
    benchmark_design_case(criterion, 100_000, 64, Coverage::HotPath);

    benchmark_penalty_case(criterion, 16);
    benchmark_penalty_case(criterion, 64);
    benchmark_penalty_case(criterion, 256);
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(15)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    targets = spline_hot_paths
}
criterion_main!(benches);
