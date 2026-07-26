#![allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]

use std::{hint::black_box, time::Duration};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use gamlss_core::{LinearPredictorBlock, LinearPredictorGeometry, Penalty, PredictorBlock};
use gamlss_spline::{
    BSplineBasis, CyclicSplineDesign, DifferencePenalty, DifferencePenaltyKernel, DuchonSmoothness,
    DuchonSplineBasis, ISplineBasis, MSplineBasis, NaturalCubicSplineBasis,
    OpenUniformSplineDesign, PreparedDifferencePenalty, ScaledPenalty, SplineOrder, SplineRowBasis,
    SplineRowBasisExt,
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

fn phases(nobs: usize) -> Vec<f64> {
    (0..nobs)
        .map(|row| (row as f64 + 0.5) / nobs as f64)
        .collect()
}

const fn order_name(order: SplineOrder) -> &'static str {
    match order {
        SplineOrder::Linear => "linear",
        SplineOrder::Quadratic => "quadratic",
        SplineOrder::Cubic => "cubic",
    }
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
    order: SplineOrder,
    coverage: Coverage,
) {
    let x = coordinates(nobs);
    let prepared = OpenUniformSplineDesign::with_range(&x, 0.0, 1.0, n_basis, order).unwrap();
    let on_demand = prepared.basis().on_demand_design(&x).unwrap();
    let dense_design = prepared.to_dense_design().unwrap();
    let dense = LinearPredictorBlock::new(&dense_design);
    let beta = coefficients(n_basis);
    let scores = row_scores(nobs);
    let weights = row_weights(nobs);

    let mut group = criterion.benchmark_group(format!(
        "open_uniform_{}/n{nobs}/k{n_basis}",
        order_name(order)
    ));
    group.throughput(Throughput::Elements(nobs as u64));

    group.bench_function("eta_rows_prepared", |bencher| {
        bencher.iter(|| {
            let beta = black_box(beta.as_slice());
            let mut checksum = 0.0;
            for row in 0..nobs {
                checksum += prepared.eta_row(row, beta);
            }
            black_box(checksum);
        });
    });

    group.bench_function("eta_rows_on_demand", |bencher| {
        bencher.iter(|| {
            let beta = black_box(beta.as_slice());
            let mut checksum = 0.0;
            for row in 0..nobs {
                checksum += on_demand.eta_row(row, beta);
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

    let mut prepared_gradient = vec![0.0; n_basis];
    group.bench_function("vjp_prepared", |bencher| {
        bencher.iter(|| {
            prepared_gradient.fill(0.0);
            prepared.add_gradient(
                black_box(&scores),
                black_box(&beta),
                black_box(&mut prepared_gradient),
            );
            black_box(&prepared_gradient);
        });
    });

    let mut on_demand_gradient = vec![0.0; n_basis];
    group.bench_function("vjp_on_demand", |bencher| {
        bencher.iter(|| {
            on_demand_gradient.fill(0.0);
            on_demand.add_gradient(
                black_box(&scores),
                black_box(&beta),
                black_box(&mut on_demand_gradient),
            );
            black_box(&on_demand_gradient);
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
        let mut prepared_gram = vec![0.0; n_basis * n_basis];
        group.bench_function("weighted_gram_prepared", |bencher| {
            bencher.iter(|| {
                prepared_gram.fill(0.0);
                prepared
                    .add_weighted_gram(black_box(&weights), black_box(&mut prepared_gram))
                    .unwrap();
                black_box(&prepared_gram);
            });
        });

        let mut on_demand_gram = vec![0.0; n_basis * n_basis];
        group.bench_function("weighted_gram_on_demand", |bencher| {
            bencher.iter(|| {
                on_demand_gram.fill(0.0);
                on_demand
                    .add_weighted_gram(black_box(&weights), black_box(&mut on_demand_gram))
                    .unwrap();
                black_box(&on_demand_gram);
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

        let sparse_basis = prepared.basis();
        group.bench_function("basis_rows_sparse", |bencher| {
            bencher.iter(|| {
                let mut checksum = 0.0;
                for value in black_box(&x) {
                    sparse_basis
                        .for_each_value_basis(*value, |_, weight| checksum += weight)
                        .unwrap();
                }
                black_box(checksum);
            });
        });

        let full_basis = BSplineBasis::open_uniform_from_data(&x, n_basis, order.degree()).unwrap();
        group.bench_function("basis_rows_general_sparse", |bencher| {
            bencher.iter(|| {
                let mut checksum = 0.0;
                for value in black_box(&x) {
                    full_basis.for_each_basis(*value, |_, weight| checksum += weight);
                }
                black_box(checksum);
            });
        });

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

fn benchmark_prepared_basis_case<P, O>(
    criterion: &mut Criterion,
    name: &str,
    prepared: &P,
    on_demand: &O,
) where
    P: LinearPredictorGeometry + SplineRowBasis,
    O: LinearPredictorGeometry,
{
    let nobs = PredictorBlock::nrows(prepared);
    let nparams = PredictorBlock::nparams(prepared);
    assert_eq!(on_demand.nrows(), nobs);
    assert_eq!(on_demand.nparams(), nparams);
    let dense_design = prepared.to_dense_design().unwrap();
    let dense = LinearPredictorBlock::new(&dense_design);
    let beta = coefficients(nparams);
    let scores = row_scores(nobs);
    let weights = row_weights(nobs);

    let mut group = criterion.benchmark_group(name);
    group.throughput(Throughput::Elements(nobs as u64));
    group.bench_function("eta_rows_prepared", |bencher| {
        bencher.iter(|| {
            let mut checksum = 0.0;
            for row in 0..nobs {
                checksum += prepared.eta_row(row, black_box(&beta));
            }
            black_box(checksum);
        });
    });
    group.bench_function("eta_rows_on_demand", |bencher| {
        bencher.iter(|| {
            let mut checksum = 0.0;
            for row in 0..nobs {
                checksum += on_demand.eta_row(row, black_box(&beta));
            }
            black_box(checksum);
        });
    });
    group.bench_function("eta_rows_dense", |bencher| {
        bencher.iter(|| {
            let mut checksum = 0.0;
            for row in 0..nobs {
                checksum += dense.eta_row(row, black_box(&beta));
            }
            black_box(checksum);
        });
    });

    let mut prepared_gradient = vec![0.0; nparams];
    group.bench_function("vjp_prepared", |bencher| {
        bencher.iter(|| {
            prepared_gradient.fill(0.0);
            prepared.add_gradient(black_box(&scores), black_box(&beta), &mut prepared_gradient);
            black_box(&mut prepared_gradient);
        });
    });
    let mut on_demand_gradient = vec![0.0; nparams];
    group.bench_function("vjp_on_demand", |bencher| {
        bencher.iter(|| {
            on_demand_gradient.fill(0.0);
            on_demand.add_gradient(
                black_box(&scores),
                black_box(&beta),
                &mut on_demand_gradient,
            );
            black_box(&mut on_demand_gradient);
        });
    });
    let mut dense_gradient = vec![0.0; nparams];
    group.bench_function("vjp_dense", |bencher| {
        bencher.iter(|| {
            dense_gradient.fill(0.0);
            dense.add_gradient(black_box(&scores), black_box(&beta), &mut dense_gradient);
            black_box(&mut dense_gradient);
        });
    });

    let mut prepared_gram = vec![0.0; nparams * nparams];
    group.bench_function("weighted_gram_prepared", |bencher| {
        bencher.iter(|| {
            prepared_gram.fill(0.0);
            prepared
                .add_weighted_gram(black_box(&weights), &mut prepared_gram)
                .unwrap();
            black_box(&mut prepared_gram);
        });
    });
    let mut on_demand_gram = vec![0.0; nparams * nparams];
    group.bench_function("weighted_gram_on_demand", |bencher| {
        bencher.iter(|| {
            on_demand_gram.fill(0.0);
            on_demand
                .add_weighted_gram(black_box(&weights), &mut on_demand_gram)
                .unwrap();
            black_box(&mut on_demand_gram);
        });
    });
    let mut dense_gram = vec![0.0; nparams * nparams];
    group.bench_function("weighted_gram_dense", |bencher| {
        bencher.iter(|| {
            dense_gram.fill(0.0);
            dense
                .add_weighted_gram(black_box(&weights), &mut dense_gram)
                .unwrap();
            black_box(&mut dense_gram);
        });
    });
    group.finish();
}

fn benchmark_cyclic_case(
    criterion: &mut Criterion,
    nobs: usize,
    n_basis: usize,
    order: SplineOrder,
) {
    let phi = phases(nobs);
    let prepared = CyclicSplineDesign::new(&phi, n_basis, order).unwrap();
    let on_demand = prepared.spec().on_demand_design(&phi).unwrap();
    let dense_design = prepared.to_dense_design().unwrap();
    let dense = LinearPredictorBlock::new(&dense_design);
    let beta = coefficients(n_basis);
    let scores = row_scores(nobs);

    let mut group =
        criterion.benchmark_group(format!("cyclic_{}/n{nobs}/k{n_basis}", order_name(order)));
    group.throughput(Throughput::Elements(nobs as u64));

    group.bench_function("eta_rows_prepared", |bencher| {
        bencher.iter(|| {
            let beta = black_box(beta.as_slice());
            let mut checksum = 0.0;
            for row in 0..nobs {
                checksum += prepared.eta_row(row, beta);
            }
            black_box(checksum);
        });
    });

    group.bench_function("eta_rows_on_demand", |bencher| {
        bencher.iter(|| {
            let beta = black_box(beta.as_slice());
            let mut checksum = 0.0;
            for row in 0..nobs {
                checksum += on_demand.eta_row(row, beta);
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

    let mut prepared_gradient = vec![0.0; n_basis];
    group.bench_function("vjp_prepared", |bencher| {
        bencher.iter(|| {
            prepared_gradient.fill(0.0);
            prepared.add_gradient(
                black_box(&scores),
                black_box(&beta),
                black_box(&mut prepared_gradient),
            );
            black_box(&prepared_gradient);
        });
    });

    let mut on_demand_gradient = vec![0.0; n_basis];
    group.bench_function("vjp_on_demand", |bencher| {
        bencher.iter(|| {
            on_demand_gradient.fill(0.0);
            on_demand.add_gradient(
                black_box(&scores),
                black_box(&beta),
                black_box(&mut on_demand_gradient),
            );
            black_box(&on_demand_gradient);
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

    group.finish();
}

fn benchmark_penalty_case(criterion: &mut Criterion, n_basis: usize) {
    let beta = coefficients(n_basis);
    let unprepared = DifferencePenalty::new_unchecked(0.7, 2);
    let prepared = PreparedDifferencePenalty::new_unchecked(0.7, 2);
    let separated =
        ScaledPenalty::try_new(0.7, DifferencePenaltyKernel::try_new(n_basis, 2).unwrap()).unwrap();
    let mut group = criterion.benchmark_group(format!("difference_penalty/k{n_basis}/order2"));
    group.throughput(Throughput::Elements(n_basis as u64));

    group.bench_function("value_unprepared", |bencher| {
        bencher.iter(|| black_box(unprepared.value(black_box(&beta))));
    });
    group.bench_function("value_prepared", |bencher| {
        bencher.iter(|| black_box(prepared.value(black_box(&beta))));
    });
    group.bench_function("value_separated_kernel", |bencher| {
        bencher.iter(|| black_box(separated.value(black_box(&beta))));
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

    let mut separated_gradient = vec![0.0; n_basis];
    group.bench_function("gradient_separated_kernel", |bencher| {
        bencher.iter(|| {
            separated_gradient.fill(0.0);
            separated.add_gradient(black_box(&beta), black_box(&mut separated_gradient));
            black_box(&separated_gradient);
        });
    });

    group.finish();
}

fn benchmark_duchon_case(criterion: &mut Criterion) {
    let centers = (0..7)
        .flat_map(|row| {
            (0..7).map(move |column| {
                [
                    f64::from(column) / 6.0,
                    f64::from(row) / 6.0 + 0.015 * f64::from(column % 2),
                ]
            })
        })
        .collect::<Vec<_>>();
    let points = (0..1_000)
        .map(|index| {
            [
                f64::from(index * 37 % 997) / 996.0,
                f64::from(index * 61 % 991) / 990.0,
            ]
        })
        .collect::<Vec<_>>();
    let basis =
        DuchonSplineBasis::try_new(&centers, 20, DuchonSmoothness::thin_plate(2).unwrap()).unwrap();
    let design = basis.design(&points).unwrap();
    let beta = coefficients(basis.n_basis());
    let scores = row_scores(points.len());
    let penalty = basis.penalty(0.7).unwrap();
    let mut group = criterion.benchmark_group("duchon_thin_plate/d2/n1000/centers49/k20");
    group.throughput(Throughput::Elements(points.len() as u64));

    let mut basis_row = vec![0.0; basis.n_basis()];
    group.bench_function("basis_rows", |bencher| {
        bencher.iter(|| {
            let mut checksum = 0.0;
            for point in black_box(&points) {
                basis.evaluate_into(point, &mut basis_row).unwrap();
                checksum += basis_row.iter().sum::<f64>();
            }
            black_box((checksum, &basis_row));
        });
    });

    group.bench_function("eta_rows_prepared", |bencher| {
        bencher.iter(|| {
            let mut checksum = 0.0;
            for row in 0..points.len() {
                checksum += design.eta_row(row, black_box(&beta));
            }
            black_box(checksum);
        });
    });

    let mut gradient = vec![0.0; basis.n_basis()];
    group.bench_function("vjp_prepared", |bencher| {
        bencher.iter(|| {
            gradient.fill(0.0);
            design.add_gradient(black_box(&scores), black_box(&beta), &mut gradient);
            black_box(&gradient);
        });
    });

    group.bench_function("diagonal_penalty_value", |bencher| {
        bencher.iter(|| black_box(penalty.value(black_box(&beta))));
    });
    let mut penalty_gradient = vec![0.0; basis.n_basis()];
    group.bench_function("diagonal_penalty_gradient", |bencher| {
        bencher.iter(|| {
            penalty_gradient.fill(0.0);
            penalty.add_gradient(black_box(&beta), &mut penalty_gradient);
            black_box(&penalty_gradient);
        });
    });

    group.finish();
}

fn spline_hot_paths(criterion: &mut Criterion) {
    benchmark_design_case(criterion, 1_000, 16, SplineOrder::Cubic, Coverage::Full);
    benchmark_design_case(criterion, 1_000, 64, SplineOrder::Cubic, Coverage::Full);
    benchmark_design_case(
        criterion,
        100_000,
        16,
        SplineOrder::Cubic,
        Coverage::HotPath,
    );
    benchmark_design_case(
        criterion,
        100_000,
        64,
        SplineOrder::Cubic,
        Coverage::HotPath,
    );
    benchmark_design_case(
        criterion,
        100_000,
        16,
        SplineOrder::Quadratic,
        Coverage::HotPath,
    );
    benchmark_cyclic_case(criterion, 100_000, 16, SplineOrder::Cubic);

    let x = coordinates(10_000);
    let m_basis = MSplineBasis::open_uniform_from_data(&x, 64, 3).unwrap();
    let m_prepared = m_basis.design(&x).unwrap();
    let m_on_demand = m_basis.on_demand_design(&x).unwrap();
    benchmark_prepared_basis_case(
        criterion,
        "mspline_cubic/n10000/k64",
        &m_prepared,
        &m_on_demand,
    );

    let i_basis = ISplineBasis::open_uniform_from_data(&x, 16, 3).unwrap();
    let i_prepared = i_basis.design(&x).unwrap();
    let i_on_demand = i_basis.on_demand_design(&x).unwrap();
    benchmark_prepared_basis_case(
        criterion,
        "ispline_cubic/n10000/k16",
        &i_prepared,
        &i_on_demand,
    );

    let natural_basis = NaturalCubicSplineBasis::uniform_from_data(&x, 32).unwrap();
    let natural_prepared = natural_basis.design(&x).unwrap();
    let natural_on_demand = natural_basis.on_demand_design(&x).unwrap();
    benchmark_prepared_basis_case(
        criterion,
        "natural_cubic/n10000/k32",
        &natural_prepared,
        &natural_on_demand,
    );

    benchmark_penalty_case(criterion, 16);
    benchmark_penalty_case(criterion, 64);
    benchmark_penalty_case(criterion, 256);
    benchmark_duchon_case(criterion);
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
