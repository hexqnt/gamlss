#![allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]

use std::{array, hint::black_box, time::Duration};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use gamlss_special::{
    digamma, ln_beta, ln_gamma, ln_gamma_delta, ln_multivariate_beta, log_ndtr, normal_mills_ratio,
    owens_t, regularized_beta, regularized_beta_complement, regularized_gamma_lower,
    regularized_gamma_upper, student_t_cdf_standardized, unit_normal_cdf, unit_normal_log_sf,
    unit_normal_quantile,
};

const GRID: usize = 256;

fn fraction(index: usize, len: usize) -> f64 {
    (index as f64 + 0.5) / len as f64
}

fn linear_grid(len: usize, lower: f64, upper: f64) -> Vec<f64> {
    (0..len)
        .map(|index| (upper - lower).mul_add(fraction(index, len), lower))
        .collect()
}

fn sum_unary<F>(inputs: &[f64], function: F) -> f64
where
    F: Fn(f64) -> f64,
{
    inputs
        .iter()
        .copied()
        .fold(0.0, |sum, value| sum + function(value))
}

fn sum_binary<F>(inputs: &[(f64, f64)], function: F) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    inputs
        .iter()
        .copied()
        .fold(0.0, |sum, (left, right)| sum + function(left, right))
}

fn sum_ternary<F>(inputs: &[(f64, f64, f64)], function: F) -> f64
where
    F: Fn(f64, f64, f64) -> f64,
{
    inputs
        .iter()
        .copied()
        .fold(0.0, |sum, (first, second, third)| {
            sum + function(first, second, third)
        })
}

fn assert_finite_outputs<const N: usize>(outputs: [f64; N]) {
    assert!(
        outputs.into_iter().all(f64::is_finite),
        "benchmark inputs must exercise finite-output paths"
    );
}

fn benchmark_gamma_beta(criterion: &mut Criterion) {
    let gamma_central = linear_grid(GRID, 0.1, 50.0);
    let gamma_reflection = (0..GRID)
        .map(|index| -(index as f64 + 0.25) / 16.0)
        .collect::<Vec<_>>();
    let gamma_large = linear_grid(GRID, 1_000.0, 1_000_000.0);
    let digamma_small = (0..GRID)
        .map(|index| (index as f64 + 0.5) / 32.0)
        .collect::<Vec<_>>();
    let digamma_large = linear_grid(GRID, 8.0, 1_000_000.0);
    let gamma_delta = (0..GRID)
        .map(|index| {
            (
                10_000.0 + 990_000.0 * fraction(index, GRID),
                0.125 + (index % 17) as f64 / 8.0,
            )
        })
        .collect::<Vec<_>>();
    let beta_balanced = (0..GRID)
        .map(|index| {
            (
                0.25 + (index % 31) as f64 * 0.5,
                0.5 + (index % 37) as f64 * 0.5,
            )
        })
        .collect::<Vec<_>>();
    let beta_imbalanced = (0..GRID)
        .map(|index| {
            (
                0.1 + (index % 11) as f64 * 0.1,
                1_000.0 + (index % 29) as f64 * 1_000.0,
            )
        })
        .collect::<Vec<_>>();
    assert_finite_outputs([
        sum_unary(&gamma_central, ln_gamma),
        sum_unary(&gamma_reflection, ln_gamma),
        sum_unary(&gamma_large, ln_gamma),
        sum_unary(&digamma_small, digamma),
        sum_unary(&digamma_large, digamma),
        sum_binary(&gamma_delta, ln_gamma_delta),
        sum_binary(&beta_balanced, ln_beta),
        sum_binary(&beta_imbalanced, ln_beta),
    ]);

    {
        let mut group = criterion.benchmark_group("gamma_beta");
        group.throughput(Throughput::Elements(GRID as u64));
        group.bench_function("ln_gamma_central", |bencher| {
            bencher.iter(|| black_box(sum_unary(black_box(&gamma_central), ln_gamma)));
        });
        group.bench_function("ln_gamma_reflection", |bencher| {
            bencher.iter(|| black_box(sum_unary(black_box(&gamma_reflection), ln_gamma)));
        });
        group.bench_function("ln_gamma_large", |bencher| {
            bencher.iter(|| black_box(sum_unary(black_box(&gamma_large), ln_gamma)));
        });
        group.bench_function("digamma_small", |bencher| {
            bencher.iter(|| black_box(sum_unary(black_box(&digamma_small), digamma)));
        });
        group.bench_function("digamma_large", |bencher| {
            bencher.iter(|| black_box(sum_unary(black_box(&digamma_large), digamma)));
        });
        group.bench_function("ln_gamma_delta_large_base", |bencher| {
            bencher.iter(|| black_box(sum_binary(black_box(&gamma_delta), ln_gamma_delta)));
        });
        group.bench_function("ln_beta_balanced", |bencher| {
            bencher.iter(|| black_box(sum_binary(black_box(&beta_balanced), ln_beta)));
        });
        group.bench_function("ln_beta_imbalanced", |bencher| {
            bencher.iter(|| black_box(sum_binary(black_box(&beta_imbalanced), ln_beta)));
        });
        group.finish();
    }

    benchmark_multivariate_beta::<4>(criterion);
    benchmark_multivariate_beta::<32>(criterion);
}

fn benchmark_multivariate_beta<const D: usize>(criterion: &mut Criterion) {
    let inputs = (0..64)
        .map(|row| {
            array::from_fn(|component| 0.2 + ((row * 17 + component * 11) % 101) as f64 / 10.0)
        })
        .collect::<Vec<[f64; D]>>();
    assert_finite_outputs([inputs
        .iter()
        .fold(0.0, |sum, alpha| sum + ln_multivariate_beta(alpha))]);
    let mut group = criterion.benchmark_group(format!("multivariate_beta/d{D}"));
    group.throughput(Throughput::Elements(inputs.len() as u64));
    group.bench_function("ln_multivariate_beta", |bencher| {
        bencher.iter(|| {
            let sum = black_box(&inputs)
                .iter()
                .fold(0.0, |sum, alpha| sum + ln_multivariate_beta(alpha));
            black_box(sum);
        });
    });
    group.finish();
}

fn benchmark_regularized_functions(criterion: &mut Criterion) {
    let beta_central = (0..GRID)
        .map(|index| {
            (
                0.25 + (index % 31) as f64 * 0.5,
                0.5 + (index % 37) as f64 * 0.5,
                0.001 + 0.998 * fraction(index, GRID),
            )
        })
        .collect::<Vec<_>>();
    let beta_tail = (0..GRID)
        .map(|index| {
            let exponent = 2 + i32::try_from(index % 13).expect("remainder fits in i32");
            let epsilon = 10.0_f64.powi(-exponent);
            let x = if index.is_multiple_of(2) {
                epsilon
            } else {
                1.0 - epsilon
            };
            (
                0.2 + (index % 17) as f64 * 0.25,
                0.3 + (index % 19) as f64 * 0.25,
                x,
            )
        })
        .collect::<Vec<_>>();
    let gamma_central = (0..GRID)
        .map(|index| {
            let shape = 0.25 + (index % 47) as f64 * 0.5;
            (shape, shape * (0.1 + 1.8 * fraction(index, GRID)))
        })
        .collect::<Vec<_>>();
    let gamma_upper_tail = (0..GRID)
        .map(|index| {
            let shape = 0.25 + (index % 31) as f64 * 0.5;
            (shape, shape * (2.0 + 8.0 * fraction(index, GRID)))
        })
        .collect::<Vec<_>>();
    let gamma_saddlepoint = (0..GRID)
        .map(|index| {
            let shape = 1_000.0 + (index % 101) as f64 * 100.0;
            let offset = (index % 41) as f64 - 20.0;
            (shape, offset.mul_add(shape.sqrt(), shape).max(0.0))
        })
        .collect::<Vec<_>>();
    let student_central = (0..GRID)
        .map(|index| {
            (
                -8.0 + 16.0 * fraction(index, GRID),
                2.1 + (index % 48) as f64,
            )
        })
        .collect::<Vec<_>>();
    let student_tail = (0..GRID)
        .map(|index| {
            let magnitude = 8.0 + 32.0 * fraction(index, GRID);
            let t = if index.is_multiple_of(2) {
                -magnitude
            } else {
                magnitude
            };
            (t, 2.1 + (index % 48) as f64)
        })
        .collect::<Vec<_>>();
    assert_finite_outputs([
        sum_ternary(&beta_central, regularized_beta),
        sum_ternary(&beta_tail, regularized_beta),
        sum_ternary(&beta_tail, regularized_beta_complement),
        sum_binary(&gamma_central, regularized_gamma_lower),
        sum_binary(&gamma_upper_tail, regularized_gamma_upper),
        sum_binary(&gamma_saddlepoint, regularized_gamma_lower),
        sum_binary(&student_central, student_t_cdf_standardized),
        sum_binary(&student_tail, student_t_cdf_standardized),
    ]);

    let mut group = criterion.benchmark_group("regularized_functions");
    group.throughput(Throughput::Elements(GRID as u64));
    group.bench_function("beta_central", |bencher| {
        bencher.iter(|| {
            black_box(sum_ternary(black_box(&beta_central), regularized_beta));
        });
    });
    group.bench_function("beta_tail", |bencher| {
        bencher.iter(|| black_box(sum_ternary(black_box(&beta_tail), regularized_beta)));
    });
    group.bench_function("beta_complement_tail", |bencher| {
        bencher.iter(|| {
            black_box(sum_ternary(
                black_box(&beta_tail),
                regularized_beta_complement,
            ));
        });
    });
    group.bench_function("gamma_lower_central", |bencher| {
        bencher.iter(|| {
            black_box(sum_binary(
                black_box(&gamma_central),
                regularized_gamma_lower,
            ));
        });
    });
    group.bench_function("gamma_upper_tail", |bencher| {
        bencher.iter(|| {
            black_box(sum_binary(
                black_box(&gamma_upper_tail),
                regularized_gamma_upper,
            ));
        });
    });
    group.bench_function("gamma_saddlepoint", |bencher| {
        bencher.iter(|| {
            black_box(sum_binary(
                black_box(&gamma_saddlepoint),
                regularized_gamma_lower,
            ));
        });
    });
    group.bench_function("student_t_cdf_central", |bencher| {
        bencher.iter(|| {
            black_box(sum_binary(
                black_box(&student_central),
                student_t_cdf_standardized,
            ));
        });
    });
    group.bench_function("student_t_cdf_tail", |bencher| {
        bencher.iter(|| {
            black_box(sum_binary(
                black_box(&student_tail),
                student_t_cdf_standardized,
            ));
        });
    });
    group.finish();
}

fn benchmark_normal_functions(criterion: &mut Criterion) {
    let central = linear_grid(GRID, -8.0, 8.0);
    let left_tail = linear_grid(GRID, -40.0, -8.0);
    let right_tail = linear_grid(GRID, 8.0, 40.0);
    let central_probabilities = linear_grid(GRID, 0.001, 0.999);
    let tail_probabilities = (0..GRID)
        .map(|index| {
            let exponent = 2 + i32::try_from(index % 13).expect("remainder fits in i32");
            let epsilon = 10.0_f64.powi(-exponent);
            if index.is_multiple_of(2) {
                epsilon
            } else {
                1.0 - epsilon
            }
        })
        .collect::<Vec<_>>();
    let owen_cases = (0..64)
        .map(|index| {
            (
                -8.0 + 16.0 * fraction(index, 64),
                -10.0 + 20.0 * fraction((index * 29) % 64, 64),
            )
        })
        .collect::<Vec<_>>();
    assert_finite_outputs([
        sum_unary(&central, unit_normal_cdf),
        sum_unary(&left_tail, unit_normal_cdf),
        sum_unary(&left_tail, log_ndtr),
        sum_unary(&right_tail, unit_normal_log_sf),
        sum_unary(&left_tail, normal_mills_ratio),
        sum_unary(&central_probabilities, unit_normal_quantile),
        sum_unary(&tail_probabilities, unit_normal_quantile),
        sum_binary(&owen_cases, owens_t),
    ]);

    {
        let mut group = criterion.benchmark_group("normal_functions");
        group.throughput(Throughput::Elements(GRID as u64));
        group.bench_function("cdf_central", |bencher| {
            bencher.iter(|| black_box(sum_unary(black_box(&central), unit_normal_cdf)));
        });
        group.bench_function("cdf_left_tail", |bencher| {
            bencher.iter(|| black_box(sum_unary(black_box(&left_tail), unit_normal_cdf)));
        });
        group.bench_function("log_ndtr_left_tail", |bencher| {
            bencher.iter(|| black_box(sum_unary(black_box(&left_tail), log_ndtr)));
        });
        group.bench_function("log_sf_right_tail", |bencher| {
            bencher.iter(|| black_box(sum_unary(black_box(&right_tail), unit_normal_log_sf)));
        });
        group.bench_function("mills_ratio_left_tail", |bencher| {
            bencher.iter(|| black_box(sum_unary(black_box(&left_tail), normal_mills_ratio)));
        });
        group.bench_function("quantile_central", |bencher| {
            bencher.iter(|| {
                black_box(sum_unary(
                    black_box(&central_probabilities),
                    unit_normal_quantile,
                ));
            });
        });
        group.bench_function("quantile_tail", |bencher| {
            bencher.iter(|| {
                black_box(sum_unary(
                    black_box(&tail_probabilities),
                    unit_normal_quantile,
                ));
            });
        });
        group.finish();
    }

    let mut owen_group = criterion.benchmark_group("owens_t");
    owen_group.throughput(Throughput::Elements(owen_cases.len() as u64));
    owen_group.bench_function("mixed_regimes", |bencher| {
        bencher.iter(|| black_box(sum_binary(black_box(&owen_cases), owens_t)));
    });
    owen_group.finish();
}

fn numeric_kernels(criterion: &mut Criterion) {
    benchmark_gamma_beta(criterion);
    benchmark_regularized_functions(criterion);
    benchmark_normal_functions(criterion);
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    targets = numeric_kernels
}
criterion_main!(benches);
