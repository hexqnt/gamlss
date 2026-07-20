use crate::constants::HALF_LOG_2_PI;
use crate::multivariate::elliptical::{self, LowerTriangularMatrix};
use gamlss_special::unit_normal_cdf;

pub(in crate::multivariate) fn cholesky_score(
    row: usize,
    col: usize,
    z_col: f64,
    a: &[f64],
    cholesky: &impl LowerTriangularMatrix,
) -> f64 {
    let mut score = -a[row] * z_col;
    if row == col {
        score += 1.0 / cholesky.lower(row, row);
    }
    score
}

pub(in crate::multivariate) fn valid_theta(
    dimension: usize,
    mu: &[f64],
    cholesky: &impl LowerTriangularMatrix,
) -> bool {
    elliptical::valid_location_scale(dimension, mu, cholesky)
}

pub(in crate::multivariate) fn nll(
    dimension: usize,
    observation: &[f64],
    mu: &[f64],
    cholesky: &impl LowerTriangularMatrix,
    z: &mut [f64],
) -> f64 {
    let Some((quadratic, log_det_scale)) =
        elliptical::standardize(dimension, observation, mu, cholesky, z)
    else {
        return f64::INFINITY;
    };

    dimension as f64 * HALF_LOG_2_PI + log_det_scale + 0.5 * quadratic
}

pub(in crate::multivariate) fn nll_and_score(
    dimension: usize,
    observation: &[f64],
    mu: &[f64],
    cholesky: &impl LowerTriangularMatrix,
    z: &mut [f64],
    a: &mut [f64],
) -> f64 {
    let nll = nll(dimension, observation, mu, cholesky, z);
    if !nll.is_finite() || !elliptical::transpose_solve(dimension, cholesky, z, a) {
        return f64::INFINITY;
    }
    nll
}

pub(in crate::multivariate) fn marginal_scale(
    dimension: usize,
    component: usize,
    mu: &[f64],
    cholesky: &impl LowerTriangularMatrix,
) -> f64 {
    if component >= dimension || !valid_theta(dimension, mu, cholesky) {
        return f64::NAN;
    }

    (0..=component)
        .map(|col| {
            let value = cholesky.lower(component, col);
            value * value
        })
        .sum::<f64>()
        .sqrt()
}

pub(in crate::multivariate) fn conditional_cdf(
    dimension: usize,
    component: usize,
    y: f64,
    preceding: &[f64],
    mu: &[f64],
    cholesky: &impl LowerTriangularMatrix,
    standardized: &mut [f64],
) -> f64 {
    if component >= dimension
        || preceding.len() < component
        || standardized.len() < component
        || !y.is_finite()
        || !valid_theta(dimension, mu, cholesky)
    {
        return f64::NAN;
    }
    for row in 0..component {
        let mut residual = preceding[row] - mu[row];
        for (col, standardized_col) in standardized.iter().copied().take(row).enumerate() {
            residual = cholesky
                .lower(row, col)
                .mul_add(-standardized_col, residual);
        }
        standardized[row] = residual / cholesky.lower(row, row);
    }
    let conditional_mean = standardized
        .iter()
        .copied()
        .take(component)
        .enumerate()
        .fold(mu[component], |mean, (col, standardized)| {
            cholesky.lower(component, col).mul_add(standardized, mean)
        });
    unit_normal_cdf((y - conditional_mean) / cholesky.lower(component, component))
}

pub(in crate::multivariate) fn rosenblatt_into(
    dimension: usize,
    observation: &[f64],
    mu: &[f64],
    cholesky: &impl LowerTriangularMatrix,
    out: &mut [f64],
) {
    if !elliptical::forward_standardize(dimension, observation, mu, cholesky, out) {
        out.fill(f64::NAN);
        return;
    }
    for value in out {
        *value = unit_normal_cdf(*value);
    }
}
