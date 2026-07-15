use crate::constants::HALF_LOG_2_PI;
use crate::multivariate::elliptical::{self, LowerTriangularMatrix};

pub(super) fn cholesky_score(
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

pub(super) fn valid_theta(
    dimension: usize,
    mu: &[f64],
    cholesky: &impl LowerTriangularMatrix,
) -> bool {
    elliptical::valid_location_scale(dimension, mu, cholesky)
}

pub(super) fn nll(
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

pub(super) fn nll_and_score(
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

pub(super) fn marginal_scale(
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
