use crate::constants::HALF_LOG_2_PI;

pub(super) trait LowerTriangularMatrix {
    fn dimension(&self) -> usize;
    fn lower(&self, row: usize, col: usize) -> f64;
}

pub(super) const fn cholesky_len(dimension: usize) -> Option<usize> {
    match dimension.checked_add(1) {
        Some(next) => match dimension.checked_mul(next) {
            Some(product) => Some(product / 2),
            None => None,
        },
        None => None,
    }
}

pub(super) const fn packed_index(row: usize, col: usize) -> Option<usize> {
    if col <= row {
        match row.checked_add(1) {
            Some(next) => match row.checked_mul(next) {
                Some(product) => match (product / 2).checked_add(col) {
                    Some(index) => Some(index),
                    None => None,
                },
                None => None,
            },
            None => None,
        }
    } else {
        None
    }
}

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

pub(super) fn valid_observation(dimension: usize, observation: &[f64]) -> bool {
    dimension > 0
        && observation.len() == dimension
        && observation.iter().all(|value| value.is_finite())
}

pub(super) fn valid_theta(
    dimension: usize,
    mu: &[f64],
    cholesky: &impl LowerTriangularMatrix,
) -> bool {
    dimension > 0
        && cholesky.dimension() == dimension
        && mu.len() == dimension
        && mu.iter().all(|value| value.is_finite())
        && (0..dimension).all(|row| {
            (0..=row).all(|col| {
                let value = cholesky.lower(row, col);
                value.is_finite() && (row != col || value > 0.0)
            })
        })
}

fn forward_solve_in_place(
    dimension: usize,
    cholesky: &impl LowerTriangularMatrix,
    out: &mut [f64],
) {
    for row in 0..dimension {
        let mut value = out[row];
        for col in 0..row {
            value -= cholesky.lower(row, col) * out[col];
        }
        out[row] = value / cholesky.lower(row, row);
    }
}

fn transpose_solve(
    dimension: usize,
    cholesky: &impl LowerTriangularMatrix,
    rhs: &[f64],
    out: &mut [f64],
) {
    for row in (0..dimension).rev() {
        let mut value = rhs[row];
        for col in (row + 1)..dimension {
            value -= cholesky.lower(col, row) * out[col];
        }
        out[row] = value / cholesky.lower(row, row);
    }
}

pub(super) fn nll(
    dimension: usize,
    observation: &[f64],
    mu: &[f64],
    cholesky: &impl LowerTriangularMatrix,
    z: &mut [f64],
) -> f64 {
    if !valid_observation(dimension, observation)
        || !valid_theta(dimension, mu, cholesky)
        || z.len() != dimension
    {
        return f64::INFINITY;
    }

    for ((z_value, observation), mu) in z
        .iter_mut()
        .zip(observation.iter().copied())
        .zip(mu.iter().copied())
    {
        *z_value = observation - mu;
    }
    forward_solve_in_place(dimension, cholesky, z);

    let quadratic = z.iter().map(|value| value * value).sum::<f64>();
    let log_det_scale = (0..dimension)
        .map(|index| cholesky.lower(index, index).ln())
        .sum::<f64>();

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
    if !nll.is_finite() || a.len() != dimension {
        return f64::INFINITY;
    }

    transpose_solve(dimension, cholesky, z, a);
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
