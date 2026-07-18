//! Storage-neutral Cholesky geometry shared by elliptical multivariate family kernels.

use super::matrix::{FixedLowerTriangular, PackedLowerTriangular};

pub(super) trait LowerTriangularMatrix {
    fn dimension(&self) -> usize;
    fn lower(&self, row: usize, col: usize) -> f64;
}

impl<const D: usize> LowerTriangularMatrix for FixedLowerTriangular<D> {
    #[inline]
    fn dimension(&self) -> usize {
        D
    }

    #[inline]
    fn lower(&self, row: usize, col: usize) -> f64 {
        self.lower(row, col)
    }
}

impl LowerTriangularMatrix for PackedLowerTriangular {
    #[inline]
    fn dimension(&self) -> usize {
        self.dimension()
    }

    #[inline]
    fn lower(&self, row: usize, col: usize) -> f64 {
        self.lower(row, col)
    }
}

pub(super) fn valid_location_scale(
    dimension: usize,
    location: &[f64],
    cholesky: &impl LowerTriangularMatrix,
) -> bool {
    dimension > 0
        && cholesky.dimension() == dimension
        && location.len() == dimension
        && location.iter().all(|value| value.is_finite())
        && (0..dimension).all(|row| {
            (0..=row).all(|col| {
                let value = cholesky.lower(row, col);
                value.is_finite() && (row != col || value > 0.0)
            })
        })
}

/// Writes `L⁻¹(y - μ)`, returning whether all inputs and dimensions are valid.
pub(super) fn forward_standardize(
    dimension: usize,
    observation: &[f64],
    location: &[f64],
    cholesky: &impl LowerTriangularMatrix,
    standardized: &mut [f64],
) -> bool {
    if observation.len() != dimension
        || !observation.iter().all(|value| value.is_finite())
        || !valid_location_scale(dimension, location, cholesky)
        || standardized.len() != dimension
    {
        return false;
    }

    for ((standardized, observation), location) in standardized
        .iter_mut()
        .zip(observation.iter().copied())
        .zip(location.iter().copied())
    {
        *standardized = observation - location;
    }
    for row in 0..dimension {
        let mut value = standardized[row];
        for (col, standardized_col) in standardized.iter().copied().take(row).enumerate() {
            value = cholesky.lower(row, col).mul_add(-standardized_col, value);
        }
        standardized[row] = value / cholesky.lower(row, row);
    }

    true
}

/// Writes `L⁻¹(y - μ)` and returns its squared norm together with `ln(det(L))`.
pub(super) fn standardize(
    dimension: usize,
    observation: &[f64],
    location: &[f64],
    cholesky: &impl LowerTriangularMatrix,
    standardized: &mut [f64],
) -> Option<(f64, f64)> {
    if !forward_standardize(dimension, observation, location, cholesky, standardized) {
        return None;
    }

    let quadratic = standardized.iter().map(|value| value * value).sum();
    let log_det_scale = (0..dimension)
        .map(|index| cholesky.lower(index, index).ln())
        .sum();
    Some((quadratic, log_det_scale))
}

/// Solves `Lᵀ out = rhs`, returning `false` for inconsistent dimensions.
pub(super) fn transpose_solve(
    dimension: usize,
    cholesky: &impl LowerTriangularMatrix,
    rhs: &[f64],
    out: &mut [f64],
) -> bool {
    if cholesky.dimension() != dimension || rhs.len() != dimension || out.len() != dimension {
        return false;
    }
    for row in (0..dimension).rev() {
        let mut value = rhs[row];
        for (col, out_col) in out.iter().copied().enumerate().skip(row + 1) {
            value = cholesky.lower(col, row).mul_add(-out_col, value);
        }
        out[row] = value / cholesky.lower(row, row);
    }
    true
}
