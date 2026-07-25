#![allow(clippy::needless_range_loop, clippy::suboptimal_flops)]

//! Distribution-independent partial-correlation storage and Cholesky geometry.

use gamlss_core::{ModelError, shape::strict_lower_triangular_packed_len};

use super::matrix::FixedLowerTriangular;

/// Fixed-dimensional partial correlations backed by a square array.
///
/// Construction and iteration use strict-lower row-major order `(1,0), (2,0),
/// (2,1), (3,0), ...`. Depending on context, values can be either natural-scale
/// partial correlations or their unconstrained predictors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedPartialCorrelations<const D: usize> {
    values: [[f64; D]; D],
}

impl<const D: usize> FixedPartialCorrelations<D> {
    /// Creates a carrier from strict-lower values after validating their length.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `values.len()` is not
    /// `D * (D - 1) / 2`.
    pub fn try_new(values: Vec<f64>) -> Result<Self, ModelError> {
        let expected = Self::checked_len().ok_or(ModelError::ArithmeticOverflow {
            context: "strict-lower partial-correlation storage length",
        })?;
        if values.len() != expected {
            return Err(ModelError::InvalidParameter {
                parameter: "partial_corr",
                expected: "D * (D - 1) / 2 strict-lower values",
            });
        }
        let mut out = [[0.0; D]; D];
        let mut values = values.into_iter();
        for (row, row_values) in out.iter_mut().enumerate().skip(1) {
            for value in row_values.iter_mut().take(row) {
                let Some(packed_value) = values.next() else {
                    return Err(ModelError::InvalidParameter {
                        parameter: "partial_corr",
                        expected: "D * (D - 1) / 2 strict-lower values",
                    });
                };
                *value = packed_value;
            }
        }
        Ok(Self { values: out })
    }

    /// Creates a zero-valued carrier.
    #[must_use]
    pub const fn zeros() -> Self {
        Self {
            values: [[0.0; D]; D],
        }
    }

    /// Checked number of strict-lower values for dimension `D`.
    #[must_use]
    pub const fn checked_len() -> Option<usize> {
        strict_lower_triangular_packed_len(D)
    }

    /// Returns true when there are no strict-lower entries.
    #[must_use]
    pub const fn is_empty() -> bool {
        D < 2
    }

    /// Visits values in row-major strict-lower order.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = f64> + '_ {
        self.values
            .iter()
            .enumerate()
            .skip(1)
            .flat_map(|(row, values)| values[..row].iter().copied())
    }

    /// Returns a strict-lower entry, or `None` for invalid/diagonal/upper indices.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        (row < D && col < row).then(|| self.values[row][col])
    }

    /// Returns a mutable strict-lower entry, or `None` for invalid/diagonal/upper indices.
    #[must_use]
    pub const fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut f64> {
        if row < D && col < row {
            Some(&mut self.values[row][col])
        } else {
            None
        }
    }

    #[inline]
    pub(in crate::multivariate) const fn lower(&self, row: usize, col: usize) -> f64 {
        self.values[row][col]
    }

    pub(in crate::multivariate) fn from_lower_rows(values: [[f64; D]; D]) -> Self {
        let mut out = Self::zeros();
        for row in 1..D {
            out.values[row][..row].copy_from_slice(&values[row][..row]);
        }
        out
    }

    pub(in crate::multivariate) const fn lower_rows(&self) -> [[f64; D]; D] {
        self.values
    }

    pub(in crate::multivariate) const fn filled_strict_lower(value: f64) -> Self {
        let mut values = [[0.0; D]; D];
        let mut row = 1;
        while row < D {
            let mut col = 0;
            while col < row {
                values[row][col] = value;
                col += 1;
            }
            row += 1;
        }
        Self { values }
    }
}

pub(in crate::multivariate) fn partial_corr_from_eta<const D: usize>(
    eta: &FixedPartialCorrelations<D>,
) -> FixedPartialCorrelations<D> {
    let mut out = FixedPartialCorrelations::zeros();
    for row in 1..D {
        for col in 0..row {
            *out.get_mut(row, col).expect("valid strict-lower index") =
                stable_partial_corr(eta.lower(row, col)).0;
        }
    }
    out
}

fn stable_partial_corr(eta: f64) -> (f64, f64, f64) {
    if !eta.is_finite() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let interior = f64::from_bits(1.0_f64.to_bits() - 1);
    let limit = interior.atanh();
    let transformed = eta.clamp(-limit, limit);
    let partial_corr = transformed.tanh();
    let one_minus_p2 = (1.0 - partial_corr) * (1.0 + partial_corr);
    let sech = one_minus_p2.sqrt();
    let derivative = if eta.abs() <= limit {
        one_minus_p2
    } else {
        0.0
    };
    (partial_corr, sech, derivative)
}

pub(in crate::multivariate) fn correlation_cholesky_from_partial<const D: usize>(
    partial_corr: &FixedPartialCorrelations<D>,
) -> FixedLowerTriangular<D> {
    let mut out = FixedLowerTriangular::zeros();
    for row in 0..D {
        let mut prefix = 1.0;
        for col in 0..row {
            let p = partial_corr.lower(row, col);
            out.set_lower(row, col, p * prefix)
                .expect("valid lower index");
            prefix *= ((1.0 - p) * (1.0 + p)).sqrt();
        }
        out.set_lower(row, row, prefix).expect("valid lower index");
    }
    out
}

pub(in crate::multivariate) fn scale_cholesky_from_correlation<const D: usize>(
    sigma: &[f64; D],
    correlation_cholesky: &FixedLowerTriangular<D>,
) -> FixedLowerTriangular<D> {
    let mut out = FixedLowerTriangular::zeros();
    for row in 0..D {
        for col in 0..=row {
            out.set_lower(row, col, sigma[row] * correlation_cholesky.lower(row, col))
                .expect("valid lower index");
        }
    }
    out
}

pub(in crate::multivariate) fn covariance_from_cholesky<const D: usize>(
    cholesky: &FixedLowerTriangular<D>,
    row: usize,
    col: usize,
) -> Option<f64> {
    if row >= D || col >= D {
        return None;
    }
    let limit = row.min(col);
    Some(
        (0..=limit)
            .map(|index| cholesky.lower(row, index) * cholesky.lower(col, index))
            .sum(),
    )
}

/// Pulls a score on the correlation Cholesky factor back through the ordered
/// partial-correlation construction.
pub(in crate::multivariate) fn partial_corr_gradient_from_cholesky_score<const D: usize>(
    eta: &FixedPartialCorrelations<D>,
    correlation_cholesky: &FixedLowerTriangular<D>,
    cholesky_score: &[[f64; D]; D],
) -> FixedPartialCorrelations<D> {
    let mut gradient = FixedPartialCorrelations::zeros();
    for row in 1..D {
        let mut prefixes = [1.0; D];
        let mut prefix = 1.0;
        for col in 0..row {
            prefixes[col] = prefix;
            prefix *= stable_partial_corr(eta.lower(row, col)).1;
        }
        let mut later_adjoint = cholesky_score[row][row] * correlation_cholesky.lower(row, row);
        for col in (0..row).rev() {
            let (partial, _, derivative) = stable_partial_corr(eta.lower(row, col));
            let direct = cholesky_score[row][col] * prefixes[col] * derivative;
            let log_sech_derivative = if derivative == 0.0 { 0.0 } else { -partial };
            *gradient
                .get_mut(row, col)
                .expect("valid strict-lower index") = direct + log_sech_derivative * later_adjoint;
            later_adjoint += cholesky_score[row][col] * correlation_cholesky.lower(row, col);
        }
    }
    gradient
}
