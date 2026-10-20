//! Cholesky indexing and independently linked parameter helpers.

use gamlss_core::{Link, PositiveLink};

use super::matrix::FixedLowerTriangular;

pub use gamlss_core::shape::{
    lower_triangular_packed_index as packed_index, lower_triangular_packed_len as packed_len,
};

#[inline]
pub(super) fn from_eta_into<const D: usize, DiagonalLink, OffDiagonalLink>(
    eta: &FixedLowerTriangular<D>,
    out: &mut FixedLowerTriangular<D>,
) where
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    for row in 0..D {
        for col in 0..=row {
            let value = eta.lower(row, col);
            let value = if row == col {
                DiagonalLink::inverse(value)
            } else {
                OffDiagonalLink::inverse(value)
            };
            out.set_lower(row, col, value)
                .expect("valid lower-triangular index");
        }
    }
}

/// Pulls `log(det(L))` and a location score back through independently linked entries.
#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "preserve the rounding of the original family-specific score kernels"
)]
#[allow(
    clippy::needless_range_loop,
    reason = "keep the measured family kernel loop form and write directly into caller storage"
)]
pub(super) fn gradient_from_location_score_into<const D: usize, DiagonalLink, OffDiagonalLink>(
    eta: &FixedLowerTriangular<D>,
    standardized: &[f64; D],
    location_score: &[f64; D],
    gradient: &mut FixedLowerTriangular<D>,
) where
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    for row in 0..D {
        for (col, standardized) in standardized.iter().copied().take(row + 1).enumerate() {
            let value = eta.lower(row, col);
            let score = if row == col {
                DiagonalLink::derivative_log_inverse(value)
                    - location_score[row] * standardized * DiagonalLink::derivative_inverse(value)
            } else {
                -location_score[row] * standardized * OffDiagonalLink::derivative_inverse(value)
            };
            gradient
                .set_lower(row, col, score)
                .expect("valid lower-triangular index");
        }
    }
}
