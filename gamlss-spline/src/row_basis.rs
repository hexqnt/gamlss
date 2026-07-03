use gamlss_core::{DenseDesign, ModelError};

use crate::SplineError;

/// Sparse triplet matrix arrays `(rows, columns, values)`.
pub type TripletParts = (Vec<usize>, Vec<usize>, Vec<f64>);

/// Compressed sparse row matrix arrays `(row_offsets, column_indices, values)`.
pub type CsrParts = (Vec<usize>, Vec<usize>, Vec<f64>);

/// Predictor blocks that can expose a sparse row basis without allocation.
///
/// Implementations call `f(index, weight)` once for each non-zero basis value
/// in the requested row. The coefficient order is the same order used by
/// `PredictorBlock::eta_row`.
pub trait SplineRowBasis {
    /// Number of observations.
    fn nrows(&self) -> usize;

    /// Number of coefficients consumed by this basis.
    fn nparams(&self) -> usize;

    /// Visits non-zero basis values for `row`.
    fn for_each_row_basis(&self, row: usize, f: impl FnMut(usize, f64));
}

/// Extension methods for exporting any [`SplineRowBasis`] to neutral matrix
/// layouts.
///
/// These helpers intentionally return standard Rust containers instead of a
/// concrete linear-algebra backend, so callers can adapt spline designs to
/// `ndarray`, `faer`, `nalgebra`, `sprs`, FFI buffers, or custom solvers.
pub trait SplineRowBasisExt: SplineRowBasis {
    /// Number of row-major values in the dense design.
    ///
    /// # Errors
    ///
    /// Returns [`SplineError::Model`] if `nrows * nparams` overflows `usize`.
    fn row_major_len(&self) -> Result<usize, SplineError> {
        checked_matrix_len(self.nrows(), self.nparams(), "spline row-major value count")
    }

    /// Fills an existing row-major dense buffer with the design values.
    ///
    /// # Errors
    ///
    /// Returns [`SplineError::Model`] if `out.len() != nrows * nparams`, or if
    /// the expected length overflows `usize`.
    fn fill_row_major(&self, out: &mut [f64]) -> Result<(), SplineError> {
        let expected_values = self.row_major_len()?;
        if out.len() != expected_values {
            return Err(ModelError::DesignSize {
                expected_values,
                actual_values: out.len(),
            }
            .into());
        }

        out.fill(0.0);
        let ncols = self.nparams();
        for row in 0..self.nrows() {
            let row_offset = row * ncols;
            self.for_each_row_basis(row, |col, value| {
                if value != 0.0 {
                    out[row_offset + col] = value;
                }
            });
        }
        Ok(())
    }

    /// Returns row-major dense design values.
    ///
    /// # Errors
    ///
    /// Returns [`SplineError::Model`] if `nrows * nparams` overflows `usize`.
    fn to_row_major_values(&self) -> Result<Vec<f64>, SplineError> {
        let mut values = vec![0.0; self.row_major_len()?];
        self.fill_row_major(&mut values)?;
        Ok(values)
    }

    /// Returns a [`DenseDesign`] with the same row and coefficient order.
    ///
    /// # Errors
    ///
    /// Returns [`SplineError::Model`] if the dense design shape overflows.
    fn to_dense_design(&self) -> Result<DenseDesign, SplineError> {
        Ok(DenseDesign::from_row_major(
            self.nrows(),
            self.nparams(),
            self.to_row_major_values()?,
        )?)
    }

    /// Visits sparse `(row, column, value)` entries in row-major traversal
    /// order.
    ///
    /// Zero values emitted by an implementation are filtered out.
    fn for_each_triplet(&self, mut f: impl FnMut(usize, usize, f64)) {
        for row in 0..self.nrows() {
            self.for_each_row_basis(row, |col, value| {
                if value != 0.0 {
                    f(row, col, value);
                }
            });
        }
    }

    /// Returns sparse triplet arrays `(rows, columns, values)`.
    fn to_triplets(&self) -> TripletParts {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut values = Vec::new();
        self.for_each_triplet(|row, col, value| {
            rows.push(row);
            cols.push(col);
            values.push(value);
        });
        (rows, cols, values)
    }

    /// Returns compressed sparse row parts `(row_offsets, column_indices,
    /// values)`.
    ///
    /// `row_offsets.len() == nrows + 1`; each row's entries are stored in the
    /// same order as [`SplineRowBasis::for_each_row_basis`].
    ///
    /// # Errors
    ///
    /// Returns [`SplineError::Model`] if `nrows + 1` overflows `usize`.
    fn to_csr_parts(&self) -> Result<CsrParts, SplineError> {
        let row_offsets_len =
            self.nrows()
                .checked_add(1)
                .ok_or(ModelError::ArithmeticOverflow {
                    context: "spline CSR row offset count",
                })?;
        let mut row_offsets = Vec::with_capacity(row_offsets_len);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        row_offsets.push(0);
        for row in 0..self.nrows() {
            self.for_each_row_basis(row, |col, value| {
                if value != 0.0 {
                    col_indices.push(col);
                    values.push(value);
                }
            });
            row_offsets.push(values.len());
        }

        Ok((row_offsets, col_indices, values))
    }
}

impl<T> SplineRowBasisExt for T where T: SplineRowBasis {}

#[inline]
fn checked_matrix_len(
    nrows: usize,
    ncols: usize,
    context: &'static str,
) -> Result<usize, SplineError> {
    nrows
        .checked_mul(ncols)
        .ok_or_else(|| ModelError::ArithmeticOverflow { context }.into())
}
