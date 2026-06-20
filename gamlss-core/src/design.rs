use crate::ModelError;

/// Simple dense matrix in row-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseDesign {
    nrows: usize,
    ncols: usize,
    values: Vec<f64>,
}

impl DenseDesign {
    /// Creates a dense matrix from row-major values.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::DesignSize`] if `values.len() != nrows * ncols`.
    /// Returns [`ModelError::ArithmeticOverflow`] if `nrows * ncols` does not
    /// fit in `usize`.
    pub fn from_row_major(
        nrows: usize,
        ncols: usize,
        values: Vec<f64>,
    ) -> Result<Self, ModelError> {
        let expected_values = checked_len(nrows, ncols, "dense design row-major value count")?;
        let actual_values = values.len();
        if actual_values != expected_values {
            return Err(ModelError::DesignSize {
                expected_values,
                actual_values,
            });
        }

        Ok(Self {
            nrows,
            ncols,
            values,
        })
    }

    /// Creates a dense matrix from an array of fixed-width rows.
    #[must_use]
    #[inline]
    pub fn from_rows<const C: usize>(rows: &[[f64; C]]) -> Self {
        let values = rows.iter().flat_map(|row| row.iter().copied()).collect();
        Self {
            nrows: rows.len(),
            ncols: C,
            values,
        }
    }

    /// Creates a design matrix from a single intercept column.
    #[must_use]
    #[inline]
    pub fn intercept(nrows: usize) -> Self {
        Self {
            nrows,
            ncols: 1,
            values: vec![1.0; nrows],
        }
    }

    /// Creates a design matrix from a single user-specified column.
    #[must_use]
    #[inline]
    pub fn column(values: &[f64]) -> Self {
        Self {
            nrows: values.len(),
            ncols: 1,
            values: values.to_vec(),
        }
    }

    /// Creates a matrix from a set of columns, optionally prepending an
    /// intercept.
    ///
    /// All provided columns must have length `nrows`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::DesignRowMismatch`] if any column has a length
    /// different from `nrows`. Returns [`ModelError::ArithmeticOverflow`] if the
    /// number of row-major elements does not fit in `usize`.
    pub fn from_columns(
        nrows: usize,
        include_intercept: bool,
        columns: &[&[f64]],
    ) -> Result<Self, ModelError> {
        for column in columns {
            if column.len() != nrows {
                return Err(ModelError::DesignRowMismatch {
                    parameter: "column",
                    expected_rows: nrows,
                    actual_rows: column.len(),
                });
            }
        }

        let ncols = columns
            .len()
            .checked_add(usize::from(include_intercept))
            .ok_or(ModelError::ArithmeticOverflow {
                context: "dense design column count",
            })?;
        let mut values = Vec::with_capacity(checked_len(
            nrows,
            ncols,
            "dense design row-major value count",
        )?);

        for row in 0..nrows {
            if include_intercept {
                values.push(1.0);
            }
            for column in columns {
                values.push(column[row]);
            }
        }

        Self::from_row_major(nrows, ncols, values)
    }

    /// Returns the row-major values of the matrix.
    #[must_use]
    #[inline]
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

impl DesignMatrix for DenseDesign {
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline(always)]
    fn ncols(&self) -> usize {
        self.ncols
    }

    #[inline]
    fn dot_row(&self, row: usize, beta: &[f64]) -> f64 {
        debug_assert!(row < self.nrows);
        debug_assert_eq!(beta.len(), self.ncols);

        let offset = row * self.ncols;
        self.values[offset..offset + self.ncols]
            .iter()
            .zip(beta)
            .map(|(x, b)| x * b)
            .sum()
    }

    #[inline]
    fn add_t_mul_vec(&self, weights: &[f64], out: &mut [f64]) {
        debug_assert_eq!(weights.len(), self.nrows);
        debug_assert_eq!(out.len(), self.ncols);

        for (row, weight) in weights.iter().copied().enumerate() {
            let offset = row * self.ncols;
            let row_values = &self.values[offset..offset + self.ncols];
            for (out_value, x) in out.iter_mut().zip(row_values) {
                *out_value = x.mul_add(weight, *out_value);
            }
        }
    }

    #[inline]
    fn set_constant_start(&self, value: f64, out: &mut [f64]) -> bool {
        debug_assert_eq!(out.len(), self.ncols);
        if self.ncols == 0 || out.is_empty() {
            return false;
        }

        let has_intercept = (0..self.nrows).all(|row| {
            let first_value = self.values[row * self.ncols];
            first_value == 1.0
        });
        if !has_intercept {
            return false;
        }

        out[0] = value;
        true
    }

    #[inline]
    fn add_weighted_t_mul_vec(&self, weights: &[f64], multiplier: &[f64], out: &mut [f64]) {
        debug_assert_eq!(weights.len(), self.nrows);
        debug_assert_eq!(multiplier.len(), self.nrows);
        debug_assert_eq!(out.len(), self.ncols);

        for (row, (&weight, &multiplier)) in weights.iter().zip(multiplier).enumerate() {
            let scaled_weight = weight * multiplier;
            let offset = row * self.ncols;
            let row_values = &self.values[offset..offset + self.ncols];
            for (out_value, x) in out.iter_mut().zip(row_values) {
                *out_value = x.mul_add(scaled_weight, *out_value);
            }
        }
    }

    fn gram_weighted(&self, weights: &[f64], out: &mut [f64]) {
        let ncols = self.ncols;
        debug_assert_eq!(weights.len(), self.nrows);
        debug_assert_eq!(out.len(), ncols * ncols);

        for (row, weight) in weights.iter().copied().enumerate() {
            let row_offset = row * ncols;
            let row_values = &self.values[row_offset..row_offset + ncols];
            for (j, x_j) in row_values.iter().copied().enumerate() {
                let xw_j = x_j * weight;
                for (k, x_k) in row_values.iter().copied().enumerate().skip(j) {
                    let delta = x_k * xw_j;
                    out[j * ncols + k] += delta;
                    if k != j {
                        out[k * ncols + j] += delta;
                    }
                }
            }
        }
    }
}

/// Minimal design matrix contract for the model hot path.
///
/// Implementations must interpret `beta` as a vector of length `ncols()` and
/// `weights` as a vector of length `nrows()`. Methods are not required to
/// re-check lengths in release builds, so the calling code validates sizes
/// upfront.
pub trait DesignMatrix {
    /// Number of observations.
    fn nrows(&self) -> usize;
    /// Number of coefficients in the block.
    fn ncols(&self) -> usize;
    /// Dot product of row `row` with `beta`.
    fn dot_row(&self, row: usize, beta: &[f64]) -> f64;
    /// Adds `X^T weights` into `out`.
    fn add_t_mul_vec(&self, weights: &[f64], out: &mut [f64]);
    /// Writes a constant predictor start into `out` when this matrix has an
    /// intercept-like coefficient.
    ///
    /// The default is conservative and leaves `out` unchanged. Matrix
    /// implementations should return `true` only when setting a local
    /// coefficient to `value` makes the block contribution constant across
    /// rows with all other local coefficients left at zero.
    #[inline]
    fn set_constant_start(&self, _value: f64, _out: &mut [f64]) -> bool {
        false
    }
    /// Adds `X^T (weights * multiplier)` into `out`.
    ///
    /// Default implementation materializes scaled weights. Matrix
    /// implementations used in hot paths should override this method when they
    /// can fuse scaling into their transpose multiply.
    #[inline]
    fn add_weighted_t_mul_vec(&self, weights: &[f64], multiplier: &[f64], out: &mut [f64]) {
        debug_assert_eq!(weights.len(), multiplier.len());

        let scaled_weights = weights
            .iter()
            .zip(multiplier)
            .map(|(weight, multiplier)| weight * multiplier)
            .collect::<Vec<_>>();
        self.add_t_mul_vec(&scaled_weights, out);
    }

    /// Adds `X^T diag(weights) X` into `out`.
    ///
    /// `out` is a row-major matrix of size `ncols × ncols`. The result is added
    /// to the existing values in `out`, not replacing them. The matrix is
    /// symmetric; implementations may compute only the upper triangle and mirror
    /// into the lower.
    ///
    /// The default implementation builds column by column via
    /// [`Self::dot_row`] and [`Self::add_t_mul_vec`]. Implementations with
    /// direct access to values (dense, sparse) should override this method to
    /// avoid allocations and accelerate via SIMD.
    #[inline]
    fn gram_weighted(&self, weights: &[f64], out: &mut [f64]) {
        let ncols = self.ncols();
        let nrows = self.nrows();
        debug_assert_eq!(weights.len(), nrows);
        debug_assert_eq!(out.len(), ncols * ncols);

        let mut unit_beta = vec![0.0; ncols];
        let mut w_xk = vec![0.0; nrows];

        for k in 0..ncols {
            if k > 0 {
                unit_beta[k - 1] = 0.0;
            }
            unit_beta[k] = 1.0;

            for row in 0..nrows {
                w_xk[row] = self.dot_row(row, &unit_beta) * weights[row];
            }

            let gram_col = &mut out[k * ncols..(k + 1) * ncols];
            self.add_t_mul_vec(&w_xk, gram_col);
        }
    }
}

fn checked_len(nrows: usize, ncols: usize, context: &'static str) -> Result<usize, ModelError> {
    nrows
        .checked_mul(ncols)
        .ok_or(ModelError::ArithmeticOverflow { context })
}

#[cfg(test)]
mod tests {
    use super::{DenseDesign, DesignMatrix};
    use approx::assert_relative_eq;

    use crate::ModelError;

    #[test]
    fn dense_design_multiplies_rows_and_transpose() {
        let design = DenseDesign::from_rows(&[[1.0, 2.0], [3.0, 4.0]]);

        assert_relative_eq!(design.dot_row(1, &[10.0, 1.0]), 34.0);

        let mut out = vec![0.0, 0.0];
        design.add_t_mul_vec(&[0.5, 2.0], &mut out);

        assert_relative_eq!(out[0], 6.5);
        assert_relative_eq!(out[1], 9.0);
    }

    #[test]
    fn dense_design_builds_from_columns_and_weighted_transpose() {
        let first = [2.0, 3.0];
        let second = [5.0, 7.0];
        let design = DenseDesign::from_columns(2, true, &[&first, &second]).unwrap();

        assert_eq!(design.values(), &[1.0, 2.0, 5.0, 1.0, 3.0, 7.0]);
        assert_relative_eq!(design.dot_row(1, &[10.0, 1.0, 0.5]), 16.5);

        let mut out = vec![1.0, 1.0, 1.0];
        design.add_weighted_t_mul_vec(&[2.0, 3.0], &[0.5, -1.0], &mut out);

        assert_relative_eq!(out[0], -1.0);
        assert_relative_eq!(out[1], -6.0);
        assert_relative_eq!(out[2], -15.0);
    }

    #[test]
    fn dense_design_rejects_column_row_mismatch() {
        assert_eq!(
            DenseDesign::from_columns(2, false, &[&[1.0]]).unwrap_err(),
            ModelError::DesignRowMismatch {
                parameter: "column",
                expected_rows: 2,
                actual_rows: 1,
            }
        );
    }

    #[test]
    fn dense_design_rejects_overflowing_dimensions() {
        assert_eq!(
            DenseDesign::from_row_major(usize::MAX, 2, Vec::new()).unwrap_err(),
            ModelError::ArithmeticOverflow {
                context: "dense design row-major value count"
            }
        );
    }

    #[test]
    fn dense_design_weighted_gram_matches_elementwise() {
        // X = [[1, 2], [3, 4]]  weights = [0.5, 2.0]
        let design = DenseDesign::from_rows(&[[1.0, 2.0], [3.0, 4.0]]);
        let weights = vec![0.5, 2.0];
        let mut gram = vec![0.0; 4];
        design.gram_weighted(&weights, &mut gram);

        // Expected: X^T diag(w) X
        // X^T = [[1, 3], [2, 4]]
        // X^T W X = [[1*0.5*1 + 3*2.0*3, 1*0.5*2 + 3*2.0*4],
        //            [2*0.5*1 + 4*2.0*3, 2*0.5*2 + 4*2.0*4]]
        //         = [[18.5, 25.0], [25.0, 34.0]]
        assert_relative_eq!(gram[0], 18.5);
        assert_relative_eq!(gram[1], 25.0);
        assert_relative_eq!(gram[2], 25.0);
        assert_relative_eq!(gram[3], 34.0);

        // Adding into existing Gram
        let mut gram2 = vec![1.0, 2.0, 3.0, 4.0];
        design.gram_weighted(&weights, &mut gram2);
        assert_relative_eq!(gram2[0], 19.5);
        assert_relative_eq!(gram2[1], 27.0);
        assert_relative_eq!(gram2[2], 28.0);
        assert_relative_eq!(gram2[3], 38.0);
    }

    #[test]
    fn default_gram_weighted_matches_dense_override() {
        #[derive(Debug)]
        struct DefaultGramDesign(DenseDesign);

        impl DesignMatrix for DefaultGramDesign {
            fn nrows(&self) -> usize {
                self.0.nrows()
            }

            fn ncols(&self) -> usize {
                self.0.ncols()
            }

            fn dot_row(&self, row: usize, beta: &[f64]) -> f64 {
                self.0.dot_row(row, beta)
            }

            fn add_t_mul_vec(&self, weights: &[f64], out: &mut [f64]) {
                self.0.add_t_mul_vec(weights, out);
            }
        }

        let dense = DenseDesign::from_rows(&[[1.0, 2.0], [3.0, 4.0]]);
        let default = DefaultGramDesign(dense.clone());
        let weights = vec![0.5, 2.0];

        let mut gram_default = vec![0.0; 4];
        default.gram_weighted(&weights, &mut gram_default);

        let mut gram_override = vec![0.0; 4];
        dense.gram_weighted(&weights, &mut gram_override);

        assert_relative_eq!(gram_default[0], gram_override[0]);
        assert_relative_eq!(gram_default[1], gram_override[1]);
        assert_relative_eq!(gram_default[2], gram_override[2]);
        assert_relative_eq!(gram_default[3], gram_override[3]);
    }
}
