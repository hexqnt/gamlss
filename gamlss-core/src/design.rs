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

    /// Creates a dense matrix from finite row-major values.
    ///
    /// This is the strict counterpart to [`Self::from_row_major`]. The default
    /// constructor only checks shape so callers can decide how to represent
    /// missing or masked rows; this constructor also rejects `NaN` and
    /// infinities in the design.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::DesignSize`] if `values.len() != nrows * ncols`.
    /// Returns [`ModelError::ArithmeticOverflow`] if `nrows * ncols` does not
    /// fit in `usize`. Returns [`ModelError::InvalidDesignValue`] if any matrix
    /// entry is non-finite.
    pub fn from_row_major_strict(
        nrows: usize,
        ncols: usize,
        values: Vec<f64>,
    ) -> Result<Self, ModelError> {
        let design = Self::from_row_major(nrows, ncols, values)?;
        design.validate_finite()?;
        Ok(design)
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

    /// Validates that every row-major matrix entry is finite.
    ///
    /// This is opt-in because some data ingestion paths may use non-finite
    /// sentinel values before applying their own masking or row filtering.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidDesignValue`] with the row-major index of
    /// the first non-finite entry.
    pub fn validate_finite(&self) -> Result<(), ModelError> {
        for (index, value) in self.values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(ModelError::InvalidDesignValue { index });
            }
        }
        Ok(())
    }
}

impl DesignMatrix for DenseDesign {
    #[inline]
    fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline]
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

        if self.ncols == 0 {
            return;
        }

        for (weight, row_values) in weights
            .iter()
            .copied()
            .zip(self.values.chunks_exact(self.ncols))
        {
            if weight == 0.0 {
                continue;
            }

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

        #[allow(clippy::float_cmp)]
        let has_intercept = self
            .values
            .chunks_exact(self.ncols)
            .all(|row_values| row_values[0] == 1.0);
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

        self.add_weighted_t_mul_vec_by(weights, multiplier, out);
    }

    #[inline]
    fn add_weighted_t_mul_vec_by<M>(&self, weights: &[f64], multiplier: &M, out: &mut [f64])
    where
        M: RowMultiplier + ?Sized,
    {
        debug_assert_eq!(weights.len(), self.nrows);
        debug_assert_eq!(out.len(), self.ncols);

        if self.ncols == 0 {
            return;
        }

        for (row, (weight, row_values)) in weights
            .iter()
            .copied()
            .zip(self.values.chunks_exact(self.ncols))
            .enumerate()
        {
            if weight == 0.0 {
                continue;
            }

            let scaled_weight = weight * multiplier.multiplier_at(row);
            if scaled_weight == 0.0 {
                continue;
            }

            for (out_value, x) in out.iter_mut().zip(row_values) {
                *out_value = x.mul_add(scaled_weight, *out_value);
            }
        }
    }

    #[inline]
    fn gram_weighted_by<M>(&self, weights: &[f64], multiplier: &M, out: &mut [f64])
    where
        M: RowMultiplier + ?Sized,
    {
        add_dense_weighted_gram_by(
            self.nrows,
            self.ncols,
            &self.values,
            weights,
            multiplier,
            out,
        );
    }

    fn gram_weighted(&self, weights: &[f64], out: &mut [f64]) {
        add_dense_weighted_gram_by(
            self.nrows,
            self.ncols,
            &self.values,
            weights,
            &UnitRowMultiplier,
            out,
        );
    }
}

struct UnitRowMultiplier;

impl RowMultiplier for UnitRowMultiplier {
    #[inline]
    fn multiplier_at(&self, _: usize) -> f64 {
        1.0
    }
}

/// Minimal design matrix contract for the model hot path.
///
/// Implementations must interpret `beta` as a vector of length `ncols()` and
/// `weights` as a vector of length `nrows()`. Methods are not required to
/// re-check lengths in release builds, so the calling code validates sizes
/// upfront. Weighted operations must treat an exactly zero row weight as
/// disabling that row: they should not read its row multiplier or design
/// values.
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

        self.add_weighted_t_mul_vec_by(weights, multiplier, out);
    }

    /// Adds `X^T (weights * multiplier(row))` into `out`.
    ///
    /// This variant lets nested predictor blocks provide a lazily evaluated
    /// row multiplier and avoid materializing scaled weights. Matrix
    /// implementations with direct row access should override this method.
    #[inline]
    fn add_weighted_t_mul_vec_by<M>(&self, weights: &[f64], multiplier: &M, out: &mut [f64])
    where
        M: RowMultiplier + ?Sized,
    {
        let scaled_weights = scale_active_rows(weights, multiplier);
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

            for ((row, weight), out_value) in weights.iter().copied().enumerate().zip(&mut w_xk) {
                *out_value = if weight == 0.0 {
                    0.0
                } else {
                    self.dot_row(row, &unit_beta) * weight
                };
            }

            let gram_col = &mut out[k * ncols..(k + 1) * ncols];
            self.add_t_mul_vec(&w_xk, gram_col);
        }
    }

    /// Adds `X^T diag(weights * multiplier(row)) X` into `out`.
    ///
    /// This variant lets composed predictor blocks provide lazy row scaling
    /// without constructing a row-scaled design matrix. The default
    /// implementation materializes scaled weights; matrix implementations used
    /// in Gram hot paths should override it.
    #[inline]
    fn gram_weighted_by<M>(&self, weights: &[f64], multiplier: &M, out: &mut [f64])
    where
        M: RowMultiplier + ?Sized,
    {
        let scaled_weights = scale_active_rows(weights, multiplier);
        self.gram_weighted(&scaled_weights, out);
    }
}

/// Borrows an existing design matrix without cloning its storage.
///
/// Every operation is forwarded to the underlying implementation so custom
/// optimized weighted and Gram kernels remain available through a shared
/// reference.
impl<T> DesignMatrix for &T
where
    T: DesignMatrix + ?Sized,
{
    #[inline]
    fn nrows(&self) -> usize {
        T::nrows(*self)
    }

    #[inline]
    fn ncols(&self) -> usize {
        T::ncols(*self)
    }

    #[inline]
    fn dot_row(&self, row: usize, beta: &[f64]) -> f64 {
        T::dot_row(*self, row, beta)
    }

    #[inline]
    fn add_t_mul_vec(&self, weights: &[f64], out: &mut [f64]) {
        T::add_t_mul_vec(*self, weights, out);
    }

    #[inline]
    fn set_constant_start(&self, value: f64, out: &mut [f64]) -> bool {
        T::set_constant_start(*self, value, out)
    }

    #[inline]
    fn add_weighted_t_mul_vec(&self, weights: &[f64], multiplier: &[f64], out: &mut [f64]) {
        T::add_weighted_t_mul_vec(*self, weights, multiplier, out);
    }

    #[inline]
    fn add_weighted_t_mul_vec_by<M>(&self, weights: &[f64], multiplier: &M, out: &mut [f64])
    where
        M: RowMultiplier + ?Sized,
    {
        T::add_weighted_t_mul_vec_by(*self, weights, multiplier, out);
    }

    #[inline]
    fn gram_weighted(&self, weights: &[f64], out: &mut [f64]) {
        T::gram_weighted(*self, weights, out);
    }

    #[inline]
    fn gram_weighted_by<M>(&self, weights: &[f64], multiplier: &M, out: &mut [f64])
    where
        M: RowMultiplier + ?Sized,
    {
        T::gram_weighted_by(*self, weights, multiplier, out);
    }
}

/// Row-wise multiplier used by fused weighted transpose products.
pub trait RowMultiplier {
    /// Multiplier value for `row`.
    fn multiplier_at(&self, row: usize) -> f64;
}

impl RowMultiplier for [f64] {
    #[inline]
    fn multiplier_at(&self, row: usize) -> f64 {
        self[row]
    }
}

#[inline]
pub(crate) fn scale_active_rows<M>(values: &[f64], multiplier: &M) -> Vec<f64>
where
    M: RowMultiplier + ?Sized,
{
    values
        .iter()
        .copied()
        .enumerate()
        .map(|(row, value)| {
            if value == 0.0 {
                0.0
            } else {
                value * multiplier.multiplier_at(row)
            }
        })
        .collect()
}

fn add_dense_weighted_gram_by<M>(
    nrows: usize,
    ncols: usize,
    values: &[f64],
    weights: &[f64],
    multiplier: &M,
    out: &mut [f64],
) where
    M: RowMultiplier + ?Sized,
{
    debug_assert_eq!(weights.len(), nrows);
    debug_assert_eq!(values.len(), nrows * ncols);
    debug_assert_eq!(out.len(), ncols * ncols);

    if ncols == 0 {
        return;
    }

    for (row, (weight, row_values)) in weights
        .iter()
        .copied()
        .zip(values.chunks_exact(ncols))
        .enumerate()
    {
        if weight == 0.0 {
            continue;
        }

        let scaled_weight = weight * multiplier.multiplier_at(row);
        if scaled_weight == 0.0 {
            continue;
        }

        for (j, x_j) in row_values.iter().copied().enumerate() {
            let xw_j = x_j * scaled_weight;
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

fn checked_len(nrows: usize, ncols: usize, context: &'static str) -> Result<usize, ModelError> {
    nrows
        .checked_mul(ncols)
        .ok_or(ModelError::ArithmeticOverflow { context })
}

#[cfg(test)]
mod tests {
    use super::{DenseDesign, DesignMatrix, RowMultiplier};
    use approx::assert_relative_eq;

    use crate::ModelError;

    fn design_with_masked_nan_row() -> DenseDesign {
        DenseDesign::from_row_major(3, 2, vec![1.0, 2.0, f64::NAN, f64::NAN, 3.0, 4.0]).unwrap()
    }

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
    fn borrowed_design_forwards_all_specialized_operations() {
        let owned = DenseDesign::from_rows(&[[1.0, 2.0], [3.0, 4.0]]);
        let borrowed = &owned;

        assert_eq!(DesignMatrix::nrows(&borrowed), 2);
        assert_eq!(DesignMatrix::ncols(&borrowed), 2);
        assert_relative_eq!(DesignMatrix::dot_row(&borrowed, 1, &[2.0, -1.0]), 2.0);

        let mut start = [0.0; 2];
        assert!(DesignMatrix::set_constant_start(
            &&DenseDesign::intercept(2),
            3.0,
            &mut start[..1],
        ));
        assert_relative_eq!(start[0], 3.0);

        let mut transpose = [0.0; 2];
        DesignMatrix::add_weighted_t_mul_vec(&borrowed, &[0.5, 2.0], &[2.0, 0.25], &mut transpose);
        assert_relative_eq!(transpose[0], 2.5);
        assert_relative_eq!(transpose[1], 4.0);

        let mut gram = [0.0; 4];
        DesignMatrix::gram_weighted(&borrowed, &[0.5, 2.0], &mut gram);
        for (actual, expected) in gram.iter().zip([18.5, 25.0, 25.0, 34.0]) {
            assert_relative_eq!(*actual, expected);
        }
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
    fn dense_design_skips_zero_weighted_nan_rows_in_transpose_products() {
        struct PanicOnMaskedRow;

        impl RowMultiplier for PanicOnMaskedRow {
            fn multiplier_at(&self, row: usize) -> f64 {
                assert_ne!(row, 1, "zero-weight row must not read multiplier");
                2.0
            }
        }

        // Row 1 contains NaN, but its observation weight is zero. Transpose
        // products should not read that row, so outputs remain finite.
        let design = design_with_masked_nan_row();

        let mut out = vec![0.0, 0.0];
        design.add_t_mul_vec(&[0.5, 0.0, 2.0], &mut out);
        assert_relative_eq!(out[0], 6.5);
        assert_relative_eq!(out[1], 9.0);
        assert!(out.iter().all(|value| value.is_finite()));

        let mut weighted_out = vec![1.0, 1.0];
        design.add_weighted_t_mul_vec_by(&[0.5, 0.0, 2.0], &PanicOnMaskedRow, &mut weighted_out);
        assert_relative_eq!(weighted_out[0], 14.0);
        assert_relative_eq!(weighted_out[1], 19.0);
        assert!(weighted_out.iter().all(|value| value.is_finite()));
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
    fn dense_design_strict_rejects_non_finite_values() {
        assert_eq!(
            DenseDesign::from_row_major_strict(2, 1, vec![1.0, f64::NAN]).unwrap_err(),
            ModelError::InvalidDesignValue { index: 1 }
        );

        let design = DenseDesign::from_row_major(2, 1, vec![1.0, f64::INFINITY]).unwrap();
        assert_eq!(
            design.validate_finite().unwrap_err(),
            ModelError::InvalidDesignValue { index: 1 }
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
    fn dense_design_weighted_gram_skips_zero_weighted_nan_rows() {
        let design = design_with_masked_nan_row();
        let mut gram = vec![0.0; 4];

        design.gram_weighted(&[0.5, 0.0, 2.0], &mut gram);

        assert_relative_eq!(gram[0], 18.5);
        assert_relative_eq!(gram[1], 25.0);
        assert_relative_eq!(gram[2], 25.0);
        assert_relative_eq!(gram[3], 34.0);
        assert!(gram.iter().all(|value| value.is_finite()));
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

    #[test]
    fn design_defaults_do_not_read_zero_weight_rows_or_multipliers() {
        #[derive(Debug)]
        struct MaskedDesign(DenseDesign);

        impl DesignMatrix for MaskedDesign {
            fn nrows(&self) -> usize {
                self.0.nrows()
            }

            fn ncols(&self) -> usize {
                self.0.ncols()
            }

            fn dot_row(&self, row: usize, beta: &[f64]) -> f64 {
                assert_ne!(row, 1, "zero-weight row must not read design");
                self.0.dot_row(row, beta)
            }

            fn add_t_mul_vec(&self, weights: &[f64], out: &mut [f64]) {
                self.0.add_t_mul_vec(weights, out);
            }
        }

        struct PanicOnMaskedRow;

        impl RowMultiplier for PanicOnMaskedRow {
            fn multiplier_at(&self, row: usize) -> f64 {
                assert_ne!(row, 1, "zero-weight row must not read multiplier");
                2.0
            }
        }

        let design = MaskedDesign(design_with_masked_nan_row());
        let weights = [0.5, 0.0, 2.0];

        let mut transpose = vec![0.0; 2];
        design.add_weighted_t_mul_vec_by(&weights, &PanicOnMaskedRow, &mut transpose);
        assert_relative_eq!(transpose[0], 13.0);
        assert_relative_eq!(transpose[1], 18.0);

        let mut gram = vec![0.0; 4];
        design.gram_weighted(&weights, &mut gram);
        assert_eq!(gram, [18.5, 25.0, 25.0, 34.0]);

        let mut weighted_gram = vec![0.0; 4];
        design.gram_weighted_by(&weights, &PanicOnMaskedRow, &mut weighted_gram);
        assert_eq!(weighted_gram, [37.0, 50.0, 50.0, 68.0]);
    }
}
