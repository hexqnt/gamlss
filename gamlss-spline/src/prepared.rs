use std::ops::Range;

use gamlss_core::{ModelError, RowMultiplier};

use crate::geometry::{validate_gram_lengths, validate_transpose_lengths};
use crate::{SplineBasis1d, SplineError};

/// Prepared geometry for basis rows whose non-zero columns form one contiguous range.
///
/// `offsets` partitions the flat weight buffer while `starts[row]` identifies
/// the first coefficient. This supports arbitrary B-spline degree without one
/// allocation per row and keeps the evaluation hot path independent of a
/// matrix backend.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedContiguousGeometry {
    starts: Box<[usize]>,
    offsets: Box<[usize]>,
    weights: Box<[f64]>,
}

impl PreparedContiguousGeometry {
    /// Prepares contiguous rows emitted in increasing column order.
    pub(super) fn try_from_basis<B>(
        x: &[f64],
        basis: &B,
        max_row_width: usize,
    ) -> Result<Self, SplineError>
    where
        B: SplineBasis1d + ?Sized,
    {
        let capacity = x
            .len()
            .checked_mul(max_row_width)
            .ok_or(SplineError::ParameterOverflow)?;
        let mut starts = Vec::with_capacity(x.len());
        let mut offsets = Vec::with_capacity(x.len() + 1);
        let mut weights = Vec::with_capacity(capacity);
        offsets.push(0);

        for coordinate in x.iter().copied() {
            let row_offset = weights.len();
            let mut start = None;
            basis.for_each_basis(coordinate, |index, weight| {
                let row_start = *start.get_or_insert(index);
                debug_assert!(index >= row_start + weights.len() - row_offset);
                while row_start + (weights.len() - row_offset) < index {
                    weights.push(0.0);
                }
                weights.push(weight);
            })?;
            debug_assert!(weights.len() - row_offset <= max_row_width);
            starts.push(start.unwrap_or(0));
            offsets.push(weights.len());
        }

        Ok(Self::from_parts(starts, offsets, weights))
    }

    fn from_parts(starts: Vec<usize>, offsets: Vec<usize>, weights: Vec<f64>) -> Self {
        debug_assert_eq!(offsets.len(), starts.len() + 1);
        debug_assert_eq!(offsets.first(), Some(&0));
        debug_assert_eq!(offsets.last(), Some(&weights.len()));
        debug_assert!(offsets.windows(2).all(|window| window[0] <= window[1]));
        Self {
            starts: starts.into_boxed_slice(),
            offsets: offsets.into_boxed_slice(),
            weights: weights.into_boxed_slice(),
        }
    }

    #[inline]
    pub(super) fn nrows(&self) -> usize {
        self.starts.len()
    }

    #[inline]
    pub(super) fn row(&self, row: usize) -> (usize, &[f64]) {
        let range = self.offsets[row]..self.offsets[row + 1];
        (self.starts[row], &self.weights[range])
    }

    #[inline]
    pub(super) fn for_each(&self, row: usize, mut f: impl FnMut(usize, f64)) {
        let (start, weights) = self.row(row);
        for (offset, weight) in weights.iter().copied().enumerate() {
            if weight != 0.0 {
                f(start + offset, weight);
            }
        }
    }

    #[inline]
    pub(super) fn dot(&self, row: usize, beta: &[f64]) -> f64 {
        let (start, weights) = self.row(row);
        let coefficients = &beta[start..start + weights.len()];
        coefficients
            .iter()
            .copied()
            .zip(weights.iter().copied())
            .fold(0.0, |value, (coefficient, weight)| {
                coefficient.mul_add(weight, value)
            })
    }

    #[inline]
    pub(super) fn add_scaled(&self, row: usize, scale: f64, out: &mut [f64]) {
        let (start, weights) = self.row(row);
        let values = &mut out[start..start + weights.len()];
        for (value, weight) in values.iter_mut().zip(weights.iter().copied()) {
            *value = scale.mul_add(weight, *value);
        }
    }

    #[inline]
    pub(super) fn add_scaled_outer(&self, row: usize, scale: f64, nparams: usize, out: &mut [f64]) {
        debug_assert_eq!(out.len(), nparams * nparams);
        let (start, weights) = self.row(row);
        for (left_offset, left_weight) in weights.iter().copied().enumerate() {
            let left_index = start + left_offset;
            let scaled_left = scale * left_weight;
            for (right_offset, right_weight) in weights.iter().copied().enumerate() {
                let right_index = start + right_offset;
                let index = left_index * nparams + right_index;
                out[index] = scaled_left.mul_add(right_weight, out[index]);
            }
        }
    }

    #[inline]
    pub(super) fn add_gradient_range(
        &self,
        nparams: usize,
        rows: Range<usize>,
        scores: &[f64],
        grad: &mut [f64],
    ) {
        debug_assert!(rows.end <= self.nrows());
        debug_assert_eq!(scores.len(), rows.len());
        debug_assert_eq!(grad.len(), nparams);

        for (offset, score) in scores.iter().copied().enumerate() {
            if score != 0.0 {
                self.add_scaled(rows.start + offset, score, grad);
            }
        }
    }

    #[inline]
    pub(super) fn add_weighted_gradient_by_range<M>(
        &self,
        nparams: usize,
        rows: Range<usize>,
        scores: &[f64],
        multiplier: &M,
        grad: &mut [f64],
    ) where
        M: RowMultiplier + ?Sized,
    {
        debug_assert!(rows.end <= self.nrows());
        debug_assert_eq!(scores.len(), rows.len());
        debug_assert_eq!(grad.len(), nparams);

        for (offset, score) in scores.iter().copied().enumerate() {
            if score == 0.0 {
                continue;
            }
            let row = rows.start + offset;
            let scaled_score = score * multiplier.multiplier_at(row);
            if scaled_score != 0.0 {
                self.add_scaled(row, scaled_score, grad);
            }
        }
    }

    #[inline]
    pub(super) fn add_weighted_gram(
        &self,
        nparams: usize,
        row_weights: &[f64],
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        validate_gram_lengths(self.nrows(), nparams, row_weights, out)?;
        for (row, weight) in row_weights.iter().copied().enumerate() {
            if weight != 0.0 {
                self.add_scaled_outer(row, weight, nparams, out);
            }
        }
        Ok(())
    }

    #[inline]
    pub(super) fn add_weighted_gram_by<M>(
        &self,
        nparams: usize,
        row_weights: &[f64],
        multiplier: &M,
        out: &mut [f64],
    ) -> Result<(), ModelError>
    where
        M: RowMultiplier + ?Sized,
    {
        validate_gram_lengths(self.nrows(), nparams, row_weights, out)?;
        for (row, weight) in row_weights.iter().copied().enumerate() {
            if weight == 0.0 {
                continue;
            }
            let scaled_weight = weight * multiplier.multiplier_at(row);
            if scaled_weight != 0.0 {
                self.add_scaled_outer(row, scaled_weight, nparams, out);
            }
        }
        Ok(())
    }

    #[inline]
    pub(super) fn add_t_mul_vec(
        &self,
        nparams: usize,
        row_scores: &[f64],
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        validate_transpose_lengths(self.nrows(), nparams, row_scores, out)?;
        self.add_gradient_range(nparams, 0..self.nrows(), row_scores, out);
        Ok(())
    }

    #[inline]
    pub(super) fn add_t_mul_vec_by<M>(
        &self,
        nparams: usize,
        row_scores: &[f64],
        multiplier: &M,
        out: &mut [f64],
    ) -> Result<(), ModelError>
    where
        M: RowMultiplier + ?Sized,
    {
        validate_transpose_lengths(self.nrows(), nparams, row_scores, out)?;
        self.add_weighted_gradient_by_range(nparams, 0..self.nrows(), row_scores, multiplier, out);
        Ok(())
    }
}

/// Implements the public predictor traits for a design backed by
/// `PreparedContiguousGeometry` and fields named `basis` and `prepared`.
///
/// B- and M-spline designs intentionally retain distinct public metadata and
/// evaluation APIs, but their prepared linear-algebra paths are identical.
macro_rules! impl_prepared_contiguous_design {
    ($design:ty) => {
        impl crate::SplineRowBasis for $design {
            #[inline]
            fn nrows(&self) -> usize {
                self.prepared.nrows()
            }

            #[inline]
            fn nparams(&self) -> usize {
                self.basis.n_basis()
            }

            #[inline]
            fn for_each_row_basis(&self, row: usize, f: impl FnMut(usize, f64)) {
                self.prepared.for_each(row, f);
            }
        }

        impl gamlss_core::PredictorBlock for $design {
            #[inline]
            fn nrows(&self) -> usize {
                self.prepared.nrows()
            }

            #[inline]
            fn nparams(&self) -> usize {
                self.basis.n_basis()
            }

            #[inline]
            fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
                debug_assert!(row < self.prepared.nrows());
                debug_assert_eq!(beta.len(), self.basis.n_basis());
                self.prepared.dot(row, beta)
            }

            #[inline]
            fn zero_beta_constant_contribution(&self) -> Option<f64> {
                Some(0.0)
            }

            #[inline]
            fn add_gradient_range(
                &self,
                rows: std::ops::Range<usize>,
                scores: &[f64],
                _: &[f64],
                grad: &mut [f64],
            ) {
                self.prepared
                    .add_gradient_range(self.basis.n_basis(), rows, scores, grad);
            }

            #[inline]
            fn add_weighted_gradient_by_range<M>(
                &self,
                rows: std::ops::Range<usize>,
                scores: &[f64],
                multiplier: &M,
                _: &[f64],
                grad: &mut [f64],
            ) where
                M: gamlss_core::RowMultiplier + ?Sized,
            {
                self.prepared.add_weighted_gradient_by_range(
                    self.basis.n_basis(),
                    rows,
                    scores,
                    multiplier,
                    grad,
                );
            }
        }

        impl gamlss_core::LinearPredictorGeometry for $design {
            #[inline]
            fn add_weighted_gram(
                &self,
                row_weights: &[f64],
                out: &mut [f64],
            ) -> Result<(), gamlss_core::ModelError> {
                self.prepared
                    .add_weighted_gram(self.basis.n_basis(), row_weights, out)
            }

            #[inline]
            fn add_weighted_gram_by<M>(
                &self,
                row_weights: &[f64],
                multiplier: &M,
                out: &mut [f64],
            ) -> Result<(), gamlss_core::ModelError>
            where
                M: gamlss_core::RowMultiplier + ?Sized,
            {
                self.prepared.add_weighted_gram_by(
                    self.basis.n_basis(),
                    row_weights,
                    multiplier,
                    out,
                )
            }

            #[inline]
            fn add_t_mul_vec(
                &self,
                row_scores: &[f64],
                out: &mut [f64],
            ) -> Result<(), gamlss_core::ModelError> {
                self.prepared
                    .add_t_mul_vec(self.basis.n_basis(), row_scores, out)
            }

            #[inline]
            fn add_t_mul_vec_by<M>(
                &self,
                row_scores: &[f64],
                multiplier: &M,
                out: &mut [f64],
            ) -> Result<(), gamlss_core::ModelError>
            where
                M: gamlss_core::RowMultiplier + ?Sized,
            {
                self.prepared
                    .add_t_mul_vec_by(self.basis.n_basis(), row_scores, multiplier, out)
            }
        }
    };
}

pub(crate) use impl_prepared_contiguous_design;

/// Implements spline and predictor traits for a semantic wrapper around a
/// field named `prepared` containing `gamlss_core::DenseDesign`.
macro_rules! impl_prepared_dense_design {
    ([$($generics:tt)*] $design:ty) => {
        impl<$($generics)*> crate::SplineRowBasis for $design {
            fn nrows(&self) -> usize {
                gamlss_core::DesignMatrix::nrows(&self.prepared)
            }

            fn nparams(&self) -> usize {
                gamlss_core::DesignMatrix::ncols(&self.prepared)
            }

            fn for_each_row_basis(&self, row: usize, mut f: impl FnMut(usize, f64)) {
                let nparams = gamlss_core::DesignMatrix::ncols(&self.prepared);
                let start = row * nparams;
                for (index, value) in self.prepared.values()[start..start + nparams]
                    .iter()
                    .copied()
                    .enumerate()
                {
                    if value != 0.0 {
                        f(index, value);
                    }
                }
            }
        }

        impl<$($generics)*> gamlss_core::PredictorBlock for $design {
            fn nrows(&self) -> usize {
                gamlss_core::DesignMatrix::nrows(&self.prepared)
            }

            fn nparams(&self) -> usize {
                gamlss_core::DesignMatrix::ncols(&self.prepared)
            }

            fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
                gamlss_core::DesignMatrix::dot_row(&self.prepared, row, beta)
            }

            fn zero_beta_constant_contribution(&self) -> Option<f64> {
                Some(0.0)
            }

            fn add_gradient_range(
                &self,
                rows: std::ops::Range<usize>,
                scores: &[f64],
                _: &[f64],
                grad: &mut [f64],
            ) {
                gamlss_core::DesignMatrix::add_t_mul_vec_range(
                    &self.prepared,
                    rows,
                    scores,
                    grad,
                );
            }

            fn add_weighted_gradient_by_range<M>(
                &self,
                rows: std::ops::Range<usize>,
                scores: &[f64],
                multiplier: &M,
                _: &[f64],
                grad: &mut [f64],
            ) where
                M: gamlss_core::RowMultiplier + ?Sized,
            {
                gamlss_core::DesignMatrix::add_weighted_t_mul_vec_by_range(
                    &self.prepared,
                    rows,
                    scores,
                    multiplier,
                    grad,
                );
            }
        }

        impl<$($generics)*> gamlss_core::LinearPredictorGeometry for $design {
            fn add_weighted_gram(
                &self,
                row_weights: &[f64],
                out: &mut [f64],
            ) -> Result<(), gamlss_core::ModelError> {
                gamlss_core::LinearPredictorGeometry::add_weighted_gram(
                    &gamlss_core::LinearPredictorBlock::new(&self.prepared),
                    row_weights,
                    out,
                )
            }

            fn add_weighted_gram_by<M>(
                &self,
                row_weights: &[f64],
                multiplier: &M,
                out: &mut [f64],
            ) -> Result<(), gamlss_core::ModelError>
            where
                M: gamlss_core::RowMultiplier + ?Sized,
            {
                gamlss_core::LinearPredictorGeometry::add_weighted_gram_by(
                    &gamlss_core::LinearPredictorBlock::new(&self.prepared),
                    row_weights,
                    multiplier,
                    out,
                )
            }

            fn add_t_mul_vec(
                &self,
                row_scores: &[f64],
                out: &mut [f64],
            ) -> Result<(), gamlss_core::ModelError> {
                gamlss_core::LinearPredictorGeometry::add_t_mul_vec(
                    &gamlss_core::LinearPredictorBlock::new(&self.prepared),
                    row_scores,
                    out,
                )
            }

            fn add_t_mul_vec_by<M>(
                &self,
                row_scores: &[f64],
                multiplier: &M,
                out: &mut [f64],
            ) -> Result<(), gamlss_core::ModelError>
            where
                M: gamlss_core::RowMultiplier + ?Sized,
            {
                gamlss_core::LinearPredictorGeometry::add_t_mul_vec_by(
                    &gamlss_core::LinearPredictorBlock::new(&self.prepared),
                    row_scores,
                    multiplier,
                    out,
                )
            }
        }
    };
}

pub(crate) use impl_prepared_dense_design;
