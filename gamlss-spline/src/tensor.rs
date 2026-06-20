use gamlss_core::PredictorBlock;

use crate::SplineError;
use crate::row_basis::SplineRowBasis;

/// Structured tensor-product spline predictor.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorSplineDesign<A, B> {
    left: A,
    right: B,
    nrows: usize,
    nparams: usize,
    right_nparams: usize,
}

impl<A, B> TensorSplineDesign<A, B>
where
    A: SplineRowBasis,
    B: SplineRowBasis,
{
    /// Creates a row-wise Kronecker product of two spline bases.
    pub fn new(left: A, right: B) -> Result<Self, SplineError> {
        let nrows = left.nrows();
        let right_rows = right.nrows();
        if nrows != right_rows {
            return Err(SplineError::RowMismatch {
                expected: nrows,
                actual: right_rows,
            });
        }

        let right_nparams = right.nparams();
        let nparams = left
            .nparams()
            .checked_mul(right_nparams)
            .ok_or(SplineError::ParameterOverflow)?;

        Ok(Self {
            left,
            right,
            nrows,
            nparams,
            right_nparams,
        })
    }

    /// Left basis.
    #[must_use]
    #[inline(always)]
    pub fn left(&self) -> &A {
        &self.left
    }

    /// Right basis.
    #[must_use]
    #[inline(always)]
    pub fn right(&self) -> &B {
        &self.right
    }

    /// Number of parameters in the left basis.
    #[must_use]
    #[inline(always)]
    pub fn left_nparams(&self) -> usize {
        self.nparams / self.right_nparams
    }

    /// Number of parameters in the right basis.
    #[must_use]
    #[inline(always)]
    pub fn right_nparams(&self) -> usize {
        self.right_nparams
    }
}

impl<A, B> SplineRowBasis for TensorSplineDesign<A, B>
where
    A: SplineRowBasis,
    B: SplineRowBasis,
{
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline(always)]
    fn nparams(&self) -> usize {
        self.nparams
    }

    #[inline]
    fn for_each_row_basis(&self, row: usize, mut f: impl FnMut(usize, f64)) {
        self.left
            .for_each_row_basis(row, |left_index, left_weight| {
                self.right
                    .for_each_row_basis(row, |right_index, right_weight| {
                        let index = left_index * self.right_nparams + right_index;
                        f(index, left_weight * right_weight);
                    });
            });
    }
}
impl<A, B> PredictorBlock for TensorSplineDesign<A, B>
where
    A: SplineRowBasis,
    B: SplineRowBasis,
{
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline(always)]
    fn nparams(&self) -> usize {
        self.nparams
    }

    #[inline]
    fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
        debug_assert!(row < self.nrows);
        debug_assert_eq!(beta.len(), self.nparams);

        let mut value = 0.0;
        self.for_each_row_basis(row, |index, weight| {
            value += beta[index] * weight;
        });
        value
    }

    #[inline]
    fn add_gradient(&self, scores: &[f64], _: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(scores.len(), self.nrows);
        debug_assert_eq!(grad.len(), self.nparams);

        for (row, score) in scores.iter().copied().enumerate() {
            self.for_each_row_basis(row, |index, weight| {
                grad[index] += score * weight;
            });
        }
    }

    #[inline]
    fn add_weighted_gradient(
        &self,
        scores: &[f64],
        multiplier: &[f64],
        _: &[f64],
        grad: &mut [f64],
    ) {
        debug_assert_eq!(scores.len(), self.nrows);
        debug_assert_eq!(multiplier.len(), self.nrows);
        debug_assert_eq!(grad.len(), self.nparams);

        for (row, (&score, &multiplier)) in scores.iter().zip(multiplier).enumerate() {
            self.for_each_row_basis(row, |index, weight| {
                grad[index] += score * multiplier * weight;
            });
        }
    }
}
