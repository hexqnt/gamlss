use std::marker::PhantomData;

use crate::{DesignMatrix, Link, ModelError, Softplus};

/// Predictor block for one distribution parameter.
///
/// Implementations map a local coefficient slice to a scalar linear predictor
/// contribution for each observation and know how to propagate per-observation
/// scores back to that local coefficient slice.
///
/// The model validates row counts before evaluation. In release builds,
/// implementations may assume `row < nrows()`, `beta.len() == nparams()`,
/// `scores.len() == nrows()` and `grad.len() == nparams()`. `add_gradient`
/// must add into the existing `grad` buffer rather than clearing it.
pub trait PredictorBlock {
    /// Number of observations.
    fn nrows(&self) -> usize;
    /// Number of local coefficients consumed by this block.
    fn nparams(&self) -> usize;
    /// Predictor contribution for one row.
    fn eta_row(&self, row: usize, beta: &[f64]) -> f64;
    /// Adds the gradient contribution implied by `scores` into `grad`.
    fn add_gradient(&self, scores: &[f64], beta: &[f64], grad: &mut [f64]);
    /// Adds the gradient contribution implied by `scores * multiplier` into `grad`.
    ///
    /// Default implementation materializes scaled scores and delegates to
    /// [`Self::add_gradient`]. Blocks used in nested hot paths should override
    /// this method when they can fuse the multiplier into their gradient pass.
    #[inline]
    fn add_weighted_gradient(
        &self,
        scores: &[f64],
        multiplier: &[f64],
        beta: &[f64],
        grad: &mut [f64],
    ) {
        debug_assert_eq!(scores.len(), multiplier.len());

        let scaled_scores = scores
            .iter()
            .zip(multiplier)
            .map(|(score, multiplier)| score * multiplier)
            .collect::<Vec<_>>();
        self.add_gradient(&scaled_scores, beta, grad);
    }

    /// Validates internal block consistency.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] when internal dimensions or invariants do not
    /// match the block contract.
    #[inline]
    fn validate(&self) -> Result<(), ModelError> {
        Ok(())
    }
}

/// Linear predictor block backed by a [`DesignMatrix`].
///
/// This is the explicit adapter from matrix-based predictors to the more
/// general [`PredictorBlock`] extension point.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LinearPredictorBlock<X> {
    /// Design matrix used by this predictor.
    pub x: X,
}

impl<X> LinearPredictorBlock<X> {
    /// Wraps a design matrix as a predictor block.
    #[must_use]
    #[inline]
    pub const fn new(x: X) -> Self {
        Self { x }
    }

    /// Returns the wrapped design matrix.
    #[must_use]
    #[inline]
    pub fn into_inner(self) -> X {
        self.x
    }
}

impl<X> PredictorBlock for LinearPredictorBlock<X>
where
    X: DesignMatrix,
{
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.x.nrows()
    }

    #[inline(always)]
    fn nparams(&self) -> usize {
        self.x.ncols()
    }

    #[inline(always)]
    fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
        self.x.dot_row(row, beta)
    }

    #[inline]
    fn add_gradient(&self, scores: &[f64], _: &[f64], grad: &mut [f64]) {
        self.x.add_t_mul_vec(scores, grad);
    }

    #[inline]
    fn add_weighted_gradient(
        &self,
        scores: &[f64],
        multiplier: &[f64],
        _: &[f64],
        grad: &mut [f64],
    ) {
        self.x.add_weighted_t_mul_vec(scores, multiplier, grad);
    }
}

/// Predictor blocks that expose an underlying [`DesignMatrix`].
///
/// This extension trait enables Fisher Scoring solvers to construct the
/// weighted Gram matrix `X^T W X` for each parameter block. Only predictor
/// blocks with a linear structure can provide this — nonlinear blocks like
/// [`TransformedScalar`] or [`ProductBlock`] must fall back to gradient-only
/// optimizers.
///
/// Currently only [`LinearPredictorBlock`] implements this trait.
/// Future sparse or structured matrix backends will implement it as well.
pub trait HasDesignMatrix: PredictorBlock {
    /// The underlying design matrix type.
    type Matrix: DesignMatrix;

    /// Returns a reference to the design matrix.
    fn design(&self) -> &Self::Matrix;
}

impl<X: DesignMatrix> HasDesignMatrix for LinearPredictorBlock<X> {
    type Matrix = X;

    #[inline(always)]
    fn design(&self) -> &Self::Matrix {
        &self.x
    }
}

/// Transform for a single coefficient used by [`TransformedScalar`].
pub trait CoefficientTransform {
    /// Transformed coefficient value.
    fn value(beta: f64) -> f64;
    /// Derivative of [`Self::value`] with respect to `beta`.
    fn derivative(beta: f64) -> f64;
}

/// Softplus coefficient transform: `softplus(beta)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SoftplusTransform;

impl CoefficientTransform for SoftplusTransform {
    #[inline(always)]
    fn value(beta: f64) -> f64 {
        Softplus::inverse(beta)
    }

    #[inline(always)]
    fn derivative(beta: f64) -> f64 {
        Softplus::derivative_inverse(beta)
    }
}

/// Negative softplus coefficient transform: `-softplus(beta)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NegativeSoftplusTransform;

impl CoefficientTransform for NegativeSoftplusTransform {
    #[inline(always)]
    fn value(beta: f64) -> f64 {
        -Softplus::inverse(beta)
    }

    #[inline(always)]
    fn derivative(beta: f64) -> f64 {
        -Softplus::derivative_inverse(beta)
    }
}

/// One-coefficient predictor block with a scalar coefficient transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransformedScalar<T> {
    /// Number of observations this scalar contribution applies to.
    pub nrows: usize,
    marker: PhantomData<T>,
}

impl<T> TransformedScalar<T> {
    /// Creates a transformed scalar predictor for `nrows` observations.
    #[must_use]
    #[inline]
    pub const fn new(nrows: usize) -> Self {
        Self {
            nrows,
            marker: PhantomData,
        }
    }
}

impl<T> PredictorBlock for TransformedScalar<T>
where
    T: CoefficientTransform,
{
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline(always)]
    fn nparams(&self) -> usize {
        1
    }

    #[inline(always)]
    fn eta_row(&self, _: usize, beta: &[f64]) -> f64 {
        T::value(beta[0])
    }

    #[inline]
    fn add_gradient(&self, scores: &[f64], beta: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(scores.len(), self.nrows);
        debug_assert_eq!(beta.len(), 1);
        debug_assert_eq!(grad.len(), 1);

        grad[0] = scores
            .iter()
            .sum::<f64>()
            .mul_add(T::derivative(beta[0]), grad[0]);
    }

    #[inline]
    fn add_weighted_gradient(
        &self,
        scores: &[f64],
        multiplier: &[f64],
        beta: &[f64],
        grad: &mut [f64],
    ) {
        debug_assert_eq!(scores.len(), self.nrows);
        debug_assert_eq!(multiplier.len(), self.nrows);
        debug_assert_eq!(beta.len(), 1);
        debug_assert_eq!(grad.len(), 1);

        grad[0] = weighted_sum(scores, multiplier).mul_add(T::derivative(beta[0]), grad[0]);
    }
}

/// Convenience predictor block alias for `softplus(beta)`.
///
/// The generic building block is [`TransformedScalar`]; this alias is provided
/// for common scalar constraints.
pub type SoftplusScalar = TransformedScalar<SoftplusTransform>;

/// Convenience predictor block alias for `-softplus(beta)`.
///
/// The generic building block is [`TransformedScalar`]; this alias is provided
/// for common scalar constraints.
pub type NegativeSoftplusScalar = TransformedScalar<NegativeSoftplusTransform>;

/// Convenience one-coefficient predictor block: `floor + softplus(beta)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FloorSoftplusScalar {
    /// Number of observations this scalar contribution applies to.
    pub nrows: usize,
    /// Constant floor added after the softplus transform.
    pub floor: f64,
}

impl FloorSoftplusScalar {
    /// Creates a floor-plus-softplus scalar predictor.
    #[must_use]
    #[inline]
    pub const fn new(nrows: usize, floor: f64) -> Self {
        Self { nrows, floor }
    }
}

impl PredictorBlock for FloorSoftplusScalar {
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline(always)]
    fn nparams(&self) -> usize {
        1
    }

    #[inline(always)]
    fn eta_row(&self, _: usize, beta: &[f64]) -> f64 {
        self.floor + Softplus::inverse(beta[0])
    }

    #[inline]
    fn add_gradient(&self, scores: &[f64], beta: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(scores.len(), self.nrows);
        debug_assert_eq!(beta.len(), 1);
        debug_assert_eq!(grad.len(), 1);

        grad[0] = scores
            .iter()
            .sum::<f64>()
            .mul_add(Softplus::derivative_inverse(beta[0]), grad[0]);
    }

    #[inline]
    fn add_weighted_gradient(
        &self,
        scores: &[f64],
        multiplier: &[f64],
        beta: &[f64],
        grad: &mut [f64],
    ) {
        debug_assert_eq!(scores.len(), self.nrows);
        debug_assert_eq!(multiplier.len(), self.nrows);
        debug_assert_eq!(beta.len(), 1);
        debug_assert_eq!(grad.len(), 1);

        grad[0] = weighted_sum(scores, multiplier)
            .mul_add(Softplus::derivative_inverse(beta[0]), grad[0]);
    }
}

/// Zero-coefficient constant predictor block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OffsetBlock {
    /// Number of observations this offset applies to.
    pub nrows: usize,
    /// Constant contribution.
    pub value: f64,
}

impl OffsetBlock {
    /// Creates a constant predictor block.
    #[must_use]
    #[inline]
    pub const fn new(nrows: usize, value: f64) -> Self {
        Self { nrows, value }
    }
}

impl PredictorBlock for OffsetBlock {
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline(always)]
    fn nparams(&self) -> usize {
        0
    }

    #[inline(always)]
    fn eta_row(&self, _: usize, _: &[f64]) -> f64 {
        self.value
    }

    #[inline(always)]
    fn add_gradient(&self, _: &[f64], _: &[f64], _: &mut [f64]) {}

    #[inline(always)]
    fn add_weighted_gradient(&self, _: &[f64], _: &[f64], _: &[f64], _: &mut [f64]) {}
}

/// Product/interacted predictor block: `multiplier[row] * inner.eta_row(row)`.
#[derive(Debug, Clone, PartialEq)]
pub struct ProductBlock<X> {
    /// Per-observation multiplier.
    pub multiplier: Vec<f64>,
    /// Wrapped predictor block.
    pub inner: X,
}

impl<X> ProductBlock<X> {
    /// Creates a product predictor block.
    #[must_use]
    #[inline]
    pub const fn new(multiplier: Vec<f64>, inner: X) -> Self {
        Self { multiplier, inner }
    }

    /// Consumes the wrapper and returns `(multiplier, inner)`.
    #[must_use]
    #[inline]
    pub fn into_inner(self) -> (Vec<f64>, X) {
        (self.multiplier, self.inner)
    }
}

impl<X> PredictorBlock for ProductBlock<X>
where
    X: PredictorBlock,
{
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    #[inline(always)]
    fn nparams(&self) -> usize {
        self.inner.nparams()
    }

    #[inline(always)]
    fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
        self.multiplier[row] * self.inner.eta_row(row, beta)
    }

    #[inline]
    fn add_gradient(&self, scores: &[f64], beta: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(scores.len(), self.nrows());
        debug_assert_eq!(self.multiplier.len(), self.nrows());

        self.inner
            .add_weighted_gradient(scores, &self.multiplier, beta, grad);
    }

    #[inline]
    fn validate(&self) -> Result<(), ModelError> {
        self.inner.validate()?;
        if self.multiplier.len() == self.inner.nrows() {
            Ok(())
        } else {
            Err(ModelError::DesignRowMismatch {
                parameter: "product multiplier",
                expected_rows: self.inner.nrows(),
                actual_rows: self.multiplier.len(),
            })
        }
    }
}

#[inline]
fn weighted_sum(scores: &[f64], multiplier: &[f64]) -> f64 {
    scores
        .iter()
        .zip(multiplier)
        .map(|(score, multiplier)| score * multiplier)
        .sum()
}

/// Sum of several predictor blocks sharing the same observations.
///
/// The local beta slice is split between terms in tuple order. This keeps
/// nonlinear or sparse user-defined terms composable without dynamic dispatch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SumBlock<Terms> {
    /// Predictor terms summed into one parameter predictor.
    pub terms: Terms,
}

impl<Terms> SumBlock<Terms> {
    /// Creates a summed predictor from tuple terms.
    #[must_use]
    #[inline]
    pub const fn new(terms: Terms) -> Self {
        Self { terms }
    }
}

macro_rules! impl_sum_block {
    (
        terms = ($($term:ident),+);
        vars = ($($var:ident),+);
        indices = ($($idx:tt),+);
        names = ($($name:literal),+)
    ) => {
        impl<$($term,)+> PredictorBlock for SumBlock<($($term,)+)>
        where
            $($term: PredictorBlock,)+
        {
            #[inline(always)]
            fn nrows(&self) -> usize {
                self.terms.0.nrows()
            }

            #[inline(always)]
            fn nparams(&self) -> usize {
                0 $(+ self.terms.$idx.nparams())+
            }

            #[inline]
            fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
                let mut start = 0;
                let mut eta = 0.0;
                $(
                    let $var = &self.terms.$idx;
                    let end = start + $var.nparams();
                    eta += $var.eta_row(row, &beta[start..end]);
                    start = end;
                )+
                let _ = start;
                eta
            }

            #[inline]
            fn add_gradient(&self, scores: &[f64], beta: &[f64], grad: &mut [f64]) {
                let mut start = 0;
                $(
                    let $var = &self.terms.$idx;
                    let end = start + $var.nparams();
                    $var.add_gradient(scores, &beta[start..end], &mut grad[start..end]);
                    start = end;
                )+
                let _ = start;
            }

            #[inline]
            fn add_weighted_gradient(
                &self,
                scores: &[f64],
                multiplier: &[f64],
                beta: &[f64],
                grad: &mut [f64],
            ) {
                let mut start = 0;
                $(
                    let $var = &self.terms.$idx;
                    let end = start + $var.nparams();
                    $var.add_weighted_gradient(
                        scores,
                        multiplier,
                        &beta[start..end],
                        &mut grad[start..end],
                    );
                    start = end;
                )+
                let _ = start;
            }

            #[inline]
            fn validate(&self) -> Result<(), ModelError> {
                let expected_rows = self.terms.0.nrows();
                $(
                    self.terms.$idx.validate()?;
                    if self.terms.$idx.nrows() != expected_rows {
                        return Err(ModelError::DesignRowMismatch {
                            parameter: $name,
                            expected_rows,
                            actual_rows: self.terms.$idx.nrows(),
                        });
                    }
                )+
                Ok(())
            }
        }
    };
}

impl_sum_block!(
    terms = (T1);
    vars = (term1);
    indices = (0);
    names = ("sum term")
);

impl_sum_block!(
    terms = (T1, T2);
    vars = (term1, term2);
    indices = (0, 1);
    names = ("sum first term", "sum second term")
);

impl_sum_block!(
    terms = (T1, T2, T3);
    vars = (term1, term2, term3);
    indices = (0, 1, 2);
    names = ("sum first term", "sum second term", "sum third term")
);

impl_sum_block!(
    terms = (T1, T2, T3, T4);
    vars = (term1, term2, term3, term4);
    indices = (0, 1, 2, 3);
    names = (
        "sum first term",
        "sum second term",
        "sum third term",
        "sum fourth term"
    )
);

impl_sum_block!(
    terms = (T1, T2, T3, T4, T5);
    vars = (term1, term2, term3, term4, term5);
    indices = (0, 1, 2, 3, 4);
    names = (
        "sum first term",
        "sum second term",
        "sum third term",
        "sum fourth term",
        "sum fifth term"
    )
);

impl_sum_block!(
    terms = (T1, T2, T3, T4, T5, T6);
    vars = (term1, term2, term3, term4, term5, term6);
    indices = (0, 1, 2, 3, 4, 5);
    names = (
        "sum first term",
        "sum second term",
        "sum third term",
        "sum fourth term",
        "sum fifth term",
        "sum sixth term"
    )
);

impl_sum_block!(
    terms = (T1, T2, T3, T4, T5, T6, T7);
    vars = (term1, term2, term3, term4, term5, term6, term7);
    indices = (0, 1, 2, 3, 4, 5, 6);
    names = (
        "sum first term",
        "sum second term",
        "sum third term",
        "sum fourth term",
        "sum fifth term",
        "sum sixth term",
        "sum seventh term"
    )
);

impl_sum_block!(
    terms = (T1, T2, T3, T4, T5, T6, T7, T8);
    vars = (term1, term2, term3, term4, term5, term6, term7, term8);
    indices = (0, 1, 2, 3, 4, 5, 6, 7);
    names = (
        "sum first term",
        "sum second term",
        "sum third term",
        "sum fourth term",
        "sum fifth term",
        "sum sixth term",
        "sum seventh term",
        "sum eighth term"
    )
);

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{DenseDesign, ModelError, PredictorBlock};

    use super::{
        FloorSoftplusScalar, LinearPredictorBlock, NegativeSoftplusScalar, OffsetBlock,
        ProductBlock, SoftplusScalar,
    };

    #[test]
    fn linear_predictor_block_matches_design_matrix_operations() {
        let design = DenseDesign::from_rows(&[[1.0, 2.0], [3.0, 4.0]]);
        let block = LinearPredictorBlock::new(design);
        let beta = [10.0, 1.0];

        assert_relative_eq!(block.eta_row(1, &beta), 34.0);

        let mut grad = vec![0.0, 0.0];
        block.add_gradient(&[0.5, 2.0], &beta, &mut grad);

        assert_relative_eq!(grad[0], 6.5);
        assert_relative_eq!(grad[1], 9.0);
    }

    #[test]
    fn linear_predictor_block_fuses_weighted_gradient() {
        let design = DenseDesign::from_rows(&[[1.0, 2.0], [3.0, 4.0]]);
        let block = LinearPredictorBlock::new(design);
        let beta = [10.0, 1.0];
        let mut grad = vec![1.0, 1.0];

        block.add_weighted_gradient(&[0.5, 2.0], &[2.0, -1.0], &beta, &mut grad);

        assert_relative_eq!(grad[0], -4.0);
        assert_relative_eq!(grad[1], -5.0);
    }

    #[test]
    fn sum_block_supports_eight_terms() {
        let terms = (
            LinearPredictorBlock::new(DenseDesign::column(&[1.0, 2.0])),
            LinearPredictorBlock::new(DenseDesign::column(&[2.0, 3.0])),
            LinearPredictorBlock::new(DenseDesign::column(&[3.0, 4.0])),
            LinearPredictorBlock::new(DenseDesign::column(&[4.0, 5.0])),
            LinearPredictorBlock::new(DenseDesign::column(&[5.0, 6.0])),
            LinearPredictorBlock::new(DenseDesign::column(&[6.0, 7.0])),
            LinearPredictorBlock::new(DenseDesign::column(&[7.0, 8.0])),
            LinearPredictorBlock::new(DenseDesign::column(&[8.0, 9.0])),
        );
        let block = crate::SumBlock::new(terms);
        let beta = [1.0; 8];

        assert_eq!(block.nparams(), 8);
        assert_relative_eq!(block.eta_row(1, &beta), 44.0);

        let mut grad = vec![0.0; 8];
        block.add_gradient(&[0.5, 2.0], &beta, &mut grad);

        assert_relative_eq!(grad[0], 4.5);
        assert_relative_eq!(grad[7], 22.0);
    }

    #[test]
    fn transformed_scalar_blocks_match_finite_difference() {
        assert_scalar_gradient_matches_finite_difference(SoftplusScalar::new(3), &[0.5, 1.0, 2.0]);
        assert_scalar_gradient_matches_finite_difference(
            NegativeSoftplusScalar::new(3),
            &[0.5, 1.0, 2.0],
        );
        assert_scalar_gradient_matches_finite_difference(
            FloorSoftplusScalar::new(3, 10.0),
            &[0.5, 1.0, 2.0],
        );
    }

    fn assert_scalar_gradient_matches_finite_difference(
        block: impl PredictorBlock,
        scores: &[f64],
    ) {
        let beta = [0.4];
        let eps = 1.0e-6;
        let mut grad = [0.0];
        block.add_gradient(scores, &beta, &mut grad);

        let mut finite_difference = 0.0;
        for (row, score) in scores.iter().copied().enumerate() {
            let plus = block.eta_row(row, &[beta[0] + eps]);
            let minus = block.eta_row(row, &[beta[0] - eps]);
            finite_difference += score * (plus - minus) / (2.0 * eps);
        }

        assert_relative_eq!(grad[0], finite_difference, epsilon = 1.0e-6);
    }

    #[test]
    fn offset_block_is_constant_and_has_no_gradient() {
        let block = OffsetBlock::new(2, 3.5);
        let mut grad = [];

        assert_eq!(block.nparams(), 0);
        assert_relative_eq!(block.eta_row(1, &[]), 3.5);
        block.add_gradient(&[1.0, 2.0], &[], &mut grad);
    }

    #[test]
    fn product_block_scales_eta_and_gradient() {
        let inner = LinearPredictorBlock::new(DenseDesign::from_rows(&[[1.0, 2.0], [3.0, 4.0]]));
        let block = ProductBlock::new(vec![2.0, -1.0], inner);
        let beta = [0.5, 1.0];
        let scores = [0.25, 2.0];
        let mut grad = [0.0, 0.0];

        assert_relative_eq!(block.eta_row(0, &beta), 5.0);
        assert_relative_eq!(block.eta_row(1, &beta), -5.5);

        block.add_gradient(&scores, &beta, &mut grad);
        assert_relative_eq!(grad[0], 2.0 * 0.25 * 1.0 - 1.0 * 2.0 * 3.0);
        assert_relative_eq!(grad[1], 2.0 * 0.25 * 2.0 - 1.0 * 2.0 * 4.0);
    }

    #[test]
    fn product_block_validates_multiplier_length() {
        let inner = LinearPredictorBlock::new(DenseDesign::intercept(2));
        let block = ProductBlock::new(vec![1.0], inner);

        assert_eq!(
            block.validate().unwrap_err(),
            ModelError::DesignRowMismatch {
                parameter: "product multiplier",
                expected_rows: 2,
                actual_rows: 1,
            }
        );
    }
}
