use std::marker::PhantomData;

use crate::{DesignMatrix, Link, ModelError, RowMultiplier, Softplus};

const EXPECTED_FINITE: &str = "finite";

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

/// Linear predictor block backed by a [`DesignMatrix`].
///
/// This is the explicit adapter from matrix-based predictors to the more
/// general [`PredictorBlock`] extension point.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LinearPredictorBlock<X> {
    /// Design matrix used by this predictor.
    x: X,
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
    pub const fn x(&self) -> &X {
        &self.x
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
    fn set_constant_start(&self, value: f64, beta: &mut [f64]) -> bool {
        self.x.set_constant_start(value, beta)
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

    #[inline]
    fn add_weighted_gradient_by<M>(
        &self,
        scores: &[f64],
        multiplier: &M,
        _: &[f64],
        grad: &mut [f64],
    ) where
        M: RowMultiplier + ?Sized,
    {
        self.x.add_weighted_t_mul_vec_by(scores, multiplier, grad);
    }

    #[inline]
    fn zero_beta_constant_contribution(&self) -> Option<f64> {
        Some(0.0)
    }
}

impl<X: DesignMatrix> HasDesignMatrix for LinearPredictorBlock<X> {
    type Matrix = X;

    #[inline(always)]
    fn design(&self) -> &Self::Matrix {
        &self.x
    }
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
    nrows: usize,
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

    /// Returns the number of observations this scalar contribution applies to.
    #[must_use]
    #[inline]
    pub const fn nrows(&self) -> usize {
        self.nrows
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

    #[inline]
    fn zero_beta_constant_contribution(&self) -> Option<f64> {
        let value = T::value(0.0);
        value.is_finite().then_some(value)
    }
}

/// Convenience one-coefficient predictor block: `floor + softplus(beta)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FloorSoftplusScalar {
    /// Number of observations this scalar contribution applies to.
    nrows: usize,
    /// Constant floor added after the softplus transform.
    floor: f64,
}

impl FloorSoftplusScalar {
    /// Creates a floor-plus-softplus scalar predictor.
    #[must_use]
    #[inline]
    pub const fn new(nrows: usize, floor: f64) -> Self {
        Self { nrows, floor }
    }

    /// Creates a floor-plus-softplus scalar predictor with a finite floor.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `floor` is not finite.
    #[inline]
    pub fn try_new(nrows: usize, floor: f64) -> Result<Self, ModelError> {
        validate_finite("floor", floor)?;
        Ok(Self::new(nrows, floor))
    }

    /// Returns the number of observations this scalar contribution applies to.
    #[must_use]
    #[inline]
    pub const fn nrows(&self) -> usize {
        self.nrows
    }

    /// Returns the constant floor added after the softplus transform.
    #[must_use]
    #[inline]
    pub const fn floor(&self) -> f64 {
        self.floor
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

    #[inline]
    fn zero_beta_constant_contribution(&self) -> Option<f64> {
        let value = self.floor + Softplus::inverse(0.0);
        value.is_finite().then_some(value)
    }
}

/// Zero-coefficient constant predictor block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OffsetBlock {
    /// Number of observations this offset applies to.
    nrows: usize,
    /// Constant contribution.
    value: f64,
}

impl OffsetBlock {
    /// Creates a constant predictor block.
    #[must_use]
    #[inline]
    pub const fn new(nrows: usize, value: f64) -> Self {
        Self { nrows, value }
    }

    /// Creates a constant predictor block with a finite value.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `value` is not finite.
    #[inline]
    pub fn try_new(nrows: usize, value: f64) -> Result<Self, ModelError> {
        validate_finite("offset value", value)?;
        Ok(Self::new(nrows, value))
    }

    /// Returns the number of observations this offset applies to.
    #[must_use]
    #[inline]
    pub const fn nrows(&self) -> usize {
        self.nrows
    }

    /// Returns the constant contribution.
    #[must_use]
    #[inline]
    pub const fn value(&self) -> f64 {
        self.value
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

    #[inline]
    fn zero_beta_constant_contribution(&self) -> Option<f64> {
        self.value.is_finite().then_some(self.value)
    }
}

/// Product/interacted predictor block: `multiplier[row] * inner.eta_row(row)`.
#[derive(Debug, Clone, PartialEq)]
pub struct ProductBlock<X> {
    /// Per-observation multiplier.
    multiplier: Vec<f64>,
    /// Wrapped predictor block.
    inner: X,
}

impl<X> ProductBlock<X> {
    /// Creates a product predictor block without validating dimensions.
    ///
    /// Use [`Self::try_new`] when the multiplier comes from user input or
    /// dynamic model metadata.
    #[must_use]
    #[inline]
    pub const fn new(multiplier: Vec<f64>, inner: X) -> Self {
        Self { multiplier, inner }
    }

    /// Returns the per-observation multiplier.
    #[must_use]
    #[inline]
    pub fn multiplier(&self) -> &[f64] {
        &self.multiplier
    }

    /// Returns the wrapped predictor block.
    #[must_use]
    #[inline]
    pub const fn inner(&self) -> &X {
        &self.inner
    }

    /// Consumes the wrapper and returns the wrapped predictor block.
    #[must_use]
    #[inline]
    pub fn into_inner(self) -> X {
        self.inner
    }

    /// Consumes the wrapper and returns `(multiplier, inner)`.
    #[must_use]
    #[inline]
    pub fn into_parts(self) -> (Vec<f64>, X) {
        (self.multiplier, self.inner)
    }
}

impl<X> ProductBlock<X>
where
    X: PredictorBlock,
{
    /// Creates a product predictor block after validating multiplier shape.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::DesignRowMismatch`] when `multiplier.len()` does
    /// not match `inner.nrows()`, or [`ModelError::InvalidMultiplier`] when a
    /// multiplier value is not finite.
    #[inline]
    pub fn try_new(multiplier: Vec<f64>, inner: X) -> Result<Self, ModelError> {
        let block = Self::new(multiplier, inner);
        block.validate()?;
        Ok(block)
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
    fn add_weighted_gradient(
        &self,
        scores: &[f64],
        multiplier: &[f64],
        beta: &[f64],
        grad: &mut [f64],
    ) {
        debug_assert_eq!(scores.len(), self.nrows());
        debug_assert_eq!(multiplier.len(), self.nrows());
        debug_assert_eq!(self.multiplier.len(), self.nrows());

        self.add_weighted_gradient_by(scores, multiplier, beta, grad);
    }

    #[inline]
    fn add_weighted_gradient_by<M>(
        &self,
        scores: &[f64],
        multiplier: &M,
        beta: &[f64],
        grad: &mut [f64],
    ) where
        M: RowMultiplier + ?Sized,
    {
        debug_assert_eq!(scores.len(), self.nrows());
        debug_assert_eq!(self.multiplier.len(), self.nrows());

        let product_multiplier = ProductRowMultiplier {
            left: self.multiplier.as_slice(),
            right: multiplier,
        };
        self.inner
            .add_weighted_gradient_by(scores, &product_multiplier, beta, grad);
    }

    #[inline]
    fn validate(&self) -> Result<(), ModelError> {
        self.inner.validate()?;
        if self.multiplier.len() != self.inner.nrows() {
            return Err(ModelError::DesignRowMismatch {
                parameter: "product multiplier",
                expected_rows: self.inner.nrows(),
                actual_rows: self.multiplier.len(),
            });
        }

        for (index, value) in self.multiplier.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(ModelError::InvalidMultiplier { index });
            }
        }

        Ok(())
    }

    #[inline]
    fn zero_beta_constant_contribution(&self) -> Option<f64> {
        let inner = self.inner.zero_beta_constant_contribution()?;
        if inner == 0.0 {
            return Some(0.0);
        }

        let first = self.multiplier.first().copied()?;
        if !first.is_finite() || !self.multiplier.iter().all(|value| *value == first) {
            return None;
        }

        let value = first * inner;
        value.is_finite().then_some(value)
    }
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

struct ProductRowMultiplier<'a, M>
where
    M: RowMultiplier + ?Sized,
{
    left: &'a [f64],
    right: &'a M,
}

impl<M> RowMultiplier for ProductRowMultiplier<'_, M>
where
    M: RowMultiplier + ?Sized,
{
    #[inline(always)]
    fn multiplier_at(&self, row: usize) -> f64 {
        self.left[row] * self.right.multiplier_at(row)
    }
}

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
    /// Writes a constant predictor start into the local coefficient slice.
    ///
    /// Implementations should return `true` only when the write makes this
    /// block contribute `value` for every row with the rest of the local slice
    /// left at zero. Unsupported blocks should leave `beta` unchanged.
    #[inline]
    fn set_constant_start(&self, _value: f64, _beta: &mut [f64]) -> bool {
        false
    }
    /// Constant contribution when all local coefficients are zero.
    ///
    /// Returns `None` when the zero-coefficient contribution is not constant
    /// across rows or cannot be determined cheaply. [`SumBlock`] uses this to
    /// account for offsets and transformed scalar baselines when constructing
    /// constant starts.
    #[inline]
    fn zero_beta_constant_contribution(&self) -> Option<f64> {
        None
    }
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
        self.add_weighted_gradient_by(scores, multiplier, beta, grad);
    }

    /// Adds the gradient contribution implied by a lazy row multiplier.
    ///
    /// Default implementation materializes scaled scores and delegates to
    /// [`Self::add_gradient`]. Blocks used in nested hot paths should override
    /// this method to keep row scaling fused through composed predictors.
    #[inline]
    fn add_weighted_gradient_by<M>(
        &self,
        scores: &[f64],
        multiplier: &M,
        beta: &[f64],
        grad: &mut [f64],
    ) where
        M: RowMultiplier + ?Sized,
    {
        debug_assert_eq!(scores.len(), self.nrows());

        let scaled_scores = scores
            .iter()
            .enumerate()
            .map(|(row, score)| score * multiplier.multiplier_at(row))
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

/// Transform for a single coefficient used by [`TransformedScalar`].
pub trait CoefficientTransform {
    /// Transformed coefficient value.
    fn value(beta: f64) -> f64;
    /// Derivative of [`Self::value`] with respect to `beta`.
    fn derivative(beta: f64) -> f64;
}

#[inline]
fn weighted_sum(scores: &[f64], multiplier: &[f64]) -> f64 {
    scores
        .iter()
        .zip(multiplier)
        .map(|(score, multiplier)| score * multiplier)
        .sum()
}

fn validate_finite(parameter: &'static str, value: f64) -> Result<(), ModelError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ModelError::InvalidParameter {
            parameter,
            expected: EXPECTED_FINITE,
        })
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
            fn set_constant_start(&self, value: f64, beta: &mut [f64]) -> bool {
                let baselines = [$(self.terms.$idx.zero_beta_constant_contribution(),)+];
                let mut start = 0;
                $(
                    let $var = &self.terms.$idx;
                    let end = start + $var.nparams();
                    let other_baseline = baselines
                        .iter()
                        .enumerate()
                        .filter(|(index, _)| *index != $idx)
                        .try_fold(0.0, |sum, (_, baseline)| baseline.map(|value| sum + value));
                    if let Some(other_baseline) = other_baseline {
                        if $var.set_constant_start(value - other_baseline, &mut beta[start..end]) {
                            return true;
                        }
                    }
                    start = end;
                )+
                let _ = start;
                false
            }

            #[inline]
            fn zero_beta_constant_contribution(&self) -> Option<f64> {
                let mut contribution = 0.0;
                $(
                    contribution += self.terms.$idx.zero_beta_constant_contribution()?;
                )+
                contribution.is_finite().then_some(contribution)
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

    use crate::{DenseDesign, DesignMatrix, ModelError, PredictorBlock};

    use super::{
        FloorSoftplusScalar, LinearPredictorBlock, NegativeSoftplusScalar, OffsetBlock,
        ProductBlock, SoftplusScalar,
    };

    #[test]
    fn linear_predictor_block_matches_design_matrix_operations() {
        let design = DenseDesign::from_rows(&[[1.0, 2.0], [3.0, 4.0]]);
        let block = LinearPredictorBlock::new(design);
        let beta = [10.0, 1.0];

        assert_eq!(block.x().nrows(), 2);
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
        let softplus = SoftplusScalar::new(3);
        assert_eq!(softplus.nrows(), 3);
        assert_scalar_gradient_matches_finite_difference(softplus, &[0.5, 1.0, 2.0]);
        assert_scalar_gradient_matches_finite_difference(
            NegativeSoftplusScalar::new(3),
            &[0.5, 1.0, 2.0],
        );
        let floored = FloorSoftplusScalar::try_new(3, 10.0).unwrap();
        assert_eq!(floored.nrows(), 3);
        assert_relative_eq!(floored.floor(), 10.0);
        assert_scalar_gradient_matches_finite_difference(floored, &[0.5, 1.0, 2.0]);
    }

    #[test]
    fn floor_softplus_scalar_try_new_validates_floor() {
        assert_eq!(
            FloorSoftplusScalar::try_new(2, f64::NAN).unwrap_err(),
            ModelError::InvalidParameter {
                parameter: "floor",
                expected: "finite",
            }
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
        let block = OffsetBlock::try_new(2, 3.5).unwrap();
        let mut grad = [];

        assert_eq!(block.nrows(), 2);
        assert_relative_eq!(block.value(), 3.5);
        assert_eq!(block.nparams(), 0);
        assert_relative_eq!(block.eta_row(1, &[]), 3.5);
        block.add_gradient(&[1.0, 2.0], &[], &mut grad);
    }

    #[test]
    fn offset_block_try_new_validates_value() {
        assert_eq!(
            OffsetBlock::try_new(2, f64::INFINITY).unwrap_err(),
            ModelError::InvalidParameter {
                parameter: "offset value",
                expected: "finite",
            }
        );
    }

    #[test]
    fn product_block_scales_eta_and_gradient() {
        let inner = LinearPredictorBlock::new(DenseDesign::from_rows(&[[1.0, 2.0], [3.0, 4.0]]));
        let block = ProductBlock::try_new(vec![2.0, -1.0], inner).unwrap();
        let beta = [0.5, 1.0];
        let scores = [0.25, 2.0];
        let mut grad = [0.0, 0.0];

        assert_eq!(block.multiplier(), &[2.0, -1.0]);
        assert_eq!(block.inner().nparams(), 2);
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

        assert_multiplier_length_error(block.validate().unwrap_err());
    }

    #[test]
    fn product_block_try_new_validates_multiplier_length() {
        let inner = LinearPredictorBlock::new(DenseDesign::intercept(2));

        assert_multiplier_length_error(ProductBlock::try_new(vec![1.0], inner).unwrap_err());
    }

    #[test]
    fn product_block_validates_multiplier_finiteness() {
        let inner = LinearPredictorBlock::new(DenseDesign::intercept(2));
        let block = ProductBlock::new(vec![1.0, f64::INFINITY], inner);

        assert_invalid_multiplier_error(block.validate().unwrap_err());
    }

    #[test]
    fn product_block_try_new_validates_multiplier_finiteness() {
        let inner = LinearPredictorBlock::new(DenseDesign::intercept(2));

        assert_invalid_multiplier_error(
            ProductBlock::try_new(vec![1.0, f64::INFINITY], inner).unwrap_err(),
        );
    }

    fn assert_multiplier_length_error(error: ModelError) {
        assert_eq!(
            error,
            ModelError::DesignRowMismatch {
                parameter: "product multiplier",
                expected_rows: 2,
                actual_rows: 1,
            }
        );
    }

    fn assert_invalid_multiplier_error(error: ModelError) {
        assert_eq!(error, ModelError::InvalidMultiplier { index: 1 });
    }
}
