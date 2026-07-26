use std::ops::Range;

use gamlss_core::{LinearPredictorGeometry, ModelError, PredictorBlock, RowMultiplier};

use crate::geometry::{
    add_row_basis_t_mul_vec, add_row_basis_t_mul_vec_by, add_row_basis_weighted_gram,
    add_row_basis_weighted_gram_by,
};
use crate::row_basis::SplineRowBasis;
use crate::{DifferentiableSplineRowBasis, HelmertContrastDesign, SplineError};

/// Structured tensor-product spline predictor.
///
/// If one row of the left basis is $A_i(x)$ for $0\le i<n_A$ and the matching row of the right basis is $B_j(z)$ for $0\le j<n_B$, the tensor basis and predictor are
///
/// $$
/// T_{ij}(x,z)=A_i(x)B_j(z),
/// \qquad
/// \eta(x,z)=\sum_i\sum_j\beta_{ij}T_{ij}(x,z).
/// $$
///
/// Here $n_A$ and $n_B$ are [`TensorSplineDesign::left_nparams`] and [`TensorSplineDesign::right_nparams`]. Coefficients use row-major tensor order: pair $(i,j)$ is stored at $i n_B+j$. The two bases must describe the same observation rows; this is a row-wise Kronecker product, not a Cartesian product of rows.
///
/// The tensor retains its two input designs as supplied and does not
/// materialize tensor rows. Combining prepared component designs keeps their
/// prepared caches; combining [`crate::OnDemandSplineDesign`] components keeps
/// the tensor fully on demand.
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSplineDesign<A, B> {
    left: A,
    right: B,
    nrows: usize,
    left_nparams: usize,
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

        let left_nparams = left.nparams();
        let right_nparams = right.nparams();
        let nparams = left_nparams
            .checked_mul(right_nparams)
            .ok_or(SplineError::ParameterOverflow)?;

        Ok(Self {
            left,
            right,
            nrows,
            left_nparams,
            nparams,
            right_nparams,
        })
    }

    /// Left basis.
    #[must_use]
    #[inline]
    pub const fn left(&self) -> &A {
        &self.left
    }

    /// Right basis.
    #[must_use]
    #[inline]
    pub const fn right(&self) -> &B {
        &self.right
    }

    /// Number of parameters in the left basis.
    #[must_use]
    #[inline]
    pub const fn left_nparams(&self) -> usize {
        self.left_nparams
    }

    /// Number of parameters in the right basis.
    #[must_use]
    #[inline]
    pub const fn right_nparams(&self) -> usize {
        self.right_nparams
    }

    /// Number of tensor-product coefficients.
    #[must_use]
    #[inline]
    pub const fn n_basis(&self) -> usize {
        self.nparams
    }
}

impl<A, B> TensorSplineDesign<A, B>
where
    A: DifferentiableSplineRowBasis,
    B: SplineRowBasis,
{
    /// Partial derivative of the predictor with respect to the left coordinate.
    pub fn eta_left_derivative_row(
        &self,
        row: usize,
        derivative_order: usize,
        beta: &[f64],
    ) -> Result<f64, SplineError> {
        debug_assert_eq!(beta.len(), self.nparams);
        let mut value = 0.0;
        self.left.for_each_row_basis_derivative(
            row,
            derivative_order,
            |left_index, left_weight| {
                self.right
                    .for_each_row_basis(row, |right_index, right_weight| {
                        let index = left_index * self.right_nparams + right_index;
                        value = beta[index].mul_add(left_weight * right_weight, value);
                    });
            },
        )?;
        Ok(value)
    }
}

impl<A, B> TensorSplineDesign<A, B>
where
    A: SplineRowBasis,
    B: DifferentiableSplineRowBasis,
{
    /// Partial derivative of the predictor with respect to the right coordinate.
    pub fn eta_right_derivative_row(
        &self,
        row: usize,
        derivative_order: usize,
        beta: &[f64],
    ) -> Result<f64, SplineError> {
        debug_assert_eq!(beta.len(), self.nparams);
        let mut value = 0.0;
        self.right.for_each_row_basis_derivative(
            row,
            derivative_order,
            |right_index, right_weight| {
                self.left
                    .for_each_row_basis(row, |left_index, left_weight| {
                        let index = left_index * self.right_nparams + right_index;
                        value = beta[index].mul_add(left_weight * right_weight, value);
                    });
            },
        )?;
        Ok(value)
    }
}

impl<A, B> TensorSplineDesign<A, B>
where
    A: DifferentiableSplineRowBasis,
    B: DifferentiableSplineRowBasis,
{
    /// Mixed partial derivative with independent orders for both coordinates.
    pub fn eta_mixed_derivative_row(
        &self,
        row: usize,
        left_derivative_order: usize,
        right_derivative_order: usize,
        beta: &[f64],
    ) -> Result<f64, SplineError> {
        debug_assert_eq!(beta.len(), self.nparams);
        validate_derivative_order(&self.left, left_derivative_order)?;
        validate_derivative_order(&self.right, right_derivative_order)?;
        let mut value = 0.0;
        let mut right_error = None;
        self.left.for_each_row_basis_derivative(
            row,
            left_derivative_order,
            |left_index, left_weight| {
                if right_error.is_some() {
                    return;
                }
                if let Err(error) = self.right.for_each_row_basis_derivative(
                    row,
                    right_derivative_order,
                    |right_index, right_weight| {
                        let index = left_index * self.right_nparams + right_index;
                        value = beta[index].mul_add(left_weight * right_weight, value);
                    },
                ) {
                    right_error = Some(error);
                }
            },
        )?;
        right_error.map_or(Ok(value), Err)
    }
}

impl<A, B> SplineRowBasis for TensorSplineDesign<A, B>
where
    A: SplineRowBasis,
    B: SplineRowBasis,
{
    #[inline]
    fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline]
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
                        let weight = left_weight * right_weight;
                        if weight != 0.0 {
                            f(index, weight);
                        }
                    });
            });
    }
}

impl<A, B> PredictorBlock for TensorSplineDesign<A, B>
where
    A: SplineRowBasis,
    B: SplineRowBasis,
{
    #[inline]
    fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline]
    fn nparams(&self) -> usize {
        self.nparams
    }

    #[inline]
    fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
        debug_assert!(row < self.nrows);
        debug_assert_eq!(beta.len(), self.nparams);

        let mut value = 0.0;
        self.for_each_row_basis(row, |index, weight| {
            value = beta[index].mul_add(weight, value);
        });
        value
    }

    #[inline]
    fn zero_beta_constant_contribution(&self) -> Option<f64> {
        Some(0.0)
    }

    #[inline]
    fn add_gradient_range(&self, rows: Range<usize>, scores: &[f64], _: &[f64], grad: &mut [f64]) {
        debug_assert!(rows.end <= self.nrows);
        debug_assert_eq!(scores.len(), rows.len());
        debug_assert_eq!(grad.len(), self.nparams);

        for (offset, score) in scores.iter().copied().enumerate() {
            if score == 0.0 {
                continue;
            }
            let row = rows.start + offset;
            self.for_each_row_basis(row, |index, weight| {
                grad[index] = score.mul_add(weight, grad[index]);
            });
        }
    }

    #[inline]
    fn add_weighted_gradient_by_range<M>(
        &self,
        rows: Range<usize>,
        scores: &[f64],
        multiplier: &M,
        _: &[f64],
        grad: &mut [f64],
    ) where
        M: RowMultiplier + ?Sized,
    {
        debug_assert!(rows.end <= self.nrows);
        debug_assert_eq!(scores.len(), rows.len());
        debug_assert_eq!(grad.len(), self.nparams);

        for (offset, score) in scores.iter().copied().enumerate() {
            if score == 0.0 {
                continue;
            }
            let row = rows.start + offset;
            let scaled_score = score * multiplier.multiplier_at(row);
            if scaled_score == 0.0 {
                continue;
            }
            self.for_each_row_basis(row, |index, weight| {
                grad[index] = scaled_score.mul_add(weight, grad[index]);
            });
        }
    }
}

impl<A, B> LinearPredictorGeometry for TensorSplineDesign<A, B>
where
    A: SplineRowBasis,
    B: SplineRowBasis,
{
    #[inline]
    fn add_weighted_gram(&self, row_weights: &[f64], out: &mut [f64]) -> Result<(), ModelError> {
        add_row_basis_weighted_gram(self, row_weights, out)
    }

    #[inline]
    fn add_weighted_gram_by<M>(
        &self,
        row_weights: &[f64],
        multiplier: &M,
        out: &mut [f64],
    ) -> Result<(), ModelError>
    where
        M: RowMultiplier + ?Sized,
    {
        add_row_basis_weighted_gram_by(self, row_weights, multiplier, out)
    }

    #[inline]
    fn add_t_mul_vec(&self, row_scores: &[f64], out: &mut [f64]) -> Result<(), ModelError> {
        add_row_basis_t_mul_vec(self, row_scores, out)
    }

    #[inline]
    fn add_t_mul_vec_by<M>(
        &self,
        row_scores: &[f64],
        multiplier: &M,
        out: &mut [f64],
    ) -> Result<(), ModelError>
    where
        M: RowMultiplier + ?Sized,
    {
        add_row_basis_t_mul_vec_by(self, row_scores, multiplier, out)
    }
}
/// Constant-free tensor interaction built from two marginal spline designs.
///
/// Both margins are reparameterized with [`HelmertContrastDesign`] before the
/// row-wise tensor product is formed. Consequently this type cannot reproduce
/// either unconstrained marginal main effect. The transformed marginal rows
/// are prepared once and remain backend-independent.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInteractionDesign<A, B> {
    inner: TensorSplineDesign<HelmertContrastDesign<A>, HelmertContrastDesign<B>>,
}

impl<A, B> TensorInteractionDesign<A, B>
where
    A: SplineRowBasis,
    B: SplineRowBasis,
{
    /// Creates a pure tensor interaction from matching-row marginal designs.
    ///
    /// # Errors
    ///
    /// Returns an error when a margin has fewer than two parameters, row
    /// counts differ, or a transformed/tensor size overflows `usize`.
    pub fn try_new(left: A, right: B) -> Result<Self, SplineError> {
        Ok(Self {
            inner: TensorSplineDesign::new(
                HelmertContrastDesign::try_new(left)?,
                HelmertContrastDesign::try_new(right)?,
            )?,
        })
    }

    /// Constrained left marginal design.
    #[must_use]
    pub const fn left(&self) -> &HelmertContrastDesign<A> {
        self.inner.left()
    }

    /// Constrained right marginal design.
    #[must_use]
    pub const fn right(&self) -> &HelmertContrastDesign<B> {
        self.inner.right()
    }

    /// Number of interaction coefficients.
    #[must_use]
    pub const fn n_basis(&self) -> usize {
        self.inner.n_basis()
    }

    /// Underlying tensor product of constrained margins.
    #[must_use]
    pub const fn as_tensor(
        &self,
    ) -> &TensorSplineDesign<HelmertContrastDesign<A>, HelmertContrastDesign<B>> {
        &self.inner
    }
}

impl<A, B> TensorInteractionDesign<A, B>
where
    A: DifferentiableSplineRowBasis,
    B: SplineRowBasis,
{
    /// Partial derivative with respect to the left coordinate.
    pub fn eta_left_derivative_row(
        &self,
        row: usize,
        derivative_order: usize,
        beta: &[f64],
    ) -> Result<f64, SplineError> {
        self.inner
            .eta_left_derivative_row(row, derivative_order, beta)
    }
}

impl<A, B> TensorInteractionDesign<A, B>
where
    A: SplineRowBasis,
    B: DifferentiableSplineRowBasis,
{
    /// Partial derivative with respect to the right coordinate.
    pub fn eta_right_derivative_row(
        &self,
        row: usize,
        derivative_order: usize,
        beta: &[f64],
    ) -> Result<f64, SplineError> {
        self.inner
            .eta_right_derivative_row(row, derivative_order, beta)
    }
}

impl<A, B> TensorInteractionDesign<A, B>
where
    A: DifferentiableSplineRowBasis,
    B: DifferentiableSplineRowBasis,
{
    /// Mixed partial derivative with independent orders for both coordinates.
    pub fn eta_mixed_derivative_row(
        &self,
        row: usize,
        left_derivative_order: usize,
        right_derivative_order: usize,
        beta: &[f64],
    ) -> Result<f64, SplineError> {
        self.inner.eta_mixed_derivative_row(
            row,
            left_derivative_order,
            right_derivative_order,
            beta,
        )
    }
}

impl<A, B> SplineRowBasis for TensorInteractionDesign<A, B>
where
    A: SplineRowBasis,
    B: SplineRowBasis,
{
    fn nrows(&self) -> usize {
        self.inner.nrows
    }

    fn nparams(&self) -> usize {
        self.inner.nparams
    }

    fn for_each_row_basis(&self, row: usize, f: impl FnMut(usize, f64)) {
        self.inner.for_each_row_basis(row, f);
    }
}

impl<A, B> PredictorBlock for TensorInteractionDesign<A, B>
where
    A: SplineRowBasis,
    B: SplineRowBasis,
{
    fn nrows(&self) -> usize {
        PredictorBlock::nrows(&self.inner)
    }

    fn nparams(&self) -> usize {
        PredictorBlock::nparams(&self.inner)
    }

    fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
        self.inner.eta_row(row, beta)
    }

    fn zero_beta_constant_contribution(&self) -> Option<f64> {
        Some(0.0)
    }

    fn add_gradient_range(
        &self,
        rows: Range<usize>,
        scores: &[f64],
        beta: &[f64],
        grad: &mut [f64],
    ) {
        self.inner.add_gradient_range(rows, scores, beta, grad);
    }

    fn add_weighted_gradient_by_range<M>(
        &self,
        rows: Range<usize>,
        scores: &[f64],
        multiplier: &M,
        beta: &[f64],
        grad: &mut [f64],
    ) where
        M: RowMultiplier + ?Sized,
    {
        self.inner
            .add_weighted_gradient_by_range(rows, scores, multiplier, beta, grad);
    }
}

impl<A, B> LinearPredictorGeometry for TensorInteractionDesign<A, B>
where
    A: SplineRowBasis,
    B: SplineRowBasis,
{
    fn add_weighted_gram(&self, row_weights: &[f64], out: &mut [f64]) -> Result<(), ModelError> {
        self.inner.add_weighted_gram(row_weights, out)
    }

    fn add_weighted_gram_by<M>(
        &self,
        row_weights: &[f64],
        multiplier: &M,
        out: &mut [f64],
    ) -> Result<(), ModelError>
    where
        M: RowMultiplier + ?Sized,
    {
        self.inner
            .add_weighted_gram_by(row_weights, multiplier, out)
    }

    fn add_t_mul_vec(&self, row_scores: &[f64], out: &mut [f64]) -> Result<(), ModelError> {
        self.inner.add_t_mul_vec(row_scores, out)
    }

    fn add_t_mul_vec_by<M>(
        &self,
        row_scores: &[f64],
        multiplier: &M,
        out: &mut [f64],
    ) -> Result<(), ModelError>
    where
        M: RowMultiplier + ?Sized,
    {
        self.inner.add_t_mul_vec_by(row_scores, multiplier, out)
    }
}

fn validate_derivative_order<B>(basis: &B, requested: usize) -> Result<(), SplineError>
where
    B: DifferentiableSplineRowBasis + ?Sized,
{
    if let Some(max) = basis.max_derivative_order()
        && requested > max
    {
        Err(SplineError::UnsupportedDerivativeOrder { requested, max })
    } else {
        Ok(())
    }
}

