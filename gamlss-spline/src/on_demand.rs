use std::ops::Range;

use gamlss_core::{LinearPredictorGeometry, ModelError, PredictorBlock, RowMultiplier};

use crate::geometry::{validate_gram_lengths, validate_transpose_lengths};
use crate::{SplineBasis1d, SplineError, SplineRowBasis};

/// A spline predictor that retains coordinates but evaluates row geometry on demand.
///
/// The design owns `O(nrows)` coordinates and one basis metadata value. It does
/// not retain basis indices or weights for individual rows. This makes it the
/// low-memory counterpart to compact prepared designs such as
/// [`crate::OpenUniformSplineDesign`] and [`crate::CyclicSplineDesign`], at the
/// cost of reevaluating the basis during prediction, gradient, and Gram
/// operations.
///
/// `B` remains a concrete type, so choosing this representation does not add
/// runtime dispatch. For a one-shot coordinate stream that does not need the
/// repeated indexed passes required by [`PredictorBlock`], call
/// [`SplineBasis1d::for_each_basis`] directly instead.
///
/// Weighted Gram evaluation is also allocation-free, but it may reevaluate a
/// row several times to avoid retaining generic scratch storage. Prefer a
/// compact prepared or dense design when an expensive or dense basis is used
/// in repeated second-order updates.
#[derive(Debug, Clone, PartialEq)]
pub struct OnDemandSplineDesign<B> {
    x: Box<[f64]>,
    basis: B,
}

impl<B> OnDemandSplineDesign<B>
where
    B: SplineBasis1d,
{
    /// Builds an on-demand design by copying `x`.
    ///
    /// Construction validates coordinates without evaluating or retaining
    /// their basis geometry.
    ///
    /// # Errors
    ///
    /// Returns the first coordinate-validation error, including
    /// [`SplineError::NonFiniteValue`] for a non-finite coordinate.
    pub fn new(x: &[f64], basis: B) -> Result<Self, SplineError> {
        validate_coordinates(x, &basis)?;
        Ok(Self { x: x.into(), basis })
    }

    /// Builds an on-demand design while taking ownership of `x`.
    ///
    /// This avoids copying an existing coordinate buffer. Construction
    /// validates coordinates without evaluating or retaining their basis
    /// geometry.
    ///
    /// # Errors
    ///
    /// Returns the first coordinate-validation error.
    pub fn from_boxed(x: Box<[f64]>, basis: B) -> Result<Self, SplineError> {
        validate_coordinates(&x, &basis)?;
        Ok(Self { x, basis })
    }

    /// Basis metadata used by the design.
    #[must_use]
    #[inline]
    pub const fn basis(&self) -> &B {
        &self.basis
    }

    /// Retained input coordinates.
    #[must_use]
    #[inline]
    pub fn x(&self) -> &[f64] {
        &self.x
    }

    /// Number of retained observation rows.
    #[must_use]
    #[inline]
    pub fn nrows(&self) -> usize {
        self.x.len()
    }

    /// Number of spline coefficients.
    #[must_use]
    #[inline]
    pub fn n_basis(&self) -> usize {
        self.basis.n_basis()
    }

    /// Decomposes the design into its coordinate storage and basis metadata.
    #[must_use]
    #[inline]
    pub fn into_parts(self) -> (Box<[f64]>, B) {
        (self.x, self.basis)
    }

    #[inline]
    fn visit_row(&self, row: usize, mut f: impl FnMut(usize, f64)) {
        debug_assert!(row < self.x.len());
        let n_basis = self.basis.n_basis();
        let result = self.basis.for_each_basis(self.x[row], |index, weight| {
            debug_assert!(index < n_basis);
            f(index, weight);
        });
        result.expect(
            "a basis evaluation failed after OnDemandSplineDesign validated the coordinate",
        );
    }

    #[inline]
    fn add_scaled_outer_row(&self, row: usize, scale: f64, out: &mut [f64]) {
        self.basis.add_scaled_outer(self.x[row], scale, out).expect(
            "a basis evaluation failed after OnDemandSplineDesign validated the coordinate",
        );
    }
}

impl<B> SplineRowBasis for OnDemandSplineDesign<B>
where
    B: SplineBasis1d,
{
    #[inline]
    fn nrows(&self) -> usize {
        self.x.len()
    }

    #[inline]
    fn nparams(&self) -> usize {
        self.basis.n_basis()
    }

    #[inline]
    fn for_each_row_basis(&self, row: usize, f: impl FnMut(usize, f64)) {
        self.visit_row(row, f);
    }
}

impl<B> PredictorBlock for OnDemandSplineDesign<B>
where
    B: SplineBasis1d,
{
    #[inline]
    fn nrows(&self) -> usize {
        self.x.len()
    }

    #[inline]
    fn nparams(&self) -> usize {
        self.basis.n_basis()
    }

    #[inline]
    fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
        debug_assert_eq!(beta.len(), self.basis.n_basis());

        let mut value = 0.0;
        self.visit_row(row, |index, weight| {
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
        debug_assert!(rows.end <= self.x.len());
        debug_assert_eq!(scores.len(), rows.len());
        debug_assert_eq!(grad.len(), self.basis.n_basis());

        for (offset, score) in scores.iter().copied().enumerate() {
            if score == 0.0 {
                continue;
            }
            let row = rows.start + offset;
            self.visit_row(row, |index, weight| {
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
        debug_assert!(rows.end <= self.x.len());
        debug_assert_eq!(scores.len(), rows.len());
        debug_assert_eq!(grad.len(), self.basis.n_basis());

        for (offset, score) in scores.iter().copied().enumerate() {
            if score == 0.0 {
                continue;
            }
            let row = rows.start + offset;
            let scaled_score = score * multiplier.multiplier_at(row);
            if scaled_score == 0.0 {
                continue;
            }
            self.visit_row(row, |index, weight| {
                grad[index] = scaled_score.mul_add(weight, grad[index]);
            });
        }
    }
}

impl<B> LinearPredictorGeometry for OnDemandSplineDesign<B>
where
    B: SplineBasis1d,
{
    #[inline]
    fn add_weighted_gram(&self, row_weights: &[f64], out: &mut [f64]) -> Result<(), ModelError> {
        validate_gram_lengths(self.x.len(), self.basis.n_basis(), row_weights, out)?;

        for (row, weight) in row_weights.iter().copied().enumerate() {
            if weight != 0.0 {
                self.add_scaled_outer_row(row, weight, out);
            }
        }
        Ok(())
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
        validate_gram_lengths(self.x.len(), self.basis.n_basis(), row_weights, out)?;

        for (row, weight) in row_weights.iter().copied().enumerate() {
            if weight == 0.0 {
                continue;
            }
            let scaled_weight = weight * multiplier.multiplier_at(row);
            if scaled_weight != 0.0 {
                self.add_scaled_outer_row(row, scaled_weight, out);
            }
        }
        Ok(())
    }

    #[inline]
    fn add_t_mul_vec(&self, row_scores: &[f64], out: &mut [f64]) -> Result<(), ModelError> {
        validate_transpose_lengths(self.x.len(), self.basis.n_basis(), row_scores, out)?;
        self.add_gradient(row_scores, &[], out);
        Ok(())
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
        validate_transpose_lengths(self.x.len(), self.basis.n_basis(), row_scores, out)?;
        self.add_weighted_gradient_by(row_scores, multiplier, &[], out);
        Ok(())
    }
}

#[inline]
fn validate_coordinates<B>(x: &[f64], basis: &B) -> Result<(), SplineError>
where
    B: SplineBasis1d + ?Sized,
{
    for value in x.iter().copied() {
        basis.validate_coordinate(value)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use approx::assert_relative_eq;
    use gamlss_core::PredictorBlock;

    use super::OnDemandSplineDesign;
    use crate::{SplineBasis1d, SplineError};

    struct CountingBasis {
        evaluations: Cell<usize>,
    }

    impl SplineBasis1d for CountingBasis {
        fn n_basis(&self) -> usize {
            1
        }

        fn for_each_basis(&self, x: f64, mut f: impl FnMut(usize, f64)) -> Result<(), SplineError> {
            self.validate_coordinate(x)?;
            self.evaluations.set(self.evaluations.get() + 1);
            f(0, 1.0);
            Ok(())
        }
    }

    #[test]
    fn construction_validates_without_evaluating_rows() {
        let basis = CountingBasis {
            evaluations: Cell::new(0),
        };
        let design = OnDemandSplineDesign::new(&[0.0, 0.5, 1.0], &basis).unwrap();

        assert_eq!(basis.evaluations.get(), 0);
        assert_relative_eq!(design.eta_row(1, &[2.0]), 2.0);
        assert_eq!(basis.evaluations.get(), 1);
    }
}
