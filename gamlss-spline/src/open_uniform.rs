use gamlss_core::{LinearPredictorGeometry, ModelError, PredictorBlock, RowMultiplier};

use crate::local::{LocalBasis, open_uniform_local_basis};
use crate::row_basis::SplineRowBasis;
use crate::{SplineError, SplineOrder};

/// Metadata for an open-uniform spline predictor.
///
/// Stores only the basis shape and scaling range, so it can be reused for
/// building designs on training and new data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpenUniformSplineBasis {
    min: f64,
    max: f64,
    n_basis: usize,
    order: SplineOrder,
    n_intervals: f64,
}

impl OpenUniformSplineBasis {
    /// Builds metadata from the training data range.
    ///
    /// # Errors
    ///
    /// Returns an error if `x` is empty, contains non-finite values, has a
    /// degenerate range, or `n_basis` is insufficient for `order`.
    pub fn from_data(x: &[f64], n_basis: usize, order: SplineOrder) -> Result<Self, SplineError> {
        if x.is_empty() {
            return Err(SplineError::EmptyInput);
        }
        if n_basis < order.min_basis() {
            return Err(SplineError::NotEnoughBasis {
                n_basis,
                degree: order.degree(),
            });
        }

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for value in x.iter().copied() {
            if !value.is_finite() {
                return Err(SplineError::NonFiniteValue);
            }
            min = min.min(value);
            max = max.max(value);
        }
        Self::new(min, max, n_basis, order)
    }

    /// Builds metadata with an explicit finite range.
    ///
    /// # Errors
    ///
    /// Returns an error if the boundaries are not finite, `min >= max`, or
    /// `n_basis` is insufficient for `order`.
    pub fn new(
        min: f64,
        max: f64,
        n_basis: usize,
        order: SplineOrder,
    ) -> Result<Self, SplineError> {
        if !min.is_finite() || !max.is_finite() || min >= max {
            return Err(SplineError::InvalidRange);
        }
        if n_basis < order.min_basis() {
            return Err(SplineError::NotEnoughBasis {
                n_basis,
                degree: order.degree(),
            });
        }

        Ok(Self {
            min,
            max,
            n_basis,
            order,
            n_intervals: (n_basis - order.degree()).max(1) as f64,
        })
    }

    /// Builds a predictor design for a specific set of coordinates.
    ///
    /// # Errors
    ///
    /// Returns an error if `x` contains non-finite values.
    pub fn design(&self, x: &[f64]) -> Result<OpenUniformSplineDesign, SplineError> {
        if x.iter().any(|value| !value.is_finite()) {
            return Err(SplineError::NonFiniteValue);
        }

        Ok(OpenUniformSplineDesign {
            x: x.to_vec(),
            basis: *self,
        })
    }

    /// Visits non-zero basis values for one input coordinate without allocating.
    ///
    /// The callback receives `(basis_index, weight)` pairs in the same
    /// coefficient order as [`OpenUniformSplineDesign`].
    ///
    /// # Errors
    ///
    /// Returns [`SplineError::NonFiniteValue`] if `x` is not finite.
    pub fn for_each_value_basis(
        &self,
        x: f64,
        f: impl FnMut(usize, f64),
    ) -> Result<(), SplineError> {
        if !x.is_finite() {
            return Err(SplineError::NonFiniteValue);
        }

        self.local_basis(x).for_each(f);
        Ok(())
    }

    /// Lower boundary of the basis range.
    #[must_use]
    #[inline(always)]
    pub const fn min(&self) -> f64 {
        self.min
    }

    /// Upper boundary of the basis range.
    #[must_use]
    #[inline(always)]
    pub const fn max(&self) -> f64 {
        self.max
    }

    /// Number of spline coefficients.
    #[must_use]
    #[inline(always)]
    pub const fn n_basis(&self) -> usize {
        self.n_basis
    }

    /// Spline order.
    #[must_use]
    #[inline(always)]
    pub const fn order(&self) -> SplineOrder {
        self.order
    }

    #[inline(always)]
    fn span(&self) -> f64 {
        self.max - self.min
    }

    #[inline]
    fn local_basis(&self, x: f64) -> LocalBasis {
        let u = (x - self.min) / self.span();
        self.local_basis_for_unit(u)
    }

    #[inline]
    fn local_basis_for_unit(&self, u: f64) -> LocalBasis {
        open_uniform_local_basis(u, self.order, self.n_basis, self.n_intervals)
    }
}

/// Open-uniform spline predictor with local sparse row computation.
///
/// Unlike [`crate::BSplineBasis`], stores the original data and computes
/// basis functions "on the fly" via a compact `LocalBasis`, without
/// materializing the full design matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenUniformSplineDesign {
    x: Vec<f64>,
    basis: OpenUniformSplineBasis,
}

impl OpenUniformSplineDesign {
    /// Builds an open-uniform spline design from a data range.
    ///
    /// Returns an error if the data is empty or contains non-finite values.
    pub fn from_data(x: &[f64], n_basis: usize, order: SplineOrder) -> Result<Self, SplineError> {
        OpenUniformSplineBasis::from_data(x, n_basis, order)?.design(x)
    }

    /// Builds an open-uniform spline design with an explicit finite range.
    ///
    /// `min` and `max` must be finite and `min < max`.
    pub fn with_range(
        x: &[f64],
        min: f64,
        max: f64,
        n_basis: usize,
        order: SplineOrder,
    ) -> Result<Self, SplineError> {
        OpenUniformSplineBasis::new(min, max, n_basis, order)?.design(x)
    }

    /// Number of spline coefficients.
    #[must_use]
    #[inline(always)]
    pub fn n_basis(&self) -> usize {
        self.basis.n_basis()
    }

    /// Basis metadata suitable for building a design on new data.
    #[must_use]
    #[inline(always)]
    pub fn basis(&self) -> OpenUniformSplineBasis {
        self.basis
    }

    /// Returns the original coordinates of the design.
    #[must_use]
    #[inline(always)]
    pub fn x(&self) -> &[f64] {
        &self.x
    }

    #[inline]
    fn basis_for_row(&self, row: usize) -> LocalBasis {
        self.basis.local_basis(self.x[row])
    }

    #[inline]
    fn basis_for_unit(&self, u: f64) -> LocalBasis {
        self.basis.local_basis_for_unit(u)
    }

    /// Derivative of the predictor contribution with respect to the original
    /// coordinate `x`.
    #[must_use]
    #[inline]
    pub fn eta_derivative_row(&self, row: usize, beta: &[f64]) -> f64 {
        let span = self.basis.span();
        let h = 1.0e-6_f64.max(span.abs() * 1.0e-6);
        let u = (self.x[row] - self.basis.min) / span;
        let du = h / span;
        let plus = self.basis_for_unit(u + du).dot(beta);
        let minus = self.basis_for_unit(u - du).dot(beta);
        (plus - minus) / (2.0 * h)
    }
}

impl SplineRowBasis for OpenUniformSplineDesign {
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.x.len()
    }

    #[inline(always)]
    fn nparams(&self) -> usize {
        self.basis.n_basis
    }

    #[inline]
    fn for_each_row_basis(&self, row: usize, f: impl FnMut(usize, f64)) {
        self.basis_for_row(row).for_each(f);
    }
}
impl PredictorBlock for OpenUniformSplineDesign {
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.x.len()
    }

    #[inline(always)]
    fn nparams(&self) -> usize {
        self.basis.n_basis
    }

    #[inline]
    fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
        let basis = self.basis_for_row(row);
        basis.dot(beta)
    }

    #[inline]
    fn add_gradient(&self, scores: &[f64], _: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(scores.len(), self.x.len());
        debug_assert_eq!(grad.len(), self.basis.n_basis);

        for (row, score) in scores.iter().copied().enumerate() {
            if score == 0.0 {
                continue;
            }
            self.basis_for_row(row).add_scaled(score, grad);
        }
    }

    #[inline]
    fn add_weighted_gradient(
        &self,
        scores: &[f64],
        multiplier: &[f64],
        beta: &[f64],
        grad: &mut [f64],
    ) {
        debug_assert_eq!(multiplier.len(), self.x.len());
        self.add_weighted_gradient_by(scores, multiplier, beta, grad);
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
        debug_assert_eq!(scores.len(), self.x.len());
        debug_assert_eq!(grad.len(), self.basis.n_basis);

        for (row, score) in scores.iter().copied().enumerate() {
            if score == 0.0 {
                continue;
            }
            let scaled_score = score * multiplier.multiplier_at(row);
            if scaled_score == 0.0 {
                continue;
            }
            self.basis_for_row(row).add_scaled(scaled_score, grad);
        }
    }
}

impl LinearPredictorGeometry for OpenUniformSplineDesign {
    #[inline]
    fn add_weighted_gram(&self, row_weights: &[f64], out: &mut [f64]) -> Result<(), ModelError> {
        validate_gram_lengths(self.x.len(), self.basis.n_basis, row_weights, out)?;

        for (row, weight) in row_weights.iter().copied().enumerate() {
            if weight == 0.0 {
                continue;
            }
            self.basis_for_row(row)
                .add_scaled_outer(weight, self.basis.n_basis, out);
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
        validate_gram_lengths(self.x.len(), self.basis.n_basis, row_weights, out)?;

        for (row, weight) in row_weights.iter().copied().enumerate() {
            if weight == 0.0 {
                continue;
            }
            let scaled_weight = weight * multiplier.multiplier_at(row);
            if scaled_weight == 0.0 {
                continue;
            }
            self.basis_for_row(row)
                .add_scaled_outer(scaled_weight, self.basis.n_basis, out);
        }

        Ok(())
    }

    #[inline]
    fn add_t_mul_vec(&self, row_scores: &[f64], out: &mut [f64]) -> Result<(), ModelError> {
        validate_transpose_lengths(self.x.len(), self.basis.n_basis, row_scores, out)?;
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
        validate_transpose_lengths(self.x.len(), self.basis.n_basis, row_scores, out)?;
        self.add_weighted_gradient_by(row_scores, multiplier, &[], out);
        Ok(())
    }
}

#[inline]
fn validate_gram_lengths(
    nrows: usize,
    nparams: usize,
    row_weights: &[f64],
    out: &[f64],
) -> Result<(), ModelError> {
    validate_row_values_len(nrows, row_weights)?;
    let expected_values = nparams
        .checked_mul(nparams)
        .ok_or(ModelError::ArithmeticOverflow {
            context: "linear predictor geometry Gram value count",
        })?;
    if out.len() != expected_values {
        return Err(ModelError::DesignSize {
            expected_values,
            actual_values: out.len(),
        });
    }
    Ok(())
}

#[inline]
fn validate_transpose_lengths(
    nrows: usize,
    nparams: usize,
    row_scores: &[f64],
    out: &[f64],
) -> Result<(), ModelError> {
    validate_row_values_len(nrows, row_scores)?;
    if out.len() != nparams {
        return Err(ModelError::GradientLength {
            expected: nparams,
            actual: out.len(),
        });
    }
    Ok(())
}

#[inline]
const fn validate_row_values_len(nrows: usize, row_values: &[f64]) -> Result<(), ModelError> {
    if row_values.len() != nrows {
        return Err(ModelError::WeightLength {
            expected: nrows,
            actual: row_values.len(),
        });
    }
    Ok(())
}
