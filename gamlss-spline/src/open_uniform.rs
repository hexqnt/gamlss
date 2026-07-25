use std::ops::Range;

use gamlss_core::{LinearPredictorGeometry, ModelError, PredictorBlock, RowMultiplier};

use crate::geometry::{validate_gram_lengths, validate_transpose_lengths};
use crate::local::{
    LocalBasis, PreparedLocalBasis, open_uniform_local_basis, open_uniform_local_basis_derivative,
    prepare_open_uniform_local_basis,
};
use crate::row_basis::SplineRowBasis;
use crate::validation::{finite_data_range, validate_finite_range};
use crate::{OnDemandSplineDesign, SplineError, SplineOrder};

/// Metadata for an open-uniform spline predictor.
///
/// Stores only the basis shape and scaling range, so it can be reused for building designs on training and new data.
///
/// Let $a$ be [`OpenUniformSplineBasis::min`], $b$ be [`OpenUniformSplineBasis::max`], $K$ be [`OpenUniformSplineBasis::n_basis`], and $p=\mathtt{order.degree()}$. Coordinates are normalized as $u=(x-a)/(b-a)$, and $q=K-p$ equal intervals define the clamped knot vector
///
/// $$
/// t_i=
/// \begin{cases}
/// 0, & i\le p,\\\\
/// (i-p)/q, & p<i<K,\\\\
/// 1, & i\ge K.
/// \end{cases}
/// $$
///
/// For $0<u<1$, the predictor is $\eta(x)=\sum_{j=0}^{K-1}\beta_jB_{j,p}(u)$ and only $p+1$ adjacent B-spline weights are evaluated. At and beyond the normalized boundaries, the implementation uses the following linear continuation:
///
/// $$
/// \eta(x(u))=
/// \begin{cases}
/// \beta_0+pq\\,u(\beta_1-\beta_0), & u\le0,\\\\
/// \beta_{K-1}+pq\\,(u-1)(\beta_{K-1}-\beta_{K-2}), & u\ge1.
/// \end{cases}
/// $$
#[allow(clippy::doc_markdown)]
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

        let (min, max) = finite_data_range(x)?;
        Self::new(min, max, n_basis, order)
    }

    /// Builds metadata with an explicit finite range.
    ///
    /// # Errors
    ///
    /// Returns an error if the boundaries or their span are not finite,
    /// `min >= max`, or `n_basis` is insufficient for `order`.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(
        min: f64,
        max: f64,
        n_basis: usize,
        order: SplineOrder,
    ) -> Result<Self, SplineError> {
        validate_finite_range(min, max)?;
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
        let prepared_rows = x
            .iter()
            .copied()
            .map(|value| {
                let u = self.unit_coordinate(value)?;
                Ok::<_, SplineError>(prepare_open_uniform_local_basis(
                    u,
                    self.order,
                    self.n_basis,
                    self.n_intervals,
                ))
            })
            .collect::<Result<_, _>>()?;

        Ok(OpenUniformSplineDesign {
            x: x.into(),
            prepared_rows,
            basis: *self,
        })
    }

    /// Builds a low-memory design that recomputes local row geometry on demand.
    ///
    /// Unlike [`Self::design`], this retains only the coordinates and basis
    /// metadata. Use it when the additional prepared-row cache is less
    /// important than minimizing persistent memory.
    ///
    /// # Errors
    ///
    /// Returns [`SplineError::NonFiniteValue`] if `x` contains a non-finite
    /// coordinate.
    pub fn on_demand_design(&self, x: &[f64]) -> Result<OnDemandSplineDesign<Self>, SplineError> {
        OnDemandSplineDesign::new(x, *self)
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
        self.local_basis_for_unit(self.unit_coordinate(x)?)
            .for_each(f);
        Ok(())
    }

    /// Lower boundary of the basis range.
    #[must_use]
    #[inline]
    pub const fn min(&self) -> f64 {
        self.min
    }

    /// Upper boundary of the basis range.
    #[must_use]
    #[inline]
    pub const fn max(&self) -> f64 {
        self.max
    }

    /// Number of spline coefficients.
    #[must_use]
    #[inline]
    pub const fn n_basis(&self) -> usize {
        self.n_basis
    }

    /// Spline order.
    #[must_use]
    #[inline]
    pub const fn order(&self) -> SplineOrder {
        self.order
    }

    #[inline]
    fn span(&self) -> f64 {
        self.max - self.min
    }

    #[inline]
    pub(crate) fn unit_coordinate(&self, x: f64) -> Result<f64, SplineError> {
        if !x.is_finite() {
            return Err(SplineError::NonFiniteValue);
        }
        let u = (x - self.min) / self.span();
        if !u.is_finite() {
            return Err(SplineError::NonFiniteValue);
        }
        Ok(u)
    }

    #[inline]
    pub(crate) fn local_basis_for_unit(&self, u: f64) -> LocalBasis {
        open_uniform_local_basis(u, self.order, self.n_basis, self.n_intervals)
    }
}

/// Open-uniform spline predictor with compact prepared row geometry.
///
/// Construction evaluates every local basis once. Each retained row stores its original `f64` coordinate plus compact geometry containing one `usize` start index and four `f64` weights; it never materializes an `nrows × n_basis` matrix. For repeated model passes without this row cache, use [`OpenUniformSplineBasis::on_demand_design`]. For one-shot evaluation, use [`OpenUniformSplineBasis::for_each_value_basis`] or [`crate::SplineBasis1d`].
///
/// See [`OpenUniformSplineBasis`] for the knot construction and extrapolation rule.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenUniformSplineDesign {
    x: Box<[f64]>,
    prepared_rows: Box<[PreparedLocalBasis]>,
    basis: OpenUniformSplineBasis,
}

impl OpenUniformSplineDesign {
    /// Builds an open-uniform spline design from a data range.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is empty, contains non-finite values, has
    /// a degenerate range, or `n_basis` is insufficient for `order`.
    pub fn from_data(x: &[f64], n_basis: usize, order: SplineOrder) -> Result<Self, SplineError> {
        OpenUniformSplineBasis::from_data(x, n_basis, order)?.design(x)
    }

    /// Builds an open-uniform spline design with an explicit finite range.
    ///
    /// `min` and `max` must be finite and `min < max`.
    ///
    /// # Errors
    ///
    /// Returns an error if `x` or the range contains non-finite values,
    /// `min >= max`, or `n_basis` is insufficient for `order`.
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
    #[inline]
    pub const fn n_basis(&self) -> usize {
        self.basis.n_basis()
    }

    /// Basis metadata suitable for building a design on new data.
    #[must_use]
    #[inline]
    pub const fn basis(&self) -> OpenUniformSplineBasis {
        self.basis
    }

    /// Returns the original coordinates of the design.
    #[must_use]
    #[inline]
    pub fn x(&self) -> &[f64] {
        &self.x
    }

    #[inline]
    fn prepared_basis_for_row(&self, row: usize) -> &PreparedLocalBasis {
        &self.prepared_rows[row]
    }

    #[inline]
    fn active_weights_for_row(&self, row: usize) -> Range<usize> {
        let width = self.basis.order.degree() + 1;
        if self.x[row] <= self.basis.min {
            0..2
        } else if self.x[row] >= self.basis.max {
            width - 2..width
        } else {
            0..width
        }
    }

    #[inline]
    fn derivative_basis_for_unit(&self, u: f64) -> LocalBasis {
        open_uniform_local_basis_derivative(
            u,
            self.basis.order,
            self.basis.n_basis,
            self.basis.n_intervals,
        )
    }

    /// Derivative of the predictor contribution with respect to the original
    /// coordinate `x`.
    ///
    /// The normalized-basis derivative is scaled by $du/dx=1/(\text{max}-\text{min})$.
    #[must_use]
    #[inline]
    pub fn eta_derivative_row(&self, row: usize, beta: &[f64]) -> f64 {
        let span = self.basis.span();
        let u = (self.x[row] - self.basis.min) / span;
        self.derivative_basis_for_unit(u).dot(beta) / span
    }
}

impl SplineRowBasis for OpenUniformSplineDesign {
    #[inline]
    fn nrows(&self) -> usize {
        self.x.len()
    }

    #[inline]
    fn nparams(&self) -> usize {
        self.basis.n_basis
    }

    #[inline]
    fn for_each_row_basis(&self, row: usize, f: impl FnMut(usize, f64)) {
        let active = self.active_weights_for_row(row);
        self.prepared_basis_for_row(row)
            .for_each_contiguous(active, f);
    }
}
impl PredictorBlock for OpenUniformSplineDesign {
    #[inline]
    fn nrows(&self) -> usize {
        self.x.len()
    }

    #[inline]
    fn nparams(&self) -> usize {
        self.basis.n_basis
    }

    #[inline]
    fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
        let active = self.active_weights_for_row(row);
        self.prepared_basis_for_row(row)
            .dot_contiguous(active, beta)
    }

    #[inline]
    fn zero_beta_constant_contribution(&self) -> Option<f64> {
        Some(0.0)
    }

    #[inline]
    fn add_gradient_range(&self, rows: Range<usize>, scores: &[f64], _: &[f64], grad: &mut [f64]) {
        debug_assert!(rows.end <= self.x.len());
        debug_assert_eq!(scores.len(), rows.len());
        debug_assert_eq!(grad.len(), self.basis.n_basis);

        for (offset, score) in scores.iter().copied().enumerate() {
            if score == 0.0 {
                continue;
            }
            let row = rows.start + offset;
            let active = self.active_weights_for_row(row);
            self.prepared_basis_for_row(row)
                .add_scaled_contiguous(active, score, grad);
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
        debug_assert_eq!(grad.len(), self.basis.n_basis);

        for (offset, score) in scores.iter().copied().enumerate() {
            if score == 0.0 {
                continue;
            }
            let row = rows.start + offset;
            let scaled_score = score * multiplier.multiplier_at(row);
            if scaled_score == 0.0 {
                continue;
            }
            let active = self.active_weights_for_row(row);
            self.prepared_basis_for_row(row)
                .add_scaled_contiguous(active, scaled_score, grad);
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
            let active = self.active_weights_for_row(row);
            self.prepared_basis_for_row(row)
                .add_scaled_outer_contiguous(active, weight, self.basis.n_basis, out);
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
            let active = self.active_weights_for_row(row);
            self.prepared_basis_for_row(row)
                .add_scaled_outer_contiguous(active, scaled_weight, self.basis.n_basis, out);
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
impl OnDemandSplineDesign<OpenUniformSplineBasis> {
    /// Derivative of the predictor contribution with respect to the original
    /// coordinate `x`.
    #[must_use]
    #[inline]
    pub fn eta_derivative_row(&self, row: usize, beta: &[f64]) -> f64 {
        debug_assert!(row < self.x().len());
        debug_assert_eq!(beta.len(), self.n_basis());

        let basis = self.basis();
        let span = basis.span();
        let u = (self.x()[row] - basis.min) / span;
        open_uniform_local_basis_derivative(u, basis.order, basis.n_basis, basis.n_intervals)
            .dot(beta)
            / span
    }
}
