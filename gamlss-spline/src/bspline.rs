use std::ops::Range;

use gamlss_core::{
    DenseDesign, LinearPredictorGeometry, ModelError, PredictorBlock, RowMultiplier,
};

use crate::local::{bspline_active_range, bspline_local_basis, bspline_value};
use crate::prepared::PreparedContiguousGeometry;
use crate::validation::validate_coordinates;
use crate::{KnotPlacement, OnDemandSplineDesign, OpenKnotVector, SplineError, SplineRowBasis};

/// B-spline basis with degree $p$ and non-decreasing knot vector $\boldsymbol{t}=(t_0,\ldots,t_{M-1})$.
///
/// The degree-zero basis is $B_{i,0}(x)=\mathbf{1}\\!\left\\{t_i\le x<t_{i+1}\right\\}$. Higher degrees use the Cox--de Boor recursion
///
/// $$
/// B_{i,p}(x)
/// =\frac{x-t_i}{t_{i+p}-t_i}B_{i,p-1}(x)
/// \mathbin{+}\frac{t_{i+p+1}-x}{t_{i+p+1}-t_{i+1}}B_{i+1,p-1}(x).
/// $$
///
/// Here $\mathbf{1}_A$ is the indicator of set $A$. A recurrence term with a zero denominator contributes zero. At the right boundary the implementation closes the last degree-zero interval so that an open basis includes its final endpoint. The code fields satisfy $p=\mathtt{degree}$, $M=\mathtt{knots.len()}$, and [`BSplineBasis::n_basis`] returns $M-p-1$.
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, PartialEq)]
pub struct BSplineBasis {
    degree: usize,
    knots: Vec<f64>,
}

impl BSplineBasis {
    /// Creates a basis from a ready-made knot vector.
    ///
    /// The knot vector must be finite, non-decreasing and long enough for the
    /// chosen degree.
    pub fn new(degree: usize, knots: Vec<f64>) -> Result<Self, SplineError> {
        if knots.len() <= degree + 1 {
            return Err(SplineError::NotEnoughBasis { n_basis: 0, degree });
        }

        if knots
            .windows(2)
            .any(|window| !window[0].is_finite() || !window[1].is_finite() || window[0] > window[1])
        {
            return Err(SplineError::InvalidKnots);
        }

        Ok(Self { degree, knots })
    }

    /// Builds an open uniform B-spline basis over a data range.
    #[allow(clippy::cast_precision_loss)]
    pub fn open_uniform_from_data(
        x: &[f64],
        n_basis: usize,
        degree: usize,
    ) -> Result<Self, SplineError> {
        Self::open_from_data(x, n_basis, degree, KnotPlacement::Uniform)
    }

    /// Builds an open B-spline basis with a persisted knot-placement policy.
    pub fn open_from_data(
        x: &[f64],
        n_basis: usize,
        degree: usize,
        placement: KnotPlacement,
    ) -> Result<Self, SplineError> {
        Self::from_open_knots(OpenKnotVector::from_data(x, n_basis, degree, placement)?)
    }

    /// Builds an open B-spline basis using weighted empirical quantiles.
    pub fn open_from_weighted_data(
        x: &[f64],
        weights: &[f64],
        n_basis: usize,
        degree: usize,
    ) -> Result<Self, SplineError> {
        Self::from_open_knots(OpenKnotVector::from_weighted_data(
            x, weights, n_basis, degree,
        )?)
    }

    /// Builds a basis from persisted open-knot metadata.
    pub fn from_open_knots(knots: OpenKnotVector) -> Result<Self, SplineError> {
        Self::new(knots.degree(), knots.into_knots())
    }

    /// Spline degree.
    #[must_use]
    pub const fn degree(&self) -> usize {
        self.degree
    }

    /// Knot vector.
    #[must_use]
    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    /// Number of basis functions.
    #[must_use]
    pub const fn n_basis(&self) -> usize {
        self.knots.len() - self.degree - 1
    }

    /// Builds a predictor that retains `x` and evaluates basis rows on demand.
    ///
    /// This uses `O(nrows + n_basis)` storage instead of the
    /// `O(nrows * n_basis)` storage used by [`Self::design_matrix`]. The basis
    /// metadata is cloned once; row evaluation does not allocate.
    ///
    /// # Errors
    ///
    /// Returns [`SplineError::NonFiniteValue`] if `x` contains a non-finite
    /// coordinate.
    pub fn on_demand_design(&self, x: &[f64]) -> Result<OnDemandSplineDesign<Self>, SplineError> {
        OnDemandSplineDesign::new(x, self.clone())
    }

    /// Builds a compact prepared predictor for repeated model passes.
    ///
    /// Each row stores only its contiguous non-zero coefficient range and
    /// weights. Storage is `O(nrows * (degree + 1))`, independent of the total
    /// basis count for a fixed degree.
    ///
    /// # Errors
    ///
    /// Returns [`SplineError::NonFiniteValue`] for a non-finite coordinate or
    /// [`SplineError::ParameterOverflow`] if the row storage size overflows.
    pub fn design(&self, x: &[f64]) -> Result<BSplineDesign, SplineError> {
        let prepared = PreparedContiguousGeometry::try_from_basis(x, self, self.degree + 1)?;
        Ok(BSplineDesign {
            x: x.into(),
            prepared,
            basis: self.clone(),
        })
    }

    /// Values of all basis functions at point `x`.
    #[must_use]
    pub fn evaluate(&self, x: f64) -> Vec<f64> {
        let mut values = vec![0.0; self.n_basis()];
        self.evaluate_into(x, &mut values);
        values
    }

    /// Writes all basis-function values at `x` into `out`.
    ///
    /// `out.len()` must equal [`Self::n_basis`].
    pub fn evaluate_into(&self, x: f64, out: &mut [f64]) {
        self.fill_values(x, out);
    }

    /// Visits non-zero basis-function values at `x` without allocating.
    #[inline]
    #[allow(clippy::float_cmp)]
    pub fn for_each_basis(&self, x: f64, mut f: impl FnMut(usize, f64)) {
        if self.degree <= 3 {
            bspline_local_basis(&self.knots, self.n_basis(), self.degree, x).for_each(f);
        } else if let Some(active) =
            bspline_active_range(&self.knots, self.n_basis(), self.degree, x)
        {
            for index in active {
                let weight = bspline_value(&self.knots, self.n_basis(), index, self.degree, x);
                if weight != 0.0 {
                    f(index, weight);
                }
            }
        }
    }

    /// Dense design matrix where each row contains `evaluate(x_i)`.
    pub fn design_matrix(&self, x: &[f64]) -> Result<DenseDesign, SplineError> {
        validate_coordinates(x)?;

        let n_basis = self.n_basis();
        let mut values = Vec::with_capacity(x.len() * n_basis);
        values.resize(x.len() * n_basis, 0.0);
        for (row, value) in values.chunks_exact_mut(n_basis).zip(x.iter().copied()) {
            self.fill_values(value, row);
        }

        Ok(DenseDesign::from_row_major(x.len(), n_basis, values)?)
    }

    fn fill_values(&self, x: f64, out: &mut [f64]) {
        debug_assert_eq!(out.len(), self.n_basis());

        if x.is_finite() {
            out.fill(0.0);
            self.for_each_basis(x, |index, weight| out[index] = weight);
        } else {
            for (index, value) in out.iter_mut().enumerate() {
                *value = bspline_value(&self.knots, self.n_basis(), index, self.degree, x);
            }
        }
    }
}

/// General-knot B-spline predictor with compact prepared row geometry.
///
/// Unlike [`BSplineBasis::design_matrix`], this representation retains only
/// the at-most-`degree + 1` active weights per observation. Use
/// [`BSplineBasis::on_demand_design`] when even that row cache is undesirable.
#[derive(Debug, Clone, PartialEq)]
pub struct BSplineDesign {
    x: Box<[f64]>,
    prepared: PreparedContiguousGeometry,
    basis: BSplineBasis,
}

impl BSplineDesign {
    /// Basis metadata suitable for evaluating new coordinates.
    #[must_use]
    #[inline]
    pub const fn basis(&self) -> &BSplineBasis {
        &self.basis
    }

    /// Original input coordinates.
    #[must_use]
    #[inline]
    pub fn x(&self) -> &[f64] {
        &self.x
    }

    /// Number of spline coefficients.
    #[must_use]
    #[inline]
    pub const fn n_basis(&self) -> usize {
        self.basis.n_basis()
    }
}

impl SplineRowBasis for BSplineDesign {
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

impl PredictorBlock for BSplineDesign {
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
        debug_assert_eq!(beta.len(), self.basis.n_basis());
        self.prepared.dot(row, beta)
    }

    #[inline]
    fn zero_beta_constant_contribution(&self) -> Option<f64> {
        Some(0.0)
    }

    #[inline]
    fn add_gradient_range(&self, rows: Range<usize>, scores: &[f64], _: &[f64], grad: &mut [f64]) {
        self.prepared
            .add_gradient_range(self.basis.n_basis(), rows, scores, grad);
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
        self.prepared.add_weighted_gradient_by_range(
            self.basis.n_basis(),
            rows,
            scores,
            multiplier,
            grad,
        );
    }
}

impl LinearPredictorGeometry for BSplineDesign {
    #[inline]
    fn add_weighted_gram(&self, row_weights: &[f64], out: &mut [f64]) -> Result<(), ModelError> {
        self.prepared
            .add_weighted_gram(self.basis.n_basis(), row_weights, out)
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
        self.prepared
            .add_weighted_gram_by(self.basis.n_basis(), row_weights, multiplier, out)
    }

    #[inline]
    fn add_t_mul_vec(&self, row_scores: &[f64], out: &mut [f64]) -> Result<(), ModelError> {
        self.prepared
            .add_t_mul_vec(self.basis.n_basis(), row_scores, out)
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
        self.prepared
            .add_t_mul_vec_by(self.basis.n_basis(), row_scores, multiplier, out)
    }
}

/// Helper function for open uniform P-spline design matrix.
pub fn pspline_design(
    x: &[f64],
    n_basis: usize,
    degree: usize,
) -> Result<DenseDesign, SplineError> {
    BSplineBasis::open_uniform_from_data(x, n_basis, degree)?.design_matrix(x)
}
