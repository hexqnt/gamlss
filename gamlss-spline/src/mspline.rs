use std::ops::Range;

use gamlss_core::{LinearPredictorGeometry, ModelError, PredictorBlock, RowMultiplier};

use crate::local::{LocalBasis, bspline_active_range, bspline_local_basis, bspline_value};
use crate::prepared::PreparedContiguousGeometry;
use crate::row_basis::SplineRowBasis;
use crate::validation::finite_data_range;
use crate::{OnDemandSplineDesign, SplineError};

/// M-spline basis with normalized non-negative basis functions.
///
/// For degree $p$ and the B-spline $B_{i,p}$ on the same knot vector $\boldsymbol t$, this implementation uses
///
/// $$
/// M_{i,p}(x)
/// =\frac{p+1}{t_{i+p+1}-t_i}B_{i,p}(x).
/// $$
///
/// A zero denominator produces the zero basis function. Otherwise $M_{i,p}(x)\ge0$ and its integral over $\lbrack t_i,t_{i+p+1}\rbrack$ is one. The symbols $p$ and $\boldsymbol t$ correspond to [`MSplineBasis::degree`] and [`MSplineBasis::knots`]; degrees above three are rejected.
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, PartialEq)]
pub struct MSplineBasis {
    knots: Vec<f64>,
    degree: usize,
    n_basis: usize,
}

impl MSplineBasis {
    /// Creates an M-spline basis from a finite nondecreasing knot vector.
    pub fn new(knots: Vec<f64>, degree: usize) -> Result<Self, SplineError> {
        if degree > 3 {
            return Err(SplineError::UnsupportedDegree { degree });
        }
        if knots.len() <= degree + 1 {
            return Err(SplineError::NotEnoughKnots { min: degree + 2 });
        }
        if knots
            .windows(2)
            .any(|window| !window[0].is_finite() || !window[1].is_finite() || window[0] > window[1])
        {
            return Err(SplineError::InvalidKnots);
        }
        let n_basis = knots.len() - degree - 1;
        Ok(Self {
            knots,
            degree,
            n_basis,
        })
    }

    /// Builds an open-uniform knot vector from data.
    #[allow(clippy::cast_precision_loss)]
    pub fn open_uniform_from_data(
        x: &[f64],
        n_basis: usize,
        degree: usize,
    ) -> Result<Self, SplineError> {
        if x.is_empty() {
            return Err(SplineError::EmptyInput);
        }
        if n_basis <= degree {
            return Err(SplineError::NotEnoughBasis { n_basis, degree });
        }

        let (min, max) = finite_data_range(x)?;

        let interior = n_basis.saturating_sub(degree + 1);
        let mut knots = Vec::with_capacity(n_basis + degree + 1);
        knots.extend(std::iter::repeat_n(min, degree + 1));
        for index in 1..=interior {
            let fraction = index as f64 / (interior + 1) as f64;
            knots.push(min + fraction * (max - min));
        }
        knots.extend(std::iter::repeat_n(max, degree + 1));
        Self::new(knots, degree)
    }

    /// Builds a predictor design.
    pub fn design(&self, x: &[f64]) -> Result<MSplineDesign, SplineError> {
        let prepared = PreparedContiguousGeometry::try_from_basis(x, self, self.degree + 1)?;
        Ok(MSplineDesign {
            x: x.into(),
            prepared,
            basis: self.clone(),
        })
    }

    /// Builds a low-memory predictor that reevaluates row geometry on demand.
    pub fn on_demand_design(&self, x: &[f64]) -> Result<OnDemandSplineDesign<Self>, SplineError> {
        OnDemandSplineDesign::new(x, self.clone())
    }

    /// Knot vector.
    #[must_use]
    #[inline]
    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    /// Degree.
    #[must_use]
    #[inline]
    pub const fn degree(&self) -> usize {
        self.degree
    }

    /// Number of basis functions.
    #[must_use]
    #[inline]
    pub const fn n_basis(&self) -> usize {
        self.n_basis
    }

    /// Evaluates all basis functions at `x`.
    #[must_use]
    pub fn evaluate(&self, x: f64) -> Vec<f64> {
        let mut values = vec![0.0; self.n_basis];
        self.evaluate_into(x, &mut values);
        values
    }

    /// Writes all basis-function values at `x` into `out`.
    ///
    /// `out.len()` must equal [`Self::n_basis`].
    #[inline]
    pub fn evaluate_into(&self, x: f64, out: &mut [f64]) {
        debug_assert_eq!(out.len(), self.n_basis);
        out.fill(0.0);
        self.for_each_basis(x, |index, weight| out[index] = weight);
    }

    /// Evaluates first derivatives of all basis functions at `x`.
    #[must_use]
    pub fn evaluate_derivative(&self, x: f64) -> Vec<f64> {
        let mut values = vec![0.0; self.n_basis];
        self.evaluate_derivative_into(x, &mut values);
        values
    }

    /// Writes first derivatives of all basis functions at `x` into `out`.
    ///
    /// `out.len()` must equal [`Self::n_basis`].
    #[inline]
    pub fn evaluate_derivative_into(&self, x: f64, out: &mut [f64]) {
        debug_assert_eq!(out.len(), self.n_basis);
        out.fill(0.0);
        self.for_each_derivative_basis(x, |index, weight| out[index] = weight);
    }

    /// Visits non-zero basis-function values at `x` without allocating.
    #[inline]
    pub fn for_each_basis(&self, x: f64, mut f: impl FnMut(usize, f64)) {
        self.local_basis(x).for_each(&mut f);
    }

    /// Visits non-zero first derivatives at `x` without allocating.
    #[inline]
    pub fn for_each_derivative_basis(&self, x: f64, mut f: impl FnMut(usize, f64)) {
        if let Some(active) = bspline_active_range(&self.knots, self.n_basis, self.degree, x) {
            for index in active {
                let weight = self.evaluate_derivative_one(index, x);
                if weight != 0.0 {
                    f(index, weight);
                }
            }
        }
    }

    /// Evaluates one basis function at `x`.
    #[must_use]
    #[inline]
    #[allow(clippy::cast_precision_loss)]
    pub fn evaluate_one(&self, index: usize, x: f64) -> f64 {
        let denom = self.knots[index + self.degree + 1] - self.knots[index];
        if denom <= 0.0 {
            return 0.0;
        }
        (self.degree + 1) as f64 * bspline_value(&self.knots, self.n_basis, index, self.degree, x)
            / denom
    }

    /// Evaluates the first derivative of one basis function at `x`.
    #[must_use]
    #[inline]
    #[allow(clippy::cast_precision_loss)]
    pub fn evaluate_derivative_one(&self, index: usize, x: f64) -> f64 {
        if self.degree == 0 {
            return 0.0;
        }

        let denom = self.knots[index + self.degree + 1] - self.knots[index];
        if denom <= 0.0 {
            return 0.0;
        }

        let scale = (self.degree + 1) as f64 / denom;
        scale * bspline_derivative_value(&self.knots, self.n_basis, index, self.degree, x)
    }

    #[inline]
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn local_basis(&self, x: f64) -> LocalBasis {
        let mut local = LocalBasis::default();
        bspline_local_basis(&self.knots, self.n_basis, self.degree, x).for_each(|index, weight| {
            let denom = self.knots[index + self.degree + 1] - self.knots[index];
            if denom > 0.0 {
                local.push_nonzero(index, (self.degree + 1) as f64 * weight / denom);
            }
        });
        local
    }
}

/// M-spline predictor with compact prepared row geometry.
///
/// Each row stores no more than `degree + 1` contiguous weights. Use
/// [`MSplineBasis::on_demand_design`] for the lower-memory alternative.
#[derive(Debug, Clone, PartialEq)]
pub struct MSplineDesign {
    x: Box<[f64]>,
    prepared: PreparedContiguousGeometry,
    basis: MSplineBasis,
}

impl MSplineDesign {
    /// Returns the basis metadata.
    #[must_use]
    #[inline]
    pub const fn basis(&self) -> &MSplineBasis {
        &self.basis
    }

    /// Input coordinates.
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

    /// Predictor derivative with respect to `x`.
    #[must_use]
    #[inline]
    pub fn eta_derivative_row(&self, row: usize, beta: &[f64]) -> f64 {
        debug_assert!(row < self.x.len());
        debug_assert_eq!(beta.len(), self.basis.n_basis());

        let mut value = 0.0;
        self.basis
            .for_each_derivative_basis(self.x[row], |index, weight| {
                value = weight.mul_add(beta[index], value);
            });
        value
    }
}

impl SplineRowBasis for MSplineDesign {
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

impl PredictorBlock for MSplineDesign {
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
        debug_assert!(row < self.x.len());
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

impl LinearPredictorGeometry for MSplineDesign {
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

#[allow(clippy::cast_precision_loss)]
fn bspline_derivative_value(
    knots: &[f64],
    n_basis: usize,
    index: usize,
    degree: usize,
    x: f64,
) -> f64 {
    debug_assert!(degree > 0);

    let degree_f64 = degree as f64;
    let mut value = 0.0;
    let left_denom = knots[index + degree] - knots[index];
    if left_denom > 0.0 {
        value = (degree_f64 / left_denom)
            .mul_add(bspline_value(knots, n_basis, index, degree - 1, x), value);
    }

    let right_denom = knots[index + degree + 1] - knots[index + 1];
    if right_denom > 0.0 {
        value = (-degree_f64 / right_denom).mul_add(
            bspline_value(knots, n_basis, index + 1, degree - 1, x),
            value,
        );
    }

    value
}
