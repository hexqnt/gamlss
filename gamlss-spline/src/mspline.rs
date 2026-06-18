use gamlss_core::PredictorBlock;

use crate::SplineError;
use crate::row_basis::SplineRowBasis;

/// M-spline basis with normalized non-negative basis functions.
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

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for value in x.iter().copied() {
            if !value.is_finite() {
                return Err(SplineError::NonFiniteValue);
            }
            min = min.min(value);
            max = max.max(value);
        }
        if min >= max {
            return Err(SplineError::InvalidRange);
        }

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
        if x.iter().any(|value| !value.is_finite()) {
            return Err(SplineError::NonFiniteValue);
        }
        Ok(MSplineDesign {
            x: x.to_vec(),
            basis: self.clone(),
        })
    }

    /// Knot vector.
    #[must_use]
    #[inline(always)]
    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    /// Degree.
    #[must_use]
    #[inline(always)]
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Number of basis functions.
    #[must_use]
    #[inline(always)]
    pub fn n_basis(&self) -> usize {
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

        for (index, value) in out.iter_mut().enumerate() {
            *value = self.evaluate_one(index, x);
        }
    }

    /// Visits non-zero basis-function values at `x` without allocating.
    #[inline]
    pub fn for_each_basis(&self, x: f64, mut f: impl FnMut(usize, f64)) {
        for index in 0..self.n_basis {
            let weight = self.evaluate_one(index, x);
            if weight != 0.0 {
                f(index, weight);
            }
        }
    }

    /// Evaluates one basis function at `x`.
    #[must_use]
    #[inline]
    pub fn evaluate_one(&self, index: usize, x: f64) -> f64 {
        let denom = self.knots[index + self.degree + 1] - self.knots[index];
        if denom <= 0.0 {
            return 0.0;
        }
        (self.degree + 1) as f64 * bspline_value(&self.knots, self.n_basis, index, self.degree, x)
            / denom
    }
}

/// M-spline predictor.
#[derive(Debug, Clone, PartialEq)]
pub struct MSplineDesign {
    x: Vec<f64>,
    basis: MSplineBasis,
}

impl MSplineDesign {
    /// Returns the basis metadata.
    #[must_use]
    #[inline(always)]
    pub fn basis(&self) -> &MSplineBasis {
        &self.basis
    }

    /// Input coordinates.
    #[must_use]
    #[inline(always)]
    pub fn x(&self) -> &[f64] {
        &self.x
    }

    /// Number of spline coefficients.
    #[must_use]
    #[inline(always)]
    pub fn n_basis(&self) -> usize {
        self.basis.n_basis()
    }

    /// Predictor derivative with respect to `x`.
    #[must_use]
    #[inline]
    pub fn eta_derivative_row(&self, row: usize, beta: &[f64]) -> f64 {
        debug_assert!(row < self.x.len());
        debug_assert_eq!(beta.len(), self.basis.n_basis());

        let h = 1.0e-6;
        let x = self.x[row];
        let plus = self.dot_at(x + h, beta);
        let minus = self.dot_at(x - h, beta);
        (plus - minus) / (2.0 * h)
    }

    #[inline]
    fn dot_at(&self, x: f64, beta: &[f64]) -> f64 {
        let mut value = 0.0;
        self.basis.for_each_basis(x, |index, weight| {
            value += beta[index] * weight;
        });
        value
    }
}

impl PredictorBlock for MSplineDesign {
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.x.len()
    }

    #[inline(always)]
    fn nparams(&self) -> usize {
        self.basis.n_basis()
    }

    #[inline]
    fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
        debug_assert!(row < self.x.len());
        debug_assert_eq!(beta.len(), self.basis.n_basis());

        self.dot_at(self.x[row], beta)
    }

    #[inline]
    fn add_gradient(&self, scores: &[f64], _: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(scores.len(), self.x.len());
        debug_assert_eq!(grad.len(), self.basis.n_basis());

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
        debug_assert_eq!(scores.len(), self.x.len());
        debug_assert_eq!(multiplier.len(), self.x.len());
        debug_assert_eq!(grad.len(), self.basis.n_basis());

        for (row, (&score, &multiplier)) in scores.iter().zip(multiplier).enumerate() {
            self.for_each_row_basis(row, |index, weight| {
                grad[index] += score * multiplier * weight;
            });
        }
    }
}

impl SplineRowBasis for MSplineDesign {
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.x.len()
    }

    #[inline(always)]
    fn nparams(&self) -> usize {
        self.basis.n_basis()
    }

    #[inline]
    fn for_each_row_basis(&self, row: usize, mut f: impl FnMut(usize, f64)) {
        self.basis.for_each_basis(self.x[row], &mut f);
    }
}

pub(crate) fn bspline_value(
    knots: &[f64],
    n_basis: usize,
    index: usize,
    degree: usize,
    x: f64,
) -> f64 {
    if degree == 0 {
        let left = knots[index];
        let right = knots[index + 1];
        let is_last_basis = index + 1 == n_basis;
        if (left <= x && x < right) || (is_last_basis && x == right) {
            1.0
        } else {
            0.0
        }
    } else {
        let mut value = 0.0;
        let left_denom = knots[index + degree] - knots[index];
        if left_denom > 0.0 {
            value += (x - knots[index]) / left_denom
                * bspline_value(knots, n_basis, index, degree - 1, x);
        }

        let right_denom = knots[index + degree + 1] - knots[index + 1];
        if right_denom > 0.0 {
            value += (knots[index + degree + 1] - x) / right_denom
                * bspline_value(knots, n_basis, index + 1, degree - 1, x);
        }
        value
    }
}
