use gamlss_core::PredictorBlock;

use crate::SplineError;
use crate::row_basis::SplineRowBasis;

/// Natural cubic spline basis with one coefficient per knot.
#[derive(Debug, Clone, PartialEq)]
pub struct NaturalCubicSplineBasis {
    knots: Vec<f64>,
    second_derivatives: Vec<f64>,
}

impl NaturalCubicSplineBasis {
    /// Creates a natural cubic basis from strictly increasing knots.
    pub fn new(knots: Vec<f64>) -> Result<Self, SplineError> {
        validate_strict_knots(&knots)?;
        let second_derivatives = precompute_second_derivatives(&knots);
        Ok(Self {
            knots,
            second_derivatives,
        })
    }

    /// Builds uniformly spaced knots over the finite range of `x`.
    pub fn uniform_from_data(x: &[f64], n_basis: usize) -> Result<Self, SplineError> {
        if x.is_empty() {
            return Err(SplineError::EmptyInput);
        }
        if n_basis < 2 {
            return Err(SplineError::NotEnoughBasis { n_basis, degree: 1 });
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

        let step = (max - min) / (n_basis - 1) as f64;
        let knots = (0..n_basis)
            .map(|index| min + step * index as f64)
            .collect::<Vec<_>>();
        Self::new(knots)
    }

    /// Builds a predictor design for concrete `x` coordinates.
    pub fn design(&self, x: &[f64]) -> Result<NaturalCubicSplineDesign, SplineError> {
        if x.iter().any(|value| !value.is_finite()) {
            return Err(SplineError::NonFiniteValue);
        }
        Ok(NaturalCubicSplineDesign {
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

    /// Number of basis functions.
    #[must_use]
    #[inline(always)]
    pub fn n_basis(&self) -> usize {
        self.knots.len()
    }

    /// Evaluates all basis functions at `x`.
    #[must_use]
    pub fn evaluate(&self, x: f64) -> Vec<f64> {
        let mut values = vec![0.0; self.n_basis()];
        self.evaluate_into(x, &mut values);
        values
    }

    /// Writes all basis-function values at `x` into `out`.
    ///
    /// `out.len()` must equal [`Self::n_basis`].
    #[inline]
    pub fn evaluate_into(&self, x: f64, out: &mut [f64]) {
        debug_assert_eq!(out.len(), self.n_basis());

        for (basis, value) in out.iter_mut().enumerate() {
            *value = self.evaluate_one(basis, x);
        }
    }

    /// Visits non-zero basis-function values at `x` without allocating.
    #[inline]
    pub fn for_each_basis(&self, x: f64, mut f: impl FnMut(usize, f64)) {
        for basis in 0..self.n_basis() {
            let weight = self.evaluate_one(basis, x);
            if weight != 0.0 {
                f(basis, weight);
            }
        }
    }

    /// Evaluates first derivatives of all basis functions at `x`.
    #[must_use]
    pub fn evaluate_derivative(&self, x: f64) -> Vec<f64> {
        let mut values = vec![0.0; self.n_basis()];
        self.evaluate_derivative_into(x, &mut values);
        values
    }

    /// Writes first derivatives of all basis functions at `x` into `out`.
    ///
    /// `out.len()` must equal [`Self::n_basis`].
    #[inline]
    pub fn evaluate_derivative_into(&self, x: f64, out: &mut [f64]) {
        debug_assert_eq!(out.len(), self.n_basis());

        for (basis, value) in out.iter_mut().enumerate() {
            *value = self.evaluate_derivative_one(basis, x);
        }
    }

    /// Visits non-zero first derivatives at `x` without allocating.
    #[inline]
    pub fn for_each_derivative_basis(&self, x: f64, mut f: impl FnMut(usize, f64)) {
        for basis in 0..self.n_basis() {
            let weight = self.evaluate_derivative_one(basis, x);
            if weight != 0.0 {
                f(basis, weight);
            }
        }
    }

    fn evaluate_one(&self, basis: usize, x: f64) -> f64 {
        let (interval, left_extrapolate, right_extrapolate) = self.interval(x);
        if left_extrapolate || right_extrapolate {
            let edge = if left_extrapolate {
                0
            } else {
                self.knots.len() - 1
            };
            let edge_x = self.knots[edge];
            let value = f64::from(basis == edge);
            return value + (x - edge_x) * self.evaluate_derivative_one(basis, edge_x);
        }

        let x0 = self.knots[interval];
        let x1 = self.knots[interval + 1];
        let h = x1 - x0;
        let a = (x1 - x) / h;
        let b = (x - x0) / h;
        let y0 = f64::from(basis == interval);
        let y1 = f64::from(basis == interval + 1);
        let m0 = self.second_derivative(basis, interval);
        let m1 = self.second_derivative(basis, interval + 1);

        a * y0 + b * y1 + ((a * a * a - a) * m0 + (b * b * b - b) * m1) * h * h / 6.0
    }

    fn evaluate_derivative_one(&self, basis: usize, x: f64) -> f64 {
        let (interval, left_extrapolate, right_extrapolate) = self.interval(x);
        let interval = if left_extrapolate {
            0
        } else if right_extrapolate {
            self.knots.len() - 2
        } else {
            interval
        };

        let x0 = self.knots[interval];
        let x1 = self.knots[interval + 1];
        let h = x1 - x0;
        let clamped_x = x.clamp(x0, x1);
        let a = (x1 - clamped_x) / h;
        let b = (clamped_x - x0) / h;
        let y0 = f64::from(basis == interval);
        let y1 = f64::from(basis == interval + 1);
        let m0 = self.second_derivative(basis, interval);
        let m1 = self.second_derivative(basis, interval + 1);

        (y1 - y0) / h + h * ((1.0 - 3.0 * a * a) * m0 + (3.0 * b * b - 1.0) * m1) / 6.0
    }

    fn second_derivative(&self, basis: usize, knot: usize) -> f64 {
        let n = self.knots.len();
        debug_assert!(basis < n);
        debug_assert!(knot < n);
        debug_assert_eq!(self.second_derivatives.len(), n * n);
        self.second_derivatives[basis * n + knot]
    }

    fn interval(&self, x: f64) -> (usize, bool, bool) {
        let last = self.knots.len() - 1;
        if x <= self.knots[0] {
            return (0, x < self.knots[0], false);
        }
        if x >= self.knots[last] {
            return (last - 1, false, x > self.knots[last]);
        }

        let upper = self.knots.partition_point(|knot| *knot <= x);
        (upper - 1, false, false)
    }
}

/// Natural cubic spline predictor.
#[derive(Debug, Clone, PartialEq)]
pub struct NaturalCubicSplineDesign {
    x: Vec<f64>,
    basis: NaturalCubicSplineBasis,
}

impl NaturalCubicSplineDesign {
    /// Builds a design from data-derived uniform knots.
    pub fn uniform_from_data(x: &[f64], n_basis: usize) -> Result<Self, SplineError> {
        NaturalCubicSplineBasis::uniform_from_data(x, n_basis)?.design(x)
    }

    /// Returns the basis metadata.
    #[must_use]
    #[inline(always)]
    pub fn basis(&self) -> &NaturalCubicSplineBasis {
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

        let mut value = 0.0;
        self.basis
            .for_each_derivative_basis(self.x[row], |index, weight| {
                value += beta[index] * weight;
            });
        value
    }
}

impl SplineRowBasis for NaturalCubicSplineDesign {
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

impl PredictorBlock for NaturalCubicSplineDesign {
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

        let mut value = 0.0;
        self.basis.for_each_basis(self.x[row], |index, weight| {
            value += beta[index] * weight;
        });
        value
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

fn validate_strict_knots(knots: &[f64]) -> Result<(), SplineError> {
    if knots.len() < 2 {
        return Err(SplineError::NotEnoughKnots { min: 2 });
    }
    if knots
        .windows(2)
        .any(|window| !window[0].is_finite() || !window[1].is_finite() || window[0] >= window[1])
    {
        return Err(SplineError::InvalidKnots);
    }
    Ok(())
}

fn precompute_second_derivatives(knots: &[f64]) -> Vec<f64> {
    let n = knots.len();
    let mut second_derivatives = Vec::with_capacity(n * n);
    for basis in 0..n {
        second_derivatives.extend(natural_basis_second_derivatives(knots, basis));
    }
    debug_assert_eq!(second_derivatives.len(), n * n);
    second_derivatives
}

fn natural_basis_second_derivatives(x: &[f64], basis: usize) -> Vec<f64> {
    let n = x.len();
    debug_assert!(basis < n);

    let mut second = vec![0.0; n];
    if n <= 2 {
        return second;
    }

    let m = n - 2;
    let mut lower = vec![0.0; m];
    let mut diag = vec![0.0; m];
    let mut upper = vec![0.0; m];
    let mut rhs = vec![0.0; m];

    for row in 0..m {
        let i = row + 1;
        let h0 = x[i] - x[i - 1];
        let h1 = x[i + 1] - x[i];
        lower[row] = h0;
        diag[row] = 2.0 * (h0 + h1);
        upper[row] = h1;
        let y_prev = f64::from(i - 1 == basis);
        let y = f64::from(i == basis);
        let y_next = f64::from(i + 1 == basis);
        rhs[row] = 6.0 * ((y_next - y) / h1 - (y - y_prev) / h0);
    }

    for row in 1..m {
        let factor = lower[row] / diag[row - 1];
        diag[row] -= factor * upper[row - 1];
        rhs[row] -= factor * rhs[row - 1];
    }

    let mut interior = vec![0.0; m];
    interior[m - 1] = rhs[m - 1] / diag[m - 1];
    for row in (0..m - 1).rev() {
        interior[row] = (rhs[row] - upper[row] * interior[row + 1]) / diag[row];
    }

    second[1..n - 1].copy_from_slice(&interior);
    second
}
