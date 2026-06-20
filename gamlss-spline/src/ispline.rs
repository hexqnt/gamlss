use gamlss_core::PredictorBlock;

use crate::SplineError;
use crate::mspline::MSplineBasis;
use crate::row_basis::SplineRowBasis;

/// I-spline basis built by integrating normalized M-splines.
#[derive(Debug, Clone, PartialEq)]
pub struct ISplineBasis {
    mspline: MSplineBasis,
}

impl ISplineBasis {
    /// Creates an I-spline basis from a finite nondecreasing knot vector.
    pub fn new(knots: Vec<f64>, degree: usize) -> Result<Self, SplineError> {
        Ok(Self {
            mspline: MSplineBasis::new(knots, degree)?,
        })
    }

    /// Builds an open-uniform I-spline basis from data.
    pub fn open_uniform_from_data(
        x: &[f64],
        n_basis: usize,
        degree: usize,
    ) -> Result<Self, SplineError> {
        Ok(Self {
            mspline: MSplineBasis::open_uniform_from_data(x, n_basis, degree)?,
        })
    }

    /// Builds a predictor design.
    pub fn design(&self, x: &[f64]) -> Result<ISplineDesign, SplineError> {
        if x.iter().any(|value| !value.is_finite()) {
            return Err(SplineError::NonFiniteValue);
        }
        Ok(ISplineDesign {
            x: x.to_vec(),
            basis: self.clone(),
        })
    }

    /// Underlying knot vector.
    #[must_use]
    #[inline(always)]
    pub fn knots(&self) -> &[f64] {
        self.mspline.knots()
    }

    /// Degree.
    #[must_use]
    #[inline(always)]
    pub fn degree(&self) -> usize {
        self.mspline.degree()
    }

    /// Number of basis functions.
    #[must_use]
    #[inline(always)]
    pub fn n_basis(&self) -> usize {
        self.mspline.n_basis()
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

        for (index, value) in out.iter_mut().enumerate() {
            *value = self.evaluate_one(index, x);
        }
    }

    /// Visits non-zero basis-function values at `x` without allocating.
    #[inline]
    pub fn for_each_basis(&self, x: f64, mut f: impl FnMut(usize, f64)) {
        for index in 0..self.n_basis() {
            let weight = self.evaluate_one(index, x);
            if weight != 0.0 {
                f(index, weight);
            }
        }
    }

    /// Evaluates first derivatives of all I-spline basis functions.
    #[must_use]
    pub fn evaluate_derivative(&self, x: f64) -> Vec<f64> {
        self.mspline.evaluate(x)
    }

    /// Evaluates one I-spline basis function at `x`.
    #[must_use]
    #[inline]
    pub fn evaluate_one(&self, index: usize, x: f64) -> f64 {
        let knots = self.knots();
        let support_left = knots[index];
        let support_right = knots[index + self.degree() + 1];
        if support_right <= support_left || x <= support_left {
            return 0.0;
        }
        if x >= support_right {
            return 1.0;
        }

        integrate_piecewise(knots, support_left, x, |point| {
            self.mspline.evaluate_one(index, point)
        })
        .clamp(0.0, 1.0)
    }
}

/// I-spline predictor.
#[derive(Debug, Clone, PartialEq)]
pub struct ISplineDesign {
    x: Vec<f64>,
    basis: ISplineBasis,
}

impl ISplineDesign {
    /// Returns the basis metadata.
    #[must_use]
    #[inline(always)]
    pub fn basis(&self) -> &ISplineBasis {
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
            .mspline
            .for_each_basis(self.x[row], |index, weight| {
                value += beta[index] * weight;
            });
        value
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

impl SplineRowBasis for ISplineDesign {
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

impl PredictorBlock for ISplineDesign {
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

fn integrate_piecewise(knots: &[f64], start: f64, end: f64, f: impl Fn(f64) -> f64) -> f64 {
    let mut sum = 0.0;
    let mut left = start;
    for right in knots
        .iter()
        .copied()
        .filter(|knot| *knot > start && *knot < end)
    {
        if right > left {
            sum += integrate_interval(left, right, &f);
            left = right;
        }
    }
    sum + integrate_interval(left, end, &f)
}

fn integrate_interval(left: f64, right: f64, f: &impl Fn(f64) -> f64) -> f64 {
    if right <= left {
        return 0.0;
    }

    const NODES: [f64; 5] = [
        -0.906_179_845_938_664,
        -0.538_469_310_105_683_1,
        0.0,
        0.538_469_310_105_683_1,
        0.906_179_845_938_664,
    ];
    const WEIGHTS: [f64; 5] = [
        0.236_926_885_056_189_1,
        0.478_628_670_499_366_47,
        0.568_888_888_888_888_9,
        0.478_628_670_499_366_47,
        0.236_926_885_056_189_1,
    ];

    let midpoint = 0.5 * (left + right);
    let half = 0.5 * (right - left);
    half * NODES
        .iter()
        .zip(WEIGHTS)
        .map(|(node, weight)| weight * f(midpoint + half * node))
        .sum::<f64>()
}
