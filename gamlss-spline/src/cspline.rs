use std::ops::Range;

use gamlss_core::{Link, PredictorBlock, RowMultiplier, Softplus};

use crate::ispline::{MAX_PREFIX_VALUES, integrate_interval};
use crate::{
    DifferentiableSplineBasis1d, ISplineBasis, KnotPlacement, OnDemandSplineDesign, OpenKnotVector,
    SplineBasis1d, SplineError,
};

/// Sign of the enforced second derivative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurvatureDirection {
    /// Non-negative second derivative.
    Convex,
    /// Non-positive second derivative.
    Concave,
}

impl CurvatureDirection {
    const fn sign(self) -> f64 {
        match self {
            Self::Convex => 1.0,
            Self::Concave => -1.0,
        }
    }
}

/// C-spline basis obtained by integrating an I-spline basis once more.
///
/// For `C_i(x) = integral I_i(t) dt`, `C_i''(x) = M_i(x) >= 0`.
/// Positive C-spline coefficients therefore enforce convexity structurally.
/// Prefix integrals at knot boundaries are prepared once; evaluation only
/// integrates over the active knot interval. This is a constrained
/// specialization of a polynomial spline space, not a new unconstrained
/// function space.
#[derive(Debug, Clone, PartialEq)]
pub struct CSplineBasis {
    ispline: ISplineBasis,
    prefixes: Box<[[f64; MAX_PREFIX_VALUES]]>,
}

impl CSplineBasis {
    /// Creates a C-spline basis from an I-spline basis.
    #[must_use]
    pub fn new(ispline: ISplineBasis) -> Self {
        let mut prefixes = Vec::with_capacity(ispline.n_basis());
        for basis_index in 0..ispline.n_basis() {
            let mut prefix = [0.0; MAX_PREFIX_VALUES];
            let mut cumulative = 0.0;
            for (offset, value) in prefix.iter_mut().enumerate().take(ispline.degree() + 1) {
                *value = cumulative;
                let interval = basis_index + offset;
                let left = ispline.knots()[interval];
                let right = ispline.knots()[interval + 1];
                cumulative +=
                    integrate_interval(left, right, &|x| ispline.evaluate_one(basis_index, x));
            }
            prefix[ispline.degree() + 1] = cumulative;
            prefixes.push(prefix);
        }
        Self {
            ispline,
            prefixes: prefixes.into_boxed_slice(),
        }
    }

    /// Builds an open C-spline with uniform or quantile knot placement.
    pub fn open_from_data(
        x: &[f64],
        n_basis: usize,
        degree: usize,
        placement: KnotPlacement,
    ) -> Result<Self, SplineError> {
        Ok(Self::new(ISplineBasis::open_from_data(
            x, n_basis, degree, placement,
        )?))
    }

    /// Builds an open C-spline using weighted quantile knots.
    pub fn open_from_weighted_data(
        x: &[f64],
        weights: &[f64],
        n_basis: usize,
        degree: usize,
    ) -> Result<Self, SplineError> {
        Ok(Self::new(ISplineBasis::open_from_weighted_data(
            x, weights, n_basis, degree,
        )?))
    }

    /// Builds a C-spline from persisted open-knot metadata.
    pub fn from_open_knots(knots: OpenKnotVector) -> Result<Self, SplineError> {
        Ok(Self::new(ISplineBasis::from_open_knots(knots)?))
    }

    /// Builds an on-demand linear design for unconstrained C-spline columns.
    pub fn design(&self, x: &[f64]) -> Result<OnDemandSplineDesign<Self>, SplineError> {
        OnDemandSplineDesign::new(x, self.clone())
    }

    /// Underlying I-spline basis.
    #[must_use]
    pub const fn ispline(&self) -> &ISplineBasis {
        &self.ispline
    }

    /// Underlying M-/I-spline degree.
    #[must_use]
    pub const fn degree(&self) -> usize {
        self.ispline.degree()
    }

    /// Number of C-spline columns.
    #[must_use]
    pub const fn n_basis(&self) -> usize {
        self.ispline.n_basis()
    }

    /// Evaluates one C-spline basis function.
    #[must_use]
    pub fn evaluate_one(&self, index: usize, x: f64) -> f64 {
        debug_assert!(index < self.n_basis());
        let knots = self.ispline.knots();
        let support_left = knots[index];
        let support_right = knots[index + self.degree() + 1];
        if support_right <= support_left || x <= support_left {
            return 0.0;
        }
        let prefix = &self.prefixes[index];
        if x >= support_right {
            return prefix[self.degree() + 1] + (x - support_right);
        }
        let interval = knots
            .partition_point(|knot| *knot <= x)
            .saturating_sub(1)
            .clamp(index, index + self.degree());
        prefix[interval - index]
            + integrate_interval(knots[interval], x, &|point| {
                self.ispline.evaluate_one(index, point)
            })
    }

    /// Visits non-zero C-spline values at one coordinate.
    pub fn for_each_basis(&self, x: f64, mut f: impl FnMut(usize, f64)) {
        for index in 0..self.n_basis() {
            let value = self.evaluate_one(index, x);
            if value != 0.0 {
                f(index, value);
            }
        }
    }
}

impl SplineBasis1d for CSplineBasis {
    fn n_basis(&self) -> usize {
        self.n_basis()
    }

    fn for_each_basis(&self, x: f64, f: impl FnMut(usize, f64)) -> Result<(), SplineError> {
        if !x.is_finite() {
            return Err(SplineError::NonFiniteValue);
        }
        Self::for_each_basis(self, x, f);
        Ok(())
    }
}

impl DifferentiableSplineBasis1d for CSplineBasis {
    fn max_derivative_order(&self) -> Option<usize> {
        self.degree().checked_add(2)
    }

    fn for_each_basis_derivative(
        &self,
        x: f64,
        derivative_order: usize,
        f: impl FnMut(usize, f64),
    ) -> Result<(), SplineError> {
        if !x.is_finite() {
            return Err(SplineError::NonFiniteValue);
        }
        let max = self.degree() + 2;
        if derivative_order > max {
            return Err(SplineError::UnsupportedDerivativeOrder {
                requested: derivative_order,
                max,
            });
        }
        match derivative_order {
            0 => SplineBasis1d::for_each_basis(self, x, f),
            order => self.ispline.for_each_basis_derivative(x, order - 1, f),
        }
    }
}

/// Hard convex or concave predictor using positive C-spline coefficients.
///
/// The first two coefficients are an unconstrained intercept and slope. Every
/// remaining coefficient is mapped through softplus and multiplies one
/// C-spline column. The sign is selected by [`CurvatureDirection`], so the
/// requested second-derivative inequality holds for every parameter vector.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvexCSplineDesign {
    design: OnDemandSplineDesign<CSplineBasis>,
    direction: CurvatureDirection,
    origin: f64,
}

impl ConvexCSplineDesign {
    /// Creates a structurally convex or concave predictor.
    pub fn new(
        x: &[f64],
        basis: CSplineBasis,
        direction: CurvatureDirection,
    ) -> Result<Self, SplineError> {
        let origin = basis.ispline().knots()[0];
        Ok(Self {
            design: OnDemandSplineDesign::new(x, basis)?,
            direction,
            origin,
        })
    }

    /// C-spline metadata.
    #[must_use]
    pub const fn basis(&self) -> &CSplineBasis {
        self.design.basis()
    }

    /// Curvature direction.
    #[must_use]
    pub const fn direction(&self) -> CurvatureDirection {
        self.direction
    }

    /// Input coordinates.
    #[must_use]
    pub fn x(&self) -> &[f64] {
        self.design.x()
    }

    /// Predictor first derivative.
    #[must_use]
    pub fn eta_derivative_row(&self, row: usize, beta: &[f64]) -> f64 {
        debug_assert_eq!(beta.len(), self.nparams());
        let sign = self.direction.sign();
        let mut value = beta[1];
        self.basis()
            .ispline()
            .for_each_basis(self.x()[row], |index, basis| {
                value = (sign * Softplus::inverse(beta[index + 2])).mul_add(basis, value);
            });
        value
    }

    /// Predictor second derivative, guaranteed to have the requested sign.
    #[must_use]
    pub fn eta_second_derivative_row(&self, row: usize, beta: &[f64]) -> f64 {
        debug_assert_eq!(beta.len(), self.nparams());
        let sign = self.direction.sign();
        let mut value = 0.0;
        self.basis()
            .ispline()
            .for_each_derivative_basis(self.x()[row], |index, basis| {
                value = (sign * Softplus::inverse(beta[index + 2])).mul_add(basis, value);
            });
        value
    }

    fn add_row_gradient(&self, row: usize, score: f64, beta: &[f64], grad: &mut [f64]) {
        grad[0] += score;
        grad[1] = score.mul_add(self.x()[row] - self.origin, grad[1]);
        let sign = self.direction.sign();
        self.basis().for_each_basis(self.x()[row], |index, basis| {
            let scale = score * sign * Softplus::derivative_inverse(beta[index + 2]);
            grad[index + 2] = scale.mul_add(basis, grad[index + 2]);
        });
    }
}

impl PredictorBlock for ConvexCSplineDesign {
    fn nrows(&self) -> usize {
        self.design.nrows()
    }

    fn nparams(&self) -> usize {
        self.basis().n_basis() + 2
    }

    fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
        debug_assert_eq!(beta.len(), self.nparams());
        let sign = self.direction.sign();
        let mut value = beta[1].mul_add(self.x()[row] - self.origin, beta[0]);
        self.basis().for_each_basis(self.x()[row], |index, basis| {
            value = (sign * Softplus::inverse(beta[index + 2])).mul_add(basis, value);
        });
        value
    }

    fn add_gradient_range(
        &self,
        rows: Range<usize>,
        scores: &[f64],
        beta: &[f64],
        grad: &mut [f64],
    ) {
        for (offset, score) in scores.iter().copied().enumerate() {
            if score != 0.0 {
                self.add_row_gradient(rows.start + offset, score, beta, grad);
            }
        }
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
        for (offset, score) in scores.iter().copied().enumerate() {
            if score == 0.0 {
                continue;
            }
            let row = rows.start + offset;
            let score = score * multiplier.multiplier_at(row);
            if score != 0.0 {
                self.add_row_gradient(row, score, beta, grad);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::PredictorBlock;

    use super::{CSplineBasis, ConvexCSplineDesign, CurvatureDirection};
    use crate::{DifferentiableSplineBasis1d, KnotPlacement};

    #[test]
    fn c_spline_derivatives_follow_i_and_m_splines() {
        let x = [0.0, 0.2, 0.5, 0.8, 1.0];
        let basis = CSplineBasis::open_from_data(&x, 6, 2, KnotPlacement::Quantile).unwrap();
        let point = 0.37;
        let first = basis.evaluate_basis_derivative(point, 1).unwrap();
        let second = basis.evaluate_basis_derivative(point, 2).unwrap();
        assert_eq!(first, basis.ispline().evaluate(point));
        assert_eq!(second, basis.ispline().evaluate_derivative(point));
    }

    #[test]
    fn local_prefix_cache_preserves_integrals_and_linear_tails() {
        let basis =
            CSplineBasis::open_from_data(&[0.0, 0.2, 0.5, 0.8, 1.0], 6, 2, KnotPlacement::Quantile)
                .unwrap();
        let point = 0.37;
        let step = 1.0e-6;
        for index in 0..basis.n_basis() {
            let numerical_derivative = (basis.evaluate_one(index, point + step)
                - basis.evaluate_one(index, point - step))
                / (2.0 * step);
            assert_relative_eq!(
                numerical_derivative,
                basis.ispline().evaluate_one(index, point),
                epsilon = 2.0e-9
            );

            let support_right = basis.ispline().knots()[index + basis.degree() + 1];
            assert_relative_eq!(
                basis.evaluate_one(index, support_right + 0.25)
                    - basis.evaluate_one(index, support_right),
                0.25,
                epsilon = 1.0e-13
            );
        }
    }

    #[test]
    fn constrained_design_has_requested_curvature_and_gradient() {
        let x = [0.0, 0.2, 0.5, 0.8, 1.0];
        let basis = CSplineBasis::open_from_data(&x, 6, 2, KnotPlacement::Uniform).unwrap();
        for direction in [CurvatureDirection::Convex, CurvatureDirection::Concave] {
            let design = ConvexCSplineDesign::new(&x, basis.clone(), direction).unwrap();
            let beta = [0.2, -0.4, 0.1, -0.3, 0.7, -0.2, 0.5, 0.9];
            for row in 0..x.len() {
                let curvature = design.eta_second_derivative_row(row, &beta);
                match direction {
                    CurvatureDirection::Convex => assert!(curvature >= -1.0e-13),
                    CurvatureDirection::Concave => assert!(curvature <= 1.0e-13),
                }
            }
            let scores = [0.3, -0.2, 0.7, 0.1, -0.5];
            let mut gradient = [0.0; 8];
            design.add_gradient(&scores, &beta, &mut gradient);
            let step = 1.0e-6;
            for index in 0..beta.len() {
                let mut lower = beta;
                let mut upper = beta;
                lower[index] -= step;
                upper[index] += step;
                let objective = |parameters: &[f64]| {
                    (0..x.len())
                        .map(|row| scores[row] * design.eta_row(row, parameters))
                        .sum::<f64>()
                };
                let actual = (objective(&upper) - objective(&lower)) / (2.0 * step);
                assert_relative_eq!(actual, gradient[index], epsilon = 2.0e-9);
            }
        }
    }
}
