use std::ops::Range;

use gamlss_core::{LinearPredictorGeometry, ModelError, PredictorBlock, RowMultiplier};

use crate::geometry::{validate_gram_lengths, validate_transpose_lengths};
use crate::row_basis::SplineRowBasis;
use crate::validation::{finite_data_range, validate_coordinates};
use crate::{OnDemandSplineDesign, SplineError};

const MAX_NATURAL_SCRATCH_DIM: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq)]
struct NaturalRowGeometry {
    knot0: usize,
    knot1: usize,
    direct0: f64,
    direct1: f64,
    second0: f64,
    second1: f64,
}

impl NaturalRowGeometry {
    #[inline]
    fn weight(self, basis: &NaturalCubicSplineBasis, index: usize) -> f64 {
        let direct = f64::from(index == self.knot0)
            .mul_add(self.direct0, f64::from(index == self.knot1) * self.direct1);
        self.second0.mul_add(
            basis.second_derivative(index, self.knot0),
            self.second1
                .mul_add(basis.second_derivative(index, self.knot1), direct),
        )
    }

    #[inline]
    fn dot(self, basis: &NaturalCubicSplineBasis, beta: &[f64]) -> f64 {
        let mut value = beta[self.knot0].mul_add(self.direct0, beta[self.knot1] * self.direct1);
        let column0 = basis.second_derivative_column(self.knot0);
        let column1 = basis.second_derivative_column(self.knot1);
        for ((coefficient, second0), second1) in beta
            .iter()
            .copied()
            .zip(column0.iter().copied())
            .zip(column1.iter().copied())
        {
            let weight = self.second0.mul_add(second0, self.second1 * second1);
            value = coefficient.mul_add(weight, value);
        }
        value
    }

    #[inline]
    fn for_each(self, basis: &NaturalCubicSplineBasis, mut f: impl FnMut(usize, f64)) {
        for index in 0..basis.n_basis() {
            let weight = self.weight(basis, index);
            if weight != 0.0 {
                f(index, weight);
            }
        }
    }

    #[inline]
    fn add_scaled(self, basis: &NaturalCubicSplineBasis, scale: f64, out: &mut [f64]) {
        out[self.knot0] = scale.mul_add(self.direct0, out[self.knot0]);
        out[self.knot1] = scale.mul_add(self.direct1, out[self.knot1]);
        let column0 = basis.second_derivative_column(self.knot0);
        let column1 = basis.second_derivative_column(self.knot1);
        for ((value, second0), second1) in out
            .iter_mut()
            .zip(column0.iter().copied())
            .zip(column1.iter().copied())
        {
            let weight = self.second0.mul_add(second0, self.second1 * second1);
            *value = scale.mul_add(weight, *value);
        }
    }
}

/// Natural cubic cardinal-spline basis with one coefficient per knot.
///
/// For strictly increasing knots $t_0,\ldots,t_{K-1}$, the basis functions satisfy
///
/// $$
/// N_j(t_i)=\delta_{ij},
/// \qquad
/// f(x)=\sum_{j=0}^{K-1}\beta_jN_j(x).
/// $$
///
/// Here $\delta_{ij}$ is the Kronecker delta, so the coefficient $\beta_j$ is the fitted value $f(t_j)$. The natural boundary conditions are $f^{\prime\prime}(t_0)=f^{\prime\prime}(t_{K-1})=0$. Outside the knot range the implementation extends each basis function linearly using its derivative at the nearest boundary. In code, $K$ is [`NaturalCubicSplineBasis::n_basis`] and $(t_j)$ is [`NaturalCubicSplineBasis::knots`].
#[allow(clippy::doc_markdown)]
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
    #[allow(clippy::cast_precision_loss)]
    pub fn uniform_from_data(x: &[f64], n_basis: usize) -> Result<Self, SplineError> {
        if x.is_empty() {
            return Err(SplineError::EmptyInput);
        }
        if n_basis < 2 {
            return Err(SplineError::NotEnoughBasis { n_basis, degree: 1 });
        }

        let (min, max) = finite_data_range(x)?;

        let step = (max - min) / (n_basis - 1) as f64;
        let knots = (0..n_basis)
            .map(|index| min + step * index as f64)
            .collect::<Vec<_>>();
        Self::new(knots)
    }

    /// Builds a predictor design for concrete `x` coordinates.
    pub fn design(&self, x: &[f64]) -> Result<NaturalCubicSplineDesign, SplineError> {
        validate_coordinates(x)?;
        Ok(NaturalCubicSplineDesign {
            x: x.into(),
            rows: x
                .iter()
                .copied()
                .map(|value| self.value_geometry(value))
                .collect(),
            basis: self.clone(),
        })
    }

    /// Builds a low-memory predictor that recomputes row geometry on demand.
    pub fn on_demand_design(&self, x: &[f64]) -> Result<OnDemandSplineDesign<Self>, SplineError> {
        OnDemandSplineDesign::new(x, self.clone())
    }

    /// Knot vector.
    #[must_use]
    #[inline]
    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    /// Number of basis functions.
    #[must_use]
    #[inline]
    pub const fn n_basis(&self) -> usize {
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
        let geometry = self.value_geometry(x);
        for (index, value) in out.iter_mut().enumerate() {
            *value = geometry.weight(self, index);
        }
    }

    /// Visits non-zero basis-function values at `x` without allocating.
    #[inline]
    pub fn for_each_basis(&self, x: f64, mut f: impl FnMut(usize, f64)) {
        self.value_geometry(x).for_each(self, &mut f);
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
        let geometry = self.derivative_geometry(x);
        for (index, value) in out.iter_mut().enumerate() {
            *value = geometry.weight(self, index);
        }
    }

    /// Visits non-zero first derivatives at `x` without allocating.
    #[inline]
    pub fn for_each_derivative_basis(&self, x: f64, mut f: impl FnMut(usize, f64)) {
        self.derivative_geometry(x).for_each(self, &mut f);
    }

    #[allow(clippy::suboptimal_flops)]
    fn value_geometry(&self, x: f64) -> NaturalRowGeometry {
        let (interval, left_extrapolate, right_extrapolate) = self.interval(x);
        if left_extrapolate || right_extrapolate {
            let edge = if left_extrapolate {
                0
            } else {
                self.knots.len() - 1
            };
            let edge_x = self.knots[edge];
            let offset = x - edge_x;
            let mut geometry = self.derivative_geometry(edge_x);
            geometry.direct0 *= offset;
            geometry.direct1 *= offset;
            geometry.second0 *= offset;
            geometry.second1 *= offset;
            if edge == geometry.knot0 {
                geometry.direct0 += 1.0;
            } else {
                debug_assert_eq!(edge, geometry.knot1);
                geometry.direct1 += 1.0;
            }
            return geometry;
        }

        let x0 = self.knots[interval];
        let x1 = self.knots[interval + 1];
        let h = x1 - x0;
        let a = (x1 - x) / h;
        let b = (x - x0) / h;
        NaturalRowGeometry {
            knot0: interval,
            knot1: interval + 1,
            direct0: a,
            direct1: b,
            second0: (a * a * a - a) * h * h / 6.0,
            second1: (b * b * b - b) * h * h / 6.0,
        }
    }

    #[allow(clippy::suboptimal_flops)]
    fn derivative_geometry(&self, x: f64) -> NaturalRowGeometry {
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
        NaturalRowGeometry {
            knot0: interval,
            knot1: interval + 1,
            direct0: -1.0 / h,
            direct1: 1.0 / h,
            second0: h * (1.0 - 3.0 * a * a) / 6.0,
            second1: h * (3.0 * b * b - 1.0) / 6.0,
        }
    }

    fn second_derivative(&self, basis: usize, knot: usize) -> f64 {
        let n = self.knots.len();
        debug_assert!(basis < n);
        debug_assert!(knot < n);
        debug_assert_eq!(self.second_derivatives.len(), n * n);
        self.second_derivatives[knot * n + basis]
    }

    #[inline]
    pub(crate) fn second_derivative_column(&self, knot: usize) -> &[f64] {
        let n = self.knots.len();
        &self.second_derivatives[knot * n..(knot + 1) * n]
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

    #[inline]
    pub(crate) fn add_scaled_outer_at(&self, x: f64, scale: f64, out: &mut [f64]) {
        let row = self.value_geometry(x);
        let n_basis = self.n_basis();
        debug_assert_eq!(out.len(), n_basis * n_basis);
        row.for_each(self, |left_index, left_weight| {
            let scaled_left = scale * left_weight;
            row.for_each(self, |right_index, right_weight| {
                let index = left_index * n_basis + right_index;
                out[index] = scaled_left.mul_add(right_weight, out[index]);
            });
        });
    }
}

/// Natural cubic spline predictor with prepared interval geometry.
///
/// The basis is globally supported, so rows are not sparse. The design caches
/// only six scalar interval coefficients per observation and evaluates the two
/// required columns of the precomputed natural-spline operator directly.
#[derive(Debug, Clone, PartialEq)]
pub struct NaturalCubicSplineDesign {
    x: Box<[f64]>,
    rows: Box<[NaturalRowGeometry]>,
    basis: NaturalCubicSplineBasis,
}

impl NaturalCubicSplineDesign {
    /// Builds a design from data-derived uniform knots.
    pub fn uniform_from_data(x: &[f64], n_basis: usize) -> Result<Self, SplineError> {
        NaturalCubicSplineBasis::uniform_from_data(x, n_basis)?.design(x)
    }

    /// Returns the basis metadata.
    #[must_use]
    #[inline]
    pub const fn basis(&self) -> &NaturalCubicSplineBasis {
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
    #[allow(clippy::suboptimal_flops)]
    pub fn eta_derivative_row(&self, row: usize, beta: &[f64]) -> f64 {
        debug_assert!(row < self.x.len());
        debug_assert_eq!(beta.len(), self.basis.n_basis());
        self.basis
            .derivative_geometry(self.x[row])
            .dot(&self.basis, beta)
    }

    #[inline]
    fn add_scores_range<M>(
        &self,
        rows: Range<usize>,
        scores: &[f64],
        multiplier: &M,
        grad: &mut [f64],
    ) where
        M: RowMultiplier + ?Sized,
    {
        let n_basis = self.basis.n_basis();
        if n_basis <= MAX_NATURAL_SCRATCH_DIM {
            let mut curvature = [0.0; MAX_NATURAL_SCRATCH_DIM];
            for (offset, score) in scores.iter().copied().enumerate() {
                if score == 0.0 {
                    continue;
                }
                let row = rows.start + offset;
                let scale = score * multiplier.multiplier_at(row);
                if scale == 0.0 {
                    continue;
                }
                let geometry = self.rows[row];
                grad[geometry.knot0] = scale.mul_add(geometry.direct0, grad[geometry.knot0]);
                grad[geometry.knot1] = scale.mul_add(geometry.direct1, grad[geometry.knot1]);
                curvature[geometry.knot0] =
                    scale.mul_add(geometry.second0, curvature[geometry.knot0]);
                curvature[geometry.knot1] =
                    scale.mul_add(geometry.second1, curvature[geometry.knot1]);
            }

            for (knot, scale) in curvature.iter().copied().enumerate().take(n_basis) {
                if scale == 0.0 {
                    continue;
                }
                for (value, second) in grad
                    .iter_mut()
                    .zip(self.basis.second_derivative_column(knot))
                {
                    *value = scale.mul_add(*second, *value);
                }
            }
        } else {
            for (offset, score) in scores.iter().copied().enumerate() {
                if score == 0.0 {
                    continue;
                }
                let row = rows.start + offset;
                let scale = score * multiplier.multiplier_at(row);
                if scale != 0.0 {
                    self.rows[row].add_scaled(&self.basis, scale, grad);
                }
            }
        }
    }

    #[inline]
    fn add_gram_rows<M>(&self, row_weights: &[f64], multiplier: &M, out: &mut [f64])
    where
        M: RowMultiplier + ?Sized,
    {
        let n_basis = self.basis.n_basis();
        if n_basis > MAX_NATURAL_SCRATCH_DIM {
            for (row, weight) in row_weights.iter().copied().enumerate() {
                if weight == 0.0 {
                    continue;
                }
                let scale = weight * multiplier.multiplier_at(row);
                if scale == 0.0 {
                    continue;
                }
                self.rows[row].for_each(&self.basis, |left_index, left_weight| {
                    let scaled_left = scale * left_weight;
                    self.rows[row].for_each(&self.basis, |right_index, right_weight| {
                        let index = left_index * n_basis + right_index;
                        out[index] = scaled_left.mul_add(right_weight, out[index]);
                    });
                });
            }
            return;
        }

        // A row has the form d + S^T c, where d and c each have two
        // non-zero entries at adjacent knots and S is the knot-by-basis
        // natural-spline second-derivative operator. The accumulated d c^T
        // and c c^T products are therefore tridiagonal. Keeping just their
        // bands changes observation-dependent work from O(n K^2) to O(n),
        // followed by a small O(K^3) transform independent of n.
        let mut cross_diagonal = [0.0; MAX_NATURAL_SCRATCH_DIM];
        let mut cross_upper = [0.0; MAX_NATURAL_SCRATCH_DIM];
        let mut cross_lower = [0.0; MAX_NATURAL_SCRATCH_DIM];
        let mut curvature_diagonal = [0.0; MAX_NATURAL_SCRATCH_DIM];
        let mut curvature_off_diagonal = [0.0; MAX_NATURAL_SCRATCH_DIM];
        for (row, weight) in row_weights.iter().copied().enumerate() {
            if weight == 0.0 {
                continue;
            }
            let scale = weight * multiplier.multiplier_at(row);
            if scale == 0.0 {
                continue;
            }
            let geometry = self.rows[row];
            debug_assert_eq!(geometry.knot1, geometry.knot0 + 1);
            let direct = [
                (geometry.knot0, geometry.direct0),
                (geometry.knot1, geometry.direct1),
            ];
            add_scaled_pair_outer(&direct, scale, n_basis, out);

            let interval = geometry.knot0;
            cross_diagonal[interval] =
                (scale * geometry.direct0).mul_add(geometry.second0, cross_diagonal[interval]);
            cross_diagonal[interval + 1] =
                (scale * geometry.direct1).mul_add(geometry.second1, cross_diagonal[interval + 1]);
            cross_upper[interval] =
                (scale * geometry.direct0).mul_add(geometry.second1, cross_upper[interval]);
            cross_lower[interval] =
                (scale * geometry.direct1).mul_add(geometry.second0, cross_lower[interval]);

            curvature_diagonal[interval] =
                (scale * geometry.second0).mul_add(geometry.second0, curvature_diagonal[interval]);
            curvature_diagonal[interval + 1] = (scale * geometry.second1)
                .mul_add(geometry.second1, curvature_diagonal[interval + 1]);
            curvature_off_diagonal[interval] = (scale * geometry.second0)
                .mul_add(geometry.second1, curvature_off_diagonal[interval]);
        }

        // Add E S + (E S)^T for E = sum(w d c^T).
        for direct_index in 0..n_basis {
            add_cross_transform_row(
                direct_index,
                cross_diagonal[direct_index],
                self.basis.second_derivative_column(direct_index),
                n_basis,
                out,
            );
            if direct_index != 0 {
                add_cross_transform_row(
                    direct_index,
                    cross_lower[direct_index - 1],
                    self.basis.second_derivative_column(direct_index - 1),
                    n_basis,
                    out,
                );
            }
            if direct_index + 1 < n_basis {
                add_cross_transform_row(
                    direct_index,
                    cross_upper[direct_index],
                    self.basis.second_derivative_column(direct_index + 1),
                    n_basis,
                    out,
                );
            }
        }

        // Add S^T C S for the symmetric tridiagonal
        // C = sum(w c c^T).
        for knot in 0..n_basis {
            add_scaled_dense_outer(
                self.basis.second_derivative_column(knot),
                curvature_diagonal[knot],
                n_basis,
                out,
            );
            if knot + 1 < n_basis {
                add_scaled_symmetric_dense_cross(
                    self.basis.second_derivative_column(knot),
                    self.basis.second_derivative_column(knot + 1),
                    curvature_off_diagonal[knot],
                    n_basis,
                    out,
                );
            }
        }
    }
}

impl SplineRowBasis for NaturalCubicSplineDesign {
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
        self.rows[row].for_each(&self.basis, f);
    }
}

impl PredictorBlock for NaturalCubicSplineDesign {
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
        debug_assert!(row < self.x.len());
        debug_assert_eq!(beta.len(), self.basis.n_basis());

        self.rows[row].dot(&self.basis, beta)
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

        self.add_scores_range(rows, scores, &UnitMultiplier, grad);
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

        self.add_scores_range(rows, scores, multiplier, grad);
    }
}

impl LinearPredictorGeometry for NaturalCubicSplineDesign {
    #[inline]
    fn add_weighted_gram(&self, row_weights: &[f64], out: &mut [f64]) -> Result<(), ModelError> {
        let nparams = self.basis.n_basis();
        validate_gram_lengths(self.rows.len(), nparams, row_weights, out)?;
        self.add_gram_rows(row_weights, &UnitMultiplier, out);
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
        let nparams = self.basis.n_basis();
        validate_gram_lengths(self.rows.len(), nparams, row_weights, out)?;
        self.add_gram_rows(row_weights, multiplier, out);
        Ok(())
    }

    #[inline]
    fn add_t_mul_vec(&self, row_scores: &[f64], out: &mut [f64]) -> Result<(), ModelError> {
        validate_transpose_lengths(self.rows.len(), self.basis.n_basis(), row_scores, out)?;
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
        validate_transpose_lengths(self.rows.len(), self.basis.n_basis(), row_scores, out)?;
        self.add_weighted_gradient_by(row_scores, multiplier, &[], out);
        Ok(())
    }
}

struct UnitMultiplier;

impl RowMultiplier for UnitMultiplier {
    #[inline]
    fn multiplier_at(&self, _: usize) -> f64 {
        1.0
    }
}

#[inline]
fn add_scaled_pair_outer(pair: &[(usize, f64); 2], scale: f64, n_basis: usize, out: &mut [f64]) {
    for (left_index, left_value) in pair.iter().copied() {
        let scaled_left = scale * left_value;
        for (right_index, right_value) in pair.iter().copied() {
            let index = left_index * n_basis + right_index;
            out[index] = scaled_left.mul_add(right_value, out[index]);
        }
    }
}

#[inline]
fn add_cross_transform_row(
    direct_index: usize,
    scale: f64,
    second: &[f64],
    n_basis: usize,
    out: &mut [f64],
) {
    if scale == 0.0 {
        return;
    }
    for (basis_index, second) in second.iter().copied().enumerate() {
        let value = scale * second;
        let index = direct_index * n_basis + basis_index;
        let transposed = basis_index * n_basis + direct_index;
        out[index] += value;
        out[transposed] += value;
    }
}

#[inline]
fn add_scaled_dense_outer(second: &[f64], scale: f64, n_basis: usize, out: &mut [f64]) {
    if scale == 0.0 {
        return;
    }
    for (left_index, left) in second.iter().copied().enumerate() {
        let scaled_left = scale * left;
        let output = &mut out[left_index * n_basis..(left_index + 1) * n_basis];
        for (value, right) in output.iter_mut().zip(second.iter().copied()) {
            *value = scaled_left.mul_add(right, *value);
        }
    }
}

#[inline]
fn add_scaled_symmetric_dense_cross(
    left: &[f64],
    right: &[f64],
    scale: f64,
    n_basis: usize,
    out: &mut [f64],
) {
    if scale == 0.0 {
        return;
    }
    for (left_index, (left_value, right_value)) in
        left.iter().copied().zip(right.iter().copied()).enumerate()
    {
        let scaled_left = scale * left_value;
        let scaled_right = scale * right_value;
        let output = &mut out[left_index * n_basis..(left_index + 1) * n_basis];
        for ((value, left_cross), right_cross) in output
            .iter_mut()
            .zip(left.iter().copied())
            .zip(right.iter().copied())
        {
            *value = scaled_left.mul_add(right_cross, scaled_right.mul_add(left_cross, *value));
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
    let mut basis_major = Vec::with_capacity(n * n);
    for basis in 0..n {
        basis_major.extend(natural_basis_second_derivatives(knots, basis));
    }
    let mut second_derivatives = vec![0.0; n * n];
    for basis in 0..n {
        for knot in 0..n {
            second_derivatives[knot * n + basis] = basis_major[basis * n + knot];
        }
    }
    debug_assert_eq!(second_derivatives.len(), n * n);
    second_derivatives
}

#[allow(clippy::suboptimal_flops)]
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
