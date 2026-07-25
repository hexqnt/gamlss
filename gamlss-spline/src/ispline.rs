use std::ops::Range;

use gamlss_core::{LinearPredictorGeometry, ModelError, PredictorBlock, RowMultiplier};

use crate::geometry::{validate_gram_lengths, validate_transpose_lengths};
use crate::mspline::MSplineBasis;
use crate::row_basis::SplineRowBasis;
use crate::validation::validate_coordinates;
use crate::{OnDemandSplineDesign, SplineError};

const MAX_PREFIX_VALUES: usize = 5;
const MAX_PARTIAL_VALUES: usize = 4;
const GAUSS_NODES: [f64; 5] = [
    -0.906_179_845_938_664,
    -0.538_469_310_105_683_1,
    0.0,
    0.538_469_310_105_683_1,
    0.906_179_845_938_664,
];
const GAUSS_WEIGHTS: [f64; 5] = [
    0.236_926_885_056_189_1,
    0.478_628_670_499_366_47,
    0.568_888_888_888_888_9,
    0.478_628_670_499_366_47,
    0.236_926_885_056_189_1,
];

/// I-spline basis built by integrating normalized M-splines.
///
/// For degree $p$, knot vector $\boldsymbol t$, and a corresponding non-degenerate M-spline basis function,
///
/// $$
/// I_{i,p}(x)=\int_{t_i}^{x}M_{i,p}(u)\\,du,
/// \qquad
/// \frac{d}{dx}I_{i,p}(x)=M_{i,p}(x).
/// $$
///
/// Since each M-spline is non-negative, the derivative displayed above is non-negative. The implementation returns zero at and below $t_i$ and one at and above $t_{i+p+1}$; a degenerate support $t_i=t_{i+p+1}$ returns zero. Consequently every basis function is non-decreasing and lies in $\lbrack 0,1\rbrack$. The symbols $p$ and $\boldsymbol t$ correspond to [`ISplineBasis::degree`] and [`ISplineBasis::knots`].
///
/// Integral masses at knot boundaries are prepared once. Evaluation therefore
/// integrates only the one polynomial segment containing `x`, instead of
/// repeatedly traversing every preceding knot interval.
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, PartialEq)]
pub struct ISplineBasis {
    pub(crate) mspline: MSplineBasis,
    integral_prefixes: Box<[[f64; MAX_PREFIX_VALUES]]>,
    nondegenerate: Box<[bool]>,
    all_nondegenerate: bool,
}

impl ISplineBasis {
    /// Creates an I-spline basis from a finite nondecreasing knot vector.
    pub fn new(knots: Vec<f64>, degree: usize) -> Result<Self, SplineError> {
        Ok(Self::from_mspline(MSplineBasis::new(knots, degree)?))
    }

    /// Builds an open-uniform I-spline basis from data.
    pub fn open_uniform_from_data(
        x: &[f64],
        n_basis: usize,
        degree: usize,
    ) -> Result<Self, SplineError> {
        Ok(Self::from_mspline(MSplineBasis::open_uniform_from_data(
            x, n_basis, degree,
        )?))
    }

    fn from_mspline(mspline: MSplineBasis) -> Self {
        let mut integral_prefixes = Vec::with_capacity(mspline.n_basis());
        let mut nondegenerate = Vec::with_capacity(mspline.n_basis());
        for index in 0..mspline.n_basis() {
            let mut prefix = [0.0; MAX_PREFIX_VALUES];
            let support_left = mspline.knots()[index];
            let support_right = mspline.knots()[index + mspline.degree() + 1];
            let has_support = support_right > support_left;
            nondegenerate.push(has_support);
            if has_support {
                for offset in 0..=mspline.degree() {
                    let left = mspline.knots()[index + offset];
                    let right = mspline.knots()[index + offset + 1];
                    prefix[offset + 1] = prefix[offset]
                        + integrate_interval(left, right, &|point| {
                            mspline.evaluate_one(index, point)
                        });
                }
            }
            integral_prefixes.push(prefix);
        }
        let all_nondegenerate = nondegenerate.iter().all(|value| *value);

        Self {
            mspline,
            integral_prefixes: integral_prefixes.into_boxed_slice(),
            nondegenerate: nondegenerate.into_boxed_slice(),
            all_nondegenerate,
        }
    }

    /// Builds a compact prepared predictor for repeated model passes.
    pub fn design(&self, x: &[f64]) -> Result<ISplineDesign, SplineError> {
        ISplineDesign::from_owned_basis(x, self.clone())
    }

    /// Builds a low-memory predictor that reevaluates row geometry on demand.
    pub fn on_demand_design(&self, x: &[f64]) -> Result<OnDemandSplineDesign<Self>, SplineError> {
        OnDemandSplineDesign::new(x, self.clone())
    }

    /// Underlying knot vector.
    #[must_use]
    #[inline]
    pub fn knots(&self) -> &[f64] {
        self.mspline.knots()
    }

    /// Degree.
    #[must_use]
    #[inline]
    pub const fn degree(&self) -> usize {
        self.mspline.degree()
    }

    /// Number of basis functions.
    #[must_use]
    #[inline]
    pub const fn n_basis(&self) -> usize {
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
        out.fill(0.0);
        self.for_each_basis(x, |index, weight| out[index] = weight);
    }

    /// Visits non-zero basis-function values at `x` without allocating.
    #[inline]
    pub fn for_each_basis(&self, x: f64, f: impl FnMut(usize, f64)) {
        self.prepare_row(x).for_each(self, f);
    }

    /// Evaluates first derivatives of all I-spline basis functions.
    #[must_use]
    pub fn evaluate_derivative(&self, x: f64) -> Vec<f64> {
        self.mspline.evaluate(x)
    }

    /// Visits non-zero first-derivative basis values at `x` without allocating.
    #[inline]
    pub fn for_each_derivative_basis(&self, x: f64, f: impl FnMut(usize, f64)) {
        self.mspline.for_each_basis(x, f);
    }

    /// Evaluates one I-spline basis function at `x`.
    #[must_use]
    #[inline]
    pub fn evaluate_one(&self, index: usize, x: f64) -> f64 {
        let support_left = self.knots()[index];
        let support_right = self.knots()[index + self.degree() + 1];
        if support_right <= support_left || x <= support_left {
            return 0.0;
        }
        if x >= support_right {
            return 1.0;
        }

        let interval = self
            .knots()
            .partition_point(|knot| *knot <= x)
            .saturating_sub(1)
            .clamp(index, index + self.degree());
        self.evaluate_partial(index, x, interval)
    }

    #[inline]
    fn evaluate_partial(&self, index: usize, x: f64, interval: usize) -> f64 {
        let offset = interval - index;
        let prefix = self.integral_prefixes[index][offset];
        let partial = integrate_interval(self.knots()[interval], x, &|point| {
            self.mspline.evaluate_one(index, point)
        });
        (prefix + partial).clamp(0.0, 1.0)
    }

    #[inline]
    fn prepare_row(&self, x: f64) -> PreparedISplineRow {
        let degree = self.degree();
        let n_basis = self.n_basis();
        let saturated_end =
            self.knots()[degree + 1..degree + 1 + n_basis].partition_point(|right| *right <= x);
        let partial_end = self.knots()[..n_basis].partition_point(|left| *left < x);
        let partial_len = partial_end.saturating_sub(saturated_end);
        debug_assert!(partial_len <= degree + 1);
        debug_assert!(partial_len <= MAX_PARTIAL_VALUES);

        let mut partial_weights = [0.0; MAX_PARTIAL_VALUES];
        if partial_len != 0 {
            let interval = self
                .knots()
                .partition_point(|knot| *knot <= x)
                .saturating_sub(1);
            for (offset, weight) in partial_weights.iter_mut().enumerate().take(partial_len) {
                let index = saturated_end + offset;
                if self.nondegenerate[index] {
                    *weight =
                        self.evaluate_partial(index, x, interval.clamp(index, index + degree));
                }
            }
        }

        PreparedISplineRow {
            saturated_end,
            partial_len,
            partial_weights,
        }
    }

    #[inline]
    pub(crate) fn add_scaled_outer_at(&self, x: f64, scale: f64, out: &mut [f64]) {
        let row = self.prepare_row(x);
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

#[derive(Debug, Clone, Copy, PartialEq)]
struct PreparedISplineRow {
    saturated_end: usize,
    partial_len: usize,
    partial_weights: [f64; MAX_PARTIAL_VALUES],
}

impl PreparedISplineRow {
    #[inline]
    const fn partial_len(self) -> usize {
        self.partial_len
    }

    #[inline]
    fn for_each(self, basis: &ISplineBasis, mut f: impl FnMut(usize, f64)) {
        for index in 0..self.saturated_end {
            if basis.nondegenerate[index] {
                f(index, 1.0);
            }
        }
        for (offset, weight) in self
            .partial_weights
            .iter()
            .copied()
            .enumerate()
            .take(self.partial_len())
        {
            if weight != 0.0 {
                f(self.saturated_end + offset, weight);
            }
        }
    }

    #[inline]
    fn dot(self, basis: &ISplineBasis, beta: &[f64]) -> f64 {
        let mut value = if basis.all_nondegenerate {
            beta[..self.saturated_end].iter().copied().sum()
        } else {
            beta[..self.saturated_end]
                .iter()
                .copied()
                .zip(basis.nondegenerate.iter().copied())
                .filter_map(|(coefficient, keep)| keep.then_some(coefficient))
                .sum()
        };
        for (offset, weight) in self
            .partial_weights
            .iter()
            .copied()
            .enumerate()
            .take(self.partial_len())
        {
            value = beta[self.saturated_end + offset].mul_add(weight, value);
        }
        value
    }

    #[inline]
    fn add_scaled(self, basis: &ISplineBasis, scale: f64, out: &mut [f64]) {
        for (index, value) in out[..self.saturated_end].iter_mut().enumerate() {
            if basis.nondegenerate[index] {
                *value += scale;
            }
        }
        for (offset, weight) in self
            .partial_weights
            .iter()
            .copied()
            .enumerate()
            .take(self.partial_len())
        {
            let index = self.saturated_end + offset;
            out[index] = scale.mul_add(weight, out[index]);
        }
    }

    #[inline]
    fn add_scaled_difference(self, scale: f64, out: &mut [f64]) {
        if self.saturated_end != 0 {
            out[0] += scale;
            if self.saturated_end < out.len() {
                out[self.saturated_end] -= scale;
            }
        }
        for (offset, weight) in self
            .partial_weights
            .iter()
            .copied()
            .enumerate()
            .take(self.partial_len())
        {
            let index = self.saturated_end + offset;
            let value = scale * weight;
            out[index] += value;
            if index + 1 < out.len() {
                out[index + 1] -= value;
            }
        }
    }
}

/// I-spline predictor with compact prepared row geometry.
///
/// A row stores the number of already saturated unit basis functions and at
/// most `degree + 1` partial weights. This avoids both an `nrows × n_basis`
/// matrix and numerical integration during repeated predictor passes.
#[derive(Debug, Clone, PartialEq)]
pub struct ISplineDesign {
    x: Box<[f64]>,
    rows: Box<[PreparedISplineRow]>,
    basis: ISplineBasis,
}

impl ISplineDesign {
    pub(crate) fn from_owned_basis(x: &[f64], basis: ISplineBasis) -> Result<Self, SplineError> {
        validate_coordinates(x)?;
        let rows = x
            .iter()
            .copied()
            .map(|value| basis.prepare_row(value))
            .collect();
        Ok(Self {
            x: x.into(),
            rows,
            basis,
        })
    }

    /// Returns the basis metadata.
    #[must_use]
    #[inline]
    pub const fn basis(&self) -> &ISplineBasis {
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

        let mut value = 0.0;
        self.basis
            .mspline
            .for_each_basis(self.x[row], |index, weight| {
                value = beta[index].mul_add(weight, value);
            });
        value
    }

    #[inline]
    fn add_scores_range<M>(
        &self,
        rows: Range<usize>,
        scores: &[f64],
        multiplier: &M,
        out: &mut [f64],
    ) where
        M: RowMultiplier + ?Sized,
    {
        if self.basis.all_nondegenerate {
            for index in (1..out.len()).rev() {
                out[index] -= out[index - 1];
            }
            for (offset, score) in scores.iter().copied().enumerate() {
                if score == 0.0 {
                    continue;
                }
                let row = rows.start + offset;
                let scaled_score = score * multiplier.multiplier_at(row);
                if scaled_score != 0.0 {
                    self.rows[row].add_scaled_difference(scaled_score, out);
                }
            }
            for index in 1..out.len() {
                out[index] += out[index - 1];
            }
        } else {
            for (offset, score) in scores.iter().copied().enumerate() {
                if score == 0.0 {
                    continue;
                }
                let row = rows.start + offset;
                let scaled_score = score * multiplier.multiplier_at(row);
                if scaled_score != 0.0 {
                    self.rows[row].add_scaled(&self.basis, scaled_score, out);
                }
            }
        }
    }
}

impl SplineRowBasis for ISplineDesign {
    #[inline]
    fn nrows(&self) -> usize {
        self.rows.len()
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

impl PredictorBlock for ISplineDesign {
    #[inline]
    fn nrows(&self) -> usize {
        self.rows.len()
    }

    #[inline]
    fn nparams(&self) -> usize {
        self.basis.n_basis()
    }

    #[inline]
    fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
        debug_assert!(row < self.rows.len());
        debug_assert_eq!(beta.len(), self.basis.n_basis());
        self.rows[row].dot(&self.basis, beta)
    }

    #[inline]
    fn zero_beta_constant_contribution(&self) -> Option<f64> {
        Some(0.0)
    }

    #[inline]
    fn add_gradient_range(&self, rows: Range<usize>, scores: &[f64], _: &[f64], grad: &mut [f64]) {
        debug_assert!(rows.end <= self.rows.len());
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
        debug_assert!(rows.end <= self.rows.len());
        debug_assert_eq!(scores.len(), rows.len());
        debug_assert_eq!(grad.len(), self.basis.n_basis());
        self.add_scores_range(rows, scores, multiplier, grad);
    }
}

impl LinearPredictorGeometry for ISplineDesign {
    #[inline]
    fn add_weighted_gram(&self, row_weights: &[f64], out: &mut [f64]) -> Result<(), ModelError> {
        let nparams = self.basis.n_basis();
        validate_gram_lengths(self.rows.len(), nparams, row_weights, out)?;
        for (row, scale) in row_weights.iter().copied().enumerate() {
            if scale == 0.0 {
                continue;
            }
            self.for_each_row_basis(row, |left_index, left_weight| {
                let scaled_left = scale * left_weight;
                self.for_each_row_basis(row, |right_index, right_weight| {
                    let index = left_index * nparams + right_index;
                    out[index] = scaled_left.mul_add(right_weight, out[index]);
                });
            });
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
        let nparams = self.basis.n_basis();
        validate_gram_lengths(self.rows.len(), nparams, row_weights, out)?;
        for (row, weight) in row_weights.iter().copied().enumerate() {
            if weight == 0.0 {
                continue;
            }
            let scale = weight * multiplier.multiplier_at(row);
            if scale == 0.0 {
                continue;
            }
            self.for_each_row_basis(row, |left_index, left_weight| {
                let scaled_left = scale * left_weight;
                self.for_each_row_basis(row, |right_index, right_weight| {
                    let index = left_index * nparams + right_index;
                    out[index] = scaled_left.mul_add(right_weight, out[index]);
                });
            });
        }
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

fn integrate_interval(left: f64, right: f64, f: &impl Fn(f64) -> f64) -> f64 {
    if right <= left {
        return 0.0;
    }

    let midpoint = f64::midpoint(left, right);
    let half_width = 0.5 * (right - left);
    let weighted_sum = GAUSS_NODES
        .iter()
        .copied()
        .zip(GAUSS_WEIGHTS)
        .map(|(node, weight)| weight * f(half_width.mul_add(node, midpoint)))
        .sum::<f64>();
    half_width * weighted_sum
}
