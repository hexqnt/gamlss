use std::ops::Range;

use crate::SplineOrder;

/// Local basis for one row — a compact sparse representation.
///
/// Stores up to 4 non-zero indices and weights, sufficient for a cubic spline.
/// Used in the `eta_row` and `add_gradient` methods of predictors.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LocalBasis {
    indices: [usize; 4],
    weights: [f64; 4],
    len: usize,
}

impl LocalBasis {
    #[inline]
    pub(crate) fn push_nonzero(&mut self, index: usize, weight: f64) {
        if weight != 0.0 {
            debug_assert!(self.len < self.indices.len());
            self.indices[self.len] = index;
            self.weights[self.len] = weight;
            self.len += 1;
        }
    }

    /// Visits non-zero elements of the local basis.
    #[inline]
    pub(crate) fn for_each(self, mut f: impl FnMut(usize, f64)) {
        match self.len {
            0 => {}
            1 => f(self.indices[0], self.weights[0]),
            2 => {
                f(self.indices[0], self.weights[0]);
                f(self.indices[1], self.weights[1]);
            }
            3 => {
                f(self.indices[0], self.weights[0]);
                f(self.indices[1], self.weights[1]);
                f(self.indices[2], self.weights[2]);
            }
            4 => {
                f(self.indices[0], self.weights[0]);
                f(self.indices[1], self.weights[1]);
                f(self.indices[2], self.weights[2]);
                f(self.indices[3], self.weights[3]);
            }
            _ => unreachable!("local spline basis stores at most four entries"),
        }
    }

    /// Dot product of the basis with coefficients.
    #[inline]
    pub(crate) fn dot(self, beta: &[f64]) -> f64 {
        match self.len {
            0 => 0.0,
            1 => beta[self.indices[0]].mul_add(self.weights[0], 0.0),
            2 => {
                let value = beta[self.indices[0]].mul_add(self.weights[0], 0.0);
                beta[self.indices[1]].mul_add(self.weights[1], value)
            }
            3 => {
                let value = beta[self.indices[0]].mul_add(self.weights[0], 0.0);
                let value = beta[self.indices[1]].mul_add(self.weights[1], value);
                beta[self.indices[2]].mul_add(self.weights[2], value)
            }
            4 => {
                let value = beta[self.indices[0]].mul_add(self.weights[0], 0.0);
                let value = beta[self.indices[1]].mul_add(self.weights[1], value);
                let value = beta[self.indices[2]].mul_add(self.weights[2], value);
                beta[self.indices[3]].mul_add(self.weights[3], value)
            }
            _ => unreachable!("local spline basis stores at most four entries"),
        }
    }

    #[inline]
    pub(crate) fn add_scaled_outer(self, scale: f64, nparams: usize, out: &mut [f64]) {
        debug_assert_eq!(out.len(), nparams * nparams);

        self.for_each(|left_index, left_weight| {
            let scaled_left = scale * left_weight;
            self.for_each(|right_index, right_weight| {
                let index = left_index * nparams + right_index;
                out[index] = scaled_left.mul_add(right_weight, out[index]);
            });
        });
    }
}

/// Compact prepared geometry for one spline row.
///
/// The design supplies the active weight range and whether indices wrap. Keeping only one start index and four weights avoids storing four `usize` indices per observation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreparedLocalBasis {
    start: usize,
    weights: [f64; 4],
}

impl PreparedLocalBasis {
    #[inline]
    fn from_contiguous(local: LocalBasis, active: Range<usize>, n_basis: usize) -> Self {
        debug_assert!(active.start < active.end && active.end <= 4);
        debug_assert_eq!(local.len, active.len());
        debug_assert!(local.indices[0] >= active.start);

        let start = local.indices[0] - active.start;
        debug_assert!(
            (0..local.len).all(|offset| local.indices[offset] == start + active.start + offset)
        );
        debug_assert!(start + active.end <= n_basis);

        let mut weights = [0.0; 4];
        weights[active].copy_from_slice(&local.weights[..local.len]);
        Self { start, weights }
    }

    #[inline]
    fn for_each_offset(&self, active: Range<usize>, mut f: impl FnMut(usize, f64)) {
        debug_assert!(active.start < active.end && active.end <= self.weights.len());

        match (active.start, active.end) {
            (0, 2) => {
                f(0, self.weights[0]);
                f(1, self.weights[1]);
            }
            (0, 3) => {
                f(0, self.weights[0]);
                f(1, self.weights[1]);
                f(2, self.weights[2]);
            }
            (0, 4) => {
                f(0, self.weights[0]);
                f(1, self.weights[1]);
                f(2, self.weights[2]);
                f(3, self.weights[3]);
            }
            (1, 3) => {
                f(1, self.weights[1]);
                f(2, self.weights[2]);
            }
            (2, 4) => {
                f(2, self.weights[2]);
                f(3, self.weights[3]);
            }
            _ => unreachable!("unsupported prepared spline row layout"),
        }
    }

    #[inline]
    pub(crate) fn for_each_contiguous(&self, active: Range<usize>, mut f: impl FnMut(usize, f64)) {
        self.for_each_offset(active, |offset, weight| {
            f(self.start + offset, weight);
        });
    }

    #[inline]
    fn for_each_wrapped_fallback(
        &self,
        width: usize,
        remaining: usize,
        mut f: impl FnMut(usize, f64),
    ) {
        debug_assert!(remaining < width);

        self.for_each_offset(0..width, |offset, weight| {
            let index = if offset < remaining {
                self.start + offset
            } else {
                offset - remaining
            };
            f(index, weight);
        });
    }

    #[inline]
    pub(crate) fn for_each_wrapped(&self, width: usize, n_basis: usize, f: impl FnMut(usize, f64)) {
        debug_assert!(self.start < n_basis);
        debug_assert!(width <= n_basis);

        let remaining = n_basis - self.start;
        if width <= remaining {
            self.for_each_contiguous(0..width, f);
            return;
        }

        self.for_each_wrapped_fallback(width, remaining, f);
    }

    #[inline]
    pub(crate) fn dot_contiguous(&self, active: Range<usize>, beta: &[f64]) -> f64 {
        debug_assert!(active.start < active.end && active.end <= self.weights.len());

        match (active.start, active.end) {
            (0, 2) => {
                let [beta0, beta1] = &beta[self.start..self.start + 2] else {
                    unreachable!("two-element coefficient slice")
                };
                let value = beta0.mul_add(self.weights[0], 0.0);
                beta1.mul_add(self.weights[1], value)
            }
            (0, 3) => {
                let [beta0, beta1, beta2] = &beta[self.start..self.start + 3] else {
                    unreachable!("three-element coefficient slice")
                };
                let value = beta0.mul_add(self.weights[0], 0.0);
                let value = beta1.mul_add(self.weights[1], value);
                beta2.mul_add(self.weights[2], value)
            }
            (0, 4) => {
                let [beta0, beta1, beta2, beta3] = &beta[self.start..self.start + 4] else {
                    unreachable!("four-element coefficient slice")
                };
                let value = beta0.mul_add(self.weights[0], 0.0);
                let value = beta1.mul_add(self.weights[1], value);
                let value = beta2.mul_add(self.weights[2], value);
                beta3.mul_add(self.weights[3], value)
            }
            (1, 3) => {
                let start = self.start + 1;
                let [beta1, beta2] = &beta[start..start + 2] else {
                    unreachable!("two-element coefficient slice")
                };
                let value = beta1.mul_add(self.weights[1], 0.0);
                beta2.mul_add(self.weights[2], value)
            }
            (2, 4) => {
                let start = self.start + 2;
                let [beta2, beta3] = &beta[start..start + 2] else {
                    unreachable!("two-element coefficient slice")
                };
                let value = beta2.mul_add(self.weights[2], 0.0);
                beta3.mul_add(self.weights[3], value)
            }
            _ => unreachable!("unsupported prepared spline row layout"),
        }
    }

    #[inline]
    pub(crate) fn dot_wrapped(&self, width: usize, n_basis: usize, beta: &[f64]) -> f64 {
        debug_assert!(self.start < n_basis);
        debug_assert!(width <= n_basis);

        let remaining = n_basis - self.start;
        if width <= remaining {
            return self.dot_contiguous(0..width, beta);
        }

        match width {
            2 => {
                let value = beta[self.start].mul_add(self.weights[0], 0.0);
                beta[wrapped_index(self.start, 1, n_basis)].mul_add(self.weights[1], value)
            }
            3 => {
                let value = beta[self.start].mul_add(self.weights[0], 0.0);
                let value =
                    beta[wrapped_index(self.start, 1, n_basis)].mul_add(self.weights[1], value);
                beta[wrapped_index(self.start, 2, n_basis)].mul_add(self.weights[2], value)
            }
            4 => {
                let value = beta[self.start].mul_add(self.weights[0], 0.0);
                let value =
                    beta[wrapped_index(self.start, 1, n_basis)].mul_add(self.weights[1], value);
                let value =
                    beta[wrapped_index(self.start, 2, n_basis)].mul_add(self.weights[2], value);
                beta[wrapped_index(self.start, 3, n_basis)].mul_add(self.weights[3], value)
            }
            _ => unreachable!("prepared cyclic spline row has width two through four"),
        }
    }

    #[inline]
    pub(crate) fn add_scaled_contiguous(&self, active: Range<usize>, scale: f64, out: &mut [f64]) {
        debug_assert!(active.start < active.end && active.end <= self.weights.len());

        match (active.start, active.end) {
            (0, 2) => {
                let [out0, out1] = &mut out[self.start..self.start + 2] else {
                    unreachable!("two-element output slice")
                };
                *out0 = scale.mul_add(self.weights[0], *out0);
                *out1 = scale.mul_add(self.weights[1], *out1);
            }
            (0, 3) => {
                let [out0, out1, out2] = &mut out[self.start..self.start + 3] else {
                    unreachable!("three-element output slice")
                };
                *out0 = scale.mul_add(self.weights[0], *out0);
                *out1 = scale.mul_add(self.weights[1], *out1);
                *out2 = scale.mul_add(self.weights[2], *out2);
            }
            (0, 4) => {
                let [out0, out1, out2, out3] = &mut out[self.start..self.start + 4] else {
                    unreachable!("four-element output slice")
                };
                *out0 = scale.mul_add(self.weights[0], *out0);
                *out1 = scale.mul_add(self.weights[1], *out1);
                *out2 = scale.mul_add(self.weights[2], *out2);
                *out3 = scale.mul_add(self.weights[3], *out3);
            }
            (1, 3) => {
                let start = self.start + 1;
                let [out1, out2] = &mut out[start..start + 2] else {
                    unreachable!("two-element output slice")
                };
                *out1 = scale.mul_add(self.weights[1], *out1);
                *out2 = scale.mul_add(self.weights[2], *out2);
            }
            (2, 4) => {
                let start = self.start + 2;
                let [out2, out3] = &mut out[start..start + 2] else {
                    unreachable!("two-element output slice")
                };
                *out2 = scale.mul_add(self.weights[2], *out2);
                *out3 = scale.mul_add(self.weights[3], *out3);
            }
            _ => unreachable!("unsupported prepared spline row layout"),
        }
    }

    #[inline]
    pub(crate) fn add_scaled_wrapped(
        &self,
        width: usize,
        n_basis: usize,
        scale: f64,
        out: &mut [f64],
    ) {
        debug_assert!(self.start < n_basis);
        debug_assert!(width <= n_basis);

        let remaining = n_basis - self.start;
        if width <= remaining {
            self.add_scaled_contiguous(0..width, scale, out);
            return;
        }

        match width {
            2 => {
                let index1 = wrapped_index(self.start, 1, n_basis);
                out[self.start] = scale.mul_add(self.weights[0], out[self.start]);
                out[index1] = scale.mul_add(self.weights[1], out[index1]);
            }
            3 => {
                let index1 = wrapped_index(self.start, 1, n_basis);
                let index2 = wrapped_index(self.start, 2, n_basis);
                out[self.start] = scale.mul_add(self.weights[0], out[self.start]);
                out[index1] = scale.mul_add(self.weights[1], out[index1]);
                out[index2] = scale.mul_add(self.weights[2], out[index2]);
            }
            4 => {
                let index1 = wrapped_index(self.start, 1, n_basis);
                let index2 = wrapped_index(self.start, 2, n_basis);
                let index3 = wrapped_index(self.start, 3, n_basis);
                out[self.start] = scale.mul_add(self.weights[0], out[self.start]);
                out[index1] = scale.mul_add(self.weights[1], out[index1]);
                out[index2] = scale.mul_add(self.weights[2], out[index2]);
                out[index3] = scale.mul_add(self.weights[3], out[index3]);
            }
            _ => unreachable!("prepared cyclic spline row has width two through four"),
        }
    }

    #[inline]
    pub(crate) fn add_scaled_outer_contiguous(
        &self,
        active: Range<usize>,
        scale: f64,
        nparams: usize,
        out: &mut [f64],
    ) {
        debug_assert_eq!(out.len(), nparams * nparams);
        debug_assert!(self.start + active.end <= nparams);

        let end = active.end;
        for local_j in active {
            let j = self.start + local_j;
            let scaled_j = scale * self.weights[local_j];
            for local_k in local_j..end {
                let k = self.start + local_k;
                out[j * nparams + k] =
                    scaled_j.mul_add(self.weights[local_k], out[j * nparams + k]);
                if k != j {
                    out[k * nparams + j] =
                        scaled_j.mul_add(self.weights[local_k], out[k * nparams + j]);
                }
            }
        }
    }

    #[inline]
    pub(crate) fn add_scaled_outer_wrapped(
        &self,
        width: usize,
        nparams: usize,
        scale: f64,
        out: &mut [f64],
    ) {
        debug_assert_eq!(out.len(), nparams * nparams);
        debug_assert!(self.start < nparams);
        debug_assert!(width <= nparams);

        let remaining = nparams - self.start;
        if width <= remaining {
            self.add_scaled_outer_contiguous(0..width, scale, nparams, out);
            return;
        }

        for left_offset in 0..width {
            let left_index = if left_offset < remaining {
                self.start + left_offset
            } else {
                left_offset - remaining
            };
            let scaled_left = scale * self.weights[left_offset];
            for right_offset in left_offset..width {
                let right_index = if right_offset < remaining {
                    self.start + right_offset
                } else {
                    right_offset - remaining
                };
                let upper = left_index * nparams + right_index;
                out[upper] = scaled_left.mul_add(self.weights[right_offset], out[upper]);
                if right_index != left_index {
                    let lower = right_index * nparams + left_index;
                    out[lower] = scaled_left.mul_add(self.weights[right_offset], out[lower]);
                }
            }
        }
    }
}

/// Returns the degree-`degree` B-spline functions that can be non-zero at `x`.
///
/// This preserves the crate's existing edge semantics for arbitrary knot
/// vectors, including the partially supported rows outside the conventional
/// `[t_p, t_n]` parameter interval and the closed final degree-zero interval.
#[inline]
#[allow(clippy::float_cmp)]
pub fn bspline_active_range(
    knots: &[f64],
    n_basis: usize,
    degree: usize,
    x: f64,
) -> Option<std::ops::RangeInclusive<usize>> {
    let mut first = n_basis;
    let mut last = 0;
    let mut found = false;

    let upper = knots.partition_point(|knot| *knot <= x);
    if upper > 0 && upper < knots.len() {
        let interval = upper - 1;
        first = interval.saturating_sub(degree).min(n_basis);
        last = interval.min(n_basis - 1);
        found = first <= last;
    }

    let closed_interval = n_basis - 1;
    if x == knots[n_basis] {
        let closed_first = closed_interval.saturating_sub(degree);
        first = if found {
            first.min(closed_first)
        } else {
            closed_first
        };
        last = if found {
            last.max(closed_interval)
        } else {
            closed_interval
        };
        found = true;
    }

    found.then_some(first..=last)
}

/// Evaluates one B-spline basis function with Cox--de Boor recursion.
///
/// This remains the general fallback for degrees above three and the reference
/// implementation used by the fixed-width local evaluator at unusual edges.
#[allow(clippy::float_cmp, clippy::suboptimal_flops)]
pub fn bspline_value(knots: &[f64], n_basis: usize, index: usize, degree: usize, x: f64) -> f64 {
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
            value = ((x - knots[index]) / left_denom)
                .mul_add(bspline_value(knots, n_basis, index, degree - 1, x), value);
        }

        let right_denom = knots[index + degree + 1] - knots[index + 1];
        if right_denom > 0.0 {
            value = ((knots[index + degree + 1] - x) / right_denom).mul_add(
                bspline_value(knots, n_basis, index + 1, degree - 1, x),
                value,
            );
        }
        value
    }
}

/// Evaluates all non-zero degree-zero through cubic B-spline weights together.
///
/// Interior rows use the standard allocation-free `BasisFuns` recurrence in
/// `O(p^2)`. The uncommon partial-support edge rows use the recursive reference
/// kernel so arbitrary, non-clamped knot vectors retain their historical
/// semantics.
#[inline]
pub fn bspline_local_basis(knots: &[f64], n_basis: usize, degree: usize, x: f64) -> LocalBasis {
    debug_assert!(degree <= 3);

    let Some(active) = bspline_active_range(knots, n_basis, degree, x) else {
        return LocalBasis::default();
    };
    let first = *active.start();
    let last = *active.end();

    let upper = knots.partition_point(|knot| *knot <= x);
    let conventional_interior = upper > 0
        && upper < knots.len()
        && upper > degree
        && upper - 1 < n_basis
        && first == upper - 1 - degree
        && last == upper - 1;

    if conventional_interior {
        let span = upper - 1;
        let weights = arbitrary_bspline_basis_funs(knots, span, degree, x);
        let mut basis = LocalBasis::default();
        for (offset, weight) in weights.iter().copied().enumerate().take(degree + 1) {
            basis.push_nonzero(first + offset, weight);
        }
        return basis;
    }

    let mut basis = LocalBasis::default();
    for index in active {
        basis.push_nonzero(index, bspline_value(knots, n_basis, index, degree, x));
    }
    basis
}

#[inline]
fn arbitrary_bspline_basis_funs(knots: &[f64], span: usize, degree: usize, x: f64) -> [f64; 4] {
    let mut weights = [0.0; 4];
    let mut left = [0.0; 4];
    let mut right = [0.0; 4];
    weights[0] = 1.0;
    for j in 1..=degree {
        left[j] = x - knots[span + 1 - j];
        right[j] = knots[span + j] - x;
        let mut saved = 0.0;
        for r in 0..j {
            let denominator = right[r + 1] + left[j - r];
            let temp = if denominator == 0.0 {
                0.0
            } else {
                weights[r] / denominator
            };
            weights[r] = right[r + 1].mul_add(temp, saved);
            saved = left[j - r] * temp;
        }
        weights[j] = saved;
    }
    weights
}

/// Computes the local basis of an open-uniform spline for the normalized
/// coordinate `u` in the data range.
///
/// For `u <= 0` or `u >= 1` returns a linear extrapolation.
pub fn open_uniform_local_basis(
    u: f64,
    order: SplineOrder,
    n_basis: usize,
    n_intervals: f64,
) -> LocalBasis {
    let degree = order.degree();

    if u <= 0.0 {
        return edge_extrapolation_basis(u, degree, n_basis, n_intervals, false);
    }
    if u >= 1.0 {
        return edge_extrapolation_basis(u - 1.0, degree, n_basis, n_intervals, true);
    }

    let span = open_uniform_span(u, n_basis, degree);
    let weights = open_uniform_basis_funs(span, u, n_basis, degree);
    let start = span - degree;
    let mut basis = LocalBasis {
        len: degree + 1,
        ..LocalBasis::default()
    };
    for (offset, weight) in weights.iter().copied().enumerate().take(degree + 1) {
        basis.indices[offset] = start + offset;
        basis.weights[offset] = weight;
    }
    basis
}

#[inline]
pub fn prepare_open_uniform_local_basis(
    u: f64,
    order: SplineOrder,
    n_basis: usize,
    n_intervals: f64,
) -> PreparedLocalBasis {
    let local = open_uniform_local_basis(u, order, n_basis, n_intervals);
    let active = open_uniform_active_weights(u, order.degree() + 1);
    PreparedLocalBasis::from_contiguous(local, active, n_basis)
}

#[inline]
fn open_uniform_active_weights(u: f64, width: usize) -> Range<usize> {
    debug_assert!((2..=4).contains(&width));

    if u <= 0.0 {
        0..2
    } else if u >= 1.0 {
        width - 2..width
    } else {
        0..width
    }
}

/// Computes the derivative of the local open-uniform basis with respect to
/// normalized coordinate `u`.
pub fn open_uniform_local_basis_derivative(
    u: f64,
    order: SplineOrder,
    n_basis: usize,
    n_intervals: f64,
) -> LocalBasis {
    let degree = order.degree();

    if u <= 0.0 {
        return edge_extrapolation_basis_derivative(degree, n_basis, n_intervals, false);
    }
    if u >= 1.0 {
        return edge_extrapolation_basis_derivative(degree, n_basis, n_intervals, true);
    }

    let span = open_uniform_span(u, n_basis, degree);
    let start = span - degree;
    let mut basis = LocalBasis::default();
    for index in start..=span {
        let weight = open_uniform_basis_derivative_value(index, degree, u, n_basis, degree);
        basis.push_nonzero(index, weight);
    }
    basis
}

#[inline]
pub fn prepare_cyclic_local_basis(
    phi: f64,
    order: SplineOrder,
    n_basis: usize,
) -> PreparedLocalBasis {
    let (start, u) = cyclic_start_and_u(phi, order, n_basis);
    PreparedLocalBasis {
        start,
        weights: spline_weights(order, u),
    }
}

/// Computes the derivative of the local cyclic basis with respect to phase
/// `phi`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn cyclic_local_basis_derivative(phi: f64, order: SplineOrder, n_basis: usize) -> LocalBasis {
    let (start, u) = cyclic_start_and_u(phi, order, n_basis);
    let weights = spline_weight_derivatives(order, u);
    let len = order.degree() + 1;
    let scale = n_basis as f64;
    let mut basis = LocalBasis::default();
    for (idx, weight) in weights.iter().copied().enumerate().take(len) {
        basis.push_nonzero(wrapped_index(start, idx, n_basis), scale * weight);
    }
    basis
}

#[inline]
fn cyclic_start_and_u(phi: f64, order: SplineOrder, n_basis: usize) -> (usize, f64) {
    let (cell, u) = cyclic_cell_and_u(phi, n_basis);
    let start = if order == SplineOrder::Cubic {
        if cell == 0 { n_basis - 1 } else { cell - 1 }
    } else {
        cell
    };
    (start, u)
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn cyclic_cell_and_u(phi: f64, n_basis: usize) -> (usize, f64) {
    let x = phi.rem_euclid(1.0) * n_basis as f64;
    let cell_float = x.floor();
    (cell_float as usize, x - cell_float)
}

#[inline]
fn wrapped_index(start: usize, offset: usize, n_basis: usize) -> usize {
    debug_assert!(start < n_basis);
    debug_assert!(offset < n_basis);

    let remaining = n_basis - start;
    if offset >= remaining {
        offset - remaining
    } else {
        start + offset
    }
}

/// Linear extrapolation of the spline beyond the data range boundaries.
///
/// Uses the first/last two control coefficients to continue the spline beyond
/// `[0, 1)` while preserving continuity.
#[allow(clippy::cast_precision_loss)]
fn edge_extrapolation_basis(
    offset: f64,
    degree: usize,
    n_basis: usize,
    n_intervals: f64,
    right: bool,
) -> LocalBasis {
    let slope_scale = degree as f64 * n_intervals;
    if right {
        LocalBasis {
            indices: [n_basis - 2, n_basis - 1, 0, 0],
            weights: [
                -slope_scale * offset,
                slope_scale.mul_add(offset, 1.0),
                0.0,
                0.0,
            ],
            len: 2,
        }
    } else {
        LocalBasis {
            indices: [0, 1, 0, 0],
            weights: [
                slope_scale.mul_add(-offset, 1.0),
                slope_scale * offset,
                0.0,
                0.0,
            ],
            len: 2,
        }
    }
}

/// Derivative of the linear extrapolation basis with respect to normalized
/// coordinate `u`.
#[allow(clippy::cast_precision_loss)]
fn edge_extrapolation_basis_derivative(
    degree: usize,
    n_basis: usize,
    n_intervals: f64,
    right: bool,
) -> LocalBasis {
    let slope_scale = degree as f64 * n_intervals;
    if right {
        LocalBasis {
            indices: [n_basis - 2, n_basis - 1, 0, 0],
            weights: [-slope_scale, slope_scale, 0.0, 0.0],
            len: 2,
        }
    } else {
        LocalBasis {
            indices: [0, 1, 0, 0],
            weights: [-slope_scale, slope_scale, 0.0, 0.0],
            len: 2,
        }
    }
}

/// Finds the span (control point index) for an interior open-uniform coordinate
/// in constant time.
#[inline]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn open_uniform_span(u: f64, n_basis: usize, degree: usize) -> usize {
    debug_assert!(u > 0.0 && u < 1.0);
    debug_assert!(n_basis > degree);

    let n_intervals = n_basis - degree;
    let interval = (u * n_intervals as f64) as usize;
    let mut span = degree + interval.min(n_intervals - 1);

    // Division followed by multiplication is not guaranteed to recover an
    // integer exactly (for example, 15 / 22 * 22 rounds down). Correct the
    // estimate by at most one span so exact knots retain right-open semantics.
    if u < open_uniform_knot(span, n_basis, degree) {
        span -= 1;
    } else if u >= open_uniform_knot(span + 1, n_basis, degree) {
        span += 1;
    }

    debug_assert!(u >= open_uniform_knot(span, n_basis, degree));
    debug_assert!(u < open_uniform_knot(span + 1, n_basis, degree));
    span
}

/// Computes the weights of the B-spline basis functions in a given span.
///
/// Uses the Cox-de Boor recurrence algorithm.
fn open_uniform_basis_funs(span: usize, u: f64, n_basis: usize, degree: usize) -> [f64; 4] {
    let mut weights = [0.0; 4];
    let mut left = [0.0; 4];
    let mut right = [0.0; 4];
    weights[0] = 1.0;
    for j in 1..=degree {
        left[j] = u - open_uniform_knot(span + 1 - j, n_basis, degree);
        right[j] = open_uniform_knot(span + j, n_basis, degree) - u;
        let mut saved = 0.0;
        for r in 0..j {
            let denominator = right[r + 1] + left[j - r];
            let temp = if denominator == 0.0 {
                0.0
            } else {
                weights[r] / denominator
            };
            weights[r] = right[r + 1].mul_add(temp, saved);
            saved = left[j - r] * temp;
        }
        weights[j] = saved;
    }
    weights
}

#[allow(clippy::cast_precision_loss)]
fn open_uniform_basis_derivative_value(
    index: usize,
    degree: usize,
    u: f64,
    n_basis: usize,
    spline_degree: usize,
) -> f64 {
    debug_assert!(degree > 0);

    let mut value = 0.0;
    let left_denom = open_uniform_knot(index + degree, n_basis, spline_degree)
        - open_uniform_knot(index, n_basis, spline_degree);
    if left_denom > 0.0 {
        value = (degree as f64 / left_denom).mul_add(
            open_uniform_basis_value(index, degree - 1, u, n_basis, spline_degree),
            value,
        );
    }

    let right_denom = open_uniform_knot(index + degree + 1, n_basis, spline_degree)
        - open_uniform_knot(index + 1, n_basis, spline_degree);
    if right_denom > 0.0 {
        value = (degree as f64 / right_denom).mul_add(
            -open_uniform_basis_value(index + 1, degree - 1, u, n_basis, spline_degree),
            value,
        );
    }

    value
}

fn open_uniform_basis_value(
    index: usize,
    degree: usize,
    u: f64,
    n_basis: usize,
    spline_degree: usize,
) -> f64 {
    if degree == 0 {
        let left = open_uniform_knot(index, n_basis, spline_degree);
        let right = open_uniform_knot(index + 1, n_basis, spline_degree);
        return f64::from(left <= u && u < right);
    }

    let mut value = 0.0;
    let left_denom = open_uniform_knot(index + degree, n_basis, spline_degree)
        - open_uniform_knot(index, n_basis, spline_degree);
    if left_denom > 0.0 {
        value = ((u - open_uniform_knot(index, n_basis, spline_degree)) / left_denom).mul_add(
            open_uniform_basis_value(index, degree - 1, u, n_basis, spline_degree),
            value,
        );
    }

    let right_denom = open_uniform_knot(index + degree + 1, n_basis, spline_degree)
        - open_uniform_knot(index + 1, n_basis, spline_degree);
    if right_denom > 0.0 {
        value = ((open_uniform_knot(index + degree + 1, n_basis, spline_degree) - u) / right_denom)
            .mul_add(
                open_uniform_basis_value(index + 1, degree - 1, u, n_basis, spline_degree),
                value,
            );
    }

    value
}

/// Returns the normalized knot position for an open-uniform spline.
///
/// Knots are uniformly distributed between 0 and 1 with repeated boundary
/// knots.
#[allow(clippy::cast_precision_loss)]
fn open_uniform_knot(index: usize, n_basis: usize, degree: usize) -> f64 {
    if index <= degree {
        0.0
    } else if index >= n_basis {
        1.0
    } else {
        (index - degree) as f64 / (n_basis - degree) as f64
    }
}

/// Local spline weight derivatives with respect to parameter `u`.
#[allow(clippy::suboptimal_flops)]
fn spline_weight_derivatives(order: SplineOrder, u: f64) -> [f64; 4] {
    match order {
        SplineOrder::Linear => [-1.0, 1.0, 0.0, 0.0],
        SplineOrder::Quadratic => [u - 1.0, 1.0 - 2.0 * u, u, 0.0],
        SplineOrder::Cubic => {
            let u2 = u * u;
            [
                -(1.0 - u) * (1.0 - u) / 2.0,
                1.5 * u2 - 2.0 * u,
                -1.5 * u2 + u + 0.5,
                u2 / 2.0,
            ]
        }
    }
}

/// Local spline weights (linear, quadratic, cubic) for parameter `u`.
#[allow(clippy::suboptimal_flops)]
fn spline_weights(order: SplineOrder, u: f64) -> [f64; 4] {
    match order {
        SplineOrder::Linear => [1.0 - u, u, 0.0, 0.0],
        SplineOrder::Quadratic => {
            let u2 = u * u;
            [
                (1.0 - u) * (1.0 - u) / 2.0,
                (1.0 + 2.0 * u - 2.0 * u2) / 2.0,
                u2 / 2.0,
                0.0,
            ]
        }
        SplineOrder::Cubic => {
            let mu = 1.0 - u;
            let mu2 = mu * mu;
            let mu3 = mu2 * mu;
            let u2 = u * u;
            let u3 = u2 * u;
            [
                mu3 / 6.0,
                (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0,
                (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0,
                u3 / 6.0,
            ]
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use super::{
        LocalBasis, PreparedLocalBasis, bspline_local_basis, bspline_value,
        open_uniform_active_weights, open_uniform_knot, open_uniform_local_basis,
        open_uniform_span, prepare_cyclic_local_basis, prepare_open_uniform_local_basis,
        spline_weights, wrapped_index,
    };
    use crate::SplineOrder;

    fn collect_local(local: LocalBasis) -> Vec<(usize, f64)> {
        let mut values = Vec::new();
        local.for_each(|index, weight| values.push((index, weight)));
        values
    }

    #[test]
    fn arbitrary_local_bspline_matches_recursive_reference() {
        let cases: [(usize, Vec<f64>); 5] = [
            (0, vec![-1.0, 0.0, 0.5, 2.0]),
            (1, vec![0.0, 0.0, 0.3, 0.7, 1.0, 1.0]),
            (2, vec![0.0, 0.0, 0.0, 0.4, 1.0, 1.0, 1.0]),
            (2, vec![-1.0, 0.0, 0.5, 1.5, 2.0, 3.0]),
            (3, vec![0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]),
        ];

        for (degree, knots) in cases {
            let n_basis = knots.len() - degree - 1;
            let mut points = vec![-2.0, 4.0];
            for knot in knots.iter().copied() {
                points.extend([knot.next_down(), knot, knot.next_up()]);
            }

            for x in points {
                let mut actual = vec![0.0; n_basis];
                bspline_local_basis(&knots, n_basis, degree, x)
                    .for_each(|index, weight| actual[index] = weight);
                let expected = (0..n_basis)
                    .map(|index| bspline_value(&knots, n_basis, index, degree, x))
                    .collect::<Vec<_>>();
                assert!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(actual, expected)| (actual - expected).abs() <= 1.0e-14),
                    "degree={degree}, knots={knots:?}, x={x:?}, actual={actual:?}, expected={expected:?}"
                );
            }
        }
    }

    fn assert_matches_binary_search(u: f64, n_basis: usize, degree: usize) {
        assert_eq!(
            open_uniform_span(u, n_basis, degree),
            binary_search_span(u, n_basis, degree),
            "degree={degree}, n_basis={n_basis}, u={u:?}",
        );
    }

    fn binary_search_span(u: f64, n_basis: usize, degree: usize) -> usize {
        let last_control = n_basis - 1;
        let mut low = degree;
        let mut high = n_basis;
        let mut mid = usize::midpoint(low, high);
        while u < open_uniform_knot(mid, n_basis, degree)
            || u >= open_uniform_knot(mid + 1, n_basis, degree)
        {
            if u < open_uniform_knot(mid, n_basis, degree) {
                high = mid;
            } else {
                low = mid;
            }
            mid = usize::midpoint(low, high);
        }
        mid.min(last_control)
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn constant_time_span_matches_binary_search_at_knots_and_between_them() {
        for degree in 1..=3 {
            for n_intervals in 1..=64 {
                let n_basis = degree + n_intervals;

                for u in [0.0_f64.next_up(), 1.0_f64.next_down()] {
                    assert_matches_binary_search(u, n_basis, degree);
                }

                for numerator in 1..n_intervals {
                    let knot = numerator as f64 / n_intervals as f64;
                    for u in [knot.next_down(), knot, knot.next_up()] {
                        assert_matches_binary_search(u, n_basis, degree);
                    }
                }

                for numerator in 1..1024 {
                    let u = f64::from(numerator) / 1024.0;
                    assert_matches_binary_search(u, n_basis, degree);
                }
            }
        }
    }

    #[test]
    fn constant_time_span_preserves_right_open_knot_semantics_after_rounding() {
        let degree = 3;
        let n_intervals = 22;
        let n_basis = degree + n_intervals;
        let knot = open_uniform_knot(degree + 15, n_basis, degree);

        assert!(knot * 22.0 < 15.0);
        assert_eq!(open_uniform_span(knot, n_basis, degree), degree + 15);
    }

    #[test]
    fn prepared_geometry_is_smaller_than_expanded_local_basis() {
        assert!(size_of::<PreparedLocalBasis>() < size_of::<LocalBasis>());

        #[cfg(target_pointer_width = "64")]
        {
            assert_eq!(size_of::<PreparedLocalBasis>(), 40);
            assert_eq!(size_of::<LocalBasis>(), 72);
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss, clippy::float_cmp)]
    fn prepared_open_rows_preserve_indices_and_weights() {
        for order in [
            SplineOrder::Linear,
            SplineOrder::Quadratic,
            SplineOrder::Cubic,
        ] {
            let width = order.degree() + 1;
            for n_basis in order.min_basis()..=order.min_basis() + 6 {
                let n_intervals = (n_basis - order.degree()) as f64;
                let mut coordinates = vec![
                    -0.5,
                    0.0,
                    0.0_f64.next_up(),
                    0.125,
                    0.5,
                    1.0_f64.next_down(),
                    1.0,
                    1.5,
                ];
                for knot in 1..n_basis - order.degree() {
                    coordinates.push(knot as f64 / n_intervals);
                }

                for u in coordinates {
                    let expected =
                        collect_local(open_uniform_local_basis(u, order, n_basis, n_intervals));
                    let prepared = prepare_open_uniform_local_basis(u, order, n_basis, n_intervals);
                    let active = open_uniform_active_weights(u, width);
                    let mut actual = Vec::new();
                    prepared.for_each_contiguous(active.clone(), |index, weight| {
                        actual.push((index, weight));
                    });

                    assert_eq!(
                        actual, expected,
                        "order={order:?}, n_basis={n_basis}, u={u:?}"
                    );

                    let beta = (0..n_basis)
                        .map(|index| (index as f64).mul_add(0.125, -0.25))
                        .collect::<Vec<_>>();
                    let expected_dot = expected.iter().fold(0.0, |value, &(index, weight)| {
                        beta[index].mul_add(weight, value)
                    });
                    assert_eq!(
                        prepared.dot_contiguous(active.clone(), &beta),
                        expected_dot,
                        "dot product: order={order:?}, n_basis={n_basis}, u={u:?}"
                    );

                    let scale = 0.7_f64;
                    let mut expected_gradient = vec![0.5; n_basis];
                    for &(index, weight) in &expected {
                        expected_gradient[index] = scale.mul_add(weight, expected_gradient[index]);
                    }
                    let mut actual_gradient = vec![0.5; n_basis];
                    prepared.add_scaled_contiguous(active.clone(), scale, &mut actual_gradient);
                    assert_eq!(
                        actual_gradient, expected_gradient,
                        "gradient: order={order:?}, n_basis={n_basis}, u={u:?}"
                    );

                    let mut expected_outer = vec![0.0; n_basis * n_basis];
                    for (local_j, &(j, weight_j)) in expected.iter().enumerate() {
                        let scaled_j = scale * weight_j;
                        for &(k, weight_k) in &expected[local_j..] {
                            let value = scaled_j * weight_k;
                            expected_outer[j * n_basis + k] = value;
                            expected_outer[k * n_basis + j] = value;
                        }
                    }
                    let mut actual_outer = vec![0.0; n_basis * n_basis];
                    prepared.add_scaled_outer_contiguous(active, scale, n_basis, &mut actual_outer);
                    assert_eq!(
                        actual_outer, expected_outer,
                        "outer product: order={order:?}, n_basis={n_basis}, u={u:?}"
                    );
                }
            }
        }
    }

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::float_cmp
    )]
    fn prepared_cyclic_rows_match_modulo_reference() {
        for order in [
            SplineOrder::Linear,
            SplineOrder::Quadratic,
            SplineOrder::Cubic,
        ] {
            let len = order.degree() + 1;
            for n_basis in order.min_basis()..=order.min_basis() + 6 {
                for phi in [
                    -2.25,
                    -1.0,
                    -0.75,
                    -0.001,
                    0.0,
                    0.001,
                    0.125,
                    0.5,
                    1.0_f64.next_down(),
                    1.0,
                    1.25,
                    3.7,
                ] {
                    let x = phi.rem_euclid(1.0) * n_basis as f64;
                    let cell = x.floor() as usize;
                    let u = x - x.floor();
                    let start_offset = if order == SplineOrder::Cubic {
                        n_basis - 1
                    } else {
                        0
                    };
                    let weights = spline_weights(order, u);
                    let expected = weights
                        .iter()
                        .copied()
                        .take(len)
                        .enumerate()
                        .map(|(offset, weight)| ((cell + start_offset + offset) % n_basis, weight))
                        .collect::<Vec<_>>();

                    let prepared = prepare_cyclic_local_basis(phi, order, n_basis);
                    let mut actual = Vec::new();
                    prepared.for_each_wrapped(len, n_basis, |index, weight| {
                        actual.push((index, weight));
                    });

                    assert_eq!(
                        actual, expected,
                        "order={order:?}, n_basis={n_basis}, phi={phi:?}"
                    );

                    let beta = (0..n_basis)
                        .map(|index| (index as f64).mul_add(0.125, -0.25))
                        .collect::<Vec<_>>();
                    let expected_dot = expected.iter().fold(0.0, |value, &(index, weight)| {
                        beta[index].mul_add(weight, value)
                    });
                    assert_eq!(
                        prepared.dot_wrapped(len, n_basis, &beta),
                        expected_dot,
                        "dot product: order={order:?}, n_basis={n_basis}, phi={phi:?}"
                    );

                    let scale = -0.75_f64;
                    let mut expected_gradient = vec![0.5; n_basis];
                    for &(index, weight) in &expected {
                        expected_gradient[index] = scale.mul_add(weight, expected_gradient[index]);
                    }
                    let mut actual_gradient = vec![0.5; n_basis];
                    prepared.add_scaled_wrapped(len, n_basis, scale, &mut actual_gradient);
                    assert_eq!(
                        actual_gradient, expected_gradient,
                        "gradient: order={order:?}, n_basis={n_basis}, phi={phi:?}"
                    );

                    let mut expected_outer = vec![0.25; n_basis * n_basis];
                    for (left_offset, &(left_index, left_weight)) in expected.iter().enumerate() {
                        let scaled_left = scale * left_weight;
                        for &(right_index, right_weight) in &expected[left_offset..] {
                            let upper = left_index * n_basis + right_index;
                            expected_outer[upper] =
                                scaled_left.mul_add(right_weight, expected_outer[upper]);
                            if right_index != left_index {
                                let lower = right_index * n_basis + left_index;
                                expected_outer[lower] =
                                    scaled_left.mul_add(right_weight, expected_outer[lower]);
                            }
                        }
                    }
                    let mut actual_outer = vec![0.25; n_basis * n_basis];
                    prepared.add_scaled_outer_wrapped(len, n_basis, scale, &mut actual_outer);
                    assert_eq!(
                        actual_outer, expected_outer,
                        "outer product: order={order:?}, n_basis={n_basis}, phi={phi:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn conditional_wrap_matches_modulo() {
        for n_basis in 2..=128 {
            for start in 0..n_basis {
                for offset in 0..n_basis {
                    assert_eq!(
                        wrapped_index(start, offset, n_basis),
                        (start + offset) % n_basis
                    );
                }
            }
        }
    }
}
