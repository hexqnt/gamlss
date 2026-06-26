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
    /// Visits non-zero elements of the local basis.
    pub(crate) fn for_each(self, mut f: impl FnMut(usize, f64)) {
        for (&index, &weight) in self.indices[..self.len]
            .iter()
            .zip(&self.weights[..self.len])
        {
            f(index, weight);
        }
    }

    /// Dot product of the basis with coefficients.
    pub(crate) fn dot(self, beta: &[f64]) -> f64 {
        let mut value = 0.0;
        self.for_each(|index, weight| {
            value += beta[index] * weight;
        });
        value
    }

    /// Adds `scale * weights[i]` into `out[indices[i]]` for each non-zero
    /// element of the basis.
    pub(crate) fn add_scaled(self, scale: f64, out: &mut [f64]) {
        self.for_each(|index, weight| {
            out[index] += scale * weight;
        });
    }

    /// Adds `scale * self * self^T` into a row-major `nparams × nparams`
    /// matrix.
    pub(crate) fn add_scaled_outer(self, scale: f64, nparams: usize, out: &mut [f64]) {
        debug_assert_eq!(out.len(), nparams * nparams);

        for local_j in 0..self.len {
            let j = self.indices[local_j];
            let scaled_j = scale * self.weights[local_j];
            for local_k in local_j..self.len {
                let k = self.indices[local_k];
                let delta = scaled_j * self.weights[local_k];
                out[j * nparams + k] += delta;
                if k != j {
                    out[k * nparams + j] += delta;
                }
            }
        }
    }
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

/// Computes the local basis of a cyclic spline for phase `phi`.
///
/// `phi` is reduced to `[0, 1)` via `rem_euclid`.
pub fn cyclic_local_basis(phi: f64, order: SplineOrder, n_basis: usize) -> LocalBasis {
    let x = phi.rem_euclid(1.0) * n_basis as f64;
    let cell = x.floor() as usize;
    let u = x - cell as f64;
    let weights = spline_weights(order, u);
    let len = order.degree() + 1;
    let offset = if order == SplineOrder::Cubic {
        n_basis - 1
    } else {
        0
    };
    let mut basis = LocalBasis {
        len,
        ..LocalBasis::default()
    };
    for (idx, weight) in weights.iter().copied().enumerate().take(len) {
        basis.indices[idx] = (cell + offset + idx) % n_basis;
        basis.weights[idx] = weight;
    }
    basis
}

/// Linear extrapolation of the spline beyond the data range boundaries.
///
/// Uses the first/last two control coefficients to continue the spline beyond
/// `[0, 1)` while preserving continuity.
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

/// Finds the span (control point index) for an open-uniform spline via binary
/// search.
fn open_uniform_span(u: f64, n_basis: usize, degree: usize) -> usize {
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

/// Returns the normalized knot position for an open-uniform spline.
///
/// Knots are uniformly distributed between 0 and 1 with repeated boundary
/// knots.
fn open_uniform_knot(index: usize, n_basis: usize, degree: usize) -> f64 {
    if index <= degree {
        0.0
    } else if index >= n_basis {
        1.0
    } else {
        (index - degree) as f64 / (n_basis - degree) as f64
    }
}

/// Local spline weights (linear, quadratic, cubic) for parameter `u`.
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
