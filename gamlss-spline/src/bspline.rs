use gamlss_core::DenseDesign;

use crate::{OnDemandSplineDesign, SplineError};

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
        Self::new(degree, knots)
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
        let Some(active) = self.active_basis_range(x) else {
            return;
        };
        for index in active {
            let weight = self.basis_value(index, self.degree, x);
            if weight != 0.0 {
                f(index, weight);
            }
        }
    }

    /// Dense design matrix where each row contains `evaluate(x_i)`.
    pub fn design_matrix(&self, x: &[f64]) -> Result<DenseDesign, SplineError> {
        if x.iter().any(|value| !value.is_finite()) {
            return Err(SplineError::NonFiniteValue);
        }

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

        for (index, value) in out.iter_mut().enumerate() {
            *value = self.basis_value(index, self.degree, x);
        }
    }

    #[allow(clippy::float_cmp)]
    fn active_basis_range(&self, x: f64) -> Option<std::ops::RangeInclusive<usize>> {
        let n_basis = self.n_basis();
        let mut first = n_basis;
        let mut last = 0;
        let mut found = false;

        let upper = self.knots.partition_point(|knot| *knot <= x);
        if upper > 0 && upper < self.knots.len() {
            let interval = upper - 1;
            first = interval.saturating_sub(self.degree).min(n_basis);
            last = interval.min(n_basis - 1);
            found = first <= last;
        }

        let closed_interval = n_basis - 1;
        if x == self.knots[n_basis] {
            let closed_first = closed_interval.saturating_sub(self.degree);
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

    #[allow(clippy::float_cmp, clippy::suboptimal_flops)]
    fn basis_value(&self, index: usize, degree: usize, x: f64) -> f64 {
        if degree == 0 {
            let left = self.knots[index];
            let right = self.knots[index + 1];
            let is_last_basis = index + 1 == self.n_basis();
            if (left <= x && x < right) || (is_last_basis && x == right) {
                1.0
            } else {
                0.0
            }
        } else {
            let mut value = 0.0;
            let left_denom = self.knots[index + degree] - self.knots[index];
            if left_denom > 0.0 {
                value +=
                    (x - self.knots[index]) / left_denom * self.basis_value(index, degree - 1, x);
            }

            let right_denom = self.knots[index + degree + 1] - self.knots[index + 1];
            if right_denom > 0.0 {
                value += (self.knots[index + degree + 1] - x) / right_denom
                    * self.basis_value(index + 1, degree - 1, x);
            }

            value
        }
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
