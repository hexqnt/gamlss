use gamlss_core::PredictorBlock;

use crate::{SplineError, SplineOrder, SplineRowBasis};

/// Truncated power regression spline basis.
///
/// For degree `p`, the coefficient order is:
/// `[intercept?, x, x^2, ..., x^p, (x-k_1)_+^p, ..., (x-k_m)_+^p]`.
/// The knot vector contains the truncated-power knots `k_i`; it must be finite
/// and strictly increasing.
#[derive(Debug, Clone, PartialEq)]
pub struct TruncatedPowerBasis {
    knots: Vec<f64>,
    order: SplineOrder,
    include_intercept: bool,
    n_basis: usize,
}

impl TruncatedPowerBasis {
    /// Creates a truncated power basis from strictly increasing knots.
    ///
    /// # Errors
    ///
    /// Returns [`SplineError::InvalidKnots`] if knots are non-finite or not
    /// strictly increasing. Returns [`SplineError::ParameterOverflow`] if the
    /// coefficient count overflows `usize`.
    pub fn new(
        knots: Vec<f64>,
        order: SplineOrder,
        include_intercept: bool,
    ) -> Result<Self, SplineError> {
        validate_strict_knots(&knots)?;
        let n_basis = coefficient_count(knots.len(), order, include_intercept)?;

        Ok(Self {
            knots,
            order,
            include_intercept,
            n_basis,
        })
    }

    /// Builds uniformly spaced truncated-power knots from the finite data range.
    ///
    /// `n_knots` is the number of interior truncated-power knots. Endpoints are
    /// not included in the returned knot vector.
    ///
    /// # Errors
    ///
    /// Returns an error if `x` is empty, contains non-finite values, has a
    /// degenerate range, or if the coefficient count overflows `usize`.
    pub fn uniform_from_data(
        x: &[f64],
        n_knots: usize,
        order: SplineOrder,
        include_intercept: bool,
    ) -> Result<Self, SplineError> {
        if x.is_empty() {
            return Err(SplineError::EmptyInput);
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

        let denominator = n_knots
            .checked_add(1)
            .ok_or(SplineError::ParameterOverflow)?;
        let step = (max - min) / denominator as f64;
        let knots = (1..=n_knots)
            .map(|index| min + step * index as f64)
            .collect::<Vec<_>>();
        Self::new(knots, order, include_intercept)
    }

    /// Builds a predictor design for concrete `x` coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`SplineError::NonFiniteValue`] if any input coordinate is not
    /// finite.
    pub fn design(&self, x: &[f64]) -> Result<TruncatedPowerDesign, SplineError> {
        if x.iter().any(|value| !value.is_finite()) {
            return Err(SplineError::NonFiniteValue);
        }

        Ok(TruncatedPowerDesign {
            x: x.to_vec(),
            basis: self.clone(),
        })
    }

    /// Truncated-power knots.
    #[must_use]
    #[inline(always)]
    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    /// Spline order.
    #[must_use]
    #[inline(always)]
    pub fn order(&self) -> SplineOrder {
        self.order
    }

    /// Returns `true` if the first coefficient is an intercept.
    #[must_use]
    #[inline(always)]
    pub fn include_intercept(&self) -> bool {
        self.include_intercept
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
        for value in out.iter_mut() {
            *value = 0.0;
        }
        self.for_each_basis(x, |index, weight| out[index] = weight);
    }

    /// Visits non-zero basis-function values at `x` without allocating.
    #[inline]
    pub fn for_each_basis(&self, x: f64, mut f: impl FnMut(usize, f64)) {
        let degree = self.order.degree();
        let mut offset = 0;
        if self.include_intercept {
            f(0, 1.0);
            offset = 1;
        }

        for power in 1..=degree {
            let weight = pow_usize(x, power);
            if weight != 0.0 {
                f(offset + power - 1, weight);
            }
        }
        offset += degree;

        for (knot_offset, knot) in self.knots.iter().copied().enumerate() {
            if x > knot {
                f(offset + knot_offset, pow_usize(x - knot, degree));
            }
        }
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

        for value in out.iter_mut() {
            *value = 0.0;
        }
        self.for_each_derivative_basis(x, |index, weight| out[index] = weight);
    }

    /// Visits non-zero first derivatives at `x` without allocating.
    #[inline]
    pub fn for_each_derivative_basis(&self, x: f64, mut f: impl FnMut(usize, f64)) {
        let degree = self.order.degree();
        let mut offset = 0;
        if self.include_intercept {
            offset = 1;
        }

        for power in 1..=degree {
            let weight = if power == 1 {
                1.0
            } else {
                power as f64 * pow_usize(x, power - 1)
            };
            if weight != 0.0 {
                f(offset + power - 1, weight);
            }
        }
        offset += degree;

        for (knot_offset, knot) in self.knots.iter().copied().enumerate() {
            if x > knot {
                f(
                    offset + knot_offset,
                    degree as f64 * pow_usize(x - knot, degree - 1),
                );
            }
        }
    }
}

/// Truncated power spline predictor.
#[derive(Debug, Clone, PartialEq)]
pub struct TruncatedPowerDesign {
    x: Vec<f64>,
    basis: TruncatedPowerBasis,
}

impl TruncatedPowerDesign {
    /// Builds a design from data-derived uniform truncated-power knots.
    pub fn uniform_from_data(
        x: &[f64],
        n_knots: usize,
        order: SplineOrder,
        include_intercept: bool,
    ) -> Result<Self, SplineError> {
        TruncatedPowerBasis::uniform_from_data(x, n_knots, order, include_intercept)?.design(x)
    }

    /// Returns the basis metadata.
    #[must_use]
    #[inline(always)]
    pub fn basis(&self) -> &TruncatedPowerBasis {
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

impl SplineRowBasis for TruncatedPowerDesign {
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.x.len()
    }

    #[inline(always)]
    fn nparams(&self) -> usize {
        self.basis.n_basis()
    }

    #[inline]
    fn for_each_row_basis(&self, row: usize, f: impl FnMut(usize, f64)) {
        debug_assert!(row < self.x.len());
        self.basis.for_each_basis(self.x[row], f);
    }
}

impl PredictorBlock for TruncatedPowerDesign {
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
            if score == 0.0 {
                continue;
            }
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

        for (row, score) in scores.iter().copied().enumerate() {
            if score == 0.0 {
                continue;
            }
            let scaled_score = score * multiplier[row];
            if scaled_score == 0.0 {
                continue;
            }
            self.for_each_row_basis(row, |index, weight| {
                grad[index] += scaled_score * weight;
            });
        }
    }
}

fn validate_strict_knots(knots: &[f64]) -> Result<(), SplineError> {
    let mut previous = None;
    for knot in knots.iter().copied() {
        if !knot.is_finite() || previous.is_some_and(|previous| previous >= knot) {
            return Err(SplineError::InvalidKnots);
        }
        previous = Some(knot);
    }
    Ok(())
}

fn coefficient_count(
    n_knots: usize,
    order: SplineOrder,
    include_intercept: bool,
) -> Result<usize, SplineError> {
    order
        .degree()
        .checked_add(usize::from(include_intercept))
        .and_then(|count| count.checked_add(n_knots))
        .ok_or(SplineError::ParameterOverflow)
}

#[inline]
fn pow_usize(value: f64, power: usize) -> f64 {
    (0..power).fold(1.0, |product, _| product * value)
}
