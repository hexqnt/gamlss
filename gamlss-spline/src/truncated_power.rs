use crate::numeric::pow_usize;
use crate::{KnotPlacement, OnDemandSplineDesign, OpenKnotVector, SplineError, SplineOrder};

/// Truncated-power predictor using the shared allocation-free on-demand engine.
pub type TruncatedPowerDesign = OnDemandSplineDesign<TruncatedPowerBasis>;

/// Truncated power regression spline basis.
///
/// For degree $p$ and strictly increasing knots $k_{1}<\cdots<k_{m}$, the basis is
///
/// $$
/// B(x)=\left(1,\ x,\ x^2,\ldots,x^p,
/// \max(x-k_{1},0)^p,\ldots,\max(x-k_{m},0)^p\right).
/// $$
///
/// With `include_intercept = true`, a coefficient vector ordered as $(\beta_0,\beta_1,\ldots,\beta_p,\gamma_1,\ldots,\gamma_m)$ represents
///
/// $$
/// \eta(x)=\beta_0+\sum_{r=1}^{p}\beta_r x^r
/// \mathbin{+}\sum_{j=1}^{m}\gamma_j\max(x-k_{j},0)^p.
/// $$
///
/// With `include_intercept = false`, the $\beta_0$ term and leading basis value are omitted, so the stored order begins with the coefficient of $x$. Thus [`TruncatedPowerBasis::n_basis`] is $p+m+1$ with an intercept and $p+m$ without one. Here $p=\mathtt{order.degree()}$ and $m=\mathtt{knots.len()}$. Away from a knot, $\frac{d}{dx}\max(x-k,0)^p=p\max(x-k,0)^{p-1}$ on the active side and zero otherwise.
#[allow(clippy::doc_markdown)]
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
    #[allow(clippy::cast_precision_loss)]
    pub fn uniform_from_data(
        x: &[f64],
        n_knots: usize,
        order: SplineOrder,
        include_intercept: bool,
    ) -> Result<Self, SplineError> {
        Self::from_data_with_placement(x, n_knots, order, include_intercept, KnotPlacement::Uniform)
    }

    /// Builds truncated-power knots using a uniform or quantile policy.
    pub fn from_data_with_placement(
        x: &[f64],
        n_knots: usize,
        order: SplineOrder,
        include_intercept: bool,
        placement: KnotPlacement,
    ) -> Result<Self, SplineError> {
        let n_basis = n_knots
            .checked_add(order.degree())
            .and_then(|value| value.checked_add(1))
            .ok_or(SplineError::ParameterOverflow)?;
        let knots = OpenKnotVector::from_data(x, n_basis, order.degree(), placement)?;
        Self::new(unique_interior_knots(&knots), order, include_intercept)
    }

    /// Builds truncated-power knots from weighted empirical quantiles.
    ///
    /// Zero-weight coordinates do not affect the fitted boundaries or knots.
    /// Repeated empirical quantiles collapse to one truncated-power column.
    pub fn from_weighted_data(
        x: &[f64],
        weights: &[f64],
        n_knots: usize,
        order: SplineOrder,
        include_intercept: bool,
    ) -> Result<Self, SplineError> {
        let n_basis = n_knots
            .checked_add(order.degree())
            .and_then(|value| value.checked_add(1))
            .ok_or(SplineError::ParameterOverflow)?;
        let knots = OpenKnotVector::from_weighted_data(x, weights, n_basis, order.degree())?;
        Self::new(unique_interior_knots(&knots), order, include_intercept)
    }

    /// Builds a truncated-power basis from persisted open-knot metadata.
    pub fn from_open_knots(
        knots: &OpenKnotVector,
        include_intercept: bool,
    ) -> Result<Self, SplineError> {
        let order = match knots.degree() {
            1 => SplineOrder::Linear,
            2 => SplineOrder::Quadratic,
            3 => SplineOrder::Cubic,
            degree => return Err(SplineError::UnsupportedDegree { degree }),
        };
        Self::new(unique_interior_knots(knots), order, include_intercept)
    }

    /// Builds a predictor design for concrete `x` coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`SplineError::NonFiniteValue`] if any input coordinate is not
    /// finite.
    pub fn design(&self, x: &[f64]) -> Result<TruncatedPowerDesign, SplineError> {
        self.on_demand_design(x)
    }

    /// Builds the generic low-memory spline design for concrete coordinates.
    ///
    /// This is equivalent to [`Self::design`] but exposes the shared
    /// [`OnDemandSplineDesign`] representation directly.
    ///
    /// # Errors
    ///
    /// Returns [`SplineError::NonFiniteValue`] if any input coordinate is not
    /// finite.
    pub fn on_demand_design(&self, x: &[f64]) -> Result<OnDemandSplineDesign<Self>, SplineError> {
        OnDemandSplineDesign::new(x, self.clone())
    }

    /// Truncated-power knots.
    #[must_use]
    #[inline]
    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    /// Spline order; its degree is the $p$ used in the basis formula.
    #[must_use]
    #[inline]
    pub const fn order(&self) -> SplineOrder {
        self.order
    }

    /// Returns `true` if the first coefficient is an intercept.
    #[must_use]
    #[inline]
    pub const fn include_intercept(&self) -> bool {
        self.include_intercept
    }

    /// Number of basis functions.
    #[must_use]
    #[inline]
    pub const fn n_basis(&self) -> usize {
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
    #[allow(clippy::useless_let_if_seq)]
    pub fn for_each_basis(&self, x: f64, mut f: impl FnMut(usize, f64)) {
        let degree = self.order.degree();
        let mut offset = 0;
        if self.include_intercept {
            f(0, 1.0);
            offset = 1;
        }

        let mut polynomial = x;
        for power in 1..=degree {
            let weight = polynomial;
            if weight != 0.0 {
                f(offset + power - 1, weight);
            }
            polynomial *= x;
        }
        offset += degree;

        let active_knots = self.knots.partition_point(|knot| *knot <= x);
        for (knot_offset, knot) in self.knots[..active_knots].iter().copied().enumerate() {
            f(offset + knot_offset, pow_usize(x - knot, degree));
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
    #[allow(clippy::cast_precision_loss, clippy::useless_let_if_seq)]
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

        let active_knots = self.knots.partition_point(|knot| *knot <= x);
        for (knot_offset, knot) in self.knots[..active_knots].iter().copied().enumerate() {
            let weight = degree as f64 * pow_usize(x - knot, degree - 1);
            if weight != 0.0 {
                f(offset + knot_offset, weight);
            }
        }
    }
}

impl OnDemandSplineDesign<TruncatedPowerBasis> {
    /// Builds a design from data-derived uniform truncated-power knots.
    pub fn uniform_from_data(
        x: &[f64],
        n_knots: usize,
        order: SplineOrder,
        include_intercept: bool,
    ) -> Result<Self, SplineError> {
        TruncatedPowerBasis::uniform_from_data(x, n_knots, order, include_intercept)?.design(x)
    }

    /// Predictor derivative with respect to `x`.
    #[must_use]
    #[inline]
    #[allow(clippy::suboptimal_flops)]
    pub fn eta_derivative_row(&self, row: usize, beta: &[f64]) -> f64 {
        debug_assert!(row < self.nrows());
        debug_assert_eq!(beta.len(), self.n_basis());

        let mut value = 0.0;
        self.basis()
            .for_each_derivative_basis(self.x()[row], |index, weight| {
                value = beta[index].mul_add(weight, value);
            });
        value
    }
}

fn unique_interior_knots(knots: &OpenKnotVector) -> Vec<f64> {
    let mut interior = knots.interior().to_vec();
    interior.dedup_by(|left, right| left.total_cmp(right).is_eq());
    interior
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
