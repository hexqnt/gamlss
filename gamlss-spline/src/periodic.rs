use std::ops::Range;

use gamlss_core::{LinearPredictorGeometry, ModelError, PredictorBlock, RowMultiplier};

use crate::cyclic::{CyclicSplineSpec, PreparedCyclicGeometry};
use crate::local::prepare_cyclic_local_basis;
use crate::row_basis::SplineRowBasis;
use crate::{OnDemandSplineDesign, SplineError, SplineOrder};

/// Metadata for a cyclic spline over a physical period.
///
/// A coordinate $x$ is converted to the phase
///
/// $$
/// \phi(x)=\frac{x-x_0}{P},
/// $$
///
/// Here $x_0$ is [`PeriodicSplineSpec::origin`] and $P>0$ is [`PeriodicSplineSpec::period`]. The cyclic basis reduces this unwrapped phase modulo one; therefore $\eta(x+P)=\eta(x)$.
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeriodicSplineSpec {
    cyclic: CyclicSplineSpec,
    period: f64,
    origin: f64,
}

impl PeriodicSplineSpec {
    /// Creates a periodic spline spec.
    ///
    /// # Errors
    ///
    /// Returns [`SplineError::InvalidPeriod`] unless `period` is finite and
    /// positive and `origin` is finite. Returns
    /// [`SplineError::NotEnoughBasis`] if `n_basis` is insufficient for
    /// `order`.
    pub fn new(
        n_basis: usize,
        order: SplineOrder,
        period: f64,
        origin: f64,
    ) -> Result<Self, SplineError> {
        if !period.is_finite() || period <= 0.0 || !origin.is_finite() {
            return Err(SplineError::InvalidPeriod);
        }
        Ok(Self {
            cyclic: CyclicSplineSpec::new(n_basis, order)?,
            period,
            origin,
        })
    }

    /// Builds a design for physical coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`SplineError::NonFiniteValue`] if an input or its scaled phase
    /// is not finite.
    pub fn design(&self, x: &[f64]) -> Result<PeriodicSplineDesign, SplineError> {
        let prepared_rows = x
            .iter()
            .copied()
            .map(|value| {
                let phase = self.phase(value)?;
                Ok(prepare_cyclic_local_basis(
                    phase,
                    self.cyclic.order(),
                    self.cyclic.n_basis(),
                ))
            })
            .collect::<Result<Box<[_]>, SplineError>>()?;

        Ok(PeriodicSplineDesign {
            x: x.into(),
            prepared: PreparedCyclicGeometry::from_rows(prepared_rows),
            spec: *self,
        })
    }

    /// Builds a low-memory design that recomputes scaled cyclic rows on demand.
    ///
    /// Unlike [`Self::design`], this retains only the physical coordinates and
    /// spline metadata.
    ///
    /// # Errors
    ///
    /// Returns [`SplineError::NonFiniteValue`] if an input or its scaled phase
    /// is not finite.
    pub fn on_demand_design(&self, x: &[f64]) -> Result<OnDemandSplineDesign<Self>, SplineError> {
        OnDemandSplineDesign::new(x, *self)
    }

    /// Visits non-zero basis values for one physical coordinate.
    ///
    /// # Errors
    ///
    /// Returns [`SplineError::NonFiniteValue`] if `x` or its scaled phase is
    /// not finite.
    pub fn for_each_value_basis(
        &self,
        x: f64,
        f: impl FnMut(usize, f64),
    ) -> Result<(), SplineError> {
        self.cyclic.for_each_value_basis(self.phase(x)?, f)
    }

    /// Number of spline coefficients.
    #[must_use]
    #[inline]
    pub const fn n_basis(&self) -> usize {
        self.cyclic.n_basis()
    }

    /// Spline order.
    #[must_use]
    #[inline]
    pub const fn order(&self) -> SplineOrder {
        self.cyclic.order()
    }

    /// Period.
    #[must_use]
    #[inline]
    pub const fn period(&self) -> f64 {
        self.period
    }

    /// Origin.
    #[must_use]
    #[inline]
    pub const fn origin(&self) -> f64 {
        self.origin
    }

    #[inline]
    pub(crate) fn phase(&self, x: f64) -> Result<f64, SplineError> {
        if !x.is_finite() {
            return Err(SplineError::NonFiniteValue);
        }
        let phase = self.phase_unchecked(x);
        if !phase.is_finite() {
            return Err(SplineError::NonFiniteValue);
        }
        Ok(phase)
    }

    #[inline]
    fn phase_unchecked(&self, x: f64) -> f64 {
        (x - self.origin) / self.period
    }

    #[inline]
    pub(crate) fn add_scaled_outer_at(
        &self,
        x: f64,
        scale: f64,
        out: &mut [f64],
    ) -> Result<(), SplineError> {
        prepare_cyclic_local_basis(self.phase(x)?, self.cyclic.order(), self.cyclic.n_basis())
            .add_scaled_outer_wrapped(
                self.cyclic.order().degree() + 1,
                self.cyclic.n_basis(),
                scale,
                out,
            );
        Ok(())
    }
}

/// Periodic spline predictor over physical coordinates.
///
/// The design stores original coordinates and compact prepared local geometry,
/// but not a second phase-coordinate array. It evaluates the cyclic basis at
/// $(x-x_0)/P$. Derivatives obey
///
/// $$
/// \frac{d\eta}{dx}=\frac{1}{P}\frac{d\eta}{d\phi}.
/// $$
#[derive(Debug, Clone, PartialEq)]
pub struct PeriodicSplineDesign {
    x: Box<[f64]>,
    prepared: PreparedCyclicGeometry,
    spec: PeriodicSplineSpec,
}

impl PeriodicSplineDesign {
    /// Creates a periodic spline design.
    ///
    /// # Errors
    ///
    /// Returns an error if the spline metadata is invalid, an input coordinate
    /// is not finite, or scaling an input to a phase overflows.
    pub fn new(
        x: &[f64],
        n_basis: usize,
        order: SplineOrder,
        period: f64,
        origin: f64,
    ) -> Result<Self, SplineError> {
        PeriodicSplineSpec::new(n_basis, order, period, origin)?.design(x)
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
        self.spec.n_basis()
    }

    /// Metadata.
    #[must_use]
    #[inline]
    pub const fn spec(&self) -> PeriodicSplineSpec {
        self.spec
    }

    /// Predictor derivative with respect to the original coordinate.
    #[must_use]
    #[inline]
    pub fn eta_derivative_row(&self, row: usize, beta: &[f64]) -> f64 {
        debug_assert!(row < self.x.len());
        debug_assert_eq!(beta.len(), self.n_basis());

        let phase = self.spec.phase_unchecked(self.x[row]);
        self.spec.cyclic.eta_derivative_at(phase, beta) / self.spec.period
    }
}

impl SplineRowBasis for PeriodicSplineDesign {
    #[inline]
    fn nrows(&self) -> usize {
        self.prepared.nrows()
    }

    #[inline]
    fn nparams(&self) -> usize {
        self.spec.cyclic.n_basis()
    }

    #[inline]
    fn for_each_row_basis(&self, row: usize, f: impl FnMut(usize, f64)) {
        self.prepared.for_each_row_basis(self.spec.cyclic, row, f);
    }
}

impl PredictorBlock for PeriodicSplineDesign {
    #[inline]
    fn nrows(&self) -> usize {
        self.prepared.nrows()
    }

    #[inline]
    fn nparams(&self) -> usize {
        self.spec.cyclic.n_basis()
    }

    #[inline]
    fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
        self.prepared.eta_row(self.spec.cyclic, row, beta)
    }

    #[inline]
    fn zero_beta_constant_contribution(&self) -> Option<f64> {
        Some(0.0)
    }

    #[inline]
    fn add_gradient_range(&self, rows: Range<usize>, scores: &[f64], _: &[f64], grad: &mut [f64]) {
        self.prepared
            .add_gradient_range(self.spec.cyclic, rows, scores, grad);
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
        self.prepared.add_weighted_gradient_by_range(
            self.spec.cyclic,
            rows,
            scores,
            multiplier,
            grad,
        );
    }
}

impl LinearPredictorGeometry for PeriodicSplineDesign {
    #[inline]
    fn add_weighted_gram(&self, row_weights: &[f64], out: &mut [f64]) -> Result<(), ModelError> {
        self.prepared
            .add_weighted_gram(self.spec.cyclic, row_weights, out)
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
        self.prepared
            .add_weighted_gram_by(self.spec.cyclic, row_weights, multiplier, out)
    }

    #[inline]
    fn add_t_mul_vec(&self, row_scores: &[f64], out: &mut [f64]) -> Result<(), ModelError> {
        self.prepared
            .add_t_mul_vec(self.spec.cyclic, row_scores, out)
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
        self.prepared
            .add_t_mul_vec_by(self.spec.cyclic, row_scores, multiplier, out)
    }
}
impl OnDemandSplineDesign<PeriodicSplineSpec> {
    /// Derivative of the predictor contribution with respect to the original
    /// physical coordinate.
    #[must_use]
    #[inline]
    pub fn eta_derivative_row(&self, row: usize, beta: &[f64]) -> f64 {
        debug_assert!(row < self.x().len());
        debug_assert_eq!(beta.len(), self.n_basis());

        let spec = self.basis();
        let phase = spec.phase_unchecked(self.x()[row]);
        spec.cyclic.eta_derivative_at(phase, beta) / spec.period
    }
}
