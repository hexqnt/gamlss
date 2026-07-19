use std::ops::Range;

use gamlss_core::{PredictorBlock, RowMultiplier};

use crate::cyclic::{CyclicSplineDesign, CyclicSplineSpec};
use crate::row_basis::SplineRowBasis;
use crate::{SplineError, SplineOrder};

/// Metadata for a cyclic spline over a physical period.
///
/// A coordinate $x$ is converted to the phase
///
/// $$
/// \phi(x)=\frac{x-x_0}{P},
/// $$
///
/// Here $x_0$ is [`PeriodicSplineSpec::origin`] and $P>0$ is [`PeriodicSplineSpec::period`]. [`PeriodicSplineSpec::design`] passes this unwrapped phase to [`CyclicSplineDesign`], which applies reduction modulo one; therefore $\eta(x+P)=\eta(x)$.
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeriodicSplineSpec {
    cyclic: CyclicSplineSpec,
    period: f64,
    origin: f64,
}

impl PeriodicSplineSpec {
    /// Creates a periodic spline spec.
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
    pub fn design(&self, x: &[f64]) -> Result<PeriodicSplineDesign, SplineError> {
        if x.iter().any(|value| !value.is_finite()) {
            return Err(SplineError::NonFiniteValue);
        }
        let phi = x
            .iter()
            .map(|value| (value - self.origin) / self.period)
            .collect::<Vec<_>>();
        Ok(PeriodicSplineDesign {
            x: x.to_vec(),
            phase_design: self.cyclic.design(&phi)?,
            spec: *self,
        })
    }

    /// Number of spline coefficients.
    #[must_use]
    #[inline]
    pub const fn n_basis(&self) -> usize {
        self.cyclic.n_basis()
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
}

/// Periodic spline predictor over physical coordinates.
///
/// Unlike [`CyclicSplineDesign`], this design stores original coordinates and evaluates the cyclic basis at $(x-x_0)/P$. Derivatives obey
///
/// $$
/// \frac{d\eta}{dx}=\frac{1}{P}\frac{d\eta}{d\phi}.
/// $$
#[derive(Debug, Clone, PartialEq)]
pub struct PeriodicSplineDesign {
    x: Vec<f64>,
    phase_design: CyclicSplineDesign,
    spec: PeriodicSplineSpec,
}

impl PeriodicSplineDesign {
    /// Creates a periodic spline design.
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

        self.phase_design.eta_derivative_row(row, beta) / self.spec.period
    }
}

impl SplineRowBasis for PeriodicSplineDesign {
    #[inline]
    fn nrows(&self) -> usize {
        SplineRowBasis::nrows(&self.phase_design)
    }

    #[inline]
    fn nparams(&self) -> usize {
        SplineRowBasis::nparams(&self.phase_design)
    }

    #[inline]
    fn for_each_row_basis(&self, row: usize, f: impl FnMut(usize, f64)) {
        self.phase_design.for_each_row_basis(row, f);
    }
}
impl PredictorBlock for PeriodicSplineDesign {
    #[inline]
    fn nrows(&self) -> usize {
        PredictorBlock::nrows(&self.phase_design)
    }

    #[inline]
    fn nparams(&self) -> usize {
        PredictorBlock::nparams(&self.phase_design)
    }

    #[inline]
    fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
        self.phase_design.eta_row(row, beta)
    }

    #[inline]
    fn add_gradient_range(
        &self,
        rows: Range<usize>,
        scores: &[f64],
        beta: &[f64],
        grad: &mut [f64],
    ) {
        self.phase_design
            .add_gradient_range(rows, scores, beta, grad);
    }

    #[inline]
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
        self.phase_design
            .add_weighted_gradient_by_range(rows, scores, multiplier, beta, grad);
    }
}
