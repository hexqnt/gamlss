use std::ops::Range;

use gamlss_core::{LinearPredictorGeometry, ModelError, PredictorBlock, RowMultiplier};

use crate::geometry::{validate_gram_lengths, validate_transpose_lengths};
use crate::local::{PreparedLocalBasis, cyclic_local_basis_derivative, prepare_cyclic_local_basis};
use crate::row_basis::SplineRowBasis;
use crate::{OnDemandSplineDesign, SplineError, SplineOrder};

/// Metadata for a cyclic spline predictor on phase coordinates.
///
/// Let $K$ be `n_basis` and $p=\mathtt{order.degree()}$. Every input phase is reduced modulo one, $\phi^\star=\phi-\lfloor\phi\rfloor\in\lbrack0,1\rparen$, before evaluating the wrapped local basis $C_{j,p}$:
///
/// $$
/// \eta(\phi)=\sum_{j=0}^{K-1}\beta_jC_{j,p}(\phi^\star),
/// \qquad
/// \eta(\phi+\ell)=\eta(\phi),\quad \ell\in\mathbb{Z}.
/// $$
///
/// At most $p+1$ local weights are evaluated, and their coefficient indices wrap modulo $K$ at the phase boundary.
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CyclicSplineSpec {
    n_basis: usize,
    order: SplineOrder,
}

impl CyclicSplineSpec {
    /// Builds cyclic spline metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if `n_basis` is insufficient for `order`.
    pub const fn new(n_basis: usize, order: SplineOrder) -> Result<Self, SplineError> {
        if n_basis < order.min_basis() {
            return Err(SplineError::NotEnoughBasis {
                n_basis,
                degree: order.degree(),
            });
        }

        Ok(Self { n_basis, order })
    }

    /// Builds a predictor design for specific phases.
    ///
    /// # Errors
    ///
    /// Returns an error if `phi` contains non-finite values.
    pub fn design(&self, phi: &[f64]) -> Result<CyclicSplineDesign, SplineError> {
        let prepared = PreparedCyclicGeometry::try_from_phases(phi, *self)?;
        Ok(CyclicSplineDesign {
            phi: phi.into(),
            prepared,
            spec: *self,
        })
    }

    /// Builds a low-memory design that recomputes local row geometry on demand.
    ///
    /// Unlike [`Self::design`], this retains only the phases and spline
    /// metadata.
    ///
    /// # Errors
    ///
    /// Returns [`SplineError::NonFiniteValue`] if `phi` contains a non-finite
    /// phase.
    pub fn on_demand_design(&self, phi: &[f64]) -> Result<OnDemandSplineDesign<Self>, SplineError> {
        OnDemandSplineDesign::new(phi, *self)
    }

    /// Visits the local basis for one phase without allocating or retaining row geometry.
    ///
    /// This is the low-memory alternative to constructing a [`CyclicSplineDesign`] when phases are evaluated once or arrive as a stream.
    ///
    /// # Errors
    ///
    /// Returns [`SplineError::NonFiniteValue`] if `phi` is not finite.
    pub fn for_each_value_basis(
        &self,
        phi: f64,
        f: impl FnMut(usize, f64),
    ) -> Result<(), SplineError> {
        if !phi.is_finite() {
            return Err(SplineError::NonFiniteValue);
        }

        prepare_cyclic_local_basis(phi, self.order, self.n_basis).for_each_wrapped(
            self.order.degree() + 1,
            self.n_basis,
            f,
        );
        Ok(())
    }

    /// Number of spline coefficients.
    #[must_use]
    #[inline]
    pub const fn n_basis(&self) -> usize {
        self.n_basis
    }

    /// Spline order.
    #[must_use]
    #[inline]
    pub const fn order(&self) -> SplineOrder {
        self.order
    }

    #[inline]
    pub(crate) fn eta_derivative_at(&self, phi: f64, beta: &[f64]) -> f64 {
        cyclic_local_basis_derivative(phi, self.order, self.n_basis).dot(beta)
    }
}

/// Prepared cyclic row geometry shared by phase and physical-period designs.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedCyclicGeometry {
    rows: Box<[PreparedLocalBasis]>,
}

impl PreparedCyclicGeometry {
    #[inline]
    fn try_from_phases(phi: &[f64], spec: CyclicSplineSpec) -> Result<Self, SplineError> {
        let rows = phi
            .iter()
            .copied()
            .map(|value| {
                if !value.is_finite() {
                    return Err(SplineError::NonFiniteValue);
                }
                Ok(prepare_cyclic_local_basis(value, spec.order, spec.n_basis))
            })
            .collect::<Result<_, _>>()?;
        Ok(Self { rows })
    }

    #[inline]
    pub(crate) const fn from_rows(rows: Box<[PreparedLocalBasis]>) -> Self {
        Self { rows }
    }

    #[inline]
    pub(crate) fn nrows(&self) -> usize {
        self.rows.len()
    }

    #[inline]
    fn row(&self, row: usize) -> &PreparedLocalBasis {
        &self.rows[row]
    }

    #[inline]
    const fn width(spec: CyclicSplineSpec) -> usize {
        spec.order.degree() + 1
    }

    #[inline]
    pub(crate) fn for_each_row_basis(
        &self,
        spec: CyclicSplineSpec,
        row: usize,
        f: impl FnMut(usize, f64),
    ) {
        self.row(row)
            .for_each_wrapped(Self::width(spec), spec.n_basis, f);
    }

    #[inline]
    pub(crate) fn eta_row(&self, spec: CyclicSplineSpec, row: usize, beta: &[f64]) -> f64 {
        self.row(row)
            .dot_wrapped(Self::width(spec), spec.n_basis, beta)
    }

    #[inline]
    pub(crate) fn add_gradient_range(
        &self,
        spec: CyclicSplineSpec,
        rows: Range<usize>,
        scores: &[f64],
        grad: &mut [f64],
    ) {
        debug_assert!(rows.end <= self.rows.len());
        debug_assert_eq!(scores.len(), rows.len());
        debug_assert_eq!(grad.len(), spec.n_basis);

        for (offset, score) in scores.iter().copied().enumerate() {
            if score == 0.0 {
                continue;
            }
            self.row(rows.start + offset).add_scaled_wrapped(
                Self::width(spec),
                spec.n_basis,
                score,
                grad,
            );
        }
    }

    #[inline]
    pub(crate) fn add_weighted_gradient_by_range<M>(
        &self,
        spec: CyclicSplineSpec,
        rows: Range<usize>,
        scores: &[f64],
        multiplier: &M,
        grad: &mut [f64],
    ) where
        M: RowMultiplier + ?Sized,
    {
        debug_assert!(rows.end <= self.rows.len());
        debug_assert_eq!(scores.len(), rows.len());
        debug_assert_eq!(grad.len(), spec.n_basis);

        for (offset, score) in scores.iter().copied().enumerate() {
            if score == 0.0 {
                continue;
            }
            let row = rows.start + offset;
            let scaled_score = score * multiplier.multiplier_at(row);
            if scaled_score == 0.0 {
                continue;
            }
            self.row(row)
                .add_scaled_wrapped(Self::width(spec), spec.n_basis, scaled_score, grad);
        }
    }

    #[inline]
    pub(crate) fn add_weighted_gram(
        &self,
        spec: CyclicSplineSpec,
        row_weights: &[f64],
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        validate_gram_lengths(self.rows.len(), spec.n_basis, row_weights, out)?;

        for (row, weight) in row_weights.iter().copied().enumerate() {
            if weight == 0.0 {
                continue;
            }
            self.row(row)
                .add_scaled_outer_wrapped(Self::width(spec), spec.n_basis, weight, out);
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn add_weighted_gram_by<M>(
        &self,
        spec: CyclicSplineSpec,
        row_weights: &[f64],
        multiplier: &M,
        out: &mut [f64],
    ) -> Result<(), ModelError>
    where
        M: RowMultiplier + ?Sized,
    {
        validate_gram_lengths(self.rows.len(), spec.n_basis, row_weights, out)?;

        for (row, weight) in row_weights.iter().copied().enumerate() {
            if weight == 0.0 {
                continue;
            }
            let scaled_weight = weight * multiplier.multiplier_at(row);
            if scaled_weight == 0.0 {
                continue;
            }
            self.row(row).add_scaled_outer_wrapped(
                Self::width(spec),
                spec.n_basis,
                scaled_weight,
                out,
            );
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn add_t_mul_vec(
        &self,
        spec: CyclicSplineSpec,
        row_scores: &[f64],
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        validate_transpose_lengths(self.rows.len(), spec.n_basis, row_scores, out)?;
        self.add_gradient_range(spec, 0..self.rows.len(), row_scores, out);
        Ok(())
    }

    #[inline]
    pub(crate) fn add_t_mul_vec_by<M>(
        &self,
        spec: CyclicSplineSpec,
        row_scores: &[f64],
        multiplier: &M,
        out: &mut [f64],
    ) -> Result<(), ModelError>
    where
        M: RowMultiplier + ?Sized,
    {
        validate_transpose_lengths(self.rows.len(), spec.n_basis, row_scores, out)?;
        self.add_weighted_gradient_by_range(spec, 0..self.rows.len(), row_scores, multiplier, out);
        Ok(())
    }
}

/// Cyclic spline predictor for periodic covariates on $\lbrack0,1\rparen$.
///
/// Construction evaluates every local basis once. Each retained row stores its original `f64` phase plus compact geometry containing one `usize` start index and four `f64` weights; it never materializes an `nrows × n_basis` matrix. For repeated model passes without this row cache, use [`CyclicSplineSpec::on_demand_design`]. For one-shot evaluation, use [`CyclicSplineSpec::for_each_value_basis`].
///
/// Pass dimensionless phases here. For coordinates with a physical origin and period, use [`crate::PeriodicSplineDesign`], which performs the scaling and applies the chain rule to derivatives.
#[derive(Debug, Clone, PartialEq)]
pub struct CyclicSplineDesign {
    phi: Box<[f64]>,
    prepared: PreparedCyclicGeometry,
    spec: CyclicSplineSpec,
}

impl CyclicSplineDesign {
    /// Builds a cyclic spline design.
    ///
    /// # Errors
    ///
    /// Returns [`SplineError::NotEnoughBasis`] if `n_basis` is insufficient
    /// for `order`, or [`SplineError::NonFiniteValue`] if `phi` contains a
    /// non-finite phase.
    pub fn new(phi: &[f64], n_basis: usize, order: SplineOrder) -> Result<Self, SplineError> {
        CyclicSplineSpec::new(n_basis, order)?.design(phi)
    }

    /// Number of spline coefficients.
    #[must_use]
    #[inline]
    pub const fn n_basis(&self) -> usize {
        self.spec.n_basis()
    }

    /// Basis metadata suitable for building a design on new data.
    #[must_use]
    #[inline]
    pub const fn spec(&self) -> CyclicSplineSpec {
        self.spec
    }

    /// Returns the original phases of the design.
    #[must_use]
    #[inline]
    pub fn phi(&self) -> &[f64] {
        &self.phi
    }

    /// Derivative of the predictor contribution with respect to phase `phi`.
    #[must_use]
    #[inline]
    pub fn eta_derivative_row(&self, row: usize, beta: &[f64]) -> f64 {
        debug_assert!(row < self.phi.len());
        debug_assert_eq!(beta.len(), self.n_basis());

        self.spec.eta_derivative_at(self.phi[row], beta)
    }
}

impl SplineRowBasis for CyclicSplineDesign {
    #[inline]
    fn nrows(&self) -> usize {
        self.prepared.nrows()
    }

    #[inline]
    fn nparams(&self) -> usize {
        self.spec.n_basis
    }

    #[inline]
    fn for_each_row_basis(&self, row: usize, f: impl FnMut(usize, f64)) {
        self.prepared.for_each_row_basis(self.spec, row, f);
    }
}

impl PredictorBlock for CyclicSplineDesign {
    #[inline]
    fn nrows(&self) -> usize {
        self.prepared.nrows()
    }

    #[inline]
    fn nparams(&self) -> usize {
        self.spec.n_basis
    }

    #[inline]
    fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
        self.prepared.eta_row(self.spec, row, beta)
    }

    #[inline]
    fn zero_beta_constant_contribution(&self) -> Option<f64> {
        Some(0.0)
    }

    #[inline]
    fn add_gradient_range(&self, rows: Range<usize>, scores: &[f64], _: &[f64], grad: &mut [f64]) {
        self.prepared
            .add_gradient_range(self.spec, rows, scores, grad);
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
        self.prepared
            .add_weighted_gradient_by_range(self.spec, rows, scores, multiplier, grad);
    }
}

impl LinearPredictorGeometry for CyclicSplineDesign {
    #[inline]
    fn add_weighted_gram(&self, row_weights: &[f64], out: &mut [f64]) -> Result<(), ModelError> {
        self.prepared.add_weighted_gram(self.spec, row_weights, out)
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
            .add_weighted_gram_by(self.spec, row_weights, multiplier, out)
    }

    #[inline]
    fn add_t_mul_vec(&self, row_scores: &[f64], out: &mut [f64]) -> Result<(), ModelError> {
        self.prepared.add_t_mul_vec(self.spec, row_scores, out)
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
            .add_t_mul_vec_by(self.spec, row_scores, multiplier, out)
    }
}
impl OnDemandSplineDesign<CyclicSplineSpec> {
    /// Derivative of the predictor contribution with respect to phase `phi`.
    #[must_use]
    #[inline]
    pub fn eta_derivative_row(&self, row: usize, beta: &[f64]) -> f64 {
        debug_assert!(row < self.x().len());
        debug_assert_eq!(beta.len(), self.n_basis());

        self.basis().eta_derivative_at(self.x()[row], beta)
    }
}
