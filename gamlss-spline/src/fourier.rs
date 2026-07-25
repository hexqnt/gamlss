use std::ops::Range;

use gamlss_core::{LinearPredictorGeometry, ModelError, PredictorBlock, RowMultiplier};

use crate::{FourierError, OnDemandSplineDesign, SplineError, SplineRowBasis};

/// Reusable Fourier basis metadata for seasonal or periodic covariates.
///
/// Let $K$ be `order`, $P>0$ be `period`, and let $c$ equal one when
/// `include_intercept` is true and zero otherwise. The basis represents
///
/// $$
/// \eta(x)=c\beta_0+\sum_{k=1}^{K}\left\lbrack
/// \beta_{c+2k-2}\sin\left(\frac{2\pi kx}{P}\right)
/// \mathbin{+}\beta_{c+2k-1}\cos\left(\frac{2\pi kx}{P}\right)
/// \right\rbrack.
/// $$
///
/// Coefficients are ordered as
/// `[intercept?, sin(k=1), cos(k=1), ..., sin(k=K), cos(k=K)]`.
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FourierBasis {
    period: f64,
    omega: f64,
    order: usize,
    n_basis: usize,
    include_intercept: bool,
}

impl FourierBasis {
    /// Creates reusable Fourier basis metadata.
    ///
    /// # Errors
    ///
    /// Returns [`FourierError::InvalidPeriod`] unless `period` and its angular
    /// frequency are finite and positive, [`FourierError::InvalidOrder`] when
    /// `order` is zero, or [`FourierError::CoefficientOverflow`] when the basis
    /// size overflows `usize`.
    pub fn new(period: f64, order: usize, include_intercept: bool) -> Result<Self, FourierError> {
        if !period.is_finite() || period <= 0.0 {
            return Err(FourierError::InvalidPeriod);
        }
        if order == 0 {
            return Err(FourierError::InvalidOrder);
        }

        let n_basis = coefficient_count(order, include_intercept)?;
        let omega = std::f64::consts::TAU / period;
        if !omega.is_finite() {
            return Err(FourierError::InvalidPeriod);
        }

        Ok(Self {
            period,
            omega,
            order,
            n_basis,
            include_intercept,
        })
    }

    /// Builds the named Fourier predictor for concrete coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`FourierError::NonFiniteValue`] if a coordinate or its scaled
    /// phase is not finite.
    pub fn design(&self, x: &[f64]) -> Result<FourierDesign, FourierError> {
        Ok(FourierDesign {
            inner: self.on_demand_design(x)?,
        })
    }

    /// Builds the generic on-demand spline representation.
    ///
    /// # Errors
    ///
    /// Returns [`FourierError::NonFiniteValue`] if a coordinate or its scaled
    /// phase is not finite.
    pub fn on_demand_design(&self, x: &[f64]) -> Result<OnDemandSplineDesign<Self>, FourierError> {
        OnDemandSplineDesign::new(x, *self).map_err(|error| map_spline_error(&error))
    }

    /// Visits non-zero basis values for one coordinate without allocating.
    ///
    /// # Errors
    ///
    /// Returns [`FourierError::NonFiniteValue`] if the coordinate or its
    /// scaled phase is not finite.
    pub fn for_each_value_basis(
        &self,
        x: f64,
        f: impl FnMut(usize, f64),
    ) -> Result<(), FourierError> {
        let phase = self.phase(x)?;
        self.for_each_basis_at_phase(phase, f);
        Ok(())
    }

    /// Number of harmonics.
    #[inline]
    #[must_use]
    pub const fn order(&self) -> usize {
        self.order
    }

    /// Period of the Fourier basis.
    #[inline]
    #[must_use]
    pub const fn period(&self) -> f64 {
        self.period
    }

    /// Returns `true` if the basis contains an intercept.
    #[inline]
    #[must_use]
    pub const fn include_intercept(&self) -> bool {
        self.include_intercept
    }

    /// Number of basis functions.
    #[inline]
    #[must_use]
    pub const fn n_basis(&self) -> usize {
        self.n_basis
    }

    #[inline]
    pub(crate) fn phase(&self, x: f64) -> Result<f64, FourierError> {
        if !x.is_finite() {
            return Err(FourierError::NonFiniteValue);
        }
        let phase = self.omega * x;
        if !phase.is_finite() {
            return Err(FourierError::NonFiniteValue);
        }
        Ok(phase)
    }

    #[inline]
    #[allow(clippy::suboptimal_flops, clippy::useless_let_if_seq)]
    fn for_each_basis_at_phase(&self, phase: f64, mut f: impl FnMut(usize, f64)) {
        let (base_sin, base_cos) = phase.sin_cos();
        let mut harmonic_sin = base_sin;
        let mut harmonic_cos = base_cos;

        let mut offset = 0;
        if self.include_intercept {
            f(0, 1.0);
            offset = 1;
        }

        for harmonic in 1..=self.order {
            if harmonic_sin != 0.0 {
                f(offset, harmonic_sin);
            }
            if harmonic_cos != 0.0 {
                f(offset + 1, harmonic_cos);
            }
            offset += 2;

            if harmonic != self.order {
                let next_sin = harmonic_sin * base_cos + harmonic_cos * base_sin;
                let next_cos = harmonic_cos * base_cos - harmonic_sin * base_sin;
                harmonic_sin = next_sin;
                harmonic_cos = next_cos;
            }
        }
    }
}

/// Fourier predictor with allocation-free on-demand row evaluation.
///
/// The design retains coordinates and reusable [`FourierBasis`] metadata but
/// does not materialize a dense matrix. Use [`FourierDesign::basis`] to apply
/// the same fitted basis to new coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct FourierDesign {
    inner: OnDemandSplineDesign<FourierBasis>,
}

impl FourierDesign {
    /// Builds a Fourier predictor.
    ///
    /// `period` must be finite and positive, `order` must be positive, and all
    /// coordinates and scaled phases must be finite.
    pub fn new(
        x: &[f64],
        period: f64,
        order: usize,
        include_intercept: bool,
    ) -> Result<Self, FourierError> {
        FourierBasis::new(period, order, include_intercept)?.design(x)
    }

    /// Reusable basis metadata.
    #[inline]
    #[must_use]
    pub const fn basis(&self) -> &FourierBasis {
        self.inner.basis()
    }

    /// Number of harmonics.
    #[inline]
    #[must_use]
    pub const fn order(&self) -> usize {
        self.basis().order()
    }

    /// Period of the Fourier basis.
    #[inline]
    #[must_use]
    pub const fn period(&self) -> f64 {
        self.basis().period()
    }

    /// Returns `true` if the predictor contains an intercept.
    #[inline]
    #[must_use]
    pub const fn include_intercept(&self) -> bool {
        self.basis().include_intercept()
    }

    /// Number of Fourier coefficients.
    #[inline]
    #[must_use]
    pub const fn n_basis(&self) -> usize {
        self.basis().n_basis()
    }

    /// Returns the original coordinates.
    #[inline]
    #[must_use]
    pub fn x(&self) -> &[f64] {
        self.inner.x()
    }
}

impl SplineRowBasis for FourierDesign {
    #[inline]
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    #[inline]
    fn nparams(&self) -> usize {
        self.inner.n_basis()
    }

    #[inline]
    fn for_each_row_basis(&self, row: usize, f: impl FnMut(usize, f64)) {
        self.inner.for_each_row_basis(row, f);
    }
}

impl PredictorBlock for FourierDesign {
    #[inline]
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    #[inline]
    fn nparams(&self) -> usize {
        self.inner.n_basis()
    }

    #[inline]
    fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
        self.inner.eta_row(row, beta)
    }

    #[inline]
    fn zero_beta_constant_contribution(&self) -> Option<f64> {
        self.inner.zero_beta_constant_contribution()
    }

    #[inline]
    fn add_gradient_range(&self, rows: Range<usize>, scores: &[f64], _: &[f64], grad: &mut [f64]) {
        self.inner.add_gradient_range(rows, scores, &[], grad);
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
        self.inner
            .add_weighted_gradient_by_range(rows, scores, multiplier, &[], grad);
    }
}

impl LinearPredictorGeometry for FourierDesign {
    #[inline]
    fn add_weighted_gram(&self, row_weights: &[f64], out: &mut [f64]) -> Result<(), ModelError> {
        self.inner.add_weighted_gram(row_weights, out)
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
        self.inner
            .add_weighted_gram_by(row_weights, multiplier, out)
    }

    #[inline]
    fn add_t_mul_vec(&self, row_scores: &[f64], out: &mut [f64]) -> Result<(), ModelError> {
        self.inner.add_t_mul_vec(row_scores, out)
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
        self.inner.add_t_mul_vec_by(row_scores, multiplier, out)
    }
}

fn coefficient_count(order: usize, include_intercept: bool) -> Result<usize, FourierError> {
    order
        .checked_mul(2)
        .and_then(|count| count.checked_add(usize::from(include_intercept)))
        .ok_or(FourierError::CoefficientOverflow)
}

#[inline]
fn map_spline_error(error: &SplineError) -> FourierError {
    if *error != SplineError::NonFiniteValue {
        debug_assert!(false, "unexpected Fourier coordinate error");
    }
    FourierError::NonFiniteValue
}
