use gamlss_core::{PredictorBlock, RowMultiplier};

use crate::local::{LocalBasis, cyclic_local_basis, cyclic_local_basis_derivative};
use crate::row_basis::SplineRowBasis;
use crate::{SplineError, SplineOrder};

/// Metadata for a cyclic spline predictor.
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
        if phi.iter().any(|value| !value.is_finite()) {
            return Err(SplineError::NonFiniteValue);
        }

        Ok(CyclicSplineDesign {
            phi: phi.to_vec(),
            spec: *self,
        })
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
}

/// Cyclic spline predictor for periodic covariates on `[0, 1)`.
#[derive(Debug, Clone, PartialEq)]
pub struct CyclicSplineDesign {
    phi: Vec<f64>,
    spec: CyclicSplineSpec,
}

impl CyclicSplineDesign {
    /// Builds a cyclic spline design.
    ///
    /// All `phi` values must be finite.
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

    #[inline]
    fn basis_for_row(&self, row: usize) -> LocalBasis {
        cyclic_local_basis(self.phi[row], self.spec.order, self.spec.n_basis)
    }

    /// Derivative of the predictor contribution with respect to phase `phi`.
    #[must_use]
    #[inline]
    pub fn eta_derivative_row(&self, row: usize, beta: &[f64]) -> f64 {
        cyclic_local_basis_derivative(self.phi[row], self.spec.order, self.spec.n_basis).dot(beta)
    }
}

impl SplineRowBasis for CyclicSplineDesign {
    #[inline]
    fn nrows(&self) -> usize {
        self.phi.len()
    }

    #[inline]
    fn nparams(&self) -> usize {
        self.spec.n_basis
    }

    #[inline]
    fn for_each_row_basis(&self, row: usize, f: impl FnMut(usize, f64)) {
        self.basis_for_row(row).for_each(f);
    }
}
impl PredictorBlock for CyclicSplineDesign {
    #[inline]
    fn nrows(&self) -> usize {
        self.phi.len()
    }

    #[inline]
    fn nparams(&self) -> usize {
        self.spec.n_basis
    }

    #[inline]
    fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
        self.basis_for_row(row).dot(beta)
    }

    #[inline]
    fn add_gradient(&self, scores: &[f64], _: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(scores.len(), self.phi.len());
        debug_assert_eq!(grad.len(), self.spec.n_basis);

        for (row, score) in scores.iter().copied().enumerate() {
            if score == 0.0 {
                continue;
            }
            self.basis_for_row(row).add_scaled(score, grad);
        }
    }

    #[inline]
    fn add_weighted_gradient(
        &self,
        scores: &[f64],
        multiplier: &[f64],
        beta: &[f64],
        grad: &mut [f64],
    ) {
        debug_assert_eq!(multiplier.len(), self.phi.len());
        self.add_weighted_gradient_by(scores, multiplier, beta, grad);
    }

    #[inline]
    fn add_weighted_gradient_by<M>(
        &self,
        scores: &[f64],
        multiplier: &M,
        _: &[f64],
        grad: &mut [f64],
    ) where
        M: RowMultiplier + ?Sized,
    {
        debug_assert_eq!(scores.len(), self.phi.len());
        debug_assert_eq!(grad.len(), self.spec.n_basis);

        for (row, score) in scores.iter().copied().enumerate() {
            if score == 0.0 {
                continue;
            }
            let scaled_score = score * multiplier.multiplier_at(row);
            if scaled_score == 0.0 {
                continue;
            }
            self.basis_for_row(row).add_scaled(scaled_score, grad);
        }
    }
}
