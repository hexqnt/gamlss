use gamlss_core::PredictorBlock;

use crate::local::{LocalBasis, cyclic_local_basis};
use crate::row_basis::SplineRowBasis;
use crate::{SplineError, SplineOrder};

/// Metadata для cyclic spline predictor-а.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CyclicSplineSpec {
    n_basis: usize,
    order: SplineOrder,
}

impl CyclicSplineSpec {
    /// Строит cyclic spline metadata.
    ///
    /// # Errors
    ///
    /// Возвращает ошибку, если `n_basis` недостаточен для `order`.
    pub fn new(n_basis: usize, order: SplineOrder) -> Result<Self, SplineError> {
        if n_basis < order.min_basis() {
            return Err(SplineError::NotEnoughBasis {
                n_basis,
                degree: order.degree(),
            });
        }

        Ok(Self { n_basis, order })
    }

    /// Строит predictor design для конкретных фаз.
    ///
    /// # Errors
    ///
    /// Возвращает ошибку, если `phi` содержит не-finite значения.
    pub fn design(&self, phi: &[f64]) -> Result<CyclicSplineDesign, SplineError> {
        if phi.iter().any(|value| !value.is_finite()) {
            return Err(SplineError::NonFiniteValue);
        }

        Ok(CyclicSplineDesign {
            phi: phi.to_vec(),
            spec: *self,
        })
    }

    /// Число spline-коэффициентов.
    #[must_use]
    pub fn n_basis(&self) -> usize {
        self.n_basis
    }

    /// Порядок spline.
    #[must_use]
    pub fn order(&self) -> SplineOrder {
        self.order
    }
}

/// Cyclic spline predictor для периодических ковариат на `[0, 1)`.
#[derive(Debug, Clone, PartialEq)]
pub struct CyclicSplineDesign {
    phi: Vec<f64>,
    spec: CyclicSplineSpec,
}

impl CyclicSplineDesign {
    /// Строит cyclic spline design.
    ///
    /// Все значения `phi` должны быть конечными.
    pub fn new(phi: &[f64], n_basis: usize, order: SplineOrder) -> Result<Self, SplineError> {
        CyclicSplineSpec::new(n_basis, order)?.design(phi)
    }

    /// Number of spline coefficients.
    #[must_use]
    pub fn n_basis(&self) -> usize {
        self.spec.n_basis()
    }

    /// Metadata basis-а, пригодная для построения design на новых данных.
    #[must_use]
    pub fn spec(&self) -> CyclicSplineSpec {
        self.spec
    }

    /// Возвращает исходные фазы design-а.
    #[must_use]
    pub fn phi(&self) -> &[f64] {
        &self.phi
    }

    fn basis_for_row(&self, row: usize) -> LocalBasis {
        cyclic_local_basis(self.phi[row], self.spec.order, self.spec.n_basis)
    }

    /// Производная predictor contribution по фазе `phi`.
    #[must_use]
    pub fn eta_derivative_row(&self, row: usize, beta: &[f64]) -> f64 {
        let h = 1.0e-6;
        let phi = self.phi[row];
        let plus = cyclic_local_basis(phi + h, self.spec.order, self.spec.n_basis).dot(beta);
        let minus = cyclic_local_basis(phi - h, self.spec.order, self.spec.n_basis).dot(beta);
        (plus - minus) / (2.0 * h)
    }
}

impl PredictorBlock for CyclicSplineDesign {
    fn nrows(&self) -> usize {
        self.phi.len()
    }

    fn nparams(&self) -> usize {
        self.spec.n_basis
    }

    fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
        self.basis_for_row(row).dot(beta)
    }

    fn add_gradient(&self, scores: &[f64], _: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(scores.len(), self.phi.len());
        debug_assert_eq!(grad.len(), self.spec.n_basis);

        for (row, score) in scores.iter().copied().enumerate() {
            self.basis_for_row(row).add_scaled(score, grad);
        }
    }

    fn add_weighted_gradient(
        &self,
        scores: &[f64],
        multiplier: &[f64],
        _: &[f64],
        grad: &mut [f64],
    ) {
        debug_assert_eq!(scores.len(), self.phi.len());
        debug_assert_eq!(multiplier.len(), self.phi.len());
        debug_assert_eq!(grad.len(), self.spec.n_basis);

        for (row, (&score, &multiplier)) in scores.iter().zip(multiplier).enumerate() {
            self.basis_for_row(row).add_scaled(score * multiplier, grad);
        }
    }
}

impl SplineRowBasis for CyclicSplineDesign {
    fn nrows(&self) -> usize {
        self.phi.len()
    }

    fn nparams(&self) -> usize {
        self.spec.n_basis
    }

    fn for_each_row_basis(&self, row: usize, f: impl FnMut(usize, f64)) {
        self.basis_for_row(row).for_each(f);
    }
}
