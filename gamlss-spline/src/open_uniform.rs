use gamlss_core::PredictorBlock;

use crate::local::{LocalBasis, open_uniform_local_basis};
use crate::row_basis::SplineRowBasis;
use crate::{SplineError, SplineOrder};

/// Metadata для open-uniform spline predictor-а.
///
/// Хранит только shape basis-а и диапазон шкалирования, поэтому может
/// переиспользоваться для построения design на обучающих и новых данных.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpenUniformSplineBasis {
    min: f64,
    max: f64,
    n_basis: usize,
    order: SplineOrder,
    n_intervals: f64,
}

impl OpenUniformSplineBasis {
    /// Строит metadata по диапазону обучающих данных.
    ///
    /// # Errors
    ///
    /// Возвращает ошибку, если `x` пустой, содержит не-finite значения,
    /// имеет вырожденный диапазон или `n_basis` недостаточен для `order`.
    pub fn from_data(x: &[f64], n_basis: usize, order: SplineOrder) -> Result<Self, SplineError> {
        if x.is_empty() {
            return Err(SplineError::EmptyInput);
        }
        if n_basis < order.min_basis() {
            return Err(SplineError::NotEnoughBasis {
                n_basis,
                degree: order.degree(),
            });
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
        Self::new(min, max, n_basis, order)
    }

    /// Строит metadata с явным конечным диапазоном.
    ///
    /// # Errors
    ///
    /// Возвращает ошибку, если границы не конечны, `min >= max`, или
    /// `n_basis` недостаточен для `order`.
    pub fn new(
        min: f64,
        max: f64,
        n_basis: usize,
        order: SplineOrder,
    ) -> Result<Self, SplineError> {
        if !min.is_finite() || !max.is_finite() || min >= max {
            return Err(SplineError::InvalidRange);
        }
        if n_basis < order.min_basis() {
            return Err(SplineError::NotEnoughBasis {
                n_basis,
                degree: order.degree(),
            });
        }

        Ok(Self {
            min,
            max,
            n_basis,
            order,
            n_intervals: (n_basis - order.degree()).max(1) as f64,
        })
    }

    /// Строит predictor design для конкретного набора координат.
    ///
    /// # Errors
    ///
    /// Возвращает ошибку, если `x` содержит не-finite значения.
    pub fn design(&self, x: &[f64]) -> Result<OpenUniformSplineDesign, SplineError> {
        if x.iter().any(|value| !value.is_finite()) {
            return Err(SplineError::NonFiniteValue);
        }

        Ok(OpenUniformSplineDesign {
            x: x.to_vec(),
            basis: *self,
        })
    }

    /// Visits non-zero basis values for one input coordinate without allocating.
    ///
    /// The callback receives `(basis_index, weight)` pairs in the same
    /// coefficient order as [`OpenUniformSplineDesign`].
    ///
    /// # Errors
    ///
    /// Returns [`SplineError::NonFiniteValue`] if `x` is not finite.
    pub fn for_each_value_basis(
        &self,
        x: f64,
        f: impl FnMut(usize, f64),
    ) -> Result<(), SplineError> {
        if !x.is_finite() {
            return Err(SplineError::NonFiniteValue);
        }

        self.local_basis(x).for_each(f);
        Ok(())
    }

    /// Нижняя граница диапазона basis-а.
    #[must_use]
    pub fn min(&self) -> f64 {
        self.min
    }

    /// Верхняя граница диапазона basis-а.
    #[must_use]
    pub fn max(&self) -> f64 {
        self.max
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

    fn span(&self) -> f64 {
        self.max - self.min
    }

    fn local_basis(&self, x: f64) -> LocalBasis {
        let u = (x - self.min) / self.span();
        self.local_basis_for_unit(u)
    }

    fn local_basis_for_unit(&self, u: f64) -> LocalBasis {
        open_uniform_local_basis(u, self.order, self.n_basis, self.n_intervals)
    }
}

/// Open-uniform spline predictor с локальным sparse вычислением строк.
///
/// В отличие от [`crate::BSplineBasis`], хранит исходные данные и вычисляет
/// базисные функции «на лету» через компактное `LocalBasis`, не
/// материализуя полную design matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenUniformSplineDesign {
    x: Vec<f64>,
    basis: OpenUniformSplineBasis,
}

impl OpenUniformSplineDesign {
    /// Строит open-uniform spline design по диапазону данных.
    ///
    /// Если данные пусты или содержат не-finite значения, возвращает ошибку.
    pub fn from_data(x: &[f64], n_basis: usize, order: SplineOrder) -> Result<Self, SplineError> {
        OpenUniformSplineBasis::from_data(x, n_basis, order)?.design(x)
    }

    /// Строит open-uniform spline design с явным конечным диапазоном.
    ///
    /// `min` и `max` должны быть конечными и `min < max`.
    pub fn with_range(
        x: &[f64],
        min: f64,
        max: f64,
        n_basis: usize,
        order: SplineOrder,
    ) -> Result<Self, SplineError> {
        OpenUniformSplineBasis::new(min, max, n_basis, order)?.design(x)
    }

    /// Число spline-коэффициентов.
    #[must_use]
    pub fn n_basis(&self) -> usize {
        self.basis.n_basis()
    }

    /// Metadata basis-а, пригодная для построения design на новых данных.
    #[must_use]
    pub fn basis(&self) -> OpenUniformSplineBasis {
        self.basis
    }

    /// Возвращает исходные координаты design-а.
    #[must_use]
    pub fn x(&self) -> &[f64] {
        &self.x
    }

    fn basis_for_row(&self, row: usize) -> LocalBasis {
        self.basis.local_basis(self.x[row])
    }

    fn basis_for_unit(&self, u: f64) -> LocalBasis {
        self.basis.local_basis_for_unit(u)
    }

    /// Производная predictor contribution по исходной координате `x`.
    #[must_use]
    pub fn eta_derivative_row(&self, row: usize, beta: &[f64]) -> f64 {
        let span = self.basis.span();
        let h = 1.0e-6_f64.max(span.abs() * 1.0e-6);
        let u = (self.x[row] - self.basis.min) / span;
        let du = h / span;
        let plus = self.basis_for_unit(u + du).dot(beta);
        let minus = self.basis_for_unit(u - du).dot(beta);
        (plus - minus) / (2.0 * h)
    }
}

impl PredictorBlock for OpenUniformSplineDesign {
    fn nrows(&self) -> usize {
        self.x.len()
    }

    fn nparams(&self) -> usize {
        self.basis.n_basis
    }

    fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
        let basis = self.basis_for_row(row);
        basis.dot(beta)
    }

    fn add_gradient(&self, scores: &[f64], _: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(scores.len(), self.x.len());
        debug_assert_eq!(grad.len(), self.basis.n_basis);

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
        debug_assert_eq!(scores.len(), self.x.len());
        debug_assert_eq!(multiplier.len(), self.x.len());
        debug_assert_eq!(grad.len(), self.basis.n_basis);

        for (row, (&score, &multiplier)) in scores.iter().zip(multiplier).enumerate() {
            self.basis_for_row(row).add_scaled(score * multiplier, grad);
        }
    }
}

impl SplineRowBasis for OpenUniformSplineDesign {
    fn nrows(&self) -> usize {
        self.x.len()
    }

    fn nparams(&self) -> usize {
        self.basis.n_basis
    }

    fn for_each_row_basis(&self, row: usize, f: impl FnMut(usize, f64)) {
        self.basis_for_row(row).for_each(f);
    }
}
