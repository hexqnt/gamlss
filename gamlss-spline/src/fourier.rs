use gamlss_core::PredictorBlock;

use crate::{FourierError, SplineRowBasis};

/// Fourier predictor для сезонных/периодических ковариат.
///
/// Для `order = K` строит колонки
/// `sin(2πkx / period)` и `cos(2πkx / period)`, `k = 1..=K`.
/// Если `include_intercept = true`, первым коэффициентом является intercept.
/// Локальный вектор коэффициентов имеет порядок:
/// `[intercept?, sin(k=1), cos(k=1), ..., sin(k=K), cos(k=K)]`.
///
/// В отличие от dense design matrix, этот predictor не материализует базис:
/// значения вычисляются напрямую в [`PredictorBlock::eta_row`] и
/// [`PredictorBlock::add_gradient`].
#[derive(Debug, Clone, PartialEq)]
pub struct FourierDesign {
    x: Vec<f64>,
    omega: f64,
    order: usize,
    nparams: usize,
    include_intercept: bool,
}

impl FourierDesign {
    /// Строит Fourier predictor.
    ///
    /// `period` должен быть конечным и положительным, `order` — положительным.
    /// Все значения `x` должны быть конечными.
    pub fn new(
        x: &[f64],
        period: f64,
        order: usize,
        include_intercept: bool,
    ) -> Result<Self, FourierError> {
        if x.iter().any(|value| !value.is_finite()) {
            return Err(FourierError::NonFiniteValue);
        }
        if !period.is_finite() || period <= 0.0 {
            return Err(FourierError::InvalidPeriod);
        }
        if order == 0 {
            return Err(FourierError::InvalidOrder);
        }

        let nparams = coefficient_count(order, include_intercept)?;

        Ok(Self {
            x: x.to_vec(),
            omega: std::f64::consts::TAU / period,
            order,
            nparams,
            include_intercept,
        })
    }

    /// Число гармоник.
    #[inline(always)]
    pub fn order(&self) -> usize {
        self.order
    }

    /// Период Fourier basis.
    #[inline(always)]
    pub fn period(&self) -> f64 {
        std::f64::consts::TAU / self.omega
    }

    /// Возвращает `true`, если predictor содержит intercept.
    #[inline(always)]
    pub fn include_intercept(&self) -> bool {
        self.include_intercept
    }

    /// Возвращает исходные координаты.
    #[inline(always)]
    pub fn x(&self) -> &[f64] {
        &self.x
    }

    #[inline]
    fn for_each_basis_at(&self, row: usize, mut f: impl FnMut(usize, f64)) {
        let (base_sin, base_cos) = (self.omega * self.x[row]).sin_cos();
        let mut harmonic_sin = base_sin;
        let mut harmonic_cos = base_cos;

        let mut offset = 0;
        if self.include_intercept {
            f(0, 1.0);
            offset = 1;
        }

        for harmonic in 1..=self.order {
            f(offset, harmonic_sin);
            f(offset + 1, harmonic_cos);
            offset += 2;

            if harmonic != self.order {
                let next_sin = harmonic_sin * base_cos + harmonic_cos * base_sin;
                let next_cos = harmonic_cos * base_cos - harmonic_sin * base_sin;
                harmonic_sin = next_sin;
                harmonic_cos = next_cos;
            }
        }
    }

    #[inline]
    fn add_row_gradient(&self, row: usize, score: f64, grad: &mut [f64]) {
        self.for_each_basis_at(row, |index, basis| {
            grad[index] += score * basis;
        });
    }
}

impl PredictorBlock for FourierDesign {
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.x.len()
    }

    #[inline(always)]
    fn nparams(&self) -> usize {
        self.nparams
    }

    #[inline]
    fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
        debug_assert!(row < self.x.len());
        debug_assert_eq!(beta.len(), self.nparams);

        let mut value = 0.0;
        self.for_each_basis_at(row, |index, basis| {
            value += beta[index] * basis;
        });
        value
    }

    #[inline]
    fn add_gradient(&self, scores: &[f64], _: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(scores.len(), self.x.len());
        debug_assert_eq!(grad.len(), self.nparams);

        for (row, score) in scores.iter().copied().enumerate() {
            self.add_row_gradient(row, score, grad);
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
        debug_assert_eq!(grad.len(), self.nparams);

        for (row, (&score, &multiplier)) in scores.iter().zip(multiplier).enumerate() {
            self.add_row_gradient(row, score * multiplier, grad);
        }
    }
}

impl SplineRowBasis for FourierDesign {
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.x.len()
    }

    #[inline(always)]
    fn nparams(&self) -> usize {
        self.nparams
    }

    #[inline]
    fn for_each_row_basis(&self, row: usize, f: impl FnMut(usize, f64)) {
        debug_assert!(row < self.x.len());
        self.for_each_basis_at(row, f);
    }
}

fn coefficient_count(order: usize, include_intercept: bool) -> Result<usize, FourierError> {
    order
        .checked_mul(2)
        .and_then(|count| count.checked_add(usize::from(include_intercept)))
        .ok_or(FourierError::CoefficientOverflow)
}
