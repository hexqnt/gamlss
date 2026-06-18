use gamlss_core::DenseDesign;

use crate::SplineError;

/// B-spline basis с заданной степенью и knot vector.
#[derive(Debug, Clone, PartialEq)]
pub struct BSplineBasis {
    degree: usize,
    knots: Vec<f64>,
}

impl BSplineBasis {
    /// Создаёт basis из готового knot vector.
    ///
    /// Knot vector должен быть конечным, неубывающим и достаточно длинным для
    /// выбранной степени.
    pub fn new(degree: usize, knots: Vec<f64>) -> Result<Self, SplineError> {
        if knots.len() <= degree + 1 {
            return Err(SplineError::NotEnoughBasis { n_basis: 0, degree });
        }

        if knots
            .windows(2)
            .any(|window| !window[0].is_finite() || !window[1].is_finite() || window[0] > window[1])
        {
            return Err(SplineError::InvalidKnots);
        }

        Ok(Self { degree, knots })
    }

    /// Строит open uniform B-spline basis по диапазону данных.
    pub fn open_uniform_from_data(
        x: &[f64],
        n_basis: usize,
        degree: usize,
    ) -> Result<Self, SplineError> {
        if x.is_empty() {
            return Err(SplineError::EmptyInput);
        }
        if n_basis <= degree {
            return Err(SplineError::NotEnoughBasis { n_basis, degree });
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

        let interior = n_basis.saturating_sub(degree + 1);
        let mut knots = Vec::with_capacity(n_basis + degree + 1);
        knots.extend(std::iter::repeat_n(min, degree + 1));

        for index in 1..=interior {
            let fraction = index as f64 / (interior + 1) as f64;
            knots.push(min + fraction * (max - min));
        }

        knots.extend(std::iter::repeat_n(max, degree + 1));
        Self::new(degree, knots)
    }

    /// Степень spline.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Knot vector.
    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    /// Число basis-функций.
    pub fn n_basis(&self) -> usize {
        self.knots.len() - self.degree - 1
    }

    /// Значения всех basis-функций в точке `x`.
    pub fn evaluate(&self, x: f64) -> Vec<f64> {
        let mut values = vec![0.0; self.n_basis()];
        self.evaluate_into(x, &mut values);
        values
    }

    /// Writes all basis-function values at `x` into `out`.
    ///
    /// `out.len()` must equal [`Self::n_basis`].
    pub fn evaluate_into(&self, x: f64, out: &mut [f64]) {
        self.fill_values(x, out);
    }

    /// Dense design matrix, где каждая строка содержит `evaluate(x_i)`.
    pub fn design_matrix(&self, x: &[f64]) -> Result<DenseDesign, SplineError> {
        if x.iter().any(|value| !value.is_finite()) {
            return Err(SplineError::NonFiniteValue);
        }

        let n_basis = self.n_basis();
        let mut values = Vec::with_capacity(x.len() * n_basis);
        values.resize(x.len() * n_basis, 0.0);
        for (row, value) in values.chunks_exact_mut(n_basis).zip(x.iter().copied()) {
            self.fill_values(value, row);
        }

        Ok(DenseDesign::from_row_major(x.len(), n_basis, values)?)
    }

    fn fill_values(&self, x: f64, out: &mut [f64]) {
        debug_assert_eq!(out.len(), self.n_basis());

        for (index, value) in out.iter_mut().enumerate() {
            *value = self.basis_value(index, self.degree, x);
        }
    }

    fn basis_value(&self, index: usize, degree: usize, x: f64) -> f64 {
        if degree == 0 {
            let left = self.knots[index];
            let right = self.knots[index + 1];
            let is_last_basis = index + 1 == self.n_basis();
            if (left <= x && x < right) || (is_last_basis && x == right) {
                1.0
            } else {
                0.0
            }
        } else {
            let mut value = 0.0;
            let left_denom = self.knots[index + degree] - self.knots[index];
            if left_denom > 0.0 {
                value +=
                    (x - self.knots[index]) / left_denom * self.basis_value(index, degree - 1, x);
            }

            let right_denom = self.knots[index + degree + 1] - self.knots[index + 1];
            if right_denom > 0.0 {
                value += (self.knots[index + degree + 1] - x) / right_denom
                    * self.basis_value(index + 1, degree - 1, x);
            }

            value
        }
    }
}

/// Вспомогательная функция для open uniform P-spline design matrix.
pub fn pspline_design(
    x: &[f64],
    n_basis: usize,
    degree: usize,
) -> Result<DenseDesign, SplineError> {
    BSplineBasis::open_uniform_from_data(x, n_basis, degree)?.design_matrix(x)
}
