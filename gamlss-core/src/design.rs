use crate::ModelError;

/// Минимальный контракт design matrix для hot path модели.
///
/// Реализации должны интерпретировать `beta` как вектор длины `ncols()` и
/// `weights` как вектор длины `nrows()`. Методы не обязаны повторно проверять
/// длины в release-сборке, поэтому вызывающий код валидирует размеры заранее.
pub trait DesignMatrix {
    /// Число наблюдений.
    fn nrows(&self) -> usize;
    /// Число коэффициентов в блоке.
    fn ncols(&self) -> usize;
    /// Скалярное произведение строки `row` на `beta`.
    fn dot_row(&self, row: usize, beta: &[f64]) -> f64;
    /// Добавляет `X^T weights` в `out`.
    fn add_t_mul_vec(&self, weights: &[f64], out: &mut [f64]);
    /// Добавляет `X^T (weights * multiplier)` в `out`.
    fn add_weighted_t_mul_vec(&self, weights: &[f64], multiplier: &[f64], out: &mut [f64]) {
        debug_assert_eq!(weights.len(), multiplier.len());

        let scaled_weights = weights
            .iter()
            .zip(multiplier)
            .map(|(weight, multiplier)| weight * multiplier)
            .collect::<Vec<_>>();
        self.add_t_mul_vec(&scaled_weights, out);
    }
}

/// Простая dense matrix в row-major порядке.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseDesign {
    nrows: usize,
    ncols: usize,
    values: Vec<f64>,
}

impl DenseDesign {
    /// Создаёт dense matrix из row-major значений.
    ///
    /// Возвращает ошибку, если `values.len() != nrows * ncols`.
    pub fn from_row_major(
        nrows: usize,
        ncols: usize,
        values: Vec<f64>,
    ) -> Result<Self, ModelError> {
        let expected_values = checked_len(nrows, ncols, "dense design row-major value count")?;
        let actual_values = values.len();
        if actual_values != expected_values {
            return Err(ModelError::DesignSize {
                expected_values,
                actual_values,
            });
        }

        Ok(Self {
            nrows,
            ncols,
            values,
        })
    }

    /// Создаёт dense matrix из массива строк фиксированной ширины.
    pub fn from_rows<const C: usize>(rows: &[[f64; C]]) -> Self {
        let values = rows.iter().flat_map(|row| row.iter().copied()).collect();
        Self {
            nrows: rows.len(),
            ncols: C,
            values,
        }
    }

    /// Создаёт design matrix из одного intercept-столбца.
    pub fn intercept(nrows: usize) -> Self {
        Self {
            nrows,
            ncols: 1,
            values: vec![1.0; nrows],
        }
    }

    /// Создаёт design matrix из одного пользовательского столбца.
    pub fn column(values: &[f64]) -> Self {
        Self {
            nrows: values.len(),
            ncols: 1,
            values: values.to_vec(),
        }
    }

    /// Создаёт matrix из набора столбцов, опционально добавляя intercept первым.
    ///
    /// Все переданные столбцы должны иметь длину `nrows`.
    pub fn from_columns(
        nrows: usize,
        include_intercept: bool,
        columns: &[&[f64]],
    ) -> Result<Self, ModelError> {
        for column in columns {
            if column.len() != nrows {
                return Err(ModelError::DesignRowMismatch {
                    parameter: "column",
                    expected_rows: nrows,
                    actual_rows: column.len(),
                });
            }
        }

        let ncols = columns
            .len()
            .checked_add(usize::from(include_intercept))
            .ok_or(ModelError::ArithmeticOverflow {
                context: "dense design column count",
            })?;
        let mut values = Vec::with_capacity(checked_len(
            nrows,
            ncols,
            "dense design row-major value count",
        )?);

        for row in 0..nrows {
            if include_intercept {
                values.push(1.0);
            }
            for column in columns {
                values.push(column[row]);
            }
        }

        Self::from_row_major(nrows, ncols, values)
    }

    /// Возвращает row-major значения матрицы.
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

fn checked_len(nrows: usize, ncols: usize, context: &'static str) -> Result<usize, ModelError> {
    nrows
        .checked_mul(ncols)
        .ok_or(ModelError::ArithmeticOverflow { context })
}

impl DesignMatrix for DenseDesign {
    fn nrows(&self) -> usize {
        self.nrows
    }

    fn ncols(&self) -> usize {
        self.ncols
    }

    fn dot_row(&self, row: usize, beta: &[f64]) -> f64 {
        debug_assert!(row < self.nrows);
        debug_assert_eq!(beta.len(), self.ncols);

        let offset = row * self.ncols;
        self.values[offset..offset + self.ncols]
            .iter()
            .zip(beta)
            .map(|(x, b)| x * b)
            .sum()
    }

    fn add_t_mul_vec(&self, weights: &[f64], out: &mut [f64]) {
        debug_assert_eq!(weights.len(), self.nrows);
        debug_assert_eq!(out.len(), self.ncols);

        for (row, weight) in weights.iter().copied().enumerate() {
            let offset = row * self.ncols;
            for (col, out_value) in out.iter_mut().enumerate() {
                *out_value += self.values[offset + col] * weight;
            }
        }
    }

    fn add_weighted_t_mul_vec(&self, weights: &[f64], multiplier: &[f64], out: &mut [f64]) {
        debug_assert_eq!(weights.len(), self.nrows);
        debug_assert_eq!(multiplier.len(), self.nrows);
        debug_assert_eq!(out.len(), self.ncols);

        for (row, (weight, multiplier)) in weights.iter().zip(multiplier).enumerate() {
            let scaled_weight = weight * multiplier;
            let offset = row * self.ncols;
            for (col, out_value) in out.iter_mut().enumerate() {
                *out_value += self.values[offset + col] * scaled_weight;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DenseDesign, DesignMatrix};
    use approx::assert_relative_eq;

    use crate::ModelError;

    #[test]
    fn dense_design_multiplies_rows_and_transpose() {
        let design = DenseDesign::from_rows(&[[1.0, 2.0], [3.0, 4.0]]);

        assert_relative_eq!(design.dot_row(1, &[10.0, 1.0]), 34.0);

        let mut out = vec![0.0, 0.0];
        design.add_t_mul_vec(&[0.5, 2.0], &mut out);

        assert_relative_eq!(out[0], 6.5);
        assert_relative_eq!(out[1], 9.0);
    }

    #[test]
    fn dense_design_rejects_overflowing_dimensions() {
        assert_eq!(
            DenseDesign::from_row_major(usize::MAX, 2, Vec::new()).unwrap_err(),
            ModelError::ArithmeticOverflow {
                context: "dense design row-major value count"
            }
        );
    }
}
