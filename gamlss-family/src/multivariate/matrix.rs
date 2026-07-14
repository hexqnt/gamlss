//! Matrix storage primitives shared by multivariate families.

use gamlss_core::ModelError;

use super::cholesky::{packed_index, packed_len};

/// Fixed-dimensional lower-triangular matrix storage.
///
/// Only entries with `col <= row` are meaningful. Upper-triangular entries are
/// normalized to zero at construction and are never returned by [`Self::get`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedLowerTriangular<const D: usize> {
    values: [[f64; D]; D],
}

impl<const D: usize> FixedLowerTriangular<D> {
    /// Creates a lower-triangular matrix with all modeled entries set to zero.
    #[must_use]
    #[inline]
    pub const fn zeros() -> Self {
        Self {
            values: [[0.0; D]; D],
        }
    }

    /// Creates a lower-triangular matrix from full row storage, ignoring the upper triangle.
    #[must_use]
    pub fn from_lower_rows(mut values: [[f64; D]; D]) -> Self {
        for (row, row_values) in values.iter_mut().enumerate() {
            for value in row_values.iter_mut().skip(row + 1) {
                *value = 0.0;
            }
        }
        Self { values }
    }

    /// Creates a lower-triangular matrix from packed row-major lower entries.
    ///
    /// Packed order is `(0,0), (1,0), (1,1), (2,0), ...`.
    pub fn try_from_packed(values: &[f64]) -> Result<Self, ModelError> {
        let expected = packed_len(D).ok_or(ModelError::ArithmeticOverflow {
            context: "lower-triangular storage length",
        })?;
        if values.len() != expected {
            return Err(ModelError::InvalidParameter {
                parameter: "cholesky",
                expected: "D * (D + 1) / 2 lower-triangular values",
            });
        }

        let mut out = [[0.0; D]; D];
        let mut values = values.iter().copied();
        for (row, row_values) in out.iter_mut().enumerate() {
            for value in row_values.iter_mut().take(row + 1) {
                let Some(packed_value) = values.next() else {
                    return Err(ModelError::InvalidParameter {
                        parameter: "cholesky",
                        expected: "D * (D + 1) / 2 lower-triangular values",
                    });
                };
                *value = packed_value;
            }
        }
        Ok(Self { values: out })
    }

    /// Returns a lower-triangular entry, or `None` for invalid/upper entries.
    #[must_use]
    #[inline]
    pub const fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row < D && col <= row {
            Some(self.values[row][col])
        } else {
            None
        }
    }

    /// Sets a lower-triangular entry.
    pub const fn set_lower(
        &mut self,
        row: usize,
        col: usize,
        value: f64,
    ) -> Result<(), ModelError> {
        if row < D && col <= row {
            self.values[row][col] = value;
            Ok(())
        } else {
            Err(ModelError::InvalidParameter {
                parameter: "cholesky index",
                expected: "lower-triangular entry within dimension",
            })
        }
    }

    /// Returns full row storage with a zero upper triangle.
    #[must_use]
    #[inline]
    pub const fn as_full_rows(&self) -> &[[f64; D]; D] {
        &self.values
    }

    #[inline]
    pub(crate) const fn lower(&self, row: usize, col: usize) -> f64 {
        self.values[row][col]
    }
}

/// Runtime-dimensional packed lower-triangular matrix storage.
///
/// Values are stored in row-major lower-triangular order:
/// `(0,0), (1,0), (1,1), (2,0), ...`.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedLowerTriangular {
    dimension: usize,
    values: Vec<f64>,
}

impl PackedLowerTriangular {
    /// Creates packed lower-triangular storage after validating its length.
    pub fn try_new(dimension: usize, values: Vec<f64>) -> Result<Self, ModelError> {
        if dimension == 0 {
            return Err(ModelError::InvalidParameter {
                parameter: "dimension",
                expected: "positive",
            });
        }
        let expected = packed_len(dimension).ok_or(ModelError::ArithmeticOverflow {
            context: "lower-triangular storage length",
        })?;
        if values.len() != expected {
            return Err(ModelError::InvalidParameter {
                parameter: "cholesky",
                expected: "D * (D + 1) / 2 lower-triangular values",
            });
        }
        Ok(Self { dimension, values })
    }

    pub(crate) fn filled(dimension: usize, value: f64) -> Self {
        let len = packed_len(dimension)
            .expect("validated dynamic dimension has representable triangular storage");
        Self {
            dimension,
            values: vec![value; len],
        }
    }

    pub(crate) fn from_validated_parts(dimension: usize, values: Vec<f64>) -> Self {
        debug_assert_eq!(packed_len(dimension), Some(values.len()));
        Self { dimension, values }
    }

    /// Observation/parameter dimension.
    #[must_use]
    #[inline]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Packed lower-triangular values.
    #[must_use]
    #[inline]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Mutable packed lower-triangular values.
    #[must_use]
    #[inline]
    pub fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }

    /// Returns a lower-triangular entry, or `None` for invalid/upper entries.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row < self.dimension && col <= row {
            packed_index(row, col).and_then(|index| self.values.get(index).copied())
        } else {
            None
        }
    }

    /// Returns a mutable lower-triangular entry, or `None` for invalid/upper entries.
    #[must_use]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut f64> {
        if row < self.dimension && col <= row {
            packed_index(row, col).and_then(|index| self.values.get_mut(index))
        } else {
            None
        }
    }

    #[inline]
    pub(crate) fn lower(&self, row: usize, col: usize) -> f64 {
        self.values[packed_index(row, col).expect("valid lower-triangular index")]
    }
}

#[cfg(test)]
mod tests {
    use super::{FixedLowerTriangular, PackedLowerTriangular};

    #[test]
    fn fixed_storage_normalizes_upper_triangle_and_checks_indices() {
        let matrix = FixedLowerTriangular::from_lower_rows([[1.0, 9.0], [2.0, 3.0]]);
        assert_eq!(matrix.get(0, 0), Some(1.0));
        assert_eq!(matrix.get(1, 0), Some(2.0));
        assert_eq!(matrix.get(0, 1), None);
        assert!(matrix.as_full_rows()[0][1].abs() <= f64::EPSILON);
    }

    #[test]
    fn packed_storage_uses_shared_cholesky_indexing() {
        let mut matrix = PackedLowerTriangular::try_new(2, vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(matrix.get(1, 0), Some(2.0));
        *matrix.get_mut(1, 1).unwrap() = 4.0;
        assert_eq!(matrix.values(), &[1.0, 2.0, 4.0]);
        assert_eq!(matrix.get(0, 1), None);
    }
}
