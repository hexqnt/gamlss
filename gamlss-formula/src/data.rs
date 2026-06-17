use std::{fmt, marker::PhantomData, sync::Arc};

use gamlss_core::{ModelError, ObservationView};

use crate::FormulaError;

/// Marker type for categorical columns.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Category;

/// Typed reference to a named input column.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Col<T> {
    name: Arc<str>,
    marker: PhantomData<T>,
}

impl<T> Col<T> {
    /// Returns the external column name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl<T> fmt::Debug for Col<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Col").field(&self.name).finish()
    }
}

/// Creates a typed column reference.
#[must_use]
pub fn col<T>(name: impl Into<Arc<str>>) -> Col<T> {
    Col {
        name: name.into(),
        marker: PhantomData,
    }
}

/// Numeric column storage returned by [`DataView`].
#[derive(Debug, Clone, PartialEq)]
pub enum NumericCol<'a> {
    /// Borrowed contiguous `f64` storage.
    Borrowed(&'a [f64]),
    /// Owned contiguous `f64` storage.
    Owned(Vec<f64>),
}

impl<'a> NumericCol<'a> {
    /// Returns the column as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        match self {
            Self::Borrowed(values) => values,
            Self::Owned(values) => values,
        }
    }

    pub(crate) fn into_response(self) -> NumericResponse<'a> {
        match self {
            Self::Borrowed(values) => NumericResponse::Borrowed(values),
            Self::Owned(values) => NumericResponse::Owned(values),
        }
    }
}

/// Boolean column storage returned by [`DataView`].
#[derive(Debug, Clone, PartialEq)]
pub enum BoolCol<'a> {
    /// Borrowed contiguous `bool` storage.
    Borrowed(&'a [bool]),
    /// Owned contiguous `bool` storage.
    Owned(Vec<bool>),
}

impl BoolCol<'_> {
    /// Returns the column as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[bool] {
        match self {
            Self::Borrowed(values) => values,
            Self::Owned(values) => values,
        }
    }
}

/// Categorical column storage returned by [`DataView`].
#[derive(Debug, Clone, PartialEq)]
pub enum CatCol<'a> {
    /// Borrowed string levels.
    Borrowed(&'a [String]),
    /// Owned string levels.
    Owned(Vec<String>),
}

impl CatCol<'_> {
    /// Returns the column as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[String] {
        match self {
            Self::Borrowed(values) => values,
            Self::Owned(values) => values,
        }
    }
}

/// Read-only data access contract for the formula layer.
pub trait DataView {
    /// Number of rows visible to the model builder.
    fn nrows(&self) -> usize;

    /// Returns an `f64` column by typed column reference.
    ///
    /// Implementations should return [`FormulaError::UnknownColumn`] when the
    /// name is not available. The formula layer validates row counts.
    fn f64_col(&self, col: &Col<f64>) -> Result<NumericCol<'_>, FormulaError>;

    /// Returns a `bool` column by typed column reference.
    fn bool_col(&self, col: &Col<bool>) -> Result<BoolCol<'_>, FormulaError> {
        Err(FormulaError::UnsupportedColumnType {
            name: col.name().to_owned(),
            requested: "bool",
        })
    }

    /// Returns a categorical column by typed column reference.
    fn cat_col(&self, col: &Col<Category>) -> Result<CatCol<'_>, FormulaError> {
        Err(FormulaError::UnsupportedColumnType {
            name: col.name().to_owned(),
            requested: "category",
        })
    }
}

/// Response storage used by compiled formula models.
#[derive(Debug, Clone, PartialEq)]
pub enum NumericResponse<'a> {
    /// Borrowed response storage.
    Borrowed(&'a [f64]),
    /// Owned response storage.
    Owned(Vec<f64>),
    /// Response with observation weights.
    Weighted {
        /// Response values.
        values: NumericCol<'a>,
        /// Observation weights.
        weights: NumericCol<'a>,
    },
}

impl NumericResponse<'_> {
    /// Returns response values as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        match self {
            Self::Borrowed(values) => values,
            Self::Owned(values) => values,
            Self::Weighted { values, .. } => values.as_slice(),
        }
    }

    /// Returns observation weights when present.
    #[must_use]
    pub fn weights(&self) -> Option<&[f64]> {
        match self {
            Self::Borrowed(_) | Self::Owned(_) => None,
            Self::Weighted { weights, .. } => Some(weights.as_slice()),
        }
    }
}

impl<'row> ObservationView<'row> for NumericResponse<'_> {
    type Observation = f64;

    fn len(&self) -> usize {
        self.as_slice().len()
    }

    fn observation_at(&'row self, row: usize) -> Self::Observation {
        self.as_slice()[row]
    }

    fn weight_at(&self, _row: usize) -> f64 {
        self.weights().map_or(1.0, |weights| weights[_row])
    }

    fn validate(&self) -> Result<(), ModelError> {
        if let Some(weights) = self.weights() {
            let expected = self.as_slice().len();
            let actual = weights.len();
            if actual != expected {
                return Err(ModelError::WeightLength { expected, actual });
            }
            for (index, weight) in weights.iter().copied().enumerate() {
                if !weight.is_finite() || weight < 0.0 {
                    return Err(ModelError::InvalidWeight { index });
                }
            }
        }
        Ok(())
    }
}
