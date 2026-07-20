use std::num::NonZeroUsize;

use crate::ModelError;

/// Borrowed equal-width rows over one contiguous flat observation buffer.
///
/// This adapter is the standard zero-copy observation view for
/// runtime-dimensional families whose observation carrier is `&[f64]`.
/// Construction validates the row geometry and optional weights once, before
/// the likelihood hot path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseRows<'a> {
    values: &'a [f64],
    width: NonZeroUsize,
    weights: Option<&'a [f64]>,
}

impl<'a> DenseRows<'a> {
    /// Creates unweighted rows over `values` without copying.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] if `width` is zero, or
    /// [`ModelError::DenseObservationSize`] if `values.len()` is not divisible
    /// by `width`.
    pub fn try_new(values: &'a [f64], width: usize) -> Result<Self, ModelError> {
        Self::try_new_inner(values, width, None)
    }

    /// Creates weighted rows over `values` without copying.
    ///
    /// # Errors
    ///
    /// Returns the geometry errors from [`Self::try_new`],
    /// [`ModelError::WeightLength`] if there is not one weight per row, or
    /// [`ModelError::InvalidWeight`] for a non-finite or negative weight.
    pub fn try_new_weighted(
        values: &'a [f64],
        width: usize,
        weights: &'a [f64],
    ) -> Result<Self, ModelError> {
        Self::try_new_inner(values, width, Some(weights))
    }

    fn try_new_inner(
        values: &'a [f64],
        width: usize,
        weights: Option<&'a [f64]>,
    ) -> Result<Self, ModelError> {
        let Some(width) = NonZeroUsize::new(width) else {
            return Err(ModelError::InvalidParameter {
                parameter: "dense observation row width",
                expected: "positive",
            });
        };
        if !values.len().is_multiple_of(width.get()) {
            return Err(ModelError::DenseObservationSize {
                actual_values: values.len(),
                row_width: width.get(),
            });
        }
        let nrows = values.len() / width.get();
        if let Some(weights) = weights {
            if weights.len() != nrows {
                return Err(ModelError::WeightLength {
                    expected: nrows,
                    actual: weights.len(),
                });
            }
            for (index, weight) in weights.iter().copied().enumerate() {
                validate_observation_weight(index, weight)?;
            }
        }
        Ok(Self {
            values,
            width,
            weights,
        })
    }

    /// Flat row-major observation values.
    #[must_use]
    #[inline]
    pub const fn values(&self) -> &'a [f64] {
        self.values
    }

    /// Number of values in each row.
    #[must_use]
    #[inline]
    pub const fn width(&self) -> usize {
        self.width.get()
    }

    /// Number of rows in the view.
    #[must_use]
    #[inline]
    pub const fn nrows(&self) -> usize {
        self.values.len() / self.width.get()
    }

    /// Optional observation weights.
    #[must_use]
    #[inline]
    pub const fn weights(&self) -> Option<&'a [f64]> {
        self.weights
    }
}

impl<'row> ObservationView<'row> for DenseRows<'_> {
    type Observation = &'row [f64];

    #[inline]
    fn len(&self) -> usize {
        self.nrows()
    }

    #[inline]
    fn observation_at(&'row self, row: usize) -> Self::Observation {
        let start = row * self.width.get();
        &self.values[start..start + self.width.get()]
    }

    #[inline]
    fn weight_at(&self, row: usize) -> f64 {
        self.weights.map_or(1.0, |weights| weights[row])
    }

    fn validate(&self) -> Result<(), ModelError> {
        Ok(())
    }
}

/// Borrowed scalar observations that reject `NaN` and infinities at validation.
///
/// The plain `&[f64]` observation view intentionally stays permissive so
/// weighted workflows can keep rows whose response is missing or outside a
/// family's domain when their weight is zero. Use this adapter, or
/// [`Gamlss::try_new_strict`](crate::Gamlss::try_new_strict), when every scalar
/// response value must be finite before objective evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FiniteScalarObservations<'a> {
    values: &'a [f64],
}

impl<'a> FiniteScalarObservations<'a> {
    /// Creates a finite scalar observation view.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidObservation`] when any response value is
    /// `NaN`, `inf` or `-inf`.
    pub fn new(values: &'a [f64]) -> Result<Self, ModelError> {
        validate_scalar_observations(values)?;
        Ok(Self { values })
    }

    /// Returns the underlying response slice.
    #[must_use]
    #[inline]
    pub const fn values(&self) -> &'a [f64] {
        self.values
    }
}

impl<'row> ObservationView<'row> for FiniteScalarObservations<'_> {
    type Observation = f64;

    #[inline]
    fn len(&self) -> usize {
        self.values.len()
    }

    #[inline]
    fn observation_at(&'row self, row: usize) -> Self::Observation {
        self.values[row]
    }

    #[inline]
    fn weight_at(&self, _row: usize) -> f64 {
        1.0
    }

    #[inline]
    fn validate(&self) -> Result<(), ModelError> {
        validate_scalar_observations(self.values)
    }
}

/// Read-only row-wise observation access for training objective evaluation.
///
/// This trait is intentionally small: it describes the row-wise data needed by
/// the likelihood loop. Implementations should make
/// [`len`](Self::len) O(1), keep it stable for the lifetime of the model, and
/// provide deterministic, panic-free access for `row < len()`.
///
/// The trait is parameterized by the borrow lifetime so compiled objectives can
/// remain generic over borrowed storage backends. This permits zero-copy row
/// views such as `&'row [f64]`.
pub trait ObservationView<'row> {
    /// Observation representation returned for one row.
    type Observation;

    /// Number of observations.
    fn len(&self) -> usize;

    /// Returns `true` if there are no observations.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Observation value for `row`.
    fn observation_at(&'row self, row: usize) -> Self::Observation;

    /// Non-negative finite observation weight for `row`.
    fn weight_at(&self, row: usize) -> f64;

    /// Validates observation-level invariants before hot-path evaluation.
    #[inline]
    fn validate(&self) -> Result<(), ModelError> {
        for row in 0..self.len() {
            validate_observation_weight(row, self.weight_at(row))?;
        }
        Ok(())
    }
}

impl<'row> ObservationView<'row> for &[f64] {
    type Observation = f64;

    #[inline]
    fn len(&self) -> usize {
        <[f64]>::len(self)
    }

    #[inline]
    fn observation_at(&'row self, row: usize) -> Self::Observation {
        self[row]
    }

    #[inline]
    fn weight_at(&self, _row: usize) -> f64 {
        1.0
    }

    #[inline]
    fn validate(&self) -> Result<(), ModelError> {
        Ok(())
    }
}

impl<'row> ObservationView<'row> for (&[f64], &[f64]) {
    type Observation = f64;

    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    fn observation_at(&'row self, row: usize) -> Self::Observation {
        self.0[row]
    }

    #[inline]
    fn weight_at(&self, row: usize) -> f64 {
        self.1[row]
    }

    fn validate(&self) -> Result<(), ModelError> {
        let expected = self.0.len();
        let actual = self.1.len();
        if actual != expected {
            return Err(ModelError::WeightLength { expected, actual });
        }

        for (index, weight) in self.1.iter().copied().enumerate() {
            validate_observation_weight(index, weight)?;
        }

        Ok(())
    }
}

impl<'row, const N: usize> ObservationView<'row> for &[[f64; N]] {
    type Observation = [f64; N];

    #[inline]
    fn len(&self) -> usize {
        <[[f64; N]]>::len(self)
    }

    #[inline]
    fn observation_at(&'row self, row: usize) -> Self::Observation {
        self[row]
    }

    #[inline]
    fn weight_at(&self, _row: usize) -> f64 {
        1.0
    }

    #[inline]
    fn validate(&self) -> Result<(), ModelError> {
        Ok(())
    }
}

impl<'row, const N: usize> ObservationView<'row> for (&[[f64; N]], &[f64]) {
    type Observation = [f64; N];

    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    fn observation_at(&'row self, row: usize) -> Self::Observation {
        self.0[row]
    }

    #[inline]
    fn weight_at(&self, row: usize) -> f64 {
        self.1[row]
    }

    fn validate(&self) -> Result<(), ModelError> {
        let expected = self.0.len();
        let actual = self.1.len();
        if actual != expected {
            return Err(ModelError::WeightLength { expected, actual });
        }

        for (index, weight) in self.1.iter().copied().enumerate() {
            validate_observation_weight(index, weight)?;
        }

        Ok(())
    }
}

fn validate_observation_weight(index: usize, weight: f64) -> Result<(), ModelError> {
    if weight.is_finite() && weight >= 0.0 {
        Ok(())
    } else {
        Err(ModelError::InvalidWeight { index })
    }
}

fn validate_scalar_observations(values: &[f64]) -> Result<(), ModelError> {
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(ModelError::InvalidObservation { index });
        }
    }
    Ok(())
}
