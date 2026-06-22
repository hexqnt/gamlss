use crate::ModelError;

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
    #[inline(always)]
    pub const fn values(&self) -> &'a [f64] {
        self.values
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
    #[inline(always)]
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

    #[inline(always)]
    fn len(&self) -> usize {
        <[f64]>::len(self)
    }

    #[inline(always)]
    fn observation_at(&'row self, row: usize) -> Self::Observation {
        self[row]
    }

    #[inline(always)]
    fn weight_at(&self, _row: usize) -> f64 {
        1.0
    }

    #[inline(always)]
    fn validate(&self) -> Result<(), ModelError> {
        Ok(())
    }
}

impl<'row, 'a> ObservationView<'row> for FiniteScalarObservations<'a> {
    type Observation = f64;

    #[inline(always)]
    fn len(&self) -> usize {
        self.values.len()
    }

    #[inline(always)]
    fn observation_at(&'row self, row: usize) -> Self::Observation {
        self.values[row]
    }

    #[inline(always)]
    fn weight_at(&self, _row: usize) -> f64 {
        1.0
    }

    #[inline]
    fn validate(&self) -> Result<(), ModelError> {
        validate_scalar_observations(self.values)
    }
}

impl<'row> ObservationView<'row> for (&[f64], &[f64]) {
    type Observation = f64;

    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    fn observation_at(&'row self, row: usize) -> Self::Observation {
        self.0[row]
    }

    #[inline(always)]
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

    #[inline(always)]
    fn len(&self) -> usize {
        <[[f64; N]]>::len(self)
    }

    #[inline(always)]
    fn observation_at(&'row self, row: usize) -> Self::Observation {
        self[row]
    }

    #[inline(always)]
    fn weight_at(&self, _row: usize) -> f64 {
        1.0
    }

    #[inline(always)]
    fn validate(&self) -> Result<(), ModelError> {
        Ok(())
    }
}

impl<'row, const N: usize> ObservationView<'row> for (&[[f64; N]], &[f64]) {
    type Observation = [f64; N];

    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    fn observation_at(&'row self, row: usize) -> Self::Observation {
        self.0[row]
    }

    #[inline(always)]
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
