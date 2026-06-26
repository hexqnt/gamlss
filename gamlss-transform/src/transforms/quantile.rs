use gamlss_special::{unit_normal_cdf, unit_normal_quantile};

use crate::transforms::{
    TargetTransform, TransformError, validate_finite_value, validate_non_empty_finite,
};

/// Empirical quantile transform to approximately uniform values.
///
/// Fitted values are mapped to empirical probabilities in `(0, 1)`. New values
/// outside the fitted target range are clamped to the fitted boundary
/// probabilities, and inverse values outside that probability range are
/// clamped to the fitted target range.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QuantileUniform;

impl TargetTransform for QuantileUniform {
    type State = QuantileState;

    fn fit(y: &[f64]) -> Result<Self::State, TransformError> {
        QuantileState::fit(y)
    }

    #[inline]
    fn transform(state: &Self::State, y: f64) -> f64 {
        state.probability_at(y)
    }

    #[inline]
    fn inverse(state: &Self::State, value: f64) -> f64 {
        state.value_at_probability(value)
    }

    #[inline]
    fn validate_transform_value(state: &Self::State, y: f64) -> Result<(), TransformError> {
        validate_finite_value(y)?;
        state.validate()
    }

    #[inline]
    fn validate_inverse_value(state: &Self::State, value: f64) -> Result<(), TransformError> {
        validate_finite_value(value)?;
        state.validate()
    }
}

/// Empirical quantile transform to approximately standard-normal values.
///
/// This uses the same empirical state as [`QuantileUniform`] and maps
/// probabilities through the standard-normal quantile. Inverse values are
/// converted back through the standard-normal CDF and clamped to the fitted
/// empirical probability range.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QuantileNormal;

impl TargetTransform for QuantileNormal {
    type State = QuantileState;

    fn fit(y: &[f64]) -> Result<Self::State, TransformError> {
        QuantileState::fit(y)
    }

    #[inline]
    fn transform(state: &Self::State, y: f64) -> f64 {
        unit_normal_quantile(state.probability_at(y))
    }

    #[inline]
    fn inverse(state: &Self::State, value: f64) -> f64 {
        state.value_at_probability(unit_normal_cdf(value))
    }

    #[inline]
    fn validate_transform_value(state: &Self::State, y: f64) -> Result<(), TransformError> {
        validate_finite_value(y)?;
        state.validate()
    }

    #[inline]
    fn validate_inverse_value(state: &Self::State, value: f64) -> Result<(), TransformError> {
        validate_finite_value(value)?;
        state.validate()
    }
}

/// State for empirical quantile transforms.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantileState {
    /// Sorted unique original-scale values.
    pub values: Vec<f64>,
    /// Empirical probabilities associated with `values`.
    pub probabilities: Vec<f64>,
}

impl QuantileState {
    /// Creates a validated empirical quantile state.
    ///
    /// # Errors
    ///
    /// Returns [`TransformError::InvalidParameter`] if the vectors are empty,
    /// have different lengths, contain non-finite values, or are not strictly
    /// increasing.
    pub fn try_new(values: Vec<f64>, probabilities: Vec<f64>) -> Result<Self, TransformError> {
        let state = Self {
            values,
            probabilities,
        };
        state.validate()?;
        Ok(state)
    }

    fn fit(y: &[f64]) -> Result<Self, TransformError> {
        validate_non_empty_finite(y)?;

        let mut sorted = y.to_vec();
        sorted.sort_by(f64::total_cmp);

        let mut values = Vec::with_capacity(sorted.len());
        let mut probabilities = Vec::with_capacity(sorted.len());
        let mut start = 0;
        while start < sorted.len() {
            let value = sorted[start];
            let mut end = start + 1;
            while end < sorted.len() && sorted[end].total_cmp(&value).is_eq() {
                end += 1;
            }

            #[allow(clippy::cast_precision_loss)]
            let probability = 0.5f64.mul_add((start + end - 1) as f64, 0.5) / sorted.len() as f64;
            values.push(value);
            probabilities.push(probability);
            start = end;
        }

        Self::try_new(values, probabilities)
    }

    fn validate(&self) -> Result<(), TransformError> {
        if self.values.is_empty() || self.values.len() != self.probabilities.len() {
            return Err(TransformError::InvalidParameter {
                name: "quantile_state",
            });
        }
        if !self.values.iter().copied().all(f64::is_finite)
            || !self.probabilities.iter().copied().all(f64::is_finite)
        {
            return Err(TransformError::InvalidParameter {
                name: "quantile_state",
            });
        }
        if !self
            .values
            .windows(2)
            .all(|window| window[0].total_cmp(&window[1]).is_lt())
            || !self
                .probabilities
                .windows(2)
                .all(|window| window[0].total_cmp(&window[1]).is_lt())
        {
            return Err(TransformError::InvalidParameter {
                name: "quantile_state",
            });
        }
        if self.lower_probability() <= 0.0 || self.upper_probability() >= 1.0 {
            return Err(TransformError::InvalidParameter {
                name: "quantile_state",
            });
        }
        Ok(())
    }

    fn lower_probability(&self) -> f64 {
        self.probabilities[0]
    }

    fn upper_probability(&self) -> f64 {
        self.probabilities[self.probabilities.len() - 1]
    }

    fn probability_at(&self, value: f64) -> f64 {
        if value <= self.values[0] {
            return self.lower_probability();
        }
        let last = self.values.len() - 1;
        if value >= self.values[last] {
            return self.upper_probability();
        }

        match self
            .values
            .binary_search_by(|probe| probe.total_cmp(&value))
        {
            Ok(index) => self.probabilities[index],
            Err(index) => interpolate(
                value,
                self.values[index - 1],
                self.values[index],
                self.probabilities[index - 1],
                self.probabilities[index],
            ),
        }
    }

    fn value_at_probability(&self, probability: f64) -> f64 {
        let probability = probability.clamp(self.lower_probability(), self.upper_probability());
        if probability <= self.lower_probability() {
            return self.values[0];
        }
        let last = self.probabilities.len() - 1;
        if probability >= self.upper_probability() {
            return self.values[last];
        }

        match self
            .probabilities
            .binary_search_by(|probe| probe.total_cmp(&probability))
        {
            Ok(index) => self.values[index],
            Err(index) => interpolate(
                probability,
                self.probabilities[index - 1],
                self.probabilities[index],
                self.values[index - 1],
                self.values[index],
            ),
        }
    }
}

#[inline]
fn interpolate(x: f64, left_x: f64, right_x: f64, left_y: f64, right_y: f64) -> f64 {
    let weight = (x - left_x) / (right_x - left_x);
    weight.mul_add(right_y - left_y, left_y)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{QuantileNormal, QuantileUniform, TargetTransform, TransformError};

    #[test]
    fn uniform_transform_is_monotone_and_round_trips_fitted_values() {
        let y = [-2.0, -1.0, 0.0, 2.0, 8.0];
        let (state, transformed) = QuantileUniform::fit_transform(&y).unwrap();
        let restored = QuantileUniform::inverse_slice(&state, &transformed).unwrap();

        assert!(transformed.windows(2).all(|window| window[0] < window[1]));
        for (actual, expected) in restored.iter().zip(y) {
            assert_relative_eq!(*actual, expected);
        }
    }

    #[test]
    fn uniform_transform_uses_average_probability_for_ties() {
        let state = QuantileUniform::fit(&[1.0, 1.0, 3.0, 5.0]).unwrap();

        assert_eq!(state.values, vec![1.0, 3.0, 5.0]);
        assert_relative_eq!(QuantileUniform::transform(&state, 1.0), 0.25);
    }

    #[test]
    fn uniform_inverse_clamps_outside_fitted_probability_range() {
        let state = QuantileUniform::fit(&[-1.0, 1.0, 3.0]).unwrap();

        assert_relative_eq!(QuantileUniform::inverse(&state, -1.0), -1.0);
        assert_relative_eq!(QuantileUniform::inverse(&state, 2.0), 3.0);
        assert_relative_eq!(QuantileUniform::transform(&state, -10.0), 1.0 / 6.0);
    }

    #[test]
    fn normal_transform_is_finite_monotone_and_round_trips_fitted_values() {
        let y = [-2.0, -1.0, 0.0, 2.0, 8.0];
        let (state, transformed) = QuantileNormal::fit_transform(&y).unwrap();
        let restored = QuantileNormal::inverse_slice(&state, &transformed).unwrap();

        assert!(transformed.iter().all(|value| value.is_finite()));
        assert!(transformed.windows(2).all(|window| window[0] < window[1]));
        for (actual, expected) in restored.iter().zip(y) {
            assert_relative_eq!(*actual, expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn rejects_empty_and_non_finite_values() {
        assert_eq!(
            QuantileUniform::fit(&[]).unwrap_err(),
            TransformError::EmptyInput
        );
        assert_eq!(
            QuantileNormal::fit(&[1.0, f64::INFINITY]).unwrap_err(),
            TransformError::NonFiniteValue
        );
    }

    #[test]
    fn checked_api_rejects_invalid_manual_state() {
        let state = super::QuantileState {
            values: Vec::new(),
            probabilities: Vec::new(),
        };

        assert_eq!(
            QuantileUniform::checked_transform(&state, 1.0).unwrap_err(),
            TransformError::InvalidParameter {
                name: "quantile_state"
            }
        );
    }

    #[test]
    fn state_constructor_rejects_unsorted_or_boundary_probabilities() {
        assert_eq!(
            super::QuantileState::try_new(vec![1.0, 0.0], vec![0.25, 0.75]).unwrap_err(),
            TransformError::InvalidParameter {
                name: "quantile_state"
            }
        );
        assert_eq!(
            super::QuantileState::try_new(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap_err(),
            TransformError::InvalidParameter {
                name: "quantile_state"
            }
        );
    }
}
