use gamlss_core::ModelError;

use crate::SplineError;
use crate::validation::finite_data_range;

const EXPECTED_WEIGHTS: &str = "finite and >= 0 with positive mass at two distinct coordinates";

/// Strategy used to place interior knots from observed coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnotPlacement {
    /// Equally spaced interior knots over the finite data range.
    Uniform,
    /// Linearly interpolated empirical quantiles.
    Quantile,
    /// Inverse weighted empirical-CDF quantiles.
    WeightedQuantile,
    /// Explicit boundaries and interior knots supplied by the caller.
    Explicit,
}

/// Persisted clamped/open knot-vector metadata.
///
/// This type records both the exact fitted knot sequence and how it was
/// obtained. Reusing it for prediction therefore never recomputes quantiles or
/// data ranges from new observations.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenKnotVector {
    degree: usize,
    placement: KnotPlacement,
    knots: Box<[f64]>,
}

impl OpenKnotVector {
    /// Builds uniform or empirical-quantile knots from unweighted data.
    ///
    /// # Errors
    ///
    /// Returns an error for empty/non-finite/constant data, insufficient basis
    /// count, or [`KnotPlacement::WeightedQuantile`] and
    /// [`KnotPlacement::Explicit`], which require their dedicated constructors.
    pub fn from_data(
        x: &[f64],
        n_basis: usize,
        degree: usize,
        placement: KnotPlacement,
    ) -> Result<Self, SplineError> {
        validate_basis_count(n_basis, degree)?;
        let (lower, upper) = finite_data_range(x)?;
        let interior_count = n_basis - degree - 1;
        let interior = match placement {
            KnotPlacement::Uniform => uniform_interior(lower, upper, interior_count),
            KnotPlacement::Quantile => quantile_interior(x, interior_count),
            KnotPlacement::WeightedQuantile | KnotPlacement::Explicit => {
                return Err(ModelError::InvalidParameter {
                    parameter: "unweighted knot placement",
                    expected: "Uniform or Quantile",
                }
                .into());
            }
        };
        Self::from_parts_with_placement(lower, upper, interior, degree, placement)
    }

    /// Builds inverse weighted-empirical-CDF knots.
    ///
    /// Zero-weight coordinates do not affect either boundaries or quantiles.
    ///
    /// # Errors
    ///
    /// Returns an error for mismatched lengths, invalid weights, fewer than
    /// two distinct positive-weight coordinates, or insufficient basis count.
    pub fn from_weighted_data(
        x: &[f64],
        weights: &[f64],
        n_basis: usize,
        degree: usize,
    ) -> Result<Self, SplineError> {
        validate_basis_count(n_basis, degree)?;
        if x.len() != weights.len() {
            return Err(ModelError::DesignSize {
                expected_values: x.len(),
                actual_values: weights.len(),
            }
            .into());
        }
        let mut values = x
            .iter()
            .copied()
            .zip(weights.iter().copied())
            .filter_map(|(value, weight)| (weight > 0.0).then_some((value, weight)))
            .collect::<Vec<_>>();
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
            || values.iter().any(|(value, _)| !value.is_finite())
            || values.len() < 2
        {
            return Err(invalid_weight_error());
        }
        values.sort_by(|left, right| left.0.total_cmp(&right.0));
        let lower = values[0].0;
        let upper = values[values.len() - 1].0;
        let total = values.iter().map(|(_, weight)| weight).sum::<f64>();
        if !total.is_finite() || lower >= upper {
            return Err(invalid_weight_error());
        }
        let interior_count = n_basis - degree - 1;
        let interior = weighted_quantile_interior(&values, total, interior_count);
        Self::from_parts_with_placement(
            lower,
            upper,
            interior,
            degree,
            KnotPlacement::WeightedQuantile,
        )
    }

    /// Builds a clamped knot vector from explicit boundaries and interior knots.
    ///
    /// # Errors
    ///
    /// Returns an error unless boundaries are finite and strictly increasing,
    /// and interior knots are finite, non-decreasing, and inside the closed
    /// boundary interval.
    pub fn from_parts(
        lower: f64,
        upper: f64,
        interior: Vec<f64>,
        degree: usize,
    ) -> Result<Self, SplineError> {
        Self::from_parts_with_placement(lower, upper, interior, degree, KnotPlacement::Explicit)
    }

    fn from_parts_with_placement(
        lower: f64,
        upper: f64,
        interior: Vec<f64>,
        degree: usize,
        placement: KnotPlacement,
    ) -> Result<Self, SplineError> {
        if !lower.is_finite()
            || !upper.is_finite()
            || lower >= upper
            || interior.iter().any(|knot| !knot.is_finite())
            || interior.windows(2).any(|window| window[0] > window[1])
            || interior.iter().any(|knot| *knot < lower || *knot > upper)
        {
            return Err(SplineError::InvalidKnots);
        }
        let n_basis = interior
            .len()
            .checked_add(degree)
            .and_then(|value| value.checked_add(1))
            .ok_or(SplineError::ParameterOverflow)?;
        let repeats = degree
            .checked_add(1)
            .ok_or(SplineError::ParameterOverflow)?;
        let knot_count = n_basis
            .checked_add(repeats)
            .ok_or(SplineError::ParameterOverflow)?;
        let mut knots = Vec::with_capacity(knot_count);
        knots.extend(std::iter::repeat_n(lower, repeats));
        knots.extend(interior);
        knots.extend(std::iter::repeat_n(upper, repeats));
        Ok(Self {
            degree,
            placement,
            knots: knots.into_boxed_slice(),
        })
    }

    /// Spline degree.
    #[must_use]
    pub const fn degree(&self) -> usize {
        self.degree
    }

    /// Number of basis functions implied by the knot vector.
    #[must_use]
    pub const fn n_basis(&self) -> usize {
        self.knots.len() - self.degree - 1
    }

    /// Placement strategy used during construction.
    #[must_use]
    pub const fn placement(&self) -> KnotPlacement {
        self.placement
    }

    /// Full clamped knot vector.
    #[must_use]
    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    /// Interior knot slice.
    #[must_use]
    pub fn interior(&self) -> &[f64] {
        let repeats = self.degree + 1;
        &self.knots[repeats..self.knots.len() - repeats]
    }

    /// Consumes the metadata and returns its full knot vector.
    #[must_use]
    pub fn into_knots(self) -> Vec<f64> {
        self.knots.into_vec()
    }
}

const fn validate_basis_count(n_basis: usize, degree: usize) -> Result<(), SplineError> {
    if n_basis <= degree {
        Err(SplineError::NotEnoughBasis { n_basis, degree })
    } else {
        Ok(())
    }
}

#[allow(clippy::cast_precision_loss)]
fn uniform_interior(lower: f64, upper: f64, count: usize) -> Vec<f64> {
    (1..=count)
        .map(|index| lower + (upper - lower) * index as f64 / (count + 1) as f64)
        .collect()
}

fn quantile_interior(x: &[f64], count: usize) -> Vec<f64> {
    let mut sorted = x.to_vec();
    sorted.sort_by(f64::total_cmp);
    (1..=count)
        .map(|index| {
            #[allow(clippy::cast_precision_loss)]
            let probability = index as f64 / (count + 1) as f64;
            interpolated_quantile(&sorted, probability)
        })
        .collect()
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn interpolated_quantile(sorted: &[f64], probability: f64) -> f64 {
    let position = probability * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    (sorted[upper] - sorted[lower]).mul_add(fraction, sorted[lower])
}

#[allow(clippy::cast_precision_loss)]
fn weighted_quantile_interior(values: &[(f64, f64)], total: f64, count: usize) -> Vec<f64> {
    let mut interior = Vec::with_capacity(count);
    let mut cumulative = 0.0;
    let mut index = 0;
    for quantile in 1..=count {
        let target = total * quantile as f64 / (count + 1) as f64;
        while index + 1 < values.len() && cumulative + values[index].1 < target {
            cumulative += values[index].1;
            index += 1;
        }
        interior.push(values[index].0);
    }
    interior
}

fn invalid_weight_error() -> SplineError {
    ModelError::InvalidParameter {
        parameter: "weighted knot observations",
        expected: EXPECTED_WEIGHTS,
    }
    .into()
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::{KnotPlacement, OpenKnotVector};

    #[test]
    fn quantile_and_uniform_policies_persist_distinct_knots() {
        let x = [0.0, 0.01, 0.02, 0.1, 1.0, 8.0];
        let uniform = OpenKnotVector::from_data(&x, 7, 3, KnotPlacement::Uniform).unwrap();
        let quantile = OpenKnotVector::from_data(&x, 7, 3, KnotPlacement::Quantile).unwrap();
        assert_eq!(uniform.placement(), KnotPlacement::Uniform);
        assert_eq!(quantile.placement(), KnotPlacement::Quantile);
        assert_ne!(uniform.interior(), quantile.interior());
        assert_eq!(quantile.n_basis(), 7);
        assert_eq!(quantile.knots().len(), 11);
    }

    #[test]
    fn weighted_policy_ignores_zero_weight_extremes() {
        let knots = OpenKnotVector::from_weighted_data(
            &[f64::NAN, 0.0, 1.0, 2.0, f64::INFINITY],
            &[0.0, 1.0, 2.0, 1.0, 0.0],
            6,
            2,
        )
        .unwrap();
        assert_relative_eq!(knots.knots()[0], 0.0, epsilon = f64::EPSILON);
        assert_relative_eq!(
            knots.knots()[knots.knots().len() - 1],
            2.0,
            epsilon = f64::EPSILON
        );
        assert_eq!(knots.placement(), KnotPlacement::WeightedQuantile);

        assert!(OpenKnotVector::from_weighted_data(&[0.0, f64::NAN], &[1.0, 1.0], 4, 2).is_err());
    }
}
