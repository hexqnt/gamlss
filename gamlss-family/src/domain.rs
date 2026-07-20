/// Returns `true` for finite values strictly greater than zero.
#[inline]
pub fn is_positive_finite(value: f64) -> bool {
    value > 0.0 && value.is_finite()
}

/// Returns `true` for finite probabilities in the closed interval `[0, 1]`.
#[inline]
pub fn is_probability(value: f64) -> bool {
    (0.0..=1.0).contains(&value)
}

/// Returns `true` for finite probabilities in the open interval `(0, 1)`.
#[inline]
pub fn is_strict_probability(value: f64) -> bool {
    value > 0.0 && value < 1.0
}

/// Returns `true` for a representable interior simplex with at least two components.
#[inline]
#[allow(clippy::cast_precision_loss)]
pub fn is_interior_simplex(values: &[f64]) -> bool {
    values.len() >= 2
        && values.iter().all(|value| *value > 0.0 && value.is_finite())
        && (values.iter().sum::<f64>() - 1.0).abs() <= 16.0 * f64::EPSILON * values.len() as f64
}

/// Returns `true` for a finite location and a positive finite scale.
#[inline]
pub fn is_finite_location_scale(location: f64, scale: f64) -> bool {
    location.is_finite() && is_positive_finite(scale)
}
