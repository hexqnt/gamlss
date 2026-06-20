/// Returns `true` for finite values strictly greater than zero.
#[inline(always)]
pub(crate) fn is_positive_finite(value: f64) -> bool {
    value > 0.0 && value.is_finite()
}

/// Returns `true` for finite probabilities in the closed interval `[0, 1]`.
#[inline(always)]
pub(crate) fn is_probability(value: f64) -> bool {
    (0.0..=1.0).contains(&value)
}

/// Returns `true` for finite probabilities in the open interval `(0, 1)`.
#[inline(always)]
pub(crate) fn is_strict_probability(value: f64) -> bool {
    value > 0.0 && value < 1.0
}

/// Returns `true` for a finite location and a positive finite scale.
#[inline(always)]
pub(crate) fn is_finite_location_scale(location: f64, scale: f64) -> bool {
    location.is_finite() && is_positive_finite(scale)
}
