use crate::SplineError;

#[inline]
pub fn finite_data_range(x: &[f64]) -> Result<(f64, f64), SplineError> {
    if x.is_empty() {
        return Err(SplineError::EmptyInput);
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
    validate_finite_range(min, max)?;
    Ok((min, max))
}

#[inline]
pub fn validate_finite_range(min: f64, max: f64) -> Result<(), SplineError> {
    if !min.is_finite() || !max.is_finite() || min >= max || !(max - min).is_finite() {
        Err(SplineError::InvalidRange)
    } else {
        Ok(())
    }
}

#[inline]
pub fn validate_coordinates(x: &[f64]) -> Result<(), SplineError> {
    if x.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(SplineError::NonFiniteValue)
    }
}
