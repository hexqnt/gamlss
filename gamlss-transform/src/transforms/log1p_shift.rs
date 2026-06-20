use crate::{
    TargetTransform, TransformError, validate_non_empty_finite, validate_output_len,
    validate_shifted_non_negative,
};

/// `log1p` transform with a fitted shift for targets that may contain zero or
/// negative values.
///
/// The fitted shift maps the minimum training value to `margin`, so
/// `transform(y) = ln(1 + y + shift)`. New values must satisfy
/// `y + shift >= 0`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Log1pShift;

impl TargetTransform for Log1pShift {
    type State = Log1pShiftState;

    fn fit(y: &[f64]) -> Result<Self::State, TransformError> {
        validate_non_empty_finite(y)?;

        let min = y.iter().copied().fold(f64::INFINITY, f64::min);
        let margin = if min <= 0.0 { 1.0e-12 } else { 0.0 };
        let shift = (-min + margin).max(0.0);
        Ok(Log1pShiftState { shift, margin })
    }

    #[inline(always)]
    fn transform(state: &Self::State, y: f64) -> f64 {
        (y + state.shift).ln_1p()
    }

    #[inline(always)]
    fn inverse(state: &Self::State, value: f64) -> f64 {
        value.exp_m1() - state.shift
    }

    #[inline]
    fn transform_slice(state: &Self::State, y: &[f64]) -> Result<Vec<f64>, TransformError> {
        let mut out = vec![0.0; y.len()];
        Self::transform_into(state, y, &mut out)?;
        Ok(out)
    }

    #[inline]
    fn transform_into(
        state: &Self::State,
        y: &[f64],
        out: &mut [f64],
    ) -> Result<(), TransformError> {
        validate_output_len(y.len(), out.len())?;
        validate_shifted_non_negative(y, state.shift)?;
        for (out, value) in out.iter_mut().zip(y.iter().copied()) {
            *out = Self::transform(state, value);
        }
        Ok(())
    }
}

/// State for [`Log1pShift`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Log1pShiftState {
    /// Additive shift applied before `ln_1p`.
    pub shift: f64,
    /// Small non-negative distance between the shifted training minimum and
    /// zero.
    pub margin: f64,
}

impl Log1pShiftState {
    /// Lower bound accepted by [`Log1pShift::transform_slice`].
    #[must_use]
    #[inline(always)]
    pub fn lower_bound(self) -> f64 {
        -self.shift
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{Log1pShift, TargetTransform, TransformError};

    #[test]
    fn round_trips_values_with_negative_minimum() {
        let y = [-3.0, 0.0, 4.0];
        let (state, transformed) = Log1pShift::fit_transform(&y).unwrap();
        let restored = Log1pShift::inverse_slice(&state, &transformed).unwrap();

        assert!(state.shift > 0.0);
        assert_relative_eq!(state.lower_bound(), -state.shift);
        for (actual, expected) in restored.iter().zip(y) {
            assert_relative_eq!(*actual, expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn rejects_values_below_fitted_lower_bound() {
        let state = Log1pShift::fit(&[-2.0, 1.0]).unwrap();

        assert_eq!(
            Log1pShift::transform_slice(&state, &[-2.1]).unwrap_err(),
            TransformError::BelowLowerBound
        );
    }
}
