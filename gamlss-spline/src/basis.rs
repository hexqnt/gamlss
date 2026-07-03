use gamlss_core::ModelError;

use crate::{
    BSplineBasis, ISplineBasis, MSplineBasis, NaturalCubicSplineBasis, OpenUniformSplineBasis,
    SplineError, TruncatedPowerBasis,
};

/// Common one-dimensional spline basis evaluation API.
///
/// This trait is intended for feature engineering and interop code that wants
/// to evaluate a fitted basis shape without depending on GAMLSS predictor
/// blocks. It is implemented for the crate's reusable basis metadata types.
pub trait SplineBasis1d {
    /// Number of basis functions.
    fn n_basis(&self) -> usize;

    /// Visits non-zero basis-function values at `x` without allocating.
    ///
    /// # Errors
    ///
    /// Returns [`SplineError::NonFiniteValue`] when `x` is not finite.
    fn for_each_basis(&self, x: f64, f: impl FnMut(usize, f64)) -> Result<(), SplineError>;

    /// Writes all basis-function values at `x` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`SplineError::NonFiniteValue`] when `x` is not finite. Returns
    /// [`SplineError::Model`] when `out.len() != self.n_basis()`.
    fn evaluate_into(&self, x: f64, out: &mut [f64]) -> Result<(), SplineError> {
        if out.len() != self.n_basis() {
            return Err(ModelError::DesignSize {
                expected_values: self.n_basis(),
                actual_values: out.len(),
            }
            .into());
        }

        out.fill(0.0);
        self.for_each_basis(x, |index, weight| out[index] = weight)
    }

    /// Evaluates all basis functions at `x`.
    ///
    /// # Errors
    ///
    /// Returns [`SplineError::NonFiniteValue`] when `x` is not finite.
    fn evaluate(&self, x: f64) -> Result<Vec<f64>, SplineError> {
        let mut values = vec![0.0; self.n_basis()];
        self.evaluate_into(x, &mut values)?;
        Ok(values)
    }
}

impl SplineBasis1d for BSplineBasis {
    #[inline]
    fn n_basis(&self) -> usize {
        self.n_basis()
    }

    #[inline]
    fn for_each_basis(&self, x: f64, mut f: impl FnMut(usize, f64)) -> Result<(), SplineError> {
        reject_non_finite(x)?;
        for (index, weight) in self.evaluate(x).into_iter().enumerate() {
            if weight != 0.0 {
                f(index, weight);
            }
        }
        Ok(())
    }

    #[inline]
    fn evaluate_into(&self, x: f64, out: &mut [f64]) -> Result<(), SplineError> {
        reject_non_finite(x)?;
        validate_output_len(self.n_basis(), out.len())?;
        Self::evaluate_into(self, x, out);
        Ok(())
    }
}

impl SplineBasis1d for OpenUniformSplineBasis {
    #[inline]
    fn n_basis(&self) -> usize {
        self.n_basis()
    }

    #[inline]
    fn for_each_basis(&self, x: f64, f: impl FnMut(usize, f64)) -> Result<(), SplineError> {
        self.for_each_value_basis(x, f)
    }
}

impl SplineBasis1d for MSplineBasis {
    #[inline]
    fn n_basis(&self) -> usize {
        self.n_basis()
    }

    #[inline]
    fn for_each_basis(&self, x: f64, f: impl FnMut(usize, f64)) -> Result<(), SplineError> {
        reject_non_finite(x)?;
        Self::for_each_basis(self, x, f);
        Ok(())
    }

    #[inline]
    fn evaluate_into(&self, x: f64, out: &mut [f64]) -> Result<(), SplineError> {
        reject_non_finite(x)?;
        validate_output_len(self.n_basis(), out.len())?;
        Self::evaluate_into(self, x, out);
        Ok(())
    }
}

impl SplineBasis1d for ISplineBasis {
    #[inline]
    fn n_basis(&self) -> usize {
        self.n_basis()
    }

    #[inline]
    fn for_each_basis(&self, x: f64, f: impl FnMut(usize, f64)) -> Result<(), SplineError> {
        reject_non_finite(x)?;
        Self::for_each_basis(self, x, f);
        Ok(())
    }

    #[inline]
    fn evaluate_into(&self, x: f64, out: &mut [f64]) -> Result<(), SplineError> {
        reject_non_finite(x)?;
        validate_output_len(self.n_basis(), out.len())?;
        Self::evaluate_into(self, x, out);
        Ok(())
    }
}

impl SplineBasis1d for NaturalCubicSplineBasis {
    #[inline]
    fn n_basis(&self) -> usize {
        self.n_basis()
    }

    #[inline]
    fn for_each_basis(&self, x: f64, f: impl FnMut(usize, f64)) -> Result<(), SplineError> {
        reject_non_finite(x)?;
        Self::for_each_basis(self, x, f);
        Ok(())
    }

    #[inline]
    fn evaluate_into(&self, x: f64, out: &mut [f64]) -> Result<(), SplineError> {
        reject_non_finite(x)?;
        validate_output_len(self.n_basis(), out.len())?;
        Self::evaluate_into(self, x, out);
        Ok(())
    }
}

impl SplineBasis1d for TruncatedPowerBasis {
    #[inline]
    fn n_basis(&self) -> usize {
        self.n_basis()
    }

    #[inline]
    fn for_each_basis(&self, x: f64, f: impl FnMut(usize, f64)) -> Result<(), SplineError> {
        reject_non_finite(x)?;
        Self::for_each_basis(self, x, f);
        Ok(())
    }

    #[inline]
    fn evaluate_into(&self, x: f64, out: &mut [f64]) -> Result<(), SplineError> {
        reject_non_finite(x)?;
        validate_output_len(self.n_basis(), out.len())?;
        Self::evaluate_into(self, x, out);
        Ok(())
    }
}

#[inline]
const fn reject_non_finite(x: f64) -> Result<(), SplineError> {
    if x.is_finite() {
        Ok(())
    } else {
        Err(SplineError::NonFiniteValue)
    }
}

#[inline]
fn validate_output_len(expected: usize, actual: usize) -> Result<(), SplineError> {
    if actual == expected {
        Ok(())
    } else {
        Err(ModelError::DesignSize {
            expected_values: expected,
            actual_values: actual,
        }
        .into())
    }
}
