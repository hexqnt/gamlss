use gamlss_core::ModelError;

use crate::{
    BSplineBasis, CyclicSplineSpec, ISplineBasis, MSplineBasis, NaturalCubicSplineBasis,
    OpenUniformSplineBasis, PeriodicSplineSpec, SplineError, TruncatedPowerBasis,
};

/// Common one-dimensional spline basis evaluation API.
///
/// This trait is intended for feature engineering and interop code that wants
/// to evaluate a fitted basis shape without depending on GAMLSS predictor
/// blocks. It is implemented for the crate's reusable basis metadata types.
/// Wrap a basis in [`crate::OnDemandSplineDesign`] when repeated indexed model
/// passes are required without retaining per-row geometry.
///
/// Implementations must emit indices smaller than [`Self::n_basis`] at most
/// once per row. If [`Self::validate_coordinate`] succeeds, subsequent
/// [`Self::for_each_basis`] calls for the same immutable basis and coordinate
/// must also succeed.
pub trait SplineBasis1d {
    /// Number of basis functions.
    fn n_basis(&self) -> usize;

    /// Validates one input coordinate without evaluating its basis row.
    ///
    /// The default accepts every finite coordinate. Implementations whose
    /// coordinate transform can fail for a finite input must override this
    /// method. [`crate::OnDemandSplineDesign`] uses it to establish its
    /// infallible hot-path invariant without computing and discarding every
    /// row during construction.
    ///
    /// # Errors
    ///
    /// Returns [`SplineError::NonFiniteValue`] when `x` is not finite.
    #[inline]
    fn validate_coordinate(&self, x: f64) -> Result<(), SplineError> {
        reject_non_finite(x)
    }

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

impl<T> SplineBasis1d for &T
where
    T: SplineBasis1d + ?Sized,
{
    #[inline]
    fn n_basis(&self) -> usize {
        T::n_basis(*self)
    }

    #[inline]
    fn validate_coordinate(&self, x: f64) -> Result<(), SplineError> {
        T::validate_coordinate(*self, x)
    }

    #[inline]
    fn for_each_basis(&self, x: f64, f: impl FnMut(usize, f64)) -> Result<(), SplineError> {
        T::for_each_basis(*self, x, f)
    }

    #[inline]
    fn evaluate_into(&self, x: f64, out: &mut [f64]) -> Result<(), SplineError> {
        T::evaluate_into(*self, x, out)
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
        Self::for_each_basis(self, x, &mut f);
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

impl SplineBasis1d for CyclicSplineSpec {
    #[inline]
    fn n_basis(&self) -> usize {
        self.n_basis()
    }

    #[inline]
    fn for_each_basis(&self, phi: f64, f: impl FnMut(usize, f64)) -> Result<(), SplineError> {
        self.for_each_value_basis(phi, f)
    }
}

impl SplineBasis1d for PeriodicSplineSpec {
    #[inline]
    fn n_basis(&self) -> usize {
        self.n_basis()
    }

    #[inline]
    fn validate_coordinate(&self, x: f64) -> Result<(), SplineError> {
        self.phase(x).map(|_| ())
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
