use gamlss_core::ModelError;

use crate::local::{bspline_local_basis, prepare_cyclic_local_basis};
use crate::{
    BSplineBasis, CyclicSplineSpec, FourierBasis, ISplineBasis, MSplineBasis,
    NaturalCubicSplineBasis, OpenUniformSplineBasis, PeriodicSplineSpec, SplineError,
    TruncatedPowerBasis,
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

    /// Adds `scale * b(x) * b(x)^T` to a row-major square buffer.
    ///
    /// The default is allocation-free but may evaluate the row repeatedly.
    /// Local basis implementations override it so on-demand Gram products
    /// evaluate the row only once.
    ///
    /// # Errors
    ///
    /// Returns the same coordinate error as [`Self::for_each_basis`]. The
    /// caller must provide `self.n_basis().pow(2)` output values.
    #[doc(hidden)]
    fn add_scaled_outer(&self, x: f64, scale: f64, out: &mut [f64]) -> Result<(), SplineError> {
        add_scaled_outer_default(self, x, scale, out)
    }

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

    #[inline]
    fn add_scaled_outer(&self, x: f64, scale: f64, out: &mut [f64]) -> Result<(), SplineError> {
        T::add_scaled_outer(*self, x, scale, out)
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

    #[inline]
    fn add_scaled_outer(&self, x: f64, scale: f64, out: &mut [f64]) -> Result<(), SplineError> {
        reject_non_finite(x)?;
        if self.degree() <= 3 {
            bspline_local_basis(self.knots(), self.n_basis(), self.degree(), x).add_scaled_outer(
                scale,
                self.n_basis(),
                out,
            );
            Ok(())
        } else {
            add_scaled_outer_default(self, x, scale, out)
        }
    }
}

impl SplineBasis1d for OpenUniformSplineBasis {
    #[inline]
    fn n_basis(&self) -> usize {
        self.n_basis()
    }

    #[inline]
    fn validate_coordinate(&self, x: f64) -> Result<(), SplineError> {
        self.unit_coordinate(x).map(|_| ())
    }

    #[inline]
    fn for_each_basis(&self, x: f64, f: impl FnMut(usize, f64)) -> Result<(), SplineError> {
        self.for_each_value_basis(x, f)
    }

    #[inline]
    fn add_scaled_outer(&self, x: f64, scale: f64, out: &mut [f64]) -> Result<(), SplineError> {
        self.local_basis_for_unit(self.unit_coordinate(x)?)
            .add_scaled_outer(scale, self.n_basis(), out);
        Ok(())
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

    #[inline]
    fn add_scaled_outer(&self, phi: f64, scale: f64, out: &mut [f64]) -> Result<(), SplineError> {
        reject_non_finite(phi)?;
        prepare_cyclic_local_basis(phi, self.order(), self.n_basis()).add_scaled_outer_wrapped(
            self.order().degree() + 1,
            self.n_basis(),
            scale,
            out,
        );
        Ok(())
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

    #[inline]
    fn add_scaled_outer(&self, x: f64, scale: f64, out: &mut [f64]) -> Result<(), SplineError> {
        self.add_scaled_outer_at(x, scale, out)
    }
}

impl SplineBasis1d for FourierBasis {
    #[inline]
    fn n_basis(&self) -> usize {
        self.n_basis()
    }

    #[inline]
    fn validate_coordinate(&self, x: f64) -> Result<(), SplineError> {
        self.phase(x)
            .map(|_| ())
            .map_err(|_| SplineError::NonFiniteValue)
    }

    #[inline]
    fn for_each_basis(&self, x: f64, f: impl FnMut(usize, f64)) -> Result<(), SplineError> {
        self.for_each_value_basis(x, f)
            .map_err(|_| SplineError::NonFiniteValue)
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

    #[inline]
    fn add_scaled_outer(&self, x: f64, scale: f64, out: &mut [f64]) -> Result<(), SplineError> {
        reject_non_finite(x)?;
        self.local_basis(x)
            .add_scaled_outer(scale, self.n_basis(), out);
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

    #[inline]
    fn add_scaled_outer(&self, x: f64, scale: f64, out: &mut [f64]) -> Result<(), SplineError> {
        reject_non_finite(x)?;
        self.add_scaled_outer_at(x, scale, out);
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

    #[inline]
    fn add_scaled_outer(&self, x: f64, scale: f64, out: &mut [f64]) -> Result<(), SplineError> {
        reject_non_finite(x)?;
        self.add_scaled_outer_at(x, scale, out);
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

fn add_scaled_outer_default<B>(
    basis: &B,
    x: f64,
    scale: f64,
    out: &mut [f64],
) -> Result<(), SplineError>
where
    B: SplineBasis1d + ?Sized,
{
    debug_assert_eq!(out.len(), basis.n_basis() * basis.n_basis());
    let n_basis = basis.n_basis();
    let mut inner_error = None;
    basis.for_each_basis(x, |left_index, left_weight| {
        if inner_error.is_some() {
            return;
        }
        let scaled_left = scale * left_weight;
        if let Err(error) = basis.for_each_basis(x, |right_index, right_weight| {
            let index = left_index * n_basis + right_index;
            out[index] = scaled_left.mul_add(right_weight, out[index]);
        }) {
            inner_error = Some(error);
        }
    })?;
    inner_error.map_or(Ok(()), Err)
}
