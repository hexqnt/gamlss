use gamlss_core::ModelError;

use crate::SplineError;

/// Common allocation-free API for a spline basis on fixed-dimensional points.
///
/// The const dimension keeps point shapes in the type system while allowing
/// basis implementations to choose sparse or dense row visitation. Emitted
/// indices must be smaller than [`Self::n_basis`] and occur at most once.
pub trait SplineBasisNd<const D: usize> {
    /// Number of basis functions.
    fn n_basis(&self) -> usize;

    /// Validates a point without retaining it.
    ///
    /// The default accepts arrays whose coordinates are all finite.
    fn validate_point(&self, point: &[f64; D]) -> Result<(), SplineError> {
        if point.iter().all(|coordinate| coordinate.is_finite()) {
            Ok(())
        } else {
            Err(SplineError::NonFiniteValue)
        }
    }

    /// Visits the non-zero basis values at one point without allocating.
    fn for_each_basis(
        &self,
        point: &[f64; D],
        f: impl FnMut(usize, f64),
    ) -> Result<(), SplineError>;

    /// Writes a dense basis row into `out`.
    fn evaluate_into(&self, point: &[f64; D], out: &mut [f64]) -> Result<(), SplineError> {
        if out.len() != self.n_basis() {
            return Err(ModelError::DesignSize {
                expected_values: self.n_basis(),
                actual_values: out.len(),
            }
            .into());
        }
        out.fill(0.0);
        self.for_each_basis(point, |index, value| out[index] = value)
    }

    /// Returns a dense basis row.
    fn evaluate(&self, point: &[f64; D]) -> Result<Vec<f64>, SplineError> {
        let mut values = vec![0.0; self.n_basis()];
        self.evaluate_into(point, &mut values)?;
        Ok(values)
    }
}

impl<T, const D: usize> SplineBasisNd<D> for &T
where
    T: SplineBasisNd<D> + ?Sized,
{
    fn n_basis(&self) -> usize {
        T::n_basis(*self)
    }

    fn validate_point(&self, point: &[f64; D]) -> Result<(), SplineError> {
        T::validate_point(*self, point)
    }

    fn for_each_basis(
        &self,
        point: &[f64; D],
        f: impl FnMut(usize, f64),
    ) -> Result<(), SplineError> {
        T::for_each_basis(*self, point, f)
    }

    fn evaluate_into(&self, point: &[f64; D], out: &mut [f64]) -> Result<(), SplineError> {
        T::evaluate_into(*self, point, out)
    }
}
