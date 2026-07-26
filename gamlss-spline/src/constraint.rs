use gamlss_core::{DenseDesign, ModelError};

use crate::prepared::impl_prepared_dense_design;
use crate::{DifferentiableSplineRowBasis, HelmertContrast, SplineError, SplineRowBasis};

const EXPECTED_CENTERING_WEIGHTS: &str = "finite and >= 0 with a positive finite sum";

/// Prepared coefficient-space Helmert transform of an existing spline design.
///
/// A source design with `K` columns is transformed to `K - 1` orthonormal
/// contrasts, removing its constant coefficient direction. Transformed rows
/// are prepared once, so predictor, gradient, and Gram hot paths do not
/// allocate or repeat the source basis evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct HelmertContrastDesign<B> {
    source: B,
    contrast: HelmertContrast,
    prepared: DenseDesign,
}

impl<B> HelmertContrastDesign<B>
where
    B: SplineRowBasis,
{
    /// Builds a prepared constant-free design.
    ///
    /// # Errors
    ///
    /// Returns an error when the source has fewer than two parameters or the
    /// transformed row-major storage size overflows `usize`.
    pub fn try_new(source: B) -> Result<Self, SplineError> {
        let contrast = HelmertContrast::try_new(source.nparams())?;
        let nrows = source.nrows();
        let len = checked_product(
            nrows,
            contrast.target_dim(),
            "Helmert contrast design value count",
        )?;
        let mut values = vec![0.0; len];
        let mut source_row = vec![0.0; contrast.source_dim()];
        for (row, output) in values.chunks_exact_mut(contrast.target_dim()).enumerate() {
            source_row.fill(0.0);
            source.for_each_row_basis(row, |index, value| source_row[index] = value);
            contrast.transform_into(&source_row, output)?;
        }
        Ok(Self {
            source,
            contrast,
            prepared: DenseDesign::from_row_major(nrows, contrast.target_dim(), values)?,
        })
    }

    /// Original unconstrained design.
    #[must_use]
    pub const fn source(&self) -> &B {
        &self.source
    }

    /// Applied Helmert transform.
    #[must_use]
    pub const fn contrast(&self) -> HelmertContrast {
        self.contrast
    }

    /// Prepared row-major transformed values.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        self.prepared.values()
    }
}

/// Prepared column-centered view of an existing spline design.
///
/// The transformed columns are `B_ij - mean_j`, where means are optionally
/// observation-weighted. This supplies the empirical sum-to-zero constraint
/// commonly needed to identify smooth main effects. It changes coordinates
/// within a constrained subspace rather than introducing a new unconstrained
/// spline basis.
#[derive(Debug, Clone, PartialEq)]
pub struct CenteredSplineDesign<B> {
    source: B,
    means: Box<[f64]>,
    prepared: DenseDesign,
}

impl<B> CenteredSplineDesign<B>
where
    B: SplineRowBasis,
{
    /// Centers every source column with equal observation weights.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty source design or overflowing storage.
    pub fn try_new(source: B) -> Result<Self, SplineError> {
        let nrows = source.nrows();
        if nrows == 0 {
            return Err(invalid_centering_weights().into());
        }
        #[allow(clippy::cast_precision_loss)]
        let weight_sum = nrows as f64;
        Self::try_new_by(source, weight_sum, |_| 1.0)
    }

    /// Centers every source column using non-negative observation weights.
    ///
    /// # Errors
    ///
    /// Returns an error when weight length differs from the row count, a
    /// weight is invalid, their sum is not positive and finite, or storage
    /// dimensions overflow.
    pub fn try_new_weighted(source: B, weights: &[f64]) -> Result<Self, SplineError> {
        if weights.len() != source.nrows() {
            return Err(ModelError::DesignSize {
                expected_values: source.nrows(),
                actual_values: weights.len(),
            }
            .into());
        }
        let weight_sum = weights.iter().copied().sum::<f64>();
        if !weight_sum.is_finite()
            || weight_sum <= 0.0
            || weights
                .iter()
                .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(invalid_centering_weights().into());
        }

        Self::try_new_by(source, weight_sum, |row| weights[row])
    }

    fn try_new_by<W>(source: B, weight_sum: f64, weight_at: W) -> Result<Self, SplineError>
    where
        W: Fn(usize) -> f64,
    {
        let nrows = source.nrows();
        let nparams = source.nparams();
        let mut means = vec![0.0; nparams];
        for row in 0..nrows {
            let weight = weight_at(row);
            if weight != 0.0 {
                source.for_each_row_basis(row, |index, value| {
                    means[index] = weight.mul_add(value, means[index]);
                });
            }
        }
        for mean in &mut means {
            *mean /= weight_sum;
        }

        let len = checked_product(nrows, nparams, "centered spline design value count")?;
        let mut values = vec![0.0; len];
        for (row, output) in values.chunks_exact_mut(nparams).enumerate() {
            for (value, mean) in output.iter_mut().zip(means.iter().copied()) {
                *value = -mean;
            }
            source.for_each_row_basis(row, |index, value| output[index] += value);
        }
        Ok(Self {
            source,
            means: means.into_boxed_slice(),
            prepared: DenseDesign::from_row_major(nrows, nparams, values)?,
        })
    }

    /// Original uncentered design.
    #[must_use]
    pub const fn source(&self) -> &B {
        &self.source
    }

    /// Column means removed from the source design.
    #[must_use]
    pub fn means(&self) -> &[f64] {
        &self.means
    }

    /// Prepared row-major centered values.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        self.prepared.values()
    }
}

impl_prepared_dense_design!([B] HelmertContrastDesign<B>);
impl_prepared_dense_design!([B] CenteredSplineDesign<B>);

impl<B> DifferentiableSplineRowBasis for HelmertContrastDesign<B>
where
    B: DifferentiableSplineRowBasis,
{
    fn max_derivative_order(&self) -> Option<usize> {
        self.source.max_derivative_order()
    }

    fn for_each_row_basis_derivative(
        &self,
        row: usize,
        derivative_order: usize,
        mut f: impl FnMut(usize, f64),
    ) -> Result<(), SplineError> {
        for target in 0..self.contrast.target_dim() {
            let mut value = 0.0;
            self.source.for_each_row_basis_derivative(
                row,
                derivative_order,
                |source, weight| {
                    value = self
                        .contrast
                        .coefficient(source, target)
                        .mul_add(weight, value);
                },
            )?;
            if value != 0.0 {
                f(target, value);
            }
        }
        Ok(())
    }
}

impl<B> DifferentiableSplineRowBasis for CenteredSplineDesign<B>
where
    B: DifferentiableSplineRowBasis,
{
    fn max_derivative_order(&self) -> Option<usize> {
        self.source.max_derivative_order()
    }

    fn for_each_row_basis_derivative(
        &self,
        row: usize,
        derivative_order: usize,
        f: impl FnMut(usize, f64),
    ) -> Result<(), SplineError> {
        if derivative_order == 0 {
            self.for_each_row_basis(row, f);
            Ok(())
        } else {
            self.source
                .for_each_row_basis_derivative(row, derivative_order, f)
        }
    }
}

fn checked_product(left: usize, right: usize, context: &'static str) -> Result<usize, SplineError> {
    left.checked_mul(right)
        .ok_or_else(|| ModelError::ArithmeticOverflow { context }.into())
}

const fn invalid_centering_weights() -> ModelError {
    ModelError::InvalidParameter {
        parameter: "spline centering weights",
        expected: EXPECTED_CENTERING_WEIGHTS,
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::{CenteredSplineDesign, HelmertContrastDesign};
    use crate::{OpenUniformSplineBasis, SplineOrder, SplineRowBasis, SplineRowBasisExt};

    #[test]
    fn helmert_design_removes_constant_source_direction() {
        let x = [0.0, 0.2, 0.5, 0.8, 1.0];
        let source = OpenUniformSplineBasis::from_data(&x, 6, SplineOrder::Cubic)
            .unwrap()
            .design(&x)
            .unwrap();
        let source_rows = source.to_row_major_values().unwrap();
        let contrast = HelmertContrastDesign::try_new(source).unwrap();
        assert_eq!(contrast.nparams(), 5);
        let (source_rows, source_remainder) = source_rows.as_chunks::<6>();
        let (transformed_rows, transformed_remainder) = contrast.values().as_chunks::<5>();
        assert!(source_remainder.is_empty());
        assert!(transformed_remainder.is_empty());
        for (source, transformed) in source_rows.iter().zip(transformed_rows) {
            let mut expected = [0.0; 5];
            contrast
                .contrast()
                .transform_into(source, &mut expected)
                .unwrap();
            for (transformed, expected) in transformed.iter().copied().zip(expected) {
                assert_relative_eq!(transformed, expected, epsilon = 1.0e-14);
            }
        }
    }

    #[test]
    fn weighted_centered_design_has_zero_weighted_column_means() {
        let x = [0.0, 0.2, 0.5, 0.8, 1.0];
        let weights = [0.5, 2.0, 0.0, 1.5, 1.0];
        let source = OpenUniformSplineBasis::from_data(&x, 6, SplineOrder::Cubic)
            .unwrap()
            .design(&x)
            .unwrap();
        let centered = CenteredSplineDesign::try_new_weighted(source, &weights).unwrap();
        for column in 0..centered.nparams() {
            let sum = centered
                .values()
                .chunks_exact(centered.nparams())
                .zip(weights)
                .map(|(row, weight)| row[column] * weight)
                .sum::<f64>();
            assert_relative_eq!(sum, 0.0, epsilon = 1.0e-14);
        }
    }
}
