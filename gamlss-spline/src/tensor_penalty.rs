use gamlss_core::{MatrixPenalty, ModelError, Penalty};

const EXPECTED_POSITIVE_DIMENSION: &str = "> 0";
const EXPECTED_FINITE_MATRIX: &str = "finite for the supplied dimension";

#[derive(Debug, Clone, Copy, PartialEq)]
struct MatrixEntry {
    row: usize,
    col: usize,
    value: f64,
}

/// Anisotropic quadratic penalty for a row-major tensor-product coefficient array.
///
/// If the coefficient matrix has `left_dim` rows and `right_dim` columns, and
/// the marginal penalty Hessians are $H_x$ and $H_z$, this type applies
///
/// $$
/// H = H_x\otimes I_z + I_x\otimes H_z.
/// $$
///
/// Each marginal penalty retains its own smoothing weight and order. The
/// marginal matrices are prepared once during construction; value, gradient,
/// and matrix evaluation do not allocate.
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, PartialEq)]
pub struct TensorProductPenalty<L, R> {
    left_dim: usize,
    right_dim: usize,
    dim: usize,
    left: L,
    right: R,
    left_entries: Box<[MatrixEntry]>,
    right_entries: Box<[MatrixEntry]>,
}

impl<L, R> TensorProductPenalty<L, R>
where
    L: MatrixPenalty,
    R: MatrixPenalty,
{
    /// Creates an anisotropic tensor-product penalty.
    ///
    /// # Errors
    ///
    /// Returns an error when a marginal dimension is zero, their product
    /// overflows `usize`, a marginal penalty rejects its dimension, or a
    /// marginal curvature matrix contains a non-finite value.
    pub fn try_new(
        left_dim: usize,
        right_dim: usize,
        left: L,
        right: R,
    ) -> Result<Self, ModelError> {
        validate_positive_dim("left tensor penalty dimension", left_dim)?;
        validate_positive_dim("right tensor penalty dimension", right_dim)?;
        let dim = left_dim
            .checked_mul(right_dim)
            .ok_or(ModelError::ArithmeticOverflow {
                context: "tensor penalty coefficient count",
            })?;
        let left_entries = prepare_entries(&left, left_dim)?;
        let right_entries = prepare_entries(&right, right_dim)?;
        Ok(Self {
            left_dim,
            right_dim,
            dim,
            left,
            right,
            left_entries,
            right_entries,
        })
    }

    /// Number of coefficients in the left marginal basis.
    #[must_use]
    pub const fn left_dim(&self) -> usize {
        self.left_dim
    }

    /// Number of coefficients in the right marginal basis.
    #[must_use]
    pub const fn right_dim(&self) -> usize {
        self.right_dim
    }

    /// Total number of row-major tensor coefficients.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// Left marginal penalty.
    #[must_use]
    pub const fn left_penalty(&self) -> &L {
        &self.left
    }

    /// Right marginal penalty.
    #[must_use]
    pub const fn right_penalty(&self) -> &R {
        &self.right
    }

    #[inline]
    fn for_each_entry(&self, mut f: impl FnMut(usize, usize, f64)) {
        for entry in &self.left_entries {
            for right_index in 0..self.right_dim {
                f(
                    entry.row * self.right_dim + right_index,
                    entry.col * self.right_dim + right_index,
                    entry.value,
                );
            }
        }
        for entry in &self.right_entries {
            for left_index in 0..self.left_dim {
                let offset = left_index * self.right_dim;
                f(offset + entry.row, offset + entry.col, entry.value);
            }
        }
    }
}

impl<L, R> Penalty for TensorProductPenalty<L, R>
where
    L: MatrixPenalty,
    R: MatrixPenalty,
{
    fn value(&self, beta: &[f64]) -> f64 {
        debug_assert_eq!(beta.len(), self.dim);
        let mut value = 0.0;
        self.for_each_entry(|row, col, weight| {
            value = (0.5 * weight * beta[row]).mul_add(beta[col], value);
        });
        value
    }

    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(beta.len(), self.dim);
        debug_assert_eq!(grad.len(), self.dim);
        self.for_each_entry(|row, col, weight| {
            grad[row] = weight.mul_add(beta[col], grad[row]);
        });
    }

    fn validate_dim(&self, dim: usize) -> Result<(), ModelError> {
        if dim != self.dim {
            return Err(ModelError::DesignSize {
                expected_values: self.dim,
                actual_values: dim,
            });
        }
        self.left.validate_dim(self.left_dim)?;
        self.right.validate_dim(self.right_dim)
    }
}

impl<L, R> MatrixPenalty for TensorProductPenalty<L, R>
where
    L: MatrixPenalty,
    R: MatrixPenalty,
{
    fn add_penalty_matrix(&self, dim: usize, gram: &mut [f64]) {
        debug_assert_eq!(dim, self.dim);
        debug_assert_eq!(dim.checked_mul(dim), Some(gram.len()));
        self.for_each_entry(|row, col, value| {
            let index = row * dim + col;
            gram[index] += value;
        });
    }
}

/// Orthonormal Helmert transform that removes a constant coefficient direction.
///
/// For `source_dim = K`, this transform maps a source row of length `K` to
/// `K - 1` contrasts. Its columns are orthonormal and orthogonal to the
/// constant vector. [`Self::transform_into`] evaluates the transform from
/// prefix sums in linear time, without materializing the dense matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HelmertContrast {
    source_dim: usize,
}

impl HelmertContrast {
    /// Creates a Helmert transform for a source dimension of at least two.
    ///
    /// # Errors
    ///
    /// Returns an error when `source_dim < 2`.
    pub const fn try_new(source_dim: usize) -> Result<Self, ModelError> {
        if source_dim < 2 {
            return Err(ModelError::InvalidParameter {
                parameter: "contrast source dimension",
                expected: ">= 2",
            });
        }
        Ok(Self { source_dim })
    }

    /// Dimension before applying the contrast.
    #[must_use]
    pub const fn source_dim(self) -> usize {
        self.source_dim
    }

    /// Dimension after removing the constant direction.
    #[must_use]
    pub const fn target_dim(self) -> usize {
        self.source_dim - 1
    }

    /// Applies the contrast to one source basis row.
    ///
    /// # Errors
    ///
    /// Returns an error when either slice has the wrong dimension.
    pub fn transform_into(self, source: &[f64], out: &mut [f64]) -> Result<(), ModelError> {
        validate_exact_dim(self.source_dim(), source.len())?;
        validate_exact_dim(self.target_dim(), out.len())?;

        let mut prefix = 0.0;
        for (target, value) in out.iter_mut().enumerate() {
            prefix += source[target];
            #[allow(clippy::cast_precision_loss)]
            let leading = (target + 1) as f64;
            *value =
                (-leading).mul_add(source[target + 1], prefix) / (leading * (leading + 1.0)).sqrt();
        }
        Ok(())
    }

    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn coefficient(self, source: usize, target: usize) -> f64 {
        debug_assert!(source < self.source_dim());
        debug_assert!(target < self.target_dim());
        let leading = target + 1;
        let denominator = ((leading * (leading + 1)) as f64).sqrt();
        if source <= target {
            1.0 / denominator
        } else if source == target + 1 {
            -(leading as f64) / denominator
        } else {
            0.0
        }
    }
}

/// Reparameterizes a marginal quadratic penalty into an orthonormal Helmert basis.
///
/// For a source basis with $K$ coefficients, the $K-1$ Helmert columns are
/// orthonormal and orthogonal to the constant coefficient vector. A
/// partition-of-unity B-spline margin therefore loses exactly its constant
/// function. Tensoring two constrained margins yields an interaction space
/// that cannot reproduce either marginal main effect.
///
/// Orthonormality is important: it preserves the identity factors in
/// [`TensorProductPenalty`] while the source curvature is transformed as
/// $Q^T H Q$. The corresponding design transform can be evaluated from prefix
/// sums in linear time rather than by a generic dense matrix product.
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, PartialEq)]
pub struct HelmertContrastPenalty<P> {
    contrast: HelmertContrast,
    penalty: P,
    matrix: Box<[f64]>,
}

impl<P> HelmertContrastPenalty<P>
where
    P: MatrixPenalty,
{
    /// Creates the penalty $Q^T H Q$ for an orthonormal Helmert matrix $Q$.
    ///
    /// # Errors
    ///
    /// Returns an error when `source_dim < 2`, the source penalty rejects that
    /// dimension, or its curvature matrix contains a non-finite value.
    pub fn try_new(source_dim: usize, penalty: P) -> Result<Self, ModelError> {
        let contrast = HelmertContrast::try_new(source_dim)?;
        let target_dim = contrast.target_dim();
        let source_matrix = prepare_matrix(&penalty, source_dim)?;
        let transform_len = checked_product(
            source_dim,
            target_dim,
            "Helmert transformed penalty workspace size",
        )?;
        let mut intermediate = vec![0.0; transform_len];
        for source_row in 0..source_dim {
            for source_col in 0..source_dim {
                let weight = source_matrix[source_row * source_dim + source_col];
                if weight == 0.0 {
                    continue;
                }
                for target_col in 0..target_dim {
                    let index = source_row * target_dim + target_col;
                    intermediate[index] = weight.mul_add(
                        contrast.coefficient(source_col, target_col),
                        intermediate[index],
                    );
                }
            }
        }
        let matrix_len =
            target_dim
                .checked_mul(target_dim)
                .ok_or(ModelError::ArithmeticOverflow {
                    context: "Helmert penalty matrix size",
                })?;
        let mut matrix = vec![0.0; matrix_len];
        for target_row in 0..target_dim {
            for source in 0..source_dim {
                let left = contrast.coefficient(source, target_row);
                for target_col in 0..target_dim {
                    let index = target_row * target_dim + target_col;
                    matrix[index] = left.mul_add(
                        intermediate[source * target_dim + target_col],
                        matrix[index],
                    );
                }
            }
        }
        if matrix.iter().any(|value| !value.is_finite()) {
            return Err(ModelError::InvalidParameter {
                parameter: "Helmert penalty matrix",
                expected: EXPECTED_FINITE_MATRIX,
            });
        }
        Ok(Self {
            contrast,
            penalty,
            matrix: matrix.into_boxed_slice(),
        })
    }

    /// Dimension of the original marginal basis.
    #[must_use]
    pub const fn source_dim(&self) -> usize {
        self.contrast.source_dim()
    }

    /// Dimension after removing the constant direction.
    #[must_use]
    pub const fn target_dim(&self) -> usize {
        self.contrast.target_dim()
    }

    /// Helmert transform shared with the corresponding design matrix.
    #[must_use]
    pub const fn contrast(&self) -> HelmertContrast {
        self.contrast
    }

    /// Original marginal penalty.
    #[must_use]
    pub const fn source_penalty(&self) -> &P {
        &self.penalty
    }
}

impl<P> Penalty for HelmertContrastPenalty<P>
where
    P: MatrixPenalty,
{
    fn value(&self, beta: &[f64]) -> f64 {
        let target_dim = self.target_dim();
        debug_assert_eq!(beta.len(), target_dim);
        let mut value = 0.0;
        for (row, beta_row) in beta.iter().copied().enumerate() {
            let product = self.matrix[row * target_dim..(row + 1) * target_dim]
                .iter()
                .copied()
                .zip(beta.iter().copied())
                .fold(0.0, |sum, (weight, beta)| weight.mul_add(beta, sum));
            value = (0.5 * beta_row).mul_add(product, value);
        }
        value
    }

    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        let target_dim = self.target_dim();
        debug_assert_eq!(beta.len(), target_dim);
        debug_assert_eq!(grad.len(), target_dim);
        for (gradient, row) in grad.iter_mut().zip(self.matrix.chunks_exact(target_dim)) {
            let product = row
                .iter()
                .copied()
                .zip(beta.iter().copied())
                .fold(0.0, |sum, (weight, beta)| weight.mul_add(beta, sum));
            *gradient += product;
        }
    }

    fn validate_dim(&self, dim: usize) -> Result<(), ModelError> {
        validate_exact_dim(self.target_dim(), dim)?;
        self.penalty.validate_dim(self.source_dim())
    }
}

impl<P> MatrixPenalty for HelmertContrastPenalty<P>
where
    P: MatrixPenalty,
{
    fn add_penalty_matrix(&self, dim: usize, gram: &mut [f64]) {
        debug_assert_eq!(dim, self.target_dim());
        debug_assert_eq!(dim.checked_mul(dim), Some(gram.len()));
        for (value, weight) in gram.iter_mut().zip(self.matrix.iter().copied()) {
            *value += weight;
        }
    }
}

fn prepare_entries<P>(penalty: &P, dim: usize) -> Result<Box<[MatrixEntry]>, ModelError>
where
    P: MatrixPenalty,
{
    let matrix = prepare_matrix(penalty, dim)?;
    Ok(matrix
        .into_vec()
        .into_iter()
        .enumerate()
        .filter_map(|(index, value)| {
            (value != 0.0).then_some(MatrixEntry {
                row: index / dim,
                col: index % dim,
                value,
            })
        })
        .collect::<Vec<_>>()
        .into_boxed_slice())
}

fn prepare_matrix<P>(penalty: &P, dim: usize) -> Result<Box<[f64]>, ModelError>
where
    P: MatrixPenalty,
{
    penalty.validate_dim(dim)?;
    let matrix_len = dim.checked_mul(dim).ok_or(ModelError::ArithmeticOverflow {
        context: "marginal penalty matrix size",
    })?;
    let mut matrix = vec![0.0; matrix_len];
    penalty.add_penalty_matrix(dim, &mut matrix);
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(ModelError::InvalidParameter {
            parameter: "marginal penalty matrix",
            expected: EXPECTED_FINITE_MATRIX,
        });
    }
    Ok(matrix.into_boxed_slice())
}

fn checked_product(left: usize, right: usize, context: &'static str) -> Result<usize, ModelError> {
    left.checked_mul(right)
        .ok_or(ModelError::ArithmeticOverflow { context })
}

const fn validate_exact_dim(expected: usize, actual: usize) -> Result<(), ModelError> {
    if actual == expected {
        Ok(())
    } else {
        Err(ModelError::DesignSize {
            expected_values: expected,
            actual_values: actual,
        })
    }
}

const fn validate_positive_dim(parameter: &'static str, dim: usize) -> Result<(), ModelError> {
    if dim == 0 {
        Err(ModelError::InvalidParameter {
            parameter,
            expected: EXPECTED_POSITIVE_DIMENSION,
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{MatrixPenalty, Penalty};

    use crate::{DifferencePenalty, PreparedDifferencePenalty};

    use super::{HelmertContrast, HelmertContrastPenalty, TensorProductPenalty};

    #[test]
    fn tensor_penalty_has_independent_marginal_weights() {
        let penalty = TensorProductPenalty::try_new(
            3,
            2,
            DifferencePenalty::try_new(2.0, 1).unwrap(),
            DifferencePenalty::try_new(5.0, 1).unwrap(),
        )
        .unwrap();
        let varying_left = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let varying_right = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0];

        assert_relative_eq!(penalty.value(&varying_left), 4.0, epsilon = 1.0e-12);
        assert_relative_eq!(penalty.value(&varying_right), 15.0, epsilon = 1.0e-12);
        assert_gradient_and_matrix_match(&penalty, &[-0.2, 0.1, 0.7, -0.4, 0.3, 0.9]);
    }

    #[test]
    fn helmert_contrast_matches_explicit_source_transform() {
        let source = PreparedDifferencePenalty::try_new(0.7, 2).unwrap();
        let contrast = HelmertContrastPenalty::try_new(5, source.clone()).unwrap();
        let theta = [0.2, -0.4, 0.9, 0.1];
        let mut source_beta = [0.0; 5];
        for (source_index, value) in source_beta.iter_mut().enumerate() {
            for (target_index, theta) in theta.iter().copied().enumerate() {
                *value = contrast
                    .contrast()
                    .coefficient(source_index, target_index)
                    .mul_add(theta, *value);
            }
        }

        assert_relative_eq!(
            contrast.value(&theta),
            source.value(&source_beta),
            epsilon = 1.0e-12
        );
        assert_gradient_and_matrix_match(&contrast, &theta);
    }

    #[test]
    fn helmert_transform_is_orthonormal_and_removes_constants() {
        let contrast = HelmertContrast::try_new(5).unwrap();
        let mut transformed = [f64::NAN; 4];
        contrast
            .transform_into(&[3.0, 3.0, 3.0, 3.0, 3.0], &mut transformed)
            .unwrap();
        assert!(transformed.iter().all(|value| value.abs() < 1.0e-14));

        let source = [0.2, -0.5, 0.1, 0.9, -0.3];
        contrast.transform_into(&source, &mut transformed).unwrap();
        for (target, actual) in transformed.iter().copied().enumerate() {
            let expected = source
                .iter()
                .copied()
                .enumerate()
                .map(|(source, value)| value * contrast.coefficient(source, target))
                .sum::<f64>();
            assert_relative_eq!(actual, expected, epsilon = 1.0e-14);
        }

        let mut columns = [[0.0; 5]; 4];
        for (target, column) in columns.iter_mut().enumerate() {
            for (source, value) in column.iter_mut().enumerate() {
                *value = contrast.coefficient(source, target);
            }
        }
        for (left, left_column) in columns.iter().enumerate() {
            for (right, right_column) in columns.iter().enumerate() {
                let product = left_column
                    .iter()
                    .zip(right_column)
                    .map(|(left, right)| left * right)
                    .sum::<f64>();
                assert_relative_eq!(product, f64::from(left == right), epsilon = 1.0e-14);
            }
        }
    }

    fn assert_gradient_and_matrix_match<P>(penalty: &P, beta: &[f64])
    where
        P: MatrixPenalty,
    {
        let mut gradient = vec![0.0; beta.len()];
        penalty.add_gradient(beta, &mut gradient);
        let mut matrix = vec![0.0; beta.len() * beta.len()];
        penalty.add_penalty_matrix(beta.len(), &mut matrix);
        for (row, expected) in gradient.iter().copied().enumerate() {
            let actual = matrix[row * beta.len()..(row + 1) * beta.len()]
                .iter()
                .zip(beta)
                .map(|(weight, value)| weight * value)
                .sum::<f64>();
            assert_relative_eq!(actual, expected, epsilon = 1.0e-11);
        }

        let step = 1.0e-6;
        for index in 0..beta.len() {
            let mut lower = beta.to_vec();
            let mut upper = beta.to_vec();
            lower[index] -= step;
            upper[index] += step;
            let finite_difference = (penalty.value(&upper) - penalty.value(&lower)) / (2.0 * step);
            assert_relative_eq!(finite_difference, gradient[index], epsilon = 2.0e-8);
        }
    }
}
