use gamlss_core::{MatrixPenalty, ModelError, Penalty};

use crate::numeric::{dot, squared_norm};
use crate::penalty::{
    add_difference_penalty_gradient, difference_penalty_value, try_difference_coefficients,
    validate_difference_order_for_dim,
};

const EXPECTED_FINITE_NONNEGATIVE: &str = "finite and >= 0";
const EXPECTED_SYMMETRIC_FINITE: &str = "finite and symmetric";

/// Diagonal positive-semidefinite penalty kernel.
///
/// This specialization keeps value, gradient, and matrix operations linear in
/// the coefficient dimension. It is useful for spectral spline
/// parameterizations, where the construction has already diagonalized the
/// roughness operator.
#[derive(Debug, Clone, PartialEq)]
pub struct DiagonalPenaltyKernel {
    diagonal: Box<[f64]>,
    rank: usize,
}

impl DiagonalPenaltyKernel {
    /// Creates a diagonal kernel from finite non-negative entries.
    ///
    /// The structural rank is the number of strictly positive entries. Exact
    /// zeros are retained as null-space directions.
    ///
    /// # Errors
    ///
    /// Returns an error when the diagonal is empty or contains a negative or
    /// non-finite value.
    pub fn try_new(diagonal: Vec<f64>) -> Result<Self, ModelError> {
        if diagonal.is_empty() {
            return Err(ModelError::InvalidParameter {
                parameter: "diagonal penalty kernel",
                expected: "non-empty with finite entries >= 0",
            });
        }
        if diagonal
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(ModelError::InvalidParameter {
                parameter: "diagonal penalty kernel",
                expected: "non-empty with finite entries >= 0",
            });
        }
        let rank = diagonal.iter().filter(|value| **value > 0.0).count();
        Ok(Self {
            diagonal: diagonal.into_boxed_slice(),
            rank,
        })
    }

    /// Diagonal entries in coefficient order.
    #[must_use]
    pub fn diagonal(&self) -> &[f64] {
        &self.diagonal
    }

    /// Coefficient dimension.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.diagonal.len()
    }
}

impl PenaltyKernel for DiagonalPenaltyKernel {
    fn dim(&self) -> usize {
        self.diagonal.len()
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn bandwidth(&self) -> Option<usize> {
        Some(0)
    }

    fn quadratic_form(&self, beta: &[f64]) -> f64 {
        debug_assert_eq!(beta.len(), self.dim());
        self.diagonal
            .iter()
            .copied()
            .zip(beta.iter().copied())
            .fold(0.0, |value, (diagonal, coefficient)| {
                (diagonal * coefficient).mul_add(coefficient, value)
            })
    }

    fn add_product(&self, beta: &[f64], out: &mut [f64]) {
        self.add_scaled_product(1.0, beta, out);
    }

    fn add_scaled_product(&self, scale: f64, beta: &[f64], out: &mut [f64]) {
        debug_assert_eq!(beta.len(), self.dim());
        debug_assert_eq!(out.len(), self.dim());
        for ((output, diagonal), coefficient) in out
            .iter_mut()
            .zip(self.diagonal.iter().copied())
            .zip(beta.iter().copied())
        {
            *output = (scale * diagonal).mul_add(coefficient, *output);
        }
    }

    fn for_each_matrix_entry(&self, mut f: impl FnMut(usize, usize, f64)) {
        for (index, value) in self.diagonal.iter().copied().enumerate() {
            if value != 0.0 {
                f(index, index, value);
            }
        }
    }
}

/// Symmetric band storage used by exact local-basis roughness operators.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SymmetricBandPenaltyKernel {
    dim: usize,
    rank: usize,
    bandwidth: usize,
    upper: Box<[f64]>,
}

impl SymmetricBandPenaltyKernel {
    pub(crate) fn try_zeroed(
        dim: usize,
        rank: usize,
        bandwidth: usize,
    ) -> Result<Self, ModelError> {
        if dim == 0 || rank > dim || bandwidth >= dim {
            return Err(ModelError::InvalidParameter {
                parameter: "symmetric band penalty dimensions",
                expected: "dimension > 0, rank <= dimension, and bandwidth < dimension",
            });
        }
        let stride = bandwidth
            .checked_add(1)
            .ok_or(ModelError::ArithmeticOverflow {
                context: "symmetric band penalty stride",
            })?;
        let len = dim
            .checked_mul(stride)
            .ok_or(ModelError::ArithmeticOverflow {
                context: "symmetric band penalty storage size",
            })?;
        Ok(Self {
            dim,
            rank,
            bandwidth,
            upper: vec![0.0; len].into_boxed_slice(),
        })
    }

    pub(crate) const fn dim(&self) -> usize {
        self.dim
    }

    pub(crate) fn add_symmetric(&mut self, row: usize, col: usize, value: f64) {
        let (row, col) = if row <= col { (row, col) } else { (col, row) };
        debug_assert!(row < self.dim && col < self.dim);
        debug_assert!(col - row <= self.bandwidth);
        let index = row * (self.bandwidth + 1) + col - row;
        self.upper[index] += value;
    }

    pub(crate) fn is_finite(&self) -> bool {
        self.upper.iter().all(|value| value.is_finite())
    }

    fn for_each_upper(&self, mut f: impl FnMut(usize, usize, f64)) {
        let stride = self.bandwidth + 1;
        for row in 0..self.dim {
            let last = (row + self.bandwidth).min(self.dim - 1);
            for col in row..=last {
                let value = self.upper[row * stride + col - row];
                if value != 0.0 {
                    f(row, col, value);
                }
            }
        }
    }
}

impl PenaltyKernel for SymmetricBandPenaltyKernel {
    fn dim(&self) -> usize {
        self.dim
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn bandwidth(&self) -> Option<usize> {
        Some(self.bandwidth)
    }

    fn quadratic_form(&self, beta: &[f64]) -> f64 {
        debug_assert_eq!(beta.len(), self.dim);
        let mut value = 0.0;
        self.for_each_upper(|row, col, weight| {
            let symmetry = if row == col { 1.0 } else { 2.0 };
            value = (symmetry * weight * beta[row]).mul_add(beta[col], value);
        });
        value
    }

    fn add_product(&self, beta: &[f64], out: &mut [f64]) {
        self.add_scaled_product(1.0, beta, out);
    }

    fn add_scaled_product(&self, scale: f64, beta: &[f64], out: &mut [f64]) {
        debug_assert_eq!(beta.len(), self.dim);
        debug_assert_eq!(out.len(), self.dim);
        self.for_each_upper(|row, col, weight| {
            out[row] = (scale * weight).mul_add(beta[col], out[row]);
            if row != col {
                out[col] = (scale * weight).mul_add(beta[row], out[col]);
            }
        });
    }

    fn for_each_matrix_entry(&self, mut f: impl FnMut(usize, usize, f64)) {
        self.for_each_upper(|row, col, value| {
            f(row, col, value);
            if row != col {
                f(col, row, value);
            }
        });
    }
}

/// Kernel `Q Q^T` represented by orthonormal column vectors of `Q`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LowRankPenaltyKernel {
    dim: usize,
    columns: Box<[f64]>,
}

impl LowRankPenaltyKernel {
    pub(crate) fn try_from_columns(dim: usize, columns: Vec<Vec<f64>>) -> Result<Self, ModelError> {
        if dim == 0 {
            return Err(ModelError::InvalidParameter {
                parameter: "low-rank penalty dimension",
                expected: "> 0",
            });
        }
        if columns.len() > dim {
            return Err(ModelError::InvalidParameter {
                parameter: "low-rank penalty columns",
                expected: "at most dimension linearly independent columns",
            });
        }
        let capacity = dim
            .checked_mul(columns.len())
            .ok_or(ModelError::ArithmeticOverflow {
                context: "low-rank penalty basis size",
            })?;
        let mut orthonormal = Vec::with_capacity(capacity);
        for mut column in columns {
            validate_exact_dim(dim, column.len())?;
            if column.iter().any(|value| !value.is_finite()) {
                return Err(ModelError::InvalidParameter {
                    parameter: "low-rank penalty basis column",
                    expected: "finite",
                });
            }
            let original_norm = squared_norm(&column).sqrt();
            for _ in 0..2 {
                for existing in orthonormal.chunks_exact(dim) {
                    let projection = dot(existing, &column);
                    for (value, direction) in column.iter_mut().zip(existing) {
                        *value = (-projection).mul_add(*direction, *value);
                    }
                }
            }
            let norm = squared_norm(&column).sqrt();
            if !norm.is_finite()
                || norm <= f64::EPSILON.sqrt() * original_norm.max(f64::MIN_POSITIVE)
            {
                return Err(ModelError::InvalidParameter {
                    parameter: "low-rank penalty basis columns",
                    expected: "finite and linearly independent",
                });
            }
            for value in &mut column {
                *value /= norm;
            }
            orthonormal.extend(column);
        }
        Ok(Self {
            dim,
            columns: orthonormal.into_boxed_slice(),
        })
    }

    pub(crate) fn basis_column(&self, index: usize) -> Option<&[f64]> {
        (index < self.rank()).then(|| &self.columns[index * self.dim..(index + 1) * self.dim])
    }

    pub(crate) const fn dim(&self) -> usize {
        self.dim
    }

    pub(crate) const fn rank(&self) -> usize {
        self.columns.len() / self.dim
    }
}

impl PenaltyKernel for LowRankPenaltyKernel {
    fn dim(&self) -> usize {
        self.dim
    }

    fn rank(&self) -> usize {
        self.columns.len() / self.dim
    }

    fn bandwidth(&self) -> Option<usize> {
        None
    }

    fn quadratic_form(&self, beta: &[f64]) -> f64 {
        debug_assert_eq!(beta.len(), self.dim);
        self.columns
            .chunks_exact(self.dim)
            .map(|column| {
                let projection = dot(column, beta);
                projection * projection
            })
            .sum()
    }

    fn add_product(&self, beta: &[f64], out: &mut [f64]) {
        self.add_scaled_product(1.0, beta, out);
    }

    fn add_scaled_product(&self, scale: f64, beta: &[f64], out: &mut [f64]) {
        debug_assert_eq!(beta.len(), self.dim);
        debug_assert_eq!(out.len(), self.dim);
        for column in self.columns.chunks_exact(self.dim) {
            let projection = scale * dot(column, beta);
            for (output, direction) in out.iter_mut().zip(column) {
                *output = projection.mul_add(*direction, *output);
            }
        }
    }

    fn for_each_matrix_entry(&self, mut f: impl FnMut(usize, usize, f64)) {
        for column in self.columns.chunks_exact(self.dim) {
            for (row, left) in column.iter().copied().enumerate() {
                for (col, right) in column.iter().copied().enumerate() {
                    let value = left * right;
                    if value != 0.0 {
                        f(row, col, value);
                    }
                }
            }
        }
    }
}

/// Shared normalized non-wrapping finite-difference operator.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DifferenceOperator {
    dim: usize,
    order: usize,
    coefficients: Box<[f64]>,
    inverse_count: f64,
}

impl DifferenceOperator {
    pub(crate) fn try_new(dim: usize, order: usize) -> Result<Self, ModelError> {
        validate_difference_order_for_dim(order, dim)?;
        let coefficients = try_difference_coefficients(order)?.into_boxed_slice();
        let difference_count = dim - order;
        #[allow(clippy::cast_precision_loss)]
        let inverse_count = 1.0 / difference_count as f64;
        Ok(Self {
            dim,
            order,
            coefficients,
            inverse_count,
        })
    }

    pub(crate) const fn dim(&self) -> usize {
        self.dim
    }

    pub(crate) const fn order(&self) -> usize {
        self.order
    }

    pub(crate) const fn difference_count(&self) -> usize {
        self.dim - self.order
    }

    pub(crate) fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    pub(crate) fn quadratic_form(&self, beta: &[f64]) -> f64 {
        difference_penalty_value(1.0, &self.coefficients, beta)
    }

    pub(crate) fn add_scaled_product(&self, scale: f64, beta: &[f64], out: &mut [f64]) {
        add_difference_penalty_gradient(0.5 * scale, &self.coefficients, beta, out);
    }

    pub(crate) fn quadratic_form_by<W>(&self, beta: &[f64], mut weight_at: W) -> f64
    where
        W: FnMut(usize) -> f64,
    {
        debug_assert_eq!(beta.len(), self.dim);
        self.inverse_count
            * beta
                .windows(self.coefficients.len())
                .enumerate()
                .map(|(row, window)| {
                    let difference = dot(&self.coefficients, window);
                    weight_at(row) * difference * difference
                })
                .sum::<f64>()
    }

    pub(crate) fn add_scaled_product_by<W>(
        &self,
        scale: f64,
        beta: &[f64],
        out: &mut [f64],
        mut weight_at: W,
    ) where
        W: FnMut(usize) -> f64,
    {
        debug_assert_eq!(beta.len(), self.dim);
        debug_assert_eq!(out.len(), self.dim);
        for (start, window) in beta.windows(self.coefficients.len()).enumerate() {
            let weight = weight_at(start);
            if weight == 0.0 {
                continue;
            }
            let difference = scale * self.inverse_count * weight * dot(&self.coefficients, window);
            for (offset, coefficient) in self.coefficients.iter().copied().enumerate() {
                out[start + offset] = difference.mul_add(coefficient, out[start + offset]);
            }
        }
    }

    pub(crate) fn for_each_matrix_entry_by<W>(
        &self,
        mut weight_at: W,
        mut f: impl FnMut(usize, usize, f64),
    ) where
        W: FnMut(usize) -> f64,
    {
        for start in 0..self.difference_count() {
            let weight = weight_at(start);
            if weight == 0.0 {
                continue;
            }
            let scale = self.inverse_count * weight;
            for (left_offset, left) in self.coefficients.iter().copied().enumerate() {
                for (right_offset, right) in self.coefficients.iter().copied().enumerate() {
                    f(
                        start + left_offset,
                        start + right_offset,
                        scale * left * right,
                    );
                }
            }
        }
    }
}

/// A validated smoothing scale applied to a reusable penalty kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaledPenalty<K> {
    lambda: f64,
    kernel: K,
}

impl<K> ScaledPenalty<K> {
    /// Creates a scaled penalty with finite non-negative `lambda`.
    ///
    /// # Errors
    ///
    /// Returns an error when `lambda` is negative or non-finite.
    pub fn try_new(lambda: f64, kernel: K) -> Result<Self, ModelError> {
        validate_smoothing_scale("penalty smoothing scale", lambda)?;
        Ok(Self { lambda, kernel })
    }

    /// Smoothing scale.
    #[must_use]
    pub const fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Unscaled penalty kernel.
    #[must_use]
    pub const fn kernel(&self) -> &K {
        &self.kernel
    }

    /// Decomposes the penalty into its scale and kernel.
    #[must_use]
    pub fn into_parts(self) -> (f64, K) {
        (self.lambda, self.kernel)
    }
}

impl<K> ScaledPenalty<K>
where
    K: PenaltyKernel,
{
    /// Derivative of the penalty value with respect to `log(lambda)`.
    #[must_use]
    pub fn log_lambda_value_derivative(&self, beta: &[f64]) -> f64 {
        self.value(beta)
    }

    /// Adds the derivative of the curvature matrix with respect to
    /// `log(lambda)`.
    pub fn add_log_lambda_matrix_derivative(&self, dim: usize, out: &mut [f64]) {
        self.add_penalty_matrix(dim, out);
    }
}

impl<K> Penalty for ScaledPenalty<K>
where
    K: PenaltyKernel,
{
    fn value(&self, beta: &[f64]) -> f64 {
        debug_assert_eq!(beta.len(), self.kernel.dim());
        if self.lambda == 0.0 {
            0.0
        } else {
            self.lambda * self.kernel.quadratic_form(beta)
        }
    }

    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(beta.len(), self.kernel.dim());
        debug_assert_eq!(grad.len(), self.kernel.dim());
        if self.lambda == 0.0 {
            return;
        }
        self.kernel
            .add_scaled_product(2.0 * self.lambda, beta, grad);
    }

    fn validate_dim(&self, dim: usize) -> Result<(), ModelError> {
        validate_exact_dim(self.kernel.dim(), dim)?;
        validate_smoothing_scale("penalty smoothing scale", self.lambda)
    }
}

impl<K> MatrixPenalty for ScaledPenalty<K>
where
    K: PenaltyKernel,
{
    fn add_penalty_matrix(&self, dim: usize, gram: &mut [f64]) {
        debug_assert_eq!(dim, self.kernel.dim());
        debug_assert_eq!(dim.checked_mul(dim), Some(gram.len()));
        let scale = 2.0 * self.lambda;
        self.kernel.for_each_matrix_entry(|row, col, value| {
            gram[row * dim + col] = scale.mul_add(value, gram[row * dim + col]);
        });
    }
}

/// Delegates the standard penalty traits from a semantic wrapper to its
/// `ScaledPenalty` field.
macro_rules! delegate_scaled_penalty {
    ($type:ty, $field:ident) => {
        impl gamlss_core::Penalty for $type {
            fn value(&self, beta: &[f64]) -> f64 {
                self.$field.value(beta)
            }

            fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
                self.$field.add_gradient(beta, grad);
            }

            fn validate_dim(&self, dim: usize) -> Result<(), gamlss_core::ModelError> {
                self.$field.validate_dim(dim)
            }
        }

        impl gamlss_core::MatrixPenalty for $type {
            fn add_penalty_matrix(&self, dim: usize, gram: &mut [f64]) {
                self.$field.add_penalty_matrix(dim, gram);
            }
        }
    };
}

pub(crate) use delegate_scaled_penalty;

/// Fixed-dimension normalized finite-difference penalty kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct DifferencePenaltyKernel {
    operator: DifferenceOperator,
}

impl DifferencePenaltyKernel {
    /// Creates `D^T D / n_diff` for a coefficient dimension and difference order.
    ///
    /// # Errors
    ///
    /// Returns an error unless `0 < order < dim` and the difference
    /// coefficients can be represented.
    pub fn try_new(dim: usize, order: usize) -> Result<Self, ModelError> {
        Ok(Self {
            operator: DifferenceOperator::try_new(dim, order)?,
        })
    }

    /// Difference order.
    #[must_use]
    pub const fn order(&self) -> usize {
        self.operator.order()
    }

    /// Number of difference rows.
    #[must_use]
    pub const fn difference_count(&self) -> usize {
        self.operator.difference_count()
    }

    /// Cached finite-difference stencil.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        self.operator.coefficients()
    }
}

impl PenaltyKernel for DifferencePenaltyKernel {
    fn dim(&self) -> usize {
        self.operator.dim()
    }

    fn rank(&self) -> usize {
        self.difference_count()
    }

    fn bandwidth(&self) -> Option<usize> {
        Some(self.order())
    }

    fn quadratic_form(&self, beta: &[f64]) -> f64 {
        self.operator.quadratic_form(beta)
    }

    fn add_product(&self, beta: &[f64], out: &mut [f64]) {
        self.add_scaled_product(1.0, beta, out);
    }

    fn add_scaled_product(&self, scale: f64, beta: &[f64], out: &mut [f64]) {
        self.operator.add_scaled_product(scale, beta, out);
    }

    fn for_each_matrix_entry(&self, mut f: impl FnMut(usize, usize, f64)) {
        self.operator.for_each_matrix_entry_by(|_| 1.0, &mut f);
    }
}

/// Dense symmetric kernel for backend-independent prepared penalties.
#[derive(Debug, Clone, PartialEq)]
pub struct DensePenaltyKernel {
    dim: usize,
    rank: usize,
    matrix: Box<[f64]>,
}

impl DensePenaltyKernel {
    /// Creates a finite symmetric dense kernel with a caller-supplied rank.
    ///
    /// This constructor validates symmetry but does not perform an expensive
    /// positive-semidefinite decomposition. Constructors deriving kernels from
    /// mathematical operators should determine and pass the structural rank.
    ///
    /// # Errors
    ///
    /// Returns an error for a zero dimension, invalid rank, wrong matrix size,
    /// non-finite values, or a non-symmetric matrix.
    pub fn try_new(dim: usize, rank: usize, matrix: Vec<f64>) -> Result<Self, ModelError> {
        if dim == 0 || rank > dim {
            return Err(ModelError::InvalidParameter {
                parameter: "dense penalty kernel dimension and rank",
                expected: "dimension > 0 and rank <= dimension",
            });
        }
        let expected = dim.checked_mul(dim).ok_or(ModelError::ArithmeticOverflow {
            context: "dense penalty kernel matrix size",
        })?;
        validate_exact_dim(expected, matrix.len())?;
        for row in 0..dim {
            for col in 0..=row {
                let left = matrix[row * dim + col];
                let right = matrix[col * dim + row];
                let tolerance = 32.0 * f64::EPSILON * left.abs().max(right.abs()).max(1.0);
                if !left.is_finite() || !right.is_finite() || (left - right).abs() > tolerance {
                    return Err(ModelError::InvalidParameter {
                        parameter: "dense penalty kernel matrix",
                        expected: EXPECTED_SYMMETRIC_FINITE,
                    });
                }
            }
        }
        Ok(Self {
            dim,
            rank,
            matrix: matrix.into_boxed_slice(),
        })
    }

    /// Row-major kernel matrix.
    #[must_use]
    pub fn matrix(&self) -> &[f64] {
        &self.matrix
    }

    /// Coefficient dimension.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }
}

impl PenaltyKernel for DensePenaltyKernel {
    fn dim(&self) -> usize {
        self.dim
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn bandwidth(&self) -> Option<usize> {
        None
    }

    fn quadratic_form(&self, beta: &[f64]) -> f64 {
        debug_assert_eq!(beta.len(), self.dim);
        let mut value = 0.0;
        for (row, beta_row) in self.matrix.chunks_exact(self.dim).zip(beta.iter().copied()) {
            value = beta_row.mul_add(dot(row, beta), value);
        }
        value
    }

    fn add_product(&self, beta: &[f64], out: &mut [f64]) {
        self.add_scaled_product(1.0, beta, out);
    }

    fn add_scaled_product(&self, scale: f64, beta: &[f64], out: &mut [f64]) {
        debug_assert_eq!(beta.len(), self.dim);
        debug_assert_eq!(out.len(), self.dim);
        for (output, row) in out.iter_mut().zip(self.matrix.chunks_exact(self.dim)) {
            *output = scale.mul_add(dot(row, beta), *output);
        }
    }

    fn for_each_matrix_entry(&self, mut f: impl FnMut(usize, usize, f64)) {
        for (index, value) in self.matrix.iter().copied().enumerate() {
            if value != 0.0 {
                f(index / self.dim, index % self.dim, value);
            }
        }
    }
}

/// Unscaled constant-curvature penalty kernel.
///
/// A kernel represents `beta^T S beta` independently of a smoothing scale.
/// Keeping `S` separate from `lambda` lets smoothing-parameter optimizers
/// reuse geometry and obtain exact derivatives with respect to `log(lambda)`.
pub trait PenaltyKernel {
    /// Coefficient dimension.
    fn dim(&self) -> usize;

    /// Matrix rank known from the kernel construction.
    fn rank(&self) -> usize;

    /// Half-bandwidth when structurally banded; `None` for a general matrix.
    fn bandwidth(&self) -> Option<usize>;

    /// Evaluates `beta^T S beta`.
    fn quadratic_form(&self, beta: &[f64]) -> f64;

    /// Adds `S beta` to `out`.
    fn add_product(&self, beta: &[f64], out: &mut [f64]);

    /// Adds `scale * S beta` to `out`.
    ///
    /// The default uses matrix-entry visitation. Structured kernels should
    /// override this when their operator form is faster.
    fn add_scaled_product(&self, scale: f64, beta: &[f64], out: &mut [f64]) {
        self.for_each_matrix_entry(|row, col, value| {
            out[row] = (scale * value).mul_add(beta[col], out[row]);
        });
    }

    /// Visits additive matrix entries of `S`.
    ///
    /// Implementations may visit the same `(row, col)` more than once; each
    /// callback value is an additive contribution.
    fn for_each_matrix_entry(&self, f: impl FnMut(usize, usize, f64));

    /// Null-space dimension.
    #[must_use]
    fn nullity(&self) -> usize {
        self.dim() - self.rank()
    }
}

impl<K> PenaltyKernel for &K
where
    K: PenaltyKernel + ?Sized,
{
    fn dim(&self) -> usize {
        K::dim(*self)
    }

    fn rank(&self) -> usize {
        K::rank(*self)
    }

    fn bandwidth(&self) -> Option<usize> {
        K::bandwidth(*self)
    }

    fn quadratic_form(&self, beta: &[f64]) -> f64 {
        K::quadratic_form(*self, beta)
    }

    fn add_product(&self, beta: &[f64], out: &mut [f64]) {
        K::add_product(*self, beta, out);
    }

    fn add_scaled_product(&self, scale: f64, beta: &[f64], out: &mut [f64]) {
        K::add_scaled_product(*self, scale, beta, out);
    }

    fn for_each_matrix_entry(&self, f: impl FnMut(usize, usize, f64)) {
        K::for_each_matrix_entry(*self, f);
    }
}

pub(crate) fn validate_smoothing_scale(
    parameter: &'static str,
    lambda: f64,
) -> Result<(), ModelError> {
    if lambda.is_finite() && lambda >= 0.0 {
        Ok(())
    } else {
        Err(ModelError::InvalidParameter {
            parameter,
            expected: EXPECTED_FINITE_NONNEGATIVE,
        })
    }
}

const fn validate_exact_dim(expected: usize, actual: usize) -> Result<(), ModelError> {
    if expected == actual {
        Ok(())
    } else {
        Err(ModelError::DesignSize {
            expected_values: expected,
            actual_values: actual,
        })
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{MatrixPenalty, Penalty};

    use super::{
        DensePenaltyKernel, DiagonalPenaltyKernel, DifferencePenaltyKernel, PenaltyKernel,
        ScaledPenalty,
    };
    use crate::PreparedDifferencePenalty;

    #[test]
    fn separated_difference_kernel_matches_legacy_penalty() {
        let kernel = DifferencePenaltyKernel::try_new(6, 2).unwrap();
        assert_eq!(kernel.rank(), 4);
        assert_eq!(kernel.nullity(), 2);
        assert_eq!(kernel.bandwidth(), Some(2));
        let separated = ScaledPenalty::try_new(0.7, kernel).unwrap();
        let legacy = PreparedDifferencePenalty::try_new(0.7, 2).unwrap();
        let beta = [0.2, -0.4, 0.9, 0.1, 0.7, -0.3];
        assert_relative_eq!(
            separated.value(&beta),
            legacy.value(&beta),
            epsilon = 1.0e-14
        );

        let mut separated_gradient = [0.0; 6];
        let mut legacy_gradient = [0.0; 6];
        separated.add_gradient(&beta, &mut separated_gradient);
        legacy.add_gradient(&beta, &mut legacy_gradient);
        for (separated, legacy) in separated_gradient.into_iter().zip(legacy_gradient) {
            assert_relative_eq!(separated, legacy, epsilon = 1.0e-14);
        }

        let mut separated_matrix = [0.0; 36];
        let mut legacy_matrix = [0.0; 36];
        separated.add_penalty_matrix(6, &mut separated_matrix);
        legacy.add_penalty_matrix(6, &mut legacy_matrix);
        for (separated, legacy) in separated_matrix.into_iter().zip(legacy_matrix) {
            assert_relative_eq!(separated, legacy, epsilon = 1.0e-14);
        }
    }

    #[test]
    fn diagonal_kernel_uses_only_its_positive_spectrum() {
        let kernel = DiagonalPenaltyKernel::try_new(vec![2.0, 0.0, 3.0]).unwrap();
        assert_eq!(kernel.rank(), 2);
        assert_eq!(kernel.nullity(), 1);
        assert_eq!(kernel.bandwidth(), Some(0));
        assert_relative_eq!(kernel.quadratic_form(&[1.0, 7.0, 2.0]), 14.0);
        let mut product = [1.0; 3];
        kernel.add_product(&[1.0, 7.0, 2.0], &mut product);
        for (actual, expected) in product.into_iter().zip([3.0, 1.0, 7.0]) {
            assert_relative_eq!(actual, expected, epsilon = f64::EPSILON);
        }
        assert!(DiagonalPenaltyKernel::try_new(vec![1.0, f64::NAN]).is_err());
    }

    #[test]
    fn dense_kernel_product_matches_quadratic_form() {
        let kernel = DensePenaltyKernel::try_new(
            3,
            2,
            vec![2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0],
        )
        .unwrap();
        let beta = [0.2, -0.4, 0.7];
        let mut product = [0.0; 3];
        kernel.add_product(&beta, &mut product);
        assert_relative_eq!(kernel.quadratic_form(&beta), 2.10, epsilon = 1.0e-14);
        assert_relative_eq!(product[0], 0.8, epsilon = 1.0e-14);
        assert_relative_eq!(product[1], -1.7, epsilon = 1.0e-14);
        assert_relative_eq!(product[2], 1.8, epsilon = 1.0e-14);
        assert!(DensePenaltyKernel::try_new(2, 2, vec![1.0, 2.0, 0.0, 1.0]).is_err());
    }
}
