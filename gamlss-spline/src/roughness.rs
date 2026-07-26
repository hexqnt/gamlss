use gamlss_core::{MatrixPenalty, ModelError, Penalty};

use crate::derivative::bspline_derivative_value;
use crate::kernel::{LowRankPenaltyKernel, SymmetricBandPenaltyKernel, validate_smoothing_scale};
use crate::{
    BSplineBasis, DensePenaltyKernel, DiagonalPenaltyKernel, FourierBasis, NaturalCubicSplineBasis,
    ScaledPenalty,
};

const EXPECTED_DERIVATIVE_ORDER: &str = "> 0 and <= spline degree";
const EXPECTED_POSITIVE_DERIVATIVE_ORDER: &str = "> 0";
const EXPECTED_FINITE_ROUGHNESS: &str = "finite for the supplied basis";

macro_rules! delegate_scaled_penalty {
    ($type:ty) => {
        impl Penalty for $type {
            fn value(&self, beta: &[f64]) -> f64 {
                self.scaled.value(beta)
            }

            fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
                self.scaled.add_gradient(beta, grad);
            }

            fn validate_dim(&self, dim: usize) -> Result<(), ModelError> {
                self.scaled.validate_dim(dim)
            }
        }

        impl MatrixPenalty for $type {
            fn add_penalty_matrix(&self, dim: usize, gram: &mut [f64]) {
                self.scaled.add_penalty_matrix(dim, gram);
            }
        }
    };
}

/// Exact integrated squared-derivative penalty for an arbitrary-knot B-spline basis.
///
/// For derivative order $m$, this type represents
///
/// $$
/// J(\boldsymbol\beta)=\lambda\int_{t_p}^{t_n}
/// \left[\frac{d^m}{dx^m}\sum_i\beta_iB_{i,p}(x)\right]^2dx,
/// $$
///
/// where $p$ is the spline degree and $n$ is the basis count. Construction
/// uses Gauss--Legendre quadrature of sufficient order to integrate every
/// piecewise-polynomial product exactly. Only the symmetric band is retained,
/// so hot-path evaluation is allocation-free and linear in the basis count for
/// fixed degree.
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, PartialEq)]
pub struct BSplineDerivativePenalty {
    scaled: ScaledPenalty<SymmetricBandPenaltyKernel>,
    derivative_order: usize,
    degree: usize,
    null_space: LowRankPenaltyKernel,
}

impl BSplineDerivativePenalty {
    /// Builds an exact derivative roughness penalty for `basis`.
    ///
    /// # Errors
    ///
    /// Returns an error unless `lambda` is finite and non-negative,
    /// `derivative_order` is in `1..=basis.degree()`, and the conventional
    /// B-spline domain and resulting roughness matrix are finite and
    /// non-degenerate.
    pub fn try_new(
        basis: &BSplineBasis,
        lambda: f64,
        derivative_order: usize,
    ) -> Result<Self, ModelError> {
        validate_smoothing_scale("B-spline roughness lambda", lambda)?;
        let degree = basis.degree();
        if derivative_order == 0 || derivative_order > degree {
            return Err(ModelError::InvalidParameter {
                parameter: "B-spline derivative order",
                expected: EXPECTED_DERIVATIVE_ORDER,
            });
        }

        let n_basis = basis.n_basis();
        let knots = basis.knots();
        let lower = knots[degree];
        let upper = knots[n_basis];
        if !lower.is_finite() || !upper.is_finite() || lower >= upper {
            return Err(ModelError::InvalidParameter {
                parameter: "B-spline integration domain",
                expected: "finite with lower < upper",
            });
        }

        let mut kernel =
            SymmetricBandPenaltyKernel::try_zeroed(n_basis, n_basis - derivative_order, degree)?;
        let polynomial_degree = degree - derivative_order;
        let quadrature_order =
            polynomial_degree
                .checked_add(1)
                .ok_or(ModelError::ArithmeticOverflow {
                    context: "B-spline roughness quadrature order",
                })?;
        let quadrature = gauss_legendre(quadrature_order)?;
        let mut active = Vec::with_capacity(degree + 1);

        for interval in degree..n_basis {
            let left = knots[interval];
            let right = knots[interval + 1];
            if right <= left {
                continue;
            }
            let midpoint = left.midpoint(right);
            let half_width = 0.5 * (right - left);
            let first = interval.saturating_sub(degree);
            let last = interval.min(n_basis - 1);
            for &(node, weight) in &quadrature {
                let x = half_width.mul_add(node, midpoint);
                active.clear();
                for index in first..=last {
                    let value = bspline_derivative_value(
                        knots,
                        n_basis,
                        index,
                        degree,
                        derivative_order,
                        x,
                    );
                    if value != 0.0 {
                        active.push((index, value));
                    }
                }
                let scale = half_width * weight;
                for (position, &(row, row_value)) in active.iter().enumerate() {
                    for &(col, col_value) in &active[position..] {
                        kernel.add_symmetric(row, col, scale * row_value * col_value);
                    }
                }
            }
        }
        if !kernel.is_finite() {
            return Err(non_finite_roughness("B-spline derivative roughness"));
        }

        let null_space = bspline_null_space(basis, derivative_order)?;
        Ok(Self {
            scaled: ScaledPenalty::try_new(lambda, kernel)?,
            derivative_order,
            degree,
            null_space,
        })
    }

    /// Penalty weight.
    #[must_use]
    pub const fn lambda(&self) -> f64 {
        self.scaled.lambda()
    }

    /// Penalized derivative order.
    #[must_use]
    pub const fn derivative_order(&self) -> usize {
        self.derivative_order
    }

    /// Degree of the source B-spline basis.
    #[must_use]
    pub const fn degree(&self) -> usize {
        self.degree
    }

    /// Number of coefficients expected by this penalty.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.scaled.kernel().dim()
    }

    /// Dimension of the derivative operator's polynomial null space.
    #[must_use]
    pub const fn nullity(&self) -> usize {
        self.null_space.rank()
    }

    /// Returns a shrinkage penalty on this roughness penalty's null space.
    pub fn null_space_penalty(&self, lambda: f64) -> Result<NullSpacePenalty, ModelError> {
        NullSpacePenalty::from_kernel(lambda, self.null_space.clone())
    }
}

delegate_scaled_penalty!(BSplineDerivativePenalty);

/// Exact integrated squared-curvature penalty for a natural cubic cardinal basis.
///
/// This is $\lambda\int(f''(x))^2dx$ over the fitted knot range. Its null
/// space is the two-dimensional space of affine functions. The dense kernel is
/// prepared from the basis' natural-spline second-derivative operator once.
#[derive(Debug, Clone, PartialEq)]
pub struct NaturalCubicRoughnessPenalty {
    scaled: ScaledPenalty<DensePenaltyKernel>,
    null_space: LowRankPenaltyKernel,
}

impl NaturalCubicRoughnessPenalty {
    /// Builds the exact natural-cubic roughness penalty.
    pub fn try_new(basis: &NaturalCubicSplineBasis, lambda: f64) -> Result<Self, ModelError> {
        validate_smoothing_scale("natural cubic roughness lambda", lambda)?;
        let dim = basis.n_basis();
        let matrix_len = checked_square(dim, "natural cubic roughness matrix size")?;
        let mut kernel = vec![0.0; matrix_len];
        for interval in 0..dim - 1 {
            let h = basis.knots()[interval + 1] - basis.knots()[interval];
            let left = basis.second_derivative_column(interval);
            let right = basis.second_derivative_column(interval + 1);
            let diagonal_scale = h / 3.0;
            let cross_scale = h / 6.0;
            for row in 0..dim {
                let output = &mut kernel[row * dim..(row + 1) * dim];
                for col in 0..dim {
                    let same = right[row].mul_add(right[col], left[row] * left[col]);
                    let cross = right[row].mul_add(left[col], left[row] * right[col]);
                    output[col] =
                        diagonal_scale.mul_add(same, cross_scale.mul_add(cross, output[col]));
                }
            }
        }
        if kernel.iter().any(|value| !value.is_finite()) {
            return Err(non_finite_roughness("natural cubic roughness"));
        }
        let null_space = affine_null_space_kernel(basis.knots())?;
        Ok(Self {
            scaled: ScaledPenalty::try_new(
                lambda,
                DensePenaltyKernel::try_new(dim, dim - 2, kernel)?,
            )?,
            null_space,
        })
    }

    /// Penalty weight.
    #[must_use]
    pub const fn lambda(&self) -> f64 {
        self.scaled.lambda()
    }

    /// Number of natural-spline coefficients.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.scaled.kernel().dim()
    }

    /// Dimension of the affine null space (always two).
    #[must_use]
    pub const fn nullity(&self) -> usize {
        self.null_space.rank()
    }

    /// Returns a shrinkage penalty on the affine null space.
    pub fn null_space_penalty(&self, lambda: f64) -> Result<NullSpacePenalty, ModelError> {
        NullSpacePenalty::from_kernel(lambda, self.null_space.clone())
    }
}

delegate_scaled_penalty!(NaturalCubicRoughnessPenalty);

/// Exact periodic derivative roughness for a Fourier basis.
///
/// Integrating over one full period makes the penalty diagonal. Harmonic $k$
/// receives kernel weight $P(k2\pi/P)^{2m}/2$ for derivative order $m$; an
/// optional intercept is unpenalized.
#[derive(Debug, Clone, PartialEq)]
pub struct FourierRoughnessPenalty {
    scaled: ScaledPenalty<DiagonalPenaltyKernel>,
    period: f64,
    derivative_order: usize,
    null_space: LowRankPenaltyKernel,
}

impl FourierRoughnessPenalty {
    /// Builds an exact Fourier derivative penalty over one period.
    pub fn try_new(
        basis: &FourierBasis,
        lambda: f64,
        derivative_order: usize,
    ) -> Result<Self, ModelError> {
        validate_smoothing_scale("Fourier roughness lambda", lambda)?;
        if derivative_order == 0 {
            return Err(ModelError::InvalidParameter {
                parameter: "Fourier derivative order",
                expected: EXPECTED_POSITIVE_DERIVATIVE_ORDER,
            });
        }
        let exponent = derivative_order
            .checked_mul(2)
            .ok_or(ModelError::ArithmeticOverflow {
                context: "Fourier roughness derivative exponent",
            })?;
        let mut diagonal = vec![0.0; basis.n_basis()];
        let offset = usize::from(basis.include_intercept());
        let omega = std::f64::consts::TAU / basis.period();
        for harmonic in 1..=basis.order() {
            #[allow(clippy::cast_precision_loss)]
            let frequency = harmonic as f64 * omega;
            let weight =
                0.5 * basis.period() * nonnegative_integer_power(frequency.abs(), exponent);
            if !weight.is_finite() {
                return Err(non_finite_roughness("Fourier derivative roughness"));
            }
            let index = offset + 2 * (harmonic - 1);
            diagonal[index] = weight;
            diagonal[index + 1] = weight;
        }
        let columns = if basis.include_intercept() {
            let mut intercept = vec![0.0; basis.n_basis()];
            intercept[0] = 1.0;
            vec![intercept]
        } else {
            Vec::new()
        };
        let null_space = LowRankPenaltyKernel::try_from_columns(basis.n_basis(), columns)?;
        Ok(Self {
            scaled: ScaledPenalty::try_new(lambda, DiagonalPenaltyKernel::try_new(diagonal)?)?,
            period: basis.period(),
            derivative_order,
            null_space,
        })
    }

    /// Penalty weight.
    #[must_use]
    pub const fn lambda(&self) -> f64 {
        self.scaled.lambda()
    }

    /// Fourier period over which roughness is integrated.
    #[must_use]
    pub const fn period(&self) -> f64 {
        self.period
    }

    /// Penalized derivative order.
    #[must_use]
    pub const fn derivative_order(&self) -> usize {
        self.derivative_order
    }

    /// Number of Fourier coefficients.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.scaled.kernel().dim()
    }

    /// Null-space dimension: one with an intercept and zero otherwise.
    #[must_use]
    pub const fn nullity(&self) -> usize {
        self.null_space.rank()
    }

    /// Returns a shrinkage penalty on the optional intercept direction.
    pub fn null_space_penalty(&self, lambda: f64) -> Result<NullSpacePenalty, ModelError> {
        NullSpacePenalty::from_kernel(lambda, self.null_space.clone())
    }
}

delegate_scaled_penalty!(FourierRoughnessPenalty);

/// Low-rank shrinkage penalty for a supplied coefficient-space null space.
///
/// Constructor columns are orthonormalized once. If $Q$ contains the resulting
/// columns, the penalty is $\lambda\lVert Q^T\beta\rVert^2$. This augments a
/// singular roughness penalty without applying ridge shrinkage to its already
/// penalized range.
#[derive(Debug, Clone, PartialEq)]
pub struct NullSpacePenalty {
    scaled: ScaledPenalty<LowRankPenaltyKernel>,
}

impl NullSpacePenalty {
    /// Creates a null-space penalty from column vectors and orthonormalizes them.
    ///
    /// Empty `columns` are accepted and represent the zero-dimensional null
    /// space. Non-empty columns must be finite, have length `dim`, and be
    /// linearly independent.
    pub fn try_from_columns(
        lambda: f64,
        dim: usize,
        columns: Vec<Vec<f64>>,
    ) -> Result<Self, ModelError> {
        Self::from_kernel(
            lambda,
            LowRankPenaltyKernel::try_from_columns(dim, columns)?,
        )
    }

    /// Creates shrinkage on the constant coefficient direction.
    pub fn try_constant(lambda: f64, dim: usize) -> Result<Self, ModelError> {
        Self::try_from_columns(lambda, dim, vec![vec![1.0; dim]])
    }

    /// Creates shrinkage on affine coefficient values over `coordinates`.
    pub fn try_affine(lambda: f64, coordinates: &[f64]) -> Result<Self, ModelError> {
        Self::from_kernel(lambda, affine_null_space_kernel(coordinates)?)
    }

    /// Returns a copy with a different validated penalty weight.
    pub fn with_lambda(&self, lambda: f64) -> Result<Self, ModelError> {
        Self::from_kernel(lambda, self.scaled.kernel().clone())
    }

    /// Penalty weight.
    #[must_use]
    pub const fn lambda(&self) -> f64 {
        self.scaled.lambda()
    }

    /// Coefficient dimension.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.scaled.kernel().dim()
    }

    /// Number of orthonormal null-space directions.
    #[must_use]
    pub const fn nullity(&self) -> usize {
        self.scaled.kernel().rank()
    }

    /// Returns one orthonormal null-space direction.
    #[must_use]
    pub fn basis_column(&self, index: usize) -> Option<&[f64]> {
        self.scaled.kernel().basis_column(index)
    }

    fn from_kernel(lambda: f64, kernel: LowRankPenaltyKernel) -> Result<Self, ModelError> {
        validate_smoothing_scale("null-space penalty lambda", lambda)?;
        Ok(Self {
            scaled: ScaledPenalty::try_new(lambda, kernel)?,
        })
    }
}

delegate_scaled_penalty!(NullSpacePenalty);

fn bspline_null_space(
    basis: &BSplineBasis,
    derivative_order: usize,
) -> Result<LowRankPenaltyKernel, ModelError> {
    let degree = basis.degree();
    let n_basis = basis.n_basis();
    let knots = basis.knots();
    let lower = knots[degree];
    let span = knots[n_basis] - lower;
    let mut columns = Vec::with_capacity(derivative_order);
    let mut elementary = vec![0.0; derivative_order];
    for polynomial_degree in 0..derivative_order {
        let divisor = binomial_f64(degree, polynomial_degree);
        let mut column = Vec::with_capacity(n_basis);
        for index in 0..n_basis {
            elementary[..=polynomial_degree].fill(0.0);
            elementary[0] = 1.0;
            for knot in &knots[index + 1..=index + degree] {
                let normalized = (*knot - lower) / span;
                for order in (1..=polynomial_degree).rev() {
                    elementary[order] =
                        normalized.mul_add(elementary[order - 1], elementary[order]);
                }
            }
            column.push(elementary[polynomial_degree] / divisor);
        }
        columns.push(column);
    }
    LowRankPenaltyKernel::try_from_columns(n_basis, columns)
}

fn affine_null_space_kernel(coordinates: &[f64]) -> Result<LowRankPenaltyKernel, ModelError> {
    if coordinates.len() < 2
        || coordinates.iter().any(|value| !value.is_finite())
        || coordinates[0] >= coordinates[coordinates.len() - 1]
    {
        return Err(ModelError::InvalidParameter {
            parameter: "affine null-space coordinates",
            expected: "at least two finite values with first < last",
        });
    }
    let lower = coordinates[0];
    let span = coordinates[coordinates.len() - 1] - lower;
    let linear = coordinates
        .iter()
        .map(|value| (value - lower) / span)
        .collect();
    LowRankPenaltyKernel::try_from_columns(
        coordinates.len(),
        vec![vec![1.0; coordinates.len()], linear],
    )
}

#[allow(clippy::cast_precision_loss)]
fn binomial_f64(n: usize, k: usize) -> f64 {
    let k = k.min(n - k);
    (0..k).fold(1.0, |value, index| {
        value * (n - index) as f64 / (index + 1) as f64
    })
}

fn gauss_legendre(order: usize) -> Result<Vec<(f64, f64)>, ModelError> {
    debug_assert!(order > 0);
    let mut nodes = vec![(0.0, 0.0); order];
    let half = order.div_ceil(2);
    #[allow(clippy::cast_precision_loss)]
    let order_f64 = order as f64;
    for index in 0..half {
        #[allow(clippy::cast_precision_loss)]
        let index_f64 = index as f64;
        let mut root = (std::f64::consts::PI * (index_f64 + 0.75) / (order_f64 + 0.5)).cos();
        let mut converged = false;
        for _ in 0..64 {
            let (polynomial, previous) = legendre_pair(order, root);
            let derivative = order_f64 * (root * polynomial - previous) / (root * root - 1.0);
            let next = root - polynomial / derivative;
            if (next - root).abs() <= 8.0 * f64::EPSILON * next.abs().max(1.0) {
                root = next;
                converged = true;
                break;
            }
            root = next;
        }
        let (polynomial, previous) = legendre_pair(order, root);
        let derivative = order_f64 * (root * polynomial - previous) / (root * root - 1.0);
        let weight = 2.0 / ((1.0 - root * root) * derivative * derivative);
        if !converged || !root.is_finite() || !weight.is_finite() || weight <= 0.0 {
            return Err(non_finite_roughness("Gauss-Legendre quadrature"));
        }
        nodes[index] = (-root, weight);
        nodes[order - 1 - index] = (root, weight);
    }
    Ok(nodes)
}

#[allow(clippy::cast_precision_loss)]
fn legendre_pair(order: usize, x: f64) -> (f64, f64) {
    let mut previous = 1.0;
    if order == 0 {
        return (previous, 0.0);
    }
    let mut current = x;
    for degree in 2..=order {
        let degree_f64 = degree as f64;
        let next = ((degree - 1) as f64)
            .mul_add(-previous, ((2 * degree - 1) as f64) * x * current)
            / degree_f64;
        previous = current;
        current = next;
    }
    (current, previous)
}

fn nonnegative_integer_power(mut base: f64, mut exponent: usize) -> f64 {
    let mut value = 1.0;
    while exponent > 0 {
        if exponent & 1 == 1 {
            value *= base;
        }
        exponent >>= 1;
        if exponent > 0 {
            base *= base;
        }
    }
    value
}

fn checked_square(dim: usize, context: &'static str) -> Result<usize, ModelError> {
    dim.checked_mul(dim)
        .ok_or(ModelError::ArithmeticOverflow { context })
}

const fn non_finite_roughness(parameter: &'static str) -> ModelError {
    ModelError::InvalidParameter {
        parameter,
        expected: EXPECTED_FINITE_ROUGHNESS,
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{MatrixPenalty, Penalty};

    use crate::{BSplineBasis, FourierBasis, NaturalCubicSplineBasis};

    use super::{
        BSplineDerivativePenalty, FourierRoughnessPenalty, NaturalCubicRoughnessPenalty,
        NullSpacePenalty, binomial_f64,
    };

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn bspline_derivative_penalty_integrates_polynomials_exactly() {
        let basis =
            BSplineBasis::new(3, vec![0.0, 0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 1.0, 1.0, 1.0]).unwrap();
        let knots = basis.knots();
        let degree = basis.degree();
        let linear = (0..basis.n_basis())
            .map(|index| knots[index + 1..=index + degree].iter().sum::<f64>() / degree as f64)
            .collect::<Vec<_>>();
        let quadratic = (0..basis.n_basis())
            .map(|index| {
                let local = &knots[index + 1..=index + degree];
                let mut pairs = 0.0;
                for left in 0..local.len() {
                    for right in left + 1..local.len() {
                        pairs = local[left].mul_add(local[right], pairs);
                    }
                }
                pairs / binomial_f64(degree, 2)
            })
            .collect::<Vec<_>>();
        let first = BSplineDerivativePenalty::try_new(&basis, 0.7, 1).unwrap();
        let second = BSplineDerivativePenalty::try_new(&basis, 0.7, 2).unwrap();

        assert_relative_eq!(first.value(&linear), 0.7, epsilon = 2.0e-12);
        assert_relative_eq!(second.value(&linear), 0.0, epsilon = 2.0e-12);
        assert_relative_eq!(second.value(&quadratic), 2.8, epsilon = 2.0e-11);
        assert_eq!(second.nullity(), 2);
        assert_gradient_and_matrix_match(&second, &[0.2, -0.4, 0.7, 0.1, 0.8, -0.2]);
    }

    #[test]
    fn natural_cubic_penalty_has_affine_null_space() {
        let basis = NaturalCubicSplineBasis::new(vec![-1.0, -0.2, 0.4, 1.5, 2.0]).unwrap();
        let penalty = NaturalCubicRoughnessPenalty::try_new(&basis, 1.3).unwrap();
        let affine = basis
            .knots()
            .iter()
            .map(|value| 2.0 - 0.7 * value)
            .collect::<Vec<_>>();
        assert_relative_eq!(penalty.value(&affine), 0.0, epsilon = 2.0e-12);
        assert!(penalty.value(&[0.0, 1.0, -0.5, 0.7, 0.1]) > 0.0);
        assert_eq!(penalty.nullity(), 2);
        assert_gradient_and_matrix_match(&penalty, &[0.2, -0.4, 0.7, 0.1, -0.2]);
    }

    #[test]
    fn fourier_penalty_matches_closed_form_integral() {
        let basis = FourierBasis::new(4.0, 2, true).unwrap();
        let penalty = FourierRoughnessPenalty::try_new(&basis, 0.6, 1).unwrap();
        let beta = [9.0, 2.0, 0.5, -0.3, 0.8];
        let omega = std::f64::consts::TAU / 4.0;
        let first = omega * omega * 2.0_f64.mul_add(2.0, 0.5 * 0.5);
        let second = (2.0 * omega)
            .powi(2)
            .mul_add((-0.3_f64).mul_add(-0.3, 0.8 * 0.8), first);
        let expected = 0.6 * 2.0 * second;
        assert_relative_eq!(penalty.value(&beta), expected, epsilon = 1.0e-12);
        assert_eq!(penalty.nullity(), 1);
        assert_gradient_and_matrix_match(&penalty, &beta);
    }

    #[test]
    fn null_space_penalty_projects_only_onto_supplied_space() {
        let penalty = NullSpacePenalty::try_from_columns(
            2.0,
            3,
            vec![vec![1.0, 1.0, 1.0], vec![-1.0, 0.0, 1.0]],
        )
        .unwrap();
        assert_eq!(penalty.nullity(), 2);
        assert_relative_eq!(penalty.value(&[1.0, -2.0, 1.0]), 0.0, epsilon = 1.0e-12);
        assert!(penalty.value(&[1.0, 1.0, 1.0]) > 0.0);
        assert_gradient_and_matrix_match(&penalty, &[0.2, -0.4, 0.7]);

        let zero = penalty.with_lambda(0.0).unwrap();
        assert_relative_eq!(
            zero.value(&[f64::NAN, f64::INFINITY, 0.0]),
            0.0,
            epsilon = f64::EPSILON
        );
    }

    #[test]
    fn roughness_constructors_reject_invalid_parameters() {
        let cubic = BSplineBasis::open_uniform_from_data(&[0.0, 0.5, 1.0], 6, 3).unwrap();
        assert!(BSplineDerivativePenalty::try_new(&cubic, f64::NAN, 2).is_err());
        assert!(BSplineDerivativePenalty::try_new(&cubic, 1.0, 0).is_err());
        assert!(BSplineDerivativePenalty::try_new(&cubic, 1.0, 4).is_err());

        let degenerate = BSplineBasis::new(1, vec![0.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(BSplineDerivativePenalty::try_new(&degenerate, 1.0, 1).is_err());

        let natural = NaturalCubicSplineBasis::new(vec![0.0, 0.5, 1.0]).unwrap();
        assert!(NaturalCubicRoughnessPenalty::try_new(&natural, -1.0).is_err());

        let fourier = FourierBasis::new(1.0, 2, false).unwrap();
        assert!(FourierRoughnessPenalty::try_new(&fourier, 1.0, 0).is_err());
        assert_eq!(
            FourierRoughnessPenalty::try_new(&fourier, 1.0, 1)
                .unwrap()
                .nullity(),
            0
        );

        assert!(
            NullSpacePenalty::try_from_columns(1.0, 3, vec![vec![1.0; 3], vec![2.0; 3]]).is_err()
        );
        assert!(
            NullSpacePenalty::try_from_columns(1.0, 3, vec![vec![0.0, f64::NAN, 1.0]]).is_err()
        );
        assert!(NullSpacePenalty::try_from_columns(1.0, 3, vec![vec![1.0; 2]]).is_err());
    }

    #[test]
    fn zero_weight_exact_roughness_penalties_are_noops() {
        let cubic = BSplineBasis::open_uniform_from_data(&[0.0, 0.5, 1.0], 6, 3).unwrap();
        let bspline = BSplineDerivativePenalty::try_new(&cubic, 0.0, 2).unwrap();
        let natural_basis = NaturalCubicSplineBasis::new(vec![0.0, 0.5, 1.0]).unwrap();
        let natural = NaturalCubicRoughnessPenalty::try_new(&natural_basis, 0.0).unwrap();
        let fourier_basis = FourierBasis::new(1.0, 2, true).unwrap();
        let fourier = FourierRoughnessPenalty::try_new(&fourier_basis, 0.0, 1).unwrap();

        assert_zero_penalty_is_noop(&bspline, cubic.n_basis());
        assert_zero_penalty_is_noop(&natural, natural_basis.n_basis());
        assert_zero_penalty_is_noop(&fourier, fourier_basis.n_basis());
    }

    fn assert_zero_penalty_is_noop<P>(penalty: &P, dim: usize)
    where
        P: Penalty,
    {
        let beta = vec![f64::NAN; dim];
        let mut gradient = vec![1.0; dim];
        penalty.add_gradient(&beta, &mut gradient);
        assert_relative_eq!(penalty.value(&beta), 0.0, epsilon = f64::EPSILON);
        assert!(
            gradient
                .iter()
                .all(|value| value.to_bits() == 1.0_f64.to_bits())
        );
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
            assert_relative_eq!(actual, expected, epsilon = 2.0e-10);
        }

        let step = 1.0e-6;
        for index in 0..beta.len() {
            let mut lower = beta.to_vec();
            let mut upper = beta.to_vec();
            lower[index] -= step;
            upper[index] += step;
            let finite_difference = (penalty.value(&upper) - penalty.value(&lower)) / (2.0 * step);
            assert_relative_eq!(finite_difference, gradient[index], epsilon = 2.0e-7);
        }
    }
}
