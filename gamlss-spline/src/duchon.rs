use std::num::NonZeroUsize;
use std::ops::Range;

use gamlss_core::{
    DenseDesign, DesignMatrix, LinearPredictorBlock, LinearPredictorGeometry, ModelError,
    PredictorBlock, RowMultiplier,
};

use crate::{
    DiagonalPenaltyKernel, PenaltyKernel, ScaledPenalty, SplineBasisNd, SplineError, SplineRowBasis,
};

/// Exact smoothness specification for a Duchon spline.
///
/// `m` is the positive integer derivative order. Duchon's frequency weight
/// `s` is represented by the integer `2s`, so the supported integer and
/// half-integer values cannot suffer floating-point parsing ambiguity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DuchonSmoothness {
    derivative_order: NonZeroUsize,
    twice_s: i32,
}

impl DuchonSmoothness {
    /// Creates a specification from `m` and the exact half-step numerator `2s`.
    ///
    /// Dimension-dependent continuity conditions are checked when a basis is
    /// constructed.
    ///
    /// # Errors
    ///
    /// Returns an error when `derivative_order == 0`.
    pub fn try_new(derivative_order: usize, twice_s: i32) -> Result<Self, SplineError> {
        let derivative_order =
            NonZeroUsize::new(derivative_order).ok_or(ModelError::InvalidParameter {
                parameter: "Duchon derivative order m",
                expected: "> 0",
            })?;
        Ok(Self {
            derivative_order,
            twice_s,
        })
    }

    /// Creates the thin-plate specialization (`s = 0`).
    ///
    /// # Errors
    ///
    /// Returns an error when `derivative_order == 0`.
    pub fn thin_plate(derivative_order: usize) -> Result<Self, SplineError> {
        Self::try_new(derivative_order, 0)
    }

    /// Derivative order `m` in the Duchon seminorm.
    #[must_use]
    pub const fn derivative_order(self) -> usize {
        self.derivative_order.get()
    }

    /// Exact integer representation of twice the frequency exponent `s`.
    #[must_use]
    pub const fn twice_s(self) -> i32 {
        self.twice_s
    }

    /// Frequency exponent `s`.
    #[must_use]
    pub fn s(self) -> f64 {
        f64::from(self.twice_s) / 2.0
    }

    fn radial_power<const D: usize>(self) -> Result<i32, SplineError> {
        let dimension = i64::try_from(D).map_err(|_| SplineError::ParameterOverflow)?;
        let derivative_order =
            i64::try_from(self.derivative_order()).map_err(|_| SplineError::ParameterOverflow)?;
        let twice_s = i64::from(self.twice_s);
        let valid = -dimension < twice_s
            && twice_s < dimension
            && derivative_order
                .checked_mul(2)
                .and_then(|twice_m| twice_m.checked_add(twice_s))
                .is_some_and(|smoothness| smoothness > dimension);
        if D == 0 || !valid {
            return Err(SplineError::InvalidDuchonSmoothness {
                dimension: D,
                derivative_order: self.derivative_order(),
                twice_s: self.twice_s,
            });
        }
        let power = derivative_order
            .checked_mul(2)
            .and_then(|twice_m| twice_m.checked_add(twice_s))
            .and_then(|smoothness| smoothness.checked_sub(dimension))
            .ok_or(SplineError::ParameterOverflow)?;
        i32::try_from(power).map_err(|_| SplineError::ParameterOverflow)
    }
}

/// Low-rank isotropic Duchon regression-spline basis in `D` dimensions.
///
/// The basis is built from the radial semi-kernel
///
/// $$
/// \phi(r) \propto
/// \begin{cases}
/// r^q\log(r), & q\text{ even},\\\\
/// r^q, & q\text{ odd},
/// \end{cases}
/// \qquad q=2m+2s-D,
/// $$
///
/// with the sign chosen so the constrained kernel is positive definite. The
/// omitted positive proportionality constant is absorbed by the smoothing
/// scale. Polynomials of total degree below `m` form the exact null space.
/// Thin-plate regression splines are therefore the `s = 0` member of this
/// type, not a separate basis implementation.
///
/// Construction follows the low-rank spectral reduction: the requested
/// largest-magnitude kernel eigenvectors are constrained against the
/// polynomial space, then the remaining penalty is diagonalized. Coefficients
/// are ordered as penalized radial components followed by unpenalized
/// polynomial components. Construction is deterministic and backend-free; it
/// uses a full symmetric eigendecomposition, so callers with many observations
/// should deliberately supply a representative center subset.
///
/// # References
///
/// - J. Duchon, [Splines minimizing rotation-invariant semi-norms in Sobolev spaces](https://doi.org/10.1007/BFb0086566), 1977.
/// - S. N. Wood, [Thin plate regression splines](https://doi.org/10.1111/1467-9868.00374), 2003.
#[derive(Debug, Clone, PartialEq)]
pub struct DuchonSplineBasis<const D: usize> {
    centers: Box<[[f64; D]]>,
    shift: [f64; D],
    smoothness: DuchonSmoothness,
    radial_power: i32,
    monomial_powers: Box<[[u32; D]]>,
    radial_transform: Box<[f64]>,
    penalty_kernel: DiagonalPenaltyKernel,
}

impl<const D: usize> DuchonSplineBasis<D> {
    /// Constructs a rank-reduced Duchon basis from explicit centers.
    ///
    /// `rank` is the total number of returned columns, including the
    /// polynomial null space. Center order has no statistical meaning but is
    /// retained so the fitted basis can evaluate new points exactly.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid smoothness, non-finite or duplicate
    /// centers, insufficient rank, a rank-deficient polynomial geometry, size
    /// overflow, or failure of a numerical setup decomposition.
    pub fn try_new(
        centers: &[[f64; D]],
        rank: usize,
        smoothness: DuchonSmoothness,
    ) -> Result<Self, SplineError> {
        let radial_power = smoothness.radial_power::<D>()?;
        validate_centers(centers)?;

        let monomial_powers = monomial_powers::<D>(smoothness.derivative_order())?;
        let nullity = monomial_powers.len();
        let min_rank = nullity
            .checked_add(1)
            .ok_or(SplineError::ParameterOverflow)?;
        if rank < min_rank || rank > centers.len() {
            return Err(SplineError::InvalidDuchonRank {
                rank,
                min: min_rank,
                max: centers.len(),
            });
        }
        let radial_rank = rank - nullity;
        let shift = coordinate_means(centers)?;
        let polynomial = polynomial_matrix(centers, &shift, &monomial_powers)?;
        let radial_matrix = radial_matrix(centers, radial_power)?;
        let (all_eigenvalues, all_eigenvectors) =
            symmetric_eigendecomposition(radial_matrix, centers.len(), "Duchon radial kernel")?;
        let selected = largest_magnitude_indices(&all_eigenvalues, rank);
        let mut eigenvalues = Vec::with_capacity(rank);
        let mut eigenvectors =
            checked_zeros(centers.len(), rank, "Duchon truncated eigenvector matrix")?;
        for (new_column, old_column) in selected.into_iter().enumerate() {
            eigenvalues.push(all_eigenvalues[old_column]);
            for center in 0..centers.len() {
                eigenvectors[center * rank + new_column] =
                    all_eigenvectors[center * centers.len() + old_column];
            }
        }

        let constraint =
            projected_constraint(&eigenvectors, centers.len(), rank, &polynomial, nullity)?;
        let constraint_space = orthonormal_column_space(&constraint, rank, nullity)?;
        let complement = orthogonal_complement(&constraint_space, rank, radial_rank)?;
        let initial_transform = multiply_row_major_by_column_major(
            &eigenvectors,
            centers.len(),
            rank,
            &complement,
            radial_rank,
        )?;
        let constrained_penalty = projected_diagonal(&eigenvalues, &complement, rank, radial_rank)?;
        let (penalty_values, penalty_vectors) = symmetric_eigendecomposition(
            constrained_penalty,
            radial_rank,
            "Duchon constrained penalty",
        )?;
        let penalty_order = descending_indices(&penalty_values);
        let largest_penalty = penalty_values
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0, f64::max);
        let tolerance = decomposition_tolerance(largest_penalty, radial_rank);
        let mut diagonal = Vec::with_capacity(rank);
        let mut rotation = checked_zeros(radial_rank, radial_rank, "Duchon penalty rotation")?;
        for (new_column, old_column) in penalty_order.into_iter().enumerate() {
            let value = penalty_values[old_column];
            if !value.is_finite() || value <= tolerance {
                return Err(SplineError::NumericalFailure {
                    context: "a positive-rank Duchon penalty",
                });
            }
            diagonal.push(value);
            for row in 0..radial_rank {
                rotation[row * radial_rank + new_column] =
                    penalty_vectors[row * radial_rank + old_column];
            }
        }
        let radial_transform = multiply_row_major(
            &initial_transform,
            centers.len(),
            radial_rank,
            &rotation,
            radial_rank,
        )?;
        diagonal.resize(rank, 0.0);
        let penalty_kernel = DiagonalPenaltyKernel::try_new(diagonal)?;
        debug_assert_eq!(penalty_kernel.rank(), radial_rank);

        Ok(Self {
            centers: centers.into(),
            shift,
            smoothness,
            radial_power,
            monomial_powers: monomial_powers.into_boxed_slice(),
            radial_transform: radial_transform.into_boxed_slice(),
            penalty_kernel,
        })
    }

    /// Builds a prepared dense design at observation points.
    pub fn design(&self, points: &[[f64; D]]) -> Result<DuchonSplineDesign<D>, SplineError> {
        DuchonSplineDesign::new(points, self.clone())
    }

    /// Number of basis columns, including the polynomial null space.
    #[must_use]
    pub const fn n_basis(&self) -> usize {
        self.penalty_kernel.dim()
    }

    /// Number of penalized radial columns.
    #[must_use]
    pub fn radial_rank(&self) -> usize {
        self.penalty_kernel.rank()
    }

    /// Number of unpenalized polynomial columns.
    #[must_use]
    pub fn nullity(&self) -> usize {
        self.monomial_powers.len()
    }

    /// Centers used by the radial expansion.
    #[must_use]
    pub fn centers(&self) -> &[[f64; D]] {
        &self.centers
    }

    /// Per-coordinate center used by the polynomial null-space columns.
    #[must_use]
    pub const fn shift(&self) -> &[f64; D] {
        &self.shift
    }

    /// Duchon smoothness specification.
    #[must_use]
    pub const fn smoothness(&self) -> DuchonSmoothness {
        self.smoothness
    }

    /// Radial exponent `q = 2m + 2s - D`.
    #[must_use]
    pub const fn radial_power(&self) -> i32 {
        self.radial_power
    }

    /// Whether the radial semi-kernel contains the logarithmic factor.
    #[must_use]
    pub const fn uses_log_kernel(&self) -> bool {
        self.radial_power % 2 == 0
    }

    /// Polynomial exponents in null-space column order.
    #[must_use]
    pub fn monomial_powers(&self) -> &[[u32; D]] {
        &self.monomial_powers
    }

    /// Row-major map from radial center values to penalized basis columns.
    ///
    /// Its shape is `centers().len() × radial_rank()`.
    #[must_use]
    pub fn radial_transform(&self) -> &[f64] {
        &self.radial_transform
    }

    /// Diagonal unscaled roughness kernel in basis coefficient order.
    #[must_use]
    pub const fn penalty_kernel(&self) -> &DiagonalPenaltyKernel {
        &self.penalty_kernel
    }

    /// Borrows the prepared kernel and applies a smoothing scale.
    pub fn penalty(
        &self,
        lambda: f64,
    ) -> Result<ScaledPenalty<&DiagonalPenaltyKernel>, ModelError> {
        ScaledPenalty::try_new(lambda, &self.penalty_kernel)
    }

    /// Writes a dense basis row into `out`.
    pub fn evaluate_into(&self, point: &[f64; D], out: &mut [f64]) -> Result<(), SplineError> {
        SplineBasisNd::evaluate_into(self, point, out)
    }

    /// Returns a dense basis row.
    pub fn evaluate(&self, point: &[f64; D]) -> Result<Vec<f64>, SplineError> {
        SplineBasisNd::evaluate(self, point)
    }

    /// Visits non-zero basis values at a point without allocating.
    pub fn for_each_basis(
        &self,
        point: &[f64; D],
        f: impl FnMut(usize, f64),
    ) -> Result<(), SplineError> {
        SplineBasisNd::for_each_basis(self, point, f)
    }
}

impl<const D: usize> SplineBasisNd<D> for DuchonSplineBasis<D> {
    fn n_basis(&self) -> usize {
        self.n_basis()
    }

    fn for_each_basis(
        &self,
        point: &[f64; D],
        mut f: impl FnMut(usize, f64),
    ) -> Result<(), SplineError> {
        self.validate_point(point)?;
        let radial_rank = self.radial_rank();
        for column in 0..radial_rank {
            let mut value = 0.0;
            for (center_index, center) in self.centers.iter().enumerate() {
                let radial = radial_value(distance(point, center), self.radial_power)?;
                value = self.radial_transform[center_index * radial_rank + column]
                    .mul_add(radial, value);
            }
            if !value.is_finite() {
                return Err(SplineError::NumericalFailure {
                    context: "a Duchon prediction row",
                });
            }
            if value != 0.0 {
                f(column, value);
            }
        }
        for (offset, powers) in self.monomial_powers.iter().enumerate() {
            let value = monomial_value(point, &self.shift, powers);
            if !value.is_finite() {
                return Err(SplineError::NumericalFailure {
                    context: "a Duchon polynomial prediction row",
                });
            }
            if value != 0.0 {
                f(radial_rank + offset, value);
            }
        }
        Ok(())
    }
}

/// Prepared dense predictor geometry for a [`DuchonSplineBasis`].
///
/// Radial rows are dense and comparatively expensive to evaluate, so this
/// design computes them once. The reusable basis remains available for
/// out-of-sample prediction.
#[derive(Debug, Clone, PartialEq)]
pub struct DuchonSplineDesign<const D: usize> {
    basis: DuchonSplineBasis<D>,
    prepared: DenseDesign,
}

impl<const D: usize> DuchonSplineDesign<D> {
    /// Prepares all basis rows for the supplied points.
    pub fn new(points: &[[f64; D]], basis: DuchonSplineBasis<D>) -> Result<Self, SplineError> {
        let nparams = basis.n_basis();
        let mut values = checked_zeros(points.len(), nparams, "Duchon design matrix")?;
        for (point, row) in points.iter().zip(values.chunks_exact_mut(nparams)) {
            basis.evaluate_into(point, row)?;
        }
        Ok(Self {
            basis,
            prepared: DenseDesign::from_row_major(points.len(), nparams, values)?,
        })
    }

    /// Reusable fitted basis metadata.
    #[must_use]
    pub const fn basis(&self) -> &DuchonSplineBasis<D> {
        &self.basis
    }

    /// Number of prepared rows.
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.prepared.nrows()
    }

    /// Number of coefficients.
    #[must_use]
    pub const fn n_basis(&self) -> usize {
        self.basis.n_basis()
    }

    /// Prepared row-major design values.
    #[must_use]
    pub fn row_major_values(&self) -> &[f64] {
        self.prepared.values()
    }

    fn row(&self, row: usize) -> &[f64] {
        let nparams = self.n_basis();
        &self.prepared.values()[row * nparams..(row + 1) * nparams]
    }

    #[inline]
    const fn linear_block(&self) -> LinearPredictorBlock<&DenseDesign> {
        LinearPredictorBlock::new(&self.prepared)
    }
}

impl<const D: usize> SplineRowBasis for DuchonSplineDesign<D> {
    fn nrows(&self) -> usize {
        self.prepared.nrows()
    }

    fn nparams(&self) -> usize {
        self.prepared.ncols()
    }

    fn for_each_row_basis(&self, row: usize, mut f: impl FnMut(usize, f64)) {
        for (index, value) in self.row(row).iter().copied().enumerate() {
            if value != 0.0 {
                f(index, value);
            }
        }
    }
}

impl<const D: usize> PredictorBlock for DuchonSplineDesign<D> {
    fn nrows(&self) -> usize {
        self.prepared.nrows()
    }

    fn nparams(&self) -> usize {
        self.prepared.ncols()
    }

    fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
        self.prepared.dot_row(row, beta)
    }

    fn zero_beta_constant_contribution(&self) -> Option<f64> {
        Some(0.0)
    }

    fn add_gradient_range(
        &self,
        rows: Range<usize>,
        scores: &[f64],
        _beta: &[f64],
        grad: &mut [f64],
    ) {
        self.prepared.add_t_mul_vec_range(rows, scores, grad);
    }

    fn add_weighted_gradient_by_range<M>(
        &self,
        rows: Range<usize>,
        scores: &[f64],
        multiplier: &M,
        _beta: &[f64],
        grad: &mut [f64],
    ) where
        M: RowMultiplier + ?Sized,
    {
        self.prepared
            .add_weighted_t_mul_vec_by_range(rows, scores, multiplier, grad);
    }
}

impl<const D: usize> LinearPredictorGeometry for DuchonSplineDesign<D> {
    fn add_weighted_gram(&self, row_weights: &[f64], out: &mut [f64]) -> Result<(), ModelError> {
        self.linear_block().add_weighted_gram(row_weights, out)
    }

    fn add_weighted_gram_by<M>(
        &self,
        row_weights: &[f64],
        multiplier: &M,
        out: &mut [f64],
    ) -> Result<(), ModelError>
    where
        M: RowMultiplier + ?Sized,
    {
        self.linear_block()
            .add_weighted_gram_by(row_weights, multiplier, out)
    }

    fn add_t_mul_vec(&self, row_scores: &[f64], out: &mut [f64]) -> Result<(), ModelError> {
        self.linear_block().add_t_mul_vec(row_scores, out)
    }

    fn add_t_mul_vec_by<M>(
        &self,
        row_scores: &[f64],
        multiplier: &M,
        out: &mut [f64],
    ) -> Result<(), ModelError>
    where
        M: RowMultiplier + ?Sized,
    {
        self.linear_block()
            .add_t_mul_vec_by(row_scores, multiplier, out)
    }
}

fn validate_centers<const D: usize>(centers: &[[f64; D]]) -> Result<(), SplineError> {
    if centers.is_empty() {
        return Err(SplineError::EmptyInput);
    }
    if centers.iter().flatten().any(|value| !value.is_finite()) {
        return Err(SplineError::NonFiniteValue);
    }
    for left in 0..centers.len() {
        if centers[left + 1..]
            .iter()
            .any(|right| points_have_same_bits(right, &centers[left]))
        {
            return Err(SplineError::DegenerateDuchonCenters);
        }
    }
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn coordinate_means<const D: usize>(centers: &[[f64; D]]) -> Result<[f64; D], SplineError> {
    let denominator = centers.len() as f64;
    let mut means = [0.0; D];
    for center in centers {
        for (mean, coordinate) in means.iter_mut().zip(center.iter().copied()) {
            *mean += coordinate / denominator;
        }
    }
    if means.iter().all(|value| value.is_finite()) {
        Ok(means)
    } else {
        Err(SplineError::NumericalFailure {
            context: "Duchon coordinate centering",
        })
    }
}

fn monomial_powers<const D: usize>(order: usize) -> Result<Vec<[u32; D]>, SplineError> {
    let total_degree = order.checked_sub(1).ok_or(SplineError::ParameterOverflow)?;
    let expected = checked_binomial(
        total_degree
            .checked_add(D)
            .ok_or(SplineError::ParameterOverflow)?,
        D,
    )?;
    let mut output = Vec::with_capacity(expected);
    let mut current = [0_u32; D];
    for degree in 0..=total_degree {
        append_degree_powers(0, degree, &mut current, &mut output)?;
    }
    debug_assert_eq!(output.len(), expected);
    Ok(output)
}

fn append_degree_powers<const D: usize>(
    axis: usize,
    remaining: usize,
    current: &mut [u32; D],
    output: &mut Vec<[u32; D]>,
) -> Result<(), SplineError> {
    debug_assert!(D > 0);
    if axis + 1 == D {
        current[axis] = u32::try_from(remaining).map_err(|_| SplineError::ParameterOverflow)?;
        output.push(*current);
        return Ok(());
    }
    for power in (0..=remaining).rev() {
        current[axis] = u32::try_from(power).map_err(|_| SplineError::ParameterOverflow)?;
        append_degree_powers(axis + 1, remaining - power, current, output)?;
    }
    Ok(())
}

fn checked_binomial(n: usize, k: usize) -> Result<usize, SplineError> {
    let k = k.min(n - k);
    let mut value = 1_usize;
    for index in 1..=k {
        let mut numerator = n - k + index;
        let mut denominator = index;
        let common = gcd(numerator, denominator);
        numerator /= common;
        denominator /= common;
        let common = gcd(value, denominator);
        value /= common;
        denominator /= common;
        value = value
            .checked_mul(numerator)
            .ok_or(SplineError::ParameterOverflow)?;
        debug_assert_eq!(value % denominator, 0);
        value /= denominator;
    }
    Ok(value)
}

const fn gcd(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn polynomial_matrix<const D: usize>(
    points: &[[f64; D]],
    shift: &[f64; D],
    powers: &[[u32; D]],
) -> Result<Vec<f64>, SplineError> {
    let mut matrix = checked_zeros(points.len(), powers.len(), "Duchon polynomial matrix")?;
    let ncols = powers.len();
    for (row, point) in points.iter().enumerate() {
        for (column, powers) in powers.iter().enumerate() {
            let value = monomial_value(point, shift, powers);
            if !value.is_finite() {
                return Err(SplineError::NumericalFailure {
                    context: "Duchon polynomial matrix",
                });
            }
            matrix[row * ncols + column] = value;
        }
    }
    Ok(matrix)
}

fn monomial_value<const D: usize>(point: &[f64; D], shift: &[f64; D], powers: &[u32; D]) -> f64 {
    point
        .iter()
        .copied()
        .zip(shift.iter().copied())
        .zip(powers.iter().copied())
        .fold(1.0, |value, ((coordinate, shift), power)| {
            value * pow_u32(coordinate - shift, power)
        })
}

fn pow_u32(mut base: f64, mut exponent: u32) -> f64 {
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

fn radial_matrix<const D: usize>(
    centers: &[[f64; D]],
    power: i32,
) -> Result<Vec<f64>, SplineError> {
    let mut matrix = checked_zeros(centers.len(), centers.len(), "Duchon radial matrix")?;
    for row in 0..centers.len() {
        for column in row..centers.len() {
            let value = radial_value(distance(&centers[row], &centers[column]), power)?;
            matrix[row * centers.len() + column] = value;
            matrix[column * centers.len() + row] = value;
        }
    }
    Ok(matrix)
}

fn distance<const D: usize>(left: &[f64; D], right: &[f64; D]) -> f64 {
    left.iter()
        .copied()
        .zip(right.iter().copied())
        .fold(0.0, |norm, (left, right)| norm.hypot(left - right))
}

fn radial_value(distance: f64, power: i32) -> Result<f64, SplineError> {
    if distance == 0.0 {
        return Ok(0.0);
    }
    if !distance.is_finite() {
        return Err(SplineError::NumericalFailure {
            context: "a Duchon radial distance",
        });
    }
    let powered = distance.powi(power);
    let unsigned = if power % 2 == 0 {
        powered * distance.ln()
    } else {
        powered
    };
    let half_power = power / 2;
    let sign = if (half_power + 1) % 2 == 0 { 1.0 } else { -1.0 };
    let value = sign * unsigned;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SplineError::NumericalFailure {
            context: "a Duchon radial kernel value",
        })
    }
}

fn projected_constraint(
    eigenvectors: &[f64],
    nrows: usize,
    rank: usize,
    polynomial: &[f64],
    nullity: usize,
) -> Result<Vec<f64>, SplineError> {
    let mut constraint = checked_zeros(rank, nullity, "Duchon projected constraint")?;
    for eigenvector in 0..rank {
        for polynomial_column in 0..nullity {
            let mut value = 0.0;
            for row in 0..nrows {
                value = eigenvectors[row * rank + eigenvector]
                    .mul_add(polynomial[row * nullity + polynomial_column], value);
            }
            constraint[eigenvector * nullity + polynomial_column] = value;
        }
    }
    Ok(constraint)
}

fn orthonormal_column_space(
    matrix: &[f64],
    nrows: usize,
    ncols: usize,
) -> Result<Vec<f64>, SplineError> {
    let mut columns = Vec::with_capacity(
        nrows
            .checked_mul(ncols)
            .ok_or(SplineError::ParameterOverflow)?,
    );
    let mut candidate = vec![0.0; nrows];
    for column in 0..ncols {
        for row in 0..nrows {
            candidate[row] = matrix[row * ncols + column];
        }
        let original_norm = squared_norm(&candidate).sqrt();
        project_out_twice(&mut candidate, &columns, nrows);
        let norm = squared_norm(&candidate).sqrt();
        let tolerance = decomposition_tolerance(original_norm, nrows);
        if !norm.is_finite() || norm <= tolerance {
            return Err(SplineError::DegenerateDuchonCenters);
        }
        for value in &mut candidate {
            *value /= norm;
        }
        columns.extend_from_slice(&candidate);
    }
    Ok(columns)
}

fn orthogonal_complement(
    column_space: &[f64],
    dimension: usize,
    expected_columns: usize,
) -> Result<Vec<f64>, SplineError> {
    let mut complement = Vec::with_capacity(
        dimension
            .checked_mul(expected_columns)
            .ok_or(SplineError::ParameterOverflow)?,
    );
    let mut candidate = vec![0.0; dimension];
    for axis in 0..dimension {
        candidate.fill(0.0);
        candidate[axis] = 1.0;
        project_out_twice(&mut candidate, column_space, dimension);
        project_out_twice(&mut candidate, &complement, dimension);
        let norm = squared_norm(&candidate).sqrt();
        if norm > decomposition_tolerance(1.0, dimension) {
            for value in &mut candidate {
                *value /= norm;
            }
            complement.extend_from_slice(&candidate);
            if complement.len() / dimension == expected_columns {
                return Ok(complement);
            }
        }
    }
    Err(SplineError::NumericalFailure {
        context: "the Duchon constraint null space",
    })
}

fn project_out_twice(candidate: &mut [f64], columns: &[f64], nrows: usize) {
    debug_assert_eq!(candidate.len(), nrows);
    debug_assert_eq!(columns.len() % nrows, 0);
    for _ in 0..2 {
        for column in columns.chunks_exact(nrows) {
            let projection = dot(candidate, column);
            add_scaled(-projection, column, candidate);
        }
    }
}

fn multiply_row_major_by_column_major(
    left: &[f64],
    nrows: usize,
    inner: usize,
    right_columns: &[f64],
    ncols: usize,
) -> Result<Vec<f64>, SplineError> {
    let mut output = checked_zeros(nrows, ncols, "Duchon radial transform")?;
    for row in 0..nrows {
        for column in 0..ncols {
            output[row * ncols + column] = dot(
                &left[row * inner..(row + 1) * inner],
                &right_columns[column * inner..(column + 1) * inner],
            );
        }
    }
    Ok(output)
}

fn projected_diagonal(
    diagonal: &[f64],
    columns: &[f64],
    dimension: usize,
    ncols: usize,
) -> Result<Vec<f64>, SplineError> {
    let mut output = checked_zeros(ncols, ncols, "Duchon projected penalty")?;
    for left in 0..ncols {
        for right in left..ncols {
            let mut value = 0.0;
            for index in 0..dimension {
                value = (columns[left * dimension + index] * diagonal[index])
                    .mul_add(columns[right * dimension + index], value);
            }
            output[left * ncols + right] = value;
            output[right * ncols + left] = value;
        }
    }
    Ok(output)
}

fn multiply_row_major(
    left: &[f64],
    nrows: usize,
    inner: usize,
    right: &[f64],
    ncols: usize,
) -> Result<Vec<f64>, SplineError> {
    let mut output = checked_zeros(nrows, ncols, "Duchon rotated radial transform")?;
    for row in 0..nrows {
        for column in 0..ncols {
            let mut value = 0.0;
            for index in 0..inner {
                value = left[row * inner + index].mul_add(right[index * ncols + column], value);
            }
            output[row * ncols + column] = value;
        }
    }
    Ok(output)
}

fn symmetric_eigendecomposition(
    mut matrix: Vec<f64>,
    dimension: usize,
    context: &'static str,
) -> Result<(Vec<f64>, Vec<f64>), SplineError> {
    if dimension == 0 || matrix.len() != dimension.saturating_mul(dimension) {
        return Err(SplineError::NumericalFailure { context });
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(SplineError::NumericalFailure { context });
    }
    let mut vectors = checked_zeros(dimension, dimension, context)?;
    for index in 0..dimension {
        vectors[index * dimension + index] = 1.0;
    }
    if dimension == 1 {
        return Ok((matrix, vectors));
    }
    let max_iterations = dimension
        .checked_mul(dimension)
        .and_then(|value| value.checked_mul(64))
        .ok_or(SplineError::ParameterOverflow)?;
    let mut converged = false;
    for _ in 0..max_iterations {
        let (left, right, off_diagonal) = largest_off_diagonal(&matrix, dimension);
        let scale = matrix[left * dimension + left]
            .abs()
            .max(matrix[right * dimension + right].abs())
            .max(off_diagonal);
        if off_diagonal <= decomposition_tolerance(scale, dimension) {
            converged = true;
            break;
        }
        jacobi_rotation(&mut matrix, &mut vectors, dimension, left, right);
    }
    if !converged {
        return Err(SplineError::NumericalFailure { context });
    }
    let values = (0..dimension)
        .map(|index| matrix[index * dimension + index])
        .collect();
    Ok((values, vectors))
}

fn largest_off_diagonal(matrix: &[f64], dimension: usize) -> (usize, usize, f64) {
    let mut best = (0, 1, matrix[1].abs());
    for row in 0..dimension {
        for column in row + 1..dimension {
            let value = matrix[row * dimension + column].abs();
            if value > best.2 {
                best = (row, column, value);
            }
        }
    }
    best
}

fn jacobi_rotation(
    matrix: &mut [f64],
    vectors: &mut [f64],
    dimension: usize,
    left: usize,
    right: usize,
) {
    let diagonal_left = matrix[left * dimension + left];
    let diagonal_right = matrix[right * dimension + right];
    let off_diagonal = matrix[left * dimension + right];
    let tau = (diagonal_right - diagonal_left) / (2.0 * off_diagonal);
    let tangent = if tau >= 0.0 {
        1.0 / (tau + tau.hypot(1.0))
    } else {
        -1.0 / (-tau + tau.hypot(1.0))
    };
    let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
    let sine = tangent * cosine;
    for index in 0..dimension {
        if index == left || index == right {
            continue;
        }
        let value_left = matrix[index * dimension + left];
        let value_right = matrix[index * dimension + right];
        let rotated_left = cosine.mul_add(value_left, -sine * value_right);
        let rotated_right = sine.mul_add(value_left, cosine * value_right);
        matrix[index * dimension + left] = rotated_left;
        matrix[left * dimension + index] = rotated_left;
        matrix[index * dimension + right] = rotated_right;
        matrix[right * dimension + index] = rotated_right;
    }
    matrix[left * dimension + left] = diagonal_left - tangent * off_diagonal;
    matrix[right * dimension + right] = diagonal_right + tangent * off_diagonal;
    matrix[left * dimension + right] = 0.0;
    matrix[right * dimension + left] = 0.0;
    for row in 0..dimension {
        let value_left = vectors[row * dimension + left];
        let value_right = vectors[row * dimension + right];
        vectors[row * dimension + left] = cosine.mul_add(value_left, -sine * value_right);
        vectors[row * dimension + right] = sine.mul_add(value_left, cosine * value_right);
    }
}

fn largest_magnitude_indices(values: &[f64], count: usize) -> Vec<usize> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        values[*right]
            .abs()
            .total_cmp(&values[*left].abs())
            .then_with(|| values[*right].total_cmp(&values[*left]))
            .then_with(|| left.cmp(right))
    });
    indices.truncate(count);
    indices
}

fn descending_indices(values: &[f64]) -> Vec<usize> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        values[*right]
            .total_cmp(&values[*left])
            .then_with(|| left.cmp(right))
    });
    indices
}

fn decomposition_tolerance(scale: f64, dimension: usize) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let dimension = dimension as f64;
    256.0 * f64::EPSILON * dimension.max(1.0) * scale.max(f64::MIN_POSITIVE)
}

fn checked_zeros(
    nrows: usize,
    ncols: usize,
    _context: &'static str,
) -> Result<Vec<f64>, SplineError> {
    let len = nrows
        .checked_mul(ncols)
        .ok_or(SplineError::ParameterOverflow)?;
    Ok(vec![0.0; len])
}

fn points_have_same_bits<const D: usize>(left: &[f64; D], right: &[f64; D]) -> bool {
    left.iter()
        .copied()
        .zip(right.iter().copied())
        .all(|(left, right)| canonical_float_bits(left) == canonical_float_bits(right))
}

const fn canonical_float_bits(value: f64) -> u64 {
    let bits = value.to_bits();
    if bits.trailing_zeros() >= 63 { 0 } else { bits }
}

fn squared_norm(values: &[f64]) -> f64 {
    dot(values, values)
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .copied()
        .zip(right.iter().copied())
        .fold(0.0, |value, (left, right)| left.mul_add(right, value))
}

fn add_scaled(scale: f64, source: &[f64], output: &mut [f64]) {
    for (output, source) in output.iter_mut().zip(source.iter().copied()) {
        *output = scale.mul_add(source, *output);
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{LinearPredictorGeometry, Penalty, PredictorBlock, RowMultiplier};

    use super::{DuchonSmoothness, DuchonSplineBasis, symmetric_eigendecomposition};
    use crate::{PenaltyKernel, SplineError, SplineRowBasisExt};

    const CENTERS: [[f64; 2]; 9] = [
        [-1.0, -1.0],
        [0.0, -1.0],
        [1.0, -1.0],
        [-1.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [-1.0, 1.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ];

    #[test]
    fn thin_plate_member_has_expected_null_space_and_diagonal_penalty() {
        let basis =
            DuchonSplineBasis::try_new(&CENTERS, 8, DuchonSmoothness::thin_plate(2).unwrap())
                .unwrap();
        assert_eq!(basis.n_basis(), 8);
        assert_eq!(basis.radial_rank(), 5);
        assert_eq!(basis.nullity(), 3);
        assert_eq!(basis.radial_power(), 2);
        assert!(basis.uses_log_kernel());
        assert_eq!(basis.monomial_powers(), &[[0, 0], [1, 0], [0, 1]]);
        assert_eq!(basis.penalty_kernel().rank(), 5);
        assert_eq!(basis.penalty_kernel().nullity(), 3);
        assert!(
            basis.penalty_kernel().diagonal()[..5]
                .iter()
                .all(|x| *x > 0.0)
        );
        assert_eq!(basis.penalty_kernel().diagonal()[5..], [0.0; 3]);

        for radial_column in 0..basis.radial_rank() {
            for polynomial_column in 0..basis.nullity() {
                let constraint = basis
                    .centers()
                    .iter()
                    .enumerate()
                    .map(|(center, point)| {
                        basis.radial_transform()[center * basis.radial_rank() + radial_column]
                            * super::monomial_value(
                                point,
                                basis.shift(),
                                &basis.monomial_powers()[polynomial_column],
                            )
                    })
                    .sum::<f64>();
                assert_relative_eq!(constraint, 0.0, epsilon = 2.0e-12);
            }
        }

        let values = basis.evaluate(&[0.25, -0.5]).unwrap();
        assert_relative_eq!(values[5], 1.0, epsilon = 1.0e-14);
        assert_relative_eq!(values[6], 0.25, epsilon = 1.0e-14);
        assert_relative_eq!(values[7], -0.5, epsilon = 1.0e-14);
    }

    #[test]
    fn generalized_member_is_not_only_a_thin_plate_alias() {
        let generalized =
            DuchonSplineBasis::try_new(&CENTERS, 8, DuchonSmoothness::try_new(1, 1).unwrap())
                .unwrap();
        assert_relative_eq!(generalized.smoothness().s(), 0.5, epsilon = f64::EPSILON);
        assert_eq!(generalized.radial_power(), 1);
        assert!(!generalized.uses_log_kernel());
        assert_eq!(generalized.nullity(), 1);
        assert_eq!(generalized.radial_rank(), 7);
    }

    #[test]
    fn isotropic_construction_is_rotation_invariant() {
        let rotated = CENTERS.map(|[x, y]| [-y, x]);
        let smoothness = DuchonSmoothness::thin_plate(2).unwrap();
        let original = DuchonSplineBasis::try_new(&CENTERS, 8, smoothness).unwrap();
        let rotated = DuchonSplineBasis::try_new(&rotated, 8, smoothness).unwrap();
        for (original, rotated) in original
            .penalty_kernel()
            .diagonal()
            .iter()
            .zip(rotated.penalty_kernel().diagonal())
        {
            assert_relative_eq!(original, rotated, epsilon = 2.0e-10);
        }
    }

    #[test]
    fn prepared_design_matches_basis_and_linear_geometry() {
        let basis =
            DuchonSplineBasis::try_new(&CENTERS, 7, DuchonSmoothness::thin_plate(2).unwrap())
                .unwrap();
        let points = [[-0.8, 0.2], [0.1, -0.3], [0.7, 0.9]];
        let design = basis.design(&points).unwrap();
        for (row, point) in points.iter().enumerate() {
            assert_eq!(
                design.row_major_values()[row * 7..(row + 1) * 7],
                basis.evaluate(point).unwrap()
            );
        }
        let beta = [0.3, -0.4, 0.2, 0.8, -0.1, 0.5, -0.7];
        let scores = [0.2, -0.5, 0.9];
        let mut gradient = [0.0; 7];
        design.add_gradient(&scores, &beta, &mut gradient);
        let mut transpose = [0.0; 7];
        design.add_t_mul_vec(&scores, &mut transpose).unwrap();
        for (gradient, transpose) in gradient.into_iter().zip(transpose) {
            assert_relative_eq!(gradient, transpose, epsilon = 1.0e-13);
        }
        assert_eq!(
            design.to_row_major_values().unwrap(),
            design.row_major_values()
        );
        for row in 0..points.len() {
            let expected = dot(&design.row_major_values()[row * 7..(row + 1) * 7], &beta);
            assert_relative_eq!(design.eta_row(row, &beta), expected, epsilon = 1.0e-13);
        }
        let penalty = basis.penalty(0.7).unwrap();
        assert!(penalty.value(&beta) >= 0.0);
    }

    #[test]
    fn invalid_geometry_and_smoothness_are_rejected() {
        let discontinuous =
            DuchonSplineBasis::try_new(&CENTERS, 5, DuchonSmoothness::try_new(1, 0).unwrap());
        assert!(matches!(
            discontinuous,
            Err(SplineError::InvalidDuchonSmoothness { .. })
        ));
        let mut duplicate = CENTERS;
        duplicate[8] = duplicate[0];
        assert_eq!(
            DuchonSplineBasis::try_new(&duplicate, 5, DuchonSmoothness::thin_plate(2).unwrap())
                .unwrap_err(),
            SplineError::DegenerateDuchonCenters
        );
    }

    #[test]
    fn symmetric_eigendecomposition_reconstructs_its_input() {
        let matrix = vec![4.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0, 0.5, 5.0];
        let (values, vectors) =
            symmetric_eigendecomposition(matrix.clone(), 3, "test matrix").unwrap();
        for row in 0..3 {
            for column in 0..3 {
                let reconstructed = (0..3)
                    .map(|index| {
                        vectors[row * 3 + index] * values[index] * vectors[column * 3 + index]
                    })
                    .sum::<f64>();
                assert_relative_eq!(reconstructed, matrix[row * 3 + column], epsilon = 1.0e-12);
            }
        }
    }

    #[test]
    fn fused_geometry_does_not_read_masked_multipliers() {
        struct PanicOnFirstRow;

        impl RowMultiplier for PanicOnFirstRow {
            fn multiplier_at(&self, row: usize) -> f64 {
                assert_ne!(row, 0, "masked multiplier was evaluated");
                1.0
            }
        }

        let basis =
            DuchonSplineBasis::try_new(&CENTERS, 7, DuchonSmoothness::thin_plate(2).unwrap())
                .unwrap();
        let design = basis
            .design(&[[-0.8, 0.2], [0.1, -0.3], [0.7, 0.9]])
            .unwrap();
        let mut gram = [0.0; 49];
        design
            .add_weighted_gram_by(&[0.0, 1.0, 1.0], &PanicOnFirstRow, &mut gram)
            .unwrap();
        let mut transpose = [0.0; 7];
        design
            .add_t_mul_vec_by(&[0.0, 1.0, 1.0], &PanicOnFirstRow, &mut transpose)
            .unwrap();
    }

    fn dot(left: &[f64], right: &[f64]) -> f64 {
        left.iter()
            .zip(right)
            .map(|(left, right)| left * right)
            .sum()
    }
}
