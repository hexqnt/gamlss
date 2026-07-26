use gamlss_core::ModelError;

use crate::local::{
    bspline_active_range, bspline_value, cyclic_local_basis_derivative_order,
    open_uniform_local_basis_derivative,
};
use crate::{
    BSplineBasis, BSplineDesign, CyclicSplineDesign, CyclicSplineSpec, FourierBasis, FourierDesign,
    ISplineBasis, ISplineDesign, MSplineBasis, MSplineDesign, NaturalCubicSplineBasis,
    NaturalCubicSplineDesign, OnDemandSplineDesign, OpenUniformSplineBasis,
    OpenUniformSplineDesign, PeriodicSplineDesign, PeriodicSplineSpec, SplineBasis1d, SplineError,
    SplineRowBasis, TruncatedPowerBasis,
};

/// Common allocation-free API for derivatives of one-dimensional spline bases.
///
/// Derivative order zero is always the original basis row. Polynomial spline
/// implementations support derivatives through their degree; Fourier bases
/// support every order whose finite values remain representable. At interior
/// knots, piecewise derivatives use the same right-open convention as basis
/// evaluation.
pub trait DifferentiableSplineBasis1d: SplineBasis1d {
    /// Largest supported derivative order, or `None` when it is unbounded.
    fn max_derivative_order(&self) -> Option<usize>;

    /// Visits non-zero derivative basis values without allocating.
    ///
    /// # Errors
    ///
    /// Returns [`SplineError::UnsupportedDerivativeOrder`] when the requested
    /// order exceeds [`Self::max_derivative_order`], or the basis' coordinate
    /// error when `x` is invalid.
    fn for_each_basis_derivative(
        &self,
        x: f64,
        derivative_order: usize,
        f: impl FnMut(usize, f64),
    ) -> Result<(), SplineError>;

    /// Writes one derivative basis row into `out`.
    ///
    /// # Errors
    ///
    /// In addition to [`Self::for_each_basis_derivative`] errors, returns a
    /// design-size error when `out.len() != self.n_basis()`.
    fn evaluate_basis_derivative_into(
        &self,
        x: f64,
        derivative_order: usize,
        out: &mut [f64],
    ) -> Result<(), SplineError> {
        validate_output_len(self.n_basis(), out.len())?;
        out.fill(0.0);
        self.for_each_basis_derivative(x, derivative_order, |index, value| {
            out[index] = value;
        })
    }

    /// Allocates and returns one derivative basis row.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::for_each_basis_derivative`].
    fn evaluate_basis_derivative(
        &self,
        x: f64,
        derivative_order: usize,
    ) -> Result<Vec<f64>, SplineError> {
        let mut values = vec![0.0; self.n_basis()];
        self.evaluate_basis_derivative_into(x, derivative_order, &mut values)?;
        Ok(values)
    }
}

/// Row-indexed counterpart of [`DifferentiableSplineBasis1d`].
///
/// This lets tensor products compute partial derivatives without knowing how a
/// marginal design stores or recomputes its rows.
pub trait DifferentiableSplineRowBasis: SplineRowBasis {
    /// Largest supported derivative order, or `None` when it is unbounded.
    fn max_derivative_order(&self) -> Option<usize>;

    /// Visits one derivative row.
    ///
    /// # Errors
    ///
    /// Returns an error when `derivative_order` is unsupported.
    fn for_each_row_basis_derivative(
        &self,
        row: usize,
        derivative_order: usize,
        f: impl FnMut(usize, f64),
    ) -> Result<(), SplineError>;
}

impl<B> DifferentiableSplineRowBasis for OnDemandSplineDesign<B>
where
    B: DifferentiableSplineBasis1d,
{
    fn max_derivative_order(&self) -> Option<usize> {
        self.basis().max_derivative_order()
    }

    fn for_each_row_basis_derivative(
        &self,
        row: usize,
        derivative_order: usize,
        f: impl FnMut(usize, f64),
    ) -> Result<(), SplineError> {
        debug_assert!(row < self.nrows());
        self.basis()
            .for_each_basis_derivative(self.x()[row], derivative_order, f)
    }
}

macro_rules! impl_prepared_derivative_rows {
    ($design:ty, $basis:ident, $coordinates:ident) => {
        impl DifferentiableSplineRowBasis for $design {
            fn max_derivative_order(&self) -> Option<usize> {
                self.$basis().max_derivative_order()
            }

            fn for_each_row_basis_derivative(
                &self,
                row: usize,
                derivative_order: usize,
                f: impl FnMut(usize, f64),
            ) -> Result<(), SplineError> {
                debug_assert!(row < SplineRowBasis::nrows(self));
                self.$basis().for_each_basis_derivative(
                    self.$coordinates()[row],
                    derivative_order,
                    f,
                )
            }
        }
    };
}

impl_prepared_derivative_rows!(BSplineDesign, basis, x);
impl_prepared_derivative_rows!(OpenUniformSplineDesign, basis, x);
impl_prepared_derivative_rows!(CyclicSplineDesign, spec, phi);
impl_prepared_derivative_rows!(PeriodicSplineDesign, spec, x);
impl_prepared_derivative_rows!(FourierDesign, basis, x);
impl_prepared_derivative_rows!(MSplineDesign, basis, x);
impl_prepared_derivative_rows!(ISplineDesign, basis, x);
impl_prepared_derivative_rows!(NaturalCubicSplineDesign, basis, x);

impl<T> DifferentiableSplineBasis1d for &T
where
    T: DifferentiableSplineBasis1d + ?Sized,
{
    fn max_derivative_order(&self) -> Option<usize> {
        T::max_derivative_order(*self)
    }

    fn for_each_basis_derivative(
        &self,
        x: f64,
        derivative_order: usize,
        f: impl FnMut(usize, f64),
    ) -> Result<(), SplineError> {
        T::for_each_basis_derivative(*self, x, derivative_order, f)
    }
}

impl DifferentiableSplineBasis1d for BSplineBasis {
    fn max_derivative_order(&self) -> Option<usize> {
        Some(self.degree())
    }

    fn for_each_basis_derivative(
        &self,
        x: f64,
        derivative_order: usize,
        mut f: impl FnMut(usize, f64),
    ) -> Result<(), SplineError> {
        validate_coordinate_and_order(self, x, derivative_order)?;
        if derivative_order == 0 {
            return SplineBasis1d::for_each_basis(self, x, f);
        }
        if let Some(active) = bspline_active_range(self.knots(), self.n_basis(), self.degree(), x) {
            for index in active {
                let value = bspline_derivative_value(
                    self.knots(),
                    self.n_basis(),
                    index,
                    self.degree(),
                    derivative_order,
                    x,
                );
                if value != 0.0 {
                    f(index, value);
                }
            }
        }
        Ok(())
    }
}

impl DifferentiableSplineBasis1d for MSplineBasis {
    fn max_derivative_order(&self) -> Option<usize> {
        Some(self.degree())
    }

    fn for_each_basis_derivative(
        &self,
        x: f64,
        derivative_order: usize,
        mut f: impl FnMut(usize, f64),
    ) -> Result<(), SplineError> {
        validate_coordinate_and_order(self, x, derivative_order)?;
        if derivative_order == 0 {
            return SplineBasis1d::for_each_basis(self, x, f);
        }
        self.bspline()
            .for_each_basis_derivative(x, derivative_order, |index, value| {
                let support = self.knots()[index + self.degree() + 1] - self.knots()[index];
                if support > 0.0 {
                    #[allow(clippy::cast_precision_loss)]
                    let scale = (self.degree() + 1) as f64 / support;
                    let value = scale * value;
                    if value != 0.0 {
                        f(index, value);
                    }
                }
            })
    }
}

impl DifferentiableSplineBasis1d for ISplineBasis {
    fn max_derivative_order(&self) -> Option<usize> {
        self.degree().checked_add(1)
    }

    fn for_each_basis_derivative(
        &self,
        x: f64,
        derivative_order: usize,
        f: impl FnMut(usize, f64),
    ) -> Result<(), SplineError> {
        validate_coordinate_and_order(self, x, derivative_order)?;
        match derivative_order {
            0 => SplineBasis1d::for_each_basis(self, x, f),
            1 => {
                self.for_each_derivative_basis(x, f);
                Ok(())
            }
            order => self.mspline.for_each_basis_derivative(x, order - 1, f),
        }
    }
}

impl DifferentiableSplineBasis1d for OpenUniformSplineBasis {
    fn max_derivative_order(&self) -> Option<usize> {
        Some(self.order().degree())
    }

    fn for_each_basis_derivative(
        &self,
        x: f64,
        derivative_order: usize,
        mut f: impl FnMut(usize, f64),
    ) -> Result<(), SplineError> {
        validate_coordinate_and_order(self, x, derivative_order)?;
        if derivative_order == 0 {
            return SplineBasis1d::for_each_basis(self, x, f);
        }
        let span = self.max() - self.min();
        let u = (x - self.min()) / span;
        #[allow(clippy::cast_precision_loss)]
        let n_intervals = (self.n_basis() - self.order().degree()) as f64;
        if u <= 0.0 || u >= 1.0 {
            if derivative_order == 1 {
                open_uniform_local_basis_derivative(u, self.order(), self.n_basis(), n_intervals)
                    .for_each(|index, value| f(index, value / span));
            }
            return Ok(());
        }
        let scale = inverse_power(span, derivative_order);
        for index in 0..self.n_basis() {
            let value = scale
                * open_uniform_derivative_value(
                    index,
                    self.order().degree(),
                    derivative_order,
                    u,
                    self.n_basis(),
                    self.order().degree(),
                );
            if value != 0.0 {
                f(index, value);
            }
        }
        Ok(())
    }
}

impl DifferentiableSplineBasis1d for CyclicSplineSpec {
    fn max_derivative_order(&self) -> Option<usize> {
        Some(self.order().degree())
    }

    fn for_each_basis_derivative(
        &self,
        x: f64,
        derivative_order: usize,
        f: impl FnMut(usize, f64),
    ) -> Result<(), SplineError> {
        validate_coordinate_and_order(self, x, derivative_order)?;
        if derivative_order == 0 {
            return SplineBasis1d::for_each_basis(self, x, f);
        }
        cyclic_local_basis_derivative_order(x, self.order(), self.n_basis(), derivative_order)
            .for_each(f);
        Ok(())
    }
}

impl DifferentiableSplineBasis1d for PeriodicSplineSpec {
    fn max_derivative_order(&self) -> Option<usize> {
        Some(self.order().degree())
    }

    fn for_each_basis_derivative(
        &self,
        x: f64,
        derivative_order: usize,
        mut f: impl FnMut(usize, f64),
    ) -> Result<(), SplineError> {
        validate_coordinate_and_order(self, x, derivative_order)?;
        if derivative_order == 0 {
            return SplineBasis1d::for_each_basis(self, x, f);
        }
        let phase = (x - self.origin()) / self.period();
        let scale = inverse_power(self.period(), derivative_order);
        self.cyclic_spec()
            .for_each_basis_derivative(phase, derivative_order, |index, value| {
                f(index, scale * value);
            })
    }
}

impl DifferentiableSplineBasis1d for FourierBasis {
    fn max_derivative_order(&self) -> Option<usize> {
        None
    }

    fn for_each_basis_derivative(
        &self,
        x: f64,
        derivative_order: usize,
        mut f: impl FnMut(usize, f64),
    ) -> Result<(), SplineError> {
        self.validate_coordinate(x)?;
        if derivative_order == 0 {
            return SplineBasis1d::for_each_basis(self, x, f);
        }
        let omega = std::f64::consts::TAU / self.period();
        let base_phase = omega * x;
        let offset = usize::from(self.include_intercept());
        for harmonic in 1..=self.order() {
            #[allow(clippy::cast_precision_loss)]
            let frequency = harmonic as f64 * omega;
            let magnitude = pow_usize(frequency, derivative_order);
            if !magnitude.is_finite() {
                return Err(SplineError::NonFiniteValue);
            }
            #[allow(clippy::cast_precision_loss)]
            let phase = harmonic as f64 * base_phase;
            let (sin, cos) = phase.sin_cos();
            let (sin_derivative, cos_derivative) = match derivative_order % 4 {
                0 => (sin, cos),
                1 => (cos, -sin),
                2 => (-sin, -cos),
                _ => (-cos, sin),
            };
            let index = offset + 2 * (harmonic - 1);
            let sin_value = magnitude * sin_derivative;
            let cos_value = magnitude * cos_derivative;
            if sin_value != 0.0 {
                f(index, sin_value);
            }
            if cos_value != 0.0 {
                f(index + 1, cos_value);
            }
        }
        Ok(())
    }
}

impl DifferentiableSplineBasis1d for NaturalCubicSplineBasis {
    fn max_derivative_order(&self) -> Option<usize> {
        Some(3)
    }

    fn for_each_basis_derivative(
        &self,
        x: f64,
        derivative_order: usize,
        mut f: impl FnMut(usize, f64),
    ) -> Result<(), SplineError> {
        validate_coordinate_and_order(self, x, derivative_order)?;
        match derivative_order {
            0 => SplineBasis1d::for_each_basis(self, x, f),
            1 => {
                self.for_each_derivative_basis(x, f);
                Ok(())
            }
            2 | 3 => {
                let knots = self.knots();
                if x <= knots[0] || x >= knots[knots.len() - 1] {
                    return Ok(());
                }
                let interval = knots.partition_point(|knot| *knot <= x) - 1;
                let left = knots[interval];
                let right = knots[interval + 1];
                let width = right - left;
                let left_column = self.second_derivative_column(interval);
                let right_column = self.second_derivative_column(interval + 1);
                for index in 0..self.n_basis() {
                    let value = if derivative_order == 2 {
                        (x - left).mul_add(right_column[index], (right - x) * left_column[index])
                            / width
                    } else {
                        (right_column[index] - left_column[index]) / width
                    };
                    if value != 0.0 {
                        f(index, value);
                    }
                }
                Ok(())
            }
            _ => unreachable!(),
        }
    }
}

impl DifferentiableSplineBasis1d for TruncatedPowerBasis {
    fn max_derivative_order(&self) -> Option<usize> {
        Some(self.order().degree())
    }

    fn for_each_basis_derivative(
        &self,
        x: f64,
        derivative_order: usize,
        mut f: impl FnMut(usize, f64),
    ) -> Result<(), SplineError> {
        validate_coordinate_and_order(self, x, derivative_order)?;
        if derivative_order == 0 {
            return SplineBasis1d::for_each_basis(self, x, f);
        }
        let degree = self.order().degree();
        let offset = usize::from(self.include_intercept());
        for power in derivative_order..=degree {
            let value =
                falling_factorial(power, derivative_order) * pow_usize(x, power - derivative_order);
            if value != 0.0 {
                f(offset + power - 1, value);
            }
        }
        let knot_offset = offset + degree;
        let active = self.knots().partition_point(|knot| *knot <= x);
        let scale = falling_factorial(degree, derivative_order);
        for (index, knot) in self.knots()[..active].iter().copied().enumerate() {
            let value = scale * pow_usize(x - knot, degree - derivative_order);
            if value != 0.0 {
                f(knot_offset + index, value);
            }
        }
        Ok(())
    }
}

/// Evaluates one arbitrary-knot B-spline derivative recursively.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn bspline_derivative_value(
    knots: &[f64],
    n_basis: usize,
    index: usize,
    degree: usize,
    derivative_order: usize,
    x: f64,
) -> f64 {
    spline_derivative_value(
        index,
        degree,
        derivative_order,
        x,
        &|knot| knots[knot],
        &|basis, basis_degree, coordinate| {
            bspline_value(knots, n_basis, basis, basis_degree, coordinate)
        },
    )
}

fn validate_coordinate_and_order<B>(
    basis: &B,
    x: f64,
    derivative_order: usize,
) -> Result<(), SplineError>
where
    B: DifferentiableSplineBasis1d + ?Sized,
{
    basis.validate_coordinate(x)?;
    if let Some(max) = basis.max_derivative_order()
        && derivative_order > max
    {
        return Err(SplineError::UnsupportedDerivativeOrder {
            requested: derivative_order,
            max,
        });
    }
    Ok(())
}

fn validate_output_len(expected: usize, actual: usize) -> Result<(), SplineError> {
    if expected == actual {
        Ok(())
    } else {
        Err(ModelError::DesignSize {
            expected_values: expected,
            actual_values: actual,
        }
        .into())
    }
}

fn open_uniform_derivative_value(
    index: usize,
    degree: usize,
    derivative_order: usize,
    u: f64,
    n_basis: usize,
    spline_degree: usize,
) -> f64 {
    spline_derivative_value(
        index,
        degree,
        derivative_order,
        u,
        &|knot| open_uniform_knot(knot, n_basis, spline_degree),
        &|basis, basis_degree, coordinate| {
            open_uniform_basis_value(basis, basis_degree, coordinate, n_basis, spline_degree)
        },
    )
}

#[allow(clippy::cast_precision_loss)]
fn spline_derivative_value<K, B>(
    index: usize,
    degree: usize,
    derivative_order: usize,
    x: f64,
    knot: &K,
    basis_value: &B,
) -> f64
where
    K: Fn(usize) -> f64,
    B: Fn(usize, usize, f64) -> f64,
{
    if derivative_order == 0 {
        return basis_value(index, degree, x);
    }
    debug_assert!(derivative_order <= degree);
    let mut value = 0.0;
    let degree_scale = degree as f64;
    let left_denominator = knot(index + degree) - knot(index);
    if left_denominator > 0.0 {
        value = (degree_scale / left_denominator).mul_add(
            spline_derivative_value(
                index,
                degree - 1,
                derivative_order - 1,
                x,
                knot,
                basis_value,
            ),
            value,
        );
    }
    let right_denominator = knot(index + degree + 1) - knot(index + 1);
    if right_denominator > 0.0 {
        value = (-degree_scale / right_denominator).mul_add(
            spline_derivative_value(
                index + 1,
                degree - 1,
                derivative_order - 1,
                x,
                knot,
                basis_value,
            ),
            value,
        );
    }
    value
}

fn open_uniform_basis_value(
    index: usize,
    degree: usize,
    u: f64,
    n_basis: usize,
    spline_degree: usize,
) -> f64 {
    if degree == 0 {
        let left = open_uniform_knot(index, n_basis, spline_degree);
        let right = open_uniform_knot(index + 1, n_basis, spline_degree);
        return f64::from(left <= u && u < right);
    }
    let left_knot = open_uniform_knot(index, n_basis, spline_degree);
    let left_end = open_uniform_knot(index + degree, n_basis, spline_degree);
    let right_start = open_uniform_knot(index + 1, n_basis, spline_degree);
    let right_knot = open_uniform_knot(index + degree + 1, n_basis, spline_degree);
    let mut value = 0.0;
    if left_end > left_knot {
        value = ((u - left_knot) / (left_end - left_knot)).mul_add(
            open_uniform_basis_value(index, degree - 1, u, n_basis, spline_degree),
            value,
        );
    }
    if right_knot > right_start {
        value = ((right_knot - u) / (right_knot - right_start)).mul_add(
            open_uniform_basis_value(index + 1, degree - 1, u, n_basis, spline_degree),
            value,
        );
    }
    value
}

#[allow(clippy::cast_precision_loss)]
fn open_uniform_knot(index: usize, n_basis: usize, degree: usize) -> f64 {
    if index <= degree {
        0.0
    } else if index >= n_basis {
        1.0
    } else {
        (index - degree) as f64 / (n_basis - degree) as f64
    }
}

#[allow(clippy::cast_precision_loss)]
fn falling_factorial(value: usize, count: usize) -> f64 {
    (0..count).fold(1.0, |product, offset| product * (value - offset) as f64)
}

fn inverse_power(value: f64, exponent: usize) -> f64 {
    1.0 / pow_usize(value, exponent)
}

fn pow_usize(mut base: f64, mut exponent: usize) -> f64 {
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

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::DifferentiableSplineBasis1d;
    use crate::{
        BSplineBasis, FourierBasis, ISplineBasis, NaturalCubicSplineBasis, OpenUniformSplineBasis,
        PeriodicSplineSpec, SplineOrder, TruncatedPowerBasis,
    };

    #[test]
    fn common_derivatives_match_centered_finite_differences() {
        let x = [-1.0, -0.3, 0.2, 0.8, 1.4];
        assert_first_derivative(
            &BSplineBasis::open_uniform_from_data(&x, 7, 3).unwrap(),
            0.17,
        );
        assert_first_derivative(
            &OpenUniformSplineBasis::from_data(&x, 7, SplineOrder::Cubic).unwrap(),
            0.17,
        );
        assert_first_derivative(
            &ISplineBasis::open_uniform_from_data(&x, 7, 3).unwrap(),
            0.17,
        );
        assert_first_derivative(
            &NaturalCubicSplineBasis::uniform_from_data(&x, 6).unwrap(),
            0.17,
        );
        assert_first_derivative(
            &TruncatedPowerBasis::uniform_from_data(&x, 3, SplineOrder::Cubic, true).unwrap(),
            0.17,
        );
        assert_first_derivative(&FourierBasis::new(3.0, 3, true).unwrap(), 0.17);
        assert_first_derivative(
            &PeriodicSplineSpec::new(7, SplineOrder::Cubic, 3.0, -1.0).unwrap(),
            0.17,
        );
    }

    fn assert_first_derivative<B>(basis: &B, x: f64)
    where
        B: DifferentiableSplineBasis1d,
    {
        let step = 1.0e-6;
        let lower = basis.evaluate(x - step).unwrap();
        let upper = basis.evaluate(x + step).unwrap();
        let derivative = basis.evaluate_basis_derivative(x, 1).unwrap();
        for ((lower, upper), actual) in lower.iter().zip(&upper).zip(&derivative) {
            assert_relative_eq!(*actual, (upper - lower) / (2.0 * step), epsilon = 3.0e-6);
        }
    }

    #[test]
    fn higher_derivatives_are_consistent() {
        let basis = BSplineBasis::open_uniform_from_data(&[0.0, 0.5, 1.0], 7, 3).unwrap();
        let step = 1.0e-5;
        let lower = basis.evaluate_basis_derivative(0.37 - step, 1).unwrap();
        let upper = basis.evaluate_basis_derivative(0.37 + step, 1).unwrap();
        let second = basis.evaluate_basis_derivative(0.37, 2).unwrap();
        for ((lower, upper), actual) in lower.iter().zip(&upper).zip(&second) {
            assert_relative_eq!(*actual, (upper - lower) / (2.0 * step), epsilon = 2.0e-4);
        }
    }

    #[test]
    fn truncated_power_uses_right_hand_derivative_at_a_knot() {
        let basis = TruncatedPowerBasis::new(vec![0.0], SplineOrder::Cubic, true).unwrap();
        let third = basis.evaluate_basis_derivative(0.0, 3).unwrap();
        assert_relative_eq!(third[4], 6.0, epsilon = f64::EPSILON);
    }
}
