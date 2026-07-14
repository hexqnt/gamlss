//! Static algebra of compiled distribution-parameter shapes.

use std::marker::PhantomData;

mod sealed {
    pub trait Sealed {}
}

/// One scalar predictor coordinate with semantic role `P`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Scalar<P>(PhantomData<P>);

impl<P> ParameterShape for Scalar<P> {
    type Values = f64;

    #[inline]
    fn zeros() -> Self::Values {
        0.0
    }

    fn scale(values: &mut Self::Values, factor: f64) {
        if factor == 0.0 {
            *values = 0.0;
        } else {
            *values *= factor;
        }
    }
    fn add_assign(target: &mut Self::Values, source: &Self::Values) {
        *target += *source;
    }
    fn add_to_first(values: &mut Self::Values, delta: f64) {
        *values += delta;
    }
}

impl<P> sealed::Sealed for Scalar<P> {}
/// Ergonomic flat tuple of scalar coordinates.
///
/// This is intentionally a primitive alongside recursive [`Product`]: ordinary
/// univariate families should not expose deeply nested binary tuple types.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScalarTuple<Params, const K: usize>(PhantomData<Params>);

impl<Params, const K: usize> ParameterShape for ScalarTuple<Params, K> {
    type Values = [f64; K];

    #[inline]
    fn zeros() -> Self::Values {
        [0.0; K]
    }

    fn scale(values: &mut Self::Values, factor: f64) {
        if factor == 0.0 {
            values.fill(0.0);
        } else {
            for value in values.iter_mut() {
                *value *= factor;
            }
        }
    }
    fn add_assign(target: &mut Self::Values, source: &Self::Values) {
        target
            .iter_mut()
            .zip(source)
            .for_each(|(target, source)| *target += *source);
    }
    fn add_to_first(values: &mut Self::Values, delta: f64) {
        if let Some(first) = values.first_mut() {
            *first += delta;
        }
    }
}

impl<Params, const K: usize> sealed::Sealed for ScalarTuple<Params, K> {}
/// `D` independent scalar coordinates with the same semantic role.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Vector<P, const D: usize>(PhantomData<P>);

impl<P, const D: usize> ParameterShape for Vector<P, D> {
    type Values = [f64; D];

    #[inline]
    fn zeros() -> Self::Values {
        [0.0; D]
    }

    fn scale(values: &mut Self::Values, factor: f64) {
        if factor == 0.0 {
            values.fill(0.0);
        } else {
            for value in values.iter_mut() {
                *value *= factor;
            }
        }
    }
    fn add_assign(target: &mut Self::Values, source: &Self::Values) {
        target
            .iter_mut()
            .zip(source)
            .for_each(|(target, source)| *target += *source);
    }
    fn add_to_first(values: &mut Self::Values, delta: f64) {
        if let Some(first) = values.first_mut() {
            *first += delta;
        }
    }
}

impl<P, const D: usize> sealed::Sealed for Vector<P, D> {}
/// Row-major lower-triangular scalar coordinates.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Lower<P, const D: usize>(PhantomData<P>);

impl<P, const D: usize> ParameterShape for Lower<P, D> {
    type Values = [[f64; D]; D];

    #[inline]
    fn zeros() -> Self::Values {
        [[0.0; D]; D]
    }

    fn scale(values: &mut Self::Values, factor: f64) {
        if factor == 0.0 {
            *values = Self::zeros();
        } else {
            values
                .iter_mut()
                .flatten()
                .for_each(|value| *value *= factor);
        }
    }
    fn add_assign(target: &mut Self::Values, source: &Self::Values) {
        for (target_row, source_row) in target.iter_mut().zip(source) {
            target_row
                .iter_mut()
                .zip(source_row)
                .for_each(|(target, source)| *target += *source);
        }
    }
    fn add_to_first(values: &mut Self::Values, delta: f64) {
        if D > 0 {
            values[0][0] += delta;
        }
    }
}

impl<P, const D: usize> sealed::Sealed for Lower<P, D> {}
/// Row-major strict-lower-triangular scalar coordinates.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StrictLower<P, const D: usize>(PhantomData<P>);

impl<P, const D: usize> ParameterShape for StrictLower<P, D> {
    type Values = [[f64; D]; D];

    #[inline]
    fn zeros() -> Self::Values {
        [[0.0; D]; D]
    }

    fn scale(values: &mut Self::Values, factor: f64) {
        if factor == 0.0 {
            *values = Self::zeros();
        } else {
            values
                .iter_mut()
                .flatten()
                .for_each(|value| *value *= factor);
        }
    }
    fn add_assign(target: &mut Self::Values, source: &Self::Values) {
        for (target_row, source_row) in target.iter_mut().zip(source) {
            target_row
                .iter_mut()
                .zip(source_row)
                .for_each(|(target, source)| *target += *source);
        }
    }
    fn add_to_first(values: &mut Self::Values, delta: f64) {
        if D > 1 {
            values[1][0] += delta;
        }
    }
}

impl<P, const D: usize> sealed::Sealed for StrictLower<P, D> {}
/// `C - 1` free baseline-softmax logits stored in a `C`-value carrier.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Simplex<P, const C: usize>(PhantomData<P>);

impl<P, const C: usize> ParameterShape for Simplex<P, C> {
    type Values = [f64; C];

    #[inline]
    fn zeros() -> Self::Values {
        [0.0; C]
    }

    fn scale(values: &mut Self::Values, factor: f64) {
        if factor == 0.0 {
            values.fill(0.0);
        } else {
            for value in values.iter_mut() {
                *value *= factor;
            }
        }
    }
    fn add_assign(target: &mut Self::Values, source: &Self::Values) {
        target
            .iter_mut()
            .zip(source)
            .for_each(|(target, source)| *target += *source);
    }
    fn add_to_first(values: &mut Self::Values, delta: f64) {
        if C > 1 {
            values[0] += delta;
        }
    }
}

impl<P, const C: usize> sealed::Sealed for Simplex<P, C> {}
/// Static product of two independent shape subtrees.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Product<A, B>(PhantomData<(A, B)>);

impl<A: ParameterShape, B: ParameterShape> ParameterShape for Product<A, B> {
    type Values = (A::Values, B::Values);

    #[inline]
    fn zeros() -> Self::Values {
        (A::zeros(), B::zeros())
    }

    fn scale(values: &mut Self::Values, factor: f64) {
        A::scale(&mut values.0, factor);
        B::scale(&mut values.1, factor);
    }
    fn add_assign(target: &mut Self::Values, source: &Self::Values) {
        A::add_assign(&mut target.0, &source.0);
        B::add_assign(&mut target.1, &source.1);
    }
    fn add_to_first(values: &mut Self::Values, delta: f64) {
        A::add_to_first(&mut values.0, delta);
    }
}

impl<A: ParameterShape, B: ParameterShape> sealed::Sealed for Product<A, B> {}
/// `C` independent repetitions of one shape subtree.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Repeated<A, const C: usize>(PhantomData<A>);

impl<A: ParameterShape, const C: usize> ParameterShape for Repeated<A, C> {
    type Values = [A::Values; C];

    #[inline]
    fn zeros() -> Self::Values {
        std::array::from_fn(|_| A::zeros())
    }

    fn scale(values: &mut Self::Values, factor: f64) {
        for value in values.iter_mut() {
            A::scale(value, factor);
        }
    }
    fn add_assign(target: &mut Self::Values, source: &Self::Values) {
        target
            .iter_mut()
            .zip(source)
            .for_each(|(target, source)| A::add_assign(target, source));
    }
    fn add_to_first(values: &mut Self::Values, delta: f64) {
        if let Some(first) = values.first_mut() {
            A::add_to_first(first, delta);
        }
    }
}

impl<A: ParameterShape, const C: usize> sealed::Sealed for Repeated<A, C> {}
/// One coefficient-owning subtree broadcast to `C` distribution consumers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Broadcast<A, const C: usize>(PhantomData<A>);

impl<A: ParameterShape, const C: usize> ParameterShape for Broadcast<A, C> {
    type Values = [A::Values; C];

    #[inline]
    fn zeros() -> Self::Values {
        std::array::from_fn(|_| A::zeros())
    }

    fn scale(values: &mut Self::Values, factor: f64) {
        for value in values.iter_mut() {
            A::scale(value, factor);
        }
    }
    fn add_assign(target: &mut Self::Values, source: &Self::Values) {
        target
            .iter_mut()
            .zip(source)
            .for_each(|(target, source)| A::add_assign(target, source));
    }
    fn add_to_first(values: &mut Self::Values, delta: f64) {
        if let Some(first) = values.first_mut() {
            A::add_to_first(first, delta);
        }
    }
}
impl<A: ParameterShape, const C: usize> sealed::Sealed for Broadcast<A, C> {}
/// Geometry of link-scale predictor coordinates used by a compiled family.
///
/// Shapes contain no distribution mathematics and no link functions. They only
/// describe the recursively typed scalar leaves consumed by the model executor.
pub trait ParameterShape: sealed::Sealed {
    /// Materialized scalar values in the topology of this shape.
    type Values: Clone;

    /// Creates a zero-valued shape, used as a conservative initializer.
    fn zeros() -> Self::Values;

    /// Multiplies every scalar coordinate in `values` by `factor`.
    ///
    /// A zero factor must overwrite every coordinate with exact zero without
    /// reading it. This keeps inactive mixture/shared branches from propagating
    /// non-finite scores through `0 * NaN`.
    fn scale(values: &mut Self::Values, factor: f64);

    /// Adds every scalar coordinate in `source` to `target`.
    fn add_assign(target: &mut Self::Values, source: &Self::Values);

    /// Adds `delta` to the first scalar coordinate, if the shape is non-empty.
    fn add_to_first(values: &mut Self::Values, delta: f64);
}

#[cfg(test)]
mod tests {
    use crate::{Mu, Sigma};

    use super::{Broadcast, Lower, ParameterShape, Product, Scalar, Simplex, StrictLower, Vector};

    #[test]
    fn zero_scaling_overwrites_non_finite_values_across_nested_shapes() {
        type Shape =
            Product<Lower<Mu, 2>, Product<StrictLower<Sigma, 2>, Broadcast<Vector<Mu, 2>, 2>>>;
        let mut values = (
            [[f64::NAN; 2]; 2],
            ([[f64::INFINITY; 2]; 2], [[f64::NEG_INFINITY; 2]; 2]),
        );

        Shape::scale(&mut values, 0.0);

        assert!(
            values
                .0
                .iter()
                .flatten()
                .all(|value| value.abs() <= f64::EPSILON)
        );
        assert!(
            values
                .1
                .0
                .iter()
                .flatten()
                .all(|value| value.abs() <= f64::EPSILON)
        );
        assert!(
            values
                .1
                .1
                .iter()
                .flatten()
                .all(|value| value.abs() <= f64::EPSILON)
        );
    }

    #[test]
    fn zero_scaling_overwrites_non_finite_scalar_and_simplex_values() {
        let mut scalar = f64::NAN;
        let mut simplex = [f64::NAN; 3];

        Scalar::<Mu>::scale(&mut scalar, 0.0);
        Simplex::<Sigma, 3>::scale(&mut simplex, 0.0);

        assert!(scalar.abs() <= f64::EPSILON);
        assert!(simplex.iter().all(|value| value.abs() <= f64::EPSILON));
    }
}
