use std::{marker::PhantomData, ops::Range};

use crate::{DesignMatrix, LinearPredictorBlock, ModelError, PredictorBlock};

/// Stable public name for a distribution parameter marker.
pub trait ParameterName {
    /// Name used in parameter layouts and unpacked coefficient views.
    const NAME: &'static str;
}

/// Helper for assigning sequential offsets to typed parameter block tuples.
///
/// This is the safe construction path for ordinary models: create each
/// [`ParameterBlock`] with any placeholder offset, then call
/// `ParameterBlocks::new((...))` to lay the tuple out from zero. Low-level
/// constructors that accept explicit offsets remain available for advanced
/// layouts and integration code.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ParameterBlocks;

/// Tuple contract implemented for typed parameter block tuples up to arity 8.
pub trait AssignParameterOffsets: Sized {
    /// Returns `self` with sequential offsets starting at `start`.
    #[must_use]
    fn assign_offsets(self, start: usize) -> Self;
}

impl ParameterBlocks {
    /// Assigns sequential offsets starting at zero.
    #[allow(clippy::new_ret_no_self)]
    #[must_use]
    #[inline]
    pub fn new<Blocks>(blocks: Blocks) -> Blocks
    where
        Blocks: AssignParameterOffsets,
    {
        Self::with_start(0, blocks)
    }

    /// Assigns sequential offsets starting at `start`.
    #[must_use]
    #[inline]
    pub fn with_start<Blocks>(start: usize, blocks: Blocks) -> Blocks
    where
        Blocks: AssignParameterOffsets,
    {
        blocks.assign_offsets(start)
    }
}

/// Marker for the location parameter `mu`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Mu;

/// Marker for the scale parameter `sigma`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Sigma;

/// Marker for the third GAMLSS parameter `nu`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Nu;

/// Marker for the fourth GAMLSS parameter `tau`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Tau;

/// Marker for the rate parameter of a distribution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Rate;

/// Marker for the shape parameter of a distribution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Shape;

/// Marker for the scale parameter of a distribution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Scale;

/// Marker for the precision parameter of a distribution.
///
/// Used for mean/precision parameterizations, e.g. the beta distribution, where
/// `precision > 0` controls the concentration around the mean.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Precision;

impl ParameterName for Mu {
    const NAME: &'static str = "mu";
}

impl ParameterName for Sigma {
    const NAME: &'static str = "sigma";
}

impl ParameterName for Nu {
    const NAME: &'static str = "nu";
}

impl ParameterName for Tau {
    const NAME: &'static str = "tau";
}

impl ParameterName for Rate {
    const NAME: &'static str = "rate";
}

impl ParameterName for Shape {
    const NAME: &'static str = "shape";
}

impl ParameterName for Scale {
    const NAME: &'static str = "scale";
}

impl ParameterName for Precision {
    const NAME: &'static str = "precision";
}

/// Typed coefficient block for a single distribution parameter.
///
/// `P` specifies the parameter role, `L` specifies the link function, `X` holds
/// the predictor block, and `Penalty` adds regularization. `offset` and `len`
/// describe the coefficient range of the block within the common beta vector.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterBlock<P, L, X, Penalty> {
    /// Predictor block.
    pub x: X,
    /// Penalty applied to the block's coefficients.
    pub penalty: Penalty,
    /// Start position of the block in the common beta vector.
    pub offset: usize,
    /// Number of coefficients in the block.
    pub len: usize,
    marker: PhantomData<(P, L)>,
}

impl<P, L, X, Penalty> ParameterBlock<P, L, X, Penalty>
where
    X: PredictorBlock,
{
    /// Creates a block, taking `len` from `x.nparams()`.
    #[must_use]
    #[inline]
    pub fn new(x: X, penalty: Penalty, offset: usize) -> Self {
        let len = x.nparams();
        Self::from_len(x, penalty, offset, len)
    }

    /// Creates a block from a generic predictor.
    ///
    /// This is a synonym for [`Self::new`], kept for code where the explicit
    /// `predictor` word makes the call more readable.
    #[must_use]
    #[inline]
    pub fn from_predictor(x: X, penalty: Penalty, offset: usize) -> Self {
        Self::new(x, penalty, offset)
    }
}

impl<P, L, X, Penalty> ParameterBlock<P, L, LinearPredictorBlock<X>, Penalty>
where
    X: DesignMatrix,
{
    /// Creates a linear block from a design matrix.
    #[must_use]
    #[inline]
    pub fn linear(x: X, penalty: Penalty, offset: usize) -> Self {
        Self::new(LinearPredictorBlock::new(x), penalty, offset)
    }
}

impl<P, L, X, Penalty> ParameterBlock<P, L, X, Penalty> {
    #[inline]
    fn from_len(x: X, penalty: Penalty, offset: usize, len: usize) -> Self {
        Self {
            x,
            penalty,
            offset,
            len,
            marker: PhantomData,
        }
    }

    /// Returns a copy of the block with a new offset.
    #[must_use]
    #[inline]
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Coefficient range of the block in the common beta vector.
    ///
    /// # Panics
    ///
    /// Panics if `offset + len` overflows. Use [`Self::try_range`] when the
    /// offset may come from unchecked external input.
    #[must_use]
    #[inline]
    pub fn range(&self) -> Range<usize> {
        self.offset..self.end()
    }

    /// Index immediately after the last coefficient of the block.
    ///
    /// # Panics
    ///
    /// Panics if `offset + len` overflows. Use [`Self::try_range`] for
    /// recoverable validation.
    #[must_use]
    #[inline]
    pub fn end(&self) -> usize {
        self.offset
            .checked_add(self.len)
            .expect("parameter block range end must fit in usize")
    }

    /// Number of coefficients in the block.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if the block contains no coefficients.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<P, L, X, Penalty> ParameterBlock<P, L, X, Penalty>
where
    P: ParameterName,
{
    /// Validates and returns the block's coefficient range.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::BlockRangeOverflow`] if `offset + len` does not fit
    /// in `usize`.
    #[inline]
    pub fn try_range(&self) -> Result<Range<usize>, ModelError> {
        let end = self
            .offset
            .checked_add(self.len)
            .ok_or(ModelError::BlockRangeOverflow {
                parameter: P::NAME,
                offset: self.offset,
                len: self.len,
            })?;
        Ok(self.offset..end)
    }
}

macro_rules! impl_assign_offsets {
    (
        types = ($($block:ident),+);
        vars = ($($var:ident),+)
    ) => {
        impl<$($block,)+> AssignParameterOffsets for ($($block,)+)
        where
            $($block: OffsetAssignable,)+
        {
            #[inline]
            fn assign_offsets(self, start: usize) -> Self {
                let ($($var,)+) = self;
                let mut offset = start;
                $(
                    let $var = $var.with_assigned_offset(offset);
                    offset = offset.saturating_add($var.assigned_len());
                )+
                let _ = offset;
                ($($var,)+)
            }
        }
    };
}

trait OffsetAssignable: Sized {
    fn with_assigned_offset(self, offset: usize) -> Self;
    fn assigned_len(&self) -> usize;
}

impl<P, L, X, Penalty> OffsetAssignable for ParameterBlock<P, L, X, Penalty> {
    fn with_assigned_offset(self, offset: usize) -> Self {
        self.with_offset(offset)
    }

    fn assigned_len(&self) -> usize {
        self.len()
    }
}

impl_assign_offsets!(types = (B1); vars = (b1));
impl_assign_offsets!(types = (B1, B2); vars = (b1, b2));
impl_assign_offsets!(types = (B1, B2, B3); vars = (b1, b2, b3));
impl_assign_offsets!(types = (B1, B2, B3, B4); vars = (b1, b2, b3, b4));
impl_assign_offsets!(types = (B1, B2, B3, B4, B5); vars = (b1, b2, b3, b4, b5));
impl_assign_offsets!(types = (B1, B2, B3, B4, B5, B6); vars = (b1, b2, b3, b4, b5, b6));
impl_assign_offsets!(
    types = (B1, B2, B3, B4, B5, B6, B7);
    vars = (b1, b2, b3, b4, b5, b6, b7)
);
impl_assign_offsets!(
    types = (B1, B2, B3, B4, B5, B6, B7, B8);
    vars = (b1, b2, b3, b4, b5, b6, b7, b8)
);

#[cfg(test)]
mod tests {
    use crate::{DenseDesign, Identity, LinearPredictorBlock, NoPenalty};

    use super::{
        Mu, Nu, ParameterBlock, ParameterBlocks, Precision, Rate, Scale, Shape, Sigma, Tau,
    };

    #[test]
    fn parameter_blocks_assign_offsets_for_one_block() {
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(
            DenseDesign::from_rows(&[[1.0, 2.0]]),
            NoPenalty,
            99,
        );

        let (mu,) = ParameterBlocks::new((mu,));

        assert_eq!(mu.range(), 0..2);
    }

    #[test]
    fn parameter_blocks_assign_offsets_for_two_blocks() {
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(
            DenseDesign::from_rows(&[[1.0, 2.0]]),
            NoPenalty,
            99,
        );
        let sigma = ParameterBlock::<Sigma, Identity, _, _>::linear(
            DenseDesign::from_rows(&[[1.0, 2.0, 3.0]]),
            NoPenalty,
            99,
        );

        let (mu, sigma) = ParameterBlocks::new((mu, sigma));

        assert_eq!(mu.range(), 0..2);
        assert_eq!(sigma.range(), 2..5);
    }

    #[test]
    fn parameter_blocks_assign_offsets_for_eight_blocks_with_start() {
        let blocks = (
            intercept_block::<Mu>(),
            intercept_block::<Sigma>(),
            intercept_block::<Nu>(),
            intercept_block::<Tau>(),
            intercept_block::<Shape>(),
            intercept_block::<Scale>(),
            intercept_block::<Rate>(),
            intercept_block::<Precision>(),
        );

        let (b1, b2, b3, b4, b5, b6, b7, b8) = ParameterBlocks::with_start(10, blocks);

        assert_eq!(b1.range(), 10..11);
        assert_eq!(b2.range(), 11..12);
        assert_eq!(b3.range(), 12..13);
        assert_eq!(b4.range(), 13..14);
        assert_eq!(b5.range(), 14..15);
        assert_eq!(b6.range(), 15..16);
        assert_eq!(b7.range(), 16..17);
        assert_eq!(b8.range(), 17..18);
    }

    #[test]
    fn parameter_block_try_range_reports_overflow() {
        let block = ParameterBlock::<Mu, Identity, _, _>::linear(
            DenseDesign::from_rows(&[[1.0, 2.0]]),
            NoPenalty,
            usize::MAX,
        );

        assert_eq!(
            block.try_range().unwrap_err(),
            crate::ModelError::BlockRangeOverflow {
                parameter: "mu",
                offset: usize::MAX,
                len: 2,
            }
        );
    }

    fn intercept_block<P>()
    -> ParameterBlock<P, Identity, LinearPredictorBlock<DenseDesign>, NoPenalty> {
        ParameterBlock::linear(DenseDesign::intercept(1), NoPenalty, 99)
    }
}
