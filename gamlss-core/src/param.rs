use std::{marker::PhantomData, ops::Range};

use crate::{DesignMatrix, LinearPredictorBlock, ModelError, PredictorBlock};

/// Helper for assigning sequential offsets to typed parameter block tuples.
///
/// This is the safe construction path for ordinary models: create each
/// [`ParameterBlock`] with any placeholder offset, then call
/// `ParameterBlocks::new((...))` to lay the tuple out from zero. Low-level
/// constructors that accept explicit offsets remain available for advanced
/// layouts and integration code.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ParameterBlocks;

impl ParameterBlocks {
    /// Assigns sequential offsets starting at zero.
    ///
    /// # Panics
    ///
    /// Panics if the sequential layout does not fit in `usize`. Use
    /// [`Self::try_new`] when block sizes may come from unchecked external
    /// input.
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
    ///
    /// # Panics
    ///
    /// Panics if the sequential layout does not fit in `usize`. Use
    /// [`Self::try_with_start`] when block sizes may come from unchecked
    /// external input.
    #[must_use]
    #[inline]
    pub fn with_start<Blocks>(start: usize, blocks: Blocks) -> Blocks
    where
        Blocks: AssignParameterOffsets,
    {
        blocks.assign_offsets(start)
    }

    /// Assigns sequential offsets starting at zero.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::BlockRangeOverflow`] if any assigned block range
    /// would not fit in `usize`.
    #[allow(clippy::new_ret_no_self)]
    #[inline]
    pub fn try_new<Blocks>(blocks: Blocks) -> Result<Blocks, ModelError>
    where
        Blocks: TryAssignParameterOffsets,
    {
        Self::try_with_start(0, blocks)
    }

    /// Assigns sequential offsets starting at `start`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::BlockRangeOverflow`] if any assigned block range
    /// would not fit in `usize`.
    #[inline]
    pub fn try_with_start<Blocks>(start: usize, blocks: Blocks) -> Result<Blocks, ModelError>
    where
        Blocks: TryAssignParameterOffsets,
    {
        blocks.try_assign_offsets(start)
    }
}

/// Marker for the location parameter `mu`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Mu;

impl ParameterName for Mu {
    const NAME: &'static str = "mu";
}

/// Marker for a mathematical mean parameter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Mean;

impl ParameterName for Mean {
    const NAME: &'static str = "mean";
}

/// Marker for a median parameter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Median;

impl ParameterName for Median {
    const NAME: &'static str = "median";
}

/// Marker for a component mean in mixture or zero-adjusted models.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ComponentMean;

impl ParameterName for ComponentMean {
    const NAME: &'static str = "component_mean";
}

/// Marker for an unconditional total mean in mixture or zero-adjusted models.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TotalMean;

impl ParameterName for TotalMean {
    const NAME: &'static str = "total_mean";
}

/// Marker for the scale parameter `sigma`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Sigma;

impl ParameterName for Sigma {
    const NAME: &'static str = "sigma";
}

/// Marker for a coefficient of variation parameter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Cv;

impl ParameterName for Cv {
    const NAME: &'static str = "cv";
}

/// Marker for a log-standard-deviation parameter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LogSd;

impl ParameterName for LogSd {
    const NAME: &'static str = "log_sd";
}

/// Marker for the location parameter of a log-scale distribution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LogLocation;

impl ParameterName for LogLocation {
    const NAME: &'static str = "log_location";
}

/// Marker for the third GAMLSS parameter `nu`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Nu;

impl ParameterName for Nu {
    const NAME: &'static str = "nu";
}

/// Marker for the fourth GAMLSS parameter `tau`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Tau;

impl ParameterName for Tau {
    const NAME: &'static str = "tau";
}

/// Marker for the rate parameter of a distribution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Rate;

impl ParameterName for Rate {
    const NAME: &'static str = "rate";
}

/// Marker for a dispersion parameter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Dispersion;

impl ParameterName for Dispersion {
    const NAME: &'static str = "dispersion";
}

/// Marker for the shape parameter of a distribution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Shape;

impl ParameterName for Shape {
    const NAME: &'static str = "shape";
}

/// Marker for a negative-binomial size parameter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Size;

impl ParameterName for Size {
    const NAME: &'static str = "size";
}

/// Marker for the scale parameter of a distribution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Scale;

impl ParameterName for Scale {
    const NAME: &'static str = "scale";
}

/// Marker for a probability parameter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Probability;

impl ParameterName for Probability {
    const NAME: &'static str = "probability";
}

/// Marker for a zero-mass or zero-inflation probability parameter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ZeroProbability;

impl ParameterName for ZeroProbability {
    const NAME: &'static str = "zero_probability";
}

/// Marker for a one-mass probability parameter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OneProbability;

impl ParameterName for OneProbability {
    const NAME: &'static str = "one_probability";
}

/// Marker for a power parameter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Power;

impl ParameterName for Power {
    const NAME: &'static str = "power";
}

/// Marker for the precision parameter of a distribution.
///
/// Used for mean/precision parameterizations, e.g. the beta distribution, where
/// `precision > 0` controls the concentration around the mean.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Precision;

impl ParameterName for Precision {
    const NAME: &'static str = "precision";
}

/// Typed coefficient block for a single distribution parameter.
///
/// `P` specifies the parameter role, `L` specifies the link function, `X` holds
/// the predictor block, and `Penalty` adds regularization. The block stores its
/// coefficient range within the common beta vector; use [`Self::range`] and
/// [`Self::len`] to inspect that layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterBlock<P, L, X, Penalty> {
    x: X,
    penalty: Penalty,
    offset: usize,
    len: usize,
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
    const fn from_len(x: X, penalty: Penalty, offset: usize, len: usize) -> Self {
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
    pub const fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Predictor block.
    #[must_use]
    #[inline]
    pub const fn x(&self) -> &X {
        &self.x
    }

    /// Penalty applied to the block's coefficients.
    #[must_use]
    #[inline]
    pub const fn penalty(&self) -> &Penalty {
        &self.penalty
    }

    #[must_use]
    #[inline]
    pub(crate) const fn offset(&self) -> usize {
        self.offset
    }

    /// Coefficient range of the block in the common beta vector.
    ///
    /// # Panics
    ///
    /// Panics if `offset + len` overflows. Use [`Self::try_range`] when the
    /// offset may come from unchecked external input.
    #[must_use]
    #[inline]
    pub const fn range(&self) -> Range<usize> {
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
    pub const fn end(&self) -> usize {
        self.offset
            .checked_add(self.len)
            .expect("parameter block range end must fit in usize")
    }

    /// Number of coefficients in the block.
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// `true` if the block contains no coefficients.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
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

/// Stable public name for a distribution parameter marker.
pub trait ParameterName {
    /// Name used in parameter layouts and unpacked coefficient views.
    const NAME: &'static str;
}

/// Tuple contract implemented for typed parameter block tuples up to arity 8.
pub trait AssignParameterOffsets: Sized {
    /// Returns `self` with sequential offsets starting at `start`.
    #[must_use]
    fn assign_offsets(self, start: usize) -> Self;
}

/// Fallible tuple contract for assigning typed parameter block offsets.
pub trait TryAssignParameterOffsets: Sized {
    /// Returns `self` with sequential offsets starting at `start`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::BlockRangeOverflow`] if a block range would not
    /// fit in `usize`.
    fn try_assign_offsets(self, start: usize) -> Result<Self, ModelError>;
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
                    let len = $var.assigned_len();
                    offset = offset
                        .checked_add(len)
                        .expect("parameter block layout must fit in usize");
                )+
                let _ = offset;
                ($($var,)+)
            }
        }

        impl<$($block,)+> TryAssignParameterOffsets for ($($block,)+)
        where
            $($block: TryOffsetAssignable,)+
        {
            #[inline]
            fn try_assign_offsets(self, start: usize) -> Result<Self, ModelError> {
                let ($($var,)+) = self;
                let mut offset = start;
                $(
                    let $var = $var.with_assigned_offset(offset);
                    let assigned_offset = $var.assigned_offset();
                    let assigned_len = $var.assigned_len();
                    offset = offset.checked_add(assigned_len).ok_or(
                        ModelError::BlockRangeOverflow {
                            parameter: $var.assigned_name(),
                            offset: assigned_offset,
                            len: assigned_len,
                        },
                    )?;
                )+
                let _ = offset;
                Ok(($($var,)+))
            }
        }
    };
}

impl<P, L, X, Penalty> OffsetAssignable for ParameterBlock<P, L, X, Penalty> {
    fn with_assigned_offset(self, offset: usize) -> Self {
        self.with_offset(offset)
    }

    fn assigned_len(&self) -> usize {
        self.len()
    }
}

trait OffsetAssignable: Sized {
    fn with_assigned_offset(self, offset: usize) -> Self;
    fn assigned_len(&self) -> usize;
}

impl<P, L, X, Penalty> TryOffsetAssignable for ParameterBlock<P, L, X, Penalty>
where
    P: ParameterName,
{
    fn with_assigned_offset(self, offset: usize) -> Self {
        self.with_offset(offset)
    }

    fn assigned_offset(&self) -> usize {
        self.offset()
    }

    fn assigned_len(&self) -> usize {
        self.len()
    }

    fn assigned_name(&self) -> &'static str {
        P::NAME
    }
}

trait TryOffsetAssignable: Sized {
    fn with_assigned_offset(self, offset: usize) -> Self;
    fn assigned_offset(&self) -> usize;
    fn assigned_len(&self) -> usize;
    fn assigned_name(&self) -> &'static str;
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

    #[test]
    fn parameter_blocks_try_with_start_reports_layout_overflow() {
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(
            DenseDesign::from_rows(&[[1.0, 2.0]]),
            NoPenalty,
            99,
        );

        assert_eq!(
            ParameterBlocks::try_with_start(usize::MAX, (mu,)).unwrap_err(),
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
