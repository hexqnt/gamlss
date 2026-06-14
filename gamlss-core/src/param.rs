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
    pub fn new<Blocks>(blocks: Blocks) -> Blocks
    where
        Blocks: AssignParameterOffsets,
    {
        Self::with_start(0, blocks)
    }

    /// Assigns sequential offsets starting at `start`.
    #[must_use]
    pub fn with_start<Blocks>(start: usize, blocks: Blocks) -> Blocks
    where
        Blocks: AssignParameterOffsets,
    {
        blocks.assign_offsets(start)
    }
}

/// Маркер для location-параметра `mu`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Mu;

/// Маркер для scale-параметра `sigma`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Sigma;

/// Маркер для третьего GAMLSS-параметра `nu`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Nu;

/// Маркер для четвёртого GAMLSS-параметра `tau`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Tau;

/// Маркер для rate-параметра распределения.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Rate;

/// Маркер для shape-параметра распределения.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Shape;

/// Маркер для scale-параметра распределения.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Scale;

/// Маркер для precision-параметра распределения.
///
/// Используется для параметризаций вида mean/precision, например beta
/// distribution, где `precision > 0` управляет концентрацией вокруг mean.
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

/// Типизированный block коэффициентов для одного параметра распределения.
///
/// `P` задаёт роль параметра, `L` задаёт link-функцию, `X` хранит predictor
/// block, а `Penalty` добавляет регуляризацию. `offset` и `len` описывают
/// диапазон коэффициентов блока внутри общего вектора beta.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterBlock<P, L, X, Penalty> {
    /// Predictor block.
    pub x: X,
    /// Penalty, применяемый к коэффициентам блока.
    pub penalty: Penalty,
    /// Начальная позиция блока в общем beta-векторе.
    pub offset: usize,
    /// Число коэффициентов в блоке.
    pub len: usize,
    marker: PhantomData<(P, L)>,
}

impl<P, L, X, Penalty> ParameterBlock<P, L, X, Penalty>
where
    X: PredictorBlock,
{
    /// Создаёт блок и берёт `len` из `x.nparams()`.
    #[must_use]
    pub fn new(x: X, penalty: Penalty, offset: usize) -> Self {
        let len = x.nparams();
        Self::from_len(x, penalty, offset, len)
    }

    /// Создаёт блок из generic predictor.
    ///
    /// Это синоним [`Self::new`], оставленный для кода, где явное слово
    /// `predictor` делает вызов читаемее.
    #[must_use]
    pub fn from_predictor(x: X, penalty: Penalty, offset: usize) -> Self {
        Self::new(x, penalty, offset)
    }
}

impl<P, L, X, Penalty> ParameterBlock<P, L, LinearPredictorBlock<X>, Penalty>
where
    X: DesignMatrix,
{
    /// Создаёт линейный block из design matrix.
    #[must_use]
    pub fn linear(x: X, penalty: Penalty, offset: usize) -> Self {
        Self::new(LinearPredictorBlock::new(x), penalty, offset)
    }
}

impl<P, L, X, Penalty> ParameterBlock<P, L, X, Penalty> {
    fn from_len(x: X, penalty: Penalty, offset: usize, len: usize) -> Self {
        Self {
            x,
            penalty,
            offset,
            len,
            marker: PhantomData,
        }
    }

    /// Возвращает копию блока с новым offset.
    #[must_use]
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Диапазон коэффициентов блока в общем beta-векторе.
    ///
    /// # Panics
    ///
    /// Panics if `offset + len` overflows. Use [`Self::try_range`] when the
    /// offset may come from unchecked external input.
    #[must_use]
    pub fn range(&self) -> Range<usize> {
        self.offset..self.end()
    }

    /// Индекс сразу после последнего коэффициента блока.
    ///
    /// # Panics
    ///
    /// Panics if `offset + len` overflows. Use [`Self::try_range`] for
    /// recoverable validation.
    #[must_use]
    pub fn end(&self) -> usize {
        self.offset
            .checked_add(self.len)
            .expect("parameter block range end must fit in usize")
    }

    /// Число коэффициентов блока.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true`, если block не содержит коэффициентов.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<P, L, X, Penalty> ParameterBlock<P, L, X, Penalty>
where
    P: ParameterName,
{
    /// Проверяет и возвращает диапазон коэффициентов блока.
    ///
    /// # Errors
    ///
    /// Возвращает [`ModelError::BlockRangeOverflow`], если `offset + len` не
    /// помещается в `usize`.
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
