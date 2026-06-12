use std::{marker::PhantomData, ops::Range};

use crate::{DesignMatrix, LinearPredictorBlock, PredictorBlock};

/// Stable public name for a distribution parameter marker.
pub trait ParameterName {
    /// Name used in parameter layouts and unpacked coefficient views.
    const NAME: &'static str;
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
    pub fn new(x: X, penalty: Penalty, offset: usize) -> Self {
        let len = x.nparams();
        Self::from_len(x, penalty, offset, len)
    }

    /// Создаёт блок из generic predictor.
    ///
    /// Это синоним [`Self::new`], оставленный для кода, где явное слово
    /// `predictor` делает вызов читаемее.
    pub fn from_predictor(x: X, penalty: Penalty, offset: usize) -> Self {
        Self::new(x, penalty, offset)
    }
}

impl<P, L, X, Penalty> ParameterBlock<P, L, LinearPredictorBlock<X>, Penalty>
where
    X: DesignMatrix,
{
    /// Создаёт линейный block из design matrix.
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
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Диапазон коэффициентов блока в общем beta-векторе.
    pub fn range(&self) -> Range<usize> {
        self.offset..self.offset + self.len
    }

    /// Индекс сразу после последнего коэффициента блока.
    pub fn end(&self) -> usize {
        self.offset + self.len
    }

    /// Число коэффициентов блока.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true`, если block не содержит коэффициентов.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}
