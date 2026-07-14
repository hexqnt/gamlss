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
pub struct ParameterBlocks<B = ()> {
    blocks: B,
}

impl<Blocks> ParameterBlocks<Blocks> {
    /// Assigns sequential offsets starting at zero.
    ///
    /// # Panics
    ///
    /// Panics if the sequential layout does not fit in `usize`. Use
    /// [`Self::try_new`] when block sizes may come from unchecked external
    /// input.
    #[must_use]
    #[inline]
    pub fn new(blocks: Blocks) -> Self
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
    pub fn with_start(start: usize, blocks: Blocks) -> Self
    where
        Blocks: AssignParameterOffsets,
    {
        Self {
            blocks: blocks.assign_offsets(start),
        }
    }

    /// Assigns sequential offsets starting at zero.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::BlockRangeOverflow`] if any assigned block range
    /// would not fit in `usize`.
    #[inline]
    pub fn try_new(blocks: Blocks) -> Result<Self, ModelError>
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
    pub fn try_with_start(start: usize, blocks: Blocks) -> Result<Self, ModelError>
    where
        Blocks: TryAssignParameterOffsets,
    {
        Ok(Self {
            blocks: blocks.try_assign_offsets(start)?,
        })
    }
}

impl<B> ParameterBlocks<B> {
    /// Wraps an already assigned low-level block tree without changing offsets.
    ///
    /// Prefer [`ParameterBlocks::new`] for ordinary model construction. This
    /// constructor is intended for integrations and validation tests that need
    /// to preserve explicit coefficient ranges.
    #[must_use]
    pub const fn from_assigned(blocks: B) -> Self {
        Self { blocks }
    }

    /// Borrows the statically typed block tree.
    #[must_use]
    pub const fn as_inner(&self) -> &B {
        &self.blocks
    }

    /// Consumes the container and returns its statically typed block tree.
    #[must_use]
    pub fn into_inner(self) -> B {
        self.blocks
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

/// Marker for baseline-softmax mixture weights.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MixtureWeight;

impl ParameterName for MixtureWeight {
    const NAME: &'static str = "mixture_weight";
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

/// Marker for a Cholesky scale factor parameter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CholeskyScale;

impl ParameterName for CholeskyScale {
    const NAME: &'static str = "cholesky";
}

/// Marker for strict-lower partial-correlation predictors.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PartialCorrelation;

impl ParameterName for PartialCorrelation {
    const NAME: &'static str = "partial_corr";
}

/// Typed coefficient block for a single distribution parameter.
///
/// `P` specifies the parameter role, `X` holds the predictor block, and
/// `Penalty` adds regularization. Links are owned by the family. The block stores its
/// coefficient range within the common beta vector; use [`Self::range`] and
/// [`Self::len`] to inspect that layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterBlock<P, X, Penalty> {
    x: X,
    penalty: Penalty,
    offset: usize,
    len: usize,
    marker: PhantomData<P>,
}

impl<P, X, Penalty> ParameterBlock<P, X, Penalty>
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

impl<P, X, Penalty> ParameterBlock<P, LinearPredictorBlock<X>, Penalty>
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

impl<P, X, Penalty> ParameterBlock<P, X, Penalty> {
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

impl<P, X, Penalty> ParameterBlock<P, X, Penalty>
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

/// Typed coefficient block for a vector-valued distribution parameter.
///
/// The block owns one scalar predictor per vector component and lays their
/// local coefficient ranges contiguously inside the common beta vector. It is
/// intended for structured families whose natural parameter is a vector, such
/// as a multivariate normal mean.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorParameterBlock<P, const D: usize, X, Penalty> {
    x: [X; D],
    penalty: Penalty,
    offset: usize,
    len: usize,
    component_offsets: [usize; D],
    marker: PhantomData<P>,
}

impl<P, const D: usize, X, Penalty> VectorParameterBlock<P, D, X, Penalty>
where
    X: PredictorBlock,
{
    /// Creates a vector block from one predictor per component.
    ///
    /// # Panics
    ///
    /// Panics if the sum of component predictor lengths does not fit in `usize`.
    #[must_use]
    pub fn new(x: [X; D], penalty: Penalty, offset: usize) -> Self {
        let mut component_offsets = [0; D];
        let mut len: usize = 0;
        for index in 0..D {
            component_offsets[index] = len;
            len = len
                .checked_add(x[index].nparams())
                .expect("vector parameter block length must fit in usize");
        }
        Self {
            x,
            penalty,
            offset,
            len,
            component_offsets,
            marker: PhantomData,
        }
    }
}

impl<P, const D: usize, X, Penalty> VectorParameterBlock<P, D, X, Penalty> {
    /// Returns a copy of the block with a new offset.
    #[must_use]
    #[inline]
    pub const fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Returns all component predictors.
    #[must_use]
    #[inline]
    pub const fn components(&self) -> &[X; D] {
        &self.x
    }

    /// Returns the predictor for `component`.
    #[must_use]
    #[inline]
    pub fn component(&self, component: usize) -> Option<&X> {
        self.x.get(component)
    }

    /// Penalty applied to the block's concatenated coefficients.
    #[must_use]
    #[inline]
    pub const fn penalty(&self) -> &Penalty {
        &self.penalty
    }

    /// Coefficient range of the full vector block.
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
            .expect("vector parameter block range end must fit in usize")
    }

    /// Number of coefficients in the full vector block.
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

    /// Local coefficient range for one vector component.
    #[must_use]
    #[inline]
    pub fn component_local_range(&self, component: usize) -> Option<Range<usize>>
    where
        X: PredictorBlock,
    {
        let start = *self.component_offsets.get(component)?;
        let len = self.x.get(component)?.nparams();
        Some(start..start + len)
    }

    /// Absolute beta range for one vector component.
    #[must_use]
    #[inline]
    pub fn component_range(&self, component: usize) -> Option<Range<usize>>
    where
        X: PredictorBlock,
    {
        let range = self.component_local_range(component)?;
        let start = self.offset.checked_add(range.start)?;
        let end = self.offset.checked_add(range.end)?;
        Some(start..end)
    }
}

impl<P, const D: usize, X, Penalty> VectorParameterBlock<P, D, X, Penalty>
where
    P: ParameterName,
{
    /// Validates and returns the full block coefficient range.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::BlockRangeOverflow`] if `offset + len` does not fit
    /// in `usize`.
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

/// Typed coefficient block for a packed lower-triangular matrix parameter.
///
/// Predictors are stored in row-major lower-triangular order:
/// `(0,0), (1,0), (1,1), (2,0), ...`. This matches Cholesky scale factors and
/// other structured matrix parameters whose upper-triangular entries are not
/// modeled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LowerTriangularParameterBlock<P, const D: usize, X, Penalty> {
    x: Vec<X>,
    penalty: Penalty,
    offset: usize,
    len: usize,
    entry_offsets: Vec<usize>,
    marker: PhantomData<P>,
}

impl<P, const D: usize, X, Penalty> LowerTriangularParameterBlock<P, D, X, Penalty>
where
    P: ParameterName,
    X: PredictorBlock,
{
    /// Creates a lower-triangular block from packed row-major predictors.
    ///
    /// # Panics
    ///
    /// Panics if `x.len()` is not `D * (D + 1) / 2` or if the sum of entry
    /// predictor lengths does not fit in `usize`.
    #[must_use]
    pub fn new(x: Vec<X>, penalty: Penalty, offset: usize) -> Self {
        Self::try_new(x, penalty, offset)
            .expect("lower-triangular block must have D * (D + 1) / 2 predictors")
    }

    /// Creates a lower-triangular block from packed row-major predictors.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] if `x.len()` is not `D * (D + 1) / 2`.
    /// Returns [`ModelError::ArithmeticOverflow`] if the packed length or total
    /// coefficient length does not fit in `usize`.
    pub fn try_new(x: Vec<X>, penalty: Penalty, offset: usize) -> Result<Self, ModelError> {
        let expected = lower_triangular_len(D).ok_or(ModelError::ArithmeticOverflow {
            context: "lower-triangular predictor count",
        })?;
        if x.len() != expected {
            return Err(ModelError::InvalidParameter {
                parameter: P::NAME,
                expected: "D * (D + 1) / 2 predictor blocks",
            });
        }

        let mut entry_offsets = vec![0; expected];
        let mut len: usize = 0;
        for (index, predictor) in x.iter().enumerate() {
            entry_offsets[index] = len;
            len = len
                .checked_add(predictor.nparams())
                .ok_or(ModelError::ArithmeticOverflow {
                    context: "lower-triangular parameter block length",
                })?;
        }

        Ok(Self {
            x,
            penalty,
            offset,
            len,
            entry_offsets,
            marker: PhantomData,
        })
    }
}

impl<P, const D: usize, X, Penalty> LowerTriangularParameterBlock<P, D, X, Penalty> {
    /// Returns the packed lower-triangular length for dimension `D`.
    #[must_use]
    #[inline]
    pub const fn packed_len() -> Option<usize> {
        lower_triangular_len(D)
    }

    /// Returns the packed row-major lower-triangular index for `(row, col)`.
    #[must_use]
    #[inline]
    pub const fn packed_index(row: usize, col: usize) -> Option<usize> {
        if col <= row && row < D {
            match row.checked_add(1) {
                Some(next) => match row.checked_mul(next) {
                    Some(product) => match (product / 2).checked_add(col) {
                        Some(index) => Some(index),
                        None => None,
                    },
                    None => None,
                },
                None => None,
            }
        } else {
            None
        }
    }

    /// Returns a copy of the block with a new offset.
    #[must_use]
    #[inline]
    pub const fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Returns all packed lower-triangular predictors.
    #[must_use]
    #[inline]
    pub fn entries(&self) -> &[X] {
        &self.x
    }

    /// Returns the predictor for lower-triangular entry `(row, col)`.
    #[must_use]
    #[inline]
    pub fn entry(&self, row: usize, col: usize) -> Option<&X> {
        Self::packed_index(row, col).and_then(|index| self.x.get(index))
    }

    /// Penalty applied to the block's concatenated coefficients.
    #[must_use]
    #[inline]
    pub const fn penalty(&self) -> &Penalty {
        &self.penalty
    }

    /// Coefficient range of the full lower-triangular block.
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
            .expect("lower-triangular parameter block range end must fit in usize")
    }

    /// Number of coefficients in the full lower-triangular block.
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

    /// Local coefficient range for one lower-triangular entry.
    #[must_use]
    #[inline]
    pub fn entry_local_range(&self, row: usize, col: usize) -> Option<Range<usize>>
    where
        X: PredictorBlock,
    {
        let index = Self::packed_index(row, col)?;
        let start = *self.entry_offsets.get(index)?;
        let len = self.x.get(index)?.nparams();
        Some(start..start + len)
    }

    /// Absolute beta range for one lower-triangular entry.
    #[must_use]
    #[inline]
    pub fn entry_range(&self, row: usize, col: usize) -> Option<Range<usize>>
    where
        X: PredictorBlock,
    {
        let range = self.entry_local_range(row, col)?;
        let start = self.offset.checked_add(range.start)?;
        let end = self.offset.checked_add(range.end)?;
        Some(start..end)
    }
}

impl<P, const D: usize, X, Penalty> LowerTriangularParameterBlock<P, D, X, Penalty>
where
    P: ParameterName,
{
    /// Validates and returns the full block coefficient range.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::BlockRangeOverflow`] if `offset + len` does not fit
    /// in `usize`.
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

/// Typed coefficient block for a packed strict-lower triangular matrix parameter.
///
/// Predictors are stored in row-major strict-lower order:
/// `(1,0), (2,0), (2,1), (3,0), ...`. This matches partial-correlation
/// parameterizations where diagonal entries are structural constants and must
/// not receive predictors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrictLowerTriangularParameterBlock<P, const D: usize, X, Penalty> {
    x: Vec<X>,
    penalty: Penalty,
    offset: usize,
    len: usize,
    entry_offsets: Vec<usize>,
    marker: PhantomData<P>,
}

impl<P, const D: usize, X, Penalty> StrictLowerTriangularParameterBlock<P, D, X, Penalty>
where
    P: ParameterName,
    X: PredictorBlock,
{
    /// Creates a strict-lower block from packed row-major predictors.
    ///
    /// # Panics
    ///
    /// Panics if `x.len()` is not `D * (D - 1) / 2` or if the sum of entry
    /// predictor lengths does not fit in `usize`.
    #[must_use]
    pub fn new(x: Vec<X>, penalty: Penalty, offset: usize) -> Self {
        Self::try_new(x, penalty, offset)
            .expect("strict-lower block must have D * (D - 1) / 2 predictors")
    }

    /// Creates a strict-lower block from packed row-major predictors.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] if `x.len()` is not `D * (D - 1) / 2`.
    /// Returns [`ModelError::ArithmeticOverflow`] if the packed length or total
    /// coefficient length does not fit in `usize`.
    pub fn try_new(x: Vec<X>, penalty: Penalty, offset: usize) -> Result<Self, ModelError> {
        let expected = strict_lower_triangular_len(D).ok_or(ModelError::ArithmeticOverflow {
            context: "strict-lower predictor count",
        })?;
        if x.len() != expected {
            return Err(ModelError::InvalidParameter {
                parameter: P::NAME,
                expected: "D * (D - 1) / 2 predictor blocks",
            });
        }

        let mut entry_offsets = vec![0; expected];
        let mut len: usize = 0;
        for (index, predictor) in x.iter().enumerate() {
            entry_offsets[index] = len;
            len = len
                .checked_add(predictor.nparams())
                .ok_or(ModelError::ArithmeticOverflow {
                    context: "strict-lower parameter block length",
                })?;
        }

        Ok(Self {
            x,
            penalty,
            offset,
            len,
            entry_offsets,
            marker: PhantomData,
        })
    }
}

impl<P, const D: usize, X, Penalty> StrictLowerTriangularParameterBlock<P, D, X, Penalty> {
    /// Returns the packed strict-lower length for dimension `D`.
    #[must_use]
    #[inline]
    pub const fn packed_len() -> Option<usize> {
        strict_lower_triangular_len(D)
    }

    /// Returns the packed row-major strict-lower index for `(row, col)`.
    #[must_use]
    #[inline]
    pub const fn packed_index(row: usize, col: usize) -> Option<usize> {
        if col < row && row < D {
            match row.checked_mul(row.saturating_sub(1)) {
                Some(product) => match (product / 2).checked_add(col) {
                    Some(index) => Some(index),
                    None => None,
                },
                None => None,
            }
        } else {
            None
        }
    }

    /// Returns a copy of the block with a new offset.
    #[must_use]
    #[inline]
    pub const fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Returns all packed strict-lower predictors.
    #[must_use]
    #[inline]
    pub fn entries(&self) -> &[X] {
        &self.x
    }

    /// Returns the predictor for strict-lower entry `(row, col)`.
    #[must_use]
    #[inline]
    pub fn entry(&self, row: usize, col: usize) -> Option<&X> {
        Self::packed_index(row, col).and_then(|index| self.x.get(index))
    }

    /// Penalty applied to the block's concatenated coefficients.
    #[must_use]
    #[inline]
    pub const fn penalty(&self) -> &Penalty {
        &self.penalty
    }

    /// Coefficient range of the full strict-lower block.
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
            .expect("strict-lower parameter block range end must fit in usize")
    }

    /// Number of coefficients in the full strict-lower block.
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

    /// Local coefficient range for one strict-lower entry.
    #[must_use]
    #[inline]
    pub fn entry_local_range(&self, row: usize, col: usize) -> Option<Range<usize>>
    where
        X: PredictorBlock,
    {
        let index = Self::packed_index(row, col)?;
        let start = *self.entry_offsets.get(index)?;
        let len = self.x.get(index)?.nparams();
        Some(start..start + len)
    }

    /// Absolute beta range for one strict-lower entry.
    #[must_use]
    #[inline]
    pub fn entry_range(&self, row: usize, col: usize) -> Option<Range<usize>>
    where
        X: PredictorBlock,
    {
        let range = self.entry_local_range(row, col)?;
        let start = self.offset.checked_add(range.start)?;
        let end = self.offset.checked_add(range.end)?;
        Some(start..end)
    }
}

impl<P, const D: usize, X, Penalty> StrictLowerTriangularParameterBlock<P, D, X, Penalty>
where
    P: ParameterName,
{
    /// Validates and returns the full block coefficient range.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::BlockRangeOverflow`] if `offset + len` does not fit
    /// in `usize`.
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

/// Typed coefficient block for `D - 1` baseline-softmax logits of a simplex parameter.
///
/// The last simplex logit is fixed to zero and is not represented by a
/// predictor. This makes baseline-softmax models identifiable at the block
/// layer instead of relying on family constructors to normalize an extra free
/// coefficient.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimplexLogitParameterBlock<P, const D: usize, X, Penalty> {
    x: Vec<X>,
    penalty: Penalty,
    offset: usize,
    len: usize,
    component_offsets: Vec<usize>,
    marker: PhantomData<P>,
}

impl<P, const D: usize, X, Penalty> SimplexLogitParameterBlock<P, D, X, Penalty>
where
    P: ParameterName,
    X: PredictorBlock,
{
    /// Creates a simplex-logit block from one predictor per free baseline logit.
    ///
    /// # Panics
    ///
    /// Panics if `x.len()` is not `D - 1` or if the sum of component predictor
    /// lengths does not fit in `usize`.
    #[must_use]
    pub fn new(x: Vec<X>, penalty: Penalty, offset: usize) -> Self {
        Self::try_new(x, penalty, offset).expect("simplex-logit block must have D - 1 predictors")
    }

    /// Creates a simplex-logit block from one predictor per free baseline logit.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] if `x.len()` is not `D - 1`.
    /// Returns [`ModelError::ArithmeticOverflow`] if total coefficient length
    /// does not fit in `usize`.
    pub fn try_new(x: Vec<X>, penalty: Penalty, offset: usize) -> Result<Self, ModelError> {
        let expected = D.checked_sub(1).ok_or(ModelError::InvalidParameter {
            parameter: P::NAME,
            expected: "D >= 1",
        })?;
        if x.len() != expected {
            return Err(ModelError::InvalidParameter {
                parameter: P::NAME,
                expected: "D - 1 predictor blocks",
            });
        }

        let mut component_offsets = vec![0; expected];
        let mut len: usize = 0;
        for (index, predictor) in x.iter().enumerate() {
            component_offsets[index] = len;
            len = len
                .checked_add(predictor.nparams())
                .ok_or(ModelError::ArithmeticOverflow {
                    context: "simplex-logit parameter block length",
                })?;
        }

        Ok(Self {
            x,
            penalty,
            offset,
            len,
            component_offsets,
            marker: PhantomData,
        })
    }
}

impl<P, const D: usize, X, Penalty> SimplexLogitParameterBlock<P, D, X, Penalty> {
    /// Number of free logits represented by the block.
    #[must_use]
    #[inline]
    pub const fn free_len() -> Option<usize> {
        D.checked_sub(1)
    }

    /// Returns a copy of the block with a new offset.
    #[must_use]
    #[inline]
    pub const fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Returns all free-logit predictors.
    #[must_use]
    #[inline]
    pub fn logits(&self) -> &[X] {
        &self.x
    }

    /// Returns the predictor for a free logit component.
    #[must_use]
    #[inline]
    pub fn logit(&self, component: usize) -> Option<&X> {
        self.x.get(component)
    }

    /// Penalty applied to the block's concatenated coefficients.
    #[must_use]
    #[inline]
    pub const fn penalty(&self) -> &Penalty {
        &self.penalty
    }

    /// Coefficient range of the full simplex-logit block.
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
            .expect("simplex-logit parameter block range end must fit in usize")
    }

    /// Number of coefficients in the full simplex-logit block.
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

    /// Local coefficient range for one free logit.
    #[must_use]
    #[inline]
    pub fn logit_local_range(&self, component: usize) -> Option<Range<usize>>
    where
        X: PredictorBlock,
    {
        let start = *self.component_offsets.get(component)?;
        let len = self.x.get(component)?.nparams();
        Some(start..start + len)
    }

    /// Absolute beta range for one free logit.
    #[must_use]
    #[inline]
    pub fn logit_range(&self, component: usize) -> Option<Range<usize>>
    where
        X: PredictorBlock,
    {
        let range = self.logit_local_range(component)?;
        let start = self.offset.checked_add(range.start)?;
        let end = self.offset.checked_add(range.end)?;
        Some(start..end)
    }
}

impl<P, const D: usize, X, Penalty> SimplexLogitParameterBlock<P, D, X, Penalty>
where
    P: ParameterName,
{
    /// Validates and returns the full block coefficient range.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::BlockRangeOverflow`] if `offset + len` does not fit
    /// in `usize`.
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

/// Contract implemented for typed parameter block tuples up to arity 8 and
/// repeated block arrays.
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

const fn lower_triangular_len(dimension: usize) -> Option<usize> {
    match dimension.checked_add(1) {
        Some(next) => match dimension.checked_mul(next) {
            Some(product) => Some(product / 2),
            None => None,
        },
        None => None,
    }
}

const fn strict_lower_triangular_len(dimension: usize) -> Option<usize> {
    match dimension.checked_sub(1) {
        Some(previous) => match dimension.checked_mul(previous) {
            Some(product) => Some(product / 2),
            None => None,
        },
        None => Some(0),
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

        impl<$($block,)+> OffsetAssignable for ($($block,)+)
        where
            $($block: OffsetAssignable,)+
        {
            #[inline]
            fn with_assigned_offset(self, offset: usize) -> Self {
                AssignParameterOffsets::assign_offsets(self, offset)
            }

            #[inline]
            fn assigned_len(&self) -> usize {
                let ($($var,)+) = self;
                let mut len = 0usize;
                $(
                    len = len
                        .checked_add($var.assigned_len())
                        .expect("parameter block tuple length must fit in usize");
                )+
                len
            }
        }

        impl<$($block,)+> TryOffsetAssignable for ($($block,)+)
        where
            $($block: TryOffsetAssignable,)+
        {
            #[inline]
            fn with_assigned_offset(self, start: usize) -> Self {
                let ($($var,)+) = self;
                let mut offset = start;
                $(
                    let $var = $var.with_assigned_offset(offset);
                    offset = offset
                        .checked_add($var.assigned_len())
                        .expect("parameter block tuple layout must fit in usize");
                )+
                let _ = offset;
                ($($var,)+)
            }

            #[inline]
            fn assigned_offset(&self) -> usize {
                self.0.assigned_offset()
            }

            #[inline]
            fn assigned_len(&self) -> usize {
                let ($($var,)+) = self;
                let mut len = 0usize;
                $(
                    len = len
                        .checked_add($var.assigned_len())
                        .expect("parameter block tuple length must fit in usize");
                )+
                len
            }

            #[inline]
            fn assigned_name(&self) -> &'static str {
                self.0.assigned_name()
            }
        }
    };
}

impl<P, X, Penalty> OffsetAssignable for ParameterBlock<P, X, Penalty> {
    fn with_assigned_offset(self, offset: usize) -> Self {
        self.with_offset(offset)
    }

    fn assigned_len(&self) -> usize {
        self.len()
    }
}

impl<P, X, Penalty> TryOffsetAssignable for ParameterBlock<P, X, Penalty>
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

impl<P, const D: usize, X, Penalty> OffsetAssignable for VectorParameterBlock<P, D, X, Penalty> {
    fn with_assigned_offset(self, offset: usize) -> Self {
        self.with_offset(offset)
    }

    fn assigned_len(&self) -> usize {
        self.len()
    }
}

impl<P, const D: usize, X, Penalty> TryOffsetAssignable for VectorParameterBlock<P, D, X, Penalty>
where
    P: ParameterName,
{
    fn with_assigned_offset(self, offset: usize) -> Self {
        self.with_offset(offset)
    }

    fn assigned_offset(&self) -> usize {
        self.offset
    }

    fn assigned_len(&self) -> usize {
        self.len()
    }

    fn assigned_name(&self) -> &'static str {
        P::NAME
    }
}

impl<P, const D: usize, X, Penalty> OffsetAssignable
    for LowerTriangularParameterBlock<P, D, X, Penalty>
{
    fn with_assigned_offset(self, offset: usize) -> Self {
        self.with_offset(offset)
    }

    fn assigned_len(&self) -> usize {
        self.len()
    }
}

impl<P, const D: usize, X, Penalty> TryOffsetAssignable
    for LowerTriangularParameterBlock<P, D, X, Penalty>
where
    P: ParameterName,
{
    fn with_assigned_offset(self, offset: usize) -> Self {
        self.with_offset(offset)
    }

    fn assigned_offset(&self) -> usize {
        self.offset
    }

    fn assigned_len(&self) -> usize {
        self.len()
    }

    fn assigned_name(&self) -> &'static str {
        P::NAME
    }
}

impl<P, const D: usize, X, Penalty> OffsetAssignable
    for StrictLowerTriangularParameterBlock<P, D, X, Penalty>
{
    fn with_assigned_offset(self, offset: usize) -> Self {
        self.with_offset(offset)
    }

    fn assigned_len(&self) -> usize {
        self.len()
    }
}

impl<P, const D: usize, X, Penalty> TryOffsetAssignable
    for StrictLowerTriangularParameterBlock<P, D, X, Penalty>
where
    P: ParameterName,
{
    fn with_assigned_offset(self, offset: usize) -> Self {
        self.with_offset(offset)
    }

    fn assigned_offset(&self) -> usize {
        self.offset
    }

    fn assigned_len(&self) -> usize {
        self.len()
    }

    fn assigned_name(&self) -> &'static str {
        P::NAME
    }
}

impl<P, const D: usize, X, Penalty> OffsetAssignable
    for SimplexLogitParameterBlock<P, D, X, Penalty>
{
    fn with_assigned_offset(self, offset: usize) -> Self {
        self.with_offset(offset)
    }

    fn assigned_len(&self) -> usize {
        self.len()
    }
}

impl<P, const D: usize, X, Penalty> TryOffsetAssignable
    for SimplexLogitParameterBlock<P, D, X, Penalty>
where
    P: ParameterName,
{
    fn with_assigned_offset(self, offset: usize) -> Self {
        self.with_offset(offset)
    }

    fn assigned_offset(&self) -> usize {
        self.offset
    }

    fn assigned_len(&self) -> usize {
        self.len()
    }

    fn assigned_name(&self) -> &'static str {
        P::NAME
    }
}

impl<B, const D: usize> AssignParameterOffsets for [B; D]
where
    B: OffsetAssignable,
{
    #[inline]
    fn assign_offsets(self, start: usize) -> Self {
        self.with_assigned_offset(start)
    }
}

impl<B, const D: usize> TryAssignParameterOffsets for [B; D]
where
    B: TryOffsetAssignable,
{
    #[inline]
    fn try_assign_offsets(self, start: usize) -> Result<Self, ModelError> {
        let mut offset = start;
        for block in &self {
            let len = block.assigned_len();
            offset = offset
                .checked_add(len)
                .ok_or_else(|| ModelError::BlockRangeOverflow {
                    parameter: block.assigned_name(),
                    offset,
                    len,
                })?;
        }
        Ok(self.with_assigned_offset(start))
    }
}

trait OffsetAssignable: Sized {
    fn with_assigned_offset(self, offset: usize) -> Self;
    fn assigned_len(&self) -> usize;
}

impl<B, const D: usize> OffsetAssignable for [B; D]
where
    B: OffsetAssignable,
{
    fn with_assigned_offset(self, start: usize) -> Self {
        let mut offset = start;
        self.map(|block| {
            let block = block.with_assigned_offset(offset);
            offset = offset
                .checked_add(block.assigned_len())
                .expect("repeated parameter block layout must fit in usize");
            block
        })
    }

    fn assigned_len(&self) -> usize {
        self.iter()
            .map(OffsetAssignable::assigned_len)
            .try_fold(0usize, usize::checked_add)
            .expect("repeated parameter block length must fit in usize")
    }
}

trait TryOffsetAssignable: Sized {
    fn with_assigned_offset(self, offset: usize) -> Self;
    fn assigned_offset(&self) -> usize;
    fn assigned_len(&self) -> usize;
    fn assigned_name(&self) -> &'static str;
}

impl<B, const D: usize> TryOffsetAssignable for [B; D]
where
    B: TryOffsetAssignable,
{
    fn with_assigned_offset(self, start: usize) -> Self {
        let mut offset = start;
        self.map(|block| {
            let block = block.with_assigned_offset(offset);
            offset = offset
                .checked_add(block.assigned_len())
                .expect("repeated parameter block layout must fit in usize");
            block
        })
    }

    fn assigned_offset(&self) -> usize {
        self.first().map_or(0, TryOffsetAssignable::assigned_offset)
    }

    fn assigned_len(&self) -> usize {
        self.iter()
            .map(TryOffsetAssignable::assigned_len)
            .try_fold(0usize, usize::checked_add)
            .expect("repeated parameter block length must fit in usize")
    }

    fn assigned_name(&self) -> &'static str {
        self.first()
            .map_or("repeated", TryOffsetAssignable::assigned_name)
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
    use crate::{DenseDesign, LinearPredictorBlock, NoPenalty};

    use super::{
        Mu, Nu, ParameterBlock, ParameterBlocks, Precision, Rate, Scale, Shape, Sigma, Tau,
    };

    #[test]
    fn parameter_blocks_assign_offsets_for_one_block() {
        let mu = ParameterBlock::<Mu, _, _>::linear(
            DenseDesign::from_rows(&[[1.0, 2.0]]),
            NoPenalty,
            99,
        );

        let (mu,) = ParameterBlocks::new((mu,)).into_inner();

        assert_eq!(mu.range(), 0..2);
    }

    #[test]
    fn parameter_blocks_assign_offsets_for_two_blocks() {
        let mu = ParameterBlock::<Mu, _, _>::linear(
            DenseDesign::from_rows(&[[1.0, 2.0]]),
            NoPenalty,
            99,
        );
        let sigma = ParameterBlock::<Sigma, _, _>::linear(
            DenseDesign::from_rows(&[[1.0, 2.0, 3.0]]),
            NoPenalty,
            99,
        );

        let (mu, sigma) = ParameterBlocks::new((mu, sigma)).into_inner();

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

        let (b1, b2, b3, b4, b5, b6, b7, b8) = ParameterBlocks::with_start(10, blocks).into_inner();

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
        let block = ParameterBlock::<Mu, _, _>::linear(
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
        let mu = ParameterBlock::<Mu, _, _>::linear(
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

    fn intercept_block<P>() -> ParameterBlock<P, LinearPredictorBlock<DenseDesign>, NoPenalty> {
        ParameterBlock::linear(DenseDesign::intercept(1), NoPenalty, 99)
    }
}
