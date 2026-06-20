use std::ops::Range;

use crate::ParameterName;

/// Named coefficient block inside the flat parameter vector.
///
/// Associates a stable distribution parameter name (e.g. `"mu"`) with a range
/// of positions in the common beta vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterSlice {
    /// Stable distribution parameter name, e.g. `"mu"` or `"sigma"`.
    pub name: &'static str,
    /// Coefficient range for this parameter inside the full beta vector.
    pub range: Range<usize>,
}

/// Mapping from distribution parameters to ranges in the flat beta vector.
///
/// Used for model introspection: unpacking coefficients, building diagnostics,
/// and conveying information to external optimizers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterLayout {
    slices: Vec<ParameterSlice>,
}

impl ParameterLayout {
    /// Creates a layout from named slices.
    #[must_use]
    #[inline]
    pub fn new(slices: Vec<ParameterSlice>) -> Self {
        Self { slices }
    }

    /// Number of parameter blocks represented by this layout.
    #[must_use]
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.slices.len()
    }

    /// `true` if this layout has no parameter blocks.
    #[must_use]
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.slices.is_empty()
    }

    /// Minimum coefficient-vector length needed to contain every slice.
    ///
    /// For ordinary model layouts this is equal to the model's coefficient
    /// count. For manually constructed layouts with gaps it returns the
    /// largest slice end.
    #[must_use]
    #[inline]
    pub fn ncoefficients(&self) -> usize {
        self.slices
            .iter()
            .map(|slice| slice.range.end)
            .max()
            .unwrap_or(0)
    }

    /// Returns all parameter slices in model order.
    #[must_use]
    #[inline(always)]
    pub fn slices(&self) -> &[ParameterSlice] {
        &self.slices
    }

    /// Visits parameter slices in model order without allocating.
    #[inline]
    pub fn visit_slices(&self, mut visit: impl FnMut(usize, &'static str, Range<usize>)) {
        for (index, slice) in self.slices.iter().enumerate() {
            visit(index, slice.name, slice.range.clone());
        }
    }

    /// Returns the coefficient range for `name`, if present.
    #[must_use]
    #[inline]
    pub fn slice(&self, name: &str) -> Option<Range<usize>> {
        self.slices
            .iter()
            .find(|slice| slice.name == name)
            .map(|slice| slice.range.clone())
    }

    /// Returns the coefficient range for typed parameter marker `P`, if present.
    #[must_use]
    #[inline]
    pub fn slice_of<P>(&self) -> Option<Range<usize>>
    where
        P: ParameterName,
    {
        self.slice(P::NAME)
    }
}

/// Coefficients of a single unpacked parameter block.
///
/// Returned by [`crate::Gamlss::unpack_parameters`] for a human-readable
/// representation of the flat beta vector.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterCoefficients {
    /// Stable distribution parameter name.
    pub name: &'static str,
    /// Coefficients for this parameter block.
    pub coefficients: Vec<f64>,
}

/// Human-readable representation of the flat optimizer parameter vector.
///
/// Contains one [`ParameterCoefficients`] for each distribution parameter in
/// model order.
#[derive(Debug, Clone, PartialEq)]
pub struct UnpackedParameters {
    /// Parameter blocks in model order.
    pub blocks: Vec<ParameterCoefficients>,
}

impl UnpackedParameters {
    /// Returns an unpacked coefficient block by parameter name.
    #[must_use]
    #[inline]
    pub fn block(&self, name: &str) -> Option<&ParameterCoefficients> {
        self.blocks.iter().find(|block| block.name == name)
    }

    /// Returns an unpacked coefficient block for typed parameter marker `P`.
    #[must_use]
    #[inline]
    pub fn block_of<P>(&self) -> Option<&ParameterCoefficients>
    where
        P: ParameterName,
    {
        self.block(P::NAME)
    }

    /// Returns coefficients by parameter name.
    #[must_use]
    #[inline]
    pub fn coefficients(&self, name: &str) -> Option<&[f64]> {
        self.block(name).map(|block| block.coefficients.as_slice())
    }

    /// Returns coefficients for typed parameter marker `P`.
    #[must_use]
    #[inline]
    pub fn coefficients_of<P>(&self) -> Option<&[f64]>
    where
        P: ParameterName,
    {
        self.coefficients(P::NAME)
    }
}

/// Training diagnostics for a candidate optimizer parameter vector.
///
/// Contains the objective value, scaled training negative log-likelihood
/// (without penalties), total penalty, gradient norm and the number of
/// non-finite gradient entries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainingDiagnostics {
    /// Full objective value: weighted training negative log-likelihood plus penalties.
    pub objective: f64,
    /// Training negative log-likelihood before penalties, using the model's objective scale.
    pub train_nll: f64,
    /// Total penalty contribution.
    pub penalty: f64,
    /// Euclidean norm of the objective gradient.
    pub gradient_norm: f64,
    /// Number of non-finite gradient entries.
    pub nonfinite_gradient_count: usize,
}
