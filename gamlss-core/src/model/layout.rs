use std::ops::Range;

use crate::ParameterName;

/// One axis in a nested distribution-parameter path.
///
/// Paths describe structure inside a named distribution parameter without
/// overloading the parameter role string. For example, a mixture component's
/// Cholesky entry can be represented as `component[2] / lower[1, 0]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterAxis {
    /// Component inside a repeated, product, or mixture-shaped parameter.
    Component {
        /// Zero-based component index.
        index: usize,
    },
    /// One component of a vector-valued parameter.
    Vector {
        /// Zero-based component index.
        component: usize,
    },
    /// One entry of a lower-triangular matrix parameter, including the diagonal.
    Lower {
        /// Zero-based row index.
        row: usize,
        /// Zero-based column index.
        col: usize,
    },
    /// One entry of a strict-lower triangular matrix parameter.
    StrictLower {
        /// Zero-based row index.
        row: usize,
        /// Zero-based column index.
        col: usize,
    },
    /// One entry of a dense matrix parameter.
    Matrix {
        /// Zero-based row index.
        row: usize,
        /// Zero-based column index.
        col: usize,
    },
    /// One free baseline-softmax logit.
    SimplexLogit {
        /// Zero-based class index. The baseline class has no coefficient stream.
        class: usize,
    },
    /// Named hyper-parameter or auxiliary structured axis.
    Hyper {
        /// Stable hyper-parameter axis name.
        name: &'static str,
    },
}

/// Nested position inside a distribution parameter.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParameterPath {
    axes: Vec<ParameterAxis>,
}

impl ParameterPath {
    /// Creates a path from nested axes.
    #[must_use]
    #[inline]
    pub const fn new(axes: Vec<ParameterAxis>) -> Self {
        Self { axes }
    }

    /// Creates an empty path referring to the whole named parameter role.
    #[must_use]
    #[inline]
    pub const fn whole() -> Self {
        Self { axes: Vec::new() }
    }

    /// Creates a one-axis path.
    #[must_use]
    #[inline]
    pub fn from_axis(axis: ParameterAxis) -> Self {
        Self { axes: vec![axis] }
    }

    /// Returns the nested axes in outer-to-inner order.
    #[must_use]
    #[inline]
    pub fn axes(&self) -> &[ParameterAxis] {
        &self.axes
    }

    /// Consumes the path and returns its axes.
    #[must_use]
    #[inline]
    pub fn into_axes(self) -> Vec<ParameterAxis> {
        self.axes
    }

    /// `true` when this path refers to the whole named parameter role.
    #[must_use]
    #[inline]
    pub const fn is_whole(&self) -> bool {
        self.axes.is_empty()
    }
}

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

/// Descriptor for a structured parameter or sub-parameter coefficient range.
///
/// This type is deliberately separate from [`ParameterSlice`]: slices expose
/// block-level coefficient layout, while descriptors can expose nested
/// component-level metadata for multivariate, repeated, and mixture shapes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterDescriptor {
    /// Stable distribution parameter role, e.g. `"mu"` or `"cholesky"`.
    pub role: &'static str,
    /// Nested structured position inside the named parameter role.
    pub path: ParameterPath,
    /// Coefficient range for this part inside the full beta vector.
    pub range: Range<usize>,
}

impl ParameterDescriptor {
    /// Creates a descriptor from an explicit nested path.
    #[must_use]
    #[inline]
    pub const fn new(role: &'static str, path: ParameterPath, range: Range<usize>) -> Self {
        Self { role, path, range }
    }

    /// Creates a descriptor for a whole parameter block.
    #[must_use]
    #[inline]
    pub const fn whole(role: &'static str, range: Range<usize>) -> Self {
        Self::new(role, ParameterPath::whole(), range)
    }

    /// Creates a descriptor for one repeated or mixture component.
    #[must_use]
    #[inline]
    pub fn component(role: &'static str, index: usize, range: Range<usize>) -> Self {
        Self::new(
            role,
            ParameterPath::from_axis(ParameterAxis::Component { index }),
            range,
        )
    }

    /// Creates a descriptor for one vector component.
    #[must_use]
    #[inline]
    pub fn vector_component(role: &'static str, component: usize, range: Range<usize>) -> Self {
        Self::new(
            role,
            ParameterPath::from_axis(ParameterAxis::Vector { component }),
            range,
        )
    }

    /// Creates a descriptor for one lower-triangular matrix entry.
    #[must_use]
    #[inline]
    pub fn lower_triangular_entry(
        role: &'static str,
        row: usize,
        col: usize,
        range: Range<usize>,
    ) -> Self {
        Self::new(
            role,
            ParameterPath::from_axis(ParameterAxis::Lower { row, col }),
            range,
        )
    }

    /// Creates a descriptor for one strict-lower triangular matrix entry.
    #[must_use]
    #[inline]
    pub fn strict_lower_triangular_entry(
        role: &'static str,
        row: usize,
        col: usize,
        range: Range<usize>,
    ) -> Self {
        Self::new(
            role,
            ParameterPath::from_axis(ParameterAxis::StrictLower { row, col }),
            range,
        )
    }

    /// Creates a descriptor for one dense matrix entry.
    #[must_use]
    #[inline]
    pub fn matrix_entry(role: &'static str, row: usize, col: usize, range: Range<usize>) -> Self {
        Self::new(
            role,
            ParameterPath::from_axis(ParameterAxis::Matrix { row, col }),
            range,
        )
    }

    /// Creates a descriptor for one free baseline-softmax logit.
    #[must_use]
    #[inline]
    pub fn simplex_logit(role: &'static str, class: usize, range: Range<usize>) -> Self {
        Self::new(
            role,
            ParameterPath::from_axis(ParameterAxis::SimplexLogit { class }),
            range,
        )
    }
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
    pub const fn new(slices: Vec<ParameterSlice>) -> Self {
        Self { slices }
    }

    /// Number of parameter blocks represented by this layout.
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        self.slices.len()
    }

    /// `true` if this layout has no parameter blocks.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
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
    #[inline]
    pub fn slices(&self) -> &[ParameterSlice] {
        &self.slices
    }

    /// Returns block-level descriptors for the current layout.
    ///
    /// Structured multivariate blocks may expose finer-grained descriptors in
    /// future APIs. This method provides the compatibility baseline: every
    /// existing slice is represented as a whole-parameter descriptor.
    #[must_use]
    pub fn block_descriptors(&self) -> Vec<ParameterDescriptor> {
        self.slices
            .iter()
            .map(|slice| ParameterDescriptor::whole(slice.name, slice.range.clone()))
            .collect()
    }

    /// Visits block-level descriptors in model order without allocating.
    #[inline]
    pub fn visit_block_descriptors(&self, mut visit: impl FnMut(usize, ParameterDescriptor)) {
        for (index, slice) in self.slices.iter().enumerate() {
            visit(
                index,
                ParameterDescriptor::whole(slice.name, slice.range.clone()),
            );
        }
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
