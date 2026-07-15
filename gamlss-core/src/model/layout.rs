use std::ops::Range;

use crate::{ModelError, ParameterName};

pub(super) struct UniqueParameterMatch<T> {
    value: Option<T>,
    matches: usize,
}

impl<T> UniqueParameterMatch<T> {
    #[inline]
    pub(super) const fn new() -> Self {
        Self {
            value: None,
            matches: 0,
        }
    }

    #[inline]
    pub(super) fn record(&mut self, value: T) {
        self.matches += 1;
        if self.value.is_none() {
            self.value = Some(value);
        }
    }

    pub(super) fn resolve(self, name: &str) -> Result<Option<T>, ModelError> {
        if self.matches > 1 {
            Err(ModelError::AmbiguousParameter {
                name: name.to_owned(),
                matches: self.matches,
            })
        } else {
            Ok(self.value)
        }
    }
}

/// One axis in a nested distribution-parameter path.
///
/// Paths describe structure inside a named distribution parameter without
/// overloading the parameter role string. For example, a mixture component's
/// Cholesky entry can be represented as `component[2] / lower[1, 0]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParameterAxis {
    /// Component inside a repeated or mixture-shaped parameter.
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
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
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

    /// Appends one inner axis to this path.
    #[inline]
    pub fn push_axis(&mut self, axis: ParameterAxis) {
        self.axes.push(axis);
    }

    /// Returns this path with one inner axis appended.
    #[must_use]
    #[inline]
    pub fn with_axis(mut self, axis: ParameterAxis) -> Self {
        self.push_axis(axis);
        self
    }

    /// Returns this path prefixed by an outer path.
    #[must_use]
    pub fn prefixed_by(mut self, prefix: &Self) -> Self {
        if prefix.axes.is_empty() {
            return self;
        }
        let mut axes = Vec::with_capacity(prefix.axes.len() + self.axes.len());
        axes.extend_from_slice(&prefix.axes);
        axes.append(&mut self.axes);
        Self { axes }
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

    /// Returns coarse block-level descriptors for the current layout.
    ///
    /// Every slice is represented as a whole-parameter descriptor. Compiled
    /// models expose their canonical scalar-leaf descriptors through
    /// [`crate::Gamlss::parameter_descriptors`].
    #[must_use]
    pub fn block_descriptors(&self) -> Vec<ParameterDescriptor> {
        let mut descriptors = Vec::with_capacity(self.slices.len());
        self.visit_block_descriptors(|_, descriptor| descriptors.push(descriptor));
        descriptors
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

    /// Returns every coarse coefficient range matching `name` in model order.
    #[must_use]
    pub fn ranges(&self, name: &str) -> Vec<Range<usize>> {
        let mut ranges = Vec::with_capacity(self.slices.len());
        self.visit_ranges(name, |_, range| ranges.push(range));
        ranges
    }

    /// Returns every coarse coefficient range for typed parameter marker `P`.
    #[must_use]
    #[inline]
    pub fn ranges_of<P>(&self) -> Vec<Range<usize>>
    where
        P: ParameterName,
    {
        self.ranges(P::NAME)
    }

    /// Visits every coarse coefficient range matching `name` in model order.
    #[inline]
    pub fn visit_ranges(&self, name: &str, mut visit: impl FnMut(usize, Range<usize>)) {
        for (index, slice) in self.slices.iter().enumerate() {
            if slice.name == name {
                visit(index, slice.range.clone());
            }
        }
    }

    /// Visits every coarse coefficient range for typed parameter marker `P`.
    #[inline]
    pub fn visit_ranges_of<P>(&self, visit: impl FnMut(usize, Range<usize>))
    where
        P: ParameterName,
    {
        self.visit_ranges(P::NAME, visit);
    }

    /// Returns the only coarse coefficient range matching `name`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::AmbiguousParameter`] when more than one block has
    /// the requested name. A missing parameter is represented by `Ok(None)`.
    pub fn unique_slice(&self, name: &str) -> Result<Option<Range<usize>>, ModelError> {
        let mut matched = UniqueParameterMatch::new();
        self.visit_ranges(name, |_, range| matched.record(range));
        matched.resolve(name)
    }

    /// Returns the only coarse coefficient range for typed parameter marker `P`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::AmbiguousParameter`] when more than one block has
    /// the requested parameter role.
    #[inline]
    pub fn unique_slice_of<P>(&self) -> Result<Option<Range<usize>>, ModelError>
    where
        P: ParameterName,
    {
        self.unique_slice(P::NAME)
    }
}

/// Coefficients of one unpacked scalar-leaf parameter descriptor.
///
/// Returned by [`crate::Gamlss::unpack_parameters`] for a human-readable,
/// structurally lossless representation of the flat beta vector.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterCoefficients {
    /// Stable descriptor index in the originating compiled model layout.
    pub descriptor_index: usize,
    /// Full scalar-leaf descriptor, including nested path and beta range.
    pub descriptor: ParameterDescriptor,
    /// Coefficients for this descriptor's predictor leaf.
    pub coefficients: Vec<f64>,
}

impl ParameterCoefficients {
    /// Stable distribution parameter role.
    #[must_use]
    #[inline]
    pub const fn name(&self) -> &'static str {
        self.descriptor.role
    }

    /// Nested parameter path.
    #[must_use]
    #[inline]
    pub const fn path(&self) -> &ParameterPath {
        &self.descriptor.path
    }
}

/// Human-readable representation of the flat optimizer parameter vector.
///
/// Contains one [`ParameterCoefficients`] for each canonical scalar-leaf
/// descriptor in model order. Repeated names remain distinct through their
/// descriptor indices, paths, and ranges.
#[derive(Debug, Clone, PartialEq)]
pub struct UnpackedParameters {
    /// Parameter blocks in model order.
    pub blocks: Vec<ParameterCoefficients>,
}

impl UnpackedParameters {
    /// Returns a coefficient block by stable descriptor index.
    #[must_use]
    #[inline]
    pub fn block_at(&self, descriptor_index: usize) -> Option<&ParameterCoefficients> {
        self.blocks
            .iter()
            .find(|block| block.descriptor_index == descriptor_index)
    }

    /// Returns a coefficient block by its full descriptor.
    #[must_use]
    #[inline]
    pub fn block_for_descriptor(
        &self,
        descriptor: &ParameterDescriptor,
    ) -> Option<&ParameterCoefficients> {
        self.blocks
            .iter()
            .find(|block| block.descriptor == *descriptor)
    }

    /// Iterates over every coefficient block matching `name`.
    pub fn blocks<'a>(
        &'a self,
        name: &'a str,
    ) -> impl Iterator<Item = &'a ParameterCoefficients> + 'a {
        self.blocks.iter().filter(move |block| block.name() == name)
    }

    /// Iterates over every coefficient block for typed parameter marker `P`.
    pub fn blocks_of<P>(&self) -> impl Iterator<Item = &ParameterCoefficients>
    where
        P: ParameterName,
    {
        self.blocks(P::NAME)
    }

    /// Returns the only coefficient block matching `name`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::AmbiguousParameter`] when more than one descriptor
    /// has the requested role. A missing role is represented by `Ok(None)`.
    pub fn unique_block(&self, name: &str) -> Result<Option<&ParameterCoefficients>, ModelError> {
        let mut matched = UniqueParameterMatch::new();
        for block in &self.blocks {
            if block.name() == name {
                matched.record(block);
            }
        }
        matched.resolve(name)
    }

    /// Returns the only coefficient block for typed parameter marker `P`.
    #[inline]
    pub fn unique_block_of<P>(&self) -> Result<Option<&ParameterCoefficients>, ModelError>
    where
        P: ParameterName,
    {
        self.unique_block(P::NAME)
    }

    /// Returns the only coefficient slice matching `name`.
    #[inline]
    pub fn unique_coefficients(&self, name: &str) -> Result<Option<&[f64]>, ModelError> {
        Ok(self
            .unique_block(name)?
            .map(|block| block.coefficients.as_slice()))
    }

    /// Returns the only coefficient slice for typed parameter marker `P`.
    #[inline]
    pub fn unique_coefficients_of<P>(&self) -> Result<Option<&[f64]>, ModelError>
    where
        P: ParameterName,
    {
        self.unique_coefficients(P::NAME)
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
