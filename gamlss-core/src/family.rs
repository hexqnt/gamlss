use crate::{
    ModelError,
    model::{ObservationView, ParameterPath},
    shape::{ParameterShape, ShapeValues},
};

/// Dense expected information matrix for a fixed-arity family.
///
/// This is an extension-point container for second-order fitting algorithms.
/// The core model hot path does not consume it directly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseInformation<const K: usize> {
    values: [[f64; K]; K],
}

impl<const K: usize> DenseInformation<K> {
    /// Creates a dense information matrix from row-major values.
    #[must_use]
    #[inline]
    pub const fn new(values: [[f64; K]; K]) -> Self {
        Self { values }
    }

    /// Creates a diagonal information matrix.
    #[must_use]
    #[inline]
    pub const fn diagonal(diagonal: [f64; K]) -> Self {
        let mut values = [[0.0; K]; K];
        let mut index = 0;
        while index < K {
            values[index][index] = diagonal[index];
            index += 1;
        }
        Self { values }
    }

    /// Returns the matrix entry at `row`, `col`.
    #[must_use]
    #[inline]
    pub const fn get(&self, row: usize, col: usize) -> f64 {
        self.values[row][col]
    }

    /// Returns the underlying dense matrix.
    #[must_use]
    #[inline]
    pub const fn as_array(&self) -> &[[f64; K]; K] {
        &self.values
    }
}

/// Distribution contract for the compiled GAMLSS objective.
///
/// The predictor layer is responsible for computing raw link-scale
/// [`Eta`](Self::Eta) values from covariates and coefficients. This trait owns
/// the distribution-side interpretation of those values: applying links,
/// enforcing dependent constraints, constructing valid natural-scale
/// [`Theta`](Self::Theta), and evaluating likelihood/gradient terms. Keep
/// distribution parameterizations such as Cholesky covariance, `D R D`
/// covariance, simplex weights, or ordered cutpoints in `Family::theta` and the
/// family gradient logic rather than in predictor blocks.
///
/// Implementations should treat `nll`/`nll_eta` as negative log-likelihood
/// contributions for one observation. Invalid observation or parameter domains
/// should be represented by `f64::INFINITY` rather than panicking, so
/// optimizers can reject the candidate point. `NaN` inputs may propagate as
/// `NaN`; callers can inspect diagnostics for non-finite values.
pub trait Family {
    /// Observation representation consumed by this family.
    ///
    /// Univariate families usually use `f64`. Multivariate, censored,
    /// interval, or mixture families can use small arrays, tuples, or custom
    /// row-view structs without changing the compiled model machinery. The
    /// lifetime parameter allows families to consume borrowed observations,
    /// such as `&'obs [f64]`, without forcing row copies.
    type Observation<'obs>;
    /// Additive predictors on the link scale.
    type Eta;
    /// Distribution parameters on the natural scale.
    type Theta;
    /// Exact gradient of the negative log-likelihood with respect to `Eta`.
    ///
    /// A carrier may preserve an exact factorization useful to a composite
    /// family, such as responsibility times conditional gradient in a finite
    /// mixture. Codecs must materialize the final scalar derivative for every
    /// predictor coordinate before model execution.
    type GradientEta;
    /// Reusable per-family buffers for likelihood evaluation.
    type Workspace;

    /// Creates reusable buffers for this family.
    fn workspace(&self) -> Self::Workspace;

    /// Converts link-scale predictors to distribution parameters.
    ///
    /// Per-parameter links are a convenient way to express independent scalar
    /// constraints, such as positive scales. Dependent constraints between
    /// parameters — for example correlations, covariance factors, ordered
    /// cutpoints, or simplex weights — should be handled here by transforming
    /// the full `Eta` value into a valid natural-scale [`Theta`](Self::Theta).
    /// Predictor blocks should not need to know the statistical geometry of a
    /// particular distribution to produce valid raw predictors.
    fn theta(&self, eta: &Self::Eta, workspace: &mut Self::Workspace) -> Self::Theta;
    /// Negative log-likelihood for one observation on the natural scale.
    fn nll(
        &self,
        observation: Self::Observation<'_>,
        theta: &Self::Theta,
        workspace: &mut Self::Workspace,
    ) -> f64;
    /// Negative log-likelihood for one observation on the link scale.
    fn nll_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        workspace: &mut Self::Workspace,
    ) -> f64 {
        let theta = self.theta(eta, workspace);
        self.nll(observation, &theta, workspace)
    }
    /// Negative log-likelihood and NLL gradient w.r.t. `Eta` for one observation.
    ///
    /// `GradientEta` represents the exact gradient of the negative
    /// log-likelihood with respect to the link-scale predictors `Eta`, after
    /// applying the chain rule for family links. It must map unambiguously to
    /// the same predictor coordinates and ordering as `Eta`; composite carriers
    /// may retain exact factors until their compilation codec materializes
    /// scalar scores.
    fn nll_and_gradient_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta);
}

/// Opt-in codec between a distribution family and the static compiled-model executor.
///
/// [`Family`] remains the complete likelihood contract and does not imply that
/// its runtime configuration has a static predictor topology. Implementing this
/// trait selects one known [`ParameterShape`] and maps its scalar values to and
/// from the family's named `Eta` carriers.
pub trait CompilableFamily: Family {
    /// Static predictor-coordinate geometry for this family.
    type Shape: ParameterShape;

    /// Builds the family's link-scale carrier from shape values.
    fn eta_from_shape(values: ShapeValues<Self::Shape>) -> Self::Eta;

    /// Materializes the final scalar derivative for every shape leaf.
    ///
    /// Implementations must resolve all factors stored in `GradientEta` here;
    /// predictor blocks receive plain per-coordinate scores only.
    fn gradient_to_shape(gradient: &Self::GradientEta) -> ShapeValues<Self::Shape>;

    /// Sample-aware initial link-scale values in shape topology.
    fn initial_shape<'obs, Obs>(&self, _obs: &'obs Obs) -> ShapeValues<Self::Shape>
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        Self::Shape::zeros()
    }

    /// Validates family-instance invariants required by compiled fitting.
    fn validate_compiled(&self) -> Result<(), ModelError> {
        Ok(())
    }
}

/// Exact family-local identity of a runtime predictor topology.
///
/// The key is compared only between instances of the same concrete family
/// type. Its parts must encode every instance setting that changes coordinate
/// ordering or meaning, even when the total coordinate count stays unchanged.
/// It is construction-time metadata and is never inspected in the row hot path.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct DynamicLayoutKey {
    parts: Vec<usize>,
}

impl DynamicLayoutKey {
    /// Creates a family-local layout key from exact structural parts.
    #[must_use]
    #[inline]
    pub const fn new(parts: Vec<usize>) -> Self {
        Self { parts }
    }

    /// Creates a key for a topology determined by one runtime dimension.
    #[must_use]
    #[inline]
    pub fn from_dimension(dimension: usize) -> Self {
        Self::new(vec![dimension])
    }

    /// Returns the exact family-local structural parts.
    #[must_use]
    #[inline]
    pub fn parts(&self) -> &[usize] {
        &self.parts
    }

    /// Consumes the key and returns its structural parts.
    #[must_use]
    #[inline]
    pub fn into_parts(self) -> Vec<usize> {
        self.parts
    }
}

/// Opt-in codec for runtime-dimensional compiled families.
///
/// This is intentionally separate from the const-generic [`ParameterShape`]
/// tree. It keeps runtime dimension explicit while allowing a monomorphic
/// predictor type and allocation-free row likelihood/gradient evaluation.
pub trait DynamicallyCompilableFamily: Family {
    /// Number of scalar predictor coordinates for this family instance.
    fn dynamic_parameter_count(&self) -> usize;

    /// Exact identity of this instance's runtime predictor topology.
    ///
    /// Implementations must include all configuration that affects coordinate
    /// ordering or semantics, not only the coordinate count.
    fn dynamic_layout_key(&self) -> DynamicLayoutKey;

    /// Converts a flat predictor-coordinate row into the family's eta carrier.
    fn eta_from_flat(&self, values: &[f64]) -> Self::Eta;

    /// Writes a materialized family gradient into flat coordinate order.
    fn gradient_to_flat(&self, gradient: &Self::GradientEta, out: &mut [f64]);

    /// Computes NLL directly from flat eta coordinates.
    ///
    /// Runtime-dimensional families must implement this operation without
    /// materializing [`Self::Eta`]. Compiled value and pointwise paths call it
    /// once per active observation and rely on the caller-owned workspace for
    /// reusable storage.
    fn nll_eta_flat(
        &self,
        observation: Self::Observation<'_>,
        values: &[f64],
        workspace: &mut Self::Workspace,
    ) -> f64;

    /// Computes fused NLL and an in-place flat gradient from flat eta coordinates.
    ///
    /// Runtime-dimensional families must implement this hot-path operation
    /// without materializing [`Self::Eta`] or [`Self::GradientEta`]. `gradient`
    /// uses the same coordinate order as `values` and must be fully overwritten.
    fn nll_and_gradient_eta_flat(
        &self,
        observation: Self::Observation<'_>,
        values: &[f64],
        gradient: &mut [f64],
        workspace: &mut Self::Workspace,
    ) -> f64;

    /// Semantic role and nested path for one flat coordinate.
    fn dynamic_parameter_coordinate(&self, index: usize) -> (&'static str, ParameterPath);

    /// Dataset-aware flat eta initializer.
    ///
    /// Implementations must return exactly
    /// [`dynamic_parameter_count`](Self::dynamic_parameter_count) values in the
    /// same coordinate order used by [`eta_from_flat`](Self::eta_from_flat).
    fn initial_flat<'obs, Obs>(&self, _obs: &'obs Obs) -> Vec<f64>
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        vec![0.0; self.dynamic_parameter_count()]
    }

    /// Validates runtime family configuration for compiled fitting.
    fn validate_dynamic_compiled(&self) -> Result<(), ModelError> {
        if self.dynamic_parameter_count() == 0 {
            Err(ModelError::InvalidParameter {
                parameter: "dynamic parameter count",
                expected: "positive",
            })
        } else {
            Ok(())
        }
    }
}

/// Implements [`CompilableFamily`] for an ordinary fixed-arity scalar family.
///
/// This macro is primarily intended for distribution crates. Links deliberately
/// do not appear in the shape: they remain an implementation detail of the
/// family named by the `impl` header.
#[macro_export]
macro_rules! impl_scalar_compilable_family {
    (
        impl for $family:ty;
        parameters = ($($parameter:ty),+ $(,)?);
        arity = $arity:literal $(;)?
    ) => {
        impl $crate::CompilableFamily for $family
        where
            $family: $crate::Family + $crate::InitialEtaFromObservations<$arity>,
            <$family as $crate::Family>::Eta: $crate::ParameterParts<$arity>,
            <$family as $crate::Family>::GradientEta: $crate::ParameterParts<$arity>,
        {
            type Shape = $crate::shape::ScalarTuple<($($parameter,)+), $arity>;

            fn eta_from_shape(values: [f64; $arity]) -> Self::Eta {
                <Self::Eta as $crate::ParameterParts<$arity>>::from_array(values)
            }

            fn gradient_to_shape(gradient: &Self::GradientEta) -> [f64; $arity] {
                std::array::from_fn(|index| {
                    <Self::GradientEta as $crate::ParameterParts<$arity>>::part(gradient, index)
                })
            }

            fn initial_shape<'obs, Obs>(&self, obs: &'obs Obs) -> [f64; $arity]
            where
                Obs: $crate::ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
            {
                let eta = self.initial_eta_from_observations(obs);
                std::array::from_fn(|index| {
                    <Self::Eta as $crate::ParameterParts<$arity>>::part(&eta, index)
                })
            }
        }
    };
    (
        impl<$($generic:ident),+> for $family:ty;
        parameters = ($($parameter:ty),+ $(,)?);
        arity = $arity:literal $(;)?
    ) => {
        impl<$($generic),+> $crate::CompilableFamily for $family
        where
            $family: $crate::Family + $crate::InitialEtaFromObservations<$arity>,
            <$family as $crate::Family>::Eta: $crate::ParameterParts<$arity>,
            <$family as $crate::Family>::GradientEta: $crate::ParameterParts<$arity>,
        {
            type Shape = $crate::shape::ScalarTuple<($($parameter,)+), $arity>;

            #[inline]
            fn eta_from_shape(values: [f64; $arity]) -> Self::Eta {
                <Self::Eta as $crate::ParameterParts<$arity>>::from_array(values)
            }

            #[inline]
            fn gradient_to_shape(gradient: &Self::GradientEta) -> [f64; $arity] {
                std::array::from_fn(|index| {
                    <Self::GradientEta as $crate::ParameterParts<$arity>>::part(gradient, index)
                })
            }

            fn initial_shape<'obs, Obs>(&self, obs: &'obs Obs) -> [f64; $arity]
            where
                Obs: $crate::ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
            {
                let eta = self.initial_eta_from_observations(obs);
                std::array::from_fn(|index| {
                    <Self::Eta as $crate::ParameterParts<$arity>>::part(&eta, index)
                })
            }
        }
    };
}

/// Error returned when a distribution cannot generate a sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimulationError {
    /// Natural-scale parameters are outside the sampler's domain.
    InvalidParameters(&'static str),
    /// The random backend rejected otherwise representable parameters.
    BackendRejected(&'static str),
    /// Generated normalization or arithmetic was non-finite.
    NumericalFailure(&'static str),
}

impl std::fmt::Display for SimulationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (kind, detail) = match self {
            Self::InvalidParameters(detail) => ("invalid simulation parameters", detail),
            Self::BackendRejected(detail) => ("sampling backend rejected parameters", detail),
            Self::NumericalFailure(detail) => ("numerical simulation failure", detail),
        };
        write!(formatter, "{kind}: {detail}")
    }
}

impl std::error::Error for SimulationError {}

/// Extension trait for families that provide diagonal Fisher information.
///
/// Families implementing this trait can be used with Fisher Scoring solvers
/// (Rigby–Stasinopoulos algorithm). The diagonal Fisher information is the
/// expected negative second derivative `E[-∂²ℓ/∂η²]` on the link scale.
///
/// Currently no built-in family implements this trait. It exists as an
/// explicit extension point for future solver integrations — add
/// implementations when you need Fisher Scoring for specific families.
pub trait HasDiagonalFisherInfo: Family {
    /// Negative log-likelihood, NLL gradient, and diagonal Fisher information per
    /// observation on the link scale.
    ///
    /// The `fisher` component must have the same arity and ordering as
    /// [`Family::Eta`] and [`Family::GradientEta`]. Each element is
    /// `E[-∂²ℓ/∂η_k²]`, the expected negative second derivative with
    /// respect to the k-th link-scale predictor, given the observation.
    fn nll_gradient_and_diagonal_fisher_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta, Self::GradientEta);
}

/// Extension trait for families that provide dense expected information.
///
/// This trait covers families whose second-order structure has cross-parameter
/// terms. It is intentionally not wired into [`crate::Gamlss`] evaluation yet;
/// optimizer integrations can opt into it when they need expected information.
pub trait HasExpectedInformation<const K: usize>: Family
where
    Self::Eta: ParameterParts<K>,
    Self::GradientEta: ParameterParts<K>,
{
    /// Negative log-likelihood, NLL gradient, and dense expected information
    /// per observation on the link scale.
    fn nll_gradient_and_expected_information_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta, DenseInformation<K>);
}

/// Marker trait for families with a compile-time fixed observation dimension.
pub trait FixedDimensionalFamily<const D: usize>: Family {}

/// Container for eta or NLL gradient in a family with fixed arity `K`.
///
/// `part(index)` is used in the model hot path after compile-time arity
/// selection. Callers pass `index < K`; implementations may use `unreachable!`
/// for out-of-range indices instead of returning a recoverable error.
pub trait ParameterParts<const K: usize>: Sized {
    /// Assembles a container from `K` scalar parts.
    fn from_array(values: [f64; K]) -> Self;
    /// Returns a scalar part by index.
    ///
    /// Implementations may assume that the caller passes an index less than `K`.
    fn part(&self, index: usize) -> f64;
}

impl ParameterParts<1> for f64 {
    #[inline]
    fn from_array(values: [f64; 1]) -> Self {
        values[0]
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => *self,
            _ => unreachable!("one-parameter parts only have index 0"),
        }
    }
}

impl ParameterParts<2> for (f64, f64) {
    #[inline]
    fn from_array(values: [f64; 2]) -> Self {
        values.into()
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.0,
            1 => self.1,
            _ => unreachable!("two-parameter parts only have indices 0 and 1"),
        }
    }
}

impl ParameterParts<3> for (f64, f64, f64) {
    #[inline]
    fn from_array(values: [f64; 3]) -> Self {
        values.into()
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.0,
            1 => self.1,
            2 => self.2,
            _ => unreachable!("three-parameter parts only have indices 0, 1 and 2"),
        }
    }
}

impl ParameterParts<4> for (f64, f64, f64, f64) {
    #[inline]
    fn from_array(values: [f64; 4]) -> Self {
        values.into()
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.0,
            1 => self.1,
            2 => self.2,
            3 => self.3,
            _ => unreachable!("four-parameter parts only have indices 0, 1, 2 and 3"),
        }
    }
}

impl ParameterParts<5> for (f64, f64, f64, f64, f64) {
    #[inline]
    fn from_array(values: [f64; 5]) -> Self {
        values.into()
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.0,
            1 => self.1,
            2 => self.2,
            3 => self.3,
            4 => self.4,
            _ => unreachable!("five-parameter parts only have indices 0 through 4"),
        }
    }
}

impl ParameterParts<6> for (f64, f64, f64, f64, f64, f64) {
    #[inline]
    fn from_array(values: [f64; 6]) -> Self {
        values.into()
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.0,
            1 => self.1,
            2 => self.2,
            3 => self.3,
            4 => self.4,
            5 => self.5,
            _ => unreachable!("six-parameter parts only have indices 0 through 5"),
        }
    }
}

impl ParameterParts<7> for (f64, f64, f64, f64, f64, f64, f64) {
    #[inline]
    fn from_array(values: [f64; 7]) -> Self {
        values.into()
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.0,
            1 => self.1,
            2 => self.2,
            3 => self.3,
            4 => self.4,
            5 => self.5,
            6 => self.6,
            _ => unreachable!("seven-parameter parts only have indices 0 through 6"),
        }
    }
}

impl ParameterParts<8> for (f64, f64, f64, f64, f64, f64, f64, f64) {
    #[inline]
    fn from_array(values: [f64; 8]) -> Self {
        values.into()
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.0,
            1 => self.1,
            2 => self.2,
            3 => self.3,
            4 => self.4,
            5 => self.5,
            6 => self.6,
            7 => self.7,
            _ => unreachable!("eight-parameter parts only have indices 0 through 7"),
        }
    }
}

/// Sample-aware initialization for scalar-parameter families.
pub trait InitialEtaFromObservations<const K: usize>: Family
where
    Self::Eta: ParameterParts<K>,
    Self::GradientEta: ParameterParts<K>,
{
    /// Sample-aware initial predictors on the link scale.
    fn initial_eta_from_observations<'obs, Obs>(&self, _obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        Self::Eta::from_array([0.0; K])
    }
}

/// Distribution helper for the canonical CDF of a family.
///
/// For scalar families this is the ordinary univariate CDF. For multivariate
/// families this is the joint lower-orthant CDF,
/// `P(Y_1 <= y_1, ..., Y_d <= y_d)`, evaluated at the full observation value.
pub trait HasCdf: Family {
    /// CDF at observation `y` for natural-scale parameters.
    ///
    /// Implementations should return a non-finite value for invalid query
    /// points or parameter domains rather than panicking, matching the base
    /// [`Family`] likelihood contract. For finite query points outside but
    /// below the distribution support, implementations should return the
    /// boundary probability `0.0`.
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64;
}

/// Distribution helper for component-wise marginal CDFs.
///
/// This is separate from [`HasCdf`] because the canonical multivariate CDF is a
/// joint CDF. Marginal CDFs need a component selector for multivariate
/// observations.
pub trait HasMarginalCdf: Family {
    /// Marginal CDF for `component` at scalar point `y`.
    ///
    /// Implementations should return a non-finite value for an invalid
    /// component index or invalid parameter domain rather than panicking.
    fn marginal_cdf(&self, component: usize, y: f64, theta: &Self::Theta) -> f64;
}

impl<F> HasMarginalCdf for F
where
    F: HasCdf + for<'obs> Family<Observation<'obs> = f64>,
{
    #[inline]
    fn marginal_cdf(&self, component: usize, y: f64, theta: &Self::Theta) -> f64 {
        if component == 0 {
            self.cdf(y, theta)
        } else {
            f64::NAN
        }
    }
}

/// Narrow runtime observation-dimension capability.
pub trait HasObservationDimension: Family {
    /// Number of scalar coordinates in one observation.
    fn observation_dimension(&self) -> usize;
}

/// Distribution helper for ordered conditional CDFs.
pub trait HasConditionalCdf: HasObservationDimension {
    /// Evaluates `P(Y_component <= y | Y_0..Y_component-1 = preceding)`.
    ///
    /// Invalid component indices, parameter domains, or insufficient
    /// conditioning values are represented by `NaN`.
    fn conditional_cdf(
        &self,
        component: usize,
        y: f64,
        preceding: &[f64],
        theta: &Self::Theta,
    ) -> f64;
}

/// Distribution helper for ordered Rosenblatt transforms.
pub trait HasRosenblattTransform: HasObservationDimension {
    /// Writes one conditional PIT value per observation coordinate.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::ResponseLength`] when `out` does not match the
    /// family's observation dimension.
    fn rosenblatt_into(
        &self,
        observation: Self::Observation<'_>,
        theta: &Self::Theta,
        out: &mut [f64],
    ) -> Result<(), ModelError>;
}

/// Distribution helper for the quantile function.
pub trait HasQuantile: Family {
    /// Quantile at probability level `p` for natural-scale parameters.
    ///
    /// Implementations should return a non-finite value for invalid
    /// probabilities or parameter domains rather than panicking.
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64;
}

/// Distribution helper for log-density or log-mass evaluation.
///
/// The default implementation reuses the family likelihood contract:
/// `log_density = -nll`. Continuous families expose a log-PDF through this
/// trait, while discrete families expose a log-PMF.
pub trait HasLogDensity: Family {
    /// Log-density or log-mass at `observation` for natural-scale parameters.
    fn log_density(&self, observation: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        let mut workspace = self.workspace();
        -self.nll(observation, theta, &mut workspace)
    }
}

impl<T> HasLogDensity for T where T: Family {}

/// Distribution helper for density or mass evaluation.
///
/// The default implementation exponentiates [`HasLogDensity::log_density`].
/// This may underflow to zero in far tails; fitting code should continue to use
/// [`Family::nll`] and [`Family::nll_and_gradient_eta`].
pub trait HasDensity: HasLogDensity {
    /// Density or mass at `observation` for natural-scale parameters.
    fn density(&self, observation: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        self.log_density(observation, theta).exp()
    }
}

impl<T> HasDensity for T where T: HasLogDensity {}

/// Distribution helper for continuous ranked probability score.
pub trait HasCrps: Family {
    /// CRPS for one observation and natural-scale parameters.
    ///
    /// Implementations should return a non-finite value for invalid
    /// observation or parameter domains rather than panicking.
    fn crps(&self, observation: Self::Observation<'_>, theta: &Self::Theta) -> f64;
}

/// Fallible distribution helper for compositional simulation.
pub trait TrySimulate<Rng>: Family {
    /// Generated sample representation.
    type Sample;

    /// Attempts to generate one sample for natural-scale parameters.
    fn try_sample(
        &self,
        rng: &mut Rng,
        theta: &Self::Theta,
    ) -> Result<Self::Sample, SimulationError>;
}

/// Infallible distribution helper for simulation.
///
/// Composition should prefer [`TrySimulate`]. This trait remains a separate
/// convenience surface because not every sample representation has an honest
/// invalid sentinel.
pub trait CanSimulate<Rng>: Family {
    /// Generated sample representation.
    ///
    /// Scalar families usually use `f64`; multivariate families can use arrays
    /// or custom row-value structs.
    type Sample;
    /// Generates one sample for natural-scale parameters.
    fn sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Self::Sample;
}

/// Distribution helper for per-observation deviance.
///
/// This is intentionally separate from [`Family`] so compiled likelihood
/// evaluation remains minimal. Diagnostics and residual tooling can opt into
/// this trait when a family has a meaningful deviance definition.
pub trait HasDeviance: Family {
    /// Deviance contribution for one observation on the natural parameter scale.
    ///
    /// Implementations should return a non-finite value for invalid observation
    /// or parameter domains rather than panicking, matching the rest of the
    /// family helper contracts.
    fn deviance(&self, observation: Self::Observation<'_>, theta: &Self::Theta) -> f64;
}

/// Distribution helper for family-specific link-scale initialization.
///
/// This gives future fit layers a typed place to ask the family for starting
/// predictors without hard-coding distribution heuristics outside the family
/// implementation.
pub trait HasInitialEta: Family {
    /// Initial link-scale predictors for one observation.
    ///
    /// Implementations should return finite values when the observation is
    /// inside the supported domain. Unsupported or invalid observations may
    /// produce non-finite components instead of panicking.
    fn initial_eta(&self, observation: Self::Observation<'_>) -> Self::Eta;
}
