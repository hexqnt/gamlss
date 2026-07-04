use std::marker::PhantomData;

use crate::{ParameterName, model::ObservationView};

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

/// Scalar parameter tuple specification for ordinary GAMLSS families.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScalarParams<Params, Links, const K: usize> {
    marker: PhantomData<(Params, Links)>,
}

impl<F, Params, Links, const K: usize> ParamSpec<F> for ScalarParams<Params, Links, K> where
    F: Family
{
}

impl<F, Params, Links, const K: usize> ScalarParamSpec<F, K> for ScalarParams<Params, Links, K>
where
    F: Family,
    F::Eta: ParameterParts<K>,
    F::GradientEta: ParameterParts<K>,
{
    type Params = Params;
    type Links = Links;
}

/// Location vector plus lower-triangular scale-factor parameter-shape specification.
///
/// This is a concrete structured shape for elliptical/location-scale families
/// such as a multivariate normal parameterized by `mu` and a Cholesky scale
/// factor. It is intentionally not the generic multivariate-family
/// abstraction: simplex, copula, shared-factor, sparse-precision, and
/// independent-product constructions should introduce their own [`ParamSpec`]
/// shapes when their parameter geometry differs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LocationCholesky<PVector, PLower, const D: usize> {
    marker: PhantomData<(PVector, PLower)>,
}

impl<F, PVector, PLower, const D: usize> ParamSpec<F> for LocationCholesky<PVector, PLower, D> where
    F: Family
{
}

/// Location vector, marginal scale vector, and strict-lower partial-correlation shape.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LocationScalePartialCorr<PVector, PScale, PCorr, const D: usize> {
    marker: PhantomData<(PVector, PScale, PCorr)>,
}

impl<F, PVector, PScale, PCorr, const D: usize> ParamSpec<F>
    for LocationScalePartialCorr<PVector, PScale, PCorr, D>
where
    F: Family,
{
}

/// Baseline-softmax simplex mean plus scalar precision shape.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeanPrecisionSimplex<PMean, PPrecision, const D: usize> {
    marker: PhantomData<(PMean, PPrecision)>,
}

impl<F, PMean, PPrecision, const D: usize> ParamSpec<F>
    for MeanPrecisionSimplex<PMean, PPrecision, D>
where
    F: Family,
{
}

/// Product of two parameter-shape specifications.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProductSpec<First, Second> {
    marker: PhantomData<(First, Second)>,
}

impl<F, First, Second> ParamSpec<F> for ProductSpec<First, Second> where F: Family {}

/// Baseline-softmax simplex parameter-shape specification.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SimplexWeights<P, const C: usize> {
    marker: PhantomData<P>,
}

impl<F, P, const C: usize> ParamSpec<F> for SimplexWeights<P, C> where F: Family {}

/// Repeated parameter-shape specification.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Repeated<Spec, const C: usize> {
    marker: PhantomData<Spec>,
}

impl<F, Spec, const C: usize> ParamSpec<F> for Repeated<Spec, C> where F: Family {}

/// Homogeneous mixture parameter-shape specification.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MixtureSpec<WeightSpec, ComponentSpec, const C: usize> {
    marker: PhantomData<(WeightSpec, ComponentSpec)>,
}

impl<F, WeightSpec, ComponentSpec, const C: usize> ParamSpec<F>
    for MixtureSpec<WeightSpec, ComponentSpec, C>
where
    F: Family,
{
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
    /// Gradient of the negative log-likelihood with respect to `Eta`.
    type GradientEta;
    /// Reusable per-family buffers for likelihood evaluation.
    type Workspace;
    /// Typed parameter-shape specification used by compiled model blocks.
    type ParamSpec;

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
    /// `GradientEta` is the gradient of the negative log-likelihood with
    /// respect to the link-scale predictors `Eta`, after applying the chain
    /// rule for the family links. It must have the same arity and ordering as
    /// `Eta`.
    fn nll_and_gradient_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta);
}

/// Marker trait for typed model parameter-shape specifications.
pub trait ParamSpec<F: Family + ?Sized> {}

/// Shape contract for ordinary scalar-parameter GAMLSS families.
pub trait ScalarParamSpec<F, const K: usize>: ParamSpec<F>
where
    F: Family,
    F::Eta: ParameterParts<K>,
    F::GradientEta: ParameterParts<K>,
{
    /// Parameter roles in model-block order.
    type Params;
    /// Link functions in model-block order.
    type Links;
}

/// Shape contract for a location vector and lower-triangular scale factor.
///
/// Implement this only for families whose link-scale predictors naturally split
/// into one vector block and one lower-triangular matrix block. Do not use this
/// as a catch-all marker for multivariate distributions with different
/// structure.
pub trait LocationCholeskySpec<F, const D: usize>: ParamSpec<F>
where
    F: Family,
{
    /// Parameter role represented by the vector block.
    type VectorParameter: ParameterName;
    /// Parameter role represented by the lower-triangular block.
    type LowerTriangularParameter: ParameterName;

    /// Assembles link-scale predictors from structured scalar parts.
    fn eta_from_vector_lower(vector: [f64; D], lower: [[f64; D]; D]) -> F::Eta;

    /// Returns one vector-component gradient from a link-scale gradient value.
    fn vector_gradient_part(gradient: &F::GradientEta, component: usize) -> f64;

    /// Returns one lower-triangular gradient entry from a link-scale gradient value.
    fn lower_triangular_gradient_part(gradient: &F::GradientEta, row: usize, col: usize) -> f64;

    /// Sample-aware initial predictors for the vector and lower-triangular parts.
    fn initial_vector_lower_from_observations<'obs, Obs>(
        _family: &F,
        _obs: &'obs Obs,
    ) -> ([f64; D], [[f64; D]; D])
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        ([0.0; D], [[0.0; D]; D])
    }
}

/// Shape contract for `mu`, `sigma`, and strict-lower partial-correlation predictors.
pub trait LocationScalePartialCorrSpec<F, const D: usize>: ParamSpec<F>
where
    F: Family,
{
    /// Parameter role represented by the location vector block.
    type LocationParameter: ParameterName;
    /// Parameter role represented by the marginal scale vector block.
    type ScaleParameter: ParameterName;
    /// Parameter role represented by the strict-lower partial-correlation block.
    type PartialCorrelationParameter: ParameterName;

    /// Assembles link-scale predictors from structured scalar parts.
    fn eta_from_location_scale_partial_corr(
        location: [f64; D],
        scale: [f64; D],
        partial_corr: [[f64; D]; D],
    ) -> F::Eta;

    /// Returns one location-vector gradient component.
    fn location_gradient_part(gradient: &F::GradientEta, component: usize) -> f64;

    /// Returns one scale-vector gradient component.
    fn scale_gradient_part(gradient: &F::GradientEta, component: usize) -> f64;

    /// Returns one strict-lower partial-correlation gradient entry.
    fn partial_corr_gradient_part(gradient: &F::GradientEta, row: usize, col: usize) -> f64;
}

/// Shape contract for baseline-softmax simplex logits plus scalar precision.
pub trait MeanPrecisionSimplexSpec<F, const D: usize>: ParamSpec<F>
where
    F: Family,
{
    /// Parameter role represented by the simplex mean logits.
    type MeanParameter: ParameterName;
    /// Parameter role represented by the scalar precision block.
    type PrecisionParameter: ParameterName;
    /// Link used by the scalar precision block.
    type PrecisionLink;

    /// Assembles link-scale predictors from `D - 1` free logits and scalar precision.
    fn eta_from_simplex_logits_precision(logits: [f64; D], precision: f64) -> F::Eta;

    /// Returns one free-logit gradient component.
    fn simplex_logit_gradient_part(gradient: &F::GradientEta, component: usize) -> f64;

    /// Returns the scalar precision gradient component.
    fn precision_gradient_part(gradient: &F::GradientEta) -> f64;
}

/// Shape contract for a location-Cholesky block product with one scalar block.
pub trait LocationCholeskyScalarSpec<F, const D: usize>: ParamSpec<F>
where
    F: Family,
{
    /// Parameter role represented by the location vector block.
    type VectorParameter: ParameterName;
    /// Parameter role represented by the lower-triangular block.
    type LowerTriangularParameter: ParameterName;
    /// Parameter role represented by the scalar block.
    type ScalarParameter: ParameterName;
    /// Link used by the scalar block.
    type ScalarLink;

    /// Assembles link-scale predictors from structured scalar parts.
    fn eta_from_vector_lower_scalar(vector: [f64; D], lower: [[f64; D]; D], scalar: f64) -> F::Eta;

    /// Returns one vector-component gradient.
    fn vector_gradient_part(gradient: &F::GradientEta, component: usize) -> f64;

    /// Returns one lower-triangular gradient entry.
    fn lower_triangular_gradient_part(gradient: &F::GradientEta, row: usize, col: usize) -> f64;

    /// Returns the scalar gradient.
    fn scalar_gradient_part(gradient: &F::GradientEta) -> f64;
}

/// Shape contract for repeated scalar-parameter component specs.
pub trait RepeatedScalarParamSpec<F, const D: usize, const K: usize>: ParamSpec<F>
where
    F: Family,
    Self::ComponentFamily: Family,
    <Self::ComponentFamily as Family>::Eta: ParameterParts<K>,
    <Self::ComponentFamily as Family>::GradientEta: ParameterParts<K>,
{
    /// Scalar component family repeated by the outer family.
    type ComponentFamily: Family;
    /// Component parameter roles in model-block order.
    type Params;
    /// Component links in model-block order.
    type Links;

    /// Returns the shared component family.
    fn component_family(family: &F) -> &Self::ComponentFamily;

    /// Assembles outer-family link-scale predictors from component predictors.
    fn eta_from_components(components: [<Self::ComponentFamily as Family>::Eta; D]) -> F::Eta;

    /// Returns one component gradient from the outer-family gradient value.
    fn gradient_component(
        gradient: &F::GradientEta,
        component: usize,
    ) -> &<Self::ComponentFamily as Family>::GradientEta;

    /// Sample-aware initial predictors for one repeated component.
    fn initial_component_eta_from_observations<'obs, Obs>(
        _family: &F,
        _obs: &'obs Obs,
        _component: usize,
    ) -> <Self::ComponentFamily as Family>::Eta
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
        Self::ComponentFamily: InitialEtaFromObservations<K>,
    {
        <Self::ComponentFamily as Family>::Eta::from_array([0.0; K])
    }
}

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
        (values[0], values[1])
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
        (values[0], values[1], values[2])
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
        (values[0], values[1], values[2], values[3])
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
        (values[0], values[1], values[2], values[3], values[4])
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
        (
            values[0], values[1], values[2], values[3], values[4], values[5],
        )
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
        (
            values[0], values[1], values[2], values[3], values[4], values[5], values[6],
        )
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
        (
            values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
        )
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

/// Marker extension trait for families with conditional CDF support.
///
/// This is reserved for multivariate diagnostics and forecasting APIs that
/// need conditional probability statements rather than a joint CDF.
pub trait HasConditionalCdf: Family {}

/// Marker extension trait for families with Rosenblatt transform support.
///
/// This is reserved for multivariate PIT and residual diagnostics where the
/// joint CDF is not a scalar PIT substitute.
pub trait HasRosenblattTransform: Family {}

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

/// Distribution helper for simulation.
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
