use crate::model::ObservationView;

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
/// Custom distributions implement this trait. Parameter arity, parameter roles
/// and link-function compatibility are specified through
/// [`ParameterizedFamily`], so the hot path stays typed without dynamic
/// lookup.
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
    type NllGradientEta;

    /// Converts link-scale predictors to distribution parameters.
    ///
    /// Per-parameter links are a convenient way to express independent scalar
    /// constraints, such as positive scales. Dependent constraints between
    /// parameters — for example correlations, covariance factors, ordered
    /// cutpoints, or simplex weights — should be handled here by transforming
    /// the full `Eta` value into a valid natural-scale [`Theta`](Self::Theta).
    fn theta(&self, eta: Self::Eta) -> Self::Theta;
    /// Negative log-likelihood for one observation on the natural scale.
    fn nll(&self, observation: Self::Observation<'_>, theta: Self::Theta) -> f64;
    /// Negative log-likelihood for one observation on the link scale.
    fn nll_eta(&self, observation: Self::Observation<'_>, eta: Self::Eta) -> f64 {
        self.nll(observation, self.theta(eta))
    }
    /// Negative log-likelihood and NLL gradient w.r.t. `Eta` for one observation.
    ///
    /// `NllGradientEta` is the gradient of the negative log-likelihood with
    /// respect to the link-scale predictors `Eta`, after applying the chain
    /// rule for the family links. It must have the same arity and ordering as
    /// `Eta`.
    fn nll_and_gradient_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: Self::Eta,
    ) -> (f64, Self::NllGradientEta);
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
    /// [`Family::Eta`] and [`Family::NllGradientEta`]. Each element is
    /// `E[-∂²ℓ/∂η_k²]`, the expected negative second derivative with
    /// respect to the k-th link-scale predictor, given the observation.
    fn nll_gradient_and_diagonal_fisher_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: Self::Eta,
    ) -> (f64, Self::NllGradientEta, Self::NllGradientEta);
}

/// Extension trait for families that provide dense expected information.
///
/// This trait covers families whose second-order structure has cross-parameter
/// terms. It is intentionally not wired into [`crate::Gamlss`] evaluation yet;
/// optimizer integrations can opt into it when they need expected information.
pub trait HasExpectedInformation<const K: usize>: Family
where
    Self::Eta: ParameterParts<K>,
    Self::NllGradientEta: ParameterParts<K>,
{
    /// Negative log-likelihood, NLL gradient, and dense expected information
    /// per observation on the link scale.
    fn nll_gradient_and_expected_information_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: Self::Eta,
    ) -> (f64, Self::NllGradientEta, DenseInformation<K>);
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

/// Family with a fixed number of parameters, parameter roles and link functions.
///
/// `Params` and `Links` are specified as tuples of the same length as the
/// family arity.
/// Their order defines the order of predictor blocks, gradient parts and flat
/// coefficient ranges in compiled models.
///
/// `Links` describe the independent scalar link contracts for parameter
/// blocks. More complex dependent constraints are still expressed by
/// [`Family::theta`], which sees the full link-scale parameter set.
pub trait ParameterizedFamily<const K: usize>: Family
where
    Self::Eta: ParameterParts<K>,
    Self::NllGradientEta: ParameterParts<K>,
{
    /// Parameter roles of the family.
    type Params;
    /// Link functions of the family parameters.
    type Links;

    /// Sample-aware initial predictors on the link scale.
    ///
    /// Built-in families override this with robust distribution-specific
    /// heuristics. The default keeps custom families source-compatible and
    /// starts all optimizer parameters at zero.
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
    fn cdf(&self, y: Self::Observation<'_>, theta: Self::Theta) -> f64;
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
    fn marginal_cdf(&self, component: usize, y: f64, theta: Self::Theta) -> f64;
}

impl<F> HasMarginalCdf for F
where
    F: HasCdf + for<'obs> Family<Observation<'obs> = f64>,
{
    #[inline]
    fn marginal_cdf(&self, component: usize, y: f64, theta: Self::Theta) -> f64 {
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
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64;
}

/// Distribution helper for log-density or log-mass evaluation.
///
/// The default implementation reuses the family likelihood contract:
/// `log_density = -nll`. Continuous families expose a log-PDF through this
/// trait, while discrete families expose a log-PMF.
pub trait HasLogDensity: Family {
    /// Log-density or log-mass at `observation` for natural-scale parameters.
    fn log_density(&self, observation: Self::Observation<'_>, theta: Self::Theta) -> f64 {
        -self.nll(observation, theta)
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
    fn density(&self, observation: Self::Observation<'_>, theta: Self::Theta) -> f64 {
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
    fn crps(&self, observation: Self::Observation<'_>, theta: Self::Theta) -> f64;
}

/// Distribution helper for simulation.
pub trait CanSimulate<Rng>: Family {
    /// Generated sample representation.
    ///
    /// Scalar families usually use `f64`; multivariate families can use arrays
    /// or custom row-value structs.
    type Sample;
    /// Generates one sample for natural-scale parameters.
    fn sample(&self, rng: &mut Rng, theta: Self::Theta) -> Self::Sample;
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
    fn deviance(&self, observation: Self::Observation<'_>, theta: Self::Theta) -> f64;
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
