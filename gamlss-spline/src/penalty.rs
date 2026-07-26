use gamlss_core::{MatrixPenalty, ModelError, Penalty};

const EXPECTED_FINITE_POSITIVE: &str = "finite and > 0";
const EXPECTED_FINITE_NONNEGATIVE: &str = "finite and >= 0";
const EXPECTED_DIFFERENCE_ORDER_FOR_DIM: &str = "> 0 and < dimension";
const EXPECTED_DIFFERENCE_COEFFICIENTS: &str = "consistent with difference penalty order";
const FIRST_DIFFERENCE_COEFFICIENTS: [f64; 2] = [-1.0, 1.0];
const SECOND_DIFFERENCE_COEFFICIENTS: [f64; 3] = [1.0, -2.0, 1.0];

/// Difference penalty of order `order` for neighboring spline coefficients.
///
/// For a coefficient slice $\boldsymbol\beta=(\beta_0,\ldots,\beta_{n-1})$ and difference order $m<n$, define
///
/// $$
/// \Delta^m\beta_i = \sum_{j=0}^{m}(-1)^{m-j}\binom{m}{j}\beta_{i+j}.
/// $$
///
/// Here $\binom{m}{j}$ is a binomial coefficient.
///
/// The penalty value is the mean squared non-wrapping difference,
///
/// $$
/// J_m(\boldsymbol\beta)= \frac{\lambda}{n-m}\sum_{i=0}^{n-m-1}\left(\Delta^m\beta_i\right)^2.
/// $$
///
/// In code, $n=\mathtt{beta.len()}$, $m$ is [`DifferencePenalty::order`], and $\lambda$ is [`DifferencePenalty::lambda`]. The $n-m$ denominator is the number of non-wrapping differences and makes $\lambda$ approximately scale-stable as $n$ changes.
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DifferencePenalty {
    /// Penalty weight.
    lambda: f64,
    /// Finite difference order.
    order: usize,
}

impl DifferencePenalty {
    /// Creates a difference penalty without validating penalty parameters.
    ///
    /// The penalty value is `lambda * mean(diff^2)` over non-wrapping
    /// neighboring differences, so `lambda` is approximately scale-stable as
    /// the basis size changes.
    ///
    /// By contract, `lambda` should be finite and non-negative, `order` should
    /// be positive, and its binomial coefficients should fit in `usize`.
    #[must_use]
    pub const fn new_unchecked(lambda: f64, order: usize) -> Self {
        Self { lambda, order }
    }

    /// Creates a difference penalty with validated scalar parameters.
    ///
    /// Uses the same normalized scale convention as [`Self::new_unchecked`].
    /// The block-size invariant `order < dim` is checked by
    /// [`Penalty::validate_dim`] during model validation.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `lambda` is not finite or
    /// is negative, or when `order` is zero. Returns
    /// [`ModelError::ArithmeticOverflow`] when the finite-difference
    /// coefficients cannot be represented without integer overflow.
    pub fn try_new(lambda: f64, order: usize) -> Result<Self, ModelError> {
        validate_difference_lambda(lambda)?;
        validate_difference_order(order)?;
        Ok(Self::new_unchecked(lambda, order))
    }

    /// Returns the penalty weight.
    #[must_use]
    pub const fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Returns the finite-difference order.
    #[must_use]
    pub const fn order(&self) -> usize {
        self.order
    }

    fn coefficients(&self) -> Vec<f64> {
        difference_coefficients(self.order)
    }
}

impl Penalty for DifferencePenalty {
    fn value(&self, beta: &[f64]) -> f64 {
        let coefficients = self.coefficients();
        difference_penalty_value(self.lambda, &coefficients, beta)
    }

    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        let coefficients = self.coefficients();
        add_difference_penalty_gradient(self.lambda, &coefficients, beta, grad);
    }

    fn validate_dim(&self, dim: usize) -> Result<(), ModelError> {
        validate_difference_penalty_for_dim(self.lambda, self.order, dim)
    }
}

impl MatrixPenalty for DifferencePenalty {
    fn add_penalty_matrix(&self, dim: usize, gram: &mut [f64]) {
        let coefficients = self.coefficients();
        add_difference_penalty_matrix(self.lambda, &coefficients, dim, gram);
    }
}

/// Difference penalty with finite-difference coefficients computed once.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedDifferencePenalty {
    lambda: f64,
    order: usize,
    coefficients: Vec<f64>,
}

impl PreparedDifferencePenalty {
    /// Creates a prepared difference penalty without validating penalty
    /// parameters.
    ///
    /// The penalty value is `lambda * mean(diff^2)` over non-wrapping
    /// neighboring differences, so `lambda` is approximately scale-stable as
    /// the basis size changes.
    ///
    /// By contract, `lambda` should be finite and non-negative, `order` should
    /// be positive, and its binomial coefficients should fit in `usize`.
    ///
    /// # Panics
    ///
    /// Panics when `order` violates this contract.
    #[must_use]
    pub fn new_unchecked(lambda: f64, order: usize) -> Self {
        Self {
            lambda,
            order,
            coefficients: difference_coefficients(order),
        }
    }

    /// Creates a prepared difference penalty with validated scalar parameters.
    ///
    /// Uses the same normalized scale convention as [`Self::new_unchecked`].
    /// The block-size invariant `order < dim` is checked by
    /// [`Penalty::validate_dim`] during model validation.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `lambda` is not finite or
    /// is negative, or when `order` is zero. Returns
    /// [`ModelError::ArithmeticOverflow`] when the finite-difference
    /// coefficients cannot be represented without integer overflow.
    pub fn try_new(lambda: f64, order: usize) -> Result<Self, ModelError> {
        let coefficients = validated_difference_coefficients(lambda, order)?;
        Ok(Self {
            lambda,
            order,
            coefficients,
        })
    }

    /// Returns the penalty weight.
    #[must_use]
    pub const fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Returns the finite-difference order.
    #[must_use]
    pub const fn order(&self) -> usize {
        self.order
    }

    /// Returns the cached finite-difference coefficients.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
}

impl TryFrom<DifferencePenalty> for PreparedDifferencePenalty {
    type Error = ModelError;

    fn try_from(value: DifferencePenalty) -> Result<Self, Self::Error> {
        Self::try_new(value.lambda(), value.order())
    }
}

impl Penalty for PreparedDifferencePenalty {
    fn value(&self, beta: &[f64]) -> f64 {
        difference_penalty_value(self.lambda, &self.coefficients, beta)
    }

    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        add_difference_penalty_gradient(self.lambda, &self.coefficients, beta, grad);
    }

    fn validate_dim(&self, dim: usize) -> Result<(), ModelError> {
        validate_prepared_difference_penalty_for_dim(
            self.lambda,
            self.order,
            &self.coefficients,
            dim,
        )
    }
}

impl MatrixPenalty for PreparedDifferencePenalty {
    fn add_penalty_matrix(&self, dim: usize, gram: &mut [f64]) {
        add_difference_penalty_matrix(self.lambda, &self.coefficients, dim, gram);
    }
}

/// Cyclic finite-difference penalty for periodic coefficient vectors.
///
/// For the same $\boldsymbol\beta$, $m$, and $\lambda$ notation as [`DifferencePenalty`], cyclic indexing replaces $\beta_{i+j}$ with $\beta_{(i+j)\bmod n}$ and includes one difference starting at every coefficient:
///
/// $$
/// J_m^{\mathrm{cyclic}}(\boldsymbol\beta)
/// = \frac{\lambda}{n}
///   \sum_{i=0}^{n-1}
///   \left[
///     \sum_{j=0}^{m}(-1)^{m-j}\binom{m}{j}
///     \beta_{(i+j)\bmod n}
///   \right]^2.
/// $$
///
/// The denominator is $n$ because there is one wrapped difference per coefficient. In code, $m$ and $\lambda$ are [`CyclicDifferencePenalty::order`] and [`CyclicDifferencePenalty::lambda`].
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CyclicDifferencePenalty {
    /// Penalty weight.
    lambda: f64,
    /// Finite difference order.
    order: usize,
}

impl CyclicDifferencePenalty {
    /// Creates a cyclic difference penalty without validating penalty
    /// parameters.
    ///
    /// The cyclic penalty is normalized by the number of coefficients, so its
    /// `lambda` is approximately scale-stable as the basis size changes. This
    /// matches [`DifferencePenalty`]'s normalized convention, except cyclic
    /// differences include the wrap-around rows.
    ///
    /// By contract, `lambda` should be finite and non-negative, `order` should
    /// be positive, and its binomial coefficients should fit in `usize`.
    #[must_use]
    pub const fn new_unchecked(lambda: f64, order: usize) -> Self {
        Self { lambda, order }
    }

    /// Creates a cyclic difference penalty with validated scalar parameters.
    ///
    /// Uses the same normalized scale convention as [`Self::new_unchecked`].
    /// The block-size invariant `order < dim` is checked by
    /// [`Penalty::validate_dim`] during model validation.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `lambda` is not finite or
    /// is negative, or when `order` is zero. Returns
    /// [`ModelError::ArithmeticOverflow`] when the finite-difference
    /// coefficients cannot be represented without integer overflow.
    pub fn try_new(lambda: f64, order: usize) -> Result<Self, ModelError> {
        validate_difference_lambda(lambda)?;
        validate_difference_order(order)?;
        Ok(Self::new_unchecked(lambda, order))
    }

    /// Returns the penalty weight.
    #[must_use]
    pub const fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Returns the finite-difference order.
    #[must_use]
    pub const fn order(&self) -> usize {
        self.order
    }

    fn coefficients(&self) -> Vec<f64> {
        difference_coefficients(self.order)
    }
}

impl Penalty for CyclicDifferencePenalty {
    fn value(&self, beta: &[f64]) -> f64 {
        let coefficients = self.coefficients();
        cyclic_difference_penalty_value(self.lambda, &coefficients, beta)
    }

    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        let coefficients = self.coefficients();
        add_cyclic_difference_penalty_gradient(self.lambda, &coefficients, beta, grad);
    }

    fn validate_dim(&self, dim: usize) -> Result<(), ModelError> {
        validate_difference_penalty_for_dim(self.lambda, self.order, dim)
    }
}

impl MatrixPenalty for CyclicDifferencePenalty {
    fn add_penalty_matrix(&self, dim: usize, gram: &mut [f64]) {
        let coefficients = self.coefficients();
        add_cyclic_difference_penalty_matrix(self.lambda, &coefficients, dim, gram);
    }
}

/// Cyclic difference penalty with finite-difference coefficients computed once.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedCyclicDifferencePenalty {
    lambda: f64,
    order: usize,
    coefficients: Vec<f64>,
}

impl PreparedCyclicDifferencePenalty {
    /// Creates a prepared cyclic difference penalty without validating penalty
    /// parameters.
    ///
    /// The cyclic penalty is normalized by the number of coefficients, so its
    /// `lambda` is approximately scale-stable as the basis size changes. This
    /// matches [`PreparedDifferencePenalty`]'s normalized convention, except
    /// cyclic differences include the wrap-around rows.
    ///
    /// By contract, `lambda` should be finite and non-negative, `order` should
    /// be positive, and its binomial coefficients should fit in `usize`.
    ///
    /// # Panics
    ///
    /// Panics when `order` violates this contract.
    #[must_use]
    pub fn new_unchecked(lambda: f64, order: usize) -> Self {
        Self {
            lambda,
            order,
            coefficients: difference_coefficients(order),
        }
    }

    /// Creates a prepared cyclic difference penalty with validated scalar
    /// parameters.
    ///
    /// Uses the same normalized scale convention as [`Self::new_unchecked`].
    /// The block-size invariant `order < dim` is checked by
    /// [`Penalty::validate_dim`] during model validation.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `lambda` is not finite or
    /// is negative, or when `order` is zero. Returns
    /// [`ModelError::ArithmeticOverflow`] when the finite-difference
    /// coefficients cannot be represented without integer overflow.
    pub fn try_new(lambda: f64, order: usize) -> Result<Self, ModelError> {
        let coefficients = validated_difference_coefficients(lambda, order)?;
        Ok(Self {
            lambda,
            order,
            coefficients,
        })
    }

    /// Returns the penalty weight.
    #[must_use]
    pub const fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Returns the finite-difference order.
    #[must_use]
    pub const fn order(&self) -> usize {
        self.order
    }

    /// Returns the cached finite-difference coefficients.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
}

impl TryFrom<CyclicDifferencePenalty> for PreparedCyclicDifferencePenalty {
    type Error = ModelError;

    fn try_from(value: CyclicDifferencePenalty) -> Result<Self, Self::Error> {
        Self::try_new(value.lambda(), value.order())
    }
}

impl Penalty for PreparedCyclicDifferencePenalty {
    fn value(&self, beta: &[f64]) -> f64 {
        cyclic_difference_penalty_value(self.lambda, &self.coefficients, beta)
    }

    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        add_cyclic_difference_penalty_gradient(self.lambda, &self.coefficients, beta, grad);
    }

    fn validate_dim(&self, dim: usize) -> Result<(), ModelError> {
        validate_prepared_difference_penalty_for_dim(
            self.lambda,
            self.order,
            &self.coefficients,
            dim,
        )
    }
}

impl MatrixPenalty for PreparedCyclicDifferencePenalty {
    fn add_penalty_matrix(&self, dim: usize, gram: &mut [f64]) {
        add_cyclic_difference_penalty_matrix(self.lambda, &self.coefficients, dim, gram);
    }
}

/// Quadratic penalty for violating monotonicity at the spline edges.
///
/// Penalizes positive differences `beta[1] - beta[0]` and
/// `beta[n-2] - beta[n-1]`, encouraging decrease at the edges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeMonotonicPenalty {
    weight: f64,
}

impl EdgeMonotonicPenalty {
    /// Creates an edge monotonicity penalty.
    ///
    /// This constructor is unchecked. Use [`Self::try_new`] when `weight`
    /// comes from user input or dynamic configuration.
    #[must_use]
    #[inline]
    pub const fn new(weight: f64) -> Self {
        Self { weight }
    }

    /// Creates an edge monotonicity penalty with a finite positive weight.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `weight` is not finite or
    /// is not positive.
    #[inline]
    pub fn try_new(weight: f64) -> Result<Self, ModelError> {
        validate_positive_finite("penalty weight", weight)?;
        Ok(Self::new(weight))
    }

    /// Returns the penalty weight.
    #[must_use]
    #[inline]
    pub const fn weight(&self) -> f64 {
        self.weight
    }
}

impl Penalty for EdgeMonotonicPenalty {
    fn value(&self, beta: &[f64]) -> f64 {
        if beta.len() < 2 {
            return 0.0;
        }
        let left = (beta[1] - beta[0]).max(0.0);
        let right = (beta[beta.len() - 2] - beta[beta.len() - 1]).max(0.0);
        self.weight * (left * left + right * right)
    }

    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(beta.len(), grad.len());
        if beta.len() < 2 {
            return;
        }

        let left = (beta[1] - beta[0]).max(0.0);
        if left > 0.0 {
            let d = 2.0 * self.weight * left;
            grad[0] -= d;
            grad[1] += d;
        }

        let last = beta.len() - 1;
        let prev = beta.len() - 2;
        let right = (beta[prev] - beta[last]).max(0.0);
        if right > 0.0 {
            let d = 2.0 * self.weight * right;
            grad[prev] += d;
            grad[last] -= d;
        }
    }

    fn validate_dim(&self, _dim: usize) -> Result<(), ModelError> {
        validate_positive_finite("penalty weight", self.weight)
    }
}

/// Quadratic penalty for exceeding slope limits at the cold/warm edges of a
/// spline.
///
/// Allows constraining the physical slope (e.g. consumption growth with
/// temperature) at the range edges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlopeLimitPenalty {
    weight: f64,
    scale: f64,
    cold_limit: Option<f64>,
    warm_limit: Option<f64>,
}

impl SlopeLimitPenalty {
    /// Creates a slope limit penalty with validated scalar parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `weight` or `scale` is
    /// not finite and positive, or when a provided limit is not finite and
    /// non-negative.
    #[inline]
    pub fn try_new(
        weight: f64,
        scale: f64,
        cold_limit: Option<f64>,
        warm_limit: Option<f64>,
    ) -> Result<Self, ModelError> {
        validate_positive_finite("penalty weight", weight)?;
        validate_positive_finite("penalty scale", scale)?;
        validate_limit("cold penalty limit", cold_limit)?;
        validate_limit("warm penalty limit", warm_limit)?;
        Ok(Self {
            weight,
            scale,
            cold_limit,
            warm_limit,
        })
    }

    /// Returns the penalty weight.
    #[must_use]
    #[inline]
    pub const fn weight(&self) -> f64 {
        self.weight
    }

    /// Returns the physical-slope multiplier.
    #[must_use]
    #[inline]
    pub const fn scale(&self) -> f64 {
        self.scale
    }

    /// Returns the optional cold-edge slope limit.
    #[must_use]
    #[inline]
    pub const fn cold_limit(&self) -> Option<f64> {
        self.cold_limit
    }

    /// Returns the optional warm-edge slope limit.
    #[must_use]
    #[inline]
    pub const fn warm_limit(&self) -> Option<f64> {
        self.warm_limit
    }
}

impl Penalty for SlopeLimitPenalty {
    fn value(&self, beta: &[f64]) -> f64 {
        let mut value = 0.0;
        add_slope_limit_value(beta, self, true, &mut value, None);
        add_slope_limit_value(beta, self, false, &mut value, None);
        value
    }

    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(beta.len(), grad.len());
        let mut value = 0.0;
        add_slope_limit_value(beta, self, true, &mut value, Some(&mut *grad));
        add_slope_limit_value(beta, self, false, &mut value, Some(&mut *grad));
    }

    fn validate_dim(&self, _dim: usize) -> Result<(), ModelError> {
        validate_positive_finite("penalty weight", self.weight)?;
        validate_positive_finite("penalty scale", self.scale)?;
        validate_limit("cold penalty limit", self.cold_limit)?;
        validate_limit("warm penalty limit", self.warm_limit)
    }
}

const fn cyclic_value(values: &[f64], index: usize) -> f64 {
    values[index % values.len()]
}

#[allow(clippy::cast_precision_loss)]
const fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

fn normalized_scale(lambda: f64, n_differences: usize) -> f64 {
    lambda / usize_to_f64(n_differences)
}

fn non_cyclic_difference_count(dim: usize, coefficients: &[f64]) -> Option<usize> {
    dim.checked_sub(coefficients.len())
        .map(|last_start| last_start + 1)
}

fn cyclic_difference_at(coefficients: &[f64], beta: &[f64], start: usize) -> f64 {
    coefficients
        .iter()
        .enumerate()
        .map(|(offset, coefficient)| coefficient * cyclic_value(beta, start + offset))
        .sum()
}

fn non_cyclic_difference_at(coefficients: &[f64], beta_window: &[f64]) -> f64 {
    coefficients
        .iter()
        .copied()
        .zip(beta_window.iter().copied())
        .map(|(coefficient, beta)| coefficient * beta)
        .sum()
}

fn cyclic_difference_penalty_value(lambda: f64, coefficients: &[f64], beta: &[f64]) -> f64 {
    if lambda == 0.0 {
        return 0.0;
    }
    if beta.is_empty() || beta.len() < coefficients.len() {
        return 0.0;
    }

    let n_differences = beta.len();
    let mut sum = 0.0;
    for start in 0..n_differences {
        let diff = cyclic_difference_at(coefficients, beta, start);
        sum = diff.mul_add(diff, sum);
    }

    normalized_scale(lambda, n_differences) * sum
}

fn add_cyclic_difference_penalty_gradient(
    lambda: f64,
    coefficients: &[f64],
    beta: &[f64],
    grad: &mut [f64],
) {
    debug_assert_eq!(beta.len(), grad.len());

    if lambda == 0.0 || beta.is_empty() || beta.len() < coefficients.len() {
        return;
    }

    if coefficients == FIRST_DIFFERENCE_COEFFICIENTS {
        add_cyclic_first_difference_penalty_gradient(lambda, beta, grad);
        return;
    }
    if coefficients == SECOND_DIFFERENCE_COEFFICIENTS {
        add_cyclic_second_difference_penalty_gradient(lambda, beta, grad);
        return;
    }

    let n_differences = beta.len();
    let scale = normalized_scale(lambda, n_differences);
    for start in 0..n_differences {
        let diff = cyclic_difference_at(coefficients, beta, start);

        for (offset, coefficient) in coefficients.iter().copied().enumerate() {
            let index = (start + offset) % n_differences;
            grad[index] = (2.0 * scale * diff).mul_add(coefficient, grad[index]);
        }
    }
}

fn add_cyclic_difference_penalty_matrix(
    lambda: f64,
    coefficients: &[f64],
    dim: usize,
    gram: &mut [f64],
) {
    debug_assert_eq!(dim.checked_mul(dim), Some(gram.len()));

    if dim == 0 || dim < coefficients.len() {
        return;
    }

    let scale = 2.0 * normalized_scale(lambda, dim);
    for start in 0..dim {
        add_difference_outer_product(scale, coefficients, dim, gram, |offset| {
            (start + offset) % dim
        });
    }
}

pub(crate) fn difference_penalty_value(lambda: f64, coefficients: &[f64], beta: &[f64]) -> f64 {
    if lambda == 0.0 {
        return 0.0;
    }
    if beta.len() < coefficients.len() {
        return 0.0;
    }

    let n_differences =
        non_cyclic_difference_count(beta.len(), coefficients).expect("length already checked");
    let mut sum = 0.0;
    for window in beta.windows(coefficients.len()) {
        let diff = non_cyclic_difference_at(coefficients, window);
        sum = diff.mul_add(diff, sum);
    }

    normalized_scale(lambda, n_differences) * sum
}

pub(crate) fn add_difference_penalty_gradient(
    lambda: f64,
    coefficients: &[f64],
    beta: &[f64],
    grad: &mut [f64],
) {
    debug_assert_eq!(beta.len(), grad.len());

    if lambda == 0.0 || beta.len() < coefficients.len() {
        return;
    }

    if coefficients == FIRST_DIFFERENCE_COEFFICIENTS {
        add_first_difference_penalty_gradient(lambda, beta, grad);
        return;
    }
    if coefficients == SECOND_DIFFERENCE_COEFFICIENTS {
        add_second_difference_penalty_gradient(lambda, beta, grad);
        return;
    }

    let n_differences =
        non_cyclic_difference_count(beta.len(), coefficients).expect("length already checked");
    let scale = normalized_scale(lambda, n_differences);
    for (start, beta_window) in beta.windows(coefficients.len()).enumerate() {
        let diff = non_cyclic_difference_at(coefficients, beta_window);

        for (offset, coefficient) in coefficients.iter().copied().enumerate() {
            let index = start + offset;
            grad[index] = (2.0 * scale * diff).mul_add(coefficient, grad[index]);
        }
    }
}

fn add_cyclic_first_difference_penalty_gradient(lambda: f64, beta: &[f64], grad: &mut [f64]) {
    debug_assert_eq!(beta.len(), grad.len());
    debug_assert!(beta.len() >= 2);

    let len = beta.len();
    let scale = 2.0 * normalized_scale(lambda, len);
    let last = len - 1;
    for index in 0..len {
        let previous = if index == 0 {
            beta[last]
        } else {
            beta[index - 1]
        };
        let next = if index == last {
            beta[0]
        } else {
            beta[index + 1]
        };
        let left = beta[index] - previous;
        let right = next - beta[index];
        grad[index] = scale.mul_add(left - right, grad[index]);
    }
}

fn add_cyclic_second_difference_penalty_gradient(lambda: f64, beta: &[f64], grad: &mut [f64]) {
    debug_assert_eq!(beta.len(), grad.len());
    debug_assert!(beta.len() >= 3);

    let len = beta.len();
    let scale = 2.0 * normalized_scale(lambda, len);
    let middle_scale = -2.0 * scale;
    for start in 0..len {
        let middle = wrap_offset(start, 1, len);
        let right = wrap_offset(start, 2, len);
        let diff = (beta[right] - beta[middle]) - (beta[middle] - beta[start]);
        grad[start] = scale.mul_add(diff, grad[start]);
        grad[middle] = middle_scale.mul_add(diff, grad[middle]);
        grad[right] = scale.mul_add(diff, grad[right]);
    }
}

fn add_first_difference_penalty_gradient(lambda: f64, beta: &[f64], grad: &mut [f64]) {
    debug_assert_eq!(beta.len(), grad.len());
    debug_assert!(beta.len() >= 2);

    let len = beta.len();
    let n_differences = len - 1;
    let scale = 2.0 * normalized_scale(lambda, n_differences);
    let last = len - 1;

    grad[0] = scale.mul_add(beta[0] - beta[1], grad[0]);
    for index in 1..last {
        let left = beta[index] - beta[index - 1];
        let right = beta[index + 1] - beta[index];
        grad[index] = scale.mul_add(left - right, grad[index]);
    }
    grad[last] = scale.mul_add(beta[last] - beta[last - 1], grad[last]);
}

fn add_second_difference_penalty_gradient(lambda: f64, beta: &[f64], grad: &mut [f64]) {
    debug_assert_eq!(beta.len(), grad.len());
    debug_assert!(beta.len() >= 3);

    let n_differences = beta.len() - 2;
    let scale = 2.0 * normalized_scale(lambda, n_differences);
    let middle_scale = -2.0 * scale;

    for start in 0..n_differences {
        let middle = start + 1;
        let right = start + 2;
        let diff = (beta[right] - beta[middle]) - (beta[middle] - beta[start]);
        grad[start] = scale.mul_add(diff, grad[start]);
        grad[middle] = middle_scale.mul_add(diff, grad[middle]);
        grad[right] = scale.mul_add(diff, grad[right]);
    }
}

#[inline]
const fn wrap_offset(index: usize, offset: usize, len: usize) -> usize {
    let wrapped = index + offset;
    if wrapped >= len {
        wrapped - len
    } else {
        wrapped
    }
}

fn add_difference_penalty_matrix(lambda: f64, coefficients: &[f64], dim: usize, gram: &mut [f64]) {
    debug_assert_eq!(dim.checked_mul(dim), Some(gram.len()));

    let Some(n_differences) = non_cyclic_difference_count(dim, coefficients) else {
        return;
    };

    let scale = 2.0 * normalized_scale(lambda, n_differences);
    for start in 0..n_differences {
        add_difference_outer_product(scale, coefficients, dim, gram, |offset| start + offset);
    }
}

fn add_difference_outer_product<F>(
    scale: f64,
    coefficients: &[f64],
    dim: usize,
    gram: &mut [f64],
    index: F,
) where
    F: Fn(usize) -> usize,
{
    for (left_offset, left) in coefficients.iter().copied().enumerate() {
        let row = index(left_offset);
        for (right_offset, right) in coefficients.iter().copied().enumerate() {
            let col = index(right_offset);
            let index = row * dim + col;
            gram[index] = (scale * left).mul_add(right, gram[index]);
        }
    }
}

/// Adds a penalty for exceeding the slope limit on one edge.
///
/// `cold = true` — cold edge (start), `false` — warm edge (end).
/// If `grad` is provided, also adds the gradient contribution.
fn add_slope_limit_value(
    beta: &[f64],
    penalty: &SlopeLimitPenalty,
    cold: bool,
    value: &mut f64,
    grad: Option<&mut [f64]>,
) {
    if beta.len() < 2 {
        return;
    }
    let Some(limit) = (if cold {
        penalty.cold_limit
    } else {
        penalty.warm_limit
    }) else {
        return;
    };
    let (first, second) = if cold {
        (0, 1)
    } else {
        (beta.len() - 1, beta.len() - 2)
    };
    let diff = beta[first] - beta[second];
    let abs_slope = penalty.scale * diff.abs();
    let excess = abs_slope - limit;
    if excess <= 0.0 {
        return;
    }

    let denominator = limit.max(1.0e-12);
    let relative = (excess / denominator).min(1.0e6);
    *value = (penalty.weight * relative).mul_add(relative, *value);

    if let Some(grad) = grad {
        let sign = if diff >= 0.0 { 1.0 } else { -1.0 };
        let d = 2.0 * penalty.weight * relative * penalty.scale * sign / denominator;
        grad[first] += d;
        grad[second] -= d;
    }
}

fn validate_positive_finite(parameter: &'static str, value: f64) -> Result<(), ModelError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(ModelError::InvalidParameter {
            parameter,
            expected: EXPECTED_FINITE_POSITIVE,
        })
    }
}

fn validate_nonnegative_finite(parameter: &'static str, value: f64) -> Result<(), ModelError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(ModelError::InvalidParameter {
            parameter,
            expected: EXPECTED_FINITE_NONNEGATIVE,
        })
    }
}

fn validate_limit(parameter: &'static str, value: Option<f64>) -> Result<(), ModelError> {
    match value {
        Some(value) if !value.is_finite() || value < 0.0 => Err(ModelError::InvalidParameter {
            parameter,
            expected: EXPECTED_FINITE_NONNEGATIVE,
        }),
        _ => Ok(()),
    }
}

fn validate_difference_lambda(lambda: f64) -> Result<(), ModelError> {
    validate_nonnegative_finite("penalty lambda", lambda)?;
    Ok(())
}

fn validate_difference_order(order: usize) -> Result<(), ModelError> {
    if order == 0 {
        return Err(ModelError::InvalidParameter {
            parameter: "difference penalty order",
            expected: "> 0",
        });
    }

    for index in 0..=order {
        checked_binomial(order, index)?;
    }
    Ok(())
}

fn validated_difference_coefficients(lambda: f64, order: usize) -> Result<Vec<f64>, ModelError> {
    validate_difference_lambda(lambda)?;
    try_difference_coefficients(order)
}

fn validate_difference_penalty_for_dim(
    lambda: f64,
    order: usize,
    dim: usize,
) -> Result<(), ModelError> {
    validate_difference_lambda(lambda)?;
    validate_difference_order(order)?;
    validate_difference_order_for_dim(order, dim)
}

fn validate_prepared_difference_penalty_for_dim(
    lambda: f64,
    order: usize,
    coefficients: &[f64],
    dim: usize,
) -> Result<(), ModelError> {
    validate_difference_penalty_for_dim(lambda, order, dim)?;
    let expected = try_difference_coefficients(order)?;
    if coefficients == expected {
        Ok(())
    } else {
        Err(ModelError::InvalidParameter {
            parameter: "difference penalty coefficients",
            expected: EXPECTED_DIFFERENCE_COEFFICIENTS,
        })
    }
}

pub(crate) const fn validate_difference_order_for_dim(
    order: usize,
    dim: usize,
) -> Result<(), ModelError> {
    if order < dim {
        Ok(())
    } else {
        Err(ModelError::InvalidParameter {
            parameter: "difference penalty order",
            expected: EXPECTED_DIFFERENCE_ORDER_FOR_DIM,
        })
    }
}

/// Finite-difference coefficients of the given order.
///
/// Returns alternating-sign binomial coefficients:
/// `(-1)^{order-i} * C(order, i)`.
fn difference_coefficients(order: usize) -> Vec<f64> {
    try_difference_coefficients(order)
        .expect("difference penalty order must have binomial coefficients that fit in usize")
}

pub(crate) fn try_difference_coefficients(order: usize) -> Result<Vec<f64>, ModelError> {
    validate_difference_order(order)?;
    (0..=order)
        .map(|index| {
            let sign = if (order - index).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            checked_binomial(order, index).map(|coefficient| sign * usize_to_f64(coefficient))
        })
        .collect()
}

fn checked_binomial(n: usize, k: usize) -> Result<usize, ModelError> {
    if k > n {
        return Ok(0);
    }

    let k = k.min(n - k);
    let coefficient = (0..k).try_fold(1u128, |acc, index| {
        acc.checked_mul((n - index) as u128)
            .map(|product| product / (index + 1) as u128)
            .ok_or(ModelError::ArithmeticOverflow {
                context: "difference penalty coefficients",
            })
    })?;
    usize::try_from(coefficient).map_err(|_| ModelError::ArithmeticOverflow {
        context: "difference penalty coefficients",
    })
}

#[cfg(test)]
mod tests {
    use gamlss_core::{ModelError, Penalty};

    use super::{
        CyclicDifferencePenalty, DifferencePenalty, EdgeMonotonicPenalty,
        PreparedCyclicDifferencePenalty, PreparedDifferencePenalty,
    };

    fn invalid_parameter(parameter: &'static str, expected: &'static str) -> ModelError {
        ModelError::InvalidParameter {
            parameter,
            expected,
        }
    }

    #[test]
    fn difference_penalty_validation_rejects_invalid_unchecked_parameters() {
        assert_eq!(
            DifferencePenalty::new_unchecked(f64::NAN, 1)
                .validate_dim(3)
                .unwrap_err(),
            invalid_parameter("penalty lambda", "finite and >= 0")
        );
        assert_eq!(
            DifferencePenalty::new_unchecked(1.0, 0)
                .validate_dim(3)
                .unwrap_err(),
            invalid_parameter("difference penalty order", "> 0")
        );
        assert_eq!(
            DifferencePenalty::new_unchecked(1.0, 3)
                .validate_dim(3)
                .unwrap_err(),
            invalid_parameter("difference penalty order", "> 0 and < dimension")
        );
    }

    #[test]
    fn cyclic_difference_penalty_validation_rejects_order_too_high() {
        assert_eq!(
            CyclicDifferencePenalty::new_unchecked(1.0, 2)
                .validate_dim(2)
                .unwrap_err(),
            invalid_parameter("difference penalty order", "> 0 and < dimension")
        );
    }

    #[test]
    fn prepared_difference_penalty_validation_checks_cached_coefficients() {
        let invalid = PreparedDifferencePenalty {
            lambda: 1.0,
            order: 2,
            coefficients: vec![1.0, -1.0],
        };

        assert_eq!(
            invalid.validate_dim(4).unwrap_err(),
            invalid_parameter(
                "difference penalty coefficients",
                "consistent with difference penalty order",
            )
        );
    }

    #[test]
    fn prepared_cyclic_difference_penalty_validation_checks_cached_coefficients() {
        let invalid = PreparedCyclicDifferencePenalty {
            lambda: 1.0,
            order: 1,
            coefficients: vec![1.0, 1.0],
        };

        assert_eq!(
            invalid.validate_dim(3).unwrap_err(),
            invalid_parameter(
                "difference penalty coefficients",
                "consistent with difference penalty order",
            )
        );
    }

    #[test]
    fn edge_penalty_validation_rejects_unchecked_invalid_weight() {
        assert_eq!(
            EdgeMonotonicPenalty::new(f64::NAN)
                .validate_dim(3)
                .unwrap_err(),
            invalid_parameter("penalty weight", "finite and > 0")
        );
    }
}
