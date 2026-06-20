use gamlss_core::Penalty;

/// Difference penalty of order `order` for neighboring spline coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DifferencePenalty {
    /// Penalty weight.
    pub lambda: f64,
    /// Finite difference order.
    pub order: usize,
}

impl DifferencePenalty {
    /// Creates a difference penalty.
    #[must_use]
    pub fn new(lambda: f64, order: usize) -> Self {
        Self { lambda, order }
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
}

/// Difference penalty with finite-difference coefficients computed once.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedDifferencePenalty {
    /// Penalty weight.
    pub lambda: f64,
    /// Finite difference order.
    pub order: usize,
    coefficients: Vec<f64>,
}

impl PreparedDifferencePenalty {
    /// Creates a prepared difference penalty.
    #[must_use]
    pub fn new(lambda: f64, order: usize) -> Self {
        Self {
            lambda,
            order,
            coefficients: difference_coefficients(order),
        }
    }

    /// Returns the cached finite-difference coefficients.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
}

impl From<DifferencePenalty> for PreparedDifferencePenalty {
    fn from(value: DifferencePenalty) -> Self {
        Self::new(value.lambda, value.order)
    }
}

impl Penalty for PreparedDifferencePenalty {
    fn value(&self, beta: &[f64]) -> f64 {
        difference_penalty_value(self.lambda, &self.coefficients, beta)
    }

    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        add_difference_penalty_gradient(self.lambda, &self.coefficients, beta, grad);
    }
}

/// Cyclic finite-difference penalty for periodic coefficient vectors.
///
/// Differs from [`DifferencePenalty`] in that differences are taken modulo
/// the vector length (wrap-around).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CyclicDifferencePenalty {
    /// Penalty weight.
    pub lambda: f64,
    /// Finite difference order.
    pub order: usize,
}

impl CyclicDifferencePenalty {
    /// Creates a cyclic difference penalty.
    ///
    /// `lambda` sets the penalty strength, `order` sets the finite-difference
    /// order.
    #[must_use]
    pub fn new(lambda: f64, order: usize) -> Self {
        Self { lambda, order }
    }
}

impl Penalty for CyclicDifferencePenalty {
    fn value(&self, beta: &[f64]) -> f64 {
        let coefficients = difference_coefficients(self.order);
        cyclic_difference_penalty_value(self.lambda, &coefficients, beta)
    }

    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        let coefficients = difference_coefficients(self.order);
        add_cyclic_difference_penalty_gradient(self.lambda, &coefficients, beta, grad);
    }
}

/// Cyclic difference penalty with finite-difference coefficients computed once.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedCyclicDifferencePenalty {
    /// Penalty weight.
    pub lambda: f64,
    /// Finite difference order.
    pub order: usize,
    coefficients: Vec<f64>,
}

impl PreparedCyclicDifferencePenalty {
    /// Creates a prepared cyclic difference penalty.
    #[must_use]
    pub fn new(lambda: f64, order: usize) -> Self {
        Self {
            lambda,
            order,
            coefficients: difference_coefficients(order),
        }
    }

    /// Returns the cached finite-difference coefficients.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
}

impl From<CyclicDifferencePenalty> for PreparedCyclicDifferencePenalty {
    fn from(value: CyclicDifferencePenalty) -> Self {
        Self::new(value.lambda, value.order)
    }
}

impl Penalty for PreparedCyclicDifferencePenalty {
    fn value(&self, beta: &[f64]) -> f64 {
        cyclic_difference_penalty_value(self.lambda, &self.coefficients, beta)
    }

    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        add_cyclic_difference_penalty_gradient(self.lambda, &self.coefficients, beta, grad);
    }
}

/// Quadratic penalty for violating monotonicity at the spline edges.
///
/// Penalizes positive differences `beta[1] - beta[0]` and
/// `beta[n-2] - beta[n-1]`, encouraging decrease at the edges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeMonotonicPenalty {
    /// Penalty weight.
    ///
    /// The larger `weight`, the stronger the penalty for non-monotonicity at
    /// the edges.
    pub weight: f64,
}

impl EdgeMonotonicPenalty {
    /// Creates an edge monotonicity penalty.
    pub fn new(weight: f64) -> Self {
        Self { weight }
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
}

/// Quadratic penalty for exceeding slope limits at the cold/warm edges of a
/// spline.
///
/// Allows constraining the physical slope (e.g. consumption growth with
/// temperature) at the range edges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlopeLimitPenalty {
    /// Penalty weight.
    ///
    /// The larger `weight`, the stronger the penalty.
    pub weight: f64,
    /// Converts edge coefficient differences into a physical slope.
    pub scale: f64,
    /// Optional slope limit on the cold edge.
    pub cold_limit: Option<f64>,
    /// Optional slope limit on the warm edge.
    pub warm_limit: Option<f64>,
}

impl SlopeLimitPenalty {
    /// Creates a slope limit penalty.
    ///
    /// `weight` — penalty strength, `scale` converts coefficient differences
    /// into a physical slope, `cold_limit` and `warm_limit` — optional limits
    /// (if `None`, the corresponding edge is not penalized).
    pub fn new(weight: f64, scale: f64, cold_limit: Option<f64>, warm_limit: Option<f64>) -> Self {
        Self {
            weight,
            scale,
            cold_limit,
            warm_limit,
        }
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
}

fn cyclic_value(values: &[f64], index: usize) -> f64 {
    values[index % values.len()]
}

fn cyclic_difference_penalty_value(lambda: f64, coefficients: &[f64], beta: &[f64]) -> f64 {
    if beta.is_empty() || beta.len() < coefficients.len() {
        return 0.0;
    }

    let n = beta.len();
    let mut sum = 0.0;
    for start in 0..n {
        let diff = coefficients
            .iter()
            .enumerate()
            .map(|(offset, coefficient)| coefficient * cyclic_value(beta, start + offset))
            .sum::<f64>();
        sum += diff * diff;
    }

    lambda * sum / n as f64
}

fn add_cyclic_difference_penalty_gradient(
    lambda: f64,
    coefficients: &[f64],
    beta: &[f64],
    grad: &mut [f64],
) {
    debug_assert_eq!(beta.len(), grad.len());

    if beta.is_empty() || beta.len() < coefficients.len() {
        return;
    }

    let n = beta.len();
    let scale = lambda / n as f64;
    for start in 0..n {
        let diff = coefficients
            .iter()
            .enumerate()
            .map(|(offset, coefficient)| coefficient * cyclic_value(beta, start + offset))
            .sum::<f64>();

        for (offset, coefficient) in coefficients.iter().copied().enumerate() {
            grad[(start + offset) % n] += 2.0 * scale * diff * coefficient;
        }
    }
}

fn difference_penalty_value(lambda: f64, coefficients: &[f64], beta: &[f64]) -> f64 {
    if beta.len() < coefficients.len() {
        return 0.0;
    }

    let mut sum = 0.0;
    for window in beta.windows(coefficients.len()) {
        let diff = coefficients
            .iter()
            .copied()
            .zip(window.iter().copied())
            .map(|(coefficient, beta)| coefficient * beta)
            .sum::<f64>();
        sum += diff * diff;
    }

    lambda * sum
}

fn add_difference_penalty_gradient(
    lambda: f64,
    coefficients: &[f64],
    beta: &[f64],
    grad: &mut [f64],
) {
    debug_assert_eq!(beta.len(), grad.len());

    if beta.len() < coefficients.len() {
        return;
    }

    for (start, beta_window) in beta.windows(coefficients.len()).enumerate() {
        let diff = coefficients
            .iter()
            .copied()
            .zip(beta_window.iter().copied())
            .map(|(coefficient, beta)| coefficient * beta)
            .sum::<f64>();

        for (offset, coefficient) in coefficients.iter().copied().enumerate() {
            grad[start + offset] += 2.0 * lambda * diff * coefficient;
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
    if !penalty.weight.is_finite()
        || penalty.weight <= 0.0
        || !penalty.scale.is_finite()
        || penalty.scale <= 0.0
    {
        return;
    }

    let Some(limit) = (if cold {
        penalty.cold_limit
    } else {
        penalty.warm_limit
    }) else {
        return;
    };
    if !limit.is_finite() || limit < 0.0 {
        return;
    }

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
    *value += penalty.weight * relative * relative;

    if let Some(grad) = grad {
        let sign = if diff >= 0.0 { 1.0 } else { -1.0 };
        let d = 2.0 * penalty.weight * relative * penalty.scale * sign / denominator;
        grad[first] += d;
        grad[second] -= d;
    }
}

/// Finite-difference coefficients of the given order.
///
/// Returns alternating-sign binomial coefficients:
/// `(-1)^{order-i} * C(order, i)`.
fn difference_coefficients(order: usize) -> Vec<f64> {
    (0..=order)
        .map(|index| {
            let sign = if (order - index).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            sign * binomial(order, index) as f64
        })
        .collect()
}

/// Binomial coefficient C(n, k).
fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }

    let k = k.min(n - k);
    (0..k).fold(1, |acc, index| acc * (n - index) / (index + 1))
}
