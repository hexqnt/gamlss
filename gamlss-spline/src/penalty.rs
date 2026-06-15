use gamlss_core::Penalty;

/// Difference penalty порядка `order` для соседних spline coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DifferencePenalty {
    /// Вес penalty.
    pub lambda: f64,
    /// Порядок finite difference.
    pub order: usize,
}

impl DifferencePenalty {
    /// Создаёт difference penalty.
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

        self.lambda * sum
    }

    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(beta.len(), grad.len());

        let coefficients = self.coefficients();
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
                grad[start + offset] += 2.0 * self.lambda * diff * coefficient;
            }
        }
    }
}

/// Циклический finite-difference penalty для периодических векторов
/// коэффициентов.
///
/// Отличается от [`DifferencePenalty`] тем, что разности берутся по
/// модулю длины вектора (wrap-around).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CyclicDifferencePenalty {
    /// Вес penalty.
    pub lambda: f64,
    /// Порядок finite difference.
    pub order: usize,
}

impl CyclicDifferencePenalty {
    /// Создаёт cyclic difference penalty.
    ///
    /// `lambda` задаёт силу штрафа, `order` — порядок конечной разности.
    pub fn new(lambda: f64, order: usize) -> Self {
        Self { lambda, order }
    }
}

impl Penalty for CyclicDifferencePenalty {
    fn value(&self, beta: &[f64]) -> f64 {
        let coefficients = difference_coefficients(self.order);
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

        self.lambda * sum / n as f64
    }

    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(beta.len(), grad.len());

        let coefficients = difference_coefficients(self.order);
        if beta.is_empty() || beta.len() < coefficients.len() {
            return;
        }

        let n = beta.len();
        let scale = self.lambda / n as f64;
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
}

fn cyclic_value(values: &[f64], index: usize) -> f64 {
    values[index % values.len()]
}

/// Квадратичный штраф за нарушение монотонности на краях сплайна.
///
/// Штрафует положительные разности `beta[1] - beta[0]` и
/// `beta[n-2] - beta[n-1]`, поощряя убывание на краях.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeMonotonicPenalty {
    /// Вес penalty.
    ///
    /// Чем больше `weight`, тем сильнее штраф за немонотонность на краях.
    pub weight: f64,
}

impl EdgeMonotonicPenalty {
    /// Создаёт edge monotonicity penalty.
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

/// Квадратичный штраф за превышение пределов наклона на холодном/тёплом
/// краях сплайна.
///
/// Позволяет ограничить физический наклон (например, рост потребления
/// с температурой) на краях диапазона.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlopeLimitPenalty {
    /// Вес penalty.
    ///
    /// Чем больше `weight`, тем сильнее штраф.
    pub weight: f64,
    /// Переводит разности краевых коэффициентов в физический наклон.
    pub scale: f64,
    /// Опциональный предел наклона на холодном краю.
    pub cold_limit: Option<f64>,
    /// Опциональный предел наклона на тёплом краю.
    pub warm_limit: Option<f64>,
}

impl SlopeLimitPenalty {
    /// Создаёт slope limit penalty.
    ///
    /// `weight` — сила штрафа, `scale` переводит разности коэффициентов
    /// в физический наклон, `cold_limit` и `warm_limit` — опциональные
    /// пределы (если `None`, соответствующий край не штрафуется).
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

/// Добавляет штраф за превышение предела наклона на одном краю.
///
/// `cold = true` — холодный край (начало), `false` — тёплый (конец).
/// Если `grad` передан, добавляет и градиентную составляющую.
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

/// Коэффициенты конечной разности заданного порядка.
///
/// Возвращает знакочередующиеся биномиальные коэффициенты:
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

/// Биномиальный коэффициент C(n, k).
fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }

    let k = k.min(n - k);
    (0..k).fold(1, |acc, index| acc * (n - index) / (index + 1))
}
