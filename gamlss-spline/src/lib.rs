#![forbid(unsafe_code)]
//! Spline-базисы, spline design matrices и штрафы.

use gamlss_core::{DenseDesign, ModelError, Penalty, PredictorBlock};
use thiserror::Error;

/// Поддерживаемые степени/порядки сплайнов для эффективного локального
/// вычисления базиса.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplineOrder {
    /// Linear B-spline.
    Linear = 1,
    /// Quadratic B-spline.
    Quadratic = 2,
    /// Cubic B-spline.
    Cubic = 3,
}

impl SplineOrder {
    /// Степень полинома.
    pub fn degree(self) -> usize {
        self as usize
    }

    /// Минимальное число коэффициентов для данного порядка.
    pub fn min_basis(self) -> usize {
        self.degree() + 1
    }
}

/// Ошибки построения spline basis и spline design matrix.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SplineError {
    /// Входной вектор пуст.
    #[error("spline input must contain at least one value")]
    EmptyInput,

    /// Входной вектор содержит `NaN` или infinity.
    #[error("spline input contains a non-finite value")]
    NonFiniteValue,

    /// Диапазон данных не имеет двух различных конечных границ.
    #[error("spline range must have distinct finite boundaries")]
    InvalidRange,

    /// Число basis-функций недостаточно для степени spline.
    #[error("B-spline basis count {n_basis} must be greater than degree {degree}")]
    NotEnoughBasis {
        /// Запрошенное число basis-функций.
        n_basis: usize,
        /// Степень B-spline.
        degree: usize,
    },

    /// Knot vector содержит не-finite значения или убывает.
    #[error("knot vector must be finite and nondecreasing")]
    InvalidKnots,

    /// Ошибка core design matrix.
    #[error(transparent)]
    Model(#[from] ModelError),
}

/// B-spline basis с заданной степенью и knot vector.
#[derive(Debug, Clone, PartialEq)]
pub struct BSplineBasis {
    degree: usize,
    knots: Vec<f64>,
}

impl BSplineBasis {
    /// Создаёт basis из готового knot vector.
    ///
    /// Knot vector должен быть конечным, неубывающим и достаточно длинным для
    /// выбранной степени.
    pub fn new(degree: usize, knots: Vec<f64>) -> Result<Self, SplineError> {
        if knots.len() <= degree + 1 {
            return Err(SplineError::NotEnoughBasis { n_basis: 0, degree });
        }

        if knots
            .windows(2)
            .any(|window| !window[0].is_finite() || !window[1].is_finite() || window[0] > window[1])
        {
            return Err(SplineError::InvalidKnots);
        }

        Ok(Self { degree, knots })
    }

    /// Строит open uniform B-spline basis по диапазону данных.
    pub fn open_uniform_from_data(
        x: &[f64],
        n_basis: usize,
        degree: usize,
    ) -> Result<Self, SplineError> {
        if x.is_empty() {
            return Err(SplineError::EmptyInput);
        }
        if n_basis <= degree {
            return Err(SplineError::NotEnoughBasis { n_basis, degree });
        }

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for value in x.iter().copied() {
            if !value.is_finite() {
                return Err(SplineError::NonFiniteValue);
            }
            min = min.min(value);
            max = max.max(value);
        }

        if min >= max {
            return Err(SplineError::InvalidRange);
        }

        let interior = n_basis.saturating_sub(degree + 1);
        let mut knots = Vec::with_capacity(n_basis + degree + 1);
        knots.extend(std::iter::repeat_n(min, degree + 1));

        for index in 1..=interior {
            let fraction = index as f64 / (interior + 1) as f64;
            knots.push(min + fraction * (max - min));
        }

        knots.extend(std::iter::repeat_n(max, degree + 1));
        Self::new(degree, knots)
    }

    /// Степень spline.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Knot vector.
    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    /// Число basis-функций.
    pub fn n_basis(&self) -> usize {
        self.knots.len() - self.degree - 1
    }

    /// Значения всех basis-функций в точке `x`.
    pub fn evaluate(&self, x: f64) -> Vec<f64> {
        (0..self.n_basis())
            .map(|index| self.basis_value(index, self.degree, x))
            .collect()
    }

    /// Dense design matrix, где каждая строка содержит `evaluate(x_i)`.
    pub fn design_matrix(&self, x: &[f64]) -> Result<DenseDesign, SplineError> {
        if x.iter().any(|value| !value.is_finite()) {
            return Err(SplineError::NonFiniteValue);
        }

        let n_basis = self.n_basis();
        let mut values = Vec::with_capacity(x.len() * n_basis);
        for value in x.iter().copied() {
            values.extend(self.evaluate(value));
        }

        Ok(DenseDesign::from_row_major(x.len(), n_basis, values)?)
    }

    fn basis_value(&self, index: usize, degree: usize, x: f64) -> f64 {
        if degree == 0 {
            let left = self.knots[index];
            let right = self.knots[index + 1];
            let is_last_basis = index + 1 == self.n_basis();
            if (left <= x && x < right) || (is_last_basis && x == right) {
                1.0
            } else {
                0.0
            }
        } else {
            let mut value = 0.0;
            let left_denom = self.knots[index + degree] - self.knots[index];
            if left_denom > 0.0 {
                value +=
                    (x - self.knots[index]) / left_denom * self.basis_value(index, degree - 1, x);
            }

            let right_denom = self.knots[index + degree + 1] - self.knots[index + 1];
            if right_denom > 0.0 {
                value += (self.knots[index + degree + 1] - x) / right_denom
                    * self.basis_value(index + 1, degree - 1, x);
            }

            value
        }
    }
}

/// Вспомогательная функция для open uniform P-spline design matrix.
pub fn pspline_design(
    x: &[f64],
    n_basis: usize,
    degree: usize,
) -> Result<DenseDesign, SplineError> {
    BSplineBasis::open_uniform_from_data(x, n_basis, degree)?.design_matrix(x)
}

/// Open-uniform spline predictor с локальным sparse вычислением строк.
///
/// В отличие от [`BSplineBasis`], хранит исходные данные и вычисляет
/// базисные функции «на лету» через компактный [`LocalBasis`], не
/// материализуя полную design matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenUniformSplineDesign {
    x: Vec<f64>,
    min: f64,
    span: f64,
    n_basis: usize,
    order: SplineOrder,
    n_intervals: f64,
}

impl OpenUniformSplineDesign {
    /// Строит open-uniform spline design по диапазону данных.
    ///
    /// Если данные пусты или содержат не-finite значения, возвращает ошибку.
    pub fn from_data(x: &[f64], n_basis: usize, order: SplineOrder) -> Result<Self, SplineError> {
        if x.is_empty() {
            return Err(SplineError::EmptyInput);
        }
        if n_basis < order.min_basis() {
            return Err(SplineError::NotEnoughBasis {
                n_basis,
                degree: order.degree(),
            });
        }

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for value in x.iter().copied() {
            if !value.is_finite() {
                return Err(SplineError::NonFiniteValue);
            }
            min = min.min(value);
            max = max.max(value);
        }
        Self::with_range(x, min, max, n_basis, order)
    }

    /// Строит open-uniform spline design с явным конечным диапазоном.
    ///
    /// `min` и `max` должны быть конечными и `min < max`.
    pub fn with_range(
        x: &[f64],
        min: f64,
        max: f64,
        n_basis: usize,
        order: SplineOrder,
    ) -> Result<Self, SplineError> {
        if x.iter().any(|value| !value.is_finite()) {
            return Err(SplineError::NonFiniteValue);
        }
        if !min.is_finite() || !max.is_finite() || min >= max {
            return Err(SplineError::InvalidRange);
        }
        if n_basis < order.min_basis() {
            return Err(SplineError::NotEnoughBasis {
                n_basis,
                degree: order.degree(),
            });
        }

        Ok(Self {
            x: x.to_vec(),
            min,
            span: max - min,
            n_basis,
            order,
            n_intervals: (n_basis - order.degree()).max(1) as f64,
        })
    }

    /// Число spline-коэффициентов.
    pub fn n_basis(&self) -> usize {
        self.n_basis
    }

    fn basis_for_row(&self, row: usize) -> LocalBasis {
        let u = (self.x[row] - self.min) / self.span;
        open_uniform_local_basis(u, self.order, self.n_basis, self.n_intervals)
    }
}

impl PredictorBlock for OpenUniformSplineDesign {
    fn nrows(&self) -> usize {
        self.x.len()
    }

    fn nparams(&self) -> usize {
        self.n_basis
    }

    fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
        let basis = self.basis_for_row(row);
        basis.dot(beta)
    }

    fn add_gradient(&self, scores: &[f64], _: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(scores.len(), self.x.len());
        debug_assert_eq!(grad.len(), self.n_basis);

        for (row, score) in scores.iter().copied().enumerate() {
            self.basis_for_row(row).add_scaled(score, grad);
        }
    }
}

/// Cyclic spline predictor для периодических ковариат на `[0, 1)`.
#[derive(Debug, Clone, PartialEq)]
pub struct CyclicSplineDesign {
    phi: Vec<f64>,
    n_basis: usize,
    order: SplineOrder,
}

impl CyclicSplineDesign {
    /// Строит cyclic spline design.
    ///
    /// Все значения `phi` должны быть конечными.
    pub fn new(phi: &[f64], n_basis: usize, order: SplineOrder) -> Result<Self, SplineError> {
        if phi.iter().any(|value| !value.is_finite()) {
            return Err(SplineError::NonFiniteValue);
        }
        if n_basis < order.min_basis() {
            return Err(SplineError::NotEnoughBasis {
                n_basis,
                degree: order.degree(),
            });
        }
        Ok(Self {
            phi: phi.to_vec(),
            n_basis,
            order,
        })
    }

    /// Number of spline coefficients.
    pub fn n_basis(&self) -> usize {
        self.n_basis
    }

    fn basis_for_row(&self, row: usize) -> LocalBasis {
        cyclic_local_basis(self.phi[row], self.order, self.n_basis)
    }
}

impl PredictorBlock for CyclicSplineDesign {
    fn nrows(&self) -> usize {
        self.phi.len()
    }

    fn nparams(&self) -> usize {
        self.n_basis
    }

    fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
        self.basis_for_row(row).dot(beta)
    }

    fn add_gradient(&self, scores: &[f64], _: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(scores.len(), self.phi.len());
        debug_assert_eq!(grad.len(), self.n_basis);

        for (row, score) in scores.iter().copied().enumerate() {
            self.basis_for_row(row).add_scaled(score, grad);
        }
    }
}

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
        for start in 0..=beta.len() - coefficients.len() {
            let diff = coefficients
                .iter()
                .enumerate()
                .map(|(offset, coefficient)| coefficient * beta[start + offset])
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

        for start in 0..=beta.len() - coefficients.len() {
            let diff = coefficients
                .iter()
                .enumerate()
                .map(|(offset, coefficient)| coefficient * beta[start + offset])
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

        let mut sum = 0.0;
        for start in 0..beta.len() {
            let diff = coefficients
                .iter()
                .enumerate()
                .map(|(offset, coefficient)| coefficient * beta[(start + offset) % beta.len()])
                .sum::<f64>();
            sum += diff * diff;
        }

        self.lambda * sum / beta.len() as f64
    }

    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(beta.len(), grad.len());

        let coefficients = difference_coefficients(self.order);
        if beta.is_empty() || beta.len() < coefficients.len() {
            return;
        }

        let scale = self.lambda / beta.len() as f64;
        for start in 0..beta.len() {
            let diff = coefficients
                .iter()
                .enumerate()
                .map(|(offset, coefficient)| coefficient * beta[(start + offset) % beta.len()])
                .sum::<f64>();

            for (offset, coefficient) in coefficients.iter().copied().enumerate() {
                grad[(start + offset) % beta.len()] += 2.0 * scale * diff * coefficient;
            }
        }
    }
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

/// Локальный базис для одной строки — компактное sparse-представление.
///
/// Хранит до 4 ненулевых индексов и весов, достаточных для кубического
/// сплайна. Используется в `eta_row` и `add_gradient` predictor-ов.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct LocalBasis {
    indices: [usize; 4],
    weights: [f64; 4],
    len: usize,
}

impl LocalBasis {
    /// Скалярное произведение базиса на коэффициенты.
    fn dot(self, beta: &[f64]) -> f64 {
        let mut value = 0.0;
        for idx in 0..self.len {
            value += beta[self.indices[idx]] * self.weights[idx];
        }
        value
    }

    /// Добавляет `scale * weights[i]` в `out[indices[i]]` для каждого
    /// ненулевого элемента базиса.
    fn add_scaled(self, scale: f64, out: &mut [f64]) {
        for idx in 0..self.len {
            out[self.indices[idx]] += scale * self.weights[idx];
        }
    }
}

/// Вычисляет локальный базис open-uniform сплайна для нормированной
/// координаты `u` в диапазоне данных.
///
/// При `u <= 0` или `u >= 1` возвращает линейную экстраполяцию.
fn open_uniform_local_basis(
    u: f64,
    order: SplineOrder,
    n_basis: usize,
    n_intervals: f64,
) -> LocalBasis {
    let degree = order.degree();

    if u <= 0.0 {
        return edge_extrapolation_basis(u, degree, n_basis, n_intervals, false);
    }
    if u >= 1.0 {
        return edge_extrapolation_basis(u - 1.0, degree, n_basis, n_intervals, true);
    }

    let span = open_uniform_span(u, n_basis, degree);
    let weights = open_uniform_basis_funs(span, u, n_basis, degree);
    let start = span - degree;
    let mut basis = LocalBasis {
        len: degree + 1,
        ..LocalBasis::default()
    };
    for (offset, weight) in weights.iter().copied().enumerate().take(degree + 1) {
        basis.indices[offset] = start + offset;
        basis.weights[offset] = weight;
    }
    basis
}

/// Вычисляет локальный базис циклического сплайна для фазы `phi`.
///
/// `phi` приводится к `[0, 1)` через `rem_euclid`.
fn cyclic_local_basis(phi: f64, order: SplineOrder, n_basis: usize) -> LocalBasis {
    let x = phi.rem_euclid(1.0) * n_basis as f64;
    let cell = x.floor() as usize;
    let u = x - cell as f64;
    let weights = spline_weights(order, u);
    let len = order.degree() + 1;
    let offset = if order == SplineOrder::Cubic {
        n_basis - 1
    } else {
        0
    };
    let mut basis = LocalBasis {
        len,
        ..LocalBasis::default()
    };
    for (idx, weight) in weights.iter().copied().enumerate().take(len) {
        basis.indices[idx] = (cell + offset + idx) % n_basis;
        basis.weights[idx] = weight;
    }
    basis
}

/// Линейная экстраполяция сплайна за границами диапазона данных.
///
/// Использует первые/последние два контрольных коэффициента для
/// продолжения сплайна за `[0, 1)` с сохранением непрерывности.
fn edge_extrapolation_basis(
    offset: f64,
    degree: usize,
    n_basis: usize,
    n_intervals: f64,
    right: bool,
) -> LocalBasis {
    let slope_scale = degree as f64 * n_intervals;
    if right {
        LocalBasis {
            indices: [n_basis - 2, n_basis - 1, 0, 0],
            weights: [-slope_scale * offset, 1.0 + slope_scale * offset, 0.0, 0.0],
            len: 2,
        }
    } else {
        LocalBasis {
            indices: [0, 1, 0, 0],
            weights: [1.0 - slope_scale * offset, slope_scale * offset, 0.0, 0.0],
            len: 2,
        }
    }
}

/// Находит span (индекс контрольной точки) для open-uniform сплайна
/// бинарным поиском.
fn open_uniform_span(u: f64, n_basis: usize, degree: usize) -> usize {
    let last_control = n_basis - 1;
    let mut low = degree;
    let mut high = n_basis;
    let mut mid = (low + high) / 2;
    while u < open_uniform_knot(mid, n_basis, degree)
        || u >= open_uniform_knot(mid + 1, n_basis, degree)
    {
        if u < open_uniform_knot(mid, n_basis, degree) {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) / 2;
    }
    mid.min(last_control)
}

/// Вычисляет веса B-spline базисных функций в заданном span.
///
/// Использует рекуррентный алгоритм Кокса-де Бура.
fn open_uniform_basis_funs(span: usize, u: f64, n_basis: usize, degree: usize) -> [f64; 4] {
    let mut weights = [0.0; 4];
    let mut left = [0.0; 4];
    let mut right = [0.0; 4];
    weights[0] = 1.0;
    for j in 1..=degree {
        left[j] = u - open_uniform_knot(span + 1 - j, n_basis, degree);
        right[j] = open_uniform_knot(span + j, n_basis, degree) - u;
        let mut saved = 0.0;
        for r in 0..j {
            let denominator = right[r + 1] + left[j - r];
            let temp = if denominator == 0.0 {
                0.0
            } else {
                weights[r] / denominator
            };
            weights[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        weights[j] = saved;
    }
    weights
}

/// Возвращает нормализованную позицию узла open-uniform сплайна.
///
/// Узлы равномерно распределены между 0 и 1 с кратными граничными узлами.
fn open_uniform_knot(index: usize, n_basis: usize, degree: usize) -> f64 {
    if index <= degree {
        0.0
    } else if index >= n_basis {
        1.0
    } else {
        (index - degree) as f64 / (n_basis - degree) as f64
    }
}

/// Веса локального сплайна (linear, quadratic, cubic) по параметру `u`.
fn spline_weights(order: SplineOrder, u: f64) -> [f64; 4] {
    match order {
        SplineOrder::Linear => [1.0 - u, u, 0.0, 0.0],
        SplineOrder::Quadratic => {
            let u2 = u * u;
            [
                (1.0 - u) * (1.0 - u) / 2.0,
                (1.0 + 2.0 * u - 2.0 * u2) / 2.0,
                u2 / 2.0,
                0.0,
            ]
        }
        SplineOrder::Cubic => {
            let mu = 1.0 - u;
            let mu2 = mu * mu;
            let mu3 = mu2 * mu;
            let u2 = u * u;
            let u3 = u2 * u;
            [
                mu3 / 6.0,
                (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0,
                (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0,
                u3 / 6.0,
            ]
        }
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

/// Наиболее часто используемые импорты из `gamlss-spline`.
pub mod prelude {
    pub use crate::{
        BSplineBasis, CyclicDifferencePenalty, CyclicSplineDesign, DifferencePenalty,
        EdgeMonotonicPenalty, OpenUniformSplineDesign, SlopeLimitPenalty, SplineError, SplineOrder,
        pspline_design,
    };
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{Penalty, PredictorBlock};

    use super::{
        BSplineBasis, CyclicDifferencePenalty, CyclicSplineDesign, DifferencePenalty,
        EdgeMonotonicPenalty, OpenUniformSplineDesign, SlopeLimitPenalty, SplineOrder,
    };

    #[test]
    fn open_uniform_bspline_partitions_unity_inside_range() {
        let x = [0.0, 0.25, 0.5, 0.75, 1.0];
        let basis = BSplineBasis::open_uniform_from_data(&x, 6, 3).unwrap();

        for value in x {
            let sum = basis.evaluate(value).iter().sum::<f64>();
            assert_relative_eq!(sum, 1.0, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn difference_penalty_gradient_matches_finite_difference() {
        let penalty = DifferencePenalty::new(0.7, 2);
        let beta = vec![0.2, -0.4, 0.9, 1.1, -0.3];
        let eps = 1.0e-6;
        let mut grad = vec![0.0; beta.len()];

        penalty.add_gradient(&beta, &mut grad);

        for index in 0..beta.len() {
            let mut plus = beta.clone();
            plus[index] += eps;
            let mut minus = beta.clone();
            minus[index] -= eps;
            let finite_difference = (penalty.value(&plus) - penalty.value(&minus)) / (2.0 * eps);

            assert_relative_eq!(grad[index], finite_difference, epsilon = 1.0e-6);
        }
    }

    #[test]
    fn cyclic_spline_design_wraps_and_partitions_unity() {
        let design =
            CyclicSplineDesign::new(&[0.0, 0.25, 1.0, -0.25], 8, SplineOrder::Cubic).unwrap();
        let beta = vec![1.0; design.nparams()];

        for row in 0..design.nrows() {
            assert_relative_eq!(design.eta_row(row, &beta), 1.0, epsilon = 1.0e-12);
        }

        let ramp = (0..8).map(|value| value as f64).collect::<Vec<_>>();
        assert_relative_eq!(design.eta_row(0, &ramp), design.eta_row(2, &ramp));
    }

    #[test]
    fn open_uniform_spline_design_handles_boundaries_and_extrapolation() {
        let design = OpenUniformSplineDesign::with_range(
            &[-0.5, 0.0, 0.5, 1.0, 1.5],
            0.0,
            1.0,
            6,
            SplineOrder::Cubic,
        )
        .unwrap();
        let beta = vec![1.0; design.nparams()];

        for row in 0..design.nrows() {
            assert_relative_eq!(design.eta_row(row, &beta), 1.0, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn cyclic_difference_penalty_gradient_matches_finite_difference() {
        let penalty = CyclicDifferencePenalty::new(0.7, 2);
        let beta = vec![0.2, -0.4, 0.9, 1.1, -0.3];
        assert_penalty_gradient_matches_finite_difference(&penalty, &beta);
    }

    #[test]
    fn edge_monotonic_penalty_gradient_matches_finite_difference() {
        let penalty = EdgeMonotonicPenalty::new(3.0);
        let beta = vec![0.2, 0.8, 0.4, -0.1];
        assert_penalty_gradient_matches_finite_difference(&penalty, &beta);
    }

    #[test]
    fn slope_limit_penalty_gradient_matches_finite_difference() {
        let penalty = SlopeLimitPenalty::new(5.0, 2.0, Some(0.4), Some(0.3));
        let beta = vec![0.6, 0.1, -0.1, 0.4];
        assert_penalty_gradient_matches_finite_difference(&penalty, &beta);
    }

    #[test]
    fn slope_limit_penalty_ignores_invalid_inputs() {
        let cases = [
            SlopeLimitPenalty::new(f64::NAN, 2.0, Some(0.4), Some(0.3)),
            SlopeLimitPenalty::new(5.0, f64::NAN, Some(0.4), Some(0.3)),
            SlopeLimitPenalty::new(5.0, 2.0, Some(f64::NAN), Some(-0.3)),
        ];
        let beta = vec![0.6, 0.1, -0.1, 0.4];

        for penalty in cases {
            let mut grad = vec![0.0; beta.len()];
            assert_eq!(penalty.value(&beta), 0.0);
            penalty.add_gradient(&beta, &mut grad);
            assert_eq!(grad, vec![0.0; beta.len()]);
        }
    }

    fn assert_penalty_gradient_matches_finite_difference<P>(penalty: &P, beta: &[f64])
    where
        P: Penalty,
    {
        let eps = 1.0e-6;
        let mut grad = vec![0.0; beta.len()];

        penalty.add_gradient(beta, &mut grad);

        for index in 0..beta.len() {
            let mut plus = beta.to_vec();
            plus[index] += eps;
            let mut minus = beta.to_vec();
            minus[index] -= eps;
            let finite_difference = (penalty.value(&plus) - penalty.value(&minus)) / (2.0 * eps);

            assert_relative_eq!(grad[index], finite_difference, epsilon = 1.0e-6);
        }
    }
}
