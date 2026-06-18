use crate::SplineOrder;

/// Локальный базис для одной строки — компактное sparse-представление.
///
/// Хранит до 4 ненулевых индексов и весов, достаточных для кубического
/// сплайна. Используется в `eta_row` и `add_gradient` predictor-ов.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct LocalBasis {
    indices: [usize; 4],
    weights: [f64; 4],
    len: usize,
}

impl LocalBasis {
    /// Visits non-zero elements of the local basis.
    pub(crate) fn for_each(self, mut f: impl FnMut(usize, f64)) {
        for (&index, &weight) in self.indices[..self.len]
            .iter()
            .zip(&self.weights[..self.len])
        {
            f(index, weight);
        }
    }

    /// Скалярное произведение базиса на коэффициенты.
    pub(crate) fn dot(self, beta: &[f64]) -> f64 {
        let mut value = 0.0;
        self.for_each(|index, weight| {
            value += beta[index] * weight;
        });
        value
    }

    /// Добавляет `scale * weights[i]` в `out[indices[i]]` для каждого
    /// ненулевого элемента базиса.
    pub(crate) fn add_scaled(self, scale: f64, out: &mut [f64]) {
        self.for_each(|index, weight| {
            out[index] += scale * weight;
        });
    }
}

/// Вычисляет локальный базис open-uniform сплайна для нормированной
/// координаты `u` в диапазоне данных.
///
/// При `u <= 0` или `u >= 1` возвращает линейную экстраполяцию.
pub(crate) fn open_uniform_local_basis(
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
pub(crate) fn cyclic_local_basis(phi: f64, order: SplineOrder, n_basis: usize) -> LocalBasis {
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
