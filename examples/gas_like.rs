//! Пример «gas-like» модели: подгонка нормального GAMLSS (`y_i ~ N(μ_i, σ_i)`)
//! через L-BFGS из библиотеки `argmin`.
//!
//! Демонстрирует главные концепции крейта:
//! - Композиция предикторов через `SumBlock` (intercept + сплайн + линейные эффекты).
//! - Link-функции на уровне типов: `Identity` для μ (без ограничений),
//!   `ClampedLog<-12,12>` для σ (строгая положительность + численная защита).
//! - Штрафы как часть `ParameterBlock`: `CyclicDifferencePenalty` на вторые разности
//!   сглаживает сезонный профиль волатильности.
//! - Адаптер `ArgminObjective<O>`: `RefCell` разрешает несовместимость `&self`/`&mut self`
//!   между argmin и gamlss-core без копирования буферов на каждом вызове.

#![allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]

use std::cell::RefCell;

use argmin::{
    core::{CostFunction, Error, Executor, Gradient, State},
    solver::{linesearch::MoreThuenteLineSearch, quasinewton::LBFGS},
};
use gamlss::core::Objective;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use gamlss::core::{
        ClampedLog, DenseDesign, Gamlss, Identity, LinearPredictorBlock, Mu, NoPenalty,
        ParameterBlock, Sigma, SumBlock,
    };
    use gamlss::family::Normal;
    use gamlss::spline::{
        CyclicDifferencePenalty, CyclicSplineDesign, OpenUniformSplineDesign, SplineOrder,
    };

    // Синтетические данные: истинная зависимость — квадратичная по температуре
    // (U-образный профиль потребления), минус weekend-эффект, плюс тренд.
    let data = generate_surrogate_data(90);
    let n = data.y.len();

    // ── μ-предиктор: intercept + кубический B-сплайн температуры (8 функций) +
    //                  weekend (бинарный) + линейный тренд.
    // Узлы сплайна автоматически по квантилям — нет пустых интервалов.
    let mu_predictor = SumBlock::new((
        LinearPredictorBlock::new(DenseDesign::intercept(n)),
        OpenUniformSplineDesign::from_data(&data.temperature, 8, SplineOrder::Cubic)?,
        LinearPredictorBlock::new(DenseDesign::column(&data.weekend)),
        LinearPredictorBlock::new(DenseDesign::column(&data.time_x)),
    ));

    // ── σ-предиктор: циклический кубический сплайн по фазе года phi ∈ [0,1].
    // Циклическое условие: стык декабрь-январь гладкий, без разрыва.
    let sigma_predictor = CyclicSplineDesign::new(&data.phi, 8, SplineOrder::Cubic)?;

    // ParameterBlock связывает предиктор, link и penalty. Смещение (offset)
    // задаёт позицию блока в плоском векторе параметров θ — compile-time конкатенация.
    let mu = ParameterBlock::<Mu, Identity, _, _>::new(mu_predictor, NoPenalty, 0);
    let sigma = ParameterBlock::<Sigma, ClampedLog<-12, 12>, _, _>::new(
        sigma_predictor,
        CyclicDifferencePenalty::new(0.05, 2), // λ=0.05, d=2 — сглаживание вторых разностей
        mu.len(),                              // σ-параметры идут после μ-параметров в θ
    );

    // try_new проверяет согласованность размерностей (строки дизайн-матриц = длина y,
    // offsets не перекрываются). Тип модели выводится компилятором целиком.
    let model = Gamlss::try_new(
        Normal::<Identity, ClampedLog<-12, 12>>::new(),
        (mu, sigma),
        data.y,
    )?;

    let initial_theta = model.initial_theta()?;

    // into_cached_objective кэширует значения предикторов, пересчитывая их инкрементально.
    // Для сплайновых моделей это даёт ускорение до ~10x.
    let problem = ArgminObjective::new(model.into_cached_objective());

    // L-BFGS: квази-ньютоновский метод, m=7 последних пар (s_k, y_k).
    // More-Thuente line search гарантирует условия Вольфе на каждом шаге.
    let solver = LBFGS::new(MoreThuenteLineSearch::new().with_c(1.0e-4, 0.9)?, 7)
        .with_tolerance_grad(1.0e-5)?
        .with_tolerance_cost(1.0e-10)?;

    let result = Executor::new(problem, solver)
        .configure(|state| state.param(initial_theta).max_iters(100))
        .run()?;

    // Извлекаем финальные параметры и считаем норму градиента для проверки optimality.
    let theta = result
        .state
        .get_best_param()
        .or_else(|| result.state.get_param())
        .expect("argmin returns a final parameter vector");
    let mut objective = result
        .problem
        .problem
        .expect("argmin returns the original problem");
    let mut grad = vec![0.0; objective.dim()];
    let loss = objective.value_gradient(theta, &mut grad)?;

    println!(
        "gas_like argmin l-bfgs: loss={loss:.6}, iterations={}, grad_norm={:.6}",
        result.state.get_iter(),
        grad.iter().map(|v| v * v).sum::<f64>().sqrt()
    );
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// АДАПТЕР ДЛЯ ARGMIN: RefCell разрешает конфликт &self (argmin) vs &mut self (Objective)
// ═══════════════════════════════════════════════════════════════════════════
//
// Objective требует &mut self для переиспользования внутренних буферов (векторы
// предикторов, градиентов). Выделять их заново на каждом вызове — неприемлемо.
// argmin требует &self. RefCell переносит проверку заимствования в runtime:
// поскольку argmin вызывает cost/gradient строго последовательно, паники не будет.

#[derive(Debug)]
struct ArgminObjective<O> {
    objective: RefCell<O>,
}

impl<O> ArgminObjective<O> {
    const fn new(objective: O) -> Self {
        Self {
            objective: RefCell::new(objective),
        }
    }

    fn dim(&self) -> usize
    where
        O: Objective,
    {
        self.objective.borrow().dim()
    }

    fn value_gradient(&mut self, theta: &[f64], grad: &mut [f64]) -> Result<f64, O::Error>
    where
        O: Objective,
    {
        self.objective.get_mut().value_gradient(theta, grad)
    }
}

impl<O> CostFunction for ArgminObjective<O>
where
    O: Objective,
    O::Error: std::error::Error + Send + Sync + 'static,
{
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        self.objective.borrow_mut().value(param).map_err(Error::new)
    }
}

impl<O> Gradient for ArgminObjective<O>
where
    O: Objective,
    O::Error: std::error::Error + Send + Sync + 'static,
{
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
        let mut grad = vec![0.0; self.dim()];
        self.objective
            .borrow_mut()
            .gradient(param, &mut grad)
            .map_err(Error::new)?;
        Ok(grad)
    }
}

// Синтетические данные: y = 3 - 0.08·T + 0.004·T² - 0.35·wknd + 0.2·trnd + noise.
struct SurrogateData {
    y: Vec<f64>,
    temperature: Vec<f64>,
    phi: Vec<f64>,
    weekend: Vec<f64>,
    time_x: Vec<f64>,
}

/// Генерирует surrogate-данные, имитирующие сезонную + weekend + трендовую
/// структуру потребления газа.
fn generate_surrogate_data(n: usize) -> SurrogateData {
    let mut y = Vec::with_capacity(n);
    let mut temperature = Vec::with_capacity(n);
    let mut phi = Vec::with_capacity(n);
    let mut weekend = Vec::with_capacity(n);
    let mut time_x = Vec::with_capacity(n);

    for day in 0..n {
        let day_f = day as f64;
        let seasonal = (std::f64::consts::TAU * day_f / n as f64).sin();
        let temp = 5.0 + 12.0 * seasonal;
        let weekend_value = f64::from(day % 7 >= 5);
        let trend = (day_f - (n - 1) as f64 / 2.0) / n as f64;
        let noise = 0.25 * (1.7 * day_f).sin();
        let mean = 3.0 - 0.08 * temp + 0.004 * temp * temp - 0.35 * weekend_value + 0.2 * trend;

        y.push(mean + noise);
        temperature.push(temp);
        phi.push(day_f / n as f64);
        weekend.push(weekend_value);
        time_x.push(trend);
    }

    SurrogateData {
        y,
        temperature,
        phi,
        weekend,
        time_x,
    }
}
