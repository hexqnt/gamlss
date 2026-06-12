/// Контракт распределения для скомпилированного GAMLSS-objective.
///
/// Пользовательские распределения реализуют этот trait. Арность параметров,
/// роли параметров и совместимость link-функций задаются через
/// [`ParameterizedFamily`], поэтому hot path остаётся типизированным без
/// dynamic lookup.
///
/// Implementations should treat `nll`/`nll_eta` as scalar negative
/// log-likelihood contributions for one observation. Invalid observation or
/// parameter domains should be represented by `f64::INFINITY` rather than
/// panicking, so optimizers can reject the candidate point. `NaN` inputs may
/// propagate as `NaN`; callers can inspect diagnostics for non-finite values.
pub trait Family {
    /// Аддитивные предикторы на link-шкале.
    type Eta;
    /// Параметры распределения на естественной шкале.
    type Theta;
    /// Градиент negative log-likelihood по `Eta`.
    type ScoreEta;

    /// Преобразует предикторы с link-шкалы в параметры распределения.
    fn theta(&self, eta: Self::Eta) -> Self::Theta;
    /// Negative log-likelihood для одного наблюдения на естественной шкале.
    fn nll(&self, y: f64, theta: Self::Theta) -> f64;
    /// Negative log-likelihood для одного наблюдения на link-шкале.
    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        self.nll(y, self.theta(eta))
    }
    /// Negative log-likelihood и score по `Eta` для одного наблюдения.
    ///
    /// `ScoreEta` is the gradient of the negative log-likelihood with respect
    /// to the link-scale predictors `Eta`, after applying the chain rule for
    /// the family links. It must have the same arity and ordering as `Eta`.
    fn nll_and_score_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::ScoreEta);
}

/// Контейнер для eta или score у family с фиксированной арностью `K`.
///
/// `part(index)` is used in the model hot path after compile-time arity
/// selection. Callers pass `index < K`; implementations may use `unreachable!`
/// for out-of-range indices instead of returning a recoverable error.
pub trait ParameterParts<const K: usize>: Sized {
    /// Собирает контейнер из `K` scalar-частей.
    fn from_array(values: [f64; K]) -> Self;
    /// Возвращает scalar-часть по индексу.
    ///
    /// Реализации могут считать, что вызывающий код передаёт индекс меньше `K`.
    fn part(&self, index: usize) -> f64;
}

impl ParameterParts<1> for f64 {
    #[inline(always)]
    fn from_array(values: [f64; 1]) -> Self {
        values[0]
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => *self,
            _ => unreachable!("one-parameter parts only have index 0"),
        }
    }
}

impl ParameterParts<2> for (f64, f64) {
    #[inline(always)]
    fn from_array(values: [f64; 2]) -> Self {
        (values[0], values[1])
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.0,
            1 => self.1,
            _ => unreachable!("two-parameter parts only have indices 0 and 1"),
        }
    }
}

impl ParameterParts<3> for (f64, f64, f64) {
    #[inline(always)]
    fn from_array(values: [f64; 3]) -> Self {
        (values[0], values[1], values[2])
    }

    #[inline(always)]
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
    #[inline(always)]
    fn from_array(values: [f64; 4]) -> Self {
        (values[0], values[1], values[2], values[3])
    }

    #[inline(always)]
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

/// Family с фиксированным числом параметров, ролями параметров и link-функциями.
///
/// `Params` и `Links` задаются tuple-ами той же длины, что и арность family.
/// Their order defines the order of predictor blocks, score parts and flat
/// coefficient ranges in compiled models.
pub trait ParameterizedFamily<const K: usize>: Family
where
    Self::Eta: ParameterParts<K>,
    Self::ScoreEta: ParameterParts<K>,
{
    /// Роли параметров family.
    type Params;
    /// Link-функции параметров family.
    type Links;
}

/// Distribution helper для CDF.
pub trait HasCdf: Family {
    /// CDF в точке `y` для параметров на естественной шкале.
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64;
}

/// Distribution helper для quantile function.
pub trait HasQuantile: Family {
    /// Квантиль уровня `p` для параметров на естественной шкале.
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64;
}

/// Distribution helper для simulation.
pub trait CanSimulate<Rng>: Family {
    /// Генерирует одно значение для параметров на естественной шкале.
    fn sample(&self, rng: &mut Rng, theta: Self::Theta) -> f64;
}
