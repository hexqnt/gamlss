#![forbid(unsafe_code)]
//! Динамический formula/builder слой, который компилируется в типизированные модели.

use std::collections::BTreeMap;

use gamlss_core::{
    DenseDesign, Gamlss, Identity, LinearPredictorBlock, Log, Logit, ModelError, Mu, NoPenalty,
    ParameterBlock, Precision, Rate, Scale, Shape, Sigma,
};
use gamlss_family::{
    DefaultBeta, DefaultGamma, DefaultInverseGaussian, DefaultLogNormal, DefaultNormal,
    DefaultWeibull,
};
use thiserror::Error;

/// Ошибки динамического formula/builder слоя.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum FormulaError {
    /// DataFrame не содержит колонок.
    #[error("data frame must contain at least one column")]
    EmptyData,

    /// Колонка с таким именем уже существует.
    #[error("duplicate column `{0}`")]
    DuplicateColumn(String),

    /// Запрошенная колонка отсутствует.
    #[error("unknown column `{0}`")]
    UnknownColumn(String),

    /// Длина колонки не совпадает с числом строк DataFrame.
    #[error("column `{name}` has {actual} rows, expected {expected}")]
    ColumnLength {
        /// Имя колонки.
        name: String,
        /// Ожидаемое число строк.
        expected: usize,
        /// Фактическое число строк.
        actual: usize,
    },

    /// Ошибка скомпилированной модели из `gamlss-core`.
    #[error(transparent)]
    Model(#[from] ModelError),
}

/// Минимальный column-oriented DataFrame для formula/builder слоя.
#[derive(Debug, Clone, PartialEq)]
pub struct DataFrame {
    nrows: usize,
    columns: BTreeMap<String, Vec<f64>>,
}

impl DataFrame {
    /// Создаёт пустой DataFrame с фиксированным числом строк.
    pub fn new(nrows: usize) -> Self {
        Self {
            nrows,
            columns: BTreeMap::new(),
        }
    }

    /// Создаёт DataFrame из набора колонок.
    ///
    /// Все колонки должны иметь одинаковую длину, имена не должны повторяться.
    pub fn from_columns<I, S>(columns: I) -> Result<Self, FormulaError>
    where
        I: IntoIterator<Item = (S, Vec<f64>)>,
        S: Into<String>,
    {
        let mut iter = columns.into_iter();
        let Some((first_name, first_values)) = iter.next() else {
            return Err(FormulaError::EmptyData);
        };

        let mut frame = Self::new(first_values.len());
        frame.insert_column(first_name, first_values)?;
        for (name, values) in iter {
            frame.insert_column(name, values)?;
        }
        Ok(frame)
    }

    /// Добавляет колонку, проверяя длину и уникальность имени.
    pub fn insert_column<S>(&mut self, name: S, values: Vec<f64>) -> Result<(), FormulaError>
    where
        S: Into<String>,
    {
        let name = name.into();
        if values.len() != self.nrows {
            return Err(FormulaError::ColumnLength {
                name,
                expected: self.nrows,
                actual: values.len(),
            });
        }

        if self.columns.contains_key(&name) {
            return Err(FormulaError::DuplicateColumn(name));
        }

        self.columns.insert(name, values);
        Ok(())
    }

    /// Число строк.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Возвращает колонку по имени.
    pub fn column(&self, name: &str) -> Result<&[f64], FormulaError> {
        self.columns
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| FormulaError::UnknownColumn(name.to_owned()))
    }
}

/// Динамическая спецификация family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FamilySpec {
    /// Нормальное распределение.
    Normal,
    /// Gamma distribution.
    Gamma,
    /// Log-normal distribution.
    LogNormal,
    /// Weibull distribution.
    Weibull,
    /// Inverse Gaussian distribution.
    InverseGaussian,
    /// Beta distribution.
    Beta,
}

/// Динамическая спецификация link-функции.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkSpec {
    /// Identity link-функция.
    Identity,
    /// Log link-функция.
    Log,
    /// Positive link-функция Softplus.
    Softplus,
    /// Обратная logit link-функция.
    Logit,
}

/// Динамическая спецификация term для одного predictor-а.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TermSpec {
    /// Intercept-столбец из единиц.
    Intercept,
    /// Линейный term по имени колонки.
    Linear(String),
}

impl TermSpec {
    /// Создаёт intercept term.
    pub fn intercept() -> Self {
        Self::Intercept
    }

    /// Создаёт linear term по имени колонки.
    pub fn linear<S>(name: S) -> Self
    where
        S: Into<String>,
    {
        Self::Linear(name.into())
    }
}

/// Результат компиляции `NormalSpec` в типизированную модель.
pub type CompiledNormal = Gamlss<
    DefaultNormal,
    (
        ParameterBlock<Mu, Identity, LinearPredictorBlock<DenseDesign>, NoPenalty>,
        ParameterBlock<Sigma, gamlss_core::Log, LinearPredictorBlock<DenseDesign>, NoPenalty>,
    ),
>;

/// Результат компиляции `GammaSpec` в типизированную модель.
pub type CompiledGamma = Gamlss<
    DefaultGamma,
    (
        ParameterBlock<Shape, Log, LinearPredictorBlock<DenseDesign>, NoPenalty>,
        ParameterBlock<Rate, Log, LinearPredictorBlock<DenseDesign>, NoPenalty>,
    ),
>;

/// Результат компиляции `LogNormalSpec` в типизированную модель.
pub type CompiledLogNormal = Gamlss<
    DefaultLogNormal,
    (
        ParameterBlock<Mu, Identity, LinearPredictorBlock<DenseDesign>, NoPenalty>,
        ParameterBlock<Sigma, Log, LinearPredictorBlock<DenseDesign>, NoPenalty>,
    ),
>;

/// Результат компиляции `WeibullSpec` в типизированную модель.
pub type CompiledWeibull = Gamlss<
    DefaultWeibull,
    (
        ParameterBlock<Shape, Log, LinearPredictorBlock<DenseDesign>, NoPenalty>,
        ParameterBlock<Scale, Log, LinearPredictorBlock<DenseDesign>, NoPenalty>,
    ),
>;

/// Результат компиляции `InverseGaussianSpec` в типизированную модель.
pub type CompiledInverseGaussian = Gamlss<
    DefaultInverseGaussian,
    (
        ParameterBlock<Mu, Log, LinearPredictorBlock<DenseDesign>, NoPenalty>,
        ParameterBlock<Shape, Log, LinearPredictorBlock<DenseDesign>, NoPenalty>,
    ),
>;

/// Результат компиляции `BetaSpec` в типизированную модель.
pub type CompiledBeta = Gamlss<
    DefaultBeta,
    (
        ParameterBlock<Mu, Logit, LinearPredictorBlock<DenseDesign>, NoPenalty>,
        ParameterBlock<Precision, Log, LinearPredictorBlock<DenseDesign>, NoPenalty>,
    ),
>;

/// Динамическая спецификация normal GAMLSS-модели.
///
/// Если terms для параметра не заданы, при компиляции используется intercept.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalSpec {
    mu_terms: Vec<TermSpec>,
    sigma_terms: Vec<TermSpec>,
}

impl NormalSpec {
    /// Создаёт пустую спецификацию.
    pub fn new() -> Self {
        Self {
            mu_terms: Vec::new(),
            sigma_terms: Vec::new(),
        }
    }

    /// Добавляет term в predictor параметра `mu`.
    pub fn mu(mut self, term: TermSpec) -> Self {
        self.mu_terms.push(term);
        self
    }

    /// Добавляет term в predictor параметра `sigma`.
    pub fn sigma(mut self, term: TermSpec) -> Self {
        self.sigma_terms.push(term);
        self
    }

    /// Добавляет intercept в predictor `mu`.
    pub fn mu_intercept(self) -> Self {
        self.mu(TermSpec::Intercept)
    }

    /// Добавляет intercept в predictor `sigma`.
    pub fn sigma_intercept(self) -> Self {
        self.sigma(TermSpec::Intercept)
    }

    /// Добавляет linear term в predictor `mu`.
    pub fn mu_linear<S>(self, name: S) -> Self
    where
        S: Into<String>,
    {
        self.mu(TermSpec::linear(name))
    }

    /// Добавляет linear term в predictor `sigma`.
    pub fn sigma_linear<S>(self, name: S) -> Self
    where
        S: Into<String>,
    {
        self.sigma(TermSpec::linear(name))
    }

    /// Компилирует динамическую спецификацию в типизированную normal GAMLSS-модель.
    pub fn compile(&self, data: &DataFrame, y: &str) -> Result<CompiledNormal, FormulaError> {
        let response = data.column(y)?.to_vec();
        let mu_x = design_from_terms(&self.mu_terms, data)?;
        let sigma_x = design_from_terms(&self.sigma_terms, data)?;

        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(mu_x, NoPenalty, 0);
        let sigma =
            ParameterBlock::<Sigma, gamlss_core::Log, _, _>::linear(sigma_x, NoPenalty, mu.len());

        Ok(gamlss_core::Gamlss::try_new(
            DefaultNormal::new(),
            (mu, sigma),
            response,
        )?)
    }
}

impl Default for NormalSpec {
    fn default() -> Self {
        Self::new()
    }
}

macro_rules! define_two_parameter_spec {
    (
        $(#[$meta:meta])*
        $spec:ident, $compiled:ident, $family:ty;
        first = $first_terms:ident, $first:ident, $first_intercept:ident, $first_linear:ident, $first_param:ty, $first_link:ty;
        second = $second_terms:ident, $second:ident, $second_intercept:ident, $second_linear:ident, $second_param:ty, $second_link:ty
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct $spec {
            $first_terms: Vec<TermSpec>,
            $second_terms: Vec<TermSpec>,
        }

        impl $spec {
            /// Создаёт пустую спецификацию.
            pub fn new() -> Self {
                Self {
                    $first_terms: Vec::new(),
                    $second_terms: Vec::new(),
                }
            }

            /// Добавляет term в predictor первого параметра.
            pub fn $first(mut self, term: TermSpec) -> Self {
                self.$first_terms.push(term);
                self
            }

            /// Добавляет term в predictor второго параметра.
            pub fn $second(mut self, term: TermSpec) -> Self {
                self.$second_terms.push(term);
                self
            }

            /// Добавляет intercept в predictor первого параметра.
            pub fn $first_intercept(self) -> Self {
                self.$first(TermSpec::Intercept)
            }

            /// Добавляет intercept в predictor второго параметра.
            pub fn $second_intercept(self) -> Self {
                self.$second(TermSpec::Intercept)
            }

            /// Добавляет linear term в predictor первого параметра.
            pub fn $first_linear<S>(self, name: S) -> Self
            where
                S: Into<String>,
            {
                self.$first(TermSpec::linear(name))
            }

            /// Добавляет linear term в predictor второго параметра.
            pub fn $second_linear<S>(self, name: S) -> Self
            where
                S: Into<String>,
            {
                self.$second(TermSpec::linear(name))
            }

            /// Компилирует динамическую спецификацию в типизированную модель.
            pub fn compile(&self, data: &DataFrame, y: &str) -> Result<$compiled, FormulaError> {
                let response = data.column(y)?.to_vec();
                let first_x = design_from_terms(&self.$first_terms, data)?;
                let second_x = design_from_terms(&self.$second_terms, data)?;

                let first =
                    ParameterBlock::<$first_param, $first_link, _, _>::linear(first_x, NoPenalty, 0);
                let second = ParameterBlock::<$second_param, $second_link, _, _>::linear(
                    second_x,
                    NoPenalty,
                    first.len(),
                );

                Ok(gamlss_core::Gamlss::try_new(
                    <$family>::new(),
                    (first, second),
                    response,
                )?)
            }
        }

        impl Default for $spec {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

define_two_parameter_spec!(
    /// Динамическая спецификация gamma GAMLSS-модели с default links.
    GammaSpec, CompiledGamma, DefaultGamma;
    first = shape_terms, shape, shape_intercept, shape_linear, Shape, Log;
    second = rate_terms, rate, rate_intercept, rate_linear, Rate, Log
);

define_two_parameter_spec!(
    /// Динамическая спецификация log-normal GAMLSS-модели с default links.
    LogNormalSpec, CompiledLogNormal, DefaultLogNormal;
    first = mu_terms, mu, mu_intercept, mu_linear, Mu, Identity;
    second = sigma_terms, sigma, sigma_intercept, sigma_linear, Sigma, Log
);

define_two_parameter_spec!(
    /// Динамическая спецификация Weibull GAMLSS-модели с default links.
    WeibullSpec, CompiledWeibull, DefaultWeibull;
    first = shape_terms, shape, shape_intercept, shape_linear, Shape, Log;
    second = scale_terms, scale, scale_intercept, scale_linear, Scale, Log
);

define_two_parameter_spec!(
    /// Динамическая спецификация inverse Gaussian GAMLSS-модели с default links.
    InverseGaussianSpec, CompiledInverseGaussian, DefaultInverseGaussian;
    first = mu_terms, mu, mu_intercept, mu_linear, Mu, Log;
    second = shape_terms, shape, shape_intercept, shape_linear, Shape, Log
);

define_two_parameter_spec!(
    /// Динамическая спецификация beta GAMLSS-модели с default links.
    BetaSpec, CompiledBeta, DefaultBeta;
    first = mu_terms, mu, mu_intercept, mu_linear, Mu, Logit;
    second = precision_terms, precision, precision_intercept, precision_linear, Precision, Log
);

/// Точка входа для builder-style API.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModelSpec;

impl ModelSpec {
    /// Создаёт normal model spec.
    pub fn normal() -> NormalSpec {
        NormalSpec::new()
    }

    /// Создаёт gamma model spec.
    pub fn gamma() -> GammaSpec {
        GammaSpec::new()
    }

    /// Создаёт log-normal model spec.
    pub fn log_normal() -> LogNormalSpec {
        LogNormalSpec::new()
    }

    /// Создаёт Weibull model spec.
    pub fn weibull() -> WeibullSpec {
        WeibullSpec::new()
    }

    /// Создаёт inverse Gaussian model spec.
    pub fn inverse_gaussian() -> InverseGaussianSpec {
        InverseGaussianSpec::new()
    }

    /// Создаёт beta model spec.
    pub fn beta() -> BetaSpec {
        BetaSpec::new()
    }
}

/// Вспомогательная функция для `ModelSpec::normal()`.
pub fn normal() -> NormalSpec {
    ModelSpec::normal()
}

/// Вспомогательная функция для `ModelSpec::gamma()`.
pub fn gamma() -> GammaSpec {
    ModelSpec::gamma()
}

/// Вспомогательная функция для `ModelSpec::log_normal()`.
pub fn log_normal() -> LogNormalSpec {
    ModelSpec::log_normal()
}

/// Вспомогательная функция для `ModelSpec::weibull()`.
pub fn weibull() -> WeibullSpec {
    ModelSpec::weibull()
}

/// Вспомогательная функция для `ModelSpec::inverse_gaussian()`.
pub fn inverse_gaussian() -> InverseGaussianSpec {
    ModelSpec::inverse_gaussian()
}

/// Вспомогательная функция для `ModelSpec::beta()`.
pub fn beta() -> BetaSpec {
    ModelSpec::beta()
}

fn design_from_terms(terms: &[TermSpec], data: &DataFrame) -> Result<DenseDesign, FormulaError> {
    let default_terms;
    let terms = if terms.is_empty() {
        default_terms = [TermSpec::Intercept];
        &default_terms[..]
    } else {
        terms
    };

    let nrows = data.nrows();
    let ncols = terms.len();
    let mut values = Vec::with_capacity(nrows * ncols);

    for row in 0..nrows {
        for term in terms {
            match term {
                TermSpec::Intercept => values.push(1.0),
                TermSpec::Linear(name) => values.push(data.column(name)?[row]),
            }
        }
    }

    Ok(DenseDesign::from_row_major(nrows, ncols, values)?)
}

/// Наиболее часто используемые импорты из `gamlss-formula`.
pub mod prelude {
    pub use crate::{
        BetaSpec, CompiledBeta, CompiledGamma, CompiledInverseGaussian, CompiledLogNormal,
        CompiledNormal, CompiledWeibull, DataFrame, FamilySpec, FormulaError, GammaSpec,
        InverseGaussianSpec, LinkSpec, LogNormalSpec, ModelSpec, NormalSpec, TermSpec, WeibullSpec,
        beta, gamma, inverse_gaussian, log_normal, normal, weibull,
    };
}

#[cfg(test)]
mod tests {
    use gamlss_core::Objective;

    use super::{DataFrame, ModelSpec};

    #[test]
    fn compiles_normal_spec_to_typed_model() {
        let data =
            DataFrame::from_columns([("y", vec![0.0, 1.0, 2.0]), ("x", vec![1.0, 2.0, 3.0])])
                .unwrap();
        let spec = ModelSpec::normal()
            .mu_intercept()
            .mu_linear("x")
            .sigma_intercept();

        let mut model = spec.compile(&data, "y").unwrap();
        let beta = vec![0.0, 0.5, -0.2];

        assert_eq!(model.nparams(), 3);
        assert!(model.value(&beta).unwrap().is_finite());
    }

    #[test]
    fn compiles_new_default_family_specs_to_typed_models() {
        let data = DataFrame::from_columns([
            ("y_pos", vec![0.5, 1.0, 1.5]),
            ("y_unit", vec![0.2, 0.5, 0.8]),
            ("x", vec![1.0, 2.0, 3.0]),
        ])
        .unwrap();

        let mut gamma = ModelSpec::gamma()
            .shape_intercept()
            .shape_linear("x")
            .rate_intercept()
            .compile(&data, "y_pos")
            .unwrap();
        assert_eq!(gamma.nparams(), 3);
        assert_eq!(gamma.parameter_layout().slice("shape").unwrap(), 0..2);
        assert_eq!(gamma.parameter_layout().slice("rate").unwrap(), 2..3);
        assert!(gamma.value(&[0.0, 0.1, 0.0]).unwrap().is_finite());

        let mut log_normal = ModelSpec::log_normal()
            .mu_intercept()
            .sigma_intercept()
            .sigma_linear("x")
            .compile(&data, "y_pos")
            .unwrap();
        assert_eq!(log_normal.nparams(), 3);
        assert_eq!(log_normal.parameter_layout().slice("mu").unwrap(), 0..1);
        assert_eq!(log_normal.parameter_layout().slice("sigma").unwrap(), 1..3);
        assert!(log_normal.value(&[0.0, 0.0, 0.1]).unwrap().is_finite());

        let mut weibull = ModelSpec::weibull()
            .shape_intercept()
            .scale_intercept()
            .scale_linear("x")
            .compile(&data, "y_pos")
            .unwrap();
        assert_eq!(weibull.nparams(), 3);
        assert_eq!(weibull.parameter_layout().slice("shape").unwrap(), 0..1);
        assert_eq!(weibull.parameter_layout().slice("scale").unwrap(), 1..3);
        assert!(weibull.value(&[0.0, 0.0, 0.1]).unwrap().is_finite());

        let mut inverse_gaussian = ModelSpec::inverse_gaussian()
            .mu_intercept()
            .mu_linear("x")
            .shape_intercept()
            .compile(&data, "y_pos")
            .unwrap();
        assert_eq!(inverse_gaussian.nparams(), 3);
        assert_eq!(
            inverse_gaussian.parameter_layout().slice("mu").unwrap(),
            0..2
        );
        assert_eq!(
            inverse_gaussian.parameter_layout().slice("shape").unwrap(),
            2..3
        );
        assert!(
            inverse_gaussian
                .value(&[0.0, 0.1, 0.0])
                .unwrap()
                .is_finite()
        );

        let mut beta = ModelSpec::beta()
            .mu_intercept()
            .precision_intercept()
            .precision_linear("x")
            .compile(&data, "y_unit")
            .unwrap();
        assert_eq!(beta.nparams(), 3);
        assert_eq!(beta.parameter_layout().slice("mu").unwrap(), 0..1);
        assert_eq!(beta.parameter_layout().slice("precision").unwrap(), 1..3);
        assert!(beta.value(&[0.0, 1.0, 0.1]).unwrap().is_finite());
    }
}
