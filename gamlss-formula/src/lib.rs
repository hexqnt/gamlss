#![forbid(unsafe_code)]
//! Динамический formula/builder слой, который компилируется в типизированные модели.

use std::collections::BTreeMap;

use gamlss_core::{
    DenseDesign, Gamlss, Identity, LinearPredictorBlock, ModelError, Mu, NoPenalty, ParameterBlock,
    Sigma,
};
use gamlss_family::DefaultNormal;
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

/// Точка входа для builder-style API.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModelSpec;

impl ModelSpec {
    /// Создаёт normal model spec.
    pub fn normal() -> NormalSpec {
        NormalSpec::new()
    }
}

/// Вспомогательная функция для `ModelSpec::normal()`.
pub fn normal() -> NormalSpec {
    ModelSpec::normal()
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
        CompiledNormal, DataFrame, FamilySpec, FormulaError, LinkSpec, ModelSpec, NormalSpec,
        TermSpec, normal,
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
}
