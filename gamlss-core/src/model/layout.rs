use std::ops::Range;

use crate::ParameterName;

/// Именованный блок коэффициентов внутри плоского вектора параметров.
///
/// Связывает стабильное имя параметра распределения (например, `"mu"`)
/// с диапазоном позиций в общем beta-векторе.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterSlice {
    /// Stable distribution parameter name, e.g. `"mu"` or `"sigma"`.
    pub name: &'static str,
    /// Coefficient range for this parameter inside the full beta vector.
    pub range: Range<usize>,
}

/// Отображение параметров распределения на диапазоны в плоском beta-векторе.
///
/// Используется для introspection модели: распаковки коэффициентов,
/// построения diagnostics и передачи информации внешним оптимизаторам.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterLayout {
    slices: Vec<ParameterSlice>,
}

impl ParameterLayout {
    /// Creates a layout from named slices.
    pub fn new(slices: Vec<ParameterSlice>) -> Self {
        Self { slices }
    }

    /// Returns all parameter slices in model order.
    pub fn slices(&self) -> &[ParameterSlice] {
        &self.slices
    }

    /// Returns the coefficient range for `name`, if present.
    pub fn slice(&self, name: &str) -> Option<Range<usize>> {
        self.slices
            .iter()
            .find(|slice| slice.name == name)
            .map(|slice| slice.range.clone())
    }

    /// Returns the coefficient range for typed parameter marker `P`, if present.
    pub fn slice_of<P>(&self) -> Option<Range<usize>>
    where
        P: ParameterName,
    {
        self.slice(P::NAME)
    }
}

/// Коэффициенты одного распакованного параметрического блока.
///
/// Возвращается методом [`crate::Gamlss::unpack_theta`] для human-readable
/// представления плоского beta-вектора.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterCoefficients {
    /// Stable distribution parameter name.
    pub name: &'static str,
    /// Coefficients for this parameter block.
    pub coefficients: Vec<f64>,
}

/// Человекочитаемое представление плоского beta-вектора.
///
/// Содержит по одному [`ParameterCoefficients`] для каждого параметра
/// распределения в порядке модели.
#[derive(Debug, Clone, PartialEq)]
pub struct UnpackedTheta {
    /// Parameter blocks in model order.
    pub blocks: Vec<ParameterCoefficients>,
}

impl UnpackedTheta {
    /// Returns an unpacked coefficient block by parameter name.
    pub fn block(&self, name: &str) -> Option<&ParameterCoefficients> {
        self.blocks.iter().find(|block| block.name == name)
    }

    /// Returns an unpacked coefficient block for typed parameter marker `P`.
    pub fn block_of<P>(&self) -> Option<&ParameterCoefficients>
    where
        P: ParameterName,
    {
        self.block(P::NAME)
    }

    /// Returns coefficients by parameter name.
    pub fn coefficients(&self, name: &str) -> Option<&[f64]> {
        self.block(name).map(|block| block.coefficients.as_slice())
    }

    /// Returns coefficients for typed parameter marker `P`.
    pub fn coefficients_of<P>(&self) -> Option<&[f64]>
    where
        P: ParameterName,
    {
        self.coefficients(P::NAME)
    }
}

/// Training diagnostics for a candidate theta vector.
///
/// Содержит значения objective, weighted negative log-likelihood (без штрафов),
/// суммарный штраф, норму градиента и число не-finite компонент градиента.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainingDiagnostics {
    /// Full objective value: weighted training negative log-likelihood plus penalties.
    pub objective: f64,
    /// Weighted training negative log-likelihood before penalties.
    pub train_nll: f64,
    /// Total penalty contribution.
    pub penalty: f64,
    /// Euclidean norm of the objective gradient.
    pub gradient_norm: f64,
    /// Number of non-finite gradient entries.
    pub nonfinite_gradient_count: usize,
}
