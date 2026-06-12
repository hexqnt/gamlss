use std::ops::Range;

use crate::{
    GlobalPenalty, ModelError, Objective, ParameterBlock, ParameterName, ParameterParts,
    ParameterizedFamily, Penalty, PredictorBlock,
};

/// Tuple-контракт для набора parameter blocks, совместимого с family `F`.
pub trait GamlssBlocks<F> {
    /// Число наблюдений в blocks.
    fn nrows(&self) -> usize;
    /// Длина общего beta-вектора, покрывающего все blocks.
    fn len(&self) -> usize;

    /// `true`, если blocks не требуют коэффициентов.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Проверяет, что blocks совместимы с response длины `y_len`.
    fn validate(&self, y_len: usize) -> Result<(), ModelError>;
    /// Negative log-likelihood without penalties.
    fn train_nll(&self, family: &F, y: &[f64], beta: &[f64]) -> f64;
    /// Penalty value depending on coefficient blocks.
    fn penalty_value(&self, beta: &[f64]) -> f64;
    /// Значение negative log-likelihood плюс penalties.
    fn value(&self, family: &F, y: &[f64], beta: &[f64]) -> f64 {
        self.train_nll(family, y, beta) + self.penalty_value(beta)
    }
    /// Добавляет градиент по всем blocks в `grad`.
    fn gradient_into(&self, family: &F, y: &[f64], beta: &[f64], grad: &mut [f64]);
    /// Creates reusable buffers for repeated gradient evaluations.
    fn gradient_workspace(&self, y_len: usize) -> GradientWorkspace {
        let mut workspace = GradientWorkspace::new();
        let ranges = self.block_ranges();
        workspace.prepare(ranges.len());
        for (index, range) in ranges.iter().enumerate() {
            workspace.prepare_score(index, y_len);
            let _ = workspace.local_gradient_mut(index, range.len());
        }
        workspace
    }
    /// Добавляет градиент, переиспользуя временные буферы из `workspace`.
    fn gradient_into_workspace(
        &self,
        family: &F,
        y: &[f64],
        beta: &[f64],
        grad: &mut [f64],
        _: &mut GradientWorkspace,
    ) {
        self.gradient_into(family, y, beta, grad);
    }
    /// Диапазоны коэффициентов каждого block в общем beta-векторе.
    fn block_ranges(&self) -> Vec<Range<usize>>;
    /// Возвращает размещение coefficient blocks внутри плоского beta-вектора.
    fn parameter_layout(&self) -> ParameterLayout;
}

/// Reusable scratch buffers for GAMLSS gradient evaluation.
///
/// The workspace stores one per-observation score vector and one local
/// coefficient-gradient vector per parameter block. Reusing it avoids the
/// temporary `Vec` allocations that otherwise happen inside each gradient call.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GradientWorkspace {
    scores: Vec<Vec<f64>>,
    local_gradients: Vec<Vec<f64>>,
}

impl GradientWorkspace {
    /// Creates an empty workspace. Buffers are allocated lazily on first use.
    pub fn new() -> Self {
        Self::default()
    }

    fn prepare(&mut self, block_count: usize) {
        self.scores.resize_with(block_count, Vec::new);
        self.local_gradients.resize_with(block_count, Vec::new);
    }

    fn prepare_score(&mut self, index: usize, len: usize) {
        let score = &mut self.scores[index];
        score.resize(len, 0.0);
    }

    fn set_score(&mut self, index: usize, row: usize, value: f64) {
        debug_assert!(index < self.scores.len());
        debug_assert!(row < self.scores[index].len());
        self.scores[index][row] = value;
    }

    fn local_gradient_mut(&mut self, index: usize, len: usize) -> &mut [f64] {
        let gradient = &mut self.local_gradients[index];
        gradient.resize(len, 0.0);
        gradient.fill(0.0);
        gradient
    }

    fn score_and_local_gradient_mut(
        &mut self,
        index: usize,
        local_gradient_len: usize,
    ) -> (&[f64], &mut [f64]) {
        let local_gradient = &mut self.local_gradients[index];
        local_gradient.resize(local_gradient_len, 0.0);
        local_gradient.fill(0.0);

        (self.scores[index].as_slice(), local_gradient.as_mut_slice())
    }
}

/// Скомпилированная типизированная GAMLSS-модель.
///
/// `F` задаёт распределение response, а `Blocks` задаёт по одному predictor
/// block для каждого параметра family.
#[derive(Debug, Clone, PartialEq)]
pub struct Gamlss<F, Blocks> {
    /// Family распределения response.
    pub family: F,
    /// Типизированные parameter blocks.
    pub blocks: Blocks,
    /// Response vector.
    pub y: Vec<f64>,
}

/// GAMLSS objective with reusable gradient buffers.
///
/// This wrapper is intended for optimizers that call `gradient` repeatedly.
/// It owns the compiled model and keeps a [`GradientWorkspace`] between calls,
/// avoiding per-call allocation of score and local-gradient vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct CachedGamlss<F, Blocks> {
    /// Wrapped compiled model.
    pub model: Gamlss<F, Blocks>,
    /// Reusable gradient workspace.
    pub workspace: GradientWorkspace,
}

/// Обёртка objective, добавляющая штрафы, зависящие от полного beta-вектора.
///
/// В отличие от [`Penalty`], который действует локально на один блок,
/// [`GlobalPenalty`] позволяет coupling нескольких блоков (например,
/// центрирующие или LASSO-подобные штрафы).
#[derive(Debug, Clone, PartialEq)]
pub struct WithGlobalPenalties<O, GP> {
    /// Wrapped objective.
    pub objective: O,
    /// Global penalties evaluated on the full parameter vector.
    pub penalties: GP,
}

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
}

/// Коэффициенты одного распакованного параметрического блока.
///
/// Возвращается методом [`Gamlss::unpack_theta`] для human-readable
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

    /// Returns coefficients by parameter name.
    pub fn coefficients(&self, name: &str) -> Option<&[f64]> {
        self.block(name).map(|block| block.coefficients.as_slice())
    }
}

/// Диагностики обучения для кандидата theta.
///
/// Содержит значения objective, negative log-likelihood (без штрафов),
/// суммарный штраф, норму градиента и число не-finite компонент градиента.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Diagnostics {
    /// Full objective value: training negative log-likelihood plus penalties.
    pub objective: f64,
    /// Training negative log-likelihood before penalties.
    pub train_nll: f64,
    /// Total penalty contribution.
    pub penalty: f64,
    /// Euclidean norm of the objective gradient.
    pub gradient_norm: f64,
    /// Number of non-finite gradient entries.
    pub nonfinite_gradient_count: usize,
}

impl<F, Blocks> Gamlss<F, Blocks> {
    /// Wraps the model with penalties evaluated on the full beta vector.
    pub fn with_global_penalties<GP>(self, penalties: GP) -> WithGlobalPenalties<Self, GP> {
        WithGlobalPenalties {
            objective: self,
            penalties,
        }
    }
}

impl<F, Blocks> Gamlss<F, Blocks>
where
    Blocks: GamlssBlocks<F>,
{
    /// Создаёт модель после проверки response и blocks.
    pub fn try_new(family: F, blocks: Blocks, y: Vec<f64>) -> Result<Self, ModelError> {
        if y.is_empty() {
            return Err(ModelError::EmptyResponse);
        }

        blocks.validate(y.len())?;
        Ok(Self { family, blocks, y })
    }

    /// Число наблюдений.
    pub fn nobs(&self) -> usize {
        self.y.len()
    }

    /// Число коэффициентов в общем beta-векторе.
    pub fn nparams(&self) -> usize {
        self.blocks.len()
    }

    /// Нулевой initial beta-вектор нужной длины.
    pub fn initial_zeros(&self) -> Vec<f64> {
        vec![0.0; self.nparams()]
    }

    /// Initial theta vector for external optimizers.
    pub fn initial_theta(&self) -> Result<Vec<f64>, ModelError> {
        Ok(self.initial_zeros())
    }

    /// Creates reusable gradient buffers sized for this model.
    pub fn gradient_workspace(&self) -> GradientWorkspace {
        self.blocks.gradient_workspace(self.y.len())
    }

    /// Wraps the model as an objective with reusable gradient buffers.
    pub fn into_cached_objective(self) -> CachedGamlss<F, Blocks> {
        let workspace = self.gradient_workspace();
        CachedGamlss {
            model: self,
            workspace,
        }
    }

    /// Диапазоны coefficient blocks внутри beta.
    pub fn block_ranges(&self) -> Vec<Range<usize>> {
        self.blocks.block_ranges()
    }

    /// Layout of named parameter blocks inside theta.
    pub fn parameter_layout(&self) -> ParameterLayout {
        self.blocks.parameter_layout()
    }

    /// Unpacks a flat theta vector into named coefficient blocks.
    pub fn unpack_theta(&self, theta: &[f64]) -> Result<UnpackedTheta, ModelError> {
        validate_len("theta", theta.len(), self.nparams())?;

        let blocks = self
            .parameter_layout()
            .slices()
            .iter()
            .map(|slice| ParameterCoefficients {
                name: slice.name,
                coefficients: theta[slice.range.clone()].to_vec(),
            })
            .collect();

        Ok(UnpackedTheta { blocks })
    }

    /// Computes training diagnostics for a candidate theta vector.
    pub fn diagnostics(&self, theta: &[f64]) -> Result<Diagnostics, ModelError> {
        validate_len("theta", theta.len(), self.nparams())?;

        let train_nll = self.blocks.train_nll(&self.family, &self.y, theta);
        let penalty = self.blocks.penalty_value(theta);
        let mut grad = vec![0.0; self.nparams()];
        self.try_gradient_into(theta, &mut grad)?;
        let gradient_norm = grad
            .iter()
            .filter(|value| value.is_finite())
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let nonfinite_gradient_count = grad.iter().filter(|value| !value.is_finite()).count();

        Ok(Diagnostics {
            objective: train_nll + penalty,
            train_nll,
            penalty,
            gradient_norm,
            nonfinite_gradient_count,
        })
    }

    /// Проверяет длину beta и вычисляет objective.
    pub fn try_value(&self, beta: &[f64]) -> Result<f64, ModelError> {
        let expected = self.nparams();
        let actual = beta.len();
        if actual != expected {
            return Err(ModelError::BetaLength { expected, actual });
        }

        Ok(self.blocks.value(&self.family, &self.y, beta))
    }

    /// Проверяет размеры beta/grad и записывает gradient.
    pub fn try_gradient_into(&self, beta: &[f64], grad: &mut [f64]) -> Result<(), ModelError> {
        let mut workspace = self.gradient_workspace();
        self.try_gradient_into_workspace(beta, grad, &mut workspace)
    }

    /// Проверяет размеры beta/grad и записывает gradient, переиспользуя workspace.
    pub fn try_gradient_into_workspace(
        &self,
        beta: &[f64],
        grad: &mut [f64],
        workspace: &mut GradientWorkspace,
    ) -> Result<(), ModelError> {
        let expected = self.nparams();
        let actual_beta = beta.len();
        if actual_beta != expected {
            return Err(ModelError::BetaLength {
                expected,
                actual: actual_beta,
            });
        }

        let actual_grad = grad.len();
        if actual_grad != expected {
            return Err(ModelError::GradientLength {
                expected,
                actual: actual_grad,
            });
        }

        grad.fill(0.0);
        self.blocks
            .gradient_into_workspace(&self.family, &self.y, beta, grad, workspace);
        Ok(())
    }
}

impl<F, Blocks> CachedGamlss<F, Blocks>
where
    Blocks: GamlssBlocks<F>,
{
    /// Creates a cached objective from a compiled model.
    pub fn new(model: Gamlss<F, Blocks>) -> Self {
        model.into_cached_objective()
    }

    /// Returns the wrapped model.
    pub fn model(&self) -> &Gamlss<F, Blocks> {
        &self.model
    }

    /// Returns the wrapped model mutably.
    pub fn model_mut(&mut self) -> &mut Gamlss<F, Blocks> {
        &mut self.model
    }

    /// Consumes the cached objective and returns the wrapped model.
    pub fn into_model(self) -> Gamlss<F, Blocks> {
        self.model
    }

    /// Wraps the cached objective with penalties evaluated on the full beta vector.
    pub fn with_global_penalties<GP>(self, penalties: GP) -> WithGlobalPenalties<Self, GP> {
        WithGlobalPenalties {
            objective: self,
            penalties,
        }
    }
}

impl<F, Blocks> Objective for Gamlss<F, Blocks>
where
    Blocks: GamlssBlocks<F>,
{
    type Error = ModelError;

    fn dim(&self) -> usize {
        self.nparams()
    }

    fn value(&mut self, theta: &[f64]) -> Result<f64, Self::Error> {
        self.try_value(theta)
    }

    fn gradient(&mut self, theta: &[f64], grad: &mut [f64]) -> Result<(), Self::Error> {
        self.try_gradient_into(theta, grad)
    }
}

impl<F, Blocks> Objective for CachedGamlss<F, Blocks>
where
    Blocks: GamlssBlocks<F>,
{
    type Error = ModelError;

    fn dim(&self) -> usize {
        self.model.nparams()
    }

    fn value(&mut self, theta: &[f64]) -> Result<f64, Self::Error> {
        self.model.try_value(theta)
    }

    fn gradient(&mut self, theta: &[f64], grad: &mut [f64]) -> Result<(), Self::Error> {
        self.model
            .try_gradient_into_workspace(theta, grad, &mut self.workspace)
    }
}

impl<O, GP> Objective for WithGlobalPenalties<O, GP>
where
    O: Objective,
    GP: GlobalPenalty,
{
    type Error = O::Error;

    fn dim(&self) -> usize {
        self.objective.dim()
    }

    fn value(&mut self, theta: &[f64]) -> Result<f64, Self::Error> {
        Ok(self.objective.value(theta)? + self.penalties.value(theta))
    }

    fn gradient(&mut self, theta: &[f64], grad: &mut [f64]) -> Result<(), Self::Error> {
        self.objective.gradient(theta, grad)?;
        self.penalties.add_gradient(theta, grad);
        Ok(())
    }
}

/// Макрос, генерирующий реализацию [`GamlssBlocks`] для tuple parameter blocks.
///
/// Принимает арность `K`, списки parameter-типов, link-типов, design-типов и
/// penalty-типов, а также имена внутренних переменных. На выходе даёт
/// zero-cost реализацию `train_nll`, `gradient_into`, `penalty_value` и
/// вспомогательных методов без dynamic dispatch.
macro_rules! impl_gamlss_blocks {
    (
        $k:literal;
        params = ($($param:ident),+);
        links = ($($link:ident),+);
        designs = ($($design:ident),+);
        penalties = ($($penalty:ident),+);
        blocks = ($($block:ident),+);
        beta_blocks = ($($beta_block:ident),+);
        scores = ($($score:ident),+);
        local_grads = ($($local_grad:ident),+);
        indices = ($($idx:tt),+)
    ) => {
        impl<F, $($param, $link, $design, $penalty,)+> GamlssBlocks<F>
            for ($(ParameterBlock<$param, $link, $design, $penalty>,)+)
        where
            F: ParameterizedFamily<$k, Params = ($($param,)+), Links = ($($link,)+)>,
            F::Eta: ParameterParts<$k>,
            F::ScoreEta: ParameterParts<$k>,
            $($param: ParameterName,)+
            $($link: crate::Link<f64>,)+
            $($design: PredictorBlock,)+
            $($penalty: Penalty,)+
        {
            fn nrows(&self) -> usize {
                PredictorBlock::nrows(&self.0.x)
            }

            fn len(&self) -> usize {
                0$(.max(self.$idx.end()))+
            }

            fn validate(&self, y_len: usize) -> Result<(), ModelError> {
                $(
                    self.$idx.x.validate()?;
                    validate_block_rows(
                        <$param as ParameterName>::NAME,
                        PredictorBlock::nrows(&self.$idx.x),
                        y_len,
                    )?;
                )+

                let ranges = [$((
                    <$param as ParameterName>::NAME,
                    self.$idx.range(),
                ),)+];
                for first in 0..ranges.len() {
                    for second in first + 1..ranges.len() {
                        if ranges_overlap(ranges[first].1.clone(), ranges[second].1.clone()) {
                            return Err(ModelError::BlockOverlap {
                                first: ranges[first].0,
                                second: ranges[second].0,
                            });
                        }
                    }
                }

                Ok(())
            }

            fn train_nll(&self, family: &F, y: &[f64], beta: &[f64]) -> f64 {
                $(let $block = &self.$idx;)+
                $(let $beta_block = &beta[$block.range()];)+
                let mut loss = 0.0;

                for (row, y_value) in y.iter().copied().enumerate() {
                    let eta = F::Eta::from_array([$($block.x.eta_row(row, $beta_block),)+]);
                    loss += family.nll_eta(y_value, eta);
                }

                loss
            }

            fn penalty_value(&self, beta: &[f64]) -> f64 {
                $(let $block = &self.$idx;)+
                $(let $beta_block = &beta[$block.range()];)+

                0.0 $(+ $block.penalty.value($beta_block))+
            }

            fn gradient_into(&self, family: &F, y: &[f64], beta: &[f64], grad: &mut [f64]) {
                let mut workspace = GradientWorkspace::new();
                workspace.prepare($k);
                $(
                    workspace.prepare_score($idx, y.len());
                    let _ = workspace.local_gradient_mut($idx, self.$idx.len());
                )+
                self.gradient_into_workspace(family, y, beta, grad, &mut workspace);
            }

            fn gradient_workspace(&self, y_len: usize) -> GradientWorkspace {
                let mut workspace = GradientWorkspace::new();
                workspace.prepare($k);
                $(
                    workspace.prepare_score($idx, y_len);
                    let _ = workspace.local_gradient_mut($idx, self.$idx.len());
                )+
                workspace
            }

            fn gradient_into_workspace(
                &self,
                family: &F,
                y: &[f64],
                beta: &[f64],
                grad: &mut [f64],
                workspace: &mut GradientWorkspace,
            ) {
                $(let $block = &self.$idx;)+
                $(let $beta_block = &beta[$block.range()];)+
                workspace.prepare($k);
                $(workspace.prepare_score($idx, y.len());)+

                for (row, y_value) in y.iter().copied().enumerate() {
                    let eta = F::Eta::from_array([$($block.x.eta_row(row, $beta_block),)+]);
                    let (_, score) = family.nll_and_score_eta(y_value, eta);
                    $(workspace.set_score($idx, row, score.part($idx));)+
                }

                $(
                    let ($score, $local_grad) =
                        workspace.score_and_local_gradient_mut($idx, $block.len());
                    $block.x.add_gradient($score, $beta_block, $local_grad);
                    $block.penalty.add_gradient($beta_block, $local_grad);
                    add_into(&mut grad[$block.range()], $local_grad);
                )+
            }

            fn block_ranges(&self) -> Vec<Range<usize>> {
                vec![$(self.$idx.range(),)+]
            }

            fn parameter_layout(&self) -> ParameterLayout {
                ParameterLayout::new(vec![$(
                    ParameterSlice {
                        name: <$param as ParameterName>::NAME,
                        range: self.$idx.range(),
                    },
                )+])
            }
        }
    };
}

impl_gamlss_blocks!(
    1;
    params = (P1);
    links = (L1);
    designs = (X1);
    penalties = (Pen1);
    blocks = (block1);
    beta_blocks = (beta1);
    scores = (score1);
    local_grads = (grad1);
    indices = (0)
);

impl_gamlss_blocks!(
    2;
    params = (P1, P2);
    links = (L1, L2);
    designs = (X1, X2);
    penalties = (Pen1, Pen2);
    blocks = (block1, block2);
    beta_blocks = (beta1, beta2);
    scores = (score1, score2);
    local_grads = (grad1, grad2);
    indices = (0, 1)
);

impl_gamlss_blocks!(
    3;
    params = (P1, P2, P3);
    links = (L1, L2, L3);
    designs = (X1, X2, X3);
    penalties = (Pen1, Pen2, Pen3);
    blocks = (block1, block2, block3);
    beta_blocks = (beta1, beta2, beta3);
    scores = (score1, score2, score3);
    local_grads = (grad1, grad2, grad3);
    indices = (0, 1, 2)
);

impl_gamlss_blocks!(
    4;
    params = (P1, P2, P3, P4);
    links = (L1, L2, L3, L4);
    designs = (X1, X2, X3, X4);
    penalties = (Pen1, Pen2, Pen3, Pen4);
    blocks = (block1, block2, block3, block4);
    beta_blocks = (beta1, beta2, beta3, beta4);
    scores = (score1, score2, score3, score4);
    local_grads = (grad1, grad2, grad3, grad4);
    indices = (0, 1, 2, 3)
);

/// Проверяет, что число строк predictor-а совпадает с длиной response.
fn validate_block_rows(
    parameter: &'static str,
    actual_rows: usize,
    expected_rows: usize,
) -> Result<(), ModelError> {
    if actual_rows == expected_rows {
        Ok(())
    } else {
        Err(ModelError::DesignRowMismatch {
            parameter,
            expected_rows,
            actual_rows,
        })
    }
}

/// Проверяет пересечение двух диапазонов (непустое пересечение).
fn ranges_overlap(first: Range<usize>, second: Range<usize>) -> bool {
    first.start < second.end && second.start < first.end
}

/// Поэлементно добавляет `values` к `out`.
///
/// Вызывающий код должен гарантировать `out.len() == values.len()`.
fn add_into(out: &mut [f64], values: &[f64]) {
    debug_assert_eq!(out.len(), values.len());

    for (out_value, value) in out.iter_mut().zip(values) {
        *out_value += value;
    }
}

/// Проверяет длину вектора (beta или gradient) и возвращает typed error.
fn validate_len(name: &'static str, actual: usize, expected: usize) -> Result<(), ModelError> {
    if actual == expected {
        Ok(())
    } else if name == "gradient" {
        Err(ModelError::GradientLength { expected, actual })
    } else {
        Err(ModelError::BetaLength { expected, actual })
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{
        DenseDesign, Family, Gamlss, GlobalPenalty, Identity, Mu, NoPenalty, Nu, Objective,
        ParameterBlock, ParameterizedFamily, PredictorBlock, RidgePenalty, Sigma, SumBlock,
    };

    #[derive(Debug, Clone, Copy)]
    struct FixedSigmaNormal;

    impl Family for FixedSigmaNormal {
        type Eta = f64;
        type Theta = f64;
        type ScoreEta = f64;

        fn theta(&self, eta: Self::Eta) -> Self::Theta {
            eta
        }

        fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
            let residual = y - theta;
            0.5 * residual * residual
        }

        fn nll_and_score_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::ScoreEta) {
            (self.nll(y, self.theta(eta)), eta - y)
        }
    }

    impl ParameterizedFamily<1> for FixedSigmaNormal {
        type Params = (Mu,);
        type Links = (Identity,);
    }

    #[test]
    fn custom_one_parameter_family_uses_generic_family_contract() {
        let y = vec![1.0, 2.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let mut model = Gamlss::try_new(FixedSigmaNormal, (mu,), y).unwrap();
        let beta = vec![1.5];
        let mut grad = vec![0.0];

        assert_relative_eq!(model.value(&beta).unwrap(), 0.25);

        model.gradient(&beta, &mut grad).unwrap();

        assert_relative_eq!(grad[0], 0.0);
    }

    #[test]
    fn cached_objective_matches_model_gradient_on_repeated_calls() {
        let y = vec![1.0, 2.0];
        let x = DenseDesign::from_rows(&[[1.0, 0.0], [1.0, 1.0]]);
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new(FixedSigmaNormal, (mu,), y).unwrap();
        let mut cached = model.clone().into_cached_objective();

        for beta in [vec![1.0, 0.25], vec![1.5, -0.1]] {
            let mut expected_grad = vec![0.0; beta.len()];
            let mut cached_grad = vec![0.0; beta.len()];

            model.try_gradient_into(&beta, &mut expected_grad).unwrap();
            cached.gradient(&beta, &mut cached_grad).unwrap();

            assert_relative_eq!(
                cached.value(&beta).unwrap(),
                model.try_value(&beta).unwrap()
            );
            for (actual, expected) in cached_grad.iter().zip(&expected_grad) {
                assert_relative_eq!(actual, expected);
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct SoftplusIntercept {
        nrows: usize,
    }

    impl PredictorBlock for SoftplusIntercept {
        fn nrows(&self) -> usize {
            self.nrows
        }

        fn nparams(&self) -> usize {
            1
        }

        fn eta_row(&self, _: usize, beta: &[f64]) -> f64 {
            softplus(beta[0])
        }

        fn add_gradient(&self, scores: &[f64], beta: &[f64], grad: &mut [f64]) {
            debug_assert_eq!(grad.len(), 1);
            grad[0] += scores.iter().sum::<f64>() * sigmoid(beta[0]);
        }
    }

    #[test]
    fn sum_block_supports_user_defined_nonlinear_predictors() {
        let y = vec![1.0, 2.0];
        let linear = crate::LinearPredictorBlock::new(DenseDesign::intercept(y.len()));
        let nonlinear = SoftplusIntercept { nrows: y.len() };
        let predictor = SumBlock::new((linear, nonlinear));
        let mu = ParameterBlock::<Mu, Identity, _, _>::new(predictor, NoPenalty, 0);
        let mut model = Gamlss::try_new(FixedSigmaNormal, (mu,), y).unwrap();
        let beta = vec![0.4, -0.2];
        let eps = 1.0e-6;
        let mut grad = vec![0.0; beta.len()];

        model.gradient(&beta, &mut grad).unwrap();

        for index in 0..beta.len() {
            let mut plus = beta.clone();
            plus[index] += eps;
            let mut minus = beta.clone();
            minus[index] -= eps;
            let finite_difference =
                (model.value(&plus).unwrap() - model.value(&minus).unwrap()) / (2.0 * eps);

            assert_relative_eq!(grad[index], finite_difference, epsilon = 1.0e-6);
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct StatefulLocation {
        target_shift: f64,
    }

    impl Family for StatefulLocation {
        type Eta = f64;
        type Theta = f64;
        type ScoreEta = f64;

        fn theta(&self, eta: Self::Eta) -> Self::Theta {
            eta + self.target_shift
        }

        fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
            let residual = y - theta;
            0.5 * residual * residual
        }

        fn nll_and_score_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::ScoreEta) {
            let theta = self.theta(eta);
            (self.nll(y, theta), theta - y)
        }
    }

    impl ParameterizedFamily<1> for StatefulLocation {
        type Params = (Mu,);
        type Links = (Identity,);
    }

    #[test]
    fn family_instance_state_participates_in_objective() {
        let y = vec![2.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let mut model = Gamlss::try_new(StatefulLocation { target_shift: 1.0 }, (mu,), y).unwrap();
        let beta = vec![0.5];
        let mut grad = vec![0.0];

        assert_relative_eq!(model.value(&beta).unwrap(), 0.125);

        model.gradient(&beta, &mut grad).unwrap();

        assert_relative_eq!(grad[0], -0.5);
    }

    #[derive(Debug, Clone, Copy)]
    struct ThreeParameterMock;

    impl Family for ThreeParameterMock {
        type Eta = (f64, f64, f64);
        type Theta = (f64, f64, f64);
        type ScoreEta = (f64, f64, f64);

        fn theta(&self, eta: Self::Eta) -> Self::Theta {
            eta
        }

        fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
            let first = theta.0 - y;
            let second = theta.1 - 1.0;
            let third = theta.2 + 1.0;
            0.5 * (first * first + second * second + third * third)
        }

        fn nll_and_score_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::ScoreEta) {
            let score = (eta.0 - y, eta.1 - 1.0, eta.2 + 1.0);
            (self.nll(y, eta), score)
        }
    }

    impl ParameterizedFamily<3> for ThreeParameterMock {
        type Params = (Mu, Sigma, Nu);
        type Links = (Identity, Identity, Identity);
    }

    #[test]
    fn custom_three_parameter_family_uses_generic_blocks() {
        let y = vec![2.0];
        let first = ParameterBlock::<Mu, Identity, _, _>::linear(
            DenseDesign::intercept(y.len()),
            NoPenalty,
            0,
        );
        let second = ParameterBlock::<Sigma, Identity, _, _>::linear(
            DenseDesign::intercept(y.len()),
            NoPenalty,
            1,
        );
        let third = ParameterBlock::<Nu, Identity, _, _>::linear(
            DenseDesign::intercept(y.len()),
            NoPenalty,
            2,
        );
        let mut model = Gamlss::try_new(ThreeParameterMock, (first, second, third), y).unwrap();
        let beta = vec![1.5, 0.5, -0.5];
        let mut grad = vec![0.0; 3];

        assert_relative_eq!(model.value(&beta).unwrap(), 0.375);

        model.gradient(&beta, &mut grad).unwrap();

        assert_relative_eq!(grad[0], -0.5);
        assert_relative_eq!(grad[1], -0.5);
        assert_relative_eq!(grad[2], 0.5);
    }

    #[test]
    fn parameter_layout_and_unpack_use_distribution_parameter_names() {
        let y = vec![2.0];
        let first = ParameterBlock::<Mu, Identity, _, _>::linear(
            DenseDesign::intercept(y.len()),
            NoPenalty,
            0,
        );
        let second = ParameterBlock::<Sigma, Identity, _, _>::linear(
            DenseDesign::intercept(y.len()),
            NoPenalty,
            1,
        );
        let third = ParameterBlock::<Nu, Identity, _, _>::linear(
            DenseDesign::intercept(y.len()),
            NoPenalty,
            2,
        );
        let model = Gamlss::try_new(ThreeParameterMock, (first, second, third), y).unwrap();
        let theta = vec![1.5, 0.5, -0.5];
        let layout = model.parameter_layout();
        let unpacked = model.unpack_theta(&theta).unwrap();

        assert_eq!(layout.slice("mu").unwrap(), 0..1);
        assert_eq!(layout.slice("sigma").unwrap(), 1..2);
        assert_eq!(layout.slice("nu").unwrap(), 2..3);
        assert_eq!(unpacked.coefficients("mu").unwrap(), &[1.5]);
        assert_eq!(unpacked.coefficients("sigma").unwrap(), &[0.5]);
        assert_eq!(unpacked.coefficients("nu").unwrap(), &[-0.5]);
    }

    #[test]
    fn diagnostics_report_train_nll_penalty_and_gradient_norm() {
        let y = vec![1.0, 2.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, RidgePenalty::new(0.5), 0);
        let model = Gamlss::try_new(FixedSigmaNormal, (mu,), y).unwrap();
        let theta = vec![1.5];
        let diagnostics = model.diagnostics(&theta).unwrap();

        assert_relative_eq!(diagnostics.train_nll, 0.25);
        assert_relative_eq!(diagnostics.penalty, 1.125);
        assert_relative_eq!(diagnostics.objective, 1.375);
        assert_relative_eq!(diagnostics.gradient_norm, 1.5);
        assert_eq!(diagnostics.nonfinite_gradient_count, 0);
    }

    #[derive(Debug, Clone, Copy)]
    struct DifferenceGlobalPenalty {
        lambda: f64,
    }

    impl GlobalPenalty for DifferenceGlobalPenalty {
        fn value(&self, beta: &[f64]) -> f64 {
            let diff = beta[0] - beta[1];
            self.lambda * diff * diff
        }

        fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
            let diff = beta[0] - beta[1];
            let slope = 2.0 * self.lambda * diff;
            grad[0] += slope;
            grad[1] -= slope;
        }
    }

    #[test]
    fn global_penalty_adds_value_and_gradient_to_full_objective() {
        let y = vec![0.0, 0.0];
        let x = DenseDesign::from_rows(&[[1.0, 0.0], [0.0, 1.0]]);
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let mut model = Gamlss::try_new(FixedSigmaNormal, (mu,), y)
            .unwrap()
            .with_global_penalties(DifferenceGlobalPenalty { lambda: 1.0 });
        let beta = vec![1.0, -1.0];
        let mut grad = vec![0.0; beta.len()];

        assert_relative_eq!(model.value(&beta).unwrap(), 5.0);

        model.gradient(&beta, &mut grad).unwrap();

        assert_relative_eq!(grad[0], 5.0);
        assert_relative_eq!(grad[1], -5.0);
    }

    fn softplus(value: f64) -> f64 {
        if value > 30.0 {
            value
        } else if value < -30.0 {
            value.exp()
        } else {
            value.exp().ln_1p()
        }
    }

    fn sigmoid(value: f64) -> f64 {
        if value >= 0.0 {
            1.0 / (1.0 + (-value).exp())
        } else {
            let exp_value = value.exp();
            exp_value / (1.0 + exp_value)
        }
    }
}
