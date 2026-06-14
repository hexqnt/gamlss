use std::ops::Range;

use crate::{
    BlockObjective, Family, GlobalPenalty, ModelError, Objective, ParameterBlock, ParameterName,
    ParameterParts, ParameterizedFamily, Penalty, PredictorBlock,
};

/// Tuple-контракт для набора parameter blocks, совместимого с family `F`.
///
/// Implementations are generated for typed tuples of [`ParameterBlock`]. The
/// model validates observation count, predictor row counts and coefficient
/// ranges before hot-path evaluation; generated methods may then assume
/// compatible row counts, finite non-negative weights and non-overlapping block
/// ranges.
pub trait GamlssBlocks<F> {
    /// Число наблюдений в blocks.
    fn nrows(&self) -> usize;
    /// Длина общего beta-вектора, покрывающего все blocks.
    fn len(&self) -> usize;

    /// `true`, если blocks не требуют коэффициентов.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Проверяет, что blocks совместимы с observation count `nobs`.
    fn validate(&self, nobs: usize) -> Result<(), ModelError>;
    /// Weighted negative log-likelihood without penalties.
    ///
    /// `obs` has already been validated by the model constructor. Each scalar
    /// likelihood contribution is multiplied by the corresponding observation
    /// weight.
    fn train_nll<Obs>(&self, family: &F, obs: &Obs, beta: &[f64]) -> f64
    where
        Obs: ObservationView;
    /// Аддитивные предикторы на link-шкале для одной строки.
    fn eta_row(&self, beta: &[f64], row: usize) -> F::Eta
    where
        F: Family;
    /// Penalty value depending on coefficient blocks.
    fn penalty_value(&self, beta: &[f64]) -> f64;
    /// Значение weighted negative log-likelihood плюс penalties.
    fn value<Obs>(&self, family: &F, obs: &Obs, beta: &[f64]) -> f64
    where
        Obs: ObservationView,
    {
        self.train_nll(family, obs, beta) + self.penalty_value(beta)
    }
    /// Creates reusable buffers for repeated gradient evaluations.
    fn gradient_workspace(&self, nobs: usize) -> GradientWorkspace {
        let mut workspace = GradientWorkspace::new();
        let ranges = self.block_ranges();
        workspace.prepare(ranges.len());
        for (index, range) in ranges.iter().enumerate() {
            workspace.prepare_score(index, nobs);
            let _ = workspace.local_gradient_mut(index, range.len());
        }
        workspace
    }
    /// Добавляет weighted gradient, переиспользуя временные буферы из `workspace`.
    fn gradient_into_workspace<Obs>(
        &self,
        family: &F,
        obs: &Obs,
        beta: &[f64],
        grad: &mut [f64],
        workspace: &mut GradientWorkspace,
    ) where
        Obs: ObservationView;
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
/// The workspace is model-shape specific but can be reused across different
/// beta vectors for the same compiled model.
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

/// Read-only scalar observation access for training objective evaluation.
///
/// This trait is intentionally small: it describes the row-wise data needed by
/// the current univariate likelihood loop. Implementations should make
/// [`len`](Self::len) O(1), keep it stable for the lifetime of the model, and
/// provide deterministic, panic-free access for `row < len()`.
pub trait ObservationView {
    /// Number of observations.
    fn len(&self) -> usize;

    /// Returns `true` if there are no observations.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Response value for `row`.
    fn y_at(&self, row: usize) -> f64;

    /// Non-negative finite observation weight for `row`.
    fn weight_at(&self, row: usize) -> f64;

    /// Validates observation-level invariants before hot-path evaluation.
    fn validate(&self) -> Result<(), ModelError> {
        for row in 0..self.len() {
            validate_observation_weight(row, self.weight_at(row))?;
        }
        Ok(())
    }
}

impl ObservationView for &[f64] {
    fn len(&self) -> usize {
        <[f64]>::len(self)
    }

    fn y_at(&self, row: usize) -> f64 {
        self[row]
    }

    fn weight_at(&self, _row: usize) -> f64 {
        1.0
    }

    fn validate(&self) -> Result<(), ModelError> {
        Ok(())
    }
}

impl ObservationView for (&[f64], &[f64]) {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn y_at(&self, row: usize) -> f64 {
        self.0[row]
    }

    fn weight_at(&self, row: usize) -> f64 {
        self.1[row]
    }

    fn validate(&self) -> Result<(), ModelError> {
        let expected = self.0.len();
        let actual = self.1.len();
        if actual != expected {
            return Err(ModelError::WeightLength { expected, actual });
        }

        for (index, weight) in self.1.iter().copied().enumerate() {
            validate_observation_weight(index, weight)?;
        }

        Ok(())
    }
}

/// Скомпилированная типизированная GAMLSS-модель.
///
/// `F` задаёт распределение response, а `Blocks` задаёт по одному predictor
/// block для каждого параметра family.
///
/// The model owns the family and parameter blocks, and stores an observation
/// view supplied by the caller. This keeps `gamlss-core` independent of the
/// caller's storage backend while static dispatch preserves zero-cost hot-path
/// evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct Gamlss<F, Blocks, Obs> {
    /// Family распределения response.
    pub family: F,
    /// Типизированные parameter blocks.
    pub blocks: Blocks,
    /// Observation view used for training objective evaluation.
    pub obs: Obs,
}

/// GAMLSS objective with reusable gradient buffers.
///
/// This wrapper is intended for optimizers that call `gradient` repeatedly.
/// It owns the compiled model and keeps a [`GradientWorkspace`] between calls,
/// avoiding per-call allocation of score and local-gradient vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkspaceGamlss<F, Blocks, Obs> {
    /// Wrapped compiled model.
    pub model: Gamlss<F, Blocks, Obs>,
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

/// Диагностики обучения для кандидата theta.
///
/// Содержит значения objective, weighted negative log-likelihood (без штрафов),
/// суммарный штраф, норму градиента и число не-finite компонент градиента.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Diagnostics {
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

impl<F, Blocks, Obs> Gamlss<F, Blocks, Obs> {
    /// Wraps the model with penalties evaluated on the full beta vector.
    pub fn with_global_penalties<GP>(self, penalties: GP) -> WithGlobalPenalties<Self, GP> {
        WithGlobalPenalties {
            objective: self,
            penalties,
        }
    }
}

impl<F, Blocks, Obs> Gamlss<F, Blocks, Obs>
where
    Blocks: GamlssBlocks<F>,
    Obs: ObservationView,
{
    /// Создаёт модель после проверки observation view и blocks.
    ///
    /// This is the extension point for custom storage backends. The observation
    /// view is stored by value, so callers can pass lightweight borrowed views,
    /// owned adapters, or newtypes around external dataframe/columnar storage.
    pub fn try_new_with_observations(
        family: F,
        blocks: Blocks,
        obs: Obs,
    ) -> Result<Self, ModelError> {
        if obs.is_empty() {
            return Err(ModelError::EmptyResponse);
        }

        obs.validate()?;
        blocks.validate(obs.len())?;
        Ok(Self {
            family,
            blocks,
            obs,
        })
    }

    /// Число наблюдений.
    pub fn nobs(&self) -> usize {
        self.obs.len()
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
        self.blocks.gradient_workspace(self.obs.len())
    }

    /// Wraps the model as an objective with reusable gradient buffers.
    pub fn into_workspace_objective(self) -> WorkspaceGamlss<F, Blocks, Obs> {
        let workspace = self.gradient_workspace();
        WorkspaceGamlss {
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

    /// Creates a [`BlockObjective`] projected to the coefficients of parameter `P`.
    ///
    /// This is the zero-cost building block for staged/block-wise fitting:
    /// optimise one distribution parameter (e.g. `Mu`) while keeping the
    /// remaining coefficients fixed at the values in `full_beta`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::UnknownParameter`] if the model blocks do not
    /// contain a parameter named `P::NAME`. When blocks are constructed through
    /// the typed [`ParameterBlock`] API this cannot happen in practice — the
    /// compiler guarantees that `ParameterBlock<Mu, …>` registers itself as
    /// `"mu"`.
    pub fn block_objective_for<P>(
        &mut self,
        full_beta: Vec<f64>,
    ) -> Result<BlockObjective<'_, Self>, ModelError>
    where
        P: ParameterName,
    {
        let range = self
            .parameter_layout()
            .slice_of::<P>()
            .ok_or(ModelError::UnknownParameter { name: P::NAME })?;
        Ok(BlockObjective::new(self, full_beta, range))
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

        let train_nll = self.blocks.train_nll(&self.family, &self.obs, theta);
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

    /// Predicts link-scale distribution predictors for one training row.
    pub fn predict_eta_row(&self, theta: &[f64], row: usize) -> Result<F::Eta, ModelError>
    where
        F: Family,
    {
        validate_len("theta", theta.len(), self.nparams())?;
        validate_row(row, self.nobs())?;
        Ok(self.blocks.eta_row(theta, row))
    }

    /// Predicts natural-scale distribution parameters for one training row.
    pub fn predict_theta_row(&self, theta: &[f64], row: usize) -> Result<F::Theta, ModelError>
    where
        F: Family,
    {
        Ok(self.family.theta(self.predict_eta_row(theta, row)?))
    }

    /// Predicts link-scale distribution predictors for all training rows.
    pub fn predict_eta(&self, theta: &[f64]) -> Result<Vec<F::Eta>, ModelError>
    where
        F: Family,
    {
        validate_len("theta", theta.len(), self.nparams())?;
        Ok((0..self.nobs())
            .map(|row| self.blocks.eta_row(theta, row))
            .collect())
    }

    /// Predicts natural-scale distribution parameters for all training rows.
    pub fn predict_theta(&self, theta: &[f64]) -> Result<Vec<F::Theta>, ModelError>
    where
        F: Family,
    {
        validate_len("theta", theta.len(), self.nparams())?;
        Ok((0..self.nobs())
            .map(|row| self.family.theta(self.blocks.eta_row(theta, row)))
            .collect())
    }

    /// Predicts link-scale distribution predictors for one row from compatible prediction blocks.
    ///
    /// # Errors
    ///
    /// Returns an error if `theta` has the wrong length, if `blocks` do not
    /// match this model's parameter layout, or if `row` is out of bounds for
    /// the supplied prediction blocks.
    pub fn predict_eta_row_with_blocks<PBlocks>(
        &self,
        theta: &[f64],
        blocks: &PBlocks,
        row: usize,
    ) -> Result<F::Eta, ModelError>
    where
        F: Family,
        PBlocks: GamlssBlocks<F>,
    {
        validate_len("theta", theta.len(), self.nparams())?;
        validate_prediction_blocks(blocks, self.nparams())?;
        validate_row(row, blocks.nrows())?;
        Ok(blocks.eta_row(theta, row))
    }

    /// Predicts natural-scale distribution parameters for one row from compatible prediction blocks.
    ///
    /// # Errors
    ///
    /// Returns an error if `theta` has the wrong length, if `blocks` do not
    /// match this model's parameter layout, or if `row` is out of bounds for
    /// the supplied prediction blocks.
    pub fn predict_theta_row_with_blocks<PBlocks>(
        &self,
        theta: &[f64],
        blocks: &PBlocks,
        row: usize,
    ) -> Result<F::Theta, ModelError>
    where
        F: Family,
        PBlocks: GamlssBlocks<F>,
    {
        Ok(self
            .family
            .theta(self.predict_eta_row_with_blocks(theta, blocks, row)?))
    }

    /// Predicts link-scale distribution predictors for all rows in compatible prediction blocks.
    ///
    /// # Errors
    ///
    /// Returns an error if `theta` has the wrong length or if `blocks` do not
    /// match this model's parameter layout.
    pub fn predict_eta_with_blocks<PBlocks>(
        &self,
        theta: &[f64],
        blocks: &PBlocks,
    ) -> Result<Vec<F::Eta>, ModelError>
    where
        F: Family,
        PBlocks: GamlssBlocks<F>,
    {
        validate_len("theta", theta.len(), self.nparams())?;
        validate_prediction_blocks(blocks, self.nparams())?;
        Ok((0..blocks.nrows())
            .map(|row| blocks.eta_row(theta, row))
            .collect())
    }

    /// Predicts natural-scale distribution parameters for all rows in compatible prediction blocks.
    ///
    /// # Errors
    ///
    /// Returns an error if `theta` has the wrong length or if `blocks` do not
    /// match this model's parameter layout.
    pub fn predict_theta_with_blocks<PBlocks>(
        &self,
        theta: &[f64],
        blocks: &PBlocks,
    ) -> Result<Vec<F::Theta>, ModelError>
    where
        F: Family,
        PBlocks: GamlssBlocks<F>,
    {
        validate_len("theta", theta.len(), self.nparams())?;
        validate_prediction_blocks(blocks, self.nparams())?;
        Ok((0..blocks.nrows())
            .map(|row| self.family.theta(blocks.eta_row(theta, row)))
            .collect())
    }

    /// Проверяет длину beta и вычисляет objective.
    pub fn try_value(&self, beta: &[f64]) -> Result<f64, ModelError> {
        let expected = self.nparams();
        let actual = beta.len();
        if actual != expected {
            return Err(ModelError::BetaLength { expected, actual });
        }

        Ok(self.blocks.value(&self.family, &self.obs, beta))
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
            .gradient_into_workspace(&self.family, &self.obs, beta, grad, workspace);
        Ok(())
    }
}

impl<'a, F, Blocks> Gamlss<F, Blocks, &'a [f64]>
where
    Blocks: GamlssBlocks<F>,
{
    /// Создаёт unweighted модель после проверки response и blocks.
    pub fn try_new(family: F, blocks: Blocks, y: &'a [f64]) -> Result<Self, ModelError> {
        Self::try_new_with_observations(family, blocks, y)
    }
}

impl<'a, F, Blocks> Gamlss<F, Blocks, (&'a [f64], &'a [f64])>
where
    Blocks: GamlssBlocks<F>,
{
    /// Создаёт модель с observation weights после проверки response, weights и blocks.
    ///
    /// Weights must have the same length as `y`; each weight must be finite and
    /// non-negative. Zero weights are accepted and exclude the corresponding
    /// observation from likelihood and score contributions.
    pub fn try_new_weighted(
        family: F,
        blocks: Blocks,
        y: &'a [f64],
        weights: &'a [f64],
    ) -> Result<Self, ModelError> {
        Self::try_new_with_observations(family, blocks, (y, weights))
    }
}

impl<F, Blocks, Obs> WorkspaceGamlss<F, Blocks, Obs>
where
    Blocks: GamlssBlocks<F>,
    Obs: ObservationView,
{
    /// Creates a workspace-backed objective from a compiled model.
    pub fn new(model: Gamlss<F, Blocks, Obs>) -> Self {
        model.into_workspace_objective()
    }

    /// Returns the wrapped model.
    pub fn model(&self) -> &Gamlss<F, Blocks, Obs> {
        &self.model
    }

    /// Returns the wrapped model mutably.
    pub fn model_mut(&mut self) -> &mut Gamlss<F, Blocks, Obs> {
        &mut self.model
    }

    /// Consumes the workspace-backed objective and returns the wrapped model.
    pub fn into_model(self) -> Gamlss<F, Blocks, Obs> {
        self.model
    }

    /// Creates a [`BlockObjective`] projected to the coefficients of parameter `P`.
    ///
    /// Delegates to the inner model's [`Gamlss::block_objective_for`] through
    /// [`model_mut`](Self::model_mut), so the returned objective borrows the
    /// workspace-backed model and reuses its gradient buffers.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::UnknownParameter`] if the model blocks do not
    /// contain a parameter named `P::NAME`.
    pub fn block_objective_for<P>(
        &mut self,
        full_beta: Vec<f64>,
    ) -> Result<BlockObjective<'_, Self>, ModelError>
    where
        P: ParameterName,
    {
        let range = self
            .model
            .parameter_layout()
            .slice_of::<P>()
            .ok_or(ModelError::UnknownParameter { name: P::NAME })?;
        Ok(BlockObjective::new(self, full_beta, range))
    }

    /// Wraps the workspace-backed objective with penalties evaluated on the full beta vector.
    pub fn with_global_penalties<GP>(self, penalties: GP) -> WithGlobalPenalties<Self, GP> {
        WithGlobalPenalties {
            objective: self,
            penalties,
        }
    }
}

impl<F, Blocks, Obs> Objective for Gamlss<F, Blocks, Obs>
where
    Blocks: GamlssBlocks<F>,
    Obs: ObservationView,
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

impl<F, Blocks, Obs> Objective for WorkspaceGamlss<F, Blocks, Obs>
where
    Blocks: GamlssBlocks<F>,
    Obs: ObservationView,
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
/// zero-cost реализацию `train_nll`, `gradient_into_workspace`, `penalty_value` и
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
                0$(.max(self.$idx.offset.saturating_add(self.$idx.len)))+
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
                    self.$idx.try_range()?,
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

            fn train_nll<Obs>(
                &self,
                family: &F,
                obs: &Obs,
                beta: &[f64],
            ) -> f64
            where
                Obs: ObservationView,
            {
                $(let $block = &self.$idx;)+
                $(let $beta_block = &beta[$block.range()];)+
                let mut loss = 0.0;

                for row in 0..obs.len() {
                    let y_value = obs.y_at(row);
                    let eta = F::Eta::from_array([$($block.x.eta_row(row, $beta_block),)+]);
                    loss += obs.weight_at(row) * family.nll_eta(y_value, eta);
                }

                loss
            }

            fn eta_row(&self, beta: &[f64], row: usize) -> F::Eta
            where
                F: Family,
            {
                $(let $block = &self.$idx;)+
                $(let $beta_block = &beta[$block.range()];)+
                F::Eta::from_array([$($block.x.eta_row(row, $beta_block),)+])
            }

            fn penalty_value(&self, beta: &[f64]) -> f64 {
                $(let $block = &self.$idx;)+
                $(let $beta_block = &beta[$block.range()];)+

                0.0 $(+ $block.penalty.value($beta_block))+
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

            fn gradient_into_workspace<Obs>(
                &self,
                family: &F,
                obs: &Obs,
                beta: &[f64],
                grad: &mut [f64],
                workspace: &mut GradientWorkspace,
            )
            where
                Obs: ObservationView,
            {
                $(let $block = &self.$idx;)+
                $(let $beta_block = &beta[$block.range()];)+
                workspace.prepare($k);
                $(workspace.prepare_score($idx, obs.len());)+

                for row in 0..obs.len() {
                    let y_value = obs.y_at(row);
                    let eta = F::Eta::from_array([$($block.x.eta_row(row, $beta_block),)+]);
                    let (_, score) = family.nll_and_score_eta(y_value, eta);
                    let weight = obs.weight_at(row);
                    $(workspace.set_score($idx, row, weight * score.part($idx));)+
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

impl_gamlss_blocks!(
    5;
    params = (P1, P2, P3, P4, P5);
    links = (L1, L2, L3, L4, L5);
    designs = (X1, X2, X3, X4, X5);
    penalties = (Pen1, Pen2, Pen3, Pen4, Pen5);
    blocks = (block1, block2, block3, block4, block5);
    beta_blocks = (beta1, beta2, beta3, beta4, beta5);
    scores = (score1, score2, score3, score4, score5);
    local_grads = (grad1, grad2, grad3, grad4, grad5);
    indices = (0, 1, 2, 3, 4)
);

impl_gamlss_blocks!(
    6;
    params = (P1, P2, P3, P4, P5, P6);
    links = (L1, L2, L3, L4, L5, L6);
    designs = (X1, X2, X3, X4, X5, X6);
    penalties = (Pen1, Pen2, Pen3, Pen4, Pen5, Pen6);
    blocks = (block1, block2, block3, block4, block5, block6);
    beta_blocks = (beta1, beta2, beta3, beta4, beta5, beta6);
    scores = (score1, score2, score3, score4, score5, score6);
    local_grads = (grad1, grad2, grad3, grad4, grad5, grad6);
    indices = (0, 1, 2, 3, 4, 5)
);

impl_gamlss_blocks!(
    7;
    params = (P1, P2, P3, P4, P5, P6, P7);
    links = (L1, L2, L3, L4, L5, L6, L7);
    designs = (X1, X2, X3, X4, X5, X6, X7);
    penalties = (Pen1, Pen2, Pen3, Pen4, Pen5, Pen6, Pen7);
    blocks = (block1, block2, block3, block4, block5, block6, block7);
    beta_blocks = (beta1, beta2, beta3, beta4, beta5, beta6, beta7);
    scores = (score1, score2, score3, score4, score5, score6, score7);
    local_grads = (grad1, grad2, grad3, grad4, grad5, grad6, grad7);
    indices = (0, 1, 2, 3, 4, 5, 6)
);

impl_gamlss_blocks!(
    8;
    params = (P1, P2, P3, P4, P5, P6, P7, P8);
    links = (L1, L2, L3, L4, L5, L6, L7, L8);
    designs = (X1, X2, X3, X4, X5, X6, X7, X8);
    penalties = (Pen1, Pen2, Pen3, Pen4, Pen5, Pen6, Pen7, Pen8);
    blocks = (block1, block2, block3, block4, block5, block6, block7, block8);
    beta_blocks = (beta1, beta2, beta3, beta4, beta5, beta6, beta7, beta8);
    scores = (score1, score2, score3, score4, score5, score6, score7, score8);
    local_grads = (grad1, grad2, grad3, grad4, grad5, grad6, grad7, grad8);
    indices = (0, 1, 2, 3, 4, 5, 6, 7)
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

fn validate_observation_weight(index: usize, weight: f64) -> Result<(), ModelError> {
    if weight.is_finite() && weight >= 0.0 {
        Ok(())
    } else {
        Err(ModelError::InvalidWeight { index })
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

fn validate_row(row: usize, nrows: usize) -> Result<(), ModelError> {
    if row < nrows {
        Ok(())
    } else {
        Err(ModelError::RowOutOfBounds { row, nrows })
    }
}

fn validate_prediction_blocks<F, Blocks>(
    blocks: &Blocks,
    expected_len: usize,
) -> Result<(), ModelError>
where
    Blocks: GamlssBlocks<F>,
{
    blocks.validate(blocks.nrows())?;
    let actual = blocks.len();
    if actual == expected_len {
        Ok(())
    } else {
        Err(ModelError::PredictionParameterLength {
            expected: expected_len,
            actual,
        })
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{
        DenseDesign, Family, Gamlss, GlobalPenalty, Identity, LinearPredictorBlock, ModelError, Mu,
        NoPenalty, Nu, Objective, ObservationView, ParameterBlock, ParameterBlocks, ParameterName,
        ParameterizedFamily, PredictorBlock, RidgePenalty, Sigma, SumBlock, Tau,
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

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct ShiftedObservations<'a> {
        y: &'a [f64],
        shift: f64,
        weight: f64,
    }

    impl ObservationView for ShiftedObservations<'_> {
        fn len(&self) -> usize {
            self.y.len()
        }

        fn y_at(&self, row: usize) -> f64 {
            self.y[row] + self.shift
        }

        fn weight_at(&self, _row: usize) -> f64 {
            self.weight
        }
    }

    #[test]
    fn custom_one_parameter_family_uses_generic_family_contract() {
        let y = vec![1.0, 2.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let mut model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();
        let beta = vec![1.5];
        let mut grad = vec![0.0];

        assert_relative_eq!(model.value(&beta).unwrap(), 0.25);

        model.gradient(&beta, &mut grad).unwrap();

        assert_relative_eq!(grad[0], 0.0);
    }

    #[test]
    fn model_accepts_user_defined_observation_view() {
        let y = vec![1.0, 2.0];
        let obs = ShiftedObservations {
            y: &y,
            shift: 1.0,
            weight: 0.5,
        };
        let x = DenseDesign::intercept(obs.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let mut model = Gamlss::try_new_with_observations(FixedSigmaNormal, (mu,), obs).unwrap();
        let beta = vec![2.5];
        let mut grad = vec![0.0];

        assert_relative_eq!(model.value(&beta).unwrap(), 0.125);

        model.gradient(&beta, &mut grad).unwrap();

        assert_relative_eq!(grad[0], 0.0);
    }

    #[test]
    fn custom_observation_view_rejects_invalid_weight() {
        let y = vec![1.0, 2.0];
        let obs = ShiftedObservations {
            y: &y,
            shift: 0.0,
            weight: f64::NAN,
        };
        let x = DenseDesign::intercept(obs.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);

        assert_eq!(
            Gamlss::try_new_with_observations(FixedSigmaNormal, (mu,), obs).unwrap_err(),
            ModelError::InvalidWeight { index: 0 }
        );
    }

    #[test]
    fn model_borrows_response_without_copying() {
        let y = vec![1.0, 2.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();

        assert_eq!(model.obs.as_ptr(), y.as_ptr());
        assert_eq!(model.obs, y.as_slice());
        assert_eq!(model.obs.weight_at(0), 1.0);
    }

    #[test]
    fn unweighted_model_matches_unit_weights() {
        let y = vec![1.0, 2.0];
        let unit_weights = vec![1.0, 1.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x.clone(), NoPenalty, 0);
        let weighted_mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();
        let weighted =
            Gamlss::try_new_weighted(FixedSigmaNormal, (weighted_mu,), &y, &unit_weights).unwrap();
        let beta = vec![1.5];
        let mut grad = vec![0.0];
        let mut weighted_grad = vec![0.0];

        assert_relative_eq!(
            model.try_value(&beta).unwrap(),
            weighted.try_value(&beta).unwrap()
        );

        model.try_gradient_into(&beta, &mut grad).unwrap();
        weighted
            .try_gradient_into(&beta, &mut weighted_grad)
            .unwrap();

        assert_relative_eq!(grad[0], weighted_grad[0]);
    }

    #[test]
    fn zero_weight_excludes_observation_from_value_and_gradient() {
        let y = vec![1.0, 10.0];
        let weights = vec![1.0, 0.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new_weighted(FixedSigmaNormal, (mu,), &y, &weights).unwrap();
        let beta = vec![1.0];
        let mut grad = vec![f64::NAN];

        assert_relative_eq!(model.try_value(&beta).unwrap(), 0.0);

        model.try_gradient_into(&beta, &mut grad).unwrap();

        assert_relative_eq!(grad[0], 0.0);
    }

    #[test]
    fn weighted_model_rejects_invalid_weights() {
        let y = vec![1.0, 2.0];
        let short_weights = vec![1.0];
        let infinite_weights = vec![1.0, f64::INFINITY];
        let negative_weights = vec![1.0, -0.1];
        let mu =
            ParameterBlock::<Mu, Identity, _, _>::linear(DenseDesign::intercept(2), NoPenalty, 0);

        assert_eq!(
            Gamlss::try_new_weighted(FixedSigmaNormal, (mu.clone(),), &y, &short_weights,)
                .unwrap_err(),
            ModelError::WeightLength {
                expected: 2,
                actual: 1,
            }
        );
        assert_eq!(
            Gamlss::try_new_weighted(FixedSigmaNormal, (mu.clone(),), &y, &infinite_weights,)
                .unwrap_err(),
            ModelError::InvalidWeight { index: 1 }
        );
        assert_eq!(
            Gamlss::try_new_weighted(FixedSigmaNormal, (mu,), &y, &negative_weights).unwrap_err(),
            ModelError::InvalidWeight { index: 1 }
        );
    }

    #[test]
    fn prediction_api_returns_eta_and_theta_for_one_parameter_model() {
        let y = vec![1.0, 2.0];
        let x = DenseDesign::from_rows(&[[1.0, 0.0], [1.0, 2.0]]);
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();
        let beta = vec![0.5, 0.25];

        assert_relative_eq!(model.predict_eta_row(&beta, 1).unwrap(), 1.0);
        assert_relative_eq!(model.predict_theta_row(&beta, 1).unwrap(), 1.0);
        assert_eq!(model.predict_eta(&beta).unwrap(), vec![0.5, 1.0]);
        assert_eq!(model.predict_theta(&beta).unwrap(), vec![0.5, 1.0]);
    }

    #[test]
    fn prediction_api_rejects_invalid_theta_length_and_row() {
        let y = vec![1.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();

        assert_eq!(
            model.predict_eta_row(&[], 0).unwrap_err(),
            ModelError::BetaLength {
                expected: 1,
                actual: 0,
            }
        );
        assert_eq!(
            model.predict_eta_row(&[0.0], 1).unwrap_err(),
            ModelError::RowOutOfBounds { row: 1, nrows: 1 }
        );
    }

    #[test]
    fn prediction_api_uses_compatible_blocks_for_new_rows() {
        let y = vec![1.0, 2.0];
        let train_x = DenseDesign::from_rows(&[[1.0, 0.0], [1.0, 1.0]]);
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(train_x, NoPenalty, 0);
        let model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();
        let prediction_x = DenseDesign::from_rows(&[[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]);
        let prediction_mu =
            ParameterBlock::<Mu, Identity, _, _>::linear(prediction_x, NoPenalty, 0);
        let prediction_blocks = (prediction_mu,);
        let beta = vec![0.5, 0.25];

        assert_relative_eq!(
            model
                .predict_eta_row_with_blocks(&beta, &prediction_blocks, 1)
                .unwrap(),
            1.25
        );
        assert_eq!(
            model
                .predict_eta_with_blocks(&beta, &prediction_blocks)
                .unwrap(),
            vec![1.0, 1.25, 1.5]
        );
        assert_eq!(
            model
                .predict_theta_with_blocks(&beta, &prediction_blocks)
                .unwrap(),
            vec![1.0, 1.25, 1.5]
        );
    }

    #[test]
    fn prediction_api_rejects_incompatible_prediction_blocks() {
        let y = vec![1.0, 2.0];
        let train_x = DenseDesign::from_rows(&[[1.0, 0.0], [1.0, 1.0]]);
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(train_x, NoPenalty, 0);
        let model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();
        let prediction_x = DenseDesign::from_rows(&[[1.0, 2.0, 3.0]]);
        let prediction_mu =
            ParameterBlock::<Mu, Identity, _, _>::linear(prediction_x, NoPenalty, 0);
        let prediction_blocks = (prediction_mu,);

        assert_eq!(
            model
                .predict_eta_with_blocks(&[0.5, 0.25], &prediction_blocks)
                .unwrap_err(),
            ModelError::PredictionParameterLength {
                expected: 2,
                actual: 3,
            }
        );
    }

    #[test]
    fn rejects_overflowing_parameter_block_ranges() {
        let y = vec![1.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, usize::MAX);

        assert_eq!(
            Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap_err(),
            ModelError::BlockRangeOverflow {
                parameter: "mu",
                offset: usize::MAX,
                len: 1,
            }
        );
    }

    #[test]
    fn workspace_objective_matches_model_gradient_on_repeated_calls() {
        let y = vec![1.0, 2.0];
        let x = DenseDesign::from_rows(&[[1.0, 0.0], [1.0, 1.0]]);
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, NoPenalty, 0);
        let model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();
        let mut workspace_objective = model.clone().into_workspace_objective();

        for beta in [vec![1.0, 0.25], vec![1.5, -0.1]] {
            let mut expected_grad = vec![0.0; beta.len()];
            let mut workspace_grad = vec![0.0; beta.len()];

            model.try_gradient_into(&beta, &mut expected_grad).unwrap();
            workspace_objective
                .gradient(&beta, &mut workspace_grad)
                .unwrap();

            assert_relative_eq!(
                workspace_objective.value(&beta).unwrap(),
                model.try_value(&beta).unwrap()
            );
            for (actual, expected) in workspace_grad.iter().zip(&expected_grad) {
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
        let mut model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();
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
        let mut model = Gamlss::try_new(StatefulLocation { target_shift: 1.0 }, (mu,), &y).unwrap();
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
        let mut model = Gamlss::try_new(ThreeParameterMock, (first, second, third), &y).unwrap();
        let beta = vec![1.5, 0.5, -0.5];
        let mut grad = vec![0.0; 3];

        assert_relative_eq!(model.value(&beta).unwrap(), 0.375);

        model.gradient(&beta, &mut grad).unwrap();

        assert_relative_eq!(grad[0], -0.5);
        assert_relative_eq!(grad[1], -0.5);
        assert_relative_eq!(grad[2], 0.5);
    }

    #[derive(Debug, Clone, Copy)]
    struct FourParameterMock;

    impl Family for FourParameterMock {
        type Eta = (f64, f64, f64, f64);
        type Theta = (f64, f64, f64, f64);
        type ScoreEta = (f64, f64, f64, f64);

        fn theta(&self, eta: Self::Eta) -> Self::Theta {
            (eta.0 + 1.0, eta.1 + 2.0, eta.2 + 3.0, eta.3 + 4.0)
        }

        fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
            theta.0 + theta.1 + theta.2 + theta.3 + y
        }

        fn nll_and_score_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::ScoreEta) {
            (self.nll(y, self.theta(eta)), (1.0, 1.0, 1.0, 1.0))
        }
    }

    impl ParameterizedFamily<4> for FourParameterMock {
        type Params = (Mu, Sigma, Nu, Tau);
        type Links = (Identity, Identity, Identity, Identity);
    }

    #[derive(Debug, Clone, Copy)]
    struct Fifth;

    impl ParameterName for Fifth {
        const NAME: &'static str = "fifth";
    }

    #[derive(Debug, Clone, Copy)]
    struct FiveParameterMock;

    impl Family for FiveParameterMock {
        type Eta = (f64, f64, f64, f64, f64);
        type Theta = (f64, f64, f64, f64, f64);
        type ScoreEta = (f64, f64, f64, f64, f64);

        fn theta(&self, eta: Self::Eta) -> Self::Theta {
            eta
        }

        fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
            let targets = [y, 1.0, 2.0, 3.0, 4.0];
            let values = [theta.0, theta.1, theta.2, theta.3, theta.4];
            0.5 * values
                .iter()
                .zip(targets)
                .map(|(value, target)| {
                    let residual = value - target;
                    residual * residual
                })
                .sum::<f64>()
        }

        fn nll_and_score_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::ScoreEta) {
            (
                self.nll(y, eta),
                (
                    eta.0 - y,
                    eta.1 - 1.0,
                    eta.2 - 2.0,
                    eta.3 - 3.0,
                    eta.4 - 4.0,
                ),
            )
        }
    }

    impl ParameterizedFamily<5> for FiveParameterMock {
        type Params = (Mu, Sigma, Nu, Tau, Fifth);
        type Links = (Identity, Identity, Identity, Identity, Identity);
    }

    #[test]
    fn prediction_api_returns_eta_and_theta_for_four_parameter_model() {
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
        let fourth = ParameterBlock::<Tau, Identity, _, _>::linear(
            DenseDesign::intercept(y.len()),
            NoPenalty,
            3,
        );
        let model = Gamlss::try_new(FourParameterMock, (first, second, third, fourth), &y).unwrap();
        let beta = vec![0.5, 1.5, 2.5, 3.5];

        assert_eq!(
            model.predict_eta_row(&beta, 0).unwrap(),
            (0.5, 1.5, 2.5, 3.5)
        );
        assert_eq!(
            model.predict_theta_row(&beta, 0).unwrap(),
            (1.5, 3.5, 5.5, 7.5)
        );
    }

    #[test]
    fn custom_five_parameter_family_uses_generic_blocks() {
        let y = vec![2.0];
        let blocks = ParameterBlocks::new((
            intercept_block::<Mu>(y.len()),
            intercept_block::<Sigma>(y.len()),
            intercept_block::<Nu>(y.len()),
            intercept_block::<Tau>(y.len()),
            intercept_block::<Fifth>(y.len()),
        ));
        let mut model = Gamlss::try_new(FiveParameterMock, blocks, &y).unwrap();
        let beta = vec![1.5, 0.5, 1.5, 2.5, 3.5];
        let mut grad = vec![0.0; 5];

        assert_relative_eq!(model.value(&beta).unwrap(), 0.625);

        model.gradient(&beta, &mut grad).unwrap();

        assert_eq!(model.parameter_layout().slice("fifth").unwrap(), 4..5);
        assert_relative_eq!(grad[0], -0.5);
        assert_relative_eq!(grad[1], -0.5);
        assert_relative_eq!(grad[2], -0.5);
        assert_relative_eq!(grad[3], -0.5);
        assert_relative_eq!(grad[4], -0.5);
    }

    fn intercept_block<P>(
        nrows: usize,
    ) -> ParameterBlock<P, Identity, LinearPredictorBlock<DenseDesign>, NoPenalty> {
        ParameterBlock::linear(DenseDesign::intercept(nrows), NoPenalty, 99)
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
        let model = Gamlss::try_new(ThreeParameterMock, (first, second, third), &y).unwrap();
        let theta = vec![1.5, 0.5, -0.5];
        let layout = model.parameter_layout();
        let unpacked = model.unpack_theta(&theta).unwrap();

        assert_eq!(layout.slice("mu").unwrap(), 0..1);
        assert_eq!(layout.slice_of::<Mu>().unwrap(), 0..1);
        assert_eq!(layout.slice("sigma").unwrap(), 1..2);
        assert_eq!(layout.slice_of::<Sigma>().unwrap(), 1..2);
        assert_eq!(layout.slice("nu").unwrap(), 2..3);
        assert_eq!(layout.slice_of::<Nu>().unwrap(), 2..3);
        assert_eq!(unpacked.coefficients("mu").unwrap(), &[1.5]);
        assert_eq!(unpacked.coefficients_of::<Mu>().unwrap(), &[1.5]);
        assert_eq!(unpacked.block_of::<Mu>().unwrap().name, "mu");
        assert_eq!(unpacked.coefficients("sigma").unwrap(), &[0.5]);
        assert_eq!(unpacked.coefficients("nu").unwrap(), &[-0.5]);
    }

    #[test]
    fn diagnostics_report_train_nll_penalty_and_gradient_norm() {
        let y = vec![1.0, 2.0];
        let x = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x, RidgePenalty::new(0.5), 0);
        let model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y).unwrap();
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
        let mut model = Gamlss::try_new(FixedSigmaNormal, (mu,), &y)
            .unwrap()
            .with_global_penalties(DifferenceGlobalPenalty { lambda: 1.0 });
        let beta = vec![1.0, -1.0];
        let mut grad = vec![0.0; beta.len()];

        assert_relative_eq!(model.value(&beta).unwrap(), 5.0);

        model.gradient(&beta, &mut grad).unwrap();

        assert_relative_eq!(grad[0], 5.0);
        assert_relative_eq!(grad[1], -5.0);
    }

    #[test]
    fn block_objective_for_projects_mu_coefficients() {
        // Simple 2-param mock: identity link for both, NLL = 0.5 * sum of squares.
        #[derive(Debug, Clone, Copy)]
        struct TwoParamMock;

        impl Family for TwoParamMock {
            type Eta = (f64, f64);
            type Theta = (f64, f64);
            type ScoreEta = (f64, f64);

            fn theta(&self, eta: Self::Eta) -> Self::Theta {
                eta
            }

            fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
                let first = theta.0 - y;
                let second = theta.1 - 1.0;
                0.5 * (first * first + second * second)
            }

            fn nll_and_score_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::ScoreEta) {
                let score = (eta.0 - y, eta.1 - 1.0);
                (self.nll(y, eta), score)
            }
        }

        impl ParameterizedFamily<2> for TwoParamMock {
            type Params = (Mu, Sigma);
            type Links = (Identity, Identity);
        }

        let y = vec![1.0, 2.0, 3.0];
        let x_mu = DenseDesign::from_rows(&[[1.0, 0.5], [1.0, 1.5], [1.0, 2.5]]);
        let x_sigma = DenseDesign::intercept(y.len());
        let mu = ParameterBlock::<Mu, Identity, _, _>::linear(x_mu, NoPenalty, 0);
        let sigma = ParameterBlock::<Sigma, Identity, _, _>::linear(x_sigma, NoPenalty, 0);
        let (mu, sigma) = ParameterBlocks::new((mu, sigma));
        let mut model = Gamlss::try_new(TwoParamMock, (mu, sigma), &y).unwrap();

        let beta = vec![0.5, 0.2, 0.3];
        let nparams = model.nparams();

        // Scope the block objective to release the mutable borrow on model.
        let (mu_dim, mu_grad) = {
            let mut mu_block = model.block_objective_for::<Mu>(beta.clone()).unwrap();
            let dim = mu_block.dim();
            let mut block_grad = vec![0.0; dim];
            mu_block.gradient(&beta[..2], &mut block_grad).unwrap();
            (dim, block_grad)
        };

        assert_eq!(mu_dim, 2);

        let mut full_grad = vec![0.0; nparams];
        model.gradient(&beta, &mut full_grad).unwrap();

        assert_relative_eq!(mu_grad[0], full_grad[0]);
        assert_relative_eq!(mu_grad[1], full_grad[1]);
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
