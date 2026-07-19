use std::ops::Range;

use crate::{DynamicLayoutKey, DynamicallyCompilableFamily, ModelError, Penalty, PredictorBlock};

use super::{
    GamlssBlocks, GradientWorkspace, ObservationView, ParameterDescriptor, ParameterLayout,
    ParameterPath, ParameterSlice, validate_block_rows,
};

#[derive(Debug, Clone, PartialEq)]
struct DynamicCoordinate<X, Pen> {
    role: &'static str,
    path: ParameterPath,
    predictor: X,
    penalty: Pen,
    range: Range<usize>,
}

impl<X, Pen> DynamicCoordinate<X, Pen> {
    fn descriptor(&self) -> ParameterDescriptor {
        ParameterDescriptor::new(self.role, self.path.clone(), self.range.clone())
    }
}

/// Homogeneous predictor storage for a runtime-dimensional family codec.
///
/// Each scalar distribution coordinate may own an arbitrary number of
/// coefficients through `X`. The family instance supplies coordinate roles and
/// paths, while this block collection owns predictors, penalties, and canonical
/// non-overlapping coefficient ranges.
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicParameterBlocks<X, Pen> {
    layout_key: DynamicLayoutKey,
    coordinates: Vec<DynamicCoordinate<X, Pen>>,
}

impl<X, Pen> DynamicParameterBlocks<X, Pen>
where
    X: PredictorBlock,
    Pen: Penalty,
{
    /// Builds canonical runtime-dimensional blocks for `family`.
    ///
    /// # Errors
    ///
    /// Returns an error when the number of predictor/penalty pairs differs
    /// from the family's runtime coordinate count or coefficient ranges
    /// overflow `usize`.
    pub fn try_new<F>(family: &F, blocks: Vec<(X, Pen)>) -> Result<Self, ModelError>
    where
        F: DynamicallyCompilableFamily,
    {
        family.validate_dynamic_compiled()?;
        let coordinate_count = family.dynamic_parameter_count();
        if blocks.len() != coordinate_count {
            return Err(ModelError::InvalidParameter {
                parameter: "dynamic predictor blocks",
                expected: "one block per runtime family coordinate",
            });
        }
        let layout_key = family.dynamic_layout_key();

        let mut offset = 0usize;
        let mut coordinates = Vec::with_capacity(blocks.len());
        for (index, (predictor, penalty)) in blocks.into_iter().enumerate() {
            let end =
                offset
                    .checked_add(predictor.nparams())
                    .ok_or(ModelError::ArithmeticOverflow {
                        context: "dynamic parameter coefficient layout",
                    })?;
            let (role, path) = family.dynamic_parameter_coordinate(index);
            coordinates.push(DynamicCoordinate {
                role,
                path,
                predictor,
                penalty,
                range: offset..end,
            });
            offset = end;
        }
        Ok(Self {
            layout_key,
            coordinates,
        })
    }

    /// Exact runtime family topology captured when these blocks were built.
    #[must_use]
    #[inline]
    pub const fn layout_key(&self) -> &DynamicLayoutKey {
        &self.layout_key
    }

    /// Number of scalar runtime family coordinates.
    #[must_use]
    pub const fn coordinate_count(&self) -> usize {
        self.coordinates.len()
    }

    fn coefficient_count(&self) -> usize {
        self.coordinates
            .last()
            .map_or(0, |coordinate| coordinate.range.end)
    }

    fn add_local_penalty_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        for coordinate in &self.coordinates {
            coordinate.penalty.add_gradient(
                &beta[coordinate.range.clone()],
                &mut grad[coordinate.range.clone()],
            );
        }
    }

    fn local_penalty_value(&self, beta: &[f64]) -> f64 {
        self.coordinates
            .iter()
            .map(|coordinate| coordinate.penalty.value(&beta[coordinate.range.clone()]))
            .sum()
    }

    fn fill_values(&self, beta: &[f64], row: usize, values: &mut [f64]) {
        debug_assert_eq!(values.len(), self.coordinates.len());
        for (value, coordinate) in values.iter_mut().zip(&self.coordinates) {
            *value = coordinate
                .predictor
                .eta_row(row, &beta[coordinate.range.clone()]);
        }
    }

    fn initial_beta(&self, values: &[f64]) -> Vec<f64> {
        let mut beta = vec![0.0; self.coefficient_count()];
        for (coordinate, value) in self.coordinates.iter().zip(values.iter().copied()) {
            if value.is_finite() {
                coordinate
                    .predictor
                    .set_constant_start(value, &mut beta[coordinate.range.clone()]);
            }
        }
        beta
    }
}

impl<F, X, Pen> GamlssBlocks<F> for DynamicParameterBlocks<X, Pen>
where
    F: DynamicallyCompilableFamily,
    X: PredictorBlock,
    Pen: Penalty,
{
    fn nrows(&self) -> usize {
        self.coordinates
            .first()
            .map_or(0, |coordinate| coordinate.predictor.nrows())
    }

    fn len(&self) -> usize {
        self.coefficient_count()
    }

    fn try_len(&self) -> Result<usize, ModelError> {
        Ok(self.coefficient_count())
    }

    fn validate(&self, nobs: usize) -> Result<(), ModelError> {
        for coordinate in &self.coordinates {
            coordinate.predictor.validate()?;
            validate_block_rows(coordinate.role, coordinate.predictor.nrows(), nobs)?;
            coordinate.penalty.validate_dim(coordinate.range.len())?;
        }
        Ok(())
    }

    fn validate_for(&self, family: &F, nobs: usize) -> Result<(), ModelError> {
        family.validate_dynamic_compiled()?;
        if self.coordinates.len() != family.dynamic_parameter_count() {
            return Err(ModelError::InvalidParameter {
                parameter: "dynamic predictor blocks",
                expected: "one block per runtime family coordinate",
            });
        }
        let expected_key = family.dynamic_layout_key();
        if self.layout_key != expected_key {
            return Err(ModelError::DynamicLayoutMismatch {
                expected: expected_key,
                got: self.layout_key.clone(),
            });
        }
        for (index, coordinate) in self.coordinates.iter().enumerate() {
            let (role, path) = family.dynamic_parameter_coordinate(index);
            if coordinate.role != role || coordinate.path != path {
                return Err(ModelError::DynamicCoordinateMismatch {
                    index,
                    expected: ParameterDescriptor::new(role, path, coordinate.range.clone()),
                    got: coordinate.descriptor(),
                });
            }
        }
        <Self as GamlssBlocks<F>>::validate(self, nobs)
    }

    fn train_nll<'obs, Obs>(&self, family: &F, obs: &'obs Obs, beta: &[f64]) -> f64
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        let mut workspace = family.workspace();
        let mut values = vec![0.0; self.coordinates.len()];
        let mut loss = 0.0;
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            if weight == 0.0 {
                continue;
            }
            self.fill_values(beta, row, &mut values);
            loss = weight.mul_add(
                family.nll_eta_flat(obs.observation_at(row), &values, &mut workspace),
                loss,
            );
        }
        loss
    }

    fn train_nll_into_workspace<'obs, Obs>(
        &self,
        family: &F,
        obs: &'obs Obs,
        beta: &[f64],
        family_workspace: &mut F::Workspace,
        workspace: &mut GradientWorkspace,
    ) -> f64
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        let values = workspace.dynamic_values_mut(self.coordinates.len());
        let mut loss = 0.0;
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            if weight == 0.0 {
                continue;
            }
            self.fill_values(beta, row, values);
            loss = weight.mul_add(
                family.nll_eta_flat(obs.observation_at(row), values, family_workspace),
                loss,
            );
        }
        loss
    }

    fn pointwise_nll_into_workspace<'obs, Obs>(
        &self,
        family: &F,
        obs: &'obs Obs,
        beta: &[f64],
        weighted: bool,
        out: &mut [f64],
        workspace: (&mut F::Workspace, &mut GradientWorkspace),
    ) where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        let (family_workspace, workspace) = workspace;
        let values = workspace.dynamic_values_mut(self.coordinates.len());
        for (row, value) in out.iter_mut().enumerate() {
            let weight = obs.weight_at(row);
            if weighted && weight == 0.0 {
                *value = 0.0;
                continue;
            }
            self.fill_values(beta, row, values);
            let nll = family.nll_eta_flat(obs.observation_at(row), values, family_workspace);
            *value = if weighted { weight * nll } else { nll };
        }
    }

    fn eta_row(&self, family: &F, beta: &[f64], row: usize) -> F::Eta {
        let mut values = vec![0.0; self.coordinates.len()];
        self.fill_values(beta, row, &mut values);
        family.eta_from_flat(&values)
    }

    fn penalty_value(&self, beta: &[f64]) -> f64 {
        self.local_penalty_value(beta)
    }

    fn initial_parameters<'obs, Obs>(&self, family: &F, obs: &'obs Obs) -> Vec<f64>
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        let values = family.initial_flat(obs);
        self.initial_beta(&values)
    }

    fn try_initial_parameters<'obs, Obs>(
        &self,
        family: &F,
        obs: &'obs Obs,
    ) -> Result<Vec<f64>, ModelError>
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        let values = family.initial_flat(obs);
        if values.len() != self.coordinates.len() {
            return Err(ModelError::DynamicInitialValueCount {
                expected: self.coordinates.len(),
                actual: values.len(),
            });
        }
        Ok(self.initial_beta(&values))
    }

    fn add_penalty_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        self.add_local_penalty_gradient(beta, grad);
    }

    fn gradient_workspace(&self, nobs: usize) -> GradientWorkspace {
        let mut workspace = GradientWorkspace::new();
        let _ = workspace.prepare_score_tile(self.coordinates.len(), nobs);
        let _ = workspace.dynamic_buffers_mut(self.coordinates.len());
        workspace
    }

    fn value_gradient_into_workspace<'obs, Obs>(
        &self,
        family: &F,
        obs: &'obs Obs,
        beta: &[f64],
        grad: &mut [f64],
        family_workspace: &mut F::Workspace,
        workspace: &mut GradientWorkspace,
    ) -> f64
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        let tile_rows = workspace.prepare_score_tile(self.coordinates.len(), obs.len());
        let mut loss = 0.0;

        let mut tile_start = 0;
        while tile_start < obs.len() {
            let tile_end = tile_start.saturating_add(tile_rows).min(obs.len());
            let rows = tile_start..tile_end;
            workspace.set_score_tile_len(rows.len());

            for (tile_row, row) in rows.clone().enumerate() {
                let weight = obs.weight_at(row);
                if weight == 0.0 {
                    workspace.fill_score_row(tile_row, 0.0);
                    continue;
                }
                {
                    let (values, scores) = workspace.dynamic_buffers_mut(self.coordinates.len());
                    self.fill_values(beta, row, values);
                    loss = weight.mul_add(
                        family.nll_and_gradient_eta_flat(
                            obs.observation_at(row),
                            values,
                            scores,
                            family_workspace,
                        ),
                        loss,
                    );
                }
                workspace.store_weighted_dynamic_scores(tile_row, weight);
            }

            for (index, coordinate) in self.coordinates.iter().enumerate() {
                coordinate.predictor.add_gradient_range(
                    rows.clone(),
                    workspace.scores(index),
                    &beta[coordinate.range.clone()],
                    &mut grad[coordinate.range.clone()],
                );
            }
            tile_start = tile_end;
        }
        self.add_local_penalty_gradient(beta, grad);
        loss + self.local_penalty_value(beta)
    }

    fn block_ranges(&self) -> Vec<Range<usize>> {
        let mut ranges = Vec::with_capacity(self.coordinates.len());
        <Self as GamlssBlocks<F>>::visit_block_ranges(self, |_, range| ranges.push(range));
        ranges
    }

    fn parameter_layout(&self) -> ParameterLayout {
        let mut slices = Vec::with_capacity(self.coordinates.len());
        <Self as GamlssBlocks<F>>::visit_parameter_slices(self, |_, name, range| {
            slices.push(ParameterSlice { name, range });
        });
        ParameterLayout::new(slices)
    }

    fn parameter_descriptors(&self) -> Vec<ParameterDescriptor> {
        let mut descriptors = Vec::with_capacity(self.coordinates.len());
        <Self as GamlssBlocks<F>>::visit_parameter_descriptors(self, |_, descriptor| {
            descriptors.push(descriptor);
        });
        descriptors
    }

    fn visit_block_ranges<V>(&self, mut visit: V)
    where
        V: FnMut(usize, Range<usize>),
    {
        for (index, coordinate) in self.coordinates.iter().enumerate() {
            visit(index, coordinate.range.clone());
        }
    }

    fn visit_parameter_slices<V>(&self, mut visit: V)
    where
        V: FnMut(usize, &'static str, Range<usize>),
    {
        for (index, coordinate) in self.coordinates.iter().enumerate() {
            visit(index, coordinate.role, coordinate.range.clone());
        }
    }

    fn visit_parameter_descriptors<V>(&self, mut visit: V)
    where
        V: FnMut(usize, ParameterDescriptor),
    {
        for (index, coordinate) in self.coordinates.iter().enumerate() {
            visit(index, coordinate.descriptor());
        }
    }

    fn dynamic_layout_key(&self) -> Option<&DynamicLayoutKey> {
        Some(&self.layout_key)
    }
}

impl<X, Pen> super::sealed::Sealed for DynamicParameterBlocks<X, Pen> {}
