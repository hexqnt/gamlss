use std::ops::Range;

use crate::{DynamicallyCompilableFamily, ModelError, Penalty, PredictorBlock};

use super::{
    GamlssBlocks, GradientWorkspace, ObservationView, ParameterDescriptor, ParameterLayout,
    ParameterPath, ParameterSlice, add_into, validate_block_rows,
};

#[derive(Debug, Clone, PartialEq)]
struct DynamicCoordinate<X, Pen> {
    role: &'static str,
    path: ParameterPath,
    predictor: X,
    penalty: Pen,
    range: Range<usize>,
}

/// Homogeneous predictor storage for a runtime-dimensional family codec.
///
/// Each scalar distribution coordinate may own an arbitrary number of
/// coefficients through `X`. The family instance supplies coordinate roles and
/// paths, while this block collection owns predictors, penalties, and canonical
/// non-overlapping coefficient ranges.
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicParameterBlocks<X, Pen> {
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
        if blocks.len() != family.dynamic_parameter_count() {
            return Err(ModelError::InvalidParameter {
                parameter: "dynamic predictor blocks",
                expected: "one block per runtime family coordinate",
            });
        }

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
        Ok(Self { coordinates })
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
}

impl<X, Pen> DynamicParameterBlocks<X, Pen>
where
    X: PredictorBlock,
    Pen: Penalty,
{
    fn fill_values(&self, beta: &[f64], row: usize, values: &mut [f64]) {
        for (value, coordinate) in values.iter_mut().zip(&self.coordinates) {
            *value = coordinate
                .predictor
                .eta_row(row, &beta[coordinate.range.clone()]);
        }
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

    fn eta_row(&self, beta: &[f64], row: usize) -> F::Eta {
        let mut values = vec![0.0; self.coordinates.len()];
        self.fill_values(beta, row, &mut values);
        F::eta_from_flat(&values)
    }

    fn penalty_value(&self, beta: &[f64]) -> f64 {
        self.local_penalty_value(beta)
    }

    fn initial_parameters<'obs, Obs>(&self, family: &F, obs: &'obs Obs) -> Vec<f64>
    where
        Obs: ObservationView<'obs, Observation = F::Observation<'obs>> + 'obs,
    {
        let values = family.initial_flat(obs);
        let mut beta = vec![0.0; self.coefficient_count()];
        for (coordinate, value) in self.coordinates.iter().zip(values) {
            if value.is_finite() {
                coordinate
                    .predictor
                    .set_constant_start(value, &mut beta[coordinate.range.clone()]);
            }
        }
        beta
    }

    fn add_penalty_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        self.add_local_penalty_gradient(beta, grad);
    }

    fn gradient_workspace(&self, nobs: usize) -> GradientWorkspace {
        let mut workspace = GradientWorkspace::new();
        workspace.prepare(self.coordinates.len());
        for (index, coordinate) in self.coordinates.iter().enumerate() {
            workspace.prepare_row_gradient(index, nobs);
            let _ = workspace.local_gradient_mut(index, coordinate.range.len());
        }
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
        workspace.prepare(self.coordinates.len());
        for (index, coordinate) in self.coordinates.iter().enumerate() {
            workspace.prepare_row_gradient(index, obs.len());
            let _ = workspace.local_gradient_mut(index, coordinate.range.len());
        }

        let mut loss = 0.0;
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            {
                let (values, scores) = workspace.dynamic_buffers_mut(self.coordinates.len());
                if weight == 0.0 {
                    scores.fill(0.0);
                } else {
                    for (value, coordinate) in values.iter_mut().zip(&self.coordinates) {
                        *value = coordinate
                            .predictor
                            .eta_row(row, &beta[coordinate.range.clone()]);
                    }
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
            }
            workspace.store_dynamic_scores(row, weight);
        }

        for (index, coordinate) in self.coordinates.iter().enumerate() {
            let (scores, local) =
                workspace.row_gradient_and_local_gradient_mut(index, coordinate.range.len());
            coordinate
                .predictor
                .add_gradient(scores, &beta[coordinate.range.clone()], local);
            add_into(&mut grad[coordinate.range.clone()], local);
        }
        self.add_local_penalty_gradient(beta, grad);
        loss + self.local_penalty_value(beta)
    }

    fn block_ranges(&self) -> Vec<Range<usize>> {
        self.coordinates
            .iter()
            .map(|coordinate| coordinate.range.clone())
            .collect()
    }

    fn parameter_layout(&self) -> ParameterLayout {
        ParameterLayout::new(
            self.coordinates
                .iter()
                .map(|coordinate| ParameterSlice {
                    name: coordinate.role,
                    range: coordinate.range.clone(),
                })
                .collect(),
        )
    }

    fn parameter_descriptors(&self) -> Vec<ParameterDescriptor> {
        self.coordinates
            .iter()
            .map(|coordinate| {
                ParameterDescriptor::new(
                    coordinate.role,
                    coordinate.path.clone(),
                    coordinate.range.clone(),
                )
            })
            .collect()
    }
}

impl<X, Pen> super::sealed::Sealed for DynamicParameterBlocks<X, Pen> {}
