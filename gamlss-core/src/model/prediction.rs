use crate::{Family, ModelError};

use super::{
    Gamlss, GamlssBlocks, validate_len, validate_output_len, validate_prediction_blocks,
    validate_row,
};

/// Validated prediction view over compatible parameter blocks.
///
/// The view borrows the fitted family and prediction blocks after validating
/// that the prediction block layout matches the fitted model. Reusing it avoids
/// repeating prediction-block validation across batch inference calls.
#[derive(Debug, PartialEq, Eq)]
pub struct PredictionView<'a, F, PBlocks> {
    family: &'a F,
    blocks: &'a PBlocks,
    nrows: usize,
    nparams: usize,
}

impl<'a, F, PBlocks> PredictionView<'a, F, PBlocks>
where
    F: Family,
    PBlocks: GamlssBlocks<F>,
{
    pub(super) fn new<Blocks, Obs>(
        model: &'a Gamlss<F, Blocks, Obs>,
        blocks: &'a PBlocks,
    ) -> Result<Self, ModelError>
    where
        Blocks: GamlssBlocks<F>,
    {
        validate_prediction_blocks(&model.blocks, blocks)?;
        Ok(Self {
            family: &model.family,
            blocks,
            nrows: blocks.nrows(),
            nparams: model.blocks.len(),
        })
    }

    pub(super) const fn from_parts(
        family: &'a F,
        blocks: &'a PBlocks,
        nrows: usize,
        nparams: usize,
    ) -> Self {
        Self {
            family,
            blocks,
            nrows,
            nparams,
        }
    }

    /// Response distribution family.
    #[must_use]
    #[inline]
    pub const fn family(&self) -> &'a F {
        self.family
    }

    /// Typed prediction parameter blocks.
    #[must_use]
    #[inline]
    pub const fn blocks(&self) -> &'a PBlocks {
        self.blocks
    }

    /// Number of prediction rows.
    #[must_use]
    #[inline]
    pub const fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of coefficients expected in the flat parameter vector.
    #[must_use]
    #[inline]
    pub const fn nparams(&self) -> usize {
        self.nparams
    }

    /// Predicts link-scale distribution predictors for one prediction row.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `parameters` has the wrong length or `row` is
    /// out of bounds for the validated prediction blocks.
    pub fn predict_eta_row(&self, parameters: &[f64], row: usize) -> Result<F::Eta, ModelError> {
        validate_len("parameters", parameters.len(), self.nparams)?;
        validate_row(row, self.nrows)?;
        Ok(self.blocks.eta_row(self.family, parameters, row))
    }

    /// Predicts natural-scale distribution parameters for one prediction row.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `parameters` has the wrong length or `row` is
    /// out of bounds for the validated prediction blocks.
    pub fn predict_theta_row(
        &self,
        parameters: &[f64],
        row: usize,
    ) -> Result<F::Theta, ModelError> {
        let eta = self.predict_eta_row(parameters, row)?;
        let mut workspace = self.family.workspace();
        Ok(self.family.theta(&eta, &mut workspace))
    }

    /// Predicts link-scale distribution predictors for all prediction rows.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `parameters` has the wrong length.
    pub fn predict_eta(&self, parameters: &[f64]) -> Result<Vec<F::Eta>, ModelError> {
        validate_len("parameters", parameters.len(), self.nparams)?;
        Ok((0..self.nrows)
            .map(|row| self.blocks.eta_row(self.family, parameters, row))
            .collect())
    }

    /// Predicts natural-scale distribution parameters for all prediction rows.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `parameters` has the wrong length.
    pub fn predict_theta(&self, parameters: &[f64]) -> Result<Vec<F::Theta>, ModelError> {
        validate_len("parameters", parameters.len(), self.nparams)?;
        Ok(collect_theta(self.family, self.blocks, parameters, || {
            self.nrows
        }))
    }

    /// Predicts natural-scale distribution parameters into an existing slice.
    ///
    /// `out` must have one slot per prediction row.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `parameters` or `out` have the wrong length.
    pub fn predict_theta_into(
        &self,
        parameters: &[f64],
        out: &mut [F::Theta],
    ) -> Result<(), ModelError> {
        validate_len("parameters", parameters.len(), self.nparams)?;
        validate_output_len(self.nrows, out.len())?;
        let mut workspace = self.family.workspace();
        for (row, out) in out.iter_mut().enumerate() {
            let eta = self.blocks.eta_row(self.family, parameters, row);
            *out = self.family.theta(&eta, &mut workspace);
        }
        Ok(())
    }

    /// Streams natural-scale parameters for each prediction row.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `parameters` has the wrong length.
    pub fn for_each_theta(
        &self,
        parameters: &[f64],
        visit: impl FnMut(usize, F::Theta),
    ) -> Result<(), ModelError> {
        validate_len("parameters", parameters.len(), self.nparams)?;
        visit_theta(self.family, self.blocks, parameters, || self.nrows, visit);
        Ok(())
    }
}

impl<F, PBlocks> Clone for PredictionView<'_, F, PBlocks> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, PBlocks> Copy for PredictionView<'_, F, PBlocks> {}

// Keep row-count evaluation after workspace creation, matching the training API's
// observable callback order for custom families and observation views.
pub(super) fn collect_theta<F, Blocks>(
    family: &F,
    blocks: &Blocks,
    parameters: &[f64],
    nrows: impl FnOnce() -> usize,
) -> Vec<F::Theta>
where
    F: Family,
    Blocks: GamlssBlocks<F>,
{
    let mut workspace = family.workspace();
    (0..nrows())
        .map(|row| {
            let eta = blocks.eta_row(family, parameters, row);
            family.theta(&eta, &mut workspace)
        })
        .collect()
}

pub(super) fn visit_theta<F, Blocks>(
    family: &F,
    blocks: &Blocks,
    parameters: &[f64],
    nrows: impl FnOnce() -> usize,
    mut visit: impl FnMut(usize, F::Theta),
) where
    F: Family,
    Blocks: GamlssBlocks<F>,
{
    let mut workspace = family.workspace();
    for row in 0..nrows() {
        let eta = blocks.eta_row(family, parameters, row);
        visit(row, family.theta(&eta, &mut workspace));
    }
}
