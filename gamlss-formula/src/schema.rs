use gamlss_core::ParameterLayout;

use crate::{Col, FittedTerm};

/// Metadata for all terms belonging to one distribution parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterTerms {
    /// Distribution parameter name, e.g. `"mu"` or `"sigma"`.
    pub parameter: &'static str,
    /// Fitted terms in expression order.
    pub terms: Vec<FittedTerm>,
}

/// Schema for a built formula model.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSchema {
    /// Response column.
    pub response: ResponseSchema,
    /// Per-parameter term metadata.
    pub parameters: Vec<ParameterTerms>,
}

/// Response metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct ResponseSchema {
    /// Source response column.
    pub col: Col<f64>,
    /// Optional observation weights column.
    pub weights: Option<Col<f64>>,
    /// Number of training observations.
    pub nrows: usize,
}

/// Compiled formula artifact.
#[derive(Debug, Clone, PartialEq)]
pub struct BuiltModel<M> {
    model: M,
    schema: ModelSchema,
    layout: ParameterLayout,
}

/// Reusable prediction design compiled from fitted term metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionDesign<Blocks> {
    blocks: Blocks,
}

impl<Blocks> PredictionDesign<Blocks> {
    pub(crate) fn new(blocks: Blocks) -> Self {
        Self { blocks }
    }

    /// Returns typed prediction parameter blocks.
    #[must_use]
    pub fn blocks(&self) -> &Blocks {
        &self.blocks
    }

    /// Consumes the design and returns typed prediction parameter blocks.
    #[must_use]
    pub fn into_blocks(self) -> Blocks {
        self.blocks
    }
}

impl<M> BuiltModel<M> {
    pub(crate) fn new(model: M, schema: ModelSchema, layout: ParameterLayout) -> Self {
        Self {
            model,
            schema,
            layout,
        }
    }

    /// Returns the compiled core model.
    #[must_use]
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Returns the compiled core model mutably.
    #[must_use]
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Consumes the artifact and returns the compiled core model.
    #[must_use]
    pub fn into_model(self) -> M {
        self.model
    }

    /// Returns formula schema metadata.
    #[must_use]
    pub fn schema(&self) -> &ModelSchema {
        &self.schema
    }

    /// Returns fitted terms grouped by distribution parameter.
    #[must_use]
    pub fn terms(&self) -> &[ParameterTerms] {
        &self.schema.parameters
    }

    /// Returns fitted terms for one distribution parameter.
    #[must_use]
    pub fn terms_for(&self, parameter: &str) -> Option<&[FittedTerm]> {
        self.schema
            .parameters
            .iter()
            .find(|terms| terms.parameter == parameter)
            .map(|terms| terms.terms.as_slice())
    }

    /// Returns coefficient names in model layout order.
    #[must_use]
    pub fn coefficient_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        for parameter in &self.schema.parameters {
            for term in &parameter.terms {
                term.append_coefficient_names(&mut names);
            }
        }
        names
    }

    /// Returns core parameter layout.
    #[must_use]
    pub fn layout(&self) -> &ParameterLayout {
        &self.layout
    }
}

pub(crate) fn terms_for<'a>(schema: &'a ModelSchema, parameter: &'static str) -> &'a [FittedTerm] {
    schema
        .parameters
        .iter()
        .find(|terms| terms.parameter == parameter)
        .map(|terms| terms.terms.as_slice())
        .unwrap_or(&[])
}
