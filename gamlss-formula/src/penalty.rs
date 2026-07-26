use std::ops::Range;

use gamlss_core::{MatrixPenalty, ModelError, Penalty, SegmentPenalty};
use gamlss_spline::{
    HelmertContrastPenalty, PreparedCyclicDifferencePenalty, PreparedDifferencePenalty,
    TensorProductPenalty,
};

use crate::{FittedTerm, TensorSmoothKind};

type FormulaTensorPenalty = TensorProductPenalty<TensorMarginPenalty, TensorMarginPenalty>;

#[derive(Debug, Clone, PartialEq)]
enum TensorMarginPenalty {
    Direct(PreparedDifferencePenalty),
    Contrast(HelmertContrastPenalty<PreparedDifferencePenalty>),
}

impl Penalty for TensorMarginPenalty {
    fn value(&self, beta: &[f64]) -> f64 {
        match self {
            Self::Direct(penalty) => penalty.value(beta),
            Self::Contrast(penalty) => penalty.value(beta),
        }
    }

    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        match self {
            Self::Direct(penalty) => penalty.add_gradient(beta, grad),
            Self::Contrast(penalty) => penalty.add_gradient(beta, grad),
        }
    }

    fn validate_dim(&self, dim: usize) -> Result<(), ModelError> {
        match self {
            Self::Direct(penalty) => penalty.validate_dim(dim),
            Self::Contrast(penalty) => penalty.validate_dim(dim),
        }
    }
}

impl MatrixPenalty for TensorMarginPenalty {
    fn add_penalty_matrix(&self, dim: usize, gram: &mut [f64]) {
        match self {
            Self::Direct(penalty) => penalty.add_penalty_matrix(dim, gram),
            Self::Contrast(penalty) => penalty.add_penalty_matrix(dim, gram),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum SegmentPenaltyKind {
    Difference(PreparedDifferencePenalty),
    Cyclic(PreparedCyclicDifferencePenalty),
    Tensor(FormulaTensorPenalty),
}

impl Penalty for SegmentPenaltyKind {
    fn value(&self, beta: &[f64]) -> f64 {
        match self {
            Self::Difference(penalty) => penalty.value(beta),
            Self::Cyclic(penalty) => penalty.value(beta),
            Self::Tensor(penalty) => penalty.value(beta),
        }
    }

    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        match self {
            Self::Difference(penalty) => penalty.add_gradient(beta, grad),
            Self::Cyclic(penalty) => penalty.add_gradient(beta, grad),
            Self::Tensor(penalty) => penalty.add_gradient(beta, grad),
        }
    }

    fn validate_dim(&self, dim: usize) -> Result<(), ModelError> {
        match self {
            Self::Difference(penalty) => penalty.validate_dim(dim),
            Self::Cyclic(penalty) => penalty.validate_dim(dim),
            Self::Tensor(penalty) => penalty.validate_dim(dim),
        }
    }
}

/// Formula-local segment penalty representation.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FormulaPenalty {
    segments: Vec<SegmentPenalty<SegmentPenaltyKind>>,
}

impl FormulaPenalty {
    pub(crate) fn add_spline(
        &mut self,
        range: Range<usize>,
        lambda: f64,
        order: usize,
    ) -> Result<(), ModelError> {
        let penalty = PreparedDifferencePenalty::try_new(lambda, order)?;
        self.segments.push(SegmentPenalty::new(
            range,
            SegmentPenaltyKind::Difference(penalty),
        ));
        Ok(())
    }

    pub(crate) fn add_cyclic_spline(
        &mut self,
        range: Range<usize>,
        lambda: f64,
        order: usize,
    ) -> Result<(), ModelError> {
        let penalty = PreparedCyclicDifferencePenalty::try_new(lambda, order)?;
        self.segments.push(SegmentPenalty::new(
            range,
            SegmentPenaltyKind::Cyclic(penalty),
        ));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add_tensor_spline(
        &mut self,
        range: Range<usize>,
        left_source_dim: usize,
        right_source_dim: usize,
        kind: TensorSmoothKind,
        left_lambda: f64,
        right_lambda: f64,
        left_order: usize,
        right_order: usize,
    ) -> Result<(), ModelError> {
        let penalty = prepare_tensor_penalty(
            left_source_dim,
            right_source_dim,
            kind,
            left_lambda,
            right_lambda,
            left_order,
            right_order,
        )?;
        self.segments.push(SegmentPenalty::new(
            range,
            SegmentPenaltyKind::Tensor(penalty),
        ));
        Ok(())
    }
}

impl Penalty for FormulaPenalty {
    fn value(&self, beta: &[f64]) -> f64 {
        self.segments
            .iter()
            .map(|segment| segment.value(beta))
            .sum()
    }

    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(beta.len(), grad.len());

        for segment in &self.segments {
            segment.add_gradient(beta, grad);
        }
    }

    fn validate_dim(&self, dim: usize) -> Result<(), ModelError> {
        for segment in &self.segments {
            segment.validate_dim(dim)?;
        }
        Ok(())
    }
}

pub fn prediction_penalty(terms: &[FittedTerm]) -> Result<FormulaPenalty, ModelError> {
    let mut penalty = FormulaPenalty::default();
    for term in terms {
        match term {
            FittedTerm::PSpline {
                range,
                lambda,
                penalty_order,
                ..
            } => penalty.add_spline(range.clone(), *lambda, *penalty_order)?,
            FittedTerm::CyclicPSpline {
                range,
                lambda,
                penalty_order,
                ..
            } => penalty.add_cyclic_spline(range.clone(), *lambda, *penalty_order)?,
            FittedTerm::TensorPSpline {
                range,
                left_basis,
                right_basis,
                kind,
                left_lambda,
                right_lambda,
                left_penalty_order,
                right_penalty_order,
                ..
            } => {
                penalty.add_tensor_spline(
                    range.clone(),
                    left_basis.n_basis(),
                    right_basis.n_basis(),
                    *kind,
                    *left_lambda,
                    *right_lambda,
                    *left_penalty_order,
                    *right_penalty_order,
                )?;
            }
            _ => {}
        }
    }
    Ok(penalty)
}

#[allow(clippy::too_many_arguments)]
fn prepare_tensor_penalty(
    left_source_dim: usize,
    right_source_dim: usize,
    kind: TensorSmoothKind,
    left_lambda: f64,
    right_lambda: f64,
    left_order: usize,
    right_order: usize,
) -> Result<FormulaTensorPenalty, ModelError> {
    let left = PreparedDifferencePenalty::try_new(left_lambda, left_order)?;
    let right = PreparedDifferencePenalty::try_new(right_lambda, right_order)?;
    let (left, right, left_dim, right_dim) = match kind {
        TensorSmoothKind::Full => (
            TensorMarginPenalty::Direct(left),
            TensorMarginPenalty::Direct(right),
            left_source_dim,
            right_source_dim,
        ),
        TensorSmoothKind::Interaction => {
            let left = HelmertContrastPenalty::try_new(left_source_dim, left)?;
            let right = HelmertContrastPenalty::try_new(right_source_dim, right)?;
            let left_dim = left.target_dim();
            let right_dim = right.target_dim();
            (
                TensorMarginPenalty::Contrast(left),
                TensorMarginPenalty::Contrast(right),
                left_dim,
                right_dim,
            )
        }
    };
    TensorProductPenalty::try_new(left_dim, right_dim, left, right)
}
