use std::ops::Range;

use gamlss_core::{ModelError, Penalty, SegmentPenalty};
use gamlss_spline::{PreparedCyclicDifferencePenalty, PreparedDifferencePenalty};

use crate::FittedTerm;

#[derive(Debug, Clone, PartialEq)]
enum SegmentPenaltyKind {
    Difference(PreparedDifferencePenalty),
    Cyclic(PreparedCyclicDifferencePenalty),
}

impl Penalty for SegmentPenaltyKind {
    fn value(&self, beta: &[f64]) -> f64 {
        match self {
            Self::Difference(penalty) => penalty.value(beta),
            Self::Cyclic(penalty) => penalty.value(beta),
        }
    }

    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        match self {
            Self::Difference(penalty) => penalty.add_gradient(beta, grad),
            Self::Cyclic(penalty) => penalty.add_gradient(beta, grad),
        }
    }

    fn validate_dim(&self, dim: usize) -> Result<(), ModelError> {
        match self {
            Self::Difference(penalty) => penalty.validate_dim(dim),
            Self::Cyclic(penalty) => penalty.validate_dim(dim),
        }
    }
}

/// Formula-local segment penalty representation.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FormulaPenalty {
    spline_segments: Vec<SegmentPenalty<SegmentPenaltyKind>>,
}

impl FormulaPenalty {
    pub(crate) fn add_spline(
        &mut self,
        range: Range<usize>,
        lambda: f64,
        order: usize,
    ) -> Result<(), ModelError> {
        let penalty = PreparedDifferencePenalty::try_new(lambda, order)?;
        self.push_difference_unchecked(range, penalty);
        Ok(())
    }

    pub(crate) fn add_cyclic_spline(
        &mut self,
        range: Range<usize>,
        lambda: f64,
        order: usize,
    ) -> Result<(), ModelError> {
        let penalty = PreparedCyclicDifferencePenalty::try_new(lambda, order)?;
        self.push_cyclic_unchecked(range, penalty);
        Ok(())
    }

    fn push_difference_unchecked(
        &mut self,
        range: Range<usize>,
        penalty: PreparedDifferencePenalty,
    ) {
        self.spline_segments.push(SegmentPenalty::new(
            range,
            SegmentPenaltyKind::Difference(penalty),
        ));
    }

    fn push_cyclic_unchecked(
        &mut self,
        range: Range<usize>,
        penalty: PreparedCyclicDifferencePenalty,
    ) {
        self.spline_segments.push(SegmentPenalty::new(
            range,
            SegmentPenaltyKind::Cyclic(penalty),
        ));
    }
}

impl Penalty for FormulaPenalty {
    fn value(&self, beta: &[f64]) -> f64 {
        self.spline_segments
            .iter()
            .map(|segment| segment.value(beta))
            .sum()
    }

    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(beta.len(), grad.len());

        for segment in &self.spline_segments {
            segment.add_gradient(beta, grad);
        }
    }

    fn validate_dim(&self, dim: usize) -> Result<(), ModelError> {
        for segment in &self.spline_segments {
            segment.validate_dim(dim)?;
        }
        Ok(())
    }
}

pub fn prediction_penalty(terms: &[FittedTerm]) -> FormulaPenalty {
    let mut penalty = FormulaPenalty::default();
    for term in terms {
        match term {
            FittedTerm::PSpline {
                range,
                lambda,
                penalty_order,
                ..
            } => {
                debug_assert!(PreparedDifferencePenalty::try_new(*lambda, *penalty_order).is_ok());
                penalty.push_difference_unchecked(
                    range.clone(),
                    PreparedDifferencePenalty::new_unchecked(*lambda, *penalty_order),
                );
            }
            FittedTerm::CyclicPSpline {
                range,
                lambda,
                penalty_order,
                ..
            } => {
                debug_assert!(
                    PreparedCyclicDifferencePenalty::try_new(*lambda, *penalty_order).is_ok()
                );
                penalty.push_cyclic_unchecked(
                    range.clone(),
                    PreparedCyclicDifferencePenalty::new_unchecked(*lambda, *penalty_order),
                );
            }
            _ => {}
        }
    }
    penalty
}
