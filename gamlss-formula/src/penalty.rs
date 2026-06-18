use std::ops::Range;

use gamlss_core::{Penalty, SegmentPenalty};
use gamlss_spline::{CyclicDifferencePenalty, PreparedDifferencePenalty};

use crate::FittedTerm;

/// Formula-local segment penalty representation.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FormulaPenalty {
    spline_segments: Vec<SegmentPenalty<SegmentPenaltyKind>>,
}

#[derive(Debug, Clone, PartialEq)]
enum SegmentPenaltyKind {
    Difference(PreparedDifferencePenalty),
    Cyclic(CyclicDifferencePenalty),
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
}

impl FormulaPenalty {
    pub(crate) fn add_spline(&mut self, range: Range<usize>, lambda: f64, order: usize) {
        self.spline_segments.push(SegmentPenalty::new(
            range,
            SegmentPenaltyKind::Difference(PreparedDifferencePenalty::new(lambda, order)),
        ));
    }

    pub(crate) fn add_cyclic_spline(&mut self, range: Range<usize>, lambda: f64, order: usize) {
        self.spline_segments.push(SegmentPenalty::new(
            range,
            SegmentPenaltyKind::Cyclic(CyclicDifferencePenalty::new(lambda, order)),
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
}

pub(crate) fn prediction_penalty(terms: &[FittedTerm]) -> FormulaPenalty {
    let mut penalty = FormulaPenalty::default();
    for term in terms {
        match term {
            FittedTerm::PSpline {
                range,
                lambda,
                penalty_order,
                ..
            } => penalty.add_spline(range.clone(), *lambda, *penalty_order),
            FittedTerm::CyclicPSpline {
                range,
                lambda,
                penalty_order,
                ..
            } => penalty.add_cyclic_spline(range.clone(), *lambda, *penalty_order),
            _ => {}
        }
    }
    penalty
}
