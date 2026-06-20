use std::ops::Range;

use gamlss_core::{DenseDesign, DesignMatrix, Link, ModelError, PredictorBlock, Softplus};
use gamlss_spline::{ISplineBasis, MonotoneDirection};

/// Formula predictor composed from a dense linear design, optional row offsets,
/// and nonlinear monotone spline segments.
#[derive(Debug, Clone, PartialEq)]
pub struct FormulaPredictorBlock {
    dense: DenseDesign,
    offset: Option<Vec<f64>>,
    monotone: Vec<MonotoneSegment>,
    nparams: usize,
}

impl FormulaPredictorBlock {
    /// Creates a formula predictor block.
    #[must_use]
    pub(crate) fn new(
        dense: DenseDesign,
        offset: Option<Vec<f64>>,
        monotone: Vec<MonotoneSegment>,
        nparams: usize,
    ) -> Self {
        Self {
            dense,
            offset,
            monotone,
            nparams,
        }
    }

    /// Returns the dense linear part.
    #[must_use]
    pub fn dense(&self) -> &DenseDesign {
        &self.dense
    }

    fn monotone_eta(segment: &MonotoneSegment, row: usize, beta: &[f64]) -> f64 {
        debug_assert_eq!(beta.len(), segment.range.len());

        let sign = monotone_sign(segment.direction);
        beta[0]
            + segment
                .basis
                .evaluate(segment.values[row])
                .iter()
                .zip(&beta[1..])
                .map(|(basis, beta)| sign * Softplus::inverse(*beta) * basis)
                .sum::<f64>()
    }

    fn add_monotone_gradient(
        segment: &MonotoneSegment,
        scores: &[f64],
        multiplier: Option<&[f64]>,
        beta: &[f64],
        grad: &mut [f64],
    ) {
        debug_assert_eq!(beta.len(), segment.range.len());
        debug_assert_eq!(grad.len(), segment.range.len());

        let sign = monotone_sign(segment.direction);
        for (row, score) in scores.iter().copied().enumerate() {
            let score = multiplier.map_or(score, |multiplier| score * multiplier[row]);
            grad[0] += score;
            for (index, basis) in segment
                .basis
                .evaluate(segment.values[row])
                .iter()
                .enumerate()
            {
                grad[index + 1] +=
                    score * sign * basis * Softplus::derivative_inverse(beta[index + 1]);
            }
        }
    }
}

impl PredictorBlock for FormulaPredictorBlock {
    fn nrows(&self) -> usize {
        self.dense.nrows()
    }

    fn nparams(&self) -> usize {
        self.nparams
    }

    fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
        debug_assert_eq!(beta.len(), self.nparams);

        let dense_ncols = self.dense.ncols();
        let mut eta = self.dense.dot_row(row, &beta[..dense_ncols]);
        if let Some(offset) = &self.offset {
            eta += offset[row];
        }
        for segment in &self.monotone {
            eta += Self::monotone_eta(segment, row, &beta[segment.range.clone()]);
        }
        eta
    }

    fn add_gradient(&self, scores: &[f64], beta: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(scores.len(), self.nrows());
        debug_assert_eq!(beta.len(), self.nparams);
        debug_assert_eq!(grad.len(), self.nparams);

        let dense_ncols = self.dense.ncols();
        self.dense.add_t_mul_vec(scores, &mut grad[..dense_ncols]);
        for segment in &self.monotone {
            Self::add_monotone_gradient(
                segment,
                scores,
                None,
                &beta[segment.range.clone()],
                &mut grad[segment.range.clone()],
            );
        }
    }

    fn add_weighted_gradient(
        &self,
        scores: &[f64],
        multiplier: &[f64],
        beta: &[f64],
        grad: &mut [f64],
    ) {
        debug_assert_eq!(scores.len(), self.nrows());
        debug_assert_eq!(multiplier.len(), self.nrows());
        debug_assert_eq!(beta.len(), self.nparams);
        debug_assert_eq!(grad.len(), self.nparams);

        let dense_ncols = self.dense.ncols();
        self.dense
            .add_weighted_t_mul_vec(scores, multiplier, &mut grad[..dense_ncols]);
        for segment in &self.monotone {
            Self::add_monotone_gradient(
                segment,
                scores,
                Some(multiplier),
                &beta[segment.range.clone()],
                &mut grad[segment.range.clone()],
            );
        }
    }

    fn validate(&self) -> Result<(), ModelError> {
        let nrows = self.nrows();
        let dense_ncols = self.dense.ncols();
        if dense_ncols > self.nparams {
            return Err(ModelError::InvalidParameter {
                parameter: "formula predictor",
                expected: "dense columns <= local parameter count",
            });
        }

        if let Some(offset) = &self.offset
            && offset.len() != nrows
        {
            return Err(ModelError::DesignRowMismatch {
                parameter: "formula offset",
                expected_rows: nrows,
                actual_rows: offset.len(),
            });
        }

        for segment in &self.monotone {
            if segment.range.end > self.nparams {
                return Err(ModelError::BlockRangeOverflow {
                    parameter: "formula monotone",
                    offset: segment.range.start,
                    len: segment.range.len(),
                });
            }
            if segment.range.start < dense_ncols {
                return Err(ModelError::BlockOverlap {
                    first: "formula dense",
                    second: "formula monotone",
                });
            }
            if segment.values.len() != nrows {
                return Err(ModelError::DesignRowMismatch {
                    parameter: "formula monotone",
                    expected_rows: nrows,
                    actual_rows: segment.values.len(),
                });
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MonotoneSegment {
    pub(crate) range: Range<usize>,
    pub(crate) values: Vec<f64>,
    pub(crate) basis: ISplineBasis,
    pub(crate) direction: MonotoneDirection,
}

fn monotone_sign(direction: MonotoneDirection) -> f64 {
    match direction {
        MonotoneDirection::Increasing => 1.0,
        MonotoneDirection::Decreasing => -1.0,
    }
}
