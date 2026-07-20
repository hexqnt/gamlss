use std::ops::Range;

use gamlss_core::{DenseDesign, DesignMatrix, ModelError, PredictorBlock, RowMultiplier};
use gamlss_spline::MonotoneISplineDesign;

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
    pub(crate) const fn new(
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
    pub const fn dense(&self) -> &DenseDesign {
        &self.dense
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
            eta += segment.design.eta_row(row, &beta[segment.range.clone()]);
        }
        eta
    }

    fn add_gradient_range(
        &self,
        rows: Range<usize>,
        scores: &[f64],
        beta: &[f64],
        grad: &mut [f64],
    ) {
        debug_assert!(rows.end <= self.nrows());
        debug_assert_eq!(scores.len(), rows.len());
        debug_assert_eq!(beta.len(), self.nparams);
        debug_assert_eq!(grad.len(), self.nparams);

        let dense_ncols = self.dense.ncols();
        self.dense
            .add_t_mul_vec_range(rows.clone(), scores, &mut grad[..dense_ncols]);
        for segment in &self.monotone {
            segment.design.add_gradient_range(
                rows.clone(),
                scores,
                &beta[segment.range.clone()],
                &mut grad[segment.range.clone()],
            );
        }
    }

    fn add_weighted_gradient_by_range<M>(
        &self,
        rows: Range<usize>,
        scores: &[f64],
        multiplier: &M,
        beta: &[f64],
        grad: &mut [f64],
    ) where
        M: RowMultiplier + ?Sized,
    {
        debug_assert!(rows.end <= self.nrows());
        debug_assert_eq!(scores.len(), rows.len());
        debug_assert_eq!(beta.len(), self.nparams);
        debug_assert_eq!(grad.len(), self.nparams);

        let dense_ncols = self.dense.ncols();
        self.dense.add_weighted_t_mul_vec_by_range(
            rows.clone(),
            scores,
            multiplier,
            &mut grad[..dense_ncols],
        );
        for segment in &self.monotone {
            segment.design.add_weighted_gradient_by_range(
                rows.clone(),
                scores,
                multiplier,
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
            if segment.design.nrows() != nrows {
                return Err(ModelError::DesignRowMismatch {
                    parameter: "formula monotone",
                    expected_rows: nrows,
                    actual_rows: segment.design.nrows(),
                });
            }
            if segment.design.nparams() != segment.range.len() {
                return Err(ModelError::InvalidParameter {
                    parameter: "formula monotone",
                    expected: "coefficient range matching monotone design width",
                });
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MonotoneSegment {
    pub(crate) range: Range<usize>,
    pub(crate) design: MonotoneISplineDesign,
}
