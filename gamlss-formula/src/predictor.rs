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

    fn monotone_eta(segment: &MonotoneSegment, row: usize, beta: &[f64]) -> f64 {
        debug_assert_eq!(beta.len(), segment.range.len());
        debug_assert!(row < segment.values.len());
        debug_assert!(!beta.is_empty());
        debug_assert_eq!(segment.basis.n_basis() + 1, beta.len());

        let sign = monotone_sign(segment.direction);
        let beta_tail = &beta[1..];
        let mut eta = beta[0];
        segment
            .basis
            .for_each_basis(segment.values[row], |index, basis| {
                eta = (sign * Softplus::inverse(beta_tail[index])).mul_add(basis, eta);
            });
        eta
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
        debug_assert!(!beta.is_empty());
        debug_assert_eq!(segment.basis.n_basis() + 1, beta.len());
        debug_assert_eq!(scores.len(), segment.values.len());

        let sign = monotone_sign(segment.direction);
        let (intercept_grad, grad_tail) = grad
            .split_first_mut()
            .expect("monotone segment gradient has an intercept coefficient");
        let beta_tail = &beta[1..];
        match multiplier {
            Some(multiplier) => {
                debug_assert_eq!(multiplier.len(), scores.len());
                for (row, (score, value)) in scores
                    .iter()
                    .copied()
                    .zip(segment.values.iter().copied())
                    .enumerate()
                {
                    if score == 0.0 {
                        continue;
                    }

                    let score = score * multiplier[row];
                    if score == 0.0 {
                        continue;
                    }

                    add_monotone_gradient_row(
                        &segment.basis,
                        value,
                        score,
                        sign,
                        beta_tail,
                        intercept_grad,
                        grad_tail,
                    );
                }
            }
            None => {
                for (score, value) in scores.iter().copied().zip(segment.values.iter().copied()) {
                    if score == 0.0 {
                        continue;
                    }

                    add_monotone_gradient_row(
                        &segment.basis,
                        value,
                        score,
                        sign,
                        beta_tail,
                        intercept_grad,
                        grad_tail,
                    );
                }
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
pub struct MonotoneSegment {
    pub(crate) range: Range<usize>,
    pub(crate) values: Vec<f64>,
    pub(crate) basis: ISplineBasis,
    pub(crate) direction: MonotoneDirection,
}

#[allow(clippy::suboptimal_flops)]
fn add_monotone_gradient_row(
    basis: &ISplineBasis,
    value: f64,
    score: f64,
    sign: f64,
    beta_tail: &[f64],
    intercept_grad: &mut f64,
    grad_tail: &mut [f64],
) {
    debug_assert_eq!(beta_tail.len(), grad_tail.len());

    *intercept_grad += score;
    basis.for_each_basis(value, |index, basis_value| {
        grad_tail[index] +=
            score * sign * basis_value * Softplus::derivative_inverse(beta_tail[index]);
    });
}

const fn monotone_sign(direction: MonotoneDirection) -> f64 {
    match direction {
        MonotoneDirection::Increasing => 1.0,
        MonotoneDirection::Decreasing => -1.0,
    }
}
