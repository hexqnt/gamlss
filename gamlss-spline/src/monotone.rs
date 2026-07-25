use std::ops::Range;

use gamlss_core::{Link, PredictorBlock, RowMultiplier, Softplus};

use crate::ispline::{ISplineBasis, ISplineDesign};
use crate::{SplineError, SplineRowBasis};

/// Direction of a hard-monotone I-spline predictor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonotoneDirection {
    /// Nondecreasing predictor.
    Increasing,
    /// Nonincreasing predictor.
    Decreasing,
}

impl MonotoneDirection {
    const fn sign(self) -> f64 {
        match self {
            Self::Increasing => 1.0,
            Self::Decreasing => -1.0,
        }
    }
}

/// Hard-monotone I-spline predictor using softplus-constrained increments.
///
/// Let $M$ be [`MonotoneISplineDesign::n_increments`], let $I_i(x)$ be basis index $i$ from [`ISplineBasis`], and let the unconstrained coefficient slice be $\boldsymbol\beta=(\beta_0,\ldots,\beta_M)$. The implementation uses
///
/// $$
/// \begin{aligned}
/// a_i &= \operatorname{softplus}(\beta_{i+1})>0,\qquad 0\le i<M, \\\\
/// \eta(x) &= \beta_0+s\sum_{i=0}^{M-1}a_i I_i(x),
/// \qquad
/// s=\begin{cases}1,&\text{increasing},\\\\-1,&\text{decreasing}.\end{cases}
/// \end{aligned}
/// $$
///
/// Thus `beta[0]` is an unconstrained intercept and `beta[i + 1]` controls basis index $i$. Because $I_i^{\prime}(x)\ge0$, this construction enforces $s\\,\eta^{\prime}(x)=\sum_i a_i I_i^{\prime}(x)\ge0$ for every coefficient vector; no penalty or post-fit projection is needed.
///
/// Basis rows use the compact prepared representation from [`ISplineDesign`].
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, PartialEq)]
pub struct MonotoneISplineDesign {
    design: ISplineDesign,
    direction: MonotoneDirection,
}

impl MonotoneISplineDesign {
    /// Creates a monotone I-spline predictor.
    pub fn new(
        x: &[f64],
        basis: ISplineBasis,
        direction: MonotoneDirection,
    ) -> Result<Self, SplineError> {
        Ok(Self {
            design: ISplineDesign::from_owned_basis(x, basis)?,
            direction,
        })
    }

    /// Returns the basis metadata.
    #[must_use]
    pub const fn basis(&self) -> &ISplineBasis {
        self.design.basis()
    }

    /// Input coordinates.
    #[must_use]
    pub fn x(&self) -> &[f64] {
        self.design.x()
    }

    /// Number of positive increments.
    #[must_use]
    pub const fn n_increments(&self) -> usize {
        self.design.n_basis()
    }

    /// Monotonicity direction.
    #[must_use]
    pub const fn direction(&self) -> MonotoneDirection {
        self.direction
    }

    /// Predictor derivative with respect to `x`.
    #[must_use]
    #[inline]
    pub fn eta_derivative_row(&self, row: usize, beta: &[f64]) -> f64 {
        debug_assert!(row < self.design.x().len());
        debug_assert_eq!(beta.len(), self.nparams());

        let sign = self.direction.sign();
        let beta_tail = &beta[1..];
        let mut value = 0.0;
        self.design
            .basis()
            .for_each_derivative_basis(self.design.x()[row], |index, basis| {
                value = (sign * Softplus::inverse(beta_tail[index])).mul_add(basis, value);
            });
        value
    }

    #[allow(clippy::suboptimal_flops)]
    #[inline]
    fn add_row_gradient(&self, row: usize, score: f64, beta: &[f64], grad: &mut [f64]) {
        let sign = self.direction.sign();
        grad[0] += score;
        self.design.for_each_row_basis(row, |index, basis| {
            let scale = score * sign * Softplus::derivative_inverse(beta[index + 1]);
            grad[index + 1] = scale.mul_add(basis, grad[index + 1]);
        });
    }
}

impl PredictorBlock for MonotoneISplineDesign {
    #[inline]
    fn nrows(&self) -> usize {
        self.design.x().len()
    }

    #[inline]
    fn nparams(&self) -> usize {
        1 + self.design.n_basis()
    }

    #[inline]
    fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
        debug_assert!(row < self.design.x().len());
        debug_assert_eq!(beta.len(), self.nparams());

        let sign = self.direction.sign();
        let beta_tail = &beta[1..];
        let mut eta = beta[0];
        self.design.for_each_row_basis(row, |index, basis| {
            eta = (sign * Softplus::inverse(beta_tail[index])).mul_add(basis, eta);
        });
        eta
    }

    #[inline]
    fn add_gradient_range(
        &self,
        rows: Range<usize>,
        scores: &[f64],
        beta: &[f64],
        grad: &mut [f64],
    ) {
        debug_assert!(rows.end <= self.design.x().len());
        debug_assert_eq!(scores.len(), rows.len());
        debug_assert_eq!(beta.len(), self.nparams());
        debug_assert_eq!(grad.len(), self.nparams());

        for (offset, score) in scores.iter().copied().enumerate() {
            if score == 0.0 {
                continue;
            }
            let row = rows.start + offset;
            self.add_row_gradient(row, score, beta, grad);
        }
    }

    #[inline]
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
        debug_assert!(rows.end <= self.design.x().len());
        debug_assert_eq!(scores.len(), rows.len());
        debug_assert_eq!(beta.len(), self.nparams());
        debug_assert_eq!(grad.len(), self.nparams());

        for (offset, score) in scores.iter().copied().enumerate() {
            if score == 0.0 {
                continue;
            }
            let row = rows.start + offset;
            let scaled_score = score * multiplier.multiplier_at(row);
            if scaled_score == 0.0 {
                continue;
            }
            self.add_row_gradient(row, scaled_score, beta, grad);
        }
    }
}
