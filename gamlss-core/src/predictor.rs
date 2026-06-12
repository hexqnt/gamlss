use crate::{DesignMatrix, ModelError};

/// Predictor block for one distribution parameter.
///
/// Implementations map a local coefficient slice to a scalar linear predictor
/// contribution for each observation and know how to propagate per-observation
/// scores back to that local coefficient slice.
pub trait PredictorBlock {
    /// Number of observations.
    fn nrows(&self) -> usize;
    /// Number of local coefficients consumed by this block.
    fn nparams(&self) -> usize;
    /// Predictor contribution for one row.
    fn eta_row(&self, row: usize, beta: &[f64]) -> f64;
    /// Adds the gradient contribution implied by `scores` into `grad`.
    fn add_gradient(&self, scores: &[f64], beta: &[f64], grad: &mut [f64]);

    /// Validates internal block consistency.
    fn validate(&self) -> Result<(), ModelError> {
        Ok(())
    }
}

/// Linear predictor block backed by a [`DesignMatrix`].
///
/// This is the explicit adapter from matrix-based predictors to the more
/// general [`PredictorBlock`] extension point.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LinearPredictorBlock<X> {
    /// Design matrix used by this predictor.
    pub x: X,
}

impl<X> LinearPredictorBlock<X> {
    /// Wraps a design matrix as a predictor block.
    pub fn new(x: X) -> Self {
        Self { x }
    }

    /// Returns the wrapped design matrix.
    pub fn into_inner(self) -> X {
        self.x
    }
}

impl<X> PredictorBlock for LinearPredictorBlock<X>
where
    X: DesignMatrix,
{
    fn nrows(&self) -> usize {
        self.x.nrows()
    }

    fn nparams(&self) -> usize {
        self.x.ncols()
    }

    fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
        self.x.dot_row(row, beta)
    }

    fn add_gradient(&self, scores: &[f64], _: &[f64], grad: &mut [f64]) {
        self.x.add_t_mul_vec(scores, grad);
    }
}

/// Sum of several predictor blocks sharing the same observations.
///
/// The local beta slice is split between terms in tuple order. This keeps
/// nonlinear or sparse user-defined terms composable without dynamic dispatch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SumBlock<Terms> {
    /// Predictor terms summed into one parameter predictor.
    pub terms: Terms,
}

impl<Terms> SumBlock<Terms> {
    /// Creates a summed predictor from tuple terms.
    pub fn new(terms: Terms) -> Self {
        Self { terms }
    }
}

macro_rules! impl_sum_block {
    (
        terms = ($($term:ident),+);
        vars = ($($var:ident),+);
        indices = ($($idx:tt),+);
        names = ($($name:literal),+)
    ) => {
        impl<$($term,)+> PredictorBlock for SumBlock<($($term,)+)>
        where
            $($term: PredictorBlock,)+
        {
            fn nrows(&self) -> usize {
                self.terms.0.nrows()
            }

            fn nparams(&self) -> usize {
                0 $(+ self.terms.$idx.nparams())+
            }

            fn eta_row(&self, row: usize, beta: &[f64]) -> f64 {
                let mut start = 0;
                let mut eta = 0.0;
                $(
                    let $var = &self.terms.$idx;
                    let end = start + $var.nparams();
                    eta += $var.eta_row(row, &beta[start..end]);
                    start = end;
                )+
                let _ = start;
                eta
            }

            fn add_gradient(&self, scores: &[f64], beta: &[f64], grad: &mut [f64]) {
                let mut start = 0;
                $(
                    let $var = &self.terms.$idx;
                    let end = start + $var.nparams();
                    $var.add_gradient(scores, &beta[start..end], &mut grad[start..end]);
                    start = end;
                )+
                let _ = start;
            }

            fn validate(&self) -> Result<(), ModelError> {
                let expected_rows = self.terms.0.nrows();
                $(
                    self.terms.$idx.validate()?;
                    if self.terms.$idx.nrows() != expected_rows {
                        return Err(ModelError::DesignRowMismatch {
                            parameter: $name,
                            expected_rows,
                            actual_rows: self.terms.$idx.nrows(),
                        });
                    }
                )+
                Ok(())
            }
        }
    };
}

impl_sum_block!(
    terms = (T1);
    vars = (term1);
    indices = (0);
    names = ("sum term")
);

impl_sum_block!(
    terms = (T1, T2);
    vars = (term1, term2);
    indices = (0, 1);
    names = ("sum first term", "sum second term")
);

impl_sum_block!(
    terms = (T1, T2, T3);
    vars = (term1, term2, term3);
    indices = (0, 1, 2);
    names = ("sum first term", "sum second term", "sum third term")
);

impl_sum_block!(
    terms = (T1, T2, T3, T4);
    vars = (term1, term2, term3, term4);
    indices = (0, 1, 2, 3);
    names = (
        "sum first term",
        "sum second term",
        "sum third term",
        "sum fourth term"
    )
);

impl_sum_block!(
    terms = (T1, T2, T3, T4, T5);
    vars = (term1, term2, term3, term4, term5);
    indices = (0, 1, 2, 3, 4);
    names = (
        "sum first term",
        "sum second term",
        "sum third term",
        "sum fourth term",
        "sum fifth term"
    )
);

impl_sum_block!(
    terms = (T1, T2, T3, T4, T5, T6);
    vars = (term1, term2, term3, term4, term5, term6);
    indices = (0, 1, 2, 3, 4, 5);
    names = (
        "sum first term",
        "sum second term",
        "sum third term",
        "sum fourth term",
        "sum fifth term",
        "sum sixth term"
    )
);

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{DenseDesign, PredictorBlock};

    use super::LinearPredictorBlock;

    #[test]
    fn linear_predictor_block_matches_design_matrix_operations() {
        let design = DenseDesign::from_rows(&[[1.0, 2.0], [3.0, 4.0]]);
        let block = LinearPredictorBlock::new(design);
        let beta = [10.0, 1.0];

        assert_relative_eq!(block.eta_row(1, &beta), 34.0);

        let mut grad = vec![0.0, 0.0];
        block.add_gradient(&[0.5, 2.0], &beta, &mut grad);

        assert_relative_eq!(grad[0], 6.5);
        assert_relative_eq!(grad[1], 9.0);
    }
}
