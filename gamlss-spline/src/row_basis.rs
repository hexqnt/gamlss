/// Predictor blocks that can expose a sparse row basis without allocation.
///
/// Implementations call `f(index, weight)` once for each non-zero basis value
/// in the requested row. The coefficient order is the same order used by
/// `PredictorBlock::eta_row`.
pub trait SplineRowBasis {
    /// Number of observations.
    fn nrows(&self) -> usize;

    /// Number of coefficients consumed by this basis.
    fn nparams(&self) -> usize;

    /// Visits non-zero basis values for `row`.
    fn for_each_row_basis(&self, row: usize, f: impl FnMut(usize, f64));
}
