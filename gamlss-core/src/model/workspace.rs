/// Reusable scratch buffers for GAMLSS gradient evaluation.
///
/// The workspace stores one per-observation gradient vector and one local
/// coefficient-gradient vector per parameter block. Reusing it avoids the
/// temporary `Vec` allocations that otherwise happen inside each gradient call.
/// The workspace is model-shape specific but can be reused across different
/// beta vectors for the same compiled model.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GradientWorkspace {
    row_gradients: Vec<Vec<f64>>,
    local_gradients: Vec<Vec<f64>>,
    penalty_gradient: Vec<f64>,
}

impl GradientWorkspace {
    /// Creates an empty workspace. Buffers are allocated lazily on first use.
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn prepare(&mut self, block_count: usize) {
        self.row_gradients.resize_with(block_count, Vec::new);
        self.local_gradients.resize_with(block_count, Vec::new);
    }

    pub(crate) fn prepare_row_gradient(&mut self, index: usize, len: usize) {
        let row_gradient = &mut self.row_gradients[index];
        row_gradient.resize(len, 0.0);
    }

    pub(crate) fn set_row_gradient(&mut self, index: usize, row: usize, value: f64) {
        debug_assert!(index < self.row_gradients.len());
        let row_gradient = &mut self.row_gradients[index];
        debug_assert!(row < row_gradient.len());
        row_gradient[row] = value;
    }

    pub(crate) fn local_gradient_mut(&mut self, index: usize, len: usize) -> &mut [f64] {
        let gradient = &mut self.local_gradients[index];
        gradient.resize(len, 0.0);
        gradient.fill(0.0);
        gradient
    }

    pub(crate) fn row_gradient_and_local_gradient_mut(
        &mut self,
        index: usize,
        local_gradient_len: usize,
    ) -> (&[f64], &mut [f64]) {
        let local_gradient = &mut self.local_gradients[index];
        local_gradient.resize(local_gradient_len, 0.0);
        local_gradient.fill(0.0);

        (
            self.row_gradients[index].as_slice(),
            local_gradient.as_mut_slice(),
        )
    }

    pub(crate) fn penalty_gradient_mut(&mut self, len: usize) -> &mut [f64] {
        self.penalty_gradient.resize(len, 0.0);
        self.penalty_gradient.fill(0.0);
        &mut self.penalty_gradient
    }
}
