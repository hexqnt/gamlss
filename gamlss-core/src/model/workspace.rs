use crate::{Family, ModelError};

const DEFAULT_SCORE_TILE_BYTES: usize = 4 * 1024 * 1024;
const DEFAULT_SCORE_TILE_VALUES: usize = DEFAULT_SCORE_TILE_BYTES / size_of::<f64>();

/// Reusable model scratch buffers for GAMLSS objective evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelWorkspace<F>
where
    F: Family,
{
    family: F::Workspace,
    gradient: GradientWorkspace,
}

impl<F> ModelWorkspace<F>
where
    F: Family,
{
    /// Creates reusable buffers for `family` and the model gradient path.
    #[must_use]
    pub fn new(family: &F, nobs: usize, gradient: impl FnOnce(usize) -> GradientWorkspace) -> Self {
        Self {
            family: family.workspace(),
            gradient: gradient(nobs),
        }
    }

    pub(super) fn with_gradient(family: &F, gradient: GradientWorkspace) -> Self {
        Self {
            family: family.workspace(),
            gradient,
        }
    }

    /// Family-specific likelihood workspace.
    #[inline]
    pub const fn family_mut(&mut self) -> &mut F::Workspace {
        &mut self.family
    }

    /// Gradient assembly workspace.
    #[inline]
    pub const fn gradient(&self) -> &GradientWorkspace {
        &self.gradient
    }

    /// Mutable gradient assembly workspace.
    #[inline]
    pub const fn gradient_mut(&mut self) -> &mut GradientWorkspace {
        &mut self.gradient
    }

    /// Mutable family and gradient workspaces.
    #[inline]
    pub const fn parts_mut(&mut self) -> (&mut F::Workspace, &mut GradientWorkspace) {
        (&mut self.family, &mut self.gradient)
    }
}

/// Reusable scratch buffers for GAMLSS gradient evaluation.
///
/// The workspace stores one flat, coordinate-major row-score tile. The
/// executor backpropagates each contiguous tile immediately, so its prepared
/// score length is independent of the total observation count. Reusing the
/// workspace avoids allocations across repeated objective calls.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GradientWorkspace {
    score_tile: Vec<f64>,
    score_coordinate_count: usize,
    score_tile_rows: usize,
    score_tile_len: usize,
    score_tile_row_limit: Option<usize>,
    penalty_gradient: Vec<f64>,
    dynamic_values: Vec<f64>,
    dynamic_scores: Vec<f64>,
}

impl GradientWorkspace {
    /// Creates an empty workspace. Buffers are allocated lazily on first use.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a workspace whose score tiles contain at most `tile_rows` rows.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `tile_rows == 0`.
    pub fn try_with_tile_rows(tile_rows: usize) -> Result<Self, ModelError> {
        if tile_rows == 0 {
            return Err(ModelError::InvalidParameter {
                parameter: "score tile rows",
                expected: "positive",
            });
        }
        Ok(Self {
            score_tile_row_limit: Some(tile_rows),
            ..Self::default()
        })
    }

    /// Prepares bounded score storage and returns the number of rows per tile.
    ///
    /// The automatic policy targets approximately four MiB and always keeps at
    /// least one row when `nobs > 0`. An explicit limit supplied by
    /// [`Self::try_with_tile_rows`] takes precedence.
    pub fn prepare_score_tile(&mut self, coordinate_count: usize, nobs: usize) -> usize {
        let divisor = coordinate_count.max(1);
        let automatic_rows = (DEFAULT_SCORE_TILE_VALUES / divisor).max(1);
        let requested_rows = self.score_tile_row_limit.unwrap_or(automatic_rows);
        let max_rows_without_overflow = usize::MAX / divisor;
        let tile_rows = requested_rows.min(max_rows_without_overflow).min(nobs);
        let score_value_count = coordinate_count * tile_rows;

        self.score_tile.resize(score_value_count, 0.0);
        self.score_coordinate_count = coordinate_count;
        self.score_tile_rows = tile_rows;
        self.score_tile_len = tile_rows;
        tile_rows
    }

    /// Sets the logical row count of the current, possibly shorter, score tile.
    pub fn set_score_tile_len(&mut self, tile_rows: usize) {
        debug_assert!(tile_rows <= self.score_tile_rows);
        self.score_tile_len = tile_rows;
    }

    /// Writes one score into a coordinate's current tile.
    pub fn set_score(&mut self, index: usize, tile_row: usize, value: f64) {
        debug_assert!(index < self.score_coordinate_count);
        debug_assert!(tile_row < self.score_tile_len);
        self.score_tile[index * self.score_tile_rows + tile_row] = value;
    }

    /// Fills one row across all score coordinates in the current tile.
    pub fn fill_score_row(&mut self, tile_row: usize, value: f64) {
        debug_assert!(tile_row < self.score_tile_len);
        for index in 0..self.score_coordinate_count {
            self.score_tile[index * self.score_tile_rows + tile_row] = value;
        }
    }

    /// Returns one coordinate's scores for the current tile.
    #[must_use]
    pub fn scores(&self, index: usize) -> &[f64] {
        debug_assert!(index < self.score_coordinate_count);
        let start = index * self.score_tile_rows;
        &self.score_tile[start..start + self.score_tile_len]
    }

    /// Prepared row capacity of the score tile.
    #[must_use]
    pub const fn score_tile_rows(&self) -> usize {
        self.score_tile_rows
    }

    /// Number of scalar slots in the prepared score tile.
    #[must_use]
    pub const fn score_tile_value_count(&self) -> usize {
        self.score_tile.len()
    }

    /// Returns a zero-filled temporary buffer suitable for penalty gradients.
    pub fn penalty_gradient_mut(&mut self, len: usize) -> &mut [f64] {
        self.penalty_gradient.resize(len, 0.0);
        self.penalty_gradient.fill(0.0);
        &mut self.penalty_gradient
    }

    /// Returns reusable flat eta and score buffers for runtime-dimensional families.
    pub fn dynamic_buffers_mut(&mut self, len: usize) -> (&mut [f64], &mut [f64]) {
        self.dynamic_values.resize(len, 0.0);
        self.dynamic_scores.resize(len, 0.0);
        self.dynamic_scores.fill(0.0);
        (&mut self.dynamic_values, &mut self.dynamic_scores)
    }

    /// Returns a reusable flat eta buffer for runtime-dimensional value paths.
    pub fn dynamic_values_mut(&mut self, len: usize) -> &mut [f64] {
        self.dynamic_values.resize(len, 0.0);
        &mut self.dynamic_values
    }

    /// Copies runtime-family scores into the current per-coordinate score tile.
    pub fn store_weighted_dynamic_scores(&mut self, tile_row: usize, weight: f64) {
        debug_assert_eq!(self.dynamic_scores.len(), self.score_coordinate_count);
        for index in 0..self.dynamic_scores.len() {
            let score = self.dynamic_scores[index];
            self.set_score(index, tile_row, weight * score);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_SCORE_TILE_VALUES, GradientWorkspace};
    use crate::ModelError;

    #[test]
    fn automatic_tile_bounds_high_dimensional_score_storage() {
        // D = 82: 82 location coordinates plus 82 * 83 / 2 triangular
        // covariance coordinates.
        let coordinate_count = 82 + 82 * 83 / 2;
        let mut workspace = GradientWorkspace::new();

        let tile_rows = workspace.prepare_score_tile(coordinate_count, 100_000);

        assert_eq!(tile_rows, DEFAULT_SCORE_TILE_VALUES / coordinate_count);
        assert!(workspace.score_tile_value_count() <= DEFAULT_SCORE_TILE_VALUES);
        assert_eq!(
            workspace.score_tile_value_count(),
            coordinate_count * tile_rows
        );

        workspace.set_score_tile_len(17);
        workspace.set_score(coordinate_count - 1, 16, 3.5);
        assert_eq!(workspace.scores(coordinate_count - 1).len(), 17);
        assert_eq!(
            workspace.scores(coordinate_count - 1)[16].to_bits(),
            3.5_f64.to_bits()
        );
    }

    #[test]
    fn explicit_tile_size_must_be_positive() {
        assert_eq!(
            GradientWorkspace::try_with_tile_rows(0).unwrap_err(),
            ModelError::InvalidParameter {
                parameter: "score tile rows",
                expected: "positive",
            }
        );
    }
}
