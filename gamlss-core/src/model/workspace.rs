use crate::{Family, ModelError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScoreTileLimit {
    Automatic,
    MaxBytes(usize),
    MaxRows(usize),
}

/// Controls the bounded score storage used during gradient evaluation.
///
/// The default policy targets approximately four MiB of score values. Most
/// callers should use [`Gamlss::gradient_workspace`](crate::Gamlss::gradient_workspace)
/// or [`Gamlss::into_workspace_objective`](crate::Gamlss::into_workspace_objective)
/// and rely on that default. Explicit policies are useful when profiling a
/// workload or bounding worker-local memory in an executor.
///
/// A byte budget applies only to the score tile, not to the complete model
/// workspace. At least one observation row is retained when observations are
/// present, so a single score row may exceed a very small byte budget.
///
/// # Examples
///
/// ```
/// use gamlss_core::ScoreTilePolicy;
///
/// let memory_bounded = ScoreTilePolicy::try_max_bytes(8 * 1024 * 1024)?;
/// let row_bounded = ScoreTilePolicy::try_max_rows(256)?;
/// # Ok::<(), gamlss_core::ModelError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScoreTilePolicy {
    limit: ScoreTileLimit,
}

impl ScoreTilePolicy {
    /// Default score payload budget used by [`Self::automatic`].
    pub const DEFAULT_BYTE_BUDGET: usize = 4 * 1024 * 1024;

    /// Uses the library default score-memory budget.
    #[inline]
    #[must_use]
    pub const fn automatic() -> Self {
        Self {
            limit: ScoreTileLimit::Automatic,
        }
    }

    /// Limits the score payload to approximately `bytes` bytes.
    ///
    /// The effective tile is also capped by the observation count. At least
    /// one row is retained for a non-empty data set, even when one score row is
    /// larger than `bytes`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `bytes == 0`.
    pub const fn try_max_bytes(bytes: usize) -> Result<Self, ModelError> {
        if bytes == 0 {
            return Err(ModelError::InvalidParameter {
                parameter: "score tile bytes",
                expected: "positive",
            });
        }
        Ok(Self {
            limit: ScoreTileLimit::MaxBytes(bytes),
        })
    }

    /// Limits each score tile to at most `rows` observation rows.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `rows == 0`.
    pub const fn try_max_rows(rows: usize) -> Result<Self, ModelError> {
        if rows == 0 {
            return Err(ModelError::InvalidParameter {
                parameter: "score tile rows",
                expected: "positive",
            });
        }
        Ok(Self {
            limit: ScoreTileLimit::MaxRows(rows),
        })
    }

    /// Explicit byte limit, or `None` for automatic and row-based policies.
    #[inline]
    #[must_use]
    pub const fn max_bytes(self) -> Option<usize> {
        match self.limit {
            ScoreTileLimit::MaxBytes(bytes) => Some(bytes),
            ScoreTileLimit::Automatic | ScoreTileLimit::MaxRows(_) => None,
        }
    }

    /// Explicit row limit, or `None` for automatic and byte-based policies.
    #[inline]
    #[must_use]
    pub const fn max_rows(self) -> Option<usize> {
        match self.limit {
            ScoreTileLimit::MaxRows(rows) => Some(rows),
            ScoreTileLimit::Automatic | ScoreTileLimit::MaxBytes(_) => None,
        }
    }

    fn requested_rows(self, coordinate_count: usize) -> usize {
        match self.limit {
            ScoreTileLimit::Automatic => {
                rows_for_byte_budget(Self::DEFAULT_BYTE_BUDGET, coordinate_count)
            }
            ScoreTileLimit::MaxBytes(bytes) => rows_for_byte_budget(bytes, coordinate_count),
            ScoreTileLimit::MaxRows(rows) => rows,
        }
    }
}

#[inline]
fn rows_for_byte_budget(bytes: usize, coordinate_count: usize) -> usize {
    (bytes / size_of::<f64>() / coordinate_count.max(1)).max(1)
}

impl Default for ScoreTilePolicy {
    #[inline]
    fn default() -> Self {
        Self::automatic()
    }
}

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
    score_tile_policy: ScoreTilePolicy,
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

    /// Creates an empty workspace configured with `policy`.
    ///
    /// Buffers are allocated lazily by [`Self::prepare_score_tile`].
    #[inline]
    #[must_use]
    pub fn with_score_tile_policy(policy: ScoreTilePolicy) -> Self {
        Self {
            score_tile_policy: policy,
            ..Self::default()
        }
    }

    /// Prepares bounded score storage and returns the number of rows per tile.
    ///
    /// The automatic policy targets approximately four MiB and always keeps at
    /// least one row when `nobs > 0`. An explicit [`ScoreTilePolicy`] takes
    /// precedence.
    pub fn prepare_score_tile(&mut self, coordinate_count: usize, nobs: usize) -> usize {
        let divisor = coordinate_count.max(1);
        let requested_rows = self.score_tile_policy.requested_rows(coordinate_count);
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

    /// Number of bytes in the prepared score payload.
    ///
    /// This reports `len`, not retained `Vec` capacity, and excludes all other
    /// family and gradient workspace buffers.
    #[must_use]
    pub const fn score_tile_bytes(&self) -> usize {
        self.score_tile.len() * size_of::<f64>()
    }

    /// Policy used to size score tiles.
    #[must_use]
    pub const fn score_tile_policy(&self) -> ScoreTilePolicy {
        self.score_tile_policy
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
    use super::{GradientWorkspace, ScoreTilePolicy};
    use crate::ModelError;

    #[test]
    fn automatic_tile_bounds_high_dimensional_score_storage() {
        // D = 82: 82 location coordinates plus 82 * 83 / 2 triangular
        // covariance coordinates.
        let coordinate_count = 82 + 82 * 83 / 2;
        let mut workspace = GradientWorkspace::new();

        let tile_rows = workspace.prepare_score_tile(coordinate_count, 100_000);

        let default_value_budget = ScoreTilePolicy::DEFAULT_BYTE_BUDGET / size_of::<f64>();
        assert_eq!(tile_rows, default_value_budget / coordinate_count);
        assert!(workspace.score_tile_value_count() <= default_value_budget);
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
    fn explicit_tile_limits_must_be_positive() {
        assert_eq!(
            ScoreTilePolicy::try_max_rows(0).unwrap_err(),
            ModelError::InvalidParameter {
                parameter: "score tile rows",
                expected: "positive",
            }
        );
        assert_eq!(
            ScoreTilePolicy::try_max_bytes(0).unwrap_err(),
            ModelError::InvalidParameter {
                parameter: "score tile bytes",
                expected: "positive",
            }
        );
    }

    #[test]
    fn byte_and_row_policies_select_expected_tile_sizes() {
        let byte_policy = ScoreTilePolicy::try_max_bytes(80).unwrap();
        let mut by_bytes = GradientWorkspace::with_score_tile_policy(byte_policy);
        assert_eq!(by_bytes.prepare_score_tile(2, 100), 5);
        assert_eq!(by_bytes.score_tile_bytes(), 80);
        assert_eq!(by_bytes.score_tile_policy(), byte_policy);

        let row_policy = ScoreTilePolicy::try_max_rows(7).unwrap();
        let mut by_rows = GradientWorkspace::with_score_tile_policy(row_policy);
        assert_eq!(by_rows.prepare_score_tile(2, 5), 5);
        assert_eq!(by_rows.score_tile_policy().max_rows(), Some(7));
        assert_eq!(by_rows.score_tile_policy().max_bytes(), None);
    }

    #[test]
    fn byte_policy_keeps_one_row_when_a_row_exceeds_the_budget() {
        let policy = ScoreTilePolicy::try_max_bytes(1).unwrap();
        let mut workspace = GradientWorkspace::with_score_tile_policy(policy);

        assert_eq!(workspace.prepare_score_tile(3, 10), 1);
        assert_eq!(workspace.score_tile_bytes(), 3 * size_of::<f64>());
    }
}
