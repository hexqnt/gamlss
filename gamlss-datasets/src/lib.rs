#![forbid(unsafe_code)]
//! Small built-in datasets for runnable examples and quick experiments.
//!
//! This internal workspace crate deliberately has no data-frame dependency.
//! Datasets expose typed, borrowed slices so callers can use them with the
//! low-level `gamlss` API without parsing files or allocating intermediate
//! tables.

/// A small synthetic response with a roughly linear conditional mean.
///
/// The response is suitable for the introductory normal-location example:
/// `y` is modeled by a linear predictor of `x` and an intercept-only scale.
/// Both slices have the same length and contain only finite values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearNormalDataset {
    /// One scalar predictor per observation.
    pub x: &'static [f64],
    /// Continuous response values corresponding to [`Self::x`].
    pub y: &'static [f64],
}

const LINEAR_NORMAL_X: [f64; 5] = [0.0, 1.0, 2.0, 3.0, 4.0];
const LINEAR_NORMAL_Y: [f64; 5] = [1.0, 1.5, 1.7, 2.4, 2.5];

/// Returns the synthetic dataset used by the `simple_fit` example.
#[must_use]
pub const fn linear_normal() -> LinearNormalDataset {
    LinearNormalDataset {
        x: &LINEAR_NORMAL_X,
        y: &LINEAR_NORMAL_Y,
    }
}

#[cfg(test)]
mod tests {
    use super::linear_normal;

    #[test]
    fn linear_normal_has_aligned_finite_columns() {
        let data = linear_normal();

        assert_eq!(data.x.len(), data.y.len());
        assert!(!data.x.is_empty());
        assert!(data.x.iter().chain(data.y).all(|value| value.is_finite()));
    }
}
