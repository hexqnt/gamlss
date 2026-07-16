#![forbid(unsafe_code)]
//! Small built-in datasets for runnable examples and quick experiments.
//!
//! Built-in data is authored as CSV under `data/` and compiled into static
//! `f64` arrays by `build.rs`. Loading a dataset therefore performs no I/O or
//! allocation. The optional [`simulate`] module creates owned synthetic data
//! from a parameterized data-generating process.

mod generated {
    include!(concat!(env!("OUT_DIR"), "/built_in_datasets.rs"));
}

#[cfg(feature = "rand")]
pub mod simulate;

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

/// Returns the synthetic dataset used by the `simple_fit` example.
#[must_use]
pub const fn linear_normal() -> LinearNormalDataset {
    LinearNormalDataset {
        x: &generated::LINEAR_NORMAL_X,
        y: &generated::LINEAR_NORMAL_Y,
    }
}

/// A synthetic normal response whose location and dispersion both vary with
/// the predictor.
///
/// The data is intended for examples where `mu` and `sigma` receive distinct
/// predictors. Both slices have the same length and contain only finite values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeteroscedasticNormalDataset {
    /// One scalar predictor per observation.
    pub x: &'static [f64],
    /// Continuous response values corresponding to [`Self::x`].
    pub y: &'static [f64],
}

/// Returns a small dataset with a changing conditional location and scale.
#[must_use]
pub const fn heteroscedastic_normal() -> HeteroscedasticNormalDataset {
    HeteroscedasticNormalDataset {
        x: &generated::HETEROSCEDASTIC_NORMAL_X,
        y: &generated::HETEROSCEDASTIC_NORMAL_Y,
    }
}

#[cfg(test)]
mod tests {
    use super::{heteroscedastic_normal, linear_normal};

    fn assert_aligned_finite(x: &[f64], y: &[f64]) {
        assert_eq!(x.len(), y.len());
        assert!(!x.is_empty());
        assert!(x.iter().chain(y).all(|value| value.is_finite()));
    }

    #[test]
    fn linear_normal_has_aligned_finite_columns() {
        let data = linear_normal();
        assert_aligned_finite(data.x, data.y);
    }

    #[test]
    fn heteroscedastic_normal_has_aligned_finite_columns() {
        let data = heteroscedastic_normal();
        assert_aligned_finite(data.x, data.y);
    }
}
