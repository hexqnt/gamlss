#![forbid(unsafe_code)]
//! Small built-in datasets for runnable examples and quick experiments.
//!
//! Built-in data is authored as CSV under `data/` and compiled into static
//! `f64` arrays by `build.rs`. Loading a dataset therefore performs no I/O or
//! allocation. The optional [`simulate`] module creates owned synthetic data
//! from a parameterized data-generating process.

#[allow(
    clippy::unreadable_literal,
    reason = "numeric literals are generated from CSV data"
)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/built_in_datasets.rs"));
}

#[cfg(feature = "rand")]
pub mod simulate;

/// A one-dimensional regression dataset with a continuous response.
///
/// Both slices contain 2,000 aligned, finite observations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct A1Dataset {
    /// One scalar predictor per observation.
    pub x: &'static [f64],
    /// Continuous response values corresponding to [`Self::x`].
    pub y: &'static [f64],
}

/// Returns the `a1` regression dataset.
#[must_use]
pub const fn a1() -> A1Dataset {
    A1Dataset {
        x: &generated::A1_X,
        y: &generated::A1_Y,
    }
}

#[cfg(test)]
mod tests {
    use super::a1;

    fn assert_aligned_finite(x: &[f64], y: &[f64]) {
        assert_eq!(x.len(), y.len());
        assert!(!x.is_empty());
        assert!(x.iter().chain(y).all(|value| value.is_finite()));
    }

    #[test]
    fn a1_has_aligned_finite_columns() {
        let data = a1();
        assert_aligned_finite(data.x, data.y);
        assert_eq!(data.x.len(), 2_000);
    }
}
