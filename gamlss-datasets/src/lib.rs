#![forbid(unsafe_code)]
//! Small built-in datasets for runnable examples and quick experiments.
//!
//! Built-in data is authored as CSV under `data/` and compiled into static
//! arrays by `build.rs`. Loading a dataset therefore performs no I/O or
//! allocation. The optional [`simulate`] module creates owned synthetic data
//! from a parameterized data-generating process.

#[allow(
    clippy::unreadable_literal,
    reason = "numeric literals are generated from CSV data"
)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/built_in_datasets.rs"));
}

pub use generated::*;

#[cfg(feature = "rand")]
pub mod simulate;

pub use time::{Date, Month, Time};

/// A regression dataset with type-defined predictors and responses.
///
/// `X` and `Y` deliberately have no element-type or dimensionality constraints.
/// Built-in datasets use one `x` slice and represent a scalar response as
/// `&[f64]` or fixed-width multivariate response rows as `&[[f64; D]]`.
///
/// The producer of a dataset is responsible for ensuring that all columns have
/// the same number of observations. Built-in datasets check this invariant at
/// build time.
///
/// # Example
///
/// ```
/// use gamlss_datasets::{Dataset, Date, Month};
///
/// let dates = [
///     Date::from_calendar_date(2025, Month::January, 1).unwrap(),
///     Date::from_calendar_date(2025, Month::January, 2).unwrap(),
/// ];
/// let responses = [[1.0, 2.0], [1.5, 2.5]];
/// let data = Dataset::new(&dates[..], &responses[..]);
///
/// assert_eq!(data.x[0].year(), 2025);
/// assert_eq!(data.y[0], [1.0, 2.0]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dataset<X, Y> {
    /// Predictor storage in its native representation.
    pub x: X,
    /// Response storage in its native representation.
    pub y: Y,
}

impl<X, Y> Dataset<X, Y> {
    /// Creates a dataset from aligned predictor and response storage.
    ///
    /// This constructor is intentionally `const` and performs no runtime shape
    /// checks. The caller must ensure that every column represents the same
    /// number of observations.
    #[must_use]
    pub const fn new(x: X, y: Y) -> Self {
        Self { x, y }
    }

    /// Decomposes the dataset into its predictor and response storage.
    #[must_use]
    pub fn into_parts(self) -> (X, Y) {
        (self.x, self.y)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::{Dataset, Date, Month, Time, a1};

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

    #[test]
    fn generic_dataset_preserves_predictor_type_and_response_shape() {
        let dates = [
            Date::from_calendar_date(2025, Month::January, 1).unwrap(),
            Date::from_calendar_date(2025, Month::January, 2).unwrap(),
        ];
        let responses = [[1.0, 2.0], [3.0, 4.0]];

        let data = Dataset::new(&dates[..], &responses[..]);

        assert_eq!(data.x[1].day(), 2);
        assert_eq!(data.y[1], [3.0, 4.0]);
    }

    #[test]
    fn dataset_can_own_storage() {
        let data = Dataset::new(vec![1_u32, 2], vec![[3.0, 4.0], [5.0, 6.0]]);
        let (x, y) = data.into_parts();

        assert_eq!(x, [1, 2]);
        assert_eq!(y, [[3.0, 4.0], [5.0, 6.0]]);
    }

    #[test]
    fn date_and_time_constructors_validate_components() {
        assert_eq!(
            Date::from_calendar_date(2024, Month::February, 29)
                .unwrap()
                .day(),
            29
        );
        assert!(Date::from_calendar_date(2023, Month::February, 29).is_err());
        assert_eq!(
            Time::from_hms_nano(23, 59, 58, 123).unwrap().nanosecond(),
            123
        );
        assert!(Time::from_hms_nano(24, 0, 0, 0).is_err());
    }
}
