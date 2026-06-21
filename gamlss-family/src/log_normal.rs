use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{Family, HasCdf, HasCrps, HasQuantile, Identity, Log};

use crate::special::{unit_normal_cdf, unit_normal_quantile};

pub use log_location_log_sd::{
    LogLocationLogSd, LogNormalLogLocationLogSdEta, LogNormalLogLocationLogSdTheta,
};
pub use mean_cv::{LogNormalMeanCvEta, LogNormalMeanCvTheta, MeanCv};
pub use mean_log_sd::{LogNormalMeanLogSdEta, LogNormalMeanLogSdTheta, MeanLogSd};
pub use median_log_sd::{LogNormalMedianLogSdEta, LogNormalMedianLogSdTheta, MedianLogSd};

mod log_location_log_sd;
mod mean_cv;
mod mean_log_sd;
mod median_log_sd;

const HALF_LOG_2_PI: f64 = 0.918_938_533_204_672_7;

/// Log-normal family implementation carrier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogNormal<Param = LogLocationLogSd, FirstLink = Identity, SecondLink = Log> {
    marker: PhantomData<(Param, FirstLink, SecondLink)>,
}

impl<Param, FirstLink, SecondLink> LogNormal<Param, FirstLink, SecondLink> {
    /// Creates a stateless log-normal family.
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn valid_log_location_log_sd(theta: LogNormalLogLocationLogSdTheta) -> bool {
        theta.log_location.is_finite() && theta.log_sd > 0.0 && theta.log_sd.is_finite()
    }

    #[inline(always)]
    fn nll_log_location_log_sd(y: f64, theta: LogNormalLogLocationLogSdTheta) -> f64 {
        if y <= 0.0 || !y.is_finite() || !Self::valid_log_location_log_sd(theta) {
            return f64::INFINITY;
        }

        let log_y = y.ln();
        let residual = log_y - theta.log_location;
        let z = residual / theta.log_sd;
        log_y + HALF_LOG_2_PI + theta.log_sd.ln() + 0.5 * z * z
    }

    #[inline(always)]
    fn gradient_log_location_log_sd(y: f64, theta: LogNormalLogLocationLogSdTheta) -> (f64, f64) {
        let residual = y.ln() - theta.log_location;
        let sigma2 = theta.log_sd * theta.log_sd;
        (
            (theta.log_location - y.ln()) / sigma2,
            1.0 / theta.log_sd - residual * residual / (sigma2 * theta.log_sd),
        )
    }

    #[inline(always)]
    fn cdf_log_location_log_sd(y: f64, theta: LogNormalLogLocationLogSdTheta) -> f64 {
        if !y.is_finite() || !Self::valid_log_location_log_sd(theta) {
            return f64::NAN;
        }
        if y <= 0.0 {
            return 0.0;
        }

        unit_normal_cdf((y.ln() - theta.log_location) / theta.log_sd)
    }

    #[inline(always)]
    fn quantile_log_location_log_sd(p: f64, theta: LogNormalLogLocationLogSdTheta) -> f64 {
        if !Self::valid_log_location_log_sd(theta) {
            return f64::NAN;
        }

        (theta.log_location + theta.log_sd * unit_normal_quantile(p)).exp()
    }

    #[inline(always)]
    fn crps_log_location_log_sd(y: f64, theta: LogNormalLogLocationLogSdTheta) -> f64 {
        if y < 0.0 || !y.is_finite() || !Self::valid_log_location_log_sd(theta) {
            return f64::NAN;
        }

        let mean = (theta.log_location + 0.5 * theta.log_sd * theta.log_sd).exp();
        let gini_cdf = unit_normal_cdf(theta.log_sd / std::f64::consts::SQRT_2);
        let half_gini = mean * (2.0 * gini_cdf - 1.0);
        if y == 0.0 {
            return mean - half_gini;
        }

        let z = (y.ln() - theta.log_location) / theta.log_sd;
        y * (2.0 * unit_normal_cdf(z) - 1.0)
            - 2.0 * mean * (unit_normal_cdf(z - theta.log_sd) + gini_cdf - 1.0)
    }
}

impl<Param, FirstLink, SecondLink> Default for LogNormal<Param, FirstLink, SecondLink> {
    fn default() -> Self {
        Self::new()
    }
}

macro_rules! impl_log_normal_helpers {
    ($param:ty, $first:ident, $second:ident) => {
        impl<$first, $second> HasCdf for LogNormal<$param, $first, $second>
        where
            LogNormal<$param, $first, $second>: for<'obs> Family<Observation<'obs> = f64>,
            <LogNormal<$param, $first, $second> as Family>::Theta:
                Copy + Into<LogNormalLogLocationLogSdTheta>,
        {
            fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
                Self::cdf_log_location_log_sd(y, theta.into())
            }
        }

        impl<$first, $second> HasQuantile for LogNormal<$param, $first, $second>
        where
            LogNormal<$param, $first, $second>: for<'obs> Family<Observation<'obs> = f64>,
            <LogNormal<$param, $first, $second> as Family>::Theta:
                Copy + Into<LogNormalLogLocationLogSdTheta>,
        {
            fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
                Self::quantile_log_location_log_sd(p, theta.into())
            }
        }

        impl<$first, $second> HasCrps for LogNormal<$param, $first, $second>
        where
            LogNormal<$param, $first, $second>: for<'obs> Family<Observation<'obs> = f64>,
            <LogNormal<$param, $first, $second> as Family>::Theta:
                Copy + Into<LogNormalLogLocationLogSdTheta>,
        {
            fn crps<'obs>(&self, y: Self::Observation<'obs>, theta: Self::Theta) -> f64 {
                Self::crps_log_location_log_sd(y, theta.into())
            }
        }

        #[cfg(feature = "rand")]
        impl<Rng, $first, $second> CanSimulate<Rng> for LogNormal<$param, $first, $second>
        where
            Rng: rand::Rng,
            LogNormal<$param, $first, $second>: for<'obs> Family<Observation<'obs> = f64>,
            <LogNormal<$param, $first, $second> as Family>::Theta:
                Copy + Into<LogNormalLogLocationLogSdTheta>,
        {
            fn sample(&self, rng: &mut Rng, theta: Self::Theta) -> f64 {
                let theta = theta.into();
                if !Self::valid_log_location_log_sd(theta) {
                    return f64::NAN;
                }

                rand_distr::Distribution::sample(
                    &rand_distr::LogNormal::new(theta.log_location, theta.log_sd)
                        .expect("validated log-normal parameters must construct"),
                    rng,
                )
            }
        }
    };
}

impl_log_normal_helpers!(MeanLogSd, MeanLink, LogSdLink);
impl_log_normal_helpers!(MeanCv, MeanLink, CvLink);
impl_log_normal_helpers!(MedianLogSd, MedianLink, LogSdLink);
impl_log_normal_helpers!(LogLocationLogSd, LocationLink, LogSdLink);

/// Log-normal distribution parameterized by mean and log standard deviation.
pub type LogNormalMeanLogSd = LogNormal<MeanLogSd, Log, Log>;
/// Log-normal distribution parameterized by mean and coefficient of variation.
pub type LogNormalMeanCv = LogNormal<MeanCv, Log, Log>;
/// Log-normal distribution parameterized by median and log standard deviation.
pub type LogNormalMedianLogSd = LogNormal<MedianLogSd, Log, Log>;
/// Log-normal distribution parameterized by log-location and log standard deviation.
pub type LogNormalLogLocationLogSd = LogNormal<LogLocationLogSd, Identity, Log>;

/// Backward-compatible eta alias for log-location/log-SD log-normal.
pub type LogNormalEta = LogNormalLogLocationLogSdEta;
/// Backward-compatible theta alias for log-location/log-SD log-normal.
pub type LogNormalTheta = LogNormalLogLocationLogSdTheta;

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;
    use gamlss_core::{Family, HasCdf, HasCrps, HasQuantile};
    use statrs::distribution::{ContinuousCDF, LogNormal as StatrsLogNormal};

    use super::{
        LogNormalLogLocationLogSd, LogNormalLogLocationLogSdTheta, LogNormalMeanCv,
        LogNormalMeanCvTheta, LogNormalMeanLogSd, LogNormalMeanLogSdTheta, LogNormalMedianLogSd,
    };
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn log_normal_parameterization_gradients_match_finite_difference() {
        assert_gradient_matches_finite_difference::<_, 2>(
            &LogNormalLogLocationLogSd::new(),
            1.7,
            [0.4, -0.2],
        );
        assert_gradient_matches_finite_difference::<_, 2>(
            &LogNormalMeanLogSd::new(),
            1.7,
            [1.5_f64.ln(), 0.8_f64.ln()],
        );
        assert_gradient_matches_finite_difference::<_, 2>(
            &LogNormalMeanCv::new(),
            1.7,
            [1.5_f64.ln(), 0.9_f64.ln()],
        );
        assert_gradient_matches_finite_difference::<_, 2>(
            &LogNormalMedianLogSd::new(),
            1.7,
            [1.2_f64.ln(), 0.8_f64.ln()],
        );
    }

    #[test]
    fn log_normal_mean_matches_log_location_equivalent() {
        let mean = LogNormalMeanLogSd::new();
        let canonical = LogNormalLogLocationLogSd::new();
        let theta = LogNormalMeanLogSdTheta {
            mean: 1.5,
            log_sd: 0.8,
        };
        let kernel = theta.log_location_log_sd();

        assert_relative_eq!(
            mean.nll(1.7, theta),
            canonical.nll(1.7, kernel),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean.cdf(1.7, theta),
            canonical.cdf(1.7, kernel),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean.quantile(0.4, theta),
            canonical.quantile(0.4, kernel),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean.crps(1.7, theta),
            canonical.crps(1.7, kernel),
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn log_normal_mean_cv_matches_log_location_equivalent() {
        let mean_cv = LogNormalMeanCv::new();
        let canonical = LogNormalLogLocationLogSd::new();
        let theta = LogNormalMeanCvTheta { mean: 1.5, cv: 0.9 };
        let kernel = theta.log_location_log_sd();

        assert_relative_eq!(
            mean_cv.nll(1.7, theta),
            canonical.nll(1.7, kernel),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean_cv.cdf(1.7, theta),
            canonical.cdf(1.7, kernel),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean_cv.quantile(0.4, theta),
            canonical.quantile(0.4, kernel),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean_cv.crps(1.7, theta),
            canonical.crps(1.7, kernel),
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn log_normal_rejects_invalid_domains() {
        let family = LogNormalMeanLogSd::new();
        assert!(
            family
                .nll(
                    1.7,
                    LogNormalMeanLogSdTheta {
                        mean: 1.0,
                        log_sd: 0.8
                    }
                )
                .is_finite()
        );
        assert!(
            family
                .nll(
                    0.0,
                    LogNormalMeanLogSdTheta {
                        mean: 1.0,
                        log_sd: 0.8
                    }
                )
                .is_infinite()
        );
        assert!(
            family
                .nll(
                    1.7,
                    LogNormalMeanLogSdTheta {
                        mean: 0.0,
                        log_sd: 0.8
                    }
                )
                .is_infinite()
        );
        assert!(
            family
                .nll(
                    1.7,
                    LogNormalMeanLogSdTheta {
                        mean: 1.0,
                        log_sd: 0.0
                    }
                )
                .is_infinite()
        );
    }

    #[test]
    fn log_normal_cdf_and_quantile_match_statrs_reference() {
        let family = LogNormalLogLocationLogSd::new();
        let theta = LogNormalLogLocationLogSdTheta {
            log_location: 0.4,
            log_sd: 0.8,
        };
        let reference = StatrsLogNormal::new(theta.log_location, theta.log_sd).unwrap();

        for y in [0.05, 0.25, 1.0, 2.0, 8.0] {
            assert_relative_eq!(family.cdf(y, theta), reference.cdf(y), epsilon = 1.0e-7);
        }

        for p in [0.01, 0.1, 0.5, 0.9, 0.99] {
            assert_relative_eq!(
                family.quantile(p, theta),
                reference.inverse_cdf(p),
                epsilon = 1.0e-6
            );
        }
    }

    #[test]
    fn log_normal_boundaries_and_crps_behave_like_kernel() {
        let family = LogNormalLogLocationLogSd::new();
        let theta = LogNormalLogLocationLogSdTheta {
            log_location: 0.0,
            log_sd: 1.0,
        };

        assert_eq!(family.cdf(0.0, theta), 0.0);
        assert_eq!(family.cdf(-1.0, theta), 0.0);
        assert_eq!(family.quantile(0.0, theta), 0.0);
        assert_eq!(family.quantile(1.0, theta), f64::INFINITY);
        assert!(family.quantile(f64::NAN, theta).is_nan());
        assert!(family.crps(1.0, theta).is_finite());
        assert!(family.crps(-1.0, theta).is_nan());
    }

    #[cfg(feature = "rand")]
    #[test]
    fn log_normal_sampling_returns_finite_values_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = LogNormalMeanLogSd::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family.sample(
            &mut rng,
            LogNormalMeanLogSdTheta {
                mean: 1.5,
                log_sd: 0.8,
            },
        );
        assert!(sample > 0.0 && sample.is_finite());
    }
}
