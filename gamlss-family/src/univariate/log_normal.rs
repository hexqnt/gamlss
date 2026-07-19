use std::marker::PhantomData;

use gamlss_core::{Family, HasCdf, HasCrps, HasQuantile, Identity, Log};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

use gamlss_special::{unit_normal_cdf, unit_normal_quantile};

use crate::constants::HALF_LOG_2_PI;

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

/// Log-normal family implementation carrier.
///
/// All parameterizations map to the location-scale model for $\log Y$:
///
/// $$
/// \begin{aligned}
/// \log Y &\sim \mathcal N(m,s^2), \qquad y>0,\ s>0, \\\\
/// f(y\mid m,s) &= \frac{1}{ys\sqrt{2\pi}}
/// \exp\\!\left[-\frac{(\log y-m)^2}{2s^2}\right].
/// \end{aligned}
/// $$
///
/// Its median is $\exp(m)$, while $\mathbb{E}(Y)=\exp(m+s^2/2)$ and $\operatorname{Var}(Y)=(\exp(s^2)-1)\exp(2m+s^2)$.
///
/// In the canonical carrier, [`LogNormalLogLocationLogSdTheta::log_location`] stores $m$ and [`LogNormalLogLocationLogSdTheta::log_sd`] stores $s$. Here `log_sd` means “SD of the log-transformed response”, not $\log s$.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogNormal<Param = LogLocationLogSd, FirstLink = Identity, SecondLink = Log> {
    marker: PhantomData<(Param, FirstLink, SecondLink)>,
}

impl<Param, FirstLink, SecondLink> LogNormal<Param, FirstLink, SecondLink> {
    /// Creates a stateless log-normal family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn valid_log_location_log_sd(theta: LogNormalLogLocationLogSdTheta) -> bool {
        theta.log_location.is_finite() && theta.log_sd > 0.0 && theta.log_sd.is_finite()
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn nll_log_location_log_sd(y: f64, theta: LogNormalLogLocationLogSdTheta) -> f64 {
        if y <= 0.0 || !y.is_finite() || !Self::valid_log_location_log_sd(theta) {
            return f64::INFINITY;
        }

        let log_y = y.ln();
        let residual = log_y - theta.log_location;
        let z = residual / theta.log_sd;
        log_y + HALF_LOG_2_PI + theta.log_sd.ln() + 0.5 * z * z
    }

    #[inline]
    fn gradient_log_location_log_sd(y: f64, theta: LogNormalLogLocationLogSdTheta) -> (f64, f64) {
        let residual = y.ln() - theta.log_location;
        let sigma2 = theta.log_sd * theta.log_sd;
        (
            (theta.log_location - y.ln()) / sigma2,
            1.0 / theta.log_sd - residual * residual / (sigma2 * theta.log_sd),
        )
    }

    #[inline]
    fn cdf_log_location_log_sd(y: f64, theta: LogNormalLogLocationLogSdTheta) -> f64 {
        if !y.is_finite() || !Self::valid_log_location_log_sd(theta) {
            return f64::NAN;
        }
        if y <= 0.0 {
            return 0.0;
        }

        unit_normal_cdf((y.ln() - theta.log_location) / theta.log_sd)
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn quantile_log_location_log_sd(p: f64, theta: LogNormalLogLocationLogSdTheta) -> f64 {
        if !Self::valid_log_location_log_sd(theta) {
            return f64::NAN;
        }

        (theta.log_location + theta.log_sd * unit_normal_quantile(p)).exp()
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
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
            fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
                Self::cdf_log_location_log_sd(y, (*theta).into())
            }
        }

        impl<$first, $second> HasQuantile for LogNormal<$param, $first, $second>
        where
            LogNormal<$param, $first, $second>: for<'obs> Family<Observation<'obs> = f64>,
            <LogNormal<$param, $first, $second> as Family>::Theta:
                Copy + Into<LogNormalLogLocationLogSdTheta>,
        {
            fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
                Self::quantile_log_location_log_sd(p, (*theta).into())
            }
        }

        impl<$first, $second> HasCrps for LogNormal<$param, $first, $second>
        where
            LogNormal<$param, $first, $second>: for<'obs> Family<Observation<'obs> = f64>,
            <LogNormal<$param, $first, $second> as Family>::Theta:
                Copy + Into<LogNormalLogLocationLogSdTheta>,
        {
            fn crps(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
                Self::crps_log_location_log_sd(y, (*theta).into())
            }
        }

        #[cfg(feature = "rand")]
        impl<Rng, $first, $second> TrySimulate<Rng> for LogNormal<$param, $first, $second>
        where
            Rng: rand::Rng,
            LogNormal<$param, $first, $second>: for<'obs> Family<Observation<'obs> = f64>,
            <LogNormal<$param, $first, $second> as Family>::Theta:
                Copy + Into<LogNormalLogLocationLogSdTheta>,
        {
            type Sample = f64;

            fn try_sample(
                &self,
                rng: &mut Rng,
                theta: &Self::Theta,
            ) -> Result<f64, SimulationError> {
                let theta = (*theta).into();
                if !Self::valid_log_location_log_sd(theta) {
                    return Err(SimulationError::InvalidParameters("Log-normal theta"));
                }

                let distribution = rand_distr::LogNormal::new(theta.log_location, theta.log_sd)
                    .map_err(|_| SimulationError::BackendRejected("Log-normal location/scale"))?;
                crate::simulation::ensure_finite(
                    rand_distr::Distribution::sample(&distribution, rng),
                    "Log-normal sample",
                )
            }
        }
    };
}

impl_log_normal_helpers!(MeanLogSd, MeanLink, LogSdLink);
impl_log_normal_helpers!(MeanCv, MeanLink, CvLink);
impl_log_normal_helpers!(MedianLogSd, MedianLink, LogSdLink);
impl_log_normal_helpers!(LogLocationLogSd, LocationLink, LogSdLink);

/// Log-normal distribution parameterized by mean $\mu>0$ and log-scale SD $s>0$.
///
/// [`LogNormalMeanLogSdTheta::mean`] stores $\mu$ and [`LogNormalMeanLogSdTheta::log_sd`] stores $s$. The canonical log-location is $m=\log\mu-s^2/2$; the default log links apply to the `mean` and `log_sd` predictor fields.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/distributions/log_normal_mean_log_sd.svg")
)]
#[allow(clippy::doc_markdown)]
pub type LogNormalMeanLogSd = LogNormal<MeanLogSd, Log, Log>;
/// Log-normal distribution parameterized by mean $\mu>0$ and coefficient of variation $c>0$.
///
/// Here $c=\sqrt{\operatorname{Var}(Y)}/\mathbb{E}(Y)$.
///
/// $$
/// s^2=\log(1+c^2), \qquad m=\log\mu-\frac{s^2}{2},
/// \qquad \operatorname{Var}(Y)=\mu^2c^2.
/// $$
///
/// Both parameters use log links by default.
///
/// In code, $\mu$ and $c$ are the `mean` and `cv` fields of [`LogNormalMeanCvTheta`] and [`LogNormalMeanCvEta`].
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/distributions/log_normal_mean_cv.svg")
)]
pub type LogNormalMeanCv = LogNormal<MeanCv, Log, Log>;
/// Log-normal distribution parameterized by median $q_{0.5}>0$ and log-scale SD $s>0$.
///
/// The canonical log-location is $m=\log q_{0.5}$. The natural and eta carriers use the field names `median` and `log_sd`; both predictors use log links by default.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/distributions/log_normal_median_log_sd.svg")
)]
pub type LogNormalMedianLogSd = LogNormal<MedianLogSd, Log, Log>;
/// Log-normal distribution parameterized directly by log-location $m\in\mathbb{R}$ and log-scale SD $s>0$.
///
/// The corresponding code fields are `log_location` and `log_sd`. Their default links are identity and log, respectively: $m=\eta_m$ and $s=\exp(\eta_s)$.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/distributions/log_normal.svg")
)]
#[allow(clippy::doc_markdown)]
pub type LogNormalLogLocationLogSd = LogNormal<LogLocationLogSd, Identity, Log>;

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    #[cfg(feature = "rand")]
    use gamlss_core::TrySimulate;
    use gamlss_core::{Family, HasCdf, HasCrps, HasQuantile};
    use statrs::distribution::{ContinuousCDF, LogNormal as StatrsLogNormal};

    use super::{
        LogNormalLogLocationLogSd, LogNormalLogLocationLogSdTheta, LogNormalMeanCv,
        LogNormalMeanCvEta, LogNormalMeanCvTheta, LogNormalMeanLogSd, LogNormalMeanLogSdTheta,
        LogNormalMedianLogSd,
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
    fn log_normal_mean_cv_accepts_extreme_finite_cv() {
        let family = LogNormalMeanCv::new();
        let eta = LogNormalMeanCvEta {
            mean: 0.0,
            cv: 1.0e200_f64.ln(),
        };
        let theta = LogNormalMeanCvTheta {
            mean: 1.0,
            cv: 1.0e200,
        };

        let natural_nll = family.nll(1.0, &theta, &mut ());
        let (eta_nll, gradient) = family.nll_and_gradient_eta(1.0, &eta, &mut ());

        assert!(
            natural_nll.is_finite(),
            "natural-scale nll was {natural_nll}"
        );
        assert!(eta_nll.is_finite(), "eta-scale nll was {eta_nll}");
        assert!(
            gradient.mean.is_finite(),
            "mean gradient was {}",
            gradient.mean
        );
        assert!(gradient.cv.is_finite(), "cv gradient was {}", gradient.cv);
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
            mean.nll(1.7, &theta, &mut mean.workspace()),
            canonical.nll(1.7, &kernel, &mut canonical.workspace()),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean.cdf(1.7, &theta),
            canonical.cdf(1.7, &kernel),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean.quantile(0.4, &theta),
            canonical.quantile(0.4, &kernel),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean.crps(1.7, &theta),
            canonical.crps(1.7, &kernel),
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
            mean_cv.nll(1.7, &theta, &mut mean_cv.workspace()),
            canonical.nll(1.7, &kernel, &mut canonical.workspace()),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean_cv.cdf(1.7, &theta),
            canonical.cdf(1.7, &kernel),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean_cv.quantile(0.4, &theta),
            canonical.quantile(0.4, &kernel),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean_cv.crps(1.7, &theta),
            canonical.crps(1.7, &kernel),
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
                    &LogNormalMeanLogSdTheta {
                        mean: 1.0,
                        log_sd: 0.8
                    },
                    &mut family.workspace()
                )
                .is_finite()
        );
        assert!(
            family
                .nll(
                    0.0,
                    &LogNormalMeanLogSdTheta {
                        mean: 1.0,
                        log_sd: 0.8
                    },
                    &mut family.workspace()
                )
                .is_infinite()
        );
        assert!(
            family
                .nll(
                    1.7,
                    &LogNormalMeanLogSdTheta {
                        mean: 0.0,
                        log_sd: 0.8
                    },
                    &mut family.workspace()
                )
                .is_infinite()
        );
        assert!(
            family
                .nll(
                    1.7,
                    &LogNormalMeanLogSdTheta {
                        mean: 1.0,
                        log_sd: 0.0
                    },
                    &mut family.workspace()
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
            assert_relative_eq!(family.cdf(y, &theta), reference.cdf(y), epsilon = 1.0e-7);
        }

        for p in [0.01, 0.1, 0.5, 0.9, 0.99] {
            assert_relative_eq!(
                family.quantile(p, &theta),
                reference.inverse_cdf(p),
                epsilon = 1.0e-6
            );
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn log_normal_boundaries_and_crps_behave_like_kernel() {
        let family = LogNormalLogLocationLogSd::new();
        let theta = LogNormalLogLocationLogSdTheta {
            log_location: 0.0,
            log_sd: 1.0,
        };

        assert_eq!(family.cdf(0.0, &theta), 0.0);
        assert_eq!(family.cdf(-1.0, &theta), 0.0);
        assert_eq!(family.quantile(0.0, &theta), 0.0);
        assert_eq!(family.quantile(1.0, &theta), f64::INFINITY);
        assert!(family.quantile(f64::NAN, &theta).is_nan());
        assert!(family.crps(1.0, &theta).is_finite());
        assert!(family.crps(-1.0, &theta).is_nan());
    }

    #[cfg(feature = "rand")]
    #[test]
    fn log_normal_sampling_returns_finite_values_and_errors_for_invalid_theta() {
        use rand::SeedableRng;

        let family = LogNormalMeanLogSd::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family
            .try_sample(
                &mut rng,
                &LogNormalMeanLogSdTheta {
                    mean: 1.5,
                    log_sd: 0.8,
                },
            )
            .unwrap();
        assert!(sample > 0.0 && sample.is_finite());
    }
}
