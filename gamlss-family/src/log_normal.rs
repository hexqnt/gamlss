use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Family, HasCdf, HasCrps, HasQuantile, Identity, InitialEtaFromTheta, Link, Log, LogLocation,
    LogSd, Mean, Median, ObservationView, ParameterParts, ParameterizedFamily, PositiveLink,
};

use crate::initial::{positive_floor, robust_location_scale, weighted_values};
use crate::special::{unit_normal_cdf, unit_normal_quantile};

const HALF_LOG_2_PI: f64 = 0.918_938_533_204_672_7;

/// Log-normal mean/log-SD parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeanLogSd;

/// Log-normal median/log-SD parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MedianLogSd;

/// Log-normal log-location/log-SD parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LogLocationLogSd;

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

impl<MeanLink, LogSdLink> LogNormal<MeanLogSd, MeanLink, LogSdLink>
where
    MeanLink: PositiveLink<f64>,
    LogSdLink: PositiveLink<f64>,
{
    #[inline(always)]
    fn theta_from_eta(eta: LogNormalMeanLogSdEta) -> LogNormalMeanLogSdTheta {
        LogNormalMeanLogSdTheta {
            mean: MeanLink::inverse(eta.mean),
            log_sd: LogSdLink::inverse(eta.log_sd),
        }
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(
        y: f64,
        eta: LogNormalMeanLogSdEta,
    ) -> (f64, LogNormalMeanLogSdEta) {
        let theta = Self::theta_from_eta(eta);
        let canonical = theta.log_location_log_sd();
        let nll = Self::nll_log_location_log_sd(y, canonical);
        if !nll.is_finite() {
            return (nll, LogNormalMeanLogSdEta::from_array([f64::NAN; 2]));
        }

        let (d_location, d_log_sd_kernel) = Self::gradient_log_location_log_sd(y, canonical);
        let d_mean = d_location / theta.mean;
        let d_log_sd = d_log_sd_kernel - d_location * theta.log_sd;

        (
            nll,
            LogNormalMeanLogSdEta {
                mean: d_mean * MeanLink::derivative_inverse(eta.mean),
                log_sd: d_log_sd * LogSdLink::derivative_inverse(eta.log_sd),
            },
        )
    }
}

impl<MedianLink, LogSdLink> LogNormal<MedianLogSd, MedianLink, LogSdLink>
where
    MedianLink: PositiveLink<f64>,
    LogSdLink: PositiveLink<f64>,
{
    #[inline(always)]
    fn theta_from_eta(eta: LogNormalMedianLogSdEta) -> LogNormalMedianLogSdTheta {
        LogNormalMedianLogSdTheta {
            median: MedianLink::inverse(eta.median),
            log_sd: LogSdLink::inverse(eta.log_sd),
        }
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(
        y: f64,
        eta: LogNormalMedianLogSdEta,
    ) -> (f64, LogNormalMedianLogSdEta) {
        let theta = Self::theta_from_eta(eta);
        let canonical = theta.log_location_log_sd();
        let nll = Self::nll_log_location_log_sd(y, canonical);
        if !nll.is_finite() {
            return (nll, LogNormalMedianLogSdEta::from_array([f64::NAN; 2]));
        }

        let (d_location, d_log_sd) = Self::gradient_log_location_log_sd(y, canonical);
        (
            nll,
            LogNormalMedianLogSdEta {
                median: d_location / theta.median * MedianLink::derivative_inverse(eta.median),
                log_sd: d_log_sd * LogSdLink::derivative_inverse(eta.log_sd),
            },
        )
    }
}

impl<LocationLink, LogSdLink> LogNormal<LogLocationLogSd, LocationLink, LogSdLink>
where
    LocationLink: Link<f64>,
    LogSdLink: PositiveLink<f64>,
{
    #[inline(always)]
    fn theta_from_eta(eta: LogNormalLogLocationLogSdEta) -> LogNormalLogLocationLogSdTheta {
        LogNormalLogLocationLogSdTheta {
            log_location: LocationLink::inverse(eta.log_location),
            log_sd: LogSdLink::inverse(eta.log_sd),
        }
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(
        y: f64,
        eta: LogNormalLogLocationLogSdEta,
    ) -> (f64, LogNormalLogLocationLogSdEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_log_location_log_sd(y, theta);
        if !nll.is_finite() {
            return (nll, LogNormalLogLocationLogSdEta::from_array([f64::NAN; 2]));
        }

        let (d_location, d_log_sd) = Self::gradient_log_location_log_sd(y, theta);
        (
            nll,
            LogNormalLogLocationLogSdEta {
                log_location: d_location * LocationLink::derivative_inverse(eta.log_location),
                log_sd: d_log_sd * LogSdLink::derivative_inverse(eta.log_sd),
            },
        )
    }
}

impl<Param, FirstLink, SecondLink> Default for LogNormal<Param, FirstLink, SecondLink> {
    fn default() -> Self {
        Self::new()
    }
}

impl<MeanLink, LogSdLink> Family for LogNormal<MeanLogSd, MeanLink, LogSdLink>
where
    MeanLink: PositiveLink<f64>,
    LogSdLink: PositiveLink<f64>,
{
    type Eta = LogNormalMeanLogSdEta;
    type Theta = LogNormalMeanLogSdTheta;
    type NllGradientEta = LogNormalMeanLogSdEta;
    type Observation<'obs> = f64;

    #[inline(always)]
    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline(always)]
    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::nll_log_location_log_sd(y, theta.log_location_log_sd())
    }

    #[inline(always)]
    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        Self::nll_log_location_log_sd(y, Self::theta_from_eta(eta).log_location_log_sd())
    }

    #[inline(always)]
    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        Self::nll_and_gradient_eta_values(y, eta)
    }
}

impl<MeanLink, LogSdLink> ParameterizedFamily<2> for LogNormal<MeanLogSd, MeanLink, LogSdLink>
where
    MeanLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    LogSdLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Params = (Mean, LogSd);
    type Links = (MeanLink, LogSdLink);

    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values =
            weighted_values::<Self, _, _>(obs, |y| (y.is_finite() && y > 0.0).then_some(y.ln()));
        let Some((log_location, log_sd)) = robust_location_scale(&values) else {
            return LogNormalMeanLogSdEta::from_array([0.0, 0.0]);
        };
        let log_sd = positive_floor(log_sd);
        let mean = (log_location + 0.5 * log_sd * log_sd).exp();

        LogNormalMeanLogSdEta {
            mean: MeanLink::initial_eta_from_theta(mean),
            log_sd: LogSdLink::initial_eta_from_theta(log_sd),
        }
    }
}

impl<MedianLink, LogSdLink> Family for LogNormal<MedianLogSd, MedianLink, LogSdLink>
where
    MedianLink: PositiveLink<f64>,
    LogSdLink: PositiveLink<f64>,
{
    type Eta = LogNormalMedianLogSdEta;
    type Theta = LogNormalMedianLogSdTheta;
    type NllGradientEta = LogNormalMedianLogSdEta;
    type Observation<'obs> = f64;

    #[inline(always)]
    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline(always)]
    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::nll_log_location_log_sd(y, theta.log_location_log_sd())
    }

    #[inline(always)]
    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        Self::nll_log_location_log_sd(y, Self::theta_from_eta(eta).log_location_log_sd())
    }

    #[inline(always)]
    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        Self::nll_and_gradient_eta_values(y, eta)
    }
}

impl<MedianLink, LogSdLink> ParameterizedFamily<2> for LogNormal<MedianLogSd, MedianLink, LogSdLink>
where
    MedianLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    LogSdLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Params = (Median, LogSd);
    type Links = (MedianLink, LogSdLink);

    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values =
            weighted_values::<Self, _, _>(obs, |y| (y.is_finite() && y > 0.0).then_some(y.ln()));
        let Some((log_location, log_sd)) = robust_location_scale(&values) else {
            return LogNormalMedianLogSdEta::from_array([0.0, 0.0]);
        };

        LogNormalMedianLogSdEta {
            median: MedianLink::initial_eta_from_theta(log_location.exp()),
            log_sd: LogSdLink::initial_eta_from_theta(positive_floor(log_sd)),
        }
    }
}

impl<LocationLink, LogSdLink> Family for LogNormal<LogLocationLogSd, LocationLink, LogSdLink>
where
    LocationLink: Link<f64>,
    LogSdLink: PositiveLink<f64>,
{
    type Eta = LogNormalLogLocationLogSdEta;
    type Theta = LogNormalLogLocationLogSdTheta;
    type NllGradientEta = LogNormalLogLocationLogSdEta;
    type Observation<'obs> = f64;

    #[inline(always)]
    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline(always)]
    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::nll_log_location_log_sd(y, theta)
    }

    #[inline(always)]
    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        Self::nll_log_location_log_sd(y, Self::theta_from_eta(eta))
    }

    #[inline(always)]
    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        Self::nll_and_gradient_eta_values(y, eta)
    }
}

impl<LocationLink, LogSdLink> ParameterizedFamily<2>
    for LogNormal<LogLocationLogSd, LocationLink, LogSdLink>
where
    LocationLink: InitialEtaFromTheta<f64> + Link<f64>,
    LogSdLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Params = (LogLocation, LogSd);
    type Links = (LocationLink, LogSdLink);

    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values =
            weighted_values::<Self, _, _>(obs, |y| (y.is_finite() && y > 0.0).then_some(y.ln()));
        let Some((log_location, log_sd)) = robust_location_scale(&values) else {
            return LogNormalLogLocationLogSdEta::from_array([0.0, 0.0]);
        };

        LogNormalLogLocationLogSdEta {
            log_location: LocationLink::initial_eta_from_theta(log_location),
            log_sd: LogSdLink::initial_eta_from_theta(positive_floor(log_sd)),
        }
    }
}

/// Predictors for log-normal mean/log-SD on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogNormalMeanLogSdEta {
    /// Mean predictor.
    pub mean: f64,
    /// Log-SD predictor.
    pub log_sd: f64,
}

impl ParameterParts<2> for LogNormalMeanLogSdEta {
    #[inline(always)]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            mean: values[0],
            log_sd: values[1],
        }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mean,
            1 => self.log_sd,
            _ => unreachable!("log-normal mean/log-SD eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale log-normal mean/log-SD parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogNormalMeanLogSdTheta {
    /// Positive mean.
    pub mean: f64,
    /// Positive standard deviation of `log(Y)`.
    pub log_sd: f64,
}

impl LogNormalMeanLogSdTheta {
    #[inline(always)]
    fn log_location_log_sd(self) -> LogNormalLogLocationLogSdTheta {
        LogNormalLogLocationLogSdTheta {
            log_location: self.mean.ln() - 0.5 * self.log_sd * self.log_sd,
            log_sd: self.log_sd,
        }
    }
}

/// Predictors for log-normal median/log-SD on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogNormalMedianLogSdEta {
    /// Median predictor.
    pub median: f64,
    /// Log-SD predictor.
    pub log_sd: f64,
}

impl ParameterParts<2> for LogNormalMedianLogSdEta {
    #[inline(always)]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            median: values[0],
            log_sd: values[1],
        }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.median,
            1 => self.log_sd,
            _ => unreachable!("log-normal median/log-SD eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale log-normal median/log-SD parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogNormalMedianLogSdTheta {
    /// Positive median.
    pub median: f64,
    /// Positive standard deviation of `log(Y)`.
    pub log_sd: f64,
}

impl LogNormalMedianLogSdTheta {
    #[inline(always)]
    fn log_location_log_sd(self) -> LogNormalLogLocationLogSdTheta {
        LogNormalLogLocationLogSdTheta {
            log_location: self.median.ln(),
            log_sd: self.log_sd,
        }
    }
}

/// Predictors for log-normal log-location/log-SD on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogNormalLogLocationLogSdEta {
    /// Log-location predictor.
    pub log_location: f64,
    /// Log-SD predictor.
    pub log_sd: f64,
}

impl ParameterParts<2> for LogNormalLogLocationLogSdEta {
    #[inline(always)]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            log_location: values[0],
            log_sd: values[1],
        }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.log_location,
            1 => self.log_sd,
            _ => unreachable!("log-normal log-location/log-SD eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale log-normal log-location/log-SD parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogNormalLogLocationLogSdTheta {
    /// Location parameter for `log(Y)`.
    pub log_location: f64,
    /// Positive standard deviation of `log(Y)`.
    pub log_sd: f64,
}

impl From<LogNormalMeanLogSdTheta> for LogNormalLogLocationLogSdTheta {
    #[inline(always)]
    fn from(theta: LogNormalMeanLogSdTheta) -> Self {
        theta.log_location_log_sd()
    }
}

impl From<LogNormalMedianLogSdTheta> for LogNormalLogLocationLogSdTheta {
    #[inline(always)]
    fn from(theta: LogNormalMedianLogSdTheta) -> Self {
        theta.log_location_log_sd()
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
impl_log_normal_helpers!(MedianLogSd, MedianLink, LogSdLink);
impl_log_normal_helpers!(LogLocationLogSd, LocationLink, LogSdLink);

/// Log-normal distribution parameterized by mean and log standard deviation.
pub type LogNormalMeanLogSd = LogNormal<MeanLogSd, Log, Log>;
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
        LogNormalLogLocationLogSd, LogNormalLogLocationLogSdTheta, LogNormalMeanLogSd,
        LogNormalMeanLogSdTheta, LogNormalMedianLogSd,
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
