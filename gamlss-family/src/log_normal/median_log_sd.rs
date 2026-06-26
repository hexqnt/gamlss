use gamlss_core::{
    Family, InitialEtaFromTheta, LogSd, Median, ObservationView, ParameterParts,
    ParameterizedFamily, PositiveLink,
};

use crate::initial::{positive_floor, robust_location_scale, weighted_values};

use super::{LogNormal, LogNormalLogLocationLogSdTheta};

/// Log-normal median/log-SD parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MedianLogSd;

/// Predictors for log-normal median/log-SD on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogNormalMedianLogSdEta {
    /// Median predictor.
    pub median: f64,
    /// Log-SD predictor.
    pub log_sd: f64,
}

impl ParameterParts<2> for LogNormalMedianLogSdEta {
    #[inline]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            median: values[0],
            log_sd: values[1],
        }
    }

    #[inline]
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
    #[inline]
    pub(super) fn log_location_log_sd(self) -> LogNormalLogLocationLogSdTheta {
        LogNormalLogLocationLogSdTheta {
            log_location: self.median.ln(),
            log_sd: self.log_sd,
        }
    }
}

impl<MedianLink, LogSdLink> LogNormal<MedianLogSd, MedianLink, LogSdLink>
where
    MedianLink: PositiveLink<f64>,
    LogSdLink: PositiveLink<f64>,
{
    #[inline]
    fn theta_from_eta(eta: LogNormalMedianLogSdEta) -> LogNormalMedianLogSdTheta {
        LogNormalMedianLogSdTheta {
            median: MedianLink::inverse(eta.median),
            log_sd: LogSdLink::inverse(eta.log_sd),
        }
    }

    #[inline]
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

impl<MedianLink, LogSdLink> Family for LogNormal<MedianLogSd, MedianLink, LogSdLink>
where
    MedianLink: PositiveLink<f64>,
    LogSdLink: PositiveLink<f64>,
{
    type Eta = LogNormalMedianLogSdEta;
    type Theta = LogNormalMedianLogSdTheta;
    type NllGradientEta = LogNormalMedianLogSdEta;
    type Observation<'obs> = f64;

    #[inline]
    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline]
    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::nll_log_location_log_sd(y, theta.log_location_log_sd())
    }

    #[inline]
    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        Self::nll_log_location_log_sd(y, Self::theta_from_eta(eta).log_location_log_sd())
    }

    #[inline]
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
