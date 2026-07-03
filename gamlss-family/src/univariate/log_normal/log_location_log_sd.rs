use gamlss_core::{
    Family, InitialEtaFromObservations, InitialEtaFromTheta, Link, LogLocation, LogSd,
    ObservationView, ParameterParts, PositiveLink, ScalarParams,
};

use crate::initial::{positive_floor, robust_location_scale, weighted_values};

use super::{LogNormal, LogNormalMeanCvTheta, LogNormalMeanLogSdTheta, LogNormalMedianLogSdTheta};

/// Log-normal log-location/log-SD parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LogLocationLogSd;

/// Predictors for log-normal log-location/log-SD on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogNormalLogLocationLogSdEta {
    /// Log-location predictor.
    pub log_location: f64,
    /// Log-SD predictor.
    pub log_sd: f64,
}

impl ParameterParts<2> for LogNormalLogLocationLogSdEta {
    #[inline]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            log_location: values[0],
            log_sd: values[1],
        }
    }

    #[inline]
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
    #[inline]
    fn from(theta: LogNormalMeanLogSdTheta) -> Self {
        theta.log_location_log_sd()
    }
}

impl From<LogNormalMeanCvTheta> for LogNormalLogLocationLogSdTheta {
    #[inline]
    fn from(theta: LogNormalMeanCvTheta) -> Self {
        theta.log_location_log_sd()
    }
}

impl From<LogNormalMedianLogSdTheta> for LogNormalLogLocationLogSdTheta {
    #[inline]
    fn from(theta: LogNormalMedianLogSdTheta) -> Self {
        theta.log_location_log_sd()
    }
}
impl<LocationLink, LogSdLink> LogNormal<LogLocationLogSd, LocationLink, LogSdLink>
where
    LocationLink: Link<f64>,
    LogSdLink: PositiveLink<f64>,
{
    #[inline]
    fn theta_from_eta(eta: LogNormalLogLocationLogSdEta) -> LogNormalLogLocationLogSdTheta {
        LogNormalLogLocationLogSdTheta {
            log_location: LocationLink::inverse(eta.log_location),
            log_sd: LogSdLink::inverse(eta.log_sd),
        }
    }

    #[inline]
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

impl<LocationLink, LogSdLink> Family for LogNormal<LogLocationLogSd, LocationLink, LogSdLink>
where
    LocationLink: Link<f64>,
    LogSdLink: PositiveLink<f64>,
{
    type Eta = LogNormalLogLocationLogSdEta;
    type Theta = LogNormalLogLocationLogSdTheta;
    type GradientEta = LogNormalLogLocationLogSdEta;
    type Observation<'obs> = f64;
    type Workspace = ();
    type ParamSpec = ScalarParams<(LogLocation, LogSd), (LocationLink, LogSdLink), 2>;

    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    #[inline]
    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        Self::nll_log_location_log_sd(y, *theta)
    }

    #[inline]
    fn nll_eta(&self, y: f64, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> f64 {
        Self::nll_log_location_log_sd(y, Self::theta_from_eta(*eta))
    }

    #[inline]
    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        Self::nll_and_gradient_eta_values(y, *eta)
    }
}

impl<LocationLink, LogSdLink> InitialEtaFromObservations<2>
    for LogNormal<LogLocationLogSd, LocationLink, LogSdLink>
where
    LocationLink: InitialEtaFromTheta<f64> + Link<f64>,
    LogSdLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
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
