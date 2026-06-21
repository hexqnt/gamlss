use gamlss_core::{
    Family, InitialEtaFromTheta, LogSd, Mean, ObservationView, ParameterParts, ParameterizedFamily,
    PositiveLink,
};

use crate::initial::{positive_floor, robust_location_scale, weighted_values};

use super::{LogNormal, LogNormalLogLocationLogSdTheta};

/// Log-normal mean/log-SD parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeanLogSd;

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
    pub(super) fn log_location_log_sd(self) -> LogNormalLogLocationLogSdTheta {
        LogNormalLogLocationLogSdTheta {
            log_location: self.mean.ln() - 0.5 * self.log_sd * self.log_sd,
            log_sd: self.log_sd,
        }
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
