use gamlss_core::{
    Family, InitialEtaFromObservations, InitialEtaFromTheta, LogSd, Median, ObservationView,
    ParameterParts, PositiveLink,
};

use crate::{
    initial::{positive_floor, robust_location_scale, weighted_values},
    link::positive_inverse_and_log,
};

use super::{LogNormal, LogNormalLogLocationLogSdTheta};

/// Log-normal median/log-SD parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MedianLogSd;

define_two_positive_parameter_blocks! {
    eta:
    /// Predictors for log-normal median/log-SD on the link scale.
    LogNormalMedianLogSdEta {
        /// Median predictor.
        median,
        /// Log-SD predictor.
        log_sd,
    }
    theta:
    /// Natural-scale log-normal median/log-SD parameters.
    LogNormalMedianLogSdTheta {
        /// Positive median.
        median,
        /// Positive standard deviation of `log(Y)`.
        log_sd,
    }
}

impl LogNormalMedianLogSdTheta {
    #[inline]
    pub(super) fn log_location_log_sd(self) -> LogNormalLogLocationLogSdTheta {
        self.log_location_log_sd_with_log_median(self.median.ln())
    }

    #[inline]
    const fn log_location_log_sd_with_log_median(
        self,
        log_median: f64,
    ) -> LogNormalLogLocationLogSdTheta {
        LogNormalLogLocationLogSdTheta {
            log_location: log_median,
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
        eta.theta_from_links::<MedianLink, LogSdLink>()
    }

    #[inline]
    fn valid_theta(theta: LogNormalMedianLogSdTheta) -> bool {
        theta.median > 0.0
            && theta.median.is_finite()
            && theta.log_sd > 0.0
            && theta.log_sd.is_finite()
    }

    #[inline]
    fn theta_canonical_and_log_scale_from_eta(
        eta: LogNormalMedianLogSdEta,
    ) -> (
        LogNormalMedianLogSdTheta,
        LogNormalLogLocationLogSdTheta,
        f64,
    ) {
        let (median, log_median) = positive_inverse_and_log::<MedianLink>(eta.median);
        let (log_sd, log_scale) = positive_inverse_and_log::<LogSdLink>(eta.log_sd);
        let theta = LogNormalMedianLogSdTheta { median, log_sd };
        (
            theta,
            theta.log_location_log_sd_with_log_median(log_median),
            log_scale,
        )
    }

    #[inline]
    fn nll_and_gradient_eta_values(
        y: f64,
        eta: LogNormalMedianLogSdEta,
    ) -> (f64, LogNormalMedianLogSdEta) {
        let (theta, canonical, log_scale) = Self::theta_canonical_and_log_scale_from_eta(eta);
        if !Self::valid_theta(theta) {
            return (
                f64::INFINITY,
                LogNormalMedianLogSdEta::from_array([f64::NAN; 2]),
            );
        }
        let nll = Self::nll_log_location_log_sd_with_log_scale(y, canonical, log_scale);
        if !nll.is_finite() {
            return (nll, LogNormalMedianLogSdEta::from_array([f64::NAN; 2]));
        }

        let (d_location, d_scale) = Self::gradient_log_location_log_sd(y, canonical);
        (
            nll,
            LogNormalMedianLogSdEta {
                median: d_location * MedianLink::derivative_log_inverse(eta.median),
                log_sd: d_scale * theta.log_sd * LogSdLink::derivative_log_inverse(eta.log_sd),
            },
        )
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MedianLink, LogSdLink> for LogNormal<MedianLogSd, MedianLink, LogSdLink>;
    parameters = (Median, LogSd);
    arity = 2;
);

impl<MedianLink, LogSdLink> Family for LogNormal<MedianLogSd, MedianLink, LogSdLink>
where
    MedianLink: PositiveLink<f64>,
    LogSdLink: PositiveLink<f64>,
{
    type Eta = LogNormalMedianLogSdEta;
    type Theta = LogNormalMedianLogSdTheta;
    type GradientEta = LogNormalMedianLogSdEta;
    type Observation<'obs> = f64;
    type Workspace = ();

    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    #[inline]
    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        Self::nll_log_location_log_sd(y, theta.log_location_log_sd())
    }

    #[inline]
    fn nll_eta(&self, y: f64, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> f64 {
        let (theta, canonical, log_scale) = Self::theta_canonical_and_log_scale_from_eta(*eta);
        if !Self::valid_theta(theta) {
            return f64::INFINITY;
        }
        Self::nll_log_location_log_sd_with_log_scale(y, canonical, log_scale)
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

impl<MedianLink, LogSdLink> InitialEtaFromObservations<2>
    for LogNormal<MedianLogSd, MedianLink, LogSdLink>
where
    MedianLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    LogSdLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
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
