use gamlss_core::{
    Family, InitialEtaFromObservations, InitialEtaFromTheta, LogSd, Mean, ObservationView,
    ParameterParts, PositiveLink,
};

use crate::{
    initial::{positive_floor, robust_location_scale, weighted_values},
    link::positive_inverse_and_log,
};

use super::{LogNormal, LogNormalLogLocationLogSdTheta};

/// Log-normal mean/log-SD parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeanLogSd;

define_two_positive_parameter_blocks! {
    eta:
    /// Predictors for log-normal mean/log-SD on the link scale.
    LogNormalMeanLogSdEta {
        /// Mean predictor.
        mean,
        /// Log-SD predictor.
        log_sd,
    }
    theta:
    /// Natural-scale log-normal mean/log-SD parameters.
    LogNormalMeanLogSdTheta {
        /// Positive mean.
        mean,
        /// Positive standard deviation of `log(Y)`.
        log_sd,
    }
}

impl LogNormalMeanLogSdTheta {
    #[inline]
    #[allow(clippy::suboptimal_flops)]
    pub(super) fn log_location_log_sd(self) -> LogNormalLogLocationLogSdTheta {
        self.log_location_log_sd_with_log_mean(self.mean.ln())
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn log_location_log_sd_with_log_mean(self, log_mean: f64) -> LogNormalLogLocationLogSdTheta {
        LogNormalLogLocationLogSdTheta {
            log_location: log_mean - 0.5 * self.log_sd * self.log_sd,
            log_sd: self.log_sd,
        }
    }
}

impl<MeanLink, LogSdLink> LogNormal<MeanLogSd, MeanLink, LogSdLink>
where
    MeanLink: PositiveLink<f64>,
    LogSdLink: PositiveLink<f64>,
{
    #[inline]
    fn theta_from_eta(eta: LogNormalMeanLogSdEta) -> LogNormalMeanLogSdTheta {
        eta.theta_from_links::<MeanLink, LogSdLink>()
    }

    #[inline]
    fn valid_theta(theta: LogNormalMeanLogSdTheta) -> bool {
        theta.mean > 0.0 && theta.mean.is_finite() && theta.log_sd > 0.0 && theta.log_sd.is_finite()
    }

    #[inline]
    fn theta_canonical_and_log_scale_from_eta(
        eta: LogNormalMeanLogSdEta,
    ) -> (LogNormalMeanLogSdTheta, LogNormalLogLocationLogSdTheta, f64) {
        let (mean, log_mean) = positive_inverse_and_log::<MeanLink>(eta.mean);
        let (log_sd, log_scale) = positive_inverse_and_log::<LogSdLink>(eta.log_sd);
        let theta = LogNormalMeanLogSdTheta { mean, log_sd };
        (
            theta,
            theta.log_location_log_sd_with_log_mean(log_mean),
            log_scale,
        )
    }

    #[inline]
    fn nll_and_gradient_eta_values(
        y: f64,
        eta: LogNormalMeanLogSdEta,
    ) -> (f64, LogNormalMeanLogSdEta) {
        let (theta, canonical, log_scale) = Self::theta_canonical_and_log_scale_from_eta(eta);
        if !Self::valid_theta(theta) {
            return (
                f64::INFINITY,
                LogNormalMeanLogSdEta::from_array([f64::NAN; 2]),
            );
        }
        let nll = Self::nll_log_location_log_sd_with_log_scale(y, canonical, log_scale);
        if !nll.is_finite() {
            return (nll, LogNormalMeanLogSdEta::from_array([f64::NAN; 2]));
        }

        let (d_location, d_scale_kernel) = Self::gradient_log_location_log_sd(y, canonical);
        let d_scale = d_location.mul_add(-theta.log_sd, d_scale_kernel);

        (
            nll,
            LogNormalMeanLogSdEta {
                mean: d_location * MeanLink::derivative_log_inverse(eta.mean),
                log_sd: d_scale * theta.log_sd * LogSdLink::derivative_log_inverse(eta.log_sd),
            },
        )
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MeanLink, LogSdLink> for LogNormal<MeanLogSd, MeanLink, LogSdLink>;
    parameters = (Mean, LogSd);
    arity = 2;
);

impl<MeanLink, LogSdLink> Family for LogNormal<MeanLogSd, MeanLink, LogSdLink>
where
    MeanLink: PositiveLink<f64>,
    LogSdLink: PositiveLink<f64>,
{
    type Eta = LogNormalMeanLogSdEta;
    type Theta = LogNormalMeanLogSdTheta;
    type GradientEta = LogNormalMeanLogSdEta;
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

impl<MeanLink, LogSdLink> InitialEtaFromObservations<2>
    for LogNormal<MeanLogSd, MeanLink, LogSdLink>
where
    MeanLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    LogSdLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    #[allow(clippy::suboptimal_flops)]
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
