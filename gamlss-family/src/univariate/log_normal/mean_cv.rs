use gamlss_core::{
    Cv, Family, InitialEtaFromObservations, InitialEtaFromTheta, Mean, ObservationView,
    ParameterParts, PositiveLink,
};

use crate::{
    initial::{positive_floor, weighted_summary, weighted_values},
    link::positive_inverse_and_log,
};

use super::{LogNormal, LogNormalLogLocationLogSdTheta};

/// Log-normal mean/CV parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeanCv;

define_two_positive_parameter_blocks! {
    eta:
    /// Predictors for log-normal mean/CV on the link scale.
    LogNormalMeanCvEta {
        /// Mean predictor.
        mean,
        /// Coefficient-of-variation predictor.
        cv,
    }
    theta:
    /// Natural-scale log-normal mean/CV parameters.
    LogNormalMeanCvTheta {
        /// Positive mean.
        mean,
        /// Positive coefficient of variation.
        cv,
    }
}

impl LogNormalMeanCvTheta {
    #[inline]
    pub(super) fn log_location_log_sd(self) -> LogNormalLogLocationLogSdTheta {
        self.log_location_log_sd_with_logs(self.mean.ln(), self.cv.ln())
            .0
    }

    #[inline]
    fn log_location_log_sd_with_logs(
        self,
        log_mean: f64,
        log_cv: f64,
    ) -> (LogNormalLogLocationLogSdTheta, f64) {
        let log_sd_squared = if self.cv <= 1.0 {
            (self.cv * self.cv).ln_1p()
        } else {
            2.0 * self.cv.hypot(1.0).ln()
        };
        let log_sd = if log_sd_squared == 0.0 {
            self.cv
        } else {
            log_sd_squared.sqrt()
        };
        let log_scale = if log_sd_squared == 0.0 {
            log_cv
        } else {
            0.5 * log_sd_squared.ln()
        };
        (
            LogNormalLogLocationLogSdTheta {
                log_location: 0.5f64.mul_add(-log_sd_squared, log_mean),
                log_sd,
            },
            log_scale,
        )
    }
}

impl<MeanLink, CvLink> LogNormal<MeanCv, MeanLink, CvLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
{
    #[inline]
    fn theta_from_eta(eta: LogNormalMeanCvEta) -> LogNormalMeanCvTheta {
        eta.theta_from_links::<MeanLink, CvLink>()
    }

    #[inline]
    fn valid_theta(theta: LogNormalMeanCvTheta) -> bool {
        theta.mean > 0.0 && theta.mean.is_finite() && theta.cv > 0.0 && theta.cv.is_finite()
    }

    #[inline]
    fn theta_canonical_and_log_scale_from_eta(
        eta: LogNormalMeanCvEta,
    ) -> (LogNormalMeanCvTheta, LogNormalLogLocationLogSdTheta, f64) {
        let (mean, log_mean) = positive_inverse_and_log::<MeanLink>(eta.mean);
        let (cv, log_cv) = positive_inverse_and_log::<CvLink>(eta.cv);
        let theta = LogNormalMeanCvTheta { mean, cv };
        let (canonical, log_scale) = theta.log_location_log_sd_with_logs(log_mean, log_cv);
        (theta, canonical, log_scale)
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn nll_and_gradient_eta_values(y: f64, eta: LogNormalMeanCvEta) -> (f64, LogNormalMeanCvEta) {
        let (theta, canonical, log_scale) = Self::theta_canonical_and_log_scale_from_eta(eta);
        if !Self::valid_theta(theta) {
            return (f64::INFINITY, LogNormalMeanCvEta::from_array([f64::NAN; 2]));
        }
        let nll = Self::nll_log_location_log_sd_with_log_scale(y, canonical, log_scale);
        if !nll.is_finite() {
            return (nll, LogNormalMeanCvEta::from_array([f64::NAN; 2]));
        }

        let (d_location, d_log_sd) = Self::gradient_log_location_log_sd(y, canonical);
        let inverse_hypot = 1.0 / theta.cv.hypot(1.0);
        let cv_over_one_plus_cv2 = (theta.cv * inverse_hypot) * inverse_hypot;
        let d_cv = -d_location * cv_over_one_plus_cv2
            + d_log_sd * (cv_over_one_plus_cv2 / canonical.log_sd);

        (
            nll,
            LogNormalMeanCvEta {
                mean: d_location * MeanLink::derivative_log_inverse(eta.mean),
                cv: d_cv * theta.cv * CvLink::derivative_log_inverse(eta.cv),
            },
        )
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MeanLink, CvLink> for LogNormal<MeanCv, MeanLink, CvLink>;
    parameters = (Mean, Cv);
    arity = 2;
);

impl<MeanLink, CvLink> Family for LogNormal<MeanCv, MeanLink, CvLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
{
    type Eta = LogNormalMeanCvEta;
    type Theta = LogNormalMeanCvTheta;
    type GradientEta = LogNormalMeanCvEta;
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

impl<MeanLink, CvLink> InitialEtaFromObservations<2> for LogNormal<MeanCv, MeanLink, CvLink>
where
    MeanLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    CvLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values =
            weighted_values::<Self, _, _>(obs, |y| (y.is_finite() && y > 0.0).then_some(y));
        let Some(summary) = weighted_summary(&values) else {
            return LogNormalMeanCvEta::from_array([0.0, 0.0]);
        };
        let mean = positive_floor(summary.mean);
        let cv = positive_floor(summary.variance.sqrt() / mean);

        LogNormalMeanCvEta {
            mean: MeanLink::initial_eta_from_theta(mean),
            cv: CvLink::initial_eta_from_theta(cv),
        }
    }
}
