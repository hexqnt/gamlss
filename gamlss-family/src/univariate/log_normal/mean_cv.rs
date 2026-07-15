use gamlss_core::{
    Cv, Family, InitialEtaFromObservations, InitialEtaFromTheta, Mean, ObservationView,
    ParameterParts, PositiveLink,
};

use crate::initial::{positive_floor, weighted_summary, weighted_values};

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
        LogNormalLogLocationLogSdTheta {
            log_location: 0.5f64.mul_add(-log_sd_squared, self.mean.ln()),
            log_sd,
        }
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
    #[allow(clippy::suboptimal_flops)]
    fn nll_and_gradient_eta_values(y: f64, eta: LogNormalMeanCvEta) -> (f64, LogNormalMeanCvEta) {
        let theta = Self::theta_from_eta(eta);
        let canonical = theta.log_location_log_sd();
        let nll = Self::nll_log_location_log_sd(y, canonical);
        if !nll.is_finite() {
            return (nll, LogNormalMeanCvEta::from_array([f64::NAN; 2]));
        }

        let (d_location, d_log_sd) = Self::gradient_log_location_log_sd(y, canonical);
        let inverse_hypot = 1.0 / theta.cv.hypot(1.0);
        let cv_over_one_plus_cv2 = (theta.cv * inverse_hypot) * inverse_hypot;
        let d_mean = d_location / theta.mean;
        let d_cv = -d_location * cv_over_one_plus_cv2
            + d_log_sd * (cv_over_one_plus_cv2 / canonical.log_sd);

        (nll, eta.chain_gradient::<MeanLink, CvLink>(d_mean, d_cv))
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
        Self::nll_log_location_log_sd(y, Self::theta_from_eta(*eta).log_location_log_sd())
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
