use gamlss_core::{
    Cv, Family, InitialEtaFromTheta, Mean, ObservationView, ParameterParts, ParameterizedFamily,
    PositiveLink,
};

use crate::initial::{positive_floor, weighted_summary, weighted_values};

use super::{LogNormal, LogNormalLogLocationLogSdTheta};

/// Log-normal mean/CV parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeanCv;

/// Predictors for log-normal mean/CV on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogNormalMeanCvEta {
    /// Mean predictor.
    pub mean: f64,
    /// Coefficient-of-variation predictor.
    pub cv: f64,
}

impl ParameterParts<2> for LogNormalMeanCvEta {
    #[inline(always)]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            mean: values[0],
            cv: values[1],
        }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mean,
            1 => self.cv,
            _ => unreachable!("log-normal mean/CV eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale log-normal mean/CV parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogNormalMeanCvTheta {
    /// Positive mean.
    pub mean: f64,
    /// Positive coefficient of variation.
    pub cv: f64,
}

impl LogNormalMeanCvTheta {
    #[inline(always)]
    pub(super) fn log_location_log_sd(self) -> LogNormalLogLocationLogSdTheta {
        let log_sd_squared = (self.cv * self.cv).ln_1p();
        LogNormalLogLocationLogSdTheta {
            log_location: self.mean.ln() - 0.5 * log_sd_squared,
            log_sd: log_sd_squared.sqrt(),
        }
    }
}

impl<MeanLink, CvLink> LogNormal<MeanCv, MeanLink, CvLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
{
    #[inline(always)]
    fn theta_from_eta(eta: LogNormalMeanCvEta) -> LogNormalMeanCvTheta {
        LogNormalMeanCvTheta {
            mean: MeanLink::inverse(eta.mean),
            cv: CvLink::inverse(eta.cv),
        }
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: LogNormalMeanCvEta) -> (f64, LogNormalMeanCvEta) {
        let theta = Self::theta_from_eta(eta);
        let canonical = theta.log_location_log_sd();
        let nll = Self::nll_log_location_log_sd(y, canonical);
        if !nll.is_finite() {
            return (nll, LogNormalMeanCvEta::from_array([f64::NAN; 2]));
        }

        let (d_location, d_log_sd) = Self::gradient_log_location_log_sd(y, canonical);
        let cv2_plus_one = theta.cv.mul_add(theta.cv, 1.0);
        let d_mean = d_location / theta.mean;
        let d_cv = d_location * (-theta.cv / cv2_plus_one)
            + d_log_sd * (theta.cv / (cv2_plus_one * canonical.log_sd));

        (
            nll,
            LogNormalMeanCvEta {
                mean: d_mean * MeanLink::derivative_inverse(eta.mean),
                cv: d_cv * CvLink::derivative_inverse(eta.cv),
            },
        )
    }
}

impl<MeanLink, CvLink> Family for LogNormal<MeanCv, MeanLink, CvLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
{
    type Eta = LogNormalMeanCvEta;
    type Theta = LogNormalMeanCvTheta;
    type NllGradientEta = LogNormalMeanCvEta;
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

impl<MeanLink, CvLink> ParameterizedFamily<2> for LogNormal<MeanCv, MeanLink, CvLink>
where
    MeanLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    CvLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Params = (Mean, Cv);
    type Links = (MeanLink, CvLink);

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
