use std::marker::PhantomData;

use gamlss_core::{
    Cv, Family, HasCdf, HasCrps, HasQuantile, InitialEtaFromObservations, InitialEtaFromTheta, Log,
    Mean, ObservationView, ParameterParts, PositiveLink, ScalarParams,
};

use crate::initial::{
    LARGE_SHAPE, VARIANCE_FLOOR, positive_floor, weighted_summary, weighted_values,
};

use super::{InverseGaussian, InverseGaussianTheta};

/// Inverse Gaussian distribution with log links for mean and coefficient of variation.
pub type InverseGaussianMeanCv = InverseGaussianCv<Log, Log>;

/// Predictors for inverse Gaussian mean/CV on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InverseGaussianMeanCvEta {
    /// Mean predictor.
    pub mean: f64,
    /// Coefficient-of-variation predictor.
    pub cv: f64,
}

impl ParameterParts<2> for InverseGaussianMeanCvEta {
    #[inline]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            mean: values[0],
            cv: values[1],
        }
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mean,
            1 => self.cv,
            _ => unreachable!("inverse Gaussian mean/CV eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale inverse Gaussian mean/CV parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InverseGaussianMeanCvTheta {
    /// Positive mean.
    pub mean: f64,
    /// Positive coefficient of variation.
    pub cv: f64,
}

impl InverseGaussianMeanCvTheta {
    #[inline]
    fn mean_shape(self) -> InverseGaussianTheta {
        InverseGaussianTheta {
            mu: self.mean,
            shape: self.mean / (self.cv * self.cv),
        }
    }
}

/// Inverse Gaussian mean/CV implementation carrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InverseGaussianCv<MeanLink = Log, CvLink = Log> {
    marker: PhantomData<(MeanLink, CvLink)>,
}

impl<MeanLink, CvLink> InverseGaussianCv<MeanLink, CvLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
{
    /// Creates a stateless inverse Gaussian mean/CV family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(eta: InverseGaussianMeanCvEta) -> InverseGaussianMeanCvTheta {
        InverseGaussianMeanCvTheta {
            mean: MeanLink::inverse(eta.mean),
            cv: CvLink::inverse(eta.cv),
        }
    }
}

impl<MeanLink, CvLink> Default for InverseGaussianCv<MeanLink, CvLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<MeanLink, CvLink> Family for InverseGaussianCv<MeanLink, CvLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
{
    type Eta = InverseGaussianMeanCvEta;
    type Theta = InverseGaussianMeanCvTheta;
    type GradientEta = InverseGaussianMeanCvEta;
    type Observation<'obs> = f64;
    type Workspace = ();
    type ParamSpec = ScalarParams<(Mean, Cv), (MeanLink, CvLink), 2>;
    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        InverseGaussian::<Log, Log>::nll_theta(y, theta.mean_shape())
    }

    fn nll_eta(&self, y: f64, eta: &Self::Eta, workspace: &mut Self::Workspace) -> f64 {
        let theta = Self::theta_from_eta(*eta);
        self.nll(y, &theta, workspace)
    }

    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        let theta = Self::theta_from_eta(*eta);
        let mean_shape = theta.mean_shape();
        let nll = InverseGaussian::<Log, Log>::nll_theta(y, mean_shape);
        if !nll.is_finite() {
            return (nll, InverseGaussianMeanCvEta::from_array([f64::NAN; 2]));
        }

        let residual = y - mean_shape.mu;
        let d_mu = -mean_shape.shape * residual / (mean_shape.mu * mean_shape.mu * mean_shape.mu);
        let d_shape = -0.5 / mean_shape.shape
            + residual * residual / (2.0 * mean_shape.mu * mean_shape.mu * y);
        let cv2 = theta.cv * theta.cv;
        let d_mean = d_mu + d_shape / cv2;
        let d_cv = d_shape * (-2.0 * theta.mean / (cv2 * theta.cv));

        (
            nll,
            InverseGaussianMeanCvEta {
                mean: d_mean * MeanLink::derivative_inverse(eta.mean),
                cv: d_cv * CvLink::derivative_inverse(eta.cv),
            },
        )
    }
}

impl<MeanLink, CvLink> InitialEtaFromObservations<2> for InverseGaussianCv<MeanLink, CvLink>
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
            return InverseGaussianMeanCvEta::from_array([0.0, 0.0]);
        };

        let mean = positive_floor(summary.mean);
        let cv = if summary.variance <= VARIANCE_FLOOR {
            positive_floor((mean / LARGE_SHAPE).sqrt())
        } else {
            positive_floor((summary.variance / (mean * mean)).sqrt())
        };

        InverseGaussianMeanCvEta {
            mean: MeanLink::initial_eta_from_theta(mean),
            cv: CvLink::initial_eta_from_theta(cv),
        }
    }
}

impl<MeanLink, CvLink> HasCdf for InverseGaussianCv<MeanLink, CvLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        InverseGaussian::<Log, Log>::new().cdf(y, &theta.mean_shape())
    }
}

impl<MeanLink, CvLink> HasQuantile for InverseGaussianCv<MeanLink, CvLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
{
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        InverseGaussian::<Log, Log>::new().quantile(p, &theta.mean_shape())
    }
}

impl<MeanLink, CvLink> HasCrps for InverseGaussianCv<MeanLink, CvLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
{
    fn crps(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        InverseGaussian::<Log, Log>::new().crps(y, &theta.mean_shape())
    }
}
