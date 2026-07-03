use gamlss_core::{
    Cv, Family, InitialEtaFromObservations, InitialEtaFromTheta, Log, Mean, ObservationView,
    ParameterParts, PositiveLink, ScalarParams,
};

use crate::initial::positive_floor;

use super::{Gamma, GammaShapeRateTheta};

/// Gamma distribution parameterized by mean and coefficient of variation.
pub type GammaMeanCv = Gamma<MeanCv, Log, Log>;
/// Gamma mean/coefficient-of-variation parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeanCv;

/// Predictors for gamma mean/CV on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaMeanCvEta {
    /// Mean predictor.
    pub mean: f64,
    /// Coefficient-of-variation predictor.
    pub cv: f64,
}

impl ParameterParts<2> for GammaMeanCvEta {
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
            _ => unreachable!("gamma mean/CV eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale gamma mean/CV parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaMeanCvTheta {
    /// Positive mean.
    pub mean: f64,
    /// Positive coefficient of variation.
    pub cv: f64,
}

impl GammaMeanCvTheta {
    #[inline]
    pub(super) fn shape_rate(self) -> GammaShapeRateTheta {
        let shape = 1.0 / (self.cv * self.cv);
        GammaShapeRateTheta {
            shape,
            rate: shape / self.mean,
        }
    }
}

impl<MeanLink, CvLink> Gamma<MeanCv, MeanLink, CvLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
{
    #[inline]
    fn theta_from_eta(eta: GammaMeanCvEta) -> GammaMeanCvTheta {
        GammaMeanCvTheta {
            mean: MeanLink::inverse(eta.mean),
            cv: CvLink::inverse(eta.cv),
        }
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn nll_and_gradient_eta_values(y: f64, eta: GammaMeanCvEta) -> (f64, GammaMeanCvEta) {
        let theta = Self::theta_from_eta(eta);
        let shape_rate = theta.shape_rate();
        let nll = Self::nll_shape_rate(y, shape_rate);
        if !nll.is_finite() {
            return (nll, GammaMeanCvEta::from_array([f64::NAN; 2]));
        }

        let (d_shape, d_rate) = Self::gradient_shape_rate(y, shape_rate);
        let d_mean = d_rate * (-shape_rate.rate / theta.mean);
        let d_cv = d_shape * (-2.0 * shape_rate.shape / theta.cv)
            + d_rate * (-2.0 * shape_rate.rate / theta.cv);

        (
            nll,
            GammaMeanCvEta {
                mean: d_mean * MeanLink::derivative_inverse(eta.mean),
                cv: d_cv * CvLink::derivative_inverse(eta.cv),
            },
        )
    }
}

impl From<GammaMeanCvTheta> for GammaShapeRateTheta {
    #[inline]
    fn from(theta: GammaMeanCvTheta) -> Self {
        theta.shape_rate()
    }
}

impl<MeanLink, CvLink> Family for Gamma<MeanCv, MeanLink, CvLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
{
    type Eta = GammaMeanCvEta;
    type Theta = GammaMeanCvTheta;
    type GradientEta = GammaMeanCvEta;
    type Observation<'obs> = f64;
    type Workspace = ();
    type ParamSpec = ScalarParams<(Mean, Cv), (MeanLink, CvLink), 2>;

    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    #[inline]
    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        Self::nll_shape_rate(y, theta.shape_rate())
    }

    #[inline]
    fn nll_eta(&self, y: f64, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> f64 {
        Self::nll_shape_rate(y, Self::theta_from_eta(*eta).shape_rate())
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

impl<MeanLink, CvLink> InitialEtaFromObservations<2> for Gamma<MeanCv, MeanLink, CvLink>
where
    MeanLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    CvLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let Some((mean, shape)) = Self::initial_mean_shape(obs) else {
            return GammaMeanCvEta::from_array([0.0, 0.0]);
        };
        let cv = positive_floor(1.0 / shape.sqrt());

        GammaMeanCvEta {
            mean: MeanLink::initial_eta_from_theta(mean),
            cv: CvLink::initial_eta_from_theta(cv),
        }
    }
}
