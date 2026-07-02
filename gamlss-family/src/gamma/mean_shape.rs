use gamlss_core::{
    Family, InitialEtaFromObservations, InitialEtaFromTheta, Log, Mean, ObservationView,
    ParameterParts, PositiveLink, ScalarParams, Shape,
};

use super::{Gamma, GammaShapeRateTheta};

/// Gamma distribution parameterized by mean and shape.
pub type GammaMeanShape = Gamma<MeanShape, Log, Log>;
/// Gamma mean/shape parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeanShape;

/// Predictors for gamma mean/shape on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaMeanShapeEta {
    /// Mean predictor.
    pub mean: f64,
    /// Shape predictor.
    pub shape: f64,
}

impl ParameterParts<2> for GammaMeanShapeEta {
    #[inline]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            mean: values[0],
            shape: values[1],
        }
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mean,
            1 => self.shape,
            _ => unreachable!("gamma mean/shape eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale gamma mean/shape parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaMeanShapeTheta {
    /// Positive mean.
    pub mean: f64,
    /// Positive shape.
    pub shape: f64,
}

impl GammaMeanShapeTheta {
    #[inline]
    pub(super) fn shape_rate(self) -> GammaShapeRateTheta {
        GammaShapeRateTheta {
            shape: self.shape,
            rate: self.shape / self.mean,
        }
    }
}

impl<MeanLink, ShapeLink> Gamma<MeanShape, MeanLink, ShapeLink>
where
    MeanLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    #[inline]
    fn theta_from_eta(eta: GammaMeanShapeEta) -> GammaMeanShapeTheta {
        GammaMeanShapeTheta {
            mean: MeanLink::inverse(eta.mean),
            shape: ShapeLink::inverse(eta.shape),
        }
    }

    #[inline]
    fn nll_and_gradient_eta_values(y: f64, eta: GammaMeanShapeEta) -> (f64, GammaMeanShapeEta) {
        let theta = Self::theta_from_eta(eta);
        let shape_rate = theta.shape_rate();
        let nll = Self::nll_shape_rate(y, shape_rate);
        if !nll.is_finite() {
            return (nll, GammaMeanShapeEta::from_array([f64::NAN; 2]));
        }

        let (d_shape, d_rate) = Self::gradient_shape_rate(y, shape_rate);
        let d_mean = d_rate * (-theta.shape / (theta.mean * theta.mean));
        let d_shape_param = d_shape + d_rate / theta.mean;

        (
            nll,
            GammaMeanShapeEta {
                mean: d_mean * MeanLink::derivative_inverse(eta.mean),
                shape: d_shape_param * ShapeLink::derivative_inverse(eta.shape),
            },
        )
    }
}

impl From<GammaMeanShapeTheta> for GammaShapeRateTheta {
    #[inline]
    fn from(theta: GammaMeanShapeTheta) -> Self {
        theta.shape_rate()
    }
}

impl<MeanLink, ShapeLink> Family for Gamma<MeanShape, MeanLink, ShapeLink>
where
    MeanLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    type Eta = GammaMeanShapeEta;
    type Theta = GammaMeanShapeTheta;
    type GradientEta = GammaMeanShapeEta;
    type Observation<'obs> = f64;
    type Workspace = ();
    type ParamSpec = ScalarParams<(Mean, Shape), (MeanLink, ShapeLink), 2>;

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

impl<MeanLink, ShapeLink> InitialEtaFromObservations<2> for Gamma<MeanShape, MeanLink, ShapeLink>
where
    MeanLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    ShapeLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let Some((mean, shape)) = Self::initial_mean_shape(obs) else {
            return GammaMeanShapeEta::from_array([0.0, 0.0]);
        };

        GammaMeanShapeEta {
            mean: MeanLink::initial_eta_from_theta(mean),
            shape: ShapeLink::initial_eta_from_theta(shape),
        }
    }
}
