use gamlss_core::{
    Family, InitialEtaFromObservations, InitialEtaFromTheta, Log, Mean, ObservationView,
    ParameterParts, PositiveLink, Shape,
};

use super::{Gamma, GammaShapeRateTheta};

/// Gamma distribution parameterized by mean and shape.
pub type GammaMeanShape = Gamma<MeanShape, Log, Log>;
/// Gamma mean/shape parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeanShape;

define_two_positive_parameter_blocks! {
    eta:
    /// Predictors for gamma mean/shape on the link scale.
    GammaMeanShapeEta {
        /// Mean predictor.
        mean,
        /// Shape predictor.
        shape,
    }
    theta:
    /// Natural-scale gamma mean/shape parameters.
    GammaMeanShapeTheta {
        /// Positive mean.
        mean,
        /// Positive shape.
        shape,
    }
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
        eta.theta_from_links::<MeanLink, ShapeLink>()
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
            eta.chain_gradient::<MeanLink, ShapeLink>(d_mean, d_shape_param),
        )
    }
}

impl From<GammaMeanShapeTheta> for GammaShapeRateTheta {
    #[inline]
    fn from(theta: GammaMeanShapeTheta) -> Self {
        theta.shape_rate()
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MeanLink, ShapeLink> for Gamma<MeanShape, MeanLink, ShapeLink>;
    parameters = (Mean, Shape);
    arity = 2;
);

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
