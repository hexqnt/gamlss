use gamlss_core::{
    Family, InitialEtaFromObservations, InitialEtaFromTheta, Log, ObservationView, ParameterParts,
    PositiveLink, Scale, Shape,
};

use super::Weibull;

/// Weibull distribution parameterized by scale $a>0$ and shape $k>0$.
///
/// Here $a$ is [`WeibullScaleShapeTheta::scale`], $k$ is [`WeibullScaleShapeTheta::shape`], and the matching eta fields are `scale` and `shape`. The default links give $a=\exp(\eta_a)$ and $k=\exp(\eta_k)$.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../../doc-assets/weibull.svg")
)]
#[allow(clippy::doc_markdown)]
pub type WeibullScaleShape = Weibull<ScaleShape, Log, Log>;
/// Backward-compatible eta alias for scale/shape Weibull.
pub type WeibullEta = WeibullScaleShapeEta;
/// Backward-compatible theta alias for scale/shape Weibull.
pub type WeibullTheta = WeibullScaleShapeTheta;
/// Weibull scale/shape parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScaleShape;

define_two_positive_parameter_blocks! {
    eta:
    /// Predictors for Weibull scale/shape on the link scale.
    WeibullScaleShapeEta {
        /// Scale predictor.
        scale,
        /// Shape predictor.
        shape,
    }
    theta:
    /// Natural-scale Weibull scale/shape parameters.
    WeibullScaleShapeTheta {
        /// Positive scale.
        scale,
        /// Positive shape.
        shape,
    }
}

impl<ScaleLink, ShapeLink> Weibull<ScaleShape, ScaleLink, ShapeLink>
where
    ScaleLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    #[inline]
    fn theta_from_eta(eta: WeibullScaleShapeEta) -> WeibullScaleShapeTheta {
        eta.theta_from_links::<ScaleLink, ShapeLink>()
    }

    #[inline]
    fn nll_and_gradient_eta_values(
        y: f64,
        eta: WeibullScaleShapeEta,
    ) -> (f64, WeibullScaleShapeEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_scale_shape(y, theta);
        if !nll.is_finite() {
            return (nll, WeibullScaleShapeEta::from_array([f64::NAN; 2]));
        }

        let (d_scale, d_shape) = Self::gradient_scale_shape(y, theta);
        (
            nll,
            eta.chain_gradient::<ScaleLink, ShapeLink>(d_scale, d_shape),
        )
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<ScaleLink, ShapeLink> for Weibull<ScaleShape, ScaleLink, ShapeLink>;
    parameters = (Scale, Shape);
    arity = 2;
);

impl<ScaleLink, ShapeLink> Family for Weibull<ScaleShape, ScaleLink, ShapeLink>
where
    ScaleLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    type Eta = WeibullScaleShapeEta;
    type Theta = WeibullScaleShapeTheta;
    type GradientEta = WeibullScaleShapeEta;
    type Observation<'obs> = f64;
    type Workspace = ();

    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    #[inline]
    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        Self::nll_scale_shape(y, *theta)
    }

    #[inline]
    fn nll_eta(&self, y: f64, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> f64 {
        Self::nll_scale_shape(y, Self::theta_from_eta(*eta))
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

impl<ScaleLink, ShapeLink> InitialEtaFromObservations<2>
    for Weibull<ScaleShape, ScaleLink, ShapeLink>
where
    ScaleLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    ShapeLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let Some((scale, shape)) = Self::initial_scale_shape(obs) else {
            return WeibullScaleShapeEta::from_array([0.0, 0.0]);
        };

        WeibullScaleShapeEta {
            scale: ScaleLink::initial_eta_from_theta(scale),
            shape: ShapeLink::initial_eta_from_theta(shape),
        }
    }
}
