use gamlss_core::{
    Family, InitialEtaFromObservations, InitialEtaFromTheta, Log, ObservationView, ParameterParts,
    PositiveLink, Rate, Shape,
};

use crate::initial::positive_floor;

use super::{Gamma, GammaKernel};

/// Gamma distribution parameterized directly by shape $\alpha$ and rate $\beta$.
///
/// The default log links give $\alpha=\exp(\eta_\alpha)$ and $\beta=\exp(\eta_\beta)$.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../../doc-assets/distributions/gamma.svg")
)]
pub type GammaShapeRate = Gamma<ShapeRate, Log, Log>;

/// Gamma shape/rate parameterization marker.
///
/// This is the canonical parameterization used by the shared [`Gamma`] kernel.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShapeRate;

define_two_positive_parameter_blocks! {
    eta:
    /// Predictors for gamma shape/rate on the link scale.
    GammaShapeRateEta {
        /// Shape predictor.
        shape,
        /// Rate predictor.
        rate,
    }
    theta:
    /// Natural-scale gamma shape/rate parameters.
    GammaShapeRateTheta {
        /// Positive shape.
        shape,
        /// Positive rate.
        rate,
    }
}

impl<ShapeLink, RateLink> Gamma<ShapeRate, ShapeLink, RateLink>
where
    ShapeLink: PositiveLink<f64>,
    RateLink: PositiveLink<f64>,
{
    #[inline]
    fn theta_from_eta(eta: GammaShapeRateEta) -> GammaShapeRateTheta {
        eta.theta_from_links::<ShapeLink, RateLink>()
    }

    #[inline]
    fn nll_and_gradient_eta_values(y: f64, eta: GammaShapeRateEta) -> (f64, GammaShapeRateEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = GammaKernel::nll_shape_rate(y, theta);
        if !nll.is_finite() {
            return (nll, GammaShapeRateEta::from_array([f64::NAN; 2]));
        }

        let (d_shape, d_rate) = GammaKernel::gradient_shape_rate(y, theta);
        (
            nll,
            eta.chain_gradient::<ShapeLink, RateLink>(d_shape, d_rate),
        )
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<ShapeLink, RateLink> for Gamma<ShapeRate, ShapeLink, RateLink>;
    parameters = (Shape, Rate);
    arity = 2;
);

impl<ShapeLink, RateLink> Family for Gamma<ShapeRate, ShapeLink, RateLink>
where
    ShapeLink: PositiveLink<f64>,
    RateLink: PositiveLink<f64>,
{
    type Eta = GammaShapeRateEta;
    type Theta = GammaShapeRateTheta;
    type GradientEta = GammaShapeRateEta;
    type Observation<'obs> = f64;
    type Workspace = ();

    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    #[inline]
    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        GammaKernel::nll_shape_rate(y, *theta)
    }

    #[inline]
    fn nll_eta(&self, y: f64, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> f64 {
        GammaKernel::nll_shape_rate(y, Self::theta_from_eta(*eta))
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

impl<ShapeLink, RateLink> InitialEtaFromObservations<2> for Gamma<ShapeRate, ShapeLink, RateLink>
where
    ShapeLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    RateLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let Some((mean, shape)) = GammaKernel::initial_mean_shape(obs) else {
            return GammaShapeRateEta::from_array([0.0, 0.0]);
        };
        let rate = positive_floor(shape / mean);

        GammaShapeRateEta {
            shape: ShapeLink::initial_eta_from_theta(shape),
            rate: RateLink::initial_eta_from_theta(rate),
        }
    }
}
