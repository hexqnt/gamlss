use gamlss_core::{
    Family, InitialEtaFromTheta, Log, ObservationView, ParameterParts, ParameterizedFamily,
    PositiveLink, Rate, Shape,
};

use crate::initial::positive_floor;

use super::Gamma;

/// Gamma distribution parameterized by shape and rate.
pub type GammaShapeRate = Gamma<ShapeRate, Log, Log>;

/// Backward-compatible eta alias for the shape/rate gamma.
pub type GammaEta = GammaShapeRateEta;
/// Backward-compatible theta alias for the shape/rate gamma.
pub type GammaTheta = GammaShapeRateTheta;
/// Gamma shape/rate parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShapeRate;

/// Predictors for gamma shape/rate on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaShapeRateEta {
    /// Shape predictor.
    pub shape: f64,
    /// Rate predictor.
    pub rate: f64,
}

impl ParameterParts<2> for GammaShapeRateEta {
    #[inline(always)]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            shape: values[0],
            rate: values[1],
        }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.shape,
            1 => self.rate,
            _ => unreachable!("gamma shape/rate eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale gamma shape/rate parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaShapeRateTheta {
    /// Positive shape.
    pub shape: f64,
    /// Positive rate.
    pub rate: f64,
}

impl<ShapeLink, RateLink> Gamma<ShapeRate, ShapeLink, RateLink>
where
    ShapeLink: PositiveLink<f64>,
    RateLink: PositiveLink<f64>,
{
    #[inline(always)]
    fn theta_from_eta(eta: GammaShapeRateEta) -> GammaShapeRateTheta {
        GammaShapeRateTheta {
            shape: ShapeLink::inverse(eta.shape),
            rate: RateLink::inverse(eta.rate),
        }
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: GammaShapeRateEta) -> (f64, GammaShapeRateEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_shape_rate(y, theta);
        if !nll.is_finite() {
            return (nll, GammaShapeRateEta::from_array([f64::NAN; 2]));
        }

        let (d_shape, d_rate) = Self::gradient_shape_rate(y, theta);
        (
            nll,
            GammaShapeRateEta {
                shape: d_shape * ShapeLink::derivative_inverse(eta.shape),
                rate: d_rate * RateLink::derivative_inverse(eta.rate),
            },
        )
    }
}

impl<ShapeLink, RateLink> Family for Gamma<ShapeRate, ShapeLink, RateLink>
where
    ShapeLink: PositiveLink<f64>,
    RateLink: PositiveLink<f64>,
{
    type Eta = GammaShapeRateEta;
    type Theta = GammaShapeRateTheta;
    type NllGradientEta = GammaShapeRateEta;
    type Observation<'obs> = f64;

    #[inline(always)]
    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline(always)]
    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::nll_shape_rate(y, theta)
    }

    #[inline(always)]
    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        Self::nll_shape_rate(y, Self::theta_from_eta(eta))
    }

    #[inline(always)]
    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        Self::nll_and_gradient_eta_values(y, eta)
    }
}

impl<ShapeLink, RateLink> ParameterizedFamily<2> for Gamma<ShapeRate, ShapeLink, RateLink>
where
    ShapeLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    RateLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Params = (Shape, Rate);
    type Links = (ShapeLink, RateLink);

    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let Some((mean, shape)) = Self::initial_mean_shape(obs) else {
            return GammaShapeRateEta::from_array([0.0, 0.0]);
        };
        let rate = positive_floor(shape / mean);

        GammaShapeRateEta {
            shape: ShapeLink::initial_eta_from_theta(shape),
            rate: RateLink::initial_eta_from_theta(rate),
        }
    }
}
