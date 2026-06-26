use gamlss_core::{
    Family, InitialEtaFromTheta, Log, ObservationView, ParameterParts, ParameterizedFamily,
    PositiveLink, Scale, Shape,
};

use super::Weibull;

/// Weibull distribution parameterized by scale and shape.
pub type WeibullScaleShape = Weibull<ScaleShape, Log, Log>;
/// Backward-compatible eta alias for scale/shape Weibull.
pub type WeibullEta = WeibullScaleShapeEta;
/// Backward-compatible theta alias for scale/shape Weibull.
pub type WeibullTheta = WeibullScaleShapeTheta;
/// Weibull scale/shape parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScaleShape;

/// Predictors for Weibull scale/shape on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeibullScaleShapeEta {
    /// Scale predictor.
    pub scale: f64,
    /// Shape predictor.
    pub shape: f64,
}

impl ParameterParts<2> for WeibullScaleShapeEta {
    #[inline]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            scale: values[0],
            shape: values[1],
        }
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.scale,
            1 => self.shape,
            _ => unreachable!("weibull scale/shape eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale Weibull scale/shape parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeibullScaleShapeTheta {
    /// Positive scale.
    pub scale: f64,
    /// Positive shape.
    pub shape: f64,
}

impl<ScaleLink, ShapeLink> Weibull<ScaleShape, ScaleLink, ShapeLink>
where
    ScaleLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    #[inline]
    fn theta_from_eta(eta: WeibullScaleShapeEta) -> WeibullScaleShapeTheta {
        WeibullScaleShapeTheta {
            scale: ScaleLink::inverse(eta.scale),
            shape: ShapeLink::inverse(eta.shape),
        }
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
            WeibullScaleShapeEta {
                scale: d_scale * ScaleLink::derivative_inverse(eta.scale),
                shape: d_shape * ShapeLink::derivative_inverse(eta.shape),
            },
        )
    }
}

impl<ScaleLink, ShapeLink> Family for Weibull<ScaleShape, ScaleLink, ShapeLink>
where
    ScaleLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    type Eta = WeibullScaleShapeEta;
    type Theta = WeibullScaleShapeTheta;
    type NllGradientEta = WeibullScaleShapeEta;
    type Observation<'obs> = f64;

    #[inline]
    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline]
    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::nll_scale_shape(y, theta)
    }

    #[inline]
    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        Self::nll_scale_shape(y, Self::theta_from_eta(eta))
    }

    #[inline]
    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        Self::nll_and_gradient_eta_values(y, eta)
    }
}

impl<ScaleLink, ShapeLink> ParameterizedFamily<2> for Weibull<ScaleShape, ScaleLink, ShapeLink>
where
    ScaleLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    ShapeLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Params = (Scale, Shape);
    type Links = (ScaleLink, ShapeLink);

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
