use gamlss_core::{
    Family, InitialEtaFromTheta, Log, Mean, ObservationView, ParameterParts, ParameterizedFamily,
    PositiveLink, Shape,
};

use crate::special::{digamma, ln_gamma};

use super::{Weibull, WeibullScaleShapeTheta};

/// Weibull distribution parameterized by mean and shape.
pub type WeibullMeanShape = Weibull<MeanShape, Log, Log>;
/// Weibull mean/shape parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeanShape;

/// Predictors for Weibull mean/shape on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeibullMeanShapeEta {
    /// Mean predictor.
    pub mean: f64,
    /// Shape predictor.
    pub shape: f64,
}

impl ParameterParts<2> for WeibullMeanShapeEta {
    #[inline(always)]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            mean: values[0],
            shape: values[1],
        }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mean,
            1 => self.shape,
            _ => unreachable!("weibull mean/shape eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale Weibull mean/shape parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeibullMeanShapeTheta {
    /// Positive mean.
    pub mean: f64,
    /// Positive shape.
    pub shape: f64,
}

impl WeibullMeanShapeTheta {
    #[inline(always)]
    pub(super) fn scale_shape(self) -> WeibullScaleShapeTheta {
        WeibullScaleShapeTheta {
            scale: self.mean / ln_gamma(1.0 + 1.0 / self.shape).exp(),
            shape: self.shape,
        }
    }
}

impl<MeanLink, ShapeLink> Weibull<MeanShape, MeanLink, ShapeLink>
where
    MeanLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    #[inline(always)]
    fn theta_from_eta(eta: WeibullMeanShapeEta) -> WeibullMeanShapeTheta {
        WeibullMeanShapeTheta {
            mean: MeanLink::inverse(eta.mean),
            shape: ShapeLink::inverse(eta.shape),
        }
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: WeibullMeanShapeEta) -> (f64, WeibullMeanShapeEta) {
        let theta = Self::theta_from_eta(eta);
        let scale_shape = theta.scale_shape();
        let nll = Self::nll_scale_shape(y, scale_shape);
        if !nll.is_finite() {
            return (nll, WeibullMeanShapeEta::from_array([f64::NAN; 2]));
        }

        let (d_scale, d_shape_kernel) = Self::gradient_scale_shape(y, scale_shape);
        let d_mean = d_scale * scale_shape.scale / theta.mean;
        let a = 1.0 + 1.0 / theta.shape;
        let d_scale_d_shape = scale_shape.scale * digamma(a) / (theta.shape * theta.shape);
        let d_shape = d_shape_kernel + d_scale * d_scale_d_shape;

        (
            nll,
            WeibullMeanShapeEta {
                mean: d_mean * MeanLink::derivative_inverse(eta.mean),
                shape: d_shape * ShapeLink::derivative_inverse(eta.shape),
            },
        )
    }
}

impl From<WeibullMeanShapeTheta> for WeibullScaleShapeTheta {
    #[inline(always)]
    fn from(theta: WeibullMeanShapeTheta) -> Self {
        theta.scale_shape()
    }
}

impl<MeanLink, ShapeLink> Family for Weibull<MeanShape, MeanLink, ShapeLink>
where
    MeanLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    type Eta = WeibullMeanShapeEta;
    type Theta = WeibullMeanShapeTheta;
    type NllGradientEta = WeibullMeanShapeEta;
    type Observation<'obs> = f64;

    #[inline(always)]
    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline(always)]
    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::nll_scale_shape(y, theta.scale_shape())
    }

    #[inline(always)]
    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        Self::nll_scale_shape(y, Self::theta_from_eta(eta).scale_shape())
    }

    #[inline(always)]
    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        Self::nll_and_gradient_eta_values(y, eta)
    }
}

impl<MeanLink, ShapeLink> ParameterizedFamily<2> for Weibull<MeanShape, MeanLink, ShapeLink>
where
    MeanLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    ShapeLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Params = (Mean, Shape);
    type Links = (MeanLink, ShapeLink);

    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let Some((scale, shape)) = Self::initial_scale_shape(obs) else {
            return WeibullMeanShapeEta::from_array([0.0, 0.0]);
        };
        let mean = scale * Self::mean_factor(shape);

        WeibullMeanShapeEta {
            mean: MeanLink::initial_eta_from_theta(mean),
            shape: ShapeLink::initial_eta_from_theta(shape),
        }
    }
}
