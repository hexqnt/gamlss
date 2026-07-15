use gamlss_core::{
    Family, InitialEtaFromObservations, InitialEtaFromTheta, Log, Mean, ObservationView,
    ParameterParts, PositiveLink, Shape,
};

use gamlss_special::{digamma, ln_gamma};

use super::{Weibull, WeibullScaleShapeTheta};

/// Weibull distribution parameterized by mean and shape.
pub type WeibullMeanShape = Weibull<MeanShape, Log, Log>;
/// Weibull mean/shape parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeanShape;

define_two_positive_parameter_blocks! {
    eta:
    /// Predictors for Weibull mean/shape on the link scale.
    WeibullMeanShapeEta {
        /// Mean predictor.
        mean,
        /// Shape predictor.
        shape,
    }
    theta:
    /// Natural-scale Weibull mean/shape parameters.
    WeibullMeanShapeTheta {
        /// Positive mean.
        mean,
        /// Positive shape.
        shape,
    }
}

impl WeibullMeanShapeTheta {
    #[inline]
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
    #[inline]
    fn theta_from_eta(eta: WeibullMeanShapeEta) -> WeibullMeanShapeTheta {
        eta.theta_from_links::<MeanLink, ShapeLink>()
    }

    #[inline]
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
        let d_shape = d_scale.mul_add(d_scale_d_shape, d_shape_kernel);

        (
            nll,
            eta.chain_gradient::<MeanLink, ShapeLink>(d_mean, d_shape),
        )
    }
}

impl From<WeibullMeanShapeTheta> for WeibullScaleShapeTheta {
    #[inline]
    fn from(theta: WeibullMeanShapeTheta) -> Self {
        theta.scale_shape()
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MeanLink, ShapeLink> for Weibull<MeanShape, MeanLink, ShapeLink>;
    parameters = (Mean, Shape);
    arity = 2;
);

impl<MeanLink, ShapeLink> Family for Weibull<MeanShape, MeanLink, ShapeLink>
where
    MeanLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    type Eta = WeibullMeanShapeEta;
    type Theta = WeibullMeanShapeTheta;
    type GradientEta = WeibullMeanShapeEta;
    type Observation<'obs> = f64;
    type Workspace = ();

    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    #[inline]
    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        Self::nll_scale_shape(y, theta.scale_shape())
    }

    #[inline]
    fn nll_eta(&self, y: f64, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> f64 {
        Self::nll_scale_shape(y, Self::theta_from_eta(*eta).scale_shape())
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

impl<MeanLink, ShapeLink> InitialEtaFromObservations<2> for Weibull<MeanShape, MeanLink, ShapeLink>
where
    MeanLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    ShapeLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
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
