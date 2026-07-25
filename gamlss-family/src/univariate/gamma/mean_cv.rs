use gamlss_core::{
    Cv, Family, InitialEtaFromObservations, InitialEtaFromTheta, Log, Mean, ObservationView,
    ParameterParts, PositiveLink,
};
use gamlss_special::{digamma, digamma_minus_ln};

use crate::initial::positive_floor;

use super::{Gamma, GammaKernel, GammaShapeRateTheta};

/// Gamma distribution parameterized by mean $\mu>0$ and coefficient of variation $c>0$.
///
/// Here $c=\sqrt{\operatorname{Var}(Y)}/\mathbb{E}(Y)$.
///
/// It maps to the shared shape/rate kernel as
///
/// $$
/// \alpha=c^{-2},
/// \qquad
/// \beta=\frac{1}{\mu c^2},
/// \qquad
/// \operatorname{Var}(Y)=\mu^2c^2.
/// $$
///
/// The default log links give $\mu=\exp(\eta_\mu)$ and $c=\exp(\eta_c)$.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../../doc-assets/distributions/gamma_mean_cv.svg")
)]
#[allow(clippy::doc_markdown)]
pub type GammaMeanCv = Gamma<MeanCv, Log, Log>;
/// Gamma mean/coefficient-of-variation parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeanCv;

define_two_positive_parameter_blocks! {
    eta:
    /// Predictors for gamma mean/CV on the link scale.
    GammaMeanCvEta {
        /// Mean predictor.
        mean,
        /// Coefficient-of-variation predictor.
        cv,
    }
    theta:
    /// Natural-scale gamma mean/CV parameters.
    GammaMeanCvTheta {
        /// Positive mean.
        mean,
        /// Positive coefficient of variation.
        cv,
    }
}

impl GammaMeanCvTheta {
    #[inline]
    pub(super) fn shape_rate(self) -> GammaShapeRateTheta {
        let shape = (1.0 / self.cv) / self.cv;
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
    fn scaled_digamma_minus_log(shape: f64) -> f64 {
        if shape < 1.0 {
            return shape.mul_add(digamma(shape + 1.0) - shape.ln(), -1.0);
        }
        shape * digamma_minus_ln(shape)
    }

    #[inline]
    fn scaled_ratio_deviance(y: f64, mean: f64, shape_rate: GammaShapeRateTheta) -> f64 {
        let centered = (y - mean) / mean;
        if centered.abs() <= 0.5 {
            shape_rate.shape * (centered - centered.ln_1p())
        } else {
            shape_rate.shape.mul_add(
                -(y.ln() - mean.ln()),
                shape_rate.rate.mul_add(y, -shape_rate.shape),
            )
        }
    }

    #[inline]
    fn theta_from_eta(eta: GammaMeanCvEta) -> GammaMeanCvTheta {
        eta.theta_from_links::<MeanLink, CvLink>()
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn nll_and_gradient_eta_values(y: f64, eta: GammaMeanCvEta) -> (f64, GammaMeanCvEta) {
        let theta = Self::theta_from_eta(eta);
        let shape_rate = theta.shape_rate();
        let nll = GammaKernel::nll_shape_rate(y, shape_rate);
        if !nll.is_finite() {
            return (nll, GammaMeanCvEta::from_array([f64::NAN; 2]));
        }

        let log_mean_score = shape_rate.shape - shape_rate.rate * y;
        let d_mean = log_mean_score / theta.mean;
        let log_cv_score = -2.0
            * (Self::scaled_digamma_minus_log(shape_rate.shape)
                + Self::scaled_ratio_deviance(y, theta.mean, shape_rate));
        let d_cv = log_cv_score / theta.cv;

        (nll, eta.chain_gradient::<MeanLink, CvLink>(d_mean, d_cv))
    }
}

impl From<GammaMeanCvTheta> for GammaShapeRateTheta {
    #[inline]
    fn from(theta: GammaMeanCvTheta) -> Self {
        theta.shape_rate()
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MeanLink, CvLink> for Gamma<MeanCv, MeanLink, CvLink>;
    parameters = (Mean, Cv);
    arity = 2;
);

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

    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    #[inline]
    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        GammaKernel::nll_shape_rate(y, theta.shape_rate())
    }

    #[inline]
    fn nll_eta(&self, y: f64, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> f64 {
        GammaKernel::nll_shape_rate(y, Self::theta_from_eta(*eta).shape_rate())
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
        let Some((mean, shape)) = GammaKernel::initial_mean_shape(obs) else {
            return GammaMeanCvEta::from_array([0.0, 0.0]);
        };
        let cv = positive_floor(1.0 / shape.sqrt());

        GammaMeanCvEta {
            mean: MeanLink::initial_eta_from_theta(mean),
            cv: CvLink::initial_eta_from_theta(cv),
        }
    }
}
