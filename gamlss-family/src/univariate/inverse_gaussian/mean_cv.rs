use std::marker::PhantomData;

use gamlss_core::{
    Cv, Family, HasCdf, HasCrps, HasQuantile, InitialEtaFromObservations, InitialEtaFromTheta, Log,
    Mean, ObservationView, ParameterParts, PositiveLink,
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

use crate::initial::{
    LARGE_SHAPE, VARIANCE_FLOOR, positive_floor, weighted_summary, weighted_values,
};

use super::{InverseGaussian, InverseGaussianTheta};

/// Inverse Gaussian distribution with mean $\mu>0$ and coefficient of variation $c>0$.
///
/// Here $c=\sqrt{\operatorname{Var}(Y)}/\mathbb{E}(Y)$.
///
/// The natural fields [`InverseGaussianMeanCvTheta::mean`] and [`InverseGaussianMeanCvTheta::cv`] map to the canonical carrier through $\lambda=\mu/c^2$, so $\operatorname{Var}(Y)=\mu^2c^2$. The default links give $\mu=\exp(\eta_\mu)$ and $c=\exp(\eta_c)$.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../../doc-assets/distributions/inverse_gaussian_mean_cv.svg")
)]
#[allow(clippy::doc_markdown)]
pub type InverseGaussianMeanCv = InverseGaussianCv<Log, Log>;

define_two_positive_parameter_blocks! {
    eta:
    /// Predictors for inverse Gaussian mean/CV on the link scale.
    InverseGaussianMeanCvEta {
        /// Mean predictor.
        mean,
        /// Coefficient-of-variation predictor.
        cv,
    }
    theta:
    /// Natural-scale inverse Gaussian mean/CV parameters.
    InverseGaussianMeanCvTheta {
        /// Positive mean.
        mean,
        /// Positive coefficient of variation.
        cv,
    }
}

/// Inverse Gaussian mean/CV implementation carrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InverseGaussianCv<MeanLink = Log, CvLink = Log> {
    marker: PhantomData<(MeanLink, CvLink)>,
}

impl<MeanLink, CvLink> InverseGaussianCv<MeanLink, CvLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
{
    /// Creates a stateless inverse Gaussian mean/CV family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(eta: InverseGaussianMeanCvEta) -> InverseGaussianMeanCvTheta {
        eta.theta_from_links::<MeanLink, CvLink>()
    }
}

impl<MeanLink, CvLink> Default for InverseGaussianCv<MeanLink, CvLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl InverseGaussianMeanCvTheta {
    #[inline]
    fn mean_shape(self) -> InverseGaussianTheta {
        InverseGaussianTheta {
            mu: self.mean,
            shape: (self.mean / self.cv) / self.cv,
        }
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MeanLink, CvLink> for InverseGaussianCv<MeanLink, CvLink>;
    parameters = (Mean, Cv);
    arity = 2;
);

impl<MeanLink, CvLink> Family for InverseGaussianCv<MeanLink, CvLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
{
    type Eta = InverseGaussianMeanCvEta;
    type Theta = InverseGaussianMeanCvTheta;
    type GradientEta = InverseGaussianMeanCvEta;
    type Observation<'obs> = f64;
    type Workspace = ();
    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        InverseGaussian::<Log, Log>::nll_theta(y, theta.mean_shape())
    }

    fn nll_eta(&self, y: f64, eta: &Self::Eta, workspace: &mut Self::Workspace) -> f64 {
        let theta = Self::theta_from_eta(*eta);
        self.nll(y, &theta, workspace)
    }

    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        let theta = Self::theta_from_eta(*eta);
        let mean_shape = theta.mean_shape();
        let nll = InverseGaussian::<Log, Log>::nll_theta(y, mean_shape);
        if !nll.is_finite() {
            return (nll, InverseGaussianMeanCvEta::from_array([f64::NAN; 2]));
        }

        let centered = (y - theta.mean) / theta.mean;
        let (ratio_deviance, ratio_difference) = if centered.abs() <= 0.5 {
            let denominator = 1.0 + centered;
            (
                centered * centered / denominator,
                -centered * (2.0 + centered) / denominator,
            )
        } else {
            (
                y / theta.mean + theta.mean / y - 2.0,
                theta.mean / y - y / theta.mean,
            )
        };
        let inverse_cv = 1.0 / theta.cv;
        let inverse_cv_squared = inverse_cv * inverse_cv;
        let log_mean_score = 0.5f64.mul_add(inverse_cv_squared * ratio_difference, -0.5);
        let log_cv_score = 1.0 - inverse_cv_squared * ratio_deviance;
        let d_mean = log_mean_score / theta.mean;
        let d_cv = log_cv_score / theta.cv;

        (nll, eta.chain_gradient::<MeanLink, CvLink>(d_mean, d_cv))
    }
}

impl<MeanLink, CvLink> InitialEtaFromObservations<2> for InverseGaussianCv<MeanLink, CvLink>
where
    MeanLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    CvLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values =
            weighted_values::<Self, _, _>(obs, |y| (y.is_finite() && y > 0.0).then_some(y));
        let Some(summary) = weighted_summary(&values) else {
            return InverseGaussianMeanCvEta::from_array([0.0, 0.0]);
        };

        let mean = positive_floor(summary.mean);
        let cv = if summary.variance <= VARIANCE_FLOOR {
            positive_floor((mean / LARGE_SHAPE).sqrt())
        } else {
            positive_floor((summary.variance / (mean * mean)).sqrt())
        };

        InverseGaussianMeanCvEta {
            mean: MeanLink::initial_eta_from_theta(mean),
            cv: CvLink::initial_eta_from_theta(cv),
        }
    }
}

impl<MeanLink, CvLink> HasCdf for InverseGaussianCv<MeanLink, CvLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        InverseGaussian::<Log, Log>::new().cdf(y, &theta.mean_shape())
    }
}

impl<MeanLink, CvLink> HasQuantile for InverseGaussianCv<MeanLink, CvLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
{
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        InverseGaussian::<Log, Log>::new().quantile(p, &theta.mean_shape())
    }
}

impl<MeanLink, CvLink> HasCrps for InverseGaussianCv<MeanLink, CvLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
{
    fn crps(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        InverseGaussian::<Log, Log>::new().crps(y, &theta.mean_shape())
    }
}

#[cfg(feature = "rand")]
impl<Rng, MeanLink, CvLink> TrySimulate<Rng> for InverseGaussianCv<MeanLink, CvLink>
where
    Rng: rand::Rng,
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
{
    type Sample = f64;

    fn try_sample(
        &self,
        rng: &mut Rng,
        theta: &Self::Theta,
    ) -> Result<Self::Sample, SimulationError> {
        InverseGaussian::<Log, Log>::new().try_sample(rng, &theta.mean_shape())
    }
}
