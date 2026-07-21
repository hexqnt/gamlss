use gamlss_core::{
    Family, HasCdf, HasCrps, HasQuantile, InitialEtaFromObservations, InitialEtaFromTheta, Log, Mu,
    ObservationView, ParameterParts, PositiveLink, Shape,
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

use crate::initial::{
    LARGE_SHAPE, VARIANCE_FLOOR, positive_floor, weighted_summary, weighted_values,
};

use super::{InverseGaussian, InverseGaussianKernel};

/// Inverse Gaussian distribution with mean $\mu>0$ and shape $\lambda>0$.
///
/// The code fields are `mu` and `shape`, so $\eta_\lambda$ denotes [`InverseGaussianEta::shape`]. The default links give $\mu=\exp(\eta_\mu)$ and $\lambda=\exp(\eta_\lambda)$.
pub type InverseGaussianMuShape = InverseGaussian<Log, Log>;
/// Alias for [`InverseGaussianMuShape`].
pub type InverseGaussianMeanShape = InverseGaussianMuShape;

define_two_positive_parameter_blocks! {
    eta:
    /// Predictors for the inverse Gaussian family on the link scale.
    InverseGaussianEta {
        /// Mean predictor.
        mu,
        /// Shape predictor.
        shape,
    }
    theta:
    /// Natural-scale inverse Gaussian parameters.
    InverseGaussianTheta {
        /// Positive mean parameter.
        mu,
        /// Positive shape parameter.
        shape,
    }
}

impl<MuLink, ShapeLink> InverseGaussian<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    #[inline]
    fn theta_from_eta(eta: InverseGaussianEta) -> InverseGaussianTheta {
        eta.theta_from_links::<MuLink, ShapeLink>()
    }

    #[inline]
    fn nll_and_gradient_eta_values(y: f64, eta: InverseGaussianEta) -> (f64, InverseGaussianEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = InverseGaussianKernel::nll_theta(y, theta);
        if !nll.is_finite() {
            return (
                nll,
                InverseGaussianEta {
                    mu: f64::NAN,
                    shape: f64::NAN,
                },
            );
        }

        let residual = y - theta.mu;
        let d_mu = -theta.shape * residual / (theta.mu * theta.mu * theta.mu);
        let d_shape = -0.5 / theta.shape + residual * residual / (2.0 * theta.mu * theta.mu * y);
        let gradient_eta = eta.chain_gradient::<MuLink, ShapeLink>(d_mu, d_shape);

        (nll, gradient_eta)
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MuLink, ShapeLink> for InverseGaussian<MuLink, ShapeLink>;
    parameters = (Mu, Shape);
    arity = 2;
);

impl<MuLink, ShapeLink> Family for InverseGaussian<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    type Eta = InverseGaussianEta;
    type Theta = InverseGaussianTheta;
    type GradientEta = InverseGaussianEta;
    type Observation<'obs> = f64;
    type Workspace = ();

    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    #[inline]
    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        InverseGaussianKernel::nll_theta(y, *theta)
    }

    #[inline]
    fn nll_eta(&self, y: f64, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> f64 {
        InverseGaussianKernel::nll_theta(y, Self::theta_from_eta(*eta))
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

impl<MuLink, ShapeLink> InitialEtaFromObservations<2> for InverseGaussian<MuLink, ShapeLink>
where
    MuLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    ShapeLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values =
            weighted_values::<Self, _, _>(obs, |y| (y.is_finite() && y > 0.0).then_some(y));
        let Some(summary) = weighted_summary(&values) else {
            return InverseGaussianEta::from_array([0.0, 0.0]);
        };

        let mu = positive_floor(summary.mean);
        let shape = if summary.variance <= VARIANCE_FLOOR {
            LARGE_SHAPE
        } else {
            positive_floor(mu * mu * mu / summary.variance)
        };

        InverseGaussianEta {
            mu: MuLink::initial_eta_from_theta(mu),
            shape: ShapeLink::initial_eta_from_theta(shape),
        }
    }
}

impl<MuLink, ShapeLink> HasCdf for InverseGaussian<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        InverseGaussianKernel::cdf_theta(y, *theta)
    }
}

impl<MuLink, ShapeLink> HasQuantile for InverseGaussian<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        InverseGaussianKernel::quantile_theta(p, *theta)
    }
}

impl<MuLink, ShapeLink> HasCrps for InverseGaussian<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    fn crps(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        InverseGaussianKernel::crps_theta(y, *theta)
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink, ShapeLink> TrySimulate<Rng> for InverseGaussian<MuLink, ShapeLink>
where
    Rng: rand::Rng,
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    type Sample = f64;

    fn try_sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Result<f64, SimulationError> {
        InverseGaussianKernel::try_sample(rng, *theta)
    }
}
