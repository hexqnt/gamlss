use gamlss_core::{
    Family, HasCdf, HasCrps, HasQuantile, InitialEtaFromObservations, InitialEtaFromTheta, Log, Mu,
    ObservationView, ParameterParts, PositiveLink, Shape,
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

use gamlss_special::{
    integrate_finite, invert_positive_cdf, unit_normal_cdf, unit_normal_log_sf, unit_normal_sf,
};

use crate::initial::{
    LARGE_SHAPE, VARIANCE_FLOOR, positive_floor, weighted_summary, weighted_values,
};

use super::InverseGaussian;

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
        let nll = Self::nll_theta(y, theta);
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
        Self::nll_theta(y, *theta)
    }

    #[inline]
    fn nll_eta(&self, y: f64, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> f64 {
        Self::nll_theta(y, Self::theta_from_eta(*eta))
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
        if !y.is_finite()
            || theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
        {
            return f64::NAN;
        }
        if y <= 0.0 {
            return 0.0;
        }

        let scale = (theta.shape / y).sqrt();
        let ratio = y / theta.mu;
        let first_argument = scale * (ratio - 1.0);
        let first = unit_normal_cdf(first_argument);
        let log_multiplier = 2.0 * theta.shape / theta.mu;
        let second = (log_multiplier + unit_normal_log_sf(scale * (ratio + 1.0))).exp();

        (first + second).clamp(0.0, 1.0)
    }
}

impl<MuLink, ShapeLink> HasQuantile for InverseGaussian<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        if theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
        {
            return f64::NAN;
        }

        invert_positive_cdf(p, |y| self.cdf(y, theta))
    }
}

impl<MuLink, ShapeLink> HasCrps for InverseGaussian<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    fn crps(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        if y < 0.0
            || !y.is_finite()
            || theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
        {
            return f64::NAN;
        }

        let left = integrate_finite(0.0, y, |x| {
            let cdf = self.cdf(x, theta);
            cdf * cdf
        });
        let right = integrate_finite(0.0, 1.0, |u| {
            #[allow(clippy::float_cmp)]
            if u == 1.0 {
                return 0.0;
            }

            let one_minus_u = 1.0 - u;
            let x = y + u / one_minus_u;
            let scale = (theta.shape / x).sqrt();
            let ratio = x / theta.mu;
            let first_survival = unit_normal_sf(scale * (ratio - 1.0));
            let second =
                (2.0 * theta.shape / theta.mu + unit_normal_log_sf(scale * (ratio + 1.0))).exp();
            let survival = (first_survival - second).max(0.0);
            survival * survival / (one_minus_u * one_minus_u)
        });

        left + right
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
        if theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
        {
            return Err(SimulationError::InvalidParameters("Inverse Gaussian theta"));
        }

        let distribution = rand_distr::InverseGaussian::new(theta.mu, theta.shape)
            .map_err(|_| SimulationError::BackendRejected("Inverse Gaussian mean/shape"))?;
        crate::simulation::ensure_finite(
            rand_distr::Distribution::sample(&distribution, rng),
            "Inverse Gaussian sample",
        )
    }
}
