#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Family, HasCdf, HasCrps, HasQuantile, InitialEtaFromObservations, InitialEtaFromTheta, Log, Mu,
    ObservationView, ParameterParts, PositiveLink, Shape,
};

use gamlss_special::{
    integrate_finite, invert_positive_cdf, unit_normal_cdf, unit_normal_log_sf, unit_normal_sf,
};

use crate::initial::{
    LARGE_SHAPE, VARIANCE_FLOOR, positive_floor, weighted_summary, weighted_values,
};

use super::InverseGaussian;

/// Inverse Gaussian distribution with log links for mean and shape.
pub type InverseGaussianMuShape = InverseGaussian<Log, Log>;
/// Inverse Gaussian distribution with log links for mean and shape.
pub type InverseGaussianMeanShape = InverseGaussianMuShape;

/// Predictors for the inverse Gaussian family on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InverseGaussianEta {
    /// Mean predictor.
    pub mu: f64,
    /// Shape predictor.
    pub shape: f64,
}

impl ParameterParts<2> for InverseGaussianEta {
    #[inline]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            mu: values[0],
            shape: values[1],
        }
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mu,
            1 => self.shape,
            _ => unreachable!("inverse Gaussian eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale inverse Gaussian parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InverseGaussianTheta {
    /// Positive mean parameter.
    pub mu: f64,
    /// Positive shape parameter.
    pub shape: f64,
}

impl<MuLink, ShapeLink> InverseGaussian<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    #[inline]
    fn theta_from_eta(eta: InverseGaussianEta) -> InverseGaussianTheta {
        InverseGaussianTheta {
            mu: MuLink::inverse(eta.mu),
            shape: ShapeLink::inverse(eta.shape),
        }
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
        let gradient_eta = InverseGaussianEta {
            mu: d_mu * MuLink::derivative_inverse(eta.mu),
            shape: d_shape * ShapeLink::derivative_inverse(eta.shape),
        };

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
impl<Rng, MuLink, ShapeLink> CanSimulate<Rng> for InverseGaussian<MuLink, ShapeLink>
where
    Rng: rand::Rng,
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    type Sample = f64;

    fn sample(&self, rng: &mut Rng, theta: &Self::Theta) -> f64 {
        if theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
        {
            return f64::NAN;
        }

        rand_distr::Distribution::sample(
            &rand_distr::InverseGaussian::new(theta.mu, theta.shape)
                .expect("validated inverse Gaussian parameters must construct"),
            rng,
        )
    }
}
