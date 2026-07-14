#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Family, HasCdf, HasQuantile, InitialEtaFromObservations, InitialEtaFromTheta, Log, Mu,
    ObservationView, ParameterParts, PositiveLink, Shape,
};

use gamlss_special::{discrete_quantile, is_nonnegative_integer};

use crate::initial::{LARGE_SHAPE, positive_floor, weighted_summary, weighted_values};

use super::{MAX_CDF_TERMS, NegativeBinomial, NegativeBinomialTheta};

/// Negative binomial distribution with log links for mean and shape.
pub type NegativeBinomialMeanSize = NegativeBinomial<Log, Log>;

/// Predictors for the negative binomial family on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NegativeBinomialEta {
    /// Mean predictor.
    pub mu: f64,
    /// Shape predictor.
    pub shape: f64,
}

impl ParameterParts<2> for NegativeBinomialEta {
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
            _ => unreachable!("negative binomial eta only has indices 0 and 1"),
        }
    }
}

impl<MuLink, ShapeLink> NegativeBinomial<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    #[inline]
    fn theta_from_eta(eta: NegativeBinomialEta) -> NegativeBinomialTheta {
        NegativeBinomialTheta {
            mu: MuLink::inverse(eta.mu),
            shape: ShapeLink::inverse(eta.shape),
        }
    }

    #[inline]
    fn nll_and_gradient_eta_values(y: f64, eta: NegativeBinomialEta) -> (f64, NegativeBinomialEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (
                nll,
                NegativeBinomialEta {
                    mu: f64::NAN,
                    shape: f64::NAN,
                },
            );
        }

        let total = theta.shape + theta.mu;
        let d_mu = (y + theta.shape) / total - y / theta.mu;
        let d_shape =
            Self::digamma_shape_difference(y, theta.shape) + (theta.mu / theta.shape).ln_1p() - 1.0
                + (y + theta.shape) / total;
        let gradient_eta = NegativeBinomialEta {
            mu: d_mu * MuLink::derivative_inverse(eta.mu),
            shape: d_shape * ShapeLink::derivative_inverse(eta.shape),
        };

        (nll, gradient_eta)
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MuLink, ShapeLink> for NegativeBinomial<MuLink, ShapeLink>;
    parameters = (Mu, Shape);
    arity = 2;
);

impl<MuLink, ShapeLink> Family for NegativeBinomial<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    type Eta = NegativeBinomialEta;
    type Theta = NegativeBinomialTheta;
    type GradientEta = NegativeBinomialEta;
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

impl<MuLink, ShapeLink> InitialEtaFromObservations<2> for NegativeBinomial<MuLink, ShapeLink>
where
    MuLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    ShapeLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| is_nonnegative_integer(y).then_some(y));
        let Some(summary) = weighted_summary(&values) else {
            return NegativeBinomialEta::from_array([0.0, 0.0]);
        };

        let mu = positive_floor(summary.mean);
        let shape = if summary.variance <= mu {
            LARGE_SHAPE
        } else {
            positive_floor(mu * mu / (summary.variance - mu))
        };
        NegativeBinomialEta {
            mu: MuLink::initial_eta_from_theta(mu),
            shape: ShapeLink::initial_eta_from_theta(shape),
        }
    }
}

impl<MuLink, ShapeLink> HasCdf for NegativeBinomial<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        Self::cdf_theta(y, *theta)
    }
}

impl<MuLink, ShapeLink> HasQuantile for NegativeBinomial<MuLink, ShapeLink>
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

        #[allow(clippy::cast_precision_loss)]
        discrete_quantile(p, MAX_CDF_TERMS, |count| {
            Self::cdf_theta(count as f64, *theta)
        })
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink, ShapeLink> CanSimulate<Rng> for NegativeBinomial<MuLink, ShapeLink>
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

        let lambda = rand_distr::Distribution::sample(
            &rand_distr::Gamma::new(theta.shape, theta.mu / theta.shape)
                .expect("validated gamma-poisson parameters must construct"),
            rng,
        );
        rand_distr::Distribution::sample(
            &rand_distr::Poisson::new(lambda).expect("validated poisson mean must construct"),
            rng,
        )
    }
}
