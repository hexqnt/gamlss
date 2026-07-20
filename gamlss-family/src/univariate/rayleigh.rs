use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasQuantile, InitialEtaFromObservations, InitialEtaFromTheta, Log,
    ObservationView, ParameterParts, PositiveLink, Scale,
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

use crate::domain::{is_positive_finite, is_probability};
use crate::initial::{positive_floor, weighted_values};

/// Rayleigh distribution parameterized by scale with a log link.
pub type RayleighScale = Rayleigh<Log>;

/// Rayleigh family with positive scale $\sigma$.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/distributions/rayleigh.svg")
)]
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rayleigh<ScaleLink = Log> {
    marker: PhantomData<ScaleLink>,
}

impl<ScaleLink> Rayleigh<ScaleLink>
where
    ScaleLink: PositiveLink<f64>,
{
    /// Creates a stateless Rayleigh family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(eta: RayleighEta) -> RayleighTheta {
        RayleighTheta {
            scale: ScaleLink::inverse(eta.scale),
        }
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn nll_theta(y: f64, theta: RayleighTheta) -> f64 {
        if !is_positive_finite(y) || !is_positive_finite(theta.scale) {
            return f64::INFINITY;
        }
        let ratio = y / theta.scale;
        2.0 * theta.scale.ln() - y.ln() + 0.5 * ratio * ratio
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn nll_and_gradient_eta_values(y: f64, eta: RayleighEta) -> (f64, RayleighEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (nll, RayleighEta { scale: f64::NAN });
        }
        let ratio = y / theta.scale;
        let d_scale = (2.0 - ratio * ratio) / theta.scale;
        (
            nll,
            RayleighEta {
                scale: d_scale * ScaleLink::derivative_inverse(eta.scale),
            },
        )
    }
}

impl<ScaleLink> Default for Rayleigh<ScaleLink>
where
    ScaleLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<ScaleLink> for Rayleigh<ScaleLink>;
    parameters = (Scale,);
    arity = 1;
);

impl<ScaleLink> Family for Rayleigh<ScaleLink>
where
    ScaleLink: PositiveLink<f64>,
{
    type Eta = RayleighEta;
    type Theta = RayleighTheta;
    type GradientEta = RayleighEta;
    type Observation<'obs> = f64;
    type Workspace = ();

    fn workspace(&self) -> Self::Workspace {}
    fn theta(&self, eta: &Self::Eta, _workspace: &mut ()) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }
    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut ()) -> f64 {
        Self::nll_theta(y, *theta)
    }
    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut (),
    ) -> (f64, Self::GradientEta) {
        Self::nll_and_gradient_eta_values(y, *eta)
    }
}

impl<ScaleLink> InitialEtaFromObservations<1> for Rayleigh<ScaleLink>
where
    ScaleLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = f64> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| is_positive_finite(y).then_some(y));
        let mean = crate::initial::weighted_mean(&values).unwrap_or(1.0);
        let scale = positive_floor(mean / (std::f64::consts::PI / 2.0).sqrt());
        RayleighEta {
            scale: ScaleLink::initial_eta_from_theta(scale),
        }
    }
}

impl<ScaleLink> HasCdf for Rayleigh<ScaleLink>
where
    ScaleLink: PositiveLink<f64>,
{
    fn cdf(&self, y: f64, theta: &Self::Theta) -> f64 {
        if !y.is_finite() || !is_positive_finite(theta.scale) {
            return f64::NAN;
        }
        if y <= 0.0 {
            return 0.0;
        }
        -(-0.5 * (y / theta.scale).powi(2)).exp_m1()
    }
}

impl<ScaleLink> HasQuantile for Rayleigh<ScaleLink>
where
    ScaleLink: PositiveLink<f64>,
{
    fn quantile(&self, probability: f64, theta: &Self::Theta) -> f64 {
        if !is_probability(probability) || !is_positive_finite(theta.scale) {
            return f64::NAN;
        }
        theta.scale * (-2.0 * (-probability).ln_1p()).sqrt()
    }
}

#[cfg(feature = "rand")]
impl<Rng, ScaleLink> TrySimulate<Rng> for Rayleigh<ScaleLink>
where
    Rng: rand::Rng,
    ScaleLink: PositiveLink<f64>,
{
    type Sample = f64;

    fn try_sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Result<f64, SimulationError> {
        if !is_positive_finite(theta.scale) {
            return Err(SimulationError::InvalidParameters("Rayleigh theta"));
        }
        crate::simulation::try_sample_quantile(rng, self, theta, "Rayleigh sample")
    }
}

/// Link-scale predictor for [`Rayleigh`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayleighEta {
    /// Scale predictor.
    pub scale: f64,
}

impl ParameterParts<1> for RayleighEta {
    fn from_array(values: [f64; 1]) -> Self {
        Self { scale: values[0] }
    }
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.scale,
            _ => unreachable!("Rayleigh eta only has index 0"),
        }
    }
}

/// Natural-scale parameter for [`Rayleigh`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayleighTheta {
    /// Positive scale.
    pub scale: f64,
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{Family, HasCdf, HasQuantile};

    use super::{RayleighScale, RayleighTheta};
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn gradient_matches_finite_difference() {
        assert_gradient_matches_finite_difference::<_, 1>(&RayleighScale::new(), 1.7, [0.2]);
    }

    #[test]
    fn matches_closed_form() {
        let family = RayleighScale::new();
        let theta = RayleighTheta { scale: 1.3 };
        let probability = family.cdf(1.7, &theta);
        assert_relative_eq!(family.quantile(probability, &theta), 1.7, epsilon = 1.0e-12);
        assert!(family.nll(0.0, &theta, &mut ()).is_infinite());
    }
}
