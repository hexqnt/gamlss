use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Family, HasCdf, HasQuantile, Identity, InitialEtaFromObservations, InitialEtaFromTheta, Link,
    Log, Mu, Nu, ObservationView, ParameterParts, PositiveLink, ScalarParams, Sigma,
};

use crate::initial::{robust_location_scale, weighted_values};

use super::{
    cdf_location_scale, nll_gradient_location_scale, nll_location_scale, quantile_location_scale,
};

/// Skew-normal distribution with identity/log/identity links.
pub type SkewNormalMuSigmaNu = SkewNormal<Identity, Log, Identity>;

/// Azzalini/SN1-style skew-normal family with location, scale and skewness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkewNormal<MuLink = Identity, SigmaLink = Log, NuLink = Identity> {
    marker: PhantomData<(MuLink, SigmaLink, NuLink)>,
}

impl<MuLink, SigmaLink, NuLink> SkewNormal<MuLink, SigmaLink, NuLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    /// Creates a stateless skew-normal family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(eta: SkewNormalEta) -> SkewNormalTheta {
        SkewNormalTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
            nu: NuLink::inverse(eta.nu),
        }
    }

    #[inline]
    fn nll_theta(y: f64, theta: SkewNormalTheta) -> f64 {
        nll_location_scale(y, theta.mu, theta.sigma, theta.nu)
    }

    #[inline]
    fn nll_and_gradient_eta_values(y: f64, eta: SkewNormalEta) -> (f64, SkewNormalEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (nll, SkewNormalEta::from_array([f64::NAN; 3]));
        }

        let gradient = nll_gradient_location_scale(y, theta.mu, theta.sigma, theta.nu);
        (
            nll,
            SkewNormalEta {
                mu: gradient.mu * MuLink::derivative_inverse(eta.mu),
                sigma: gradient.sigma * SigmaLink::derivative_inverse(eta.sigma),
                nu: gradient.nu * NuLink::derivative_inverse(eta.nu),
            },
        )
    }
}

impl<MuLink, SigmaLink, NuLink> Default for SkewNormal<MuLink, SigmaLink, NuLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<MuLink, SigmaLink, NuLink> Family for SkewNormal<MuLink, SigmaLink, NuLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    type Eta = SkewNormalEta;
    type Theta = SkewNormalTheta;
    type GradientEta = SkewNormalEta;
    type Observation<'obs> = f64;
    type Workspace = ();
    type ParamSpec = ScalarParams<(Mu, Sigma, Nu), (MuLink, SigmaLink, NuLink), 3>;

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

impl<MuLink, SigmaLink, NuLink> InitialEtaFromObservations<3>
    for SkewNormal<MuLink, SigmaLink, NuLink>
where
    MuLink: InitialEtaFromTheta<f64> + Link<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    NuLink: InitialEtaFromTheta<f64> + Link<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| y.is_finite().then_some(y));
        let Some((mu, sigma)) = robust_location_scale(&values) else {
            return SkewNormalEta::from_array([0.0, 0.0, 0.0]);
        };

        SkewNormalEta {
            mu: MuLink::initial_eta_from_theta(mu),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
            nu: NuLink::initial_eta_from_theta(0.0),
        }
    }
}

impl<MuLink, SigmaLink, NuLink> HasCdf for SkewNormal<MuLink, SigmaLink, NuLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        cdf_location_scale(y, theta.mu, theta.sigma, theta.nu)
    }
}

impl<MuLink, SigmaLink, NuLink> HasQuantile for SkewNormal<MuLink, SigmaLink, NuLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        quantile_location_scale(p, theta.mu, theta.sigma, theta.nu)
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink, SigmaLink, NuLink> CanSimulate<Rng> for SkewNormal<MuLink, SigmaLink, NuLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    type Sample = f64;

    fn sample(&self, rng: &mut Rng, theta: &Self::Theta) -> f64 {
        super::sample_location_scale(rng, theta.mu, theta.sigma, theta.nu)
    }
}

/// Predictors for the skew-normal family on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkewNormalEta {
    /// Location predictor.
    pub mu: f64,
    /// Scale predictor.
    pub sigma: f64,
    /// Skewness predictor.
    pub nu: f64,
}

impl ParameterParts<3> for SkewNormalEta {
    #[inline]
    fn from_array(values: [f64; 3]) -> Self {
        Self {
            mu: values[0],
            sigma: values[1],
            nu: values[2],
        }
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mu,
            1 => self.sigma,
            2 => self.nu,
            _ => unreachable!("skew-normal eta only has indices 0 through 2"),
        }
    }
}

/// Natural-scale skew-normal parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkewNormalTheta {
    /// Location parameter.
    pub mu: f64,
    /// Positive scale parameter.
    pub sigma: f64,
    /// Skewness parameter.
    pub nu: f64,
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;

    use super::{SkewNormalMuSigmaNu, SkewNormalTheta};

    #[cfg(feature = "rand")]
    #[test]
    fn skew_normal_sampling_returns_finite_values_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = SkewNormalMuSigmaNu::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        assert!(
            family
                .sample(
                    &mut rng,
                    &SkewNormalTheta {
                        mu: 0.4,
                        sigma: 1.5,
                        nu: 2.0,
                    }
                )
                .is_finite()
        );
        assert!(
            family
                .sample(
                    &mut rng,
                    &SkewNormalTheta {
                        mu: 0.4,
                        sigma: 0.0,
                        nu: 2.0,
                    }
                )
                .is_nan()
        );
    }
}
