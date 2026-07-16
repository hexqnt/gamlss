use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Family, HasCdf, HasQuantile, Identity, InitialEtaFromObservations, InitialEtaFromTheta, Link,
    Log, Mu, Nu, ObservationView, ParameterParts, PositiveLink, Sigma, Tau,
};

use gamlss_special::{unit_normal_cdf, unit_normal_log_pdf, unit_normal_quantile};

use crate::initial::{robust_location_scale, weighted_values};

/// SHASH distribution with identity/log/log/log links.
pub type ShashMuSigmaNuTau = Shash<Identity, Log, Log, Log>;
/// Sinh-arcsinh family using positive skewness and tail parameters.
///
/// `nu = 1` and `tau = 1` reduce the standardized distribution to the
/// standard normal. Values of `nu` above or below one skew the distribution
/// through `ln(nu)`, which keeps the default log link centered at the symmetric
/// case.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/shash.svg")
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shash<MuLink = Identity, SigmaLink = Log, NuLink = Log, TauLink = Log> {
    marker: PhantomData<(MuLink, SigmaLink, NuLink, TauLink)>,
}

impl<MuLink, SigmaLink, NuLink, TauLink> Shash<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    /// Creates a stateless SHASH family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(eta: ShashEta) -> ShashTheta {
        ShashTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
            nu: NuLink::inverse(eta.nu),
            tau: TauLink::inverse(eta.tau),
        }
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn transformed_z(y: f64, theta: ShashTheta) -> (f64, f64) {
        let (x, _, z) = Self::transformed_x_h_z(y, theta);
        (x, z)
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn transformed_x_h_z(y: f64, theta: ShashTheta) -> (f64, f64, f64) {
        let x = (y - theta.mu) / theta.sigma;
        let h = theta.tau * x.asinh() - theta.nu.ln();
        (x, h, h.sinh())
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn nll_theta(y: f64, theta: ShashTheta) -> f64 {
        if !y.is_finite()
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || theta.nu <= 0.0
            || !theta.nu.is_finite()
            || theta.tau <= 0.0
            || !theta.tau.is_finite()
        {
            return f64::INFINITY;
        }

        let (x, h, z) = Self::transformed_x_h_z(y, theta);
        theta.sigma.ln() - theta.tau.ln() + 0.5 * (x * x).ln_1p()
            - h.cosh().ln()
            - unit_normal_log_pdf(z)
    }

    #[inline]
    fn gradient_theta(y: f64, theta: ShashTheta) -> ShashTheta {
        let (x, h, _) = Self::transformed_x_h_z(y, theta);
        let asinh_x = x.asinh();
        let sinh_h = h.sinh();
        let cosh_h = h.cosh();
        let d_h = sinh_h.mul_add(cosh_h, -h.tanh());
        let inv_one_plus_x2 = 1.0 / x.mul_add(x, 1.0);
        let d_x = (d_h * theta.tau).mul_add(inv_one_plus_x2.sqrt(), x * inv_one_plus_x2);

        ShashTheta {
            mu: -d_x / theta.sigma,
            sigma: x.mul_add(-d_x, 1.0) / theta.sigma,
            nu: -d_h / theta.nu,
            tau: d_h.mul_add(asinh_x, -1.0 / theta.tau),
        }
    }

    #[inline]
    fn nll_and_gradient_eta_values(y: f64, eta: ShashEta) -> (f64, ShashEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (nll, ShashEta::from_array([f64::NAN; 4]));
        }

        let gradient = Self::gradient_theta(y, theta);
        (
            nll,
            ShashEta {
                mu: gradient.mu * MuLink::derivative_inverse(eta.mu),
                sigma: gradient.sigma * SigmaLink::derivative_inverse(eta.sigma),
                nu: gradient.nu * NuLink::derivative_inverse(eta.nu),
                tau: gradient.tau * TauLink::derivative_inverse(eta.tau),
            },
        )
    }
}

impl<MuLink, SigmaLink, NuLink, TauLink> Default for Shash<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MuLink, SigmaLink, NuLink, TauLink> for Shash<MuLink, SigmaLink, NuLink, TauLink>;
    parameters = (Mu, Sigma, Nu, Tau);
    arity = 4;
);

impl<MuLink, SigmaLink, NuLink, TauLink> Family for Shash<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    type Eta = ShashEta;
    type Theta = ShashTheta;
    type GradientEta = ShashEta;
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

impl<MuLink, SigmaLink, NuLink, TauLink> InitialEtaFromObservations<4>
    for Shash<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: InitialEtaFromTheta<f64> + Link<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    NuLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    TauLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| y.is_finite().then_some(y));
        let Some((mu, sigma)) = robust_location_scale(&values) else {
            return ShashEta::from_array([0.0, 0.0, 0.0, 0.0]);
        };

        ShashEta {
            mu: MuLink::initial_eta_from_theta(mu),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
            nu: NuLink::initial_eta_from_theta(1.0),
            tau: TauLink::initial_eta_from_theta(1.0),
        }
    }
}

impl<MuLink, SigmaLink, NuLink, TauLink> HasCdf for Shash<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        if !y.is_finite()
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || theta.nu <= 0.0
            || !theta.nu.is_finite()
            || theta.tau <= 0.0
            || !theta.tau.is_finite()
        {
            return f64::NAN;
        }

        unit_normal_cdf(Self::transformed_z(y, *theta).1)
    }
}

impl<MuLink, SigmaLink, NuLink, TauLink> HasQuantile for Shash<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    #[allow(clippy::suboptimal_flops)]
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        if theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || !theta.mu.is_finite()
            || theta.nu <= 0.0
            || !theta.nu.is_finite()
            || theta.tau <= 0.0
            || !theta.tau.is_finite()
        {
            return f64::NAN;
        }

        let z = unit_normal_quantile(p);
        theta.mu + theta.sigma * ((z.asinh() + theta.nu.ln()) / theta.tau).sinh()
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink, SigmaLink, NuLink, TauLink> CanSimulate<Rng>
    for Shash<MuLink, SigmaLink, NuLink, TauLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    type Sample = f64;

    fn sample(&self, rng: &mut Rng, theta: &Self::Theta) -> f64 {
        if theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || !theta.mu.is_finite()
            || theta.nu <= 0.0
            || !theta.nu.is_finite()
            || theta.tau <= 0.0
            || !theta.tau.is_finite()
        {
            return f64::NAN;
        }

        let z = crate::simulation::standard_normal(rng);
        theta
            .sigma
            .mul_add(((z.asinh() + theta.nu.ln()) / theta.tau).sinh(), theta.mu)
    }
}

/// Predictors for SHASH on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShashEta {
    /// Location predictor.
    pub mu: f64,
    /// Scale predictor.
    pub sigma: f64,
    /// Positive skewness predictor.
    pub nu: f64,
    /// Positive tail predictor.
    pub tau: f64,
}

impl ParameterParts<4> for ShashEta {
    #[inline]
    fn from_array(values: [f64; 4]) -> Self {
        Self {
            mu: values[0],
            sigma: values[1],
            nu: values[2],
            tau: values[3],
        }
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mu,
            1 => self.sigma,
            2 => self.nu,
            3 => self.tau,
            _ => unreachable!("shash eta only has indices 0 through 3"),
        }
    }
}

/// Natural-scale SHASH parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShashTheta {
    /// Location parameter.
    pub mu: f64,
    /// Positive scale parameter.
    pub sigma: f64,
    /// Positive skewness parameter.
    pub nu: f64,
    /// Positive tail parameter.
    pub tau: f64,
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;

    #[cfg(feature = "rand")]
    use super::{ShashMuSigmaNuTau, ShashTheta};

    #[cfg(feature = "rand")]
    #[test]
    fn shash_sampling_returns_finite_values_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = ShashMuSigmaNuTau::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        assert!(
            family
                .sample(
                    &mut rng,
                    &ShashTheta {
                        mu: 0.4,
                        sigma: 1.5,
                        nu: 1.2,
                        tau: 0.8,
                    }
                )
                .is_finite()
        );
        assert!(
            family
                .sample(
                    &mut rng,
                    &ShashTheta {
                        mu: 0.4,
                        sigma: 0.0,
                        nu: 1.2,
                        tau: 0.8,
                    }
                )
                .is_nan()
        );
    }
}
