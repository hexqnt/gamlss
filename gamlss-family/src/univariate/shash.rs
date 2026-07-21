use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasQuantile, Identity, InitialEtaFromObservations, InitialEtaFromTheta, Link,
    Log, Mu, Nu, ObservationView, ParameterParts, PositiveLink, Sigma, Tau,
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

use gamlss_special::{unit_normal_cdf, unit_normal_log_pdf, unit_normal_quantile};

use crate::{
    initial::{robust_location_scale, weighted_values},
    shash_kernel as kernel,
};

/// SHASH distribution with identity/log/log/log links.
pub type ShashMuSigmaNuTau = Shash<Identity, Log, Log, Log>;
/// Full-name alias for [`ShashMuSigmaNuTau`].
pub type SinhArcsinhMuSigmaNuTau = ShashMuSigmaNuTau;
/// Full-name alias for [`Shash`].
pub type SinhArcsinh<MuLink = Identity, SigmaLink = Log, NuLink = Log, TauLink = Log> =
    Shash<MuLink, SigmaLink, NuLink, TauLink>;
/// Full-name alias for [`ShashEta`].
pub type SinhArcsinhEta = ShashEta;
/// Full-name alias for [`ShashTheta`].
pub type SinhArcsinhTheta = ShashTheta;

/// Sinh-arcsinh-normal (SHASH) family using positive skew-ratio and tail parameters.
///
/// For standardized `x = (y - mu) / sigma`, the normalizing transformation is `z = sinh(tau * asinh(x) - ln(nu))`. In the original [Jones--Pewsey] notation, `epsilon = -ln(nu)` and `delta = tau`; consequently the default `nu` predictor `ln(nu)` increases in the direction of positive response skewness.
///
/// `nu = 1` is symmetric. Together, `nu = 1` and `tau = 1` reduce the family to `Normal(mu, sigma)`. Values `tau < 1` give heavier tails than the normal and values `tau > 1` give lighter tails. `mu` and `sigma` are transformation location and scale, not generally the distribution mean and standard deviation when `nu != 1` or `tau != 1`.
///
/// ### Parameterization examples
///
/// [Jones--Pewsey]: https://doi.org/10.1093/biomet/asp053
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/distributions/shash.svg")
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
    fn transformed_z(y: f64, theta: ShashTheta) -> f64 {
        Self::transform(y, theta).1.latent
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn transform(y: f64, theta: ShashTheta) -> (f64, kernel::Transform) {
        let x = (y - theta.mu) / theta.sigma;
        (x, kernel::transform_standardized(x, theta.nu, theta.tau))
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

        let (x, transformed) = Self::transform(y, theta);
        if !transformed.h.is_finite() || !transformed.latent.is_finite() {
            return f64::INFINITY;
        }

        theta.sigma.ln() - theta.tau.ln() + x.hypot(1.0).ln()
            - kernel::log_cosh(transformed.h)
            - unit_normal_log_pdf(transformed.latent)
    }

    #[inline]
    fn gradient_eta(y: f64, eta: ShashEta, theta: ShashTheta) -> ShashEta {
        let (x, transformed) = Self::transform(y, theta);
        let cosh_h = transformed.latent.hypot(1.0);
        let d_h = transformed
            .latent
            .mul_add(cosh_h, -transformed.latent / cosh_h);
        let inverse_hypot = 1.0 / x.hypot(1.0);
        let d_x = (d_h * theta.tau).mul_add(inverse_hypot, (x * inverse_hypot) * inverse_hypot);

        ShashEta {
            mu: -d_x / theta.sigma * MuLink::derivative_inverse(eta.mu),
            sigma: x.mul_add(-d_x, 1.0) * SigmaLink::derivative_log_inverse(eta.sigma),
            nu: -d_h * NuLink::derivative_log_inverse(eta.nu),
            tau: (d_h * theta.tau).mul_add(transformed.asinh_x, -1.0)
                * TauLink::derivative_log_inverse(eta.tau),
        }
    }

    #[inline]
    fn nll_and_gradient_eta_values(y: f64, eta: ShashEta) -> (f64, ShashEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (nll, ShashEta::from_array([f64::NAN; 4]));
        }

        (nll, Self::gradient_eta(y, eta, theta))
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

        unit_normal_cdf(Self::transformed_z(y, *theta))
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
        theta.mu + theta.sigma * kernel::inverse_standardized(z, theta.nu, theta.tau)
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink, SigmaLink, NuLink, TauLink> TrySimulate<Rng>
    for Shash<MuLink, SigmaLink, NuLink, TauLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    type Sample = f64;

    fn try_sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Result<f64, SimulationError> {
        if theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || !theta.mu.is_finite()
            || theta.nu <= 0.0
            || !theta.nu.is_finite()
            || theta.tau <= 0.0
            || !theta.tau.is_finite()
        {
            return Err(SimulationError::InvalidParameters("SHASH theta"));
        }

        let z = crate::simulation::standard_normal(rng);
        let sample = theta.sigma.mul_add(
            kernel::inverse_standardized(z, theta.nu, theta.tau),
            theta.mu,
        );
        if sample.is_finite() {
            Ok(sample)
        } else {
            Err(SimulationError::NumericalFailure("SHASH transform"))
        }
    }
}

/// Predictors for SHASH on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShashEta {
    /// Location predictor.
    pub mu: f64,
    /// Scale predictor.
    pub sigma: f64,
    /// Skew-ratio predictor; with the default log link this is `ln(nu)`.
    pub nu: f64,
    /// Tail-parameter predictor; the link maps it to a positive natural parameter.
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
    /// Positive skew ratio; `ln(nu)` is the signed skew predictor in this parameterization.
    pub nu: f64,
    /// Positive Jones--Pewsey tail parameter.
    pub tau: f64,
}

#[cfg(test)]
mod tests {
    use gamlss_core::Family;
    #[cfg(feature = "rand")]
    use gamlss_core::TrySimulate;

    use super::{ShashEta, ShashMuSigmaNuTau, ShashTheta};

    #[test]
    fn shash_extreme_finite_observation_remains_numerically_representable() {
        let family = ShashMuSigmaNuTau::new();
        let eta = ShashEta {
            mu: 0.0,
            sigma: 0.0,
            nu: 0.0,
            tau: 0.001_f64.ln(),
        };
        let (nll, gradient) = family.nll_and_gradient_eta(f64::MAX, &eta, &mut family.workspace());

        assert!(nll.is_finite());
        assert!(gradient.mu.is_finite());
        assert!(gradient.sigma.is_finite());
        assert!(gradient.nu.is_finite());
        assert!(gradient.tau.is_finite());
    }

    #[test]
    fn shash_log_link_gradient_remains_finite_for_subnormal_positive_parameters() {
        let family = ShashMuSigmaNuTau::new();
        for eta in [
            ShashEta {
                mu: 0.0,
                sigma: -744.0,
                nu: 0.0,
                tau: 0.0,
            },
            ShashEta {
                mu: 0.0,
                sigma: 0.0,
                nu: 0.0,
                tau: -744.0,
            },
        ] {
            let (nll, gradient) = family.nll_and_gradient_eta(0.0, &eta, &mut family.workspace());
            assert!(nll.is_finite());
            assert!(gradient.mu.is_finite());
            assert!(gradient.sigma.is_finite());
            assert!(gradient.nu.is_finite());
            assert!(gradient.tau.is_finite());
        }
    }

    #[test]
    fn shash_overflowing_normal_transform_returns_infinite_nll() {
        let family = ShashMuSigmaNuTau::new();
        let nll = family.nll(
            1.0,
            &ShashTheta {
                mu: 0.0,
                sigma: 1.0,
                nu: 1.0,
                tau: 1_000.0,
            },
            &mut family.workspace(),
        );

        assert!(nll.is_infinite() && nll.is_sign_positive());
    }

    #[cfg(feature = "rand")]
    #[test]
    fn shash_sampling_returns_finite_values_and_errors_for_invalid_theta() {
        use rand::SeedableRng;

        let family = ShashMuSigmaNuTau::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        assert!(
            family
                .try_sample(
                    &mut rng,
                    &ShashTheta {
                        mu: 0.4,
                        sigma: 1.5,
                        nu: 1.2,
                        tau: 0.8,
                    }
                )
                .is_ok_and(f64::is_finite)
        );
        assert!(
            family
                .try_sample(
                    &mut rng,
                    &ShashTheta {
                        mu: 0.4,
                        sigma: 0.0,
                        nu: 1.2,
                        tau: 0.8,
                    }
                )
                .is_err()
        );
    }
}
