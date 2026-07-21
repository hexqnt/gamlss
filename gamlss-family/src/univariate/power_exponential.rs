use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasQuantile, Identity, InitialEtaFromObservations, InitialEtaFromTheta, Link,
    Log, Mu, Nu, ObservationView, ParameterParts, PositiveLink, Sigma,
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

use gamlss_special::{
    digamma, invert_real_cdf, ln_gamma, ln_gamma_delta, regularized_gamma_lower,
    regularized_gamma_upper,
};

use crate::constants::LOG_2;
use crate::initial::{robust_location_scale, weighted_values};

#[inline]
pub(super) fn log_standardized_scale(power: f64) -> f64 {
    -0.5 * ln_gamma_delta(1.0 / power, 2.0 / power)
}

#[inline]
pub(super) fn standardized_scale(power: f64) -> f64 {
    log_standardized_scale(power).exp()
}

#[inline]
pub(super) fn d_log_standardized_scale_d_power(power: f64) -> f64 {
    3.0_f64.mul_add(digamma(3.0 / power), -digamma(1.0 / power)) / (2.0 * power * power)
}

#[inline]
pub(super) fn standardized_cdf(value: f64, power: f64) -> f64 {
    let scaled = value / standardized_scale(power);
    if scaled < 0.0 {
        0.5 * regularized_gamma_upper(1.0 / power, scaled.abs().powf(power))
    } else {
        let probability = regularized_gamma_lower(1.0 / power, scaled.powf(power));
        f64::midpoint(1.0, probability)
    }
}

/// Power exponential distribution with identity/log/log links.
pub type PowerExponentialMuSigmaNu = PowerExponential<Identity, Log, Log>;
/// Alias commonly used for the generalized error distribution.
pub type Ged<MuLink = Identity, SigmaLink = Log, NuLink = Log> =
    PowerExponential<MuLink, SigmaLink, NuLink>;
/// Default generalized error distribution alias.
pub type GedMuSigmaNu = PowerExponentialMuSigmaNu;
/// Power exponential / generalized error distribution.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/distributions/power_exponential.svg")
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PowerExponential<MuLink = Identity, SigmaLink = Log, NuLink = Log> {
    marker: PhantomData<(MuLink, SigmaLink, NuLink)>,
}

impl<MuLink, SigmaLink, NuLink> PowerExponential<MuLink, SigmaLink, NuLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: PositiveLink<f64>,
{
    /// Creates a stateless power exponential family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(eta: PowerExponentialEta) -> PowerExponentialTheta {
        PowerExponentialTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
            nu: NuLink::inverse(eta.nu),
        }
    }

    #[inline]
    fn scale_c(nu: f64) -> f64 {
        standardized_scale(nu)
    }

    #[inline]
    fn d_log_scale_c_d_nu(nu: f64) -> f64 {
        d_log_standardized_scale_d_power(nu)
    }

    #[inline]
    fn nll_theta(y: f64, theta: PowerExponentialTheta) -> f64 {
        if !y.is_finite()
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || theta.nu <= 0.0
            || !theta.nu.is_finite()
        {
            return f64::INFINITY;
        }

        let log_c = log_standardized_scale(theta.nu);
        let c = log_c.exp();
        let z = ((y - theta.mu) / (c * theta.sigma)).abs();
        LOG_2 + log_c + theta.sigma.ln() + ln_gamma(1.0 / theta.nu) - theta.nu.ln()
            + z.powf(theta.nu)
    }

    #[inline]
    fn gradient_theta(y: f64, theta: PowerExponentialTheta) -> PowerExponentialTheta {
        let c = Self::scale_c(theta.nu);
        let residual = y - theta.mu;
        let abs_residual = residual.abs();
        let z = abs_residual / (c * theta.sigma);
        let z_power = z.powf(theta.nu);
        let d_log_z = theta.nu * z_power;
        let d_log_c = Self::d_log_scale_c_d_nu(theta.nu);
        let d_log_gamma_inv_nu = -digamma(1.0 / theta.nu) / (theta.nu * theta.nu);
        let d_power = if abs_residual == 0.0 {
            0.0
        } else {
            z_power * theta.nu.mul_add(-d_log_c, z.ln())
        };
        PowerExponentialTheta {
            mu: if residual == 0.0 {
                0.0
            } else {
                -d_log_z / residual
            },
            sigma: (1.0 - d_log_z) / theta.sigma,
            nu: d_log_c + d_log_gamma_inv_nu - 1.0 / theta.nu + d_power,
        }
    }

    #[inline]
    fn nll_and_gradient_eta_values(y: f64, eta: PowerExponentialEta) -> (f64, PowerExponentialEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (
                nll,
                PowerExponentialEta {
                    mu: f64::NAN,
                    sigma: f64::NAN,
                    nu: f64::NAN,
                },
            );
        }

        let gradient = Self::gradient_theta(y, theta);
        (
            nll,
            PowerExponentialEta {
                mu: gradient.mu * MuLink::derivative_inverse(eta.mu),
                sigma: gradient.sigma * SigmaLink::derivative_inverse(eta.sigma),
                nu: gradient.nu * NuLink::derivative_inverse(eta.nu),
            },
        )
    }
}

impl<MuLink, SigmaLink, NuLink> Default for PowerExponential<MuLink, SigmaLink, NuLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MuLink, SigmaLink, NuLink> for PowerExponential<MuLink, SigmaLink, NuLink>;
    parameters = (Mu, Sigma, Nu);
    arity = 3;
);

impl<MuLink, SigmaLink, NuLink> Family for PowerExponential<MuLink, SigmaLink, NuLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: PositiveLink<f64>,
{
    type Eta = PowerExponentialEta;
    type Theta = PowerExponentialTheta;
    type GradientEta = PowerExponentialEta;
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

impl<MuLink, SigmaLink, NuLink> InitialEtaFromObservations<3>
    for PowerExponential<MuLink, SigmaLink, NuLink>
where
    MuLink: InitialEtaFromTheta<f64> + Link<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    NuLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| y.is_finite().then_some(y));
        let Some((mu, sigma)) = robust_location_scale(&values) else {
            return PowerExponentialEta::from_array([0.0, 0.0, 2.0_f64.ln()]);
        };

        PowerExponentialEta {
            mu: MuLink::initial_eta_from_theta(mu),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
            nu: NuLink::initial_eta_from_theta(2.0),
        }
    }
}

impl<MuLink, SigmaLink, NuLink> HasCdf for PowerExponential<MuLink, SigmaLink, NuLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: PositiveLink<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        if !y.is_finite()
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || theta.nu <= 0.0
            || !theta.nu.is_finite()
        {
            return f64::NAN;
        }

        standardized_cdf((y - theta.mu) / theta.sigma, theta.nu)
    }
}

impl<MuLink, SigmaLink, NuLink> HasQuantile for PowerExponential<MuLink, SigmaLink, NuLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: PositiveLink<f64>,
{
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        if theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || !theta.mu.is_finite()
            || theta.nu <= 0.0
            || !theta.nu.is_finite()
        {
            return f64::NAN;
        }

        invert_real_cdf(p, |y| self.cdf(y, theta))
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink, SigmaLink, NuLink> TrySimulate<Rng>
    for PowerExponential<MuLink, SigmaLink, NuLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: PositiveLink<f64>,
{
    type Sample = f64;

    fn try_sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Result<f64, SimulationError> {
        if !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || theta.nu <= 0.0
            || !theta.nu.is_finite()
        {
            return Err(SimulationError::InvalidParameters(
                "power exponential theta",
            ));
        }

        let distribution = rand_distr::Gamma::new(1.0 / theta.nu, 1.0)
            .map_err(|_| SimulationError::BackendRejected("power exponential shape"))?;
        let radius = rand_distr::Distribution::sample(&distribution, rng).powf(1.0 / theta.nu);
        let sample = (crate::simulation::fair_sign(rng) * Self::scale_c(theta.nu) * theta.sigma)
            .mul_add(radius, theta.mu);
        if sample.is_finite() {
            Ok(sample)
        } else {
            Err(SimulationError::NumericalFailure(
                "power exponential transform",
            ))
        }
    }
}

/// Predictors for the power exponential family on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerExponentialEta {
    /// Location predictor.
    pub mu: f64,
    /// Standard-deviation scale predictor.
    pub sigma: f64,
    /// Tail-shape predictor.
    pub nu: f64,
}

impl ParameterParts<3> for PowerExponentialEta {
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
            _ => unreachable!("power exponential eta only has indices 0 through 2"),
        }
    }
}

/// Natural-scale power exponential parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerExponentialTheta {
    /// Location parameter.
    pub mu: f64,
    /// Positive standard-deviation scale.
    pub sigma: f64,
    /// Positive tail shape.
    pub nu: f64,
}

#[cfg(test)]
mod tests {
    use gamlss_core::HasCdf;
    #[cfg(feature = "rand")]
    use gamlss_core::TrySimulate;

    use super::{PowerExponentialMuSigmaNu, PowerExponentialTheta};

    #[test]
    fn power_exponential_preserves_far_left_tail_probability() {
        let family = PowerExponentialMuSigmaNu::new();
        let cdf = family.cdf(
            -10.0,
            &PowerExponentialTheta {
                mu: 0.0,
                sigma: 1.0,
                nu: 2.0,
            },
        );

        assert!(cdf > 0.0);
        assert!(cdf < 1.0e-20);
    }

    #[cfg(feature = "rand")]
    #[test]
    fn power_exponential_sampling_returns_finite_values_and_errors_for_invalid_theta() {
        use rand::SeedableRng;

        let family = PowerExponentialMuSigmaNu::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        assert!(
            family
                .try_sample(
                    &mut rng,
                    &PowerExponentialTheta {
                        mu: 0.4,
                        sigma: 1.5,
                        nu: 1.4,
                    }
                )
                .is_ok_and(f64::is_finite)
        );
        assert!(
            family
                .try_sample(
                    &mut rng,
                    &PowerExponentialTheta {
                        mu: 0.4,
                        sigma: 0.0,
                        nu: 1.4,
                    }
                )
                .is_err()
        );
    }
}
