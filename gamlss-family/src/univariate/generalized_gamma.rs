use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Family, HasCdf, HasQuantile, Identity, InitialEtaFromObservations, InitialEtaFromTheta, Link,
    Log, Nu, ObservationView, ParameterParts, PositiveLink, ScalarParams, Scale, Sigma,
};

use gamlss_special::{
    digamma, invert_positive_cdf, ln_gamma, regularized_gamma_lower, unit_normal_cdf,
};

use crate::constants::HALF_LOG_2_PI;
use crate::initial::{positive_floor, weighted_summary, weighted_values};

const NU_EPSILON: f64 = 1.0e-4;

/// Generalized gamma scale/sigma/nu distribution with log/log/identity links.
pub type GeneralizedGammaScaleSigmaNu = GeneralizedGamma<Log, Log, Identity>;
/// Generalized gamma family with scale, sigma, and shape parameters.
///
/// The first parameter is the positive scale/location used in `(y / scale)`,
/// not the arithmetic mean except in special cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneralizedGamma<ScaleLink = Log, SigmaLink = Log, NuLink = Identity> {
    marker: PhantomData<(ScaleLink, SigmaLink, NuLink)>,
}

impl<ScaleLink, SigmaLink, NuLink> GeneralizedGamma<ScaleLink, SigmaLink, NuLink>
where
    ScaleLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    /// Creates a stateless generalized gamma family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(eta: GeneralizedGammaEta) -> GeneralizedGammaTheta {
        GeneralizedGammaTheta {
            mu: ScaleLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
            nu: NuLink::inverse(eta.nu),
        }
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn nll_theta(y: f64, theta: GeneralizedGammaTheta) -> f64 {
        if y <= 0.0
            || !y.is_finite()
            || theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || !theta.nu.is_finite()
        {
            return f64::INFINITY;
        }
        if theta.nu.abs() < NU_EPSILON {
            let log_ratio = (y / theta.mu).ln();
            let z = log_ratio / theta.sigma;
            let d_nu = Self::log_normal_limit_nu_score(log_ratio, theta.sigma);
            return y.ln() + theta.sigma.ln() + HALF_LOG_2_PI + 0.5 * z * z + theta.nu * d_nu;
        }

        let abs_nu = theta.nu.abs();
        let k = 1.0 / (theta.sigma * theta.sigma * abs_nu * abs_nu);
        let z = (y / theta.mu).powf(theta.nu);
        -(k * k.ln() + k * z.ln() + abs_nu.ln() - k * z - ln_gamma(k) - y.ln())
    }

    #[inline]
    fn log_normal_limit_nu_score(log_ratio: f64, sigma: f64) -> f64 {
        log_ratio * log_ratio * log_ratio / (6.0 * sigma * sigma) + sigma * sigma / 12.0
    }

    #[inline]
    fn gradient_theta(y: f64, theta: GeneralizedGammaTheta) -> GeneralizedGammaTheta {
        let log_ratio = (y / theta.mu).ln();
        if theta.nu.abs() < NU_EPSILON {
            return GeneralizedGammaTheta {
                mu: -log_ratio / (theta.mu * theta.sigma * theta.sigma),
                sigma: 1.0 / theta.sigma
                    - log_ratio * log_ratio / (theta.sigma * theta.sigma * theta.sigma),
                nu: Self::log_normal_limit_nu_score(log_ratio, theta.sigma),
            };
        }

        let k = 1.0 / (theta.sigma * theta.sigma * theta.nu * theta.nu);
        let log_z = theta.nu * log_ratio;
        let z = log_z.exp();
        let d_k = digamma(k) - k.ln() - 1.0 - log_z + z;
        let d_log_z = k * (z - 1.0);

        GeneralizedGammaTheta {
            mu: -d_log_z * theta.nu / theta.mu,
            sigma: d_k * (-2.0 * k / theta.sigma),
            nu: d_k * (-2.0 * k / theta.nu) + d_log_z * log_ratio - 1.0 / theta.nu,
        }
    }

    #[inline]
    fn nll_and_gradient_eta_values(y: f64, eta: GeneralizedGammaEta) -> (f64, GeneralizedGammaEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (nll, GeneralizedGammaEta::from_array([f64::NAN; 3]));
        }

        let gradient = Self::gradient_theta(y, theta);
        (
            nll,
            GeneralizedGammaEta {
                mu: gradient.mu * ScaleLink::derivative_inverse(eta.mu),
                sigma: gradient.sigma * SigmaLink::derivative_inverse(eta.sigma),
                nu: gradient.nu * NuLink::derivative_inverse(eta.nu),
            },
        )
    }
}

impl<ScaleLink, SigmaLink, NuLink> Default for GeneralizedGamma<ScaleLink, SigmaLink, NuLink>
where
    ScaleLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<ScaleLink, SigmaLink, NuLink> Family for GeneralizedGamma<ScaleLink, SigmaLink, NuLink>
where
    ScaleLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    type Eta = GeneralizedGammaEta;
    type Theta = GeneralizedGammaTheta;
    type GradientEta = GeneralizedGammaEta;
    type Observation<'obs> = f64;
    type Workspace = ();
    type ParamSpec = ScalarParams<(Scale, Sigma, Nu), (ScaleLink, SigmaLink, NuLink), 3>;

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

impl<ScaleLink, SigmaLink, NuLink> InitialEtaFromObservations<3>
    for GeneralizedGamma<ScaleLink, SigmaLink, NuLink>
where
    ScaleLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    NuLink: InitialEtaFromTheta<f64> + Link<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values =
            weighted_values::<Self, _, _>(obs, |y| (y.is_finite() && y > 0.0).then_some(y));
        let Some(summary) = weighted_summary(&values) else {
            return GeneralizedGammaEta::from_array([0.0, 0.0, 0.0]);
        };
        let scale = positive_floor(summary.mean);
        let sigma = positive_floor((summary.variance.sqrt() / scale).max(1.0e-3));

        GeneralizedGammaEta {
            mu: ScaleLink::initial_eta_from_theta(scale),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
            nu: NuLink::initial_eta_from_theta(1.0),
        }
    }
}

impl<ScaleLink, SigmaLink, NuLink> HasCdf for GeneralizedGamma<ScaleLink, SigmaLink, NuLink>
where
    ScaleLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        if !y.is_finite()
            || theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || !theta.nu.is_finite()
        {
            return f64::NAN;
        }
        if y <= 0.0 {
            return 0.0;
        }
        if theta.nu.abs() < NU_EPSILON {
            return unit_normal_cdf((y / theta.mu).ln() / theta.sigma);
        }

        let abs_nu = theta.nu.abs();
        let k = 1.0 / (theta.sigma * theta.sigma * abs_nu * abs_nu);
        let x = k * (y / theta.mu).powf(theta.nu);
        let p = regularized_gamma_lower(k, x);
        if theta.nu > 0.0 { p } else { 1.0 - p }
    }
}

impl<ScaleLink, SigmaLink, NuLink> HasQuantile for GeneralizedGamma<ScaleLink, SigmaLink, NuLink>
where
    ScaleLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        if theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || !theta.nu.is_finite()
        {
            return f64::NAN;
        }

        invert_positive_cdf(p, |y| self.cdf(y, theta))
    }
}

#[cfg(feature = "rand")]
impl<Rng, ScaleLink, SigmaLink, NuLink> CanSimulate<Rng>
    for GeneralizedGamma<ScaleLink, SigmaLink, NuLink>
where
    Rng: rand::Rng,
    ScaleLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    type Sample = f64;

    fn sample(&self, rng: &mut Rng, theta: &Self::Theta) -> f64 {
        if theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || !theta.nu.is_finite()
        {
            return f64::NAN;
        }

        if theta.nu.abs() < NU_EPSILON {
            let z = crate::simulation::standard_normal(rng);
            return theta.mu * (theta.sigma * z).exp();
        }

        let abs_nu = theta.nu.abs();
        let k = 1.0 / (theta.sigma * theta.sigma * abs_nu * abs_nu);
        let z = rand_distr::Distribution::sample(
            &rand_distr::Gamma::new(k, 1.0 / k)
                .expect("validated generalized gamma parameters must construct"),
            rng,
        );
        theta.mu * z.powf(1.0 / theta.nu)
    }
}

/// Predictors for generalized gamma on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneralizedGammaEta {
    /// Scale/location predictor.
    ///
    /// This controls the positive parameter used in `(y / scale)`; it is not
    /// generally the arithmetic mean.
    ///
    /// The field name is retained for compatibility with existing code.
    pub mu: f64,
    /// Scale predictor.
    pub sigma: f64,
    /// Shape predictor.
    pub nu: f64,
}

impl ParameterParts<3> for GeneralizedGammaEta {
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
            _ => unreachable!("generalized gamma eta only has indices 0 through 2"),
        }
    }
}

/// Natural-scale generalized gamma parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneralizedGammaTheta {
    /// Positive scale/location parameter.
    ///
    /// This is the positive parameter used in `(y / scale)`; it is not
    /// generally the arithmetic mean.
    ///
    /// The field name is retained for compatibility with existing code.
    pub mu: f64,
    /// Positive scale parameter.
    pub sigma: f64,
    /// Shape parameter; `nu = 0` is the log-normal limit.
    pub nu: f64,
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;

    use super::{GeneralizedGammaScaleSigmaNu, GeneralizedGammaTheta};

    #[cfg(feature = "rand")]
    #[test]
    fn generalized_gamma_sampling_returns_positive_values_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = GeneralizedGammaScaleSigmaNu::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family.sample(
            &mut rng,
            &GeneralizedGammaTheta {
                mu: 1.5,
                sigma: 0.7,
                nu: 0.8,
            },
        );
        assert!(sample > 0.0 && sample.is_finite());
        let log_normal_limit = family.sample(
            &mut rng,
            &GeneralizedGammaTheta {
                mu: 1.5,
                sigma: 0.7,
                nu: 0.0,
            },
        );
        assert!(log_normal_limit > 0.0 && log_normal_limit.is_finite());
        assert!(
            family
                .sample(
                    &mut rng,
                    &GeneralizedGammaTheta {
                        mu: 1.5,
                        sigma: 0.0,
                        nu: 0.8,
                    }
                )
                .is_nan()
        );
    }
}
