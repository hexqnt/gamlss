use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasQuantile, Identity, InitialEtaFromTheta, Link, Log, Mu, Nu, ObservationView,
    ParameterParts, ParameterizedFamily, PositiveLink, Sigma,
};

use crate::initial::{positive_floor, weighted_summary, weighted_values};
use crate::numeric::finite_difference_gradient_eta;
use crate::special::{invert_positive_cdf, ln_gamma, regularized_gamma_lower, unit_normal_cdf};

const NU_EPSILON: f64 = 1.0e-6;
const HALF_LOG_2_PI: f64 = 0.918_938_533_204_672_7;

/// Generalized gamma distribution with log/log/identity links.
pub type GeneralizedGammaMuSigmaNu = GeneralizedGamma<Log, Log, Identity>;
/// Generalized gamma family with GAMLSS-like mean/scale/shape parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneralizedGamma<MuLink = Log, SigmaLink = Log, NuLink = Identity> {
    marker: PhantomData<(MuLink, SigmaLink, NuLink)>,
}

impl<MuLink, SigmaLink, NuLink> GeneralizedGamma<MuLink, SigmaLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    /// Creates a stateless generalized gamma family.
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn theta_from_eta(eta: GeneralizedGammaEta) -> GeneralizedGammaTheta {
        GeneralizedGammaTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
            nu: NuLink::inverse(eta.nu),
        }
    }

    #[inline(always)]
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
            let z = (y / theta.mu).ln() / theta.sigma;
            return y.ln() + theta.sigma.ln() + HALF_LOG_2_PI + 0.5 * z * z;
        }

        let abs_nu = theta.nu.abs();
        let k = 1.0 / (theta.sigma * theta.sigma * abs_nu * abs_nu);
        let z = (y / theta.mu).powf(theta.nu);
        -(k * k.ln() + k * z.ln() + abs_nu.ln() - k * z - ln_gamma(k) - y.ln())
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: GeneralizedGammaEta) -> (f64, GeneralizedGammaEta) {
        let nll = Self::nll_theta(y, Self::theta_from_eta(eta));
        if !nll.is_finite() {
            return (nll, GeneralizedGammaEta::from_array([f64::NAN; 3]));
        }

        let gradient = finite_difference_gradient_eta::<_, GeneralizedGammaEta, 3>(eta, |probe| {
            Self::nll_theta(y, Self::theta_from_eta(probe))
        });
        (nll, GeneralizedGammaEta::from_array(gradient))
    }
}

impl<MuLink, SigmaLink, NuLink> Default for GeneralizedGamma<MuLink, SigmaLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<MuLink, SigmaLink, NuLink> Family for GeneralizedGamma<MuLink, SigmaLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    type Eta = GeneralizedGammaEta;
    type Theta = GeneralizedGammaTheta;
    type NllGradientEta = GeneralizedGammaEta;
    type Observation<'obs> = f64;

    #[inline(always)]
    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline(always)]
    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::nll_theta(y, theta)
    }

    #[inline(always)]
    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        Self::nll_theta(y, Self::theta_from_eta(eta))
    }

    #[inline(always)]
    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        Self::nll_and_gradient_eta_values(y, eta)
    }
}

impl<MuLink, SigmaLink, NuLink> ParameterizedFamily<3>
    for GeneralizedGamma<MuLink, SigmaLink, NuLink>
where
    MuLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    NuLink: InitialEtaFromTheta<f64> + Link<f64>,
{
    type Params = (Mu, Sigma, Nu);
    type Links = (MuLink, SigmaLink, NuLink);

    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values =
            weighted_values::<Self, _, _>(obs, |y| (y.is_finite() && y > 0.0).then_some(y));
        let Some(summary) = weighted_summary(&values) else {
            return GeneralizedGammaEta::from_array([0.0, 0.0, 0.0]);
        };
        let mu = positive_floor(summary.mean);
        let sigma = positive_floor((summary.variance.sqrt() / mu).max(1.0e-3));

        GeneralizedGammaEta {
            mu: MuLink::initial_eta_from_theta(mu),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
            nu: NuLink::initial_eta_from_theta(1.0),
        }
    }
}

impl<MuLink, SigmaLink, NuLink> HasCdf for GeneralizedGamma<MuLink, SigmaLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
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

impl<MuLink, SigmaLink, NuLink> HasQuantile for GeneralizedGamma<MuLink, SigmaLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
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

/// Predictors for generalized gamma on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneralizedGammaEta {
    /// Mean predictor.
    pub mu: f64,
    /// Scale predictor.
    pub sigma: f64,
    /// Shape predictor.
    pub nu: f64,
}

impl ParameterParts<3> for GeneralizedGammaEta {
    #[inline(always)]
    fn from_array(values: [f64; 3]) -> Self {
        Self {
            mu: values[0],
            sigma: values[1],
            nu: values[2],
        }
    }

    #[inline(always)]
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
    /// Positive mean parameter.
    pub mu: f64,
    /// Positive scale parameter.
    pub sigma: f64,
    /// Shape parameter; `nu = 0` is the log-normal limit.
    pub nu: f64,
}
