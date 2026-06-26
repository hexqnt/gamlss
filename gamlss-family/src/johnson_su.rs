use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasQuantile, Identity, InitialEtaFromTheta, Link, Log, Mu, Nu, ObservationView,
    ParameterParts, ParameterizedFamily, PositiveLink, Sigma, Tau,
};

use gamlss_special::{unit_normal_cdf, unit_normal_log_pdf, unit_normal_quantile};

use crate::initial::{robust_location_scale, weighted_values};
use crate::numeric::finite_difference_gradient_eta;

/// Johnson SU distribution with identity/log/identity/log links.
///
/// Its NLL gradient currently uses a finite-difference fallback and should be
/// treated as a training slow path until an analytic gradient is added.
pub type JohnsonSuMuSigmaNuTau = JohnsonSu<Identity, Log, Identity, Log>;
/// Johnson SU family in a location-scale-skewness-tail parameterization.
///
/// Its NLL gradient currently uses a finite-difference fallback and should be
/// treated as a training slow path until an analytic gradient is added.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JohnsonSu<MuLink = Identity, SigmaLink = Log, NuLink = Identity, TauLink = Log> {
    marker: PhantomData<(MuLink, SigmaLink, NuLink, TauLink)>,
}

impl<MuLink, SigmaLink, NuLink, TauLink> JohnsonSu<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
    TauLink: PositiveLink<f64>,
{
    /// Creates a stateless Johnson SU family.
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn theta_from_eta(eta: JohnsonSuEta) -> JohnsonSuTheta {
        JohnsonSuTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
            nu: NuLink::inverse(eta.nu),
            tau: TauLink::inverse(eta.tau),
        }
    }

    #[inline(always)]
    fn nll_theta(y: f64, theta: JohnsonSuTheta) -> f64 {
        if !y.is_finite()
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || !theta.nu.is_finite()
            || theta.tau <= 0.0
            || !theta.tau.is_finite()
        {
            return f64::INFINITY;
        }

        let s = (y - theta.mu) / theta.sigma;
        let z = theta.tau.mul_add(s.asinh(), theta.nu);
        theta.sigma.ln() - theta.tau.ln() + 0.5 * (1.0 + s * s).ln() - unit_normal_log_pdf(z)
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: JohnsonSuEta) -> (f64, JohnsonSuEta) {
        let nll = Self::nll_theta(y, Self::theta_from_eta(eta));
        if !nll.is_finite() {
            return (nll, JohnsonSuEta::from_array([f64::NAN; 4]));
        }

        let gradient = finite_difference_gradient_eta::<_, JohnsonSuEta, 4>(eta, |probe| {
            Self::nll_theta(y, Self::theta_from_eta(probe))
        });
        (nll, JohnsonSuEta::from_array(gradient))
    }
}

impl<MuLink, SigmaLink, NuLink, TauLink> Default for JohnsonSu<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
    TauLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<MuLink, SigmaLink, NuLink, TauLink> Family for JohnsonSu<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
    TauLink: PositiveLink<f64>,
{
    type Eta = JohnsonSuEta;
    type Theta = JohnsonSuTheta;
    type NllGradientEta = JohnsonSuEta;
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

impl<MuLink, SigmaLink, NuLink, TauLink> ParameterizedFamily<4>
    for JohnsonSu<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: InitialEtaFromTheta<f64> + Link<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    NuLink: InitialEtaFromTheta<f64> + Link<f64>,
    TauLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Params = (Mu, Sigma, Nu, Tau);
    type Links = (MuLink, SigmaLink, NuLink, TauLink);

    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| y.is_finite().then_some(y));
        let Some((mu, sigma)) = robust_location_scale(&values) else {
            return JohnsonSuEta::from_array([0.0, 0.0, 0.0, 0.0]);
        };

        JohnsonSuEta {
            mu: MuLink::initial_eta_from_theta(mu),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
            nu: NuLink::initial_eta_from_theta(0.0),
            tau: TauLink::initial_eta_from_theta(1.0),
        }
    }
}

impl<MuLink, SigmaLink, NuLink, TauLink> HasCdf for JohnsonSu<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
    TauLink: PositiveLink<f64>,
{
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
        if !y.is_finite()
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || !theta.nu.is_finite()
            || theta.tau <= 0.0
            || !theta.tau.is_finite()
        {
            return f64::NAN;
        }

        unit_normal_cdf(theta.nu + theta.tau * ((y - theta.mu) / theta.sigma).asinh())
    }
}

impl<MuLink, SigmaLink, NuLink, TauLink> HasQuantile
    for JohnsonSu<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
    TauLink: PositiveLink<f64>,
{
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
        if theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || !theta.mu.is_finite()
            || !theta.nu.is_finite()
            || theta.tau <= 0.0
            || !theta.tau.is_finite()
        {
            return f64::NAN;
        }

        theta.mu + theta.sigma * ((unit_normal_quantile(p) - theta.nu) / theta.tau).sinh()
    }
}

/// Predictors for the Johnson SU family on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JohnsonSuEta {
    /// Location predictor.
    pub mu: f64,
    /// Scale predictor.
    pub sigma: f64,
    /// Skewness predictor.
    pub nu: f64,
    /// Tail predictor.
    pub tau: f64,
}

impl ParameterParts<4> for JohnsonSuEta {
    #[inline(always)]
    fn from_array(values: [f64; 4]) -> Self {
        Self {
            mu: values[0],
            sigma: values[1],
            nu: values[2],
            tau: values[3],
        }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mu,
            1 => self.sigma,
            2 => self.nu,
            3 => self.tau,
            _ => unreachable!("johnson-su eta only has indices 0 through 3"),
        }
    }
}

/// Natural-scale Johnson SU parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JohnsonSuTheta {
    /// Location parameter.
    pub mu: f64,
    /// Positive scale parameter.
    pub sigma: f64,
    /// Skewness parameter.
    pub nu: f64,
    /// Positive tail parameter.
    pub tau: f64,
}
