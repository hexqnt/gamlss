use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasQuantile, Identity, InitialEtaFromTheta, Link, Log, Mu, Nu, ObservationView,
    ParameterParts, ParameterizedFamily, PositiveLink, Sigma,
};

use crate::initial::{robust_location_scale, weighted_values};
use crate::numeric::finite_difference_gradient_eta;

const XI_EPSILON: f64 = 1.0e-8;

/// Generalized extreme value family for block maxima.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gev<MuLink = Identity, SigmaLink = Log, NuLink = Identity> {
    marker: PhantomData<(MuLink, SigmaLink, NuLink)>,
}

impl<MuLink, SigmaLink, NuLink> Gev<MuLink, SigmaLink, NuLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    /// Creates a stateless GEV family.
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn theta_from_eta(eta: GevEta) -> GevTheta {
        GevTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
            nu: NuLink::inverse(eta.nu),
        }
    }

    #[inline(always)]
    fn nll_theta(y: f64, theta: GevTheta) -> f64 {
        if !y.is_finite()
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || !theta.nu.is_finite()
        {
            return f64::INFINITY;
        }

        let z = (y - theta.mu) / theta.sigma;
        if theta.nu.abs() < XI_EPSILON {
            return theta.sigma.ln() + z + (-z).exp();
        }

        let t = 1.0 + theta.nu * z;
        if t <= 0.0 || !t.is_finite() {
            return f64::INFINITY;
        }
        let inv = t.powf(-1.0 / theta.nu);
        theta.sigma.ln() + (1.0 / theta.nu + 1.0) * t.ln() + inv
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: GevEta) -> (f64, GevEta) {
        let nll = Self::nll_theta(y, Self::theta_from_eta(eta));
        if !nll.is_finite() {
            return (nll, GevEta::from_array([f64::NAN; 3]));
        }

        let gradient = finite_difference_gradient_eta::<_, GevEta, 3>(eta, |probe| {
            Self::nll_theta(y, Self::theta_from_eta(probe))
        });
        (nll, GevEta::from_array(gradient))
    }
}

impl<MuLink, SigmaLink, NuLink> Default for Gev<MuLink, SigmaLink, NuLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Predictors for GEV on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GevEta {
    /// Location predictor.
    pub mu: f64,
    /// Scale predictor.
    pub sigma: f64,
    /// Shape predictor.
    pub nu: f64,
}

impl ParameterParts<3> for GevEta {
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
            _ => unreachable!("gev eta only has indices 0 through 2"),
        }
    }
}

/// Natural-scale GEV parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GevTheta {
    /// Location parameter.
    pub mu: f64,
    /// Positive scale parameter.
    pub sigma: f64,
    /// Shape parameter.
    pub nu: f64,
}

impl<MuLink, SigmaLink, NuLink> Family for Gev<MuLink, SigmaLink, NuLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    type Eta = GevEta;
    type Theta = GevTheta;
    type NllGradientEta = GevEta;
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

impl<MuLink, SigmaLink, NuLink> ParameterizedFamily<3> for Gev<MuLink, SigmaLink, NuLink>
where
    MuLink: InitialEtaFromTheta<f64> + Link<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    NuLink: InitialEtaFromTheta<f64> + Link<f64>,
{
    type Params = (Mu, Sigma, Nu);
    type Links = (MuLink, SigmaLink, NuLink);

    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| y.is_finite().then_some(y));
        let Some((mu, sigma)) = robust_location_scale(&values) else {
            return GevEta::from_array([0.0, 0.0, 0.0]);
        };

        GevEta {
            mu: MuLink::initial_eta_from_theta(mu),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
            nu: NuLink::initial_eta_from_theta(0.0),
        }
    }
}

impl<MuLink, SigmaLink, NuLink> HasCdf for Gev<MuLink, SigmaLink, NuLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
        if !y.is_finite()
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || !theta.nu.is_finite()
        {
            return f64::NAN;
        }

        let z = (y - theta.mu) / theta.sigma;
        if theta.nu.abs() < XI_EPSILON {
            return (-(-z).exp()).exp();
        }
        let t = 1.0 + theta.nu * z;
        if t <= 0.0 {
            return if theta.nu > 0.0 { 0.0 } else { 1.0 };
        }
        (-t.powf(-1.0 / theta.nu)).exp()
    }
}

impl<MuLink, SigmaLink, NuLink> HasQuantile for Gev<MuLink, SigmaLink, NuLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
        if !(0.0..=1.0).contains(&p)
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || !theta.nu.is_finite()
        {
            return f64::NAN;
        }
        if p == 0.0 {
            return if theta.nu > 0.0 {
                theta.mu - theta.sigma / theta.nu
            } else {
                f64::NEG_INFINITY
            };
        }
        if p == 1.0 {
            return if theta.nu < 0.0 {
                theta.mu - theta.sigma / theta.nu
            } else {
                f64::INFINITY
            };
        }

        let log_p = -p.ln();
        if theta.nu.abs() < XI_EPSILON {
            theta.mu - theta.sigma * log_p.ln()
        } else {
            theta.mu + theta.sigma * (log_p.powf(-theta.nu) - 1.0) / theta.nu
        }
    }
}

/// GEV distribution with identity/log/identity links.
pub type DefaultGev = Gev<Identity, Log, Identity>;
