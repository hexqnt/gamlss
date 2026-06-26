use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasQuantile, Identity, InitialEtaFromTheta, Link, Log, Mu, Nu, ObservationView,
    ParameterParts, ParameterizedFamily, PositiveLink, Sigma,
};

use gamlss_special::{invert_real_cdf, ln_gamma, regularized_gamma_lower};

use crate::initial::{robust_location_scale, weighted_values};
use crate::numeric::finite_difference_gradient_eta;

const LOG_2: f64 = std::f64::consts::LN_2;

/// Power exponential distribution with identity/log/log links.
///
/// Its NLL gradient currently uses a finite-difference fallback and should be
/// treated as a training slow path until an analytic gradient is added.
pub type PowerExponentialMuSigmaNu = PowerExponential<Identity, Log, Log>;
/// Alias commonly used for the generalized error distribution.
pub type Ged<MuLink = Identity, SigmaLink = Log, NuLink = Log> =
    PowerExponential<MuLink, SigmaLink, NuLink>;
/// Default generalized error distribution alias.
pub type GedMuSigmaNu = PowerExponentialMuSigmaNu;
/// Power exponential / generalized error distribution.
///
/// Its NLL gradient currently uses a finite-difference fallback and should be
/// treated as a training slow path until an analytic gradient is added.
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
        (0.5 * (ln_gamma(1.0 / nu) - ln_gamma(3.0 / nu))).exp()
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

        let c = Self::scale_c(theta.nu);
        let z = ((y - theta.mu) / (c * theta.sigma)).abs();
        LOG_2 + c.ln() + theta.sigma.ln() + ln_gamma(1.0 / theta.nu) - theta.nu.ln()
            + z.powf(theta.nu)
    }

    #[inline]
    fn nll_and_gradient_eta_values(y: f64, eta: PowerExponentialEta) -> (f64, PowerExponentialEta) {
        let nll = Self::nll_theta(y, Self::theta_from_eta(eta));
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

        let gradient = finite_difference_gradient_eta::<_, PowerExponentialEta, 3>(eta, |probe| {
            Self::nll_theta(y, Self::theta_from_eta(probe))
        });
        (nll, PowerExponentialEta::from_array(gradient))
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

impl<MuLink, SigmaLink, NuLink> Family for PowerExponential<MuLink, SigmaLink, NuLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: PositiveLink<f64>,
{
    type Eta = PowerExponentialEta;
    type Theta = PowerExponentialTheta;
    type NllGradientEta = PowerExponentialEta;
    type Observation<'obs> = f64;

    #[inline]
    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline]
    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::nll_theta(y, theta)
    }

    #[inline]
    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        Self::nll_theta(y, Self::theta_from_eta(eta))
    }

    #[inline]
    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        Self::nll_and_gradient_eta_values(y, eta)
    }
}

impl<MuLink, SigmaLink, NuLink> ParameterizedFamily<3>
    for PowerExponential<MuLink, SigmaLink, NuLink>
where
    MuLink: InitialEtaFromTheta<f64> + Link<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    NuLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Params = (Mu, Sigma, Nu);
    type Links = (MuLink, SigmaLink, NuLink);

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
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
        if !y.is_finite()
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || theta.nu <= 0.0
            || !theta.nu.is_finite()
        {
            return f64::NAN;
        }

        let c = Self::scale_c(theta.nu);
        let z = (y - theta.mu) / (c * theta.sigma);
        let p = regularized_gamma_lower(1.0 / theta.nu, z.abs().powf(theta.nu));
        if z < 0.0 {
            0.5 * (1.0 - p)
        } else {
            f64::midpoint(1.0, p)
        }
    }
}

impl<MuLink, SigmaLink, NuLink> HasQuantile for PowerExponential<MuLink, SigmaLink, NuLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: PositiveLink<f64>,
{
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
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
