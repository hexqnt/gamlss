use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasQuantile, Identity, InitialEtaFromTheta, Link, Log, Mu, Nu, ObservationView,
    ParameterParts, ParameterizedFamily, PositiveLink, Sigma, Tau,
};

use gamlss_special::{unit_normal_cdf, unit_normal_log_pdf, unit_normal_quantile};

use crate::initial::{robust_location_scale, weighted_values};
use crate::numeric::finite_difference_gradient_eta;

/// SHASH distribution with identity/log/log/log links.
///
/// Its NLL gradient currently uses a finite-difference fallback and should be
/// treated as a training slow path until an analytic gradient is added.
pub type ShashMuSigmaNuTau = Shash<Identity, Log, Log, Log>;
/// Sinh-arcsinh family using positive skewness and tail parameters.
///
/// `nu = 1` and `tau = 1` reduce the standardized distribution to the
/// standard normal. Values of `nu` above or below one skew the distribution
/// through `ln(nu)`, which keeps the default log link centered at the symmetric
/// case.
///
/// Its NLL gradient currently uses a finite-difference fallback and should be
/// treated as a training slow path until an analytic gradient is added.
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
        let x = (y - theta.mu) / theta.sigma;
        let h = theta.tau * x.asinh() - theta.nu.ln();
        (x, h.sinh())
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

        let (x, z) = Self::transformed_z(y, theta);
        let h = theta.tau * x.asinh() - theta.nu.ln();
        theta.sigma.ln() - theta.tau.ln() + 0.5 * (x * x).ln_1p()
            - h.cosh().ln()
            - unit_normal_log_pdf(z)
    }

    #[inline]
    fn nll_and_gradient_eta_values(y: f64, eta: ShashEta) -> (f64, ShashEta) {
        let nll = Self::nll_theta(y, Self::theta_from_eta(eta));
        if !nll.is_finite() {
            return (nll, ShashEta::from_array([f64::NAN; 4]));
        }

        let gradient = finite_difference_gradient_eta::<_, ShashEta, 4>(eta, |probe| {
            Self::nll_theta(y, Self::theta_from_eta(probe))
        });
        (nll, ShashEta::from_array(gradient))
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

impl<MuLink, SigmaLink, NuLink, TauLink> Family for Shash<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    type Eta = ShashEta;
    type Theta = ShashTheta;
    type NllGradientEta = ShashEta;
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

impl<MuLink, SigmaLink, NuLink, TauLink> ParameterizedFamily<4>
    for Shash<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: InitialEtaFromTheta<f64> + Link<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    NuLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
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
    fn cdf(&self, y: Self::Observation<'_>, theta: Self::Theta) -> f64 {
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

        unit_normal_cdf(Self::transformed_z(y, theta).1)
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
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
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
