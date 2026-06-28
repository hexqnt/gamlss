use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasQuantile, Identity, InitialEtaFromTheta, Link, Log, Mu, Nu, ObservationView,
    ParameterParts, ParameterizedFamily, PositiveLink, Sigma,
};

use crate::initial::{robust_location_scale, weighted_values};

const XI_EPSILON: f64 = 1.0e-8;

/// GEV distribution with identity/log/identity links.
pub type GevMuSigmaShape = Gev<Identity, Log, Identity>;
/// Generalized extreme value family for block maxima.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(eta: GevEta) -> GevTheta {
        GevTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
            nu: NuLink::inverse(eta.nu),
        }
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
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
            let exp_neg_z = (-z).exp();
            let d_nu = Self::gumbel_limit_nu_score(z, exp_neg_z);
            return theta.sigma.ln() + z + exp_neg_z + theta.nu * d_nu;
        }

        let t = theta.nu.mul_add(z, 1.0);
        if t <= 0.0 || !t.is_finite() {
            return f64::INFINITY;
        }
        let inv = t.powf(-1.0 / theta.nu);
        theta.sigma.ln() + (1.0 / theta.nu + 1.0) * t.ln() + inv
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn gumbel_limit_nu_score(z: f64, exp_neg_z: f64) -> f64 {
        z + 0.5 * z * z * (exp_neg_z - 1.0)
    }

    #[inline]
    fn gradient_theta(y: f64, theta: GevTheta) -> GevTheta {
        let z = (y - theta.mu) / theta.sigma;
        if theta.nu.abs() < XI_EPSILON {
            let exp_neg_z = (-z).exp();
            let d_z = 1.0 - exp_neg_z;
            return GevTheta {
                mu: -d_z / theta.sigma,
                sigma: z.mul_add(-d_z, 1.0) / theta.sigma,
                nu: Self::gumbel_limit_nu_score(z, exp_neg_z),
            };
        }

        let t = theta.nu.mul_add(z, 1.0);
        let log_t = t.ln();
        let inv = t.powf(-1.0 / theta.nu);
        let d_z = (1.0 + theta.nu - inv) / t;
        let d_nu = (inv - 1.0) * log_t / (theta.nu * theta.nu)
            + z * (1.0 + theta.nu - inv) / (theta.nu * t);

        GevTheta {
            mu: -d_z / theta.sigma,
            sigma: z.mul_add(-d_z, 1.0) / theta.sigma,
            nu: d_nu,
        }
    }

    #[inline]
    fn nll_and_gradient_eta_values(y: f64, eta: GevEta) -> (f64, GevEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (nll, GevEta::from_array([f64::NAN; 3]));
        }

        let gradient = Self::gradient_theta(y, theta);
        (
            nll,
            GevEta {
                mu: gradient.mu * MuLink::derivative_inverse(eta.mu),
                sigma: gradient.sigma * SigmaLink::derivative_inverse(eta.sigma),
                nu: gradient.nu * NuLink::derivative_inverse(eta.nu),
            },
        )
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
    fn cdf(&self, y: Self::Observation<'_>, theta: Self::Theta) -> f64 {
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
        let t = theta.nu.mul_add(z, 1.0);
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
    #[allow(clippy::float_cmp)]
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
            theta.sigma.mul_add(-log_p.ln(), theta.mu)
        } else {
            theta.mu + theta.sigma * (-theta.nu * log_p.ln()).exp_m1() / theta.nu
        }
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
