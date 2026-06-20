use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasQuantile, Identity, InitialEtaFromTheta, Link, Log, Mu, Nu, ObservationView,
    ParameterParts, ParameterizedFamily, PositiveLink, Sigma,
};

use crate::initial::{robust_location_scale, weighted_values};
use crate::numeric::finite_difference_gradient_eta;
use crate::special::{invert_real_cdf, owens_t, unit_normal_cdf, unit_normal_log_pdf};

const LOG_2: f64 = std::f64::consts::LN_2;

/// Skew-normal distribution with identity/log/identity links.
pub type SkewNormalMuSigmaNu = SkewNormal<Identity, Log, Identity>;
/// Azzalini/SN1-style skew-normal family with location, scale and skewness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkewNormal<MuLink = Identity, SigmaLink = Log, NuLink = Identity> {
    marker: PhantomData<(MuLink, SigmaLink, NuLink)>,
}

impl<MuLink, SigmaLink, NuLink> SkewNormal<MuLink, SigmaLink, NuLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    /// Creates a stateless skew-normal family.
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn theta_from_eta(eta: SkewNormalEta) -> SkewNormalTheta {
        SkewNormalTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
            nu: NuLink::inverse(eta.nu),
        }
    }

    #[inline(always)]
    fn nll_theta(y: f64, theta: SkewNormalTheta) -> f64 {
        if !y.is_finite()
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || !theta.nu.is_finite()
        {
            return f64::INFINITY;
        }

        let z = (y - theta.mu) / theta.sigma;
        let skew_cdf = unit_normal_cdf(theta.nu * z);
        if skew_cdf <= 0.0 {
            return f64::INFINITY;
        }

        theta.sigma.ln() - LOG_2 - unit_normal_log_pdf(z) - skew_cdf.ln()
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: SkewNormalEta) -> (f64, SkewNormalEta) {
        let nll = Self::nll_theta(y, Self::theta_from_eta(eta));
        if !nll.is_finite() {
            return (
                nll,
                SkewNormalEta {
                    mu: f64::NAN,
                    sigma: f64::NAN,
                    nu: f64::NAN,
                },
            );
        }

        let gradient = finite_difference_gradient_eta::<_, SkewNormalEta, 3>(eta, |probe| {
            Self::nll_theta(y, Self::theta_from_eta(probe))
        });
        (nll, SkewNormalEta::from_array(gradient))
    }

    #[inline(always)]
    fn standard_cdf(z: f64, nu: f64) -> f64 {
        (unit_normal_cdf(z) - 2.0 * owens_t(z, nu)).clamp(0.0, 1.0)
    }
}

impl<MuLink, SigmaLink, NuLink> Default for SkewNormal<MuLink, SigmaLink, NuLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<MuLink, SigmaLink, NuLink> Family for SkewNormal<MuLink, SigmaLink, NuLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    type Eta = SkewNormalEta;
    type Theta = SkewNormalTheta;
    type NllGradientEta = SkewNormalEta;
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

impl<MuLink, SigmaLink, NuLink> ParameterizedFamily<3> for SkewNormal<MuLink, SigmaLink, NuLink>
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
            return SkewNormalEta::from_array([0.0, 0.0, 0.0]);
        };

        SkewNormalEta {
            mu: MuLink::initial_eta_from_theta(mu),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
            nu: NuLink::initial_eta_from_theta(0.0),
        }
    }
}

impl<MuLink, SigmaLink, NuLink> HasCdf for SkewNormal<MuLink, SigmaLink, NuLink>
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
        Self::standard_cdf((y - theta.mu) / theta.sigma, theta.nu)
    }
}

impl<MuLink, SigmaLink, NuLink> HasQuantile for SkewNormal<MuLink, SigmaLink, NuLink>
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
            return f64::NEG_INFINITY;
        }
        if p == 1.0 {
            return f64::INFINITY;
        }

        invert_real_cdf(p, |z| Self::standard_cdf(z, theta.nu)) * theta.sigma + theta.mu
    }
}

/// Predictors for the skew-normal family on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkewNormalEta {
    /// Location predictor.
    pub mu: f64,
    /// Scale predictor.
    pub sigma: f64,
    /// Skewness predictor.
    pub nu: f64,
}

impl ParameterParts<3> for SkewNormalEta {
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
            _ => unreachable!("skew-normal eta only has indices 0 through 2"),
        }
    }
}

/// Natural-scale skew-normal parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkewNormalTheta {
    /// Location parameter.
    pub mu: f64,
    /// Positive scale parameter.
    pub sigma: f64,
    /// Skewness parameter.
    pub nu: f64,
}
