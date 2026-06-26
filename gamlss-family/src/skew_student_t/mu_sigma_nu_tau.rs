use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasQuantile, Identity, InitialEtaFromTheta, Link, Log, Mu, Nu, ObservationView,
    ParameterParts, ParameterizedFamily, PositiveLink, Sigma, Tau,
};

use crate::initial::{robust_location_scale, weighted_values};
use crate::numeric::finite_difference_gradient_eta;

use super::{cdf_location_scale, nll_location_scale, quantile_location_scale};

/// Skew Student-t distribution with identity/log/identity/log links.
///
/// Its NLL gradient currently uses a finite-difference fallback and should be
/// treated as a training slow path until an analytic gradient is added.
pub type SkewStudentTMuSigmaNuTau = SkewStudentT<Identity, Log, Identity, Log>;

/// Azzalini/ST1-style skew Student-t family.
///
/// Its NLL gradient currently uses a finite-difference fallback and should be
/// treated as a training slow path until an analytic gradient is added.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkewStudentT<MuLink = Identity, SigmaLink = Log, NuLink = Identity, TauLink = Log> {
    marker: PhantomData<(MuLink, SigmaLink, NuLink, TauLink)>,
}

impl<MuLink, SigmaLink, NuLink, TauLink> SkewStudentT<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
    TauLink: PositiveLink<f64>,
{
    /// Creates a stateless skew Student-t family.
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn theta_from_eta(eta: SkewStudentTEta) -> SkewStudentTTheta {
        SkewStudentTTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
            nu: NuLink::inverse(eta.nu),
            tau: TauLink::inverse(eta.tau),
        }
    }

    #[inline(always)]
    fn nll_theta(y: f64, theta: SkewStudentTTheta) -> f64 {
        nll_location_scale(y, theta.mu, theta.sigma, theta.nu, theta.tau)
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: SkewStudentTEta) -> (f64, SkewStudentTEta) {
        let nll = Self::nll_theta(y, Self::theta_from_eta(eta));
        if !nll.is_finite() {
            return (nll, SkewStudentTEta::from_array([f64::NAN; 4]));
        }

        let gradient = finite_difference_gradient_eta::<_, SkewStudentTEta, 4>(eta, |probe| {
            Self::nll_theta(y, Self::theta_from_eta(probe))
        });
        (nll, SkewStudentTEta::from_array(gradient))
    }
}

impl<MuLink, SigmaLink, NuLink, TauLink> Default
    for SkewStudentT<MuLink, SigmaLink, NuLink, TauLink>
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

impl<MuLink, SigmaLink, NuLink, TauLink> Family for SkewStudentT<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
    TauLink: PositiveLink<f64>,
{
    type Eta = SkewStudentTEta;
    type Theta = SkewStudentTTheta;
    type NllGradientEta = SkewStudentTEta;
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
    for SkewStudentT<MuLink, SigmaLink, NuLink, TauLink>
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
            return SkewStudentTEta::from_array([0.0, 0.0, 0.0, 5.0_f64.ln()]);
        };

        SkewStudentTEta {
            mu: MuLink::initial_eta_from_theta(mu),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
            nu: NuLink::initial_eta_from_theta(0.0),
            tau: TauLink::initial_eta_from_theta(5.0),
        }
    }
}

impl<MuLink, SigmaLink, NuLink, TauLink> HasCdf for SkewStudentT<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
    TauLink: PositiveLink<f64>,
{
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
        cdf_location_scale(y, theta.mu, theta.sigma, theta.nu, theta.tau)
    }
}

impl<MuLink, SigmaLink, NuLink, TauLink> HasQuantile
    for SkewStudentT<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
    TauLink: PositiveLink<f64>,
{
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
        quantile_location_scale(p, theta.mu, theta.sigma, theta.nu, theta.tau)
    }
}

/// Predictors for skew Student-t on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkewStudentTEta {
    /// Location predictor.
    pub mu: f64,
    /// Scale predictor.
    pub sigma: f64,
    /// Skewness predictor.
    pub nu: f64,
    /// Degrees-of-freedom predictor.
    pub tau: f64,
}

impl ParameterParts<4> for SkewStudentTEta {
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
            _ => unreachable!("skew student-t eta only has indices 0 through 3"),
        }
    }
}

/// Natural-scale skew Student-t parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkewStudentTTheta {
    /// Location parameter.
    pub mu: f64,
    /// Positive scale parameter.
    pub sigma: f64,
    /// Skewness parameter.
    pub nu: f64,
    /// Positive degrees of freedom.
    pub tau: f64,
}
