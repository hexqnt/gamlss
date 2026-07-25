use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasCrps, HasQuantile, Identity, InitialEtaFromObservations,
    InitialEtaFromTheta, Link, Log, Mu, Nu, ObservationView, ParameterParts, PositiveLink, Sigma,
    Tau,
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

use gamlss_special::{student_t_cdf_standardized, student_t_log_pdf_standardized};

use crate::initial::{robust_location_scale, weighted_values};
use crate::link::positive_inverse_and_log;
use crate::numeric::finite_difference_gradient_eta;

#[cfg(feature = "rand")]
use super::try_sample_location_scale;
use super::{
    cdf_location_scale, crps_location_scale, nll_location_scale, nll_location_scale_with_log_sigma,
    quantile_location_scale, skew_argument,
};

/// Skew Student-t distribution with identity/log/identity/log links.
///
/// Its NLL gradient is analytic for location, scale and skewness. The
/// degrees-of-freedom component currently uses a finite-difference fallback.
pub type SkewStudentTMuSigmaNuTau = SkewStudentT<Identity, Log, Identity, Log>;

/// Azzalini/ST1-style skew Student-t family.
///
/// Its NLL gradient is analytic for location, scale and skewness. The
/// degrees-of-freedom component currently uses a finite-difference fallback.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../../doc-assets/distributions/skew_student_t.svg")
)]
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
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(eta: SkewStudentTEta) -> SkewStudentTTheta {
        SkewStudentTTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
            nu: NuLink::inverse(eta.nu),
            tau: TauLink::inverse(eta.tau),
        }
    }

    #[inline]
    fn theta_and_log_sigma_from_eta(eta: SkewStudentTEta) -> (SkewStudentTTheta, f64) {
        let (sigma, log_sigma) = positive_inverse_and_log::<SigmaLink>(eta.sigma);
        (
            SkewStudentTTheta {
                mu: MuLink::inverse(eta.mu),
                sigma,
                nu: NuLink::inverse(eta.nu),
                tau: TauLink::inverse(eta.tau),
            },
            log_sigma,
        )
    }

    #[inline]
    fn nll_theta(y: f64, theta: SkewStudentTTheta) -> f64 {
        nll_location_scale(y, theta.mu, theta.sigma, theta.nu, theta.tau)
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn gradient_theta_mu_sigma_nu(y: f64, theta: SkewStudentTTheta) -> (f64, f64, f64) {
        let z = (y - theta.mu) / theta.sigma;
        let skew_arg = skew_argument(z, theta.nu, theta.tau);
        let skew = student_t_cdf_standardized(skew_arg, theta.tau + 1.0);
        let skew_density = student_t_log_pdf_standardized(skew_arg, theta.tau + 1.0).exp();
        let skew_score = skew_density / skew;
        let sqrt_ratio = ((theta.tau + 1.0) / (theta.tau + z * z)).sqrt();
        let d_skew_arg_d_z = theta.nu * sqrt_ratio * theta.tau / (theta.tau + z * z);
        let d_nll_d_z = (theta.tau + 1.0) * z / (theta.tau + z * z) - skew_score * d_skew_arg_d_z;

        (
            -d_nll_d_z / theta.sigma,
            z.mul_add(-d_nll_d_z, 1.0) / theta.sigma,
            -skew_score * z * sqrt_ratio,
        )
    }

    #[inline]
    fn tau_gradient_eta(y: f64, eta_tau: f64, theta: SkewStudentTTheta, log_sigma: f64) -> f64 {
        let [tau] = finite_difference_gradient_eta::<_, f64, 1>(eta_tau, |eta_tau| {
            let theta = SkewStudentTTheta {
                tau: TauLink::inverse(eta_tau),
                ..theta
            };
            nll_location_scale_with_log_sigma(
                y,
                theta.mu,
                theta.sigma,
                theta.nu,
                theta.tau,
                log_sigma,
            )
        });
        tau
    }

    #[inline]
    fn nll_and_gradient_eta_values(y: f64, eta: SkewStudentTEta) -> (f64, SkewStudentTEta) {
        let (theta, log_sigma) = Self::theta_and_log_sigma_from_eta(eta);
        let nll = nll_location_scale_with_log_sigma(
            y,
            theta.mu,
            theta.sigma,
            theta.nu,
            theta.tau,
            log_sigma,
        );
        if !nll.is_finite() {
            return (nll, SkewStudentTEta::from_array([f64::NAN; 4]));
        }

        let (d_location, d_scale, d_skewness) = Self::gradient_theta_mu_sigma_nu(y, theta);

        (
            nll,
            SkewStudentTEta {
                mu: d_location * MuLink::derivative_inverse(eta.mu),
                sigma: d_scale * theta.sigma * SigmaLink::derivative_log_inverse(eta.sigma),
                nu: d_skewness * NuLink::derivative_inverse(eta.nu),
                tau: Self::tau_gradient_eta(y, eta.tau, theta, log_sigma),
            },
        )
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

gamlss_core::impl_scalar_compilable_family!(
    impl<MuLink, SigmaLink, NuLink, TauLink> for SkewStudentT<MuLink, SigmaLink, NuLink, TauLink>;
    parameters = (Mu, Sigma, Nu, Tau);
    arity = 4;
);

impl<MuLink, SigmaLink, NuLink, TauLink> Family for SkewStudentT<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
    TauLink: PositiveLink<f64>,
{
    type Eta = SkewStudentTEta;
    type Theta = SkewStudentTTheta;
    type GradientEta = SkewStudentTEta;
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
        let (theta, log_sigma) = Self::theta_and_log_sigma_from_eta(*eta);
        nll_location_scale_with_log_sigma(y, theta.mu, theta.sigma, theta.nu, theta.tau, log_sigma)
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

impl<MuLink, SigmaLink, NuLink, TauLink> InitialEtaFromObservations<4>
    for SkewStudentT<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: InitialEtaFromTheta<f64> + Link<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    NuLink: InitialEtaFromTheta<f64> + Link<f64>,
    TauLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
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
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
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
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        quantile_location_scale(p, theta.mu, theta.sigma, theta.nu, theta.tau)
    }
}

impl<MuLink, SigmaLink, NuLink, TauLink> HasCrps
    for SkewStudentT<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
    TauLink: PositiveLink<f64>,
{
    fn crps(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        crps_location_scale(y, theta.mu, theta.sigma, theta.nu, theta.tau)
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink, SigmaLink, NuLink, TauLink> TrySimulate<Rng>
    for SkewStudentT<MuLink, SigmaLink, NuLink, TauLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
    TauLink: PositiveLink<f64>,
{
    type Sample = f64;

    fn try_sample(
        &self,
        rng: &mut Rng,
        theta: &Self::Theta,
    ) -> Result<Self::Sample, SimulationError> {
        try_sample_location_scale(rng, theta.mu, theta.sigma, theta.nu, theta.tau)
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
