use std::marker::PhantomData;

use gamlss_core::{
    AboveTwoLink, Family, HasCdf, HasCrps, HasQuantile, InitialEtaFromObservations,
    InitialEtaFromTheta, Link, Mu, ObservationView, ParameterParts, PositiveLink, Sigma, Tau,
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

use crate::initial::{robust_location_scale, weighted_values};

use super::{
    StudentTKernel, StudentTTheta, student_t_crps_theta, student_t_standard_cdf,
    student_t_standard_quantile,
};

/// Dynamic-DF Student's t family where the natural `sigma` field is the standard deviation.
///
/// In the equations, $s$ denotes [`StudentTMuSdTauTheta::sigma`] and $\sigma_{\mathrm{scale}}$ denotes the internal [`StudentTTheta::sigma`]. The conversion is
///
/// $$
/// \sigma_{\mathrm{scale}}
/// =s\sqrt{\frac{\tau-2}{\tau}}, \qquad
/// \operatorname{Var}(Y)=s^2,\qquad \tau>2.
/// $$
///
/// The matching eta field is also named `sigma`, so the default links are $s=\exp(\eta_\sigma)$ and $\tau=2+\exp(\eta_\tau)$.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../../doc-assets/distributions/student_t_stddev.svg")
)]
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StudentTStdDev<
    MuLink = gamlss_core::Identity,
    SigmaLink = gamlss_core::Log,
    TauLink = gamlss_core::LogPlus<2>,
> {
    marker: PhantomData<(MuLink, SigmaLink, TauLink)>,
}

impl<MuLink, SigmaLink, TauLink> StudentTStdDev<MuLink, SigmaLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    TauLink: AboveTwoLink<f64>,
{
    /// Creates a stateless standard-deviation Student's t family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(eta: StudentTMuSdTauEta) -> StudentTMuSdTauTheta {
        StudentTMuSdTauTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
            tau: TauLink::inverse(eta.tau),
        }
    }

    #[inline]
    fn nll_theta(y: f64, theta: StudentTMuSdTauTheta) -> f64 {
        let Some(location_scale) = theta.location_scale() else {
            return f64::INFINITY;
        };
        StudentTKernel::try_new(theta.tau)
            .map_or(f64::INFINITY, |kernel| kernel.nll_theta(y, location_scale))
    }

    #[inline]
    fn nll_and_gradient_eta_values(y: f64, eta: StudentTMuSdTauEta) -> (f64, StudentTMuSdTauEta) {
        let theta = Self::theta_from_eta(eta);
        let Some(location_scale) = theta.location_scale() else {
            return (f64::INFINITY, StudentTMuSdTauEta::from_array([f64::NAN; 3]));
        };
        let Some(kernel) = StudentTKernel::try_new(theta.tau) else {
            return (f64::INFINITY, StudentTMuSdTauEta::from_array([f64::NAN; 3]));
        };
        let nll = kernel.nll_theta(y, location_scale);
        if !nll.is_finite() {
            return (nll, StudentTMuSdTauEta::from_array([f64::NAN; 3]));
        }

        let gradient = kernel.nll_gradient_theta(y, location_scale);
        let scale_per_sd = location_scale.sigma / theta.sigma;
        let scale_per_tau = location_scale.sigma / (theta.tau * (theta.tau - 2.0));
        let d_sigma = gradient.sigma * scale_per_sd;
        let d_tau = gradient.sigma.mul_add(scale_per_tau, gradient.tau);

        (
            nll,
            StudentTMuSdTauEta {
                mu: gradient.mu * MuLink::derivative_inverse(eta.mu),
                sigma: d_sigma * SigmaLink::derivative_inverse(eta.sigma),
                tau: d_tau * TauLink::derivative_inverse(eta.tau),
            },
        )
    }
}

impl<MuLink, SigmaLink, TauLink> Default for StudentTStdDev<MuLink, SigmaLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    TauLink: AboveTwoLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MuLink, SigmaLink, TauLink> for StudentTStdDev<MuLink, SigmaLink, TauLink>;
    parameters = (Mu, Sigma, Tau);
    arity = 3;
);

impl<MuLink, SigmaLink, TauLink> Family for StudentTStdDev<MuLink, SigmaLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    TauLink: AboveTwoLink<f64>,
{
    type Eta = StudentTMuSdTauEta;
    type Theta = StudentTMuSdTauTheta;
    type GradientEta = StudentTMuSdTauEta;
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

impl<MuLink, SigmaLink, TauLink> InitialEtaFromObservations<3>
    for StudentTStdDev<MuLink, SigmaLink, TauLink>
where
    MuLink: InitialEtaFromTheta<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    TauLink: InitialEtaFromTheta<f64> + AboveTwoLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| y.is_finite().then_some(y));
        let Some((mu, sigma)) = robust_location_scale(&values) else {
            return StudentTMuSdTauEta {
                mu: MuLink::initial_eta_from_theta(0.0),
                sigma: SigmaLink::initial_eta_from_theta(1.0),
                tau: TauLink::initial_eta_from_theta(5.0),
            };
        };

        StudentTMuSdTauEta {
            mu: MuLink::initial_eta_from_theta(mu),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
            tau: TauLink::initial_eta_from_theta(5.0),
        }
    }
}

impl<MuLink, SigmaLink, TauLink> HasCdf for StudentTStdDev<MuLink, SigmaLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    TauLink: AboveTwoLink<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        let Some(location_scale) = theta.location_scale() else {
            return f64::NAN;
        };
        if !y.is_finite() {
            return f64::NAN;
        }

        student_t_standard_cdf(theta.tau, (y - theta.mu) / location_scale.sigma)
    }
}

impl<MuLink, SigmaLink, TauLink> HasQuantile for StudentTStdDev<MuLink, SigmaLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    TauLink: AboveTwoLink<f64>,
{
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        let Some(location_scale) = theta.location_scale() else {
            return f64::NAN;
        };

        location_scale
            .sigma
            .mul_add(student_t_standard_quantile(theta.tau, p), theta.mu)
    }
}

impl<MuLink, SigmaLink, TauLink> HasCrps for StudentTStdDev<MuLink, SigmaLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    TauLink: AboveTwoLink<f64>,
{
    fn crps(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        let Some(location_scale) = theta.location_scale() else {
            return f64::NAN;
        };
        if !y.is_finite() {
            return f64::NAN;
        }

        student_t_crps_theta(theta.tau, y, location_scale)
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink, SigmaLink, TauLink> TrySimulate<Rng>
    for StudentTStdDev<MuLink, SigmaLink, TauLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    TauLink: AboveTwoLink<f64>,
{
    type Sample = f64;

    #[allow(clippy::suboptimal_flops)]
    fn try_sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Result<f64, SimulationError> {
        let Some(location_scale) = theta.location_scale() else {
            return Err(SimulationError::InvalidParameters(
                "standard-deviation Student-t theta",
            ));
        };

        let distribution = rand_distr::StudentT::new(theta.tau)
            .map_err(|_| SimulationError::BackendRejected("Student-t degrees of freedom"))?;
        let z = rand_distr::Distribution::sample(&distribution, rng);
        crate::simulation::ensure_finite(
            theta.mu + location_scale.sigma * z,
            "standard-deviation Student-t transform",
        )
    }
}

/// Predictors for standard-deviation Student's t on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudentTMuSdTauEta {
    /// Location predictor.
    pub mu: f64,
    /// Standard-deviation predictor.
    pub sigma: f64,
    /// Degrees-of-freedom predictor.
    pub tau: f64,
}

impl ParameterParts<3> for StudentTMuSdTauEta {
    #[inline]
    fn from_array(values: [f64; 3]) -> Self {
        Self {
            mu: values[0],
            sigma: values[1],
            tau: values[2],
        }
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mu,
            1 => self.sigma,
            2 => self.tau,
            _ => unreachable!("standard-deviation student-t eta only has indices 0 through 2"),
        }
    }
}

/// Standard-deviation Student's t parameters on the natural scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudentTMuSdTauTheta {
    /// Location parameter.
    pub mu: f64,
    /// Positive standard deviation.
    pub sigma: f64,
    /// Degrees of freedom; must be greater than two.
    pub tau: f64,
}

impl StudentTMuSdTauTheta {
    #[inline]
    fn location_scale(self) -> Option<StudentTTheta> {
        valid_stddev_theta(self).then(|| StudentTTheta {
            mu: self.mu,
            sigma: self.sigma * ((self.tau - 2.0) / self.tau).sqrt(),
        })
    }
}

#[inline]
fn valid_stddev_theta(theta: StudentTMuSdTauTheta) -> bool {
    theta.mu.is_finite()
        && theta.sigma > 0.0
        && theta.sigma.is_finite()
        && theta.tau > 2.0
        && theta.tau.is_finite()
}
