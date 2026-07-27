use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasCrps, HasQuantile, InitialEtaFromObservations, InitialEtaFromTheta, Link,
    Mu, ObservationView, ParameterParts, PositiveLink, Sigma, Tau,
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

use crate::initial::{robust_location_scale, weighted_values};
use crate::link::positive_inverse_and_log;

use super::{
    StudentTKernel, StudentTTheta, student_t_crps_theta, student_t_standard_cdf,
    student_t_standard_quantile,
};

/// Student's t location-scale family with estimated degrees of freedom.
///
/// The natural parameters are location $\mu\in\mathbb{R}$, scale $\sigma>0$, and degrees of freedom $\tau>0$. The fields of [`StudentTMuSigmaTauTheta`] use these same names; for the density and moment conditions, see [`StudentTTheta`].
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../../doc-assets/distributions/student_t.svg")
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StudentTDynamic<
    MuLink = gamlss_core::Identity,
    SigmaLink = gamlss_core::Log,
    TauLink = gamlss_core::Log,
> {
    marker: PhantomData<(MuLink, SigmaLink, TauLink)>,
}

impl<MuLink, SigmaLink, TauLink> StudentTDynamic<MuLink, SigmaLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    /// Creates a stateless dynamic-DF Student's t family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(eta: StudentTMuSigmaTauEta) -> StudentTMuSigmaTauTheta {
        StudentTMuSigmaTauTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
            tau: TauLink::inverse(eta.tau),
        }
    }

    #[inline]
    fn theta_and_log_sigma_from_eta(eta: StudentTMuSigmaTauEta) -> (StudentTMuSigmaTauTheta, f64) {
        let (sigma, log_sigma) = positive_inverse_and_log::<SigmaLink>(eta.sigma);
        (
            StudentTMuSigmaTauTheta {
                mu: MuLink::inverse(eta.mu),
                sigma,
                tau: TauLink::inverse(eta.tau),
            },
            log_sigma,
        )
    }

    #[inline]
    fn nll_theta(y: f64, theta: StudentTMuSigmaTauTheta) -> f64 {
        let Some(kernel) = StudentTKernel::try_new(theta.tau) else {
            return f64::INFINITY;
        };
        kernel.nll_theta(y, theta.location_scale())
    }

    #[inline]
    fn nll_and_gradient_eta_values(
        y: f64,
        eta: StudentTMuSigmaTauEta,
    ) -> (f64, StudentTMuSigmaTauEta) {
        let (theta, log_sigma) = Self::theta_and_log_sigma_from_eta(eta);
        let Some(kernel) = StudentTKernel::try_new(theta.tau) else {
            return (
                f64::INFINITY,
                StudentTMuSigmaTauEta::from_array([f64::NAN; 3]),
            );
        };
        let nll = kernel.nll_theta_with_log_sigma(y, theta.location_scale(), log_sigma);
        if !nll.is_finite() {
            return (nll, StudentTMuSigmaTauEta::from_array([f64::NAN; 3]));
        }

        let gradient = kernel.nll_gradient_theta(y, theta.location_scale());
        (
            nll,
            StudentTMuSigmaTauEta {
                mu: gradient.mu * MuLink::derivative_inverse(eta.mu),
                sigma: gradient.sigma * theta.sigma * SigmaLink::derivative_log_inverse(eta.sigma),
                tau: gradient.tau * theta.tau * TauLink::derivative_log_inverse(eta.tau),
            },
        )
    }
}

impl<MuLink, SigmaLink, TauLink> Default for StudentTDynamic<MuLink, SigmaLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MuLink, SigmaLink, TauLink> for StudentTDynamic<MuLink, SigmaLink, TauLink>;
    parameters = (Mu, Sigma, Tau);
    arity = 3;
);

impl<MuLink, SigmaLink, TauLink> Family for StudentTDynamic<MuLink, SigmaLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    type Eta = StudentTMuSigmaTauEta;
    type Theta = StudentTMuSigmaTauTheta;
    type GradientEta = StudentTMuSigmaTauEta;
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
        let Some(kernel) = StudentTKernel::try_new(theta.tau) else {
            return f64::INFINITY;
        };
        kernel.nll_theta_with_log_sigma(y, theta.location_scale(), log_sigma)
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
    for StudentTDynamic<MuLink, SigmaLink, TauLink>
where
    MuLink: InitialEtaFromTheta<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    TauLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| y.is_finite().then_some(y));
        let Some((mu, sigma)) = robust_location_scale(&values) else {
            return StudentTMuSigmaTauEta {
                mu: MuLink::initial_eta_from_theta(0.0),
                sigma: SigmaLink::initial_eta_from_theta(1.0),
                tau: TauLink::initial_eta_from_theta(5.0),
            };
        };

        StudentTMuSigmaTauEta {
            mu: MuLink::initial_eta_from_theta(mu),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
            tau: TauLink::initial_eta_from_theta(5.0),
        }
    }
}

impl<MuLink, SigmaLink, TauLink> HasCdf for StudentTDynamic<MuLink, SigmaLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        if !y.is_finite() || !valid_dynamic_theta(*theta) {
            return f64::NAN;
        }

        student_t_standard_cdf(theta.tau, (y - theta.mu) / theta.sigma)
    }
}

impl<MuLink, SigmaLink, TauLink> HasQuantile for StudentTDynamic<MuLink, SigmaLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        if !valid_dynamic_theta(*theta) {
            return f64::NAN;
        }

        theta
            .sigma
            .mul_add(student_t_standard_quantile(theta.tau, p), theta.mu)
    }
}

impl<MuLink, SigmaLink, TauLink> HasCrps for StudentTDynamic<MuLink, SigmaLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    fn crps(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        if !y.is_finite() || !valid_dynamic_theta(*theta) {
            return f64::NAN;
        }

        student_t_crps_theta(theta.tau, y, theta.location_scale())
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink, SigmaLink, TauLink> TrySimulate<Rng>
    for StudentTDynamic<MuLink, SigmaLink, TauLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    type Sample = f64;

    #[allow(clippy::suboptimal_flops)]
    fn try_sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Result<f64, SimulationError> {
        if !valid_dynamic_theta(*theta) {
            return Err(SimulationError::InvalidParameters("Student-t theta"));
        }

        let distribution = rand_distr::StudentT::new(theta.tau)
            .map_err(|_| SimulationError::BackendRejected("Student-t degrees of freedom"))?;
        let z = rand_distr::Distribution::sample(&distribution, rng);
        crate::simulation::ensure_finite(theta.mu + theta.sigma * z, "Student-t transform")
    }
}

/// Predictors for dynamic-DF Student's t on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudentTMuSigmaTauEta {
    /// Location predictor.
    pub mu: f64,
    /// Scale predictor.
    pub sigma: f64,
    /// Degrees-of-freedom predictor.
    pub tau: f64,
}

impl ParameterParts<3> for StudentTMuSigmaTauEta {
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
            _ => unreachable!("dynamic student-t eta only has indices 0 through 2"),
        }
    }
}

/// Dynamic-DF Student's t parameters on the natural scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudentTMuSigmaTauTheta {
    /// Location parameter.
    pub mu: f64,
    /// Positive scale parameter.
    pub sigma: f64,
    /// Positive degrees of freedom.
    pub tau: f64,
}

impl StudentTMuSigmaTauTheta {
    #[inline]
    const fn location_scale(self) -> StudentTTheta {
        StudentTTheta {
            mu: self.mu,
            sigma: self.sigma,
        }
    }
}

#[inline]
fn valid_dynamic_theta(theta: StudentTMuSigmaTauTheta) -> bool {
    theta.mu.is_finite()
        && theta.sigma > 0.0
        && theta.sigma.is_finite()
        && theta.tau > 0.0
        && theta.tau.is_finite()
}
