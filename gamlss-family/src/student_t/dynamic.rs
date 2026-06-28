use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Family, HasCdf, HasCrps, HasQuantile, InitialEtaFromTheta, Link, Mu, ObservationView,
    ParameterParts, ParameterizedFamily, PositiveLink, Sigma, Tau,
};

use crate::initial::{robust_location_scale, weighted_values};

use super::{
    StudentTTheta, student_t_crps_theta, student_t_nll_gradient_theta, student_t_nll_theta,
    student_t_standard_cdf, student_t_standard_quantile,
};

/// Student's t location-scale family with estimated degrees of freedom.
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
    TauLink: Link<f64>,
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
    fn nll_theta(y: f64, theta: StudentTMuSigmaTauTheta) -> f64 {
        if !valid_dynamic_theta(theta) {
            return f64::INFINITY;
        }
        student_t_nll_theta(theta.tau, y, theta.location_scale())
    }

    #[inline]
    fn nll_and_gradient_eta_values(
        y: f64,
        eta: StudentTMuSigmaTauEta,
    ) -> (f64, StudentTMuSigmaTauEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (nll, StudentTMuSigmaTauEta::from_array([f64::NAN; 3]));
        }

        let gradient = student_t_nll_gradient_theta(theta.tau, y, theta.location_scale());
        (
            nll,
            StudentTMuSigmaTauEta {
                mu: gradient.mu * MuLink::derivative_inverse(eta.mu),
                sigma: gradient.sigma * SigmaLink::derivative_inverse(eta.sigma),
                tau: gradient.tau * TauLink::derivative_inverse(eta.tau),
            },
        )
    }
}

impl<MuLink, SigmaLink, TauLink> Default for StudentTDynamic<MuLink, SigmaLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    TauLink: Link<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<MuLink, SigmaLink, TauLink> Family for StudentTDynamic<MuLink, SigmaLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    TauLink: Link<f64>,
{
    type Eta = StudentTMuSigmaTauEta;
    type Theta = StudentTMuSigmaTauTheta;
    type NllGradientEta = StudentTMuSigmaTauEta;
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

impl<MuLink, SigmaLink, TauLink> ParameterizedFamily<3>
    for StudentTDynamic<MuLink, SigmaLink, TauLink>
where
    MuLink: InitialEtaFromTheta<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    TauLink: InitialEtaFromTheta<f64> + Link<f64>,
{
    type Params = (Mu, Sigma, Tau);
    type Links = (MuLink, SigmaLink, TauLink);

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
    TauLink: Link<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: Self::Theta) -> f64 {
        if !y.is_finite() || !valid_dynamic_theta(theta) {
            return f64::NAN;
        }

        student_t_standard_cdf(theta.tau, (y - theta.mu) / theta.sigma)
    }
}

impl<MuLink, SigmaLink, TauLink> HasQuantile for StudentTDynamic<MuLink, SigmaLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    TauLink: Link<f64>,
{
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
        if !valid_dynamic_theta(theta) {
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
    TauLink: Link<f64>,
{
    fn crps(&self, y: Self::Observation<'_>, theta: Self::Theta) -> f64 {
        if !y.is_finite() || !valid_dynamic_theta(theta) {
            return f64::NAN;
        }

        student_t_crps_theta(theta.tau, y, theta.location_scale())
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink, SigmaLink, TauLink> CanSimulate<Rng>
    for StudentTDynamic<MuLink, SigmaLink, TauLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    TauLink: Link<f64>,
{
    type Sample = f64;

    #[allow(clippy::suboptimal_flops)]
    fn sample(&self, rng: &mut Rng, theta: Self::Theta) -> f64 {
        if !valid_dynamic_theta(theta) {
            return f64::NAN;
        }

        let z = rand_distr::Distribution::sample(
            &rand_distr::StudentT::new(theta.tau)
                .expect("validated degrees_of_freedom must construct"),
            rng,
        );
        theta.mu + theta.sigma * z
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
