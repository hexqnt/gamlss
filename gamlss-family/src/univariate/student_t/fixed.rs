use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasCrps, HasQuantile, InitialEtaFromObservations, InitialEtaFromTheta, Link,
    ModelError, Mu, ObservationView, ParameterParts, PositiveLink, Sigma,
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

use crate::initial::{robust_location_scale, weighted_values};

use super::{
    StudentTTheta, student_t_crps_theta, student_t_nll_gradient_theta, student_t_nll_theta,
    student_t_standard_cdf, student_t_standard_quantile,
};

/// Student's t location-scale family with a fixed number of degrees of freedom.
///
/// `MuLink` and `SigmaLink` control the link functions for the location and scale parameters respectively. Defaults to `Identity` for `mu` and `Log` for `sigma`. For the density and moment conditions, see [`StudentTTheta`].
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../../doc-assets/distributions/student_t_fixed.svg")
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudentT<MuLink = gamlss_core::Identity, SigmaLink = gamlss_core::Log> {
    degrees_of_freedom: f64,
    marker: PhantomData<(MuLink, SigmaLink)>,
}

impl<MuLink, SigmaLink> StudentT<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    /// Creates a Student's t family with finite positive degrees of freedom.
    pub fn try_new(degrees_of_freedom: f64) -> Result<Self, ModelError> {
        if !degrees_of_freedom.is_finite() || degrees_of_freedom <= 0.0 {
            return Err(ModelError::InvalidParameter {
                parameter: "degrees_of_freedom",
                expected: "finite and > 0",
            });
        }

        Ok(Self {
            degrees_of_freedom,
            marker: PhantomData,
        })
    }

    /// Returns the fixed degrees of freedom.
    #[must_use]
    pub const fn degrees_of_freedom(&self) -> f64 {
        self.degrees_of_freedom
    }

    #[inline]
    fn theta_from_eta(eta: StudentTEta) -> StudentTTheta {
        StudentTTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
        }
    }

    #[inline]
    fn nll_theta(&self, y: f64, theta: StudentTTheta) -> f64 {
        student_t_nll_theta(self.degrees_of_freedom, y, theta)
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn nll_and_gradient_eta_values(&self, y: f64, eta: StudentTEta) -> (f64, StudentTEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = self.nll_theta(y, theta);
        if !nll.is_finite() {
            return (
                nll,
                StudentTEta {
                    mu: f64::NAN,
                    sigma: f64::NAN,
                },
            );
        }

        let gradient = student_t_nll_gradient_theta(self.degrees_of_freedom, y, theta);

        (
            nll,
            StudentTEta {
                mu: gradient.mu * MuLink::derivative_inverse(eta.mu),
                sigma: gradient.sigma * SigmaLink::derivative_inverse(eta.sigma),
            },
        )
    }
}

impl<MuLink, SigmaLink> Default for StudentT<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::try_new(5.0).expect("default degrees_of_freedom is valid")
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MuLink, SigmaLink> for StudentT<MuLink, SigmaLink>;
    parameters = (Mu, Sigma);
    arity = 2;
);

impl<MuLink, SigmaLink> Family for StudentT<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    type Eta = StudentTEta;
    type Theta = StudentTTheta;
    type GradientEta = StudentTEta;
    type Observation<'obs> = f64;
    type Workspace = ();

    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    #[inline]
    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        self.nll_theta(y, *theta)
    }

    #[inline]
    fn nll_eta(&self, y: f64, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> f64 {
        self.nll_theta(y, Self::theta_from_eta(*eta))
    }

    #[inline]
    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        self.nll_and_gradient_eta_values(y, *eta)
    }
}

impl<MuLink, SigmaLink> InitialEtaFromObservations<2> for StudentT<MuLink, SigmaLink>
where
    MuLink: InitialEtaFromTheta<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| y.is_finite().then_some(y));
        let Some((mu, sigma)) = robust_location_scale(&values) else {
            return StudentTEta::from_array([0.0, 0.0]);
        };

        StudentTEta {
            mu: MuLink::initial_eta_from_theta(mu),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
        }
    }
}

impl<MuLink, SigmaLink> HasCdf for StudentT<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        if !y.is_finite() || !theta.mu.is_finite() || theta.sigma <= 0.0 || !theta.sigma.is_finite()
        {
            return f64::NAN;
        }

        student_t_standard_cdf(self.degrees_of_freedom, (y - theta.mu) / theta.sigma)
    }
}

impl<MuLink, SigmaLink> HasQuantile for StudentT<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    #[allow(clippy::suboptimal_flops)]
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        if theta.sigma <= 0.0 || !theta.sigma.is_finite() || !theta.mu.is_finite() {
            return f64::NAN;
        }

        theta.mu + theta.sigma * student_t_standard_quantile(self.degrees_of_freedom, p)
    }
}

impl<MuLink, SigmaLink> HasCrps for StudentT<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn crps(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        if !y.is_finite()
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || self.degrees_of_freedom <= 1.0
        {
            return f64::NAN;
        }

        student_t_crps_theta(self.degrees_of_freedom, y, *theta)
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink, SigmaLink> TrySimulate<Rng> for StudentT<MuLink, SigmaLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    type Sample = f64;

    #[allow(clippy::suboptimal_flops)]
    fn try_sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Result<f64, SimulationError> {
        if theta.sigma <= 0.0 || !theta.sigma.is_finite() || !theta.mu.is_finite() {
            return Err(SimulationError::InvalidParameters("Student-t theta"));
        }

        let distribution = rand_distr::StudentT::new(self.degrees_of_freedom)
            .map_err(|_| SimulationError::BackendRejected("Student-t degrees of freedom"))?;
        let z = rand_distr::Distribution::sample(&distribution, rng);
        crate::simulation::ensure_finite(theta.mu + theta.sigma * z, "Student-t transform")
    }
}

/// Predictors for the Student's t distribution on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudentTEta {
    /// Location predictor.
    pub mu: f64,
    /// Scale predictor.
    pub sigma: f64,
}

impl ParameterParts<2> for StudentTEta {
    #[inline]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            mu: values[0],
            sigma: values[1],
        }
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mu,
            1 => self.sigma,
            _ => unreachable!("student-t eta only has indices 0 and 1"),
        }
    }
}
