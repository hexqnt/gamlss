use std::marker::PhantomData;

use gamlss_core::{
    DegreesOfFreedom, Family, HasCdf, HasCrps, HasQuantile, InitialEtaFromObservations,
    InitialEtaFromTheta, Log, ObservationView, ParameterParts, PositiveLink,
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};
use gamlss_special::{invert_positive_cdf, regularized_gamma_lower};

use crate::domain::is_positive_finite;
use crate::initial::{positive_floor, weighted_values};

use super::chi_squared::{chi_squared_d_dof, chi_squared_nll};

/// Chi distribution with a log link for degrees of freedom.
pub type ChiDegreesOfFreedom = Chi<Log>;

/// Chi family parameterized by positive degrees of freedom.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/distributions/chi.svg")
)]
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chi<DofLink = Log> {
    marker: PhantomData<DofLink>,
}

impl<DofLink> Chi<DofLink>
where
    DofLink: PositiveLink<f64>,
{
    /// Creates a stateless Chi family.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    fn theta_from_eta(eta: ChiEta) -> ChiTheta {
        ChiTheta {
            degrees_of_freedom: DofLink::inverse(eta.degrees_of_freedom),
        }
    }

    fn nll_theta(y: f64, theta: ChiTheta) -> f64 {
        if !is_positive_finite(y) || !is_positive_finite(theta.degrees_of_freedom) {
            return f64::INFINITY;
        }
        let squared = y * y;
        if !squared.is_finite() {
            return f64::INFINITY;
        }
        chi_squared_nll(squared, theta.degrees_of_freedom) - (2.0 * y).ln()
    }

    fn cdf_theta(y: f64, theta: ChiTheta) -> f64 {
        if y <= 0.0 {
            0.0
        } else {
            let half_squared = 0.5 * y * y;
            if half_squared.is_infinite() {
                1.0
            } else {
                regularized_gamma_lower(0.5 * theta.degrees_of_freedom, half_squared)
            }
        }
    }
}

impl<DofLink> Default for Chi<DofLink>
where
    DofLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<DofLink> for Chi<DofLink>;
    parameters = (DegreesOfFreedom,);
    arity = 1;
);

impl<DofLink> Family for Chi<DofLink>
where
    DofLink: PositiveLink<f64>,
{
    type Eta = ChiEta;
    type Theta = ChiTheta;
    type GradientEta = ChiEta;
    type Observation<'obs> = f64;
    type Workspace = ();

    fn workspace(&self) -> Self::Workspace {}
    fn theta(&self, eta: &Self::Eta, _workspace: &mut ()) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }
    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut ()) -> f64 {
        Self::nll_theta(y, *theta)
    }
    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut (),
    ) -> (f64, Self::GradientEta) {
        let theta = Self::theta_from_eta(*eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (
                nll,
                ChiEta {
                    degrees_of_freedom: f64::NAN,
                },
            );
        }
        let d_dof = chi_squared_d_dof(y * y, theta.degrees_of_freedom);
        (
            nll,
            ChiEta {
                degrees_of_freedom: d_dof * DofLink::derivative_inverse(eta.degrees_of_freedom),
            },
        )
    }
}

impl<DofLink> InitialEtaFromObservations<1> for Chi<DofLink>
where
    DofLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = f64> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| {
            let squared = y * y;
            is_positive_finite(squared).then_some(squared)
        });
        let dof = crate::initial::weighted_mean(&values).map_or(1.0, positive_floor);
        ChiEta {
            degrees_of_freedom: DofLink::initial_eta_from_theta(dof),
        }
    }
}

impl<DofLink> HasCdf for Chi<DofLink>
where
    DofLink: PositiveLink<f64>,
{
    fn cdf(&self, y: f64, theta: &Self::Theta) -> f64 {
        if !y.is_finite() || !is_positive_finite(theta.degrees_of_freedom) {
            return f64::NAN;
        }
        Self::cdf_theta(y, *theta)
    }
}

impl<DofLink> HasQuantile for Chi<DofLink>
where
    DofLink: PositiveLink<f64>,
{
    fn quantile(&self, probability: f64, theta: &Self::Theta) -> f64 {
        if !is_positive_finite(theta.degrees_of_freedom) {
            return f64::NAN;
        }
        invert_positive_cdf(probability, |y| Self::cdf_theta(y, *theta))
    }
}

impl<DofLink> HasCrps for Chi<DofLink>
where
    DofLink: PositiveLink<f64>,
{
    fn crps(&self, y: f64, theta: &Self::Theta) -> f64 {
        if y < 0.0 || !y.is_finite() || !is_positive_finite(theta.degrees_of_freedom) {
            return f64::NAN;
        }
        crate::crps::integrate_cdf_crps(y, theta.degrees_of_freedom.sqrt(), |x| {
            Self::cdf_theta(x, *theta)
        })
    }
}

#[cfg(feature = "rand")]
impl<Rng, DofLink> TrySimulate<Rng> for Chi<DofLink>
where
    Rng: rand::Rng,
    DofLink: PositiveLink<f64>,
{
    type Sample = f64;

    fn try_sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Result<f64, SimulationError> {
        if !is_positive_finite(theta.degrees_of_freedom) {
            return Err(SimulationError::InvalidParameters("Chi theta"));
        }
        let distribution = rand_distr::ChiSquared::new(theta.degrees_of_freedom)
            .map_err(|_| SimulationError::BackendRejected("Chi degrees of freedom"))?;
        let sample: f64 = rand_distr::Distribution::sample(&distribution, rng);
        crate::simulation::ensure_finite(sample.sqrt(), "Chi sample")
    }
}

/// Link-scale degrees-of-freedom predictor for [`Chi`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChiEta {
    /// Degrees-of-freedom predictor.
    pub degrees_of_freedom: f64,
}

impl ParameterParts<1> for ChiEta {
    fn from_array(values: [f64; 1]) -> Self {
        Self {
            degrees_of_freedom: values[0],
        }
    }
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.degrees_of_freedom,
            _ => unreachable!("Chi eta only has index 0"),
        }
    }
}

/// Natural-scale degrees of freedom for [`Chi`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChiTheta {
    /// Positive degrees of freedom.
    pub degrees_of_freedom: f64,
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{HasCdf, HasQuantile};

    use super::{ChiDegreesOfFreedom, ChiTheta};
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn gradient_matches_finite_difference() {
        assert_gradient_matches_finite_difference::<_, 1>(&ChiDegreesOfFreedom::new(), 1.7, [1.2]);
    }

    #[test]
    fn quantile_inverts_cdf() {
        let family = ChiDegreesOfFreedom::new();
        let theta = ChiTheta {
            degrees_of_freedom: 4.5,
        };
        assert_relative_eq!(
            family.quantile(family.cdf(2.3, &theta), &theta),
            2.3,
            epsilon = 1.0e-9
        );
        assert_relative_eq!(family.cdf(f64::MAX, &theta), 1.0, epsilon = f64::EPSILON);
    }
}
