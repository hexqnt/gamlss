use std::marker::PhantomData;

use gamlss_core::{
    DegreesOfFreedom, Family, HasCdf, HasQuantile, InitialEtaFromObservations, InitialEtaFromTheta,
    Log, ObservationView, ParameterParts, PositiveLink,
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};
use gamlss_special::{
    digamma_minus_ln, invert_positive_cdf, ln_gamma_stirling_residual, regularized_gamma_lower,
};

use crate::domain::is_positive_finite;
use crate::initial::{positive_floor, weighted_values};

/// Chi-squared distribution with a log link for degrees of freedom.
pub type ChiSquaredDegreesOfFreedom = ChiSquared<Log>;

/// Chi-squared family parameterized by positive degrees of freedom.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/distributions/chi_squared.svg")
)]
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChiSquared<DofLink = Log> {
    marker: PhantomData<DofLink>,
}

impl<DofLink> ChiSquared<DofLink>
where
    DofLink: PositiveLink<f64>,
{
    /// Creates a stateless chi-squared family.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    fn theta_from_eta(eta: ChiSquaredEta) -> ChiSquaredTheta {
        ChiSquaredTheta {
            degrees_of_freedom: DofLink::inverse(eta.degrees_of_freedom),
        }
    }

    fn nll_theta(y: f64, theta: ChiSquaredTheta) -> f64 {
        if !is_positive_finite(y) || !is_positive_finite(theta.degrees_of_freedom) {
            return f64::INFINITY;
        }
        chi_squared_nll(y, theta.degrees_of_freedom)
    }

    fn cdf_theta(y: f64, theta: ChiSquaredTheta) -> f64 {
        if y <= 0.0 {
            0.0
        } else {
            regularized_gamma_lower(0.5 * theta.degrees_of_freedom, 0.5 * y)
        }
    }
}

#[inline]
fn log_observation_over_dof(y: f64, dof: f64) -> f64 {
    let ratio = y / dof;
    if ratio.is_finite() && ratio > 0.0 {
        let centered = ratio - 1.0;
        if centered.abs() <= 0.5 {
            return centered.ln_1p();
        }
    }
    y.ln() - dof.ln()
}

#[inline]
pub(super) fn chi_squared_nll(y: f64, dof: f64) -> f64 {
    let half_dof = 0.5 * dof;
    let ratio = y / dof;
    let deviance = if ratio.is_finite() && ratio > 0.0 && (ratio - 1.0).abs() <= 0.5 {
        let centered = ratio - 1.0;
        half_dof * (centered - centered.ln_1p())
    } else {
        half_dof.mul_add(
            -log_observation_over_dof(y, dof),
            0.5f64.mul_add(y, -half_dof),
        )
    };
    ln_gamma_stirling_residual(half_dof) + deviance + y.ln()
}

#[inline]
pub(super) fn chi_squared_d_dof(y: f64, dof: f64) -> f64 {
    0.5 * (digamma_minus_ln(0.5 * dof) - log_observation_over_dof(y, dof))
}

impl<DofLink> Default for ChiSquared<DofLink>
where
    DofLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<DofLink> for ChiSquared<DofLink>;
    parameters = (DegreesOfFreedom,);
    arity = 1;
);

impl<DofLink> Family for ChiSquared<DofLink>
where
    DofLink: PositiveLink<f64>,
{
    type Eta = ChiSquaredEta;
    type Theta = ChiSquaredTheta;
    type GradientEta = ChiSquaredEta;
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
                ChiSquaredEta {
                    degrees_of_freedom: f64::NAN,
                },
            );
        }
        let d_dof = chi_squared_d_dof(y, theta.degrees_of_freedom);
        (
            nll,
            ChiSquaredEta {
                degrees_of_freedom: d_dof * DofLink::derivative_inverse(eta.degrees_of_freedom),
            },
        )
    }
}

impl<DofLink> InitialEtaFromObservations<1> for ChiSquared<DofLink>
where
    DofLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = f64> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| is_positive_finite(y).then_some(y));
        let dof = crate::initial::weighted_mean(&values).map_or(1.0, positive_floor);
        ChiSquaredEta {
            degrees_of_freedom: DofLink::initial_eta_from_theta(dof),
        }
    }
}

impl<DofLink> HasCdf for ChiSquared<DofLink>
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

impl<DofLink> HasQuantile for ChiSquared<DofLink>
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

#[cfg(feature = "rand")]
impl<Rng, DofLink> TrySimulate<Rng> for ChiSquared<DofLink>
where
    Rng: rand::Rng,
    DofLink: PositiveLink<f64>,
{
    type Sample = f64;

    fn try_sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Result<f64, SimulationError> {
        if !is_positive_finite(theta.degrees_of_freedom) {
            return Err(SimulationError::InvalidParameters("Chi-squared theta"));
        }
        let distribution = rand_distr::ChiSquared::new(theta.degrees_of_freedom)
            .map_err(|_| SimulationError::BackendRejected("Chi-squared degrees of freedom"))?;
        crate::simulation::ensure_finite(
            rand_distr::Distribution::sample(&distribution, rng),
            "Chi-squared sample",
        )
    }
}

/// Link-scale degrees-of-freedom predictor for [`ChiSquared`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChiSquaredEta {
    /// Degrees-of-freedom predictor.
    pub degrees_of_freedom: f64,
}

impl ParameterParts<1> for ChiSquaredEta {
    fn from_array(values: [f64; 1]) -> Self {
        Self {
            degrees_of_freedom: values[0],
        }
    }
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.degrees_of_freedom,
            _ => unreachable!("chi-squared eta only has index 0"),
        }
    }
}

/// Natural-scale degrees of freedom for [`ChiSquared`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChiSquaredTheta {
    /// Positive degrees of freedom.
    pub degrees_of_freedom: f64,
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{Family, HasCdf, HasQuantile};

    use super::{ChiSquaredDegreesOfFreedom, ChiSquaredTheta};
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn gradient_matches_finite_difference() {
        assert_gradient_matches_finite_difference::<_, 1>(
            &ChiSquaredDegreesOfFreedom::new(),
            1.7,
            [1.2],
        );
    }

    #[test]
    fn quantile_inverts_cdf() {
        let family = ChiSquaredDegreesOfFreedom::new();
        let theta = ChiSquaredTheta {
            degrees_of_freedom: 4.5,
        };
        assert_relative_eq!(
            family.quantile(family.cdf(2.3, &theta), &theta),
            2.3,
            epsilon = 1.0e-9
        );
    }

    #[test]
    fn concentrated_case_preserves_small_normalizer() {
        let family = ChiSquaredDegreesOfFreedom::new();
        let dof = 1.0e16;
        let theta = ChiSquaredTheta {
            degrees_of_freedom: dof,
        };
        let nll = family.nll(dof, &theta, &mut ());
        let expected = 0.5 * (4.0 * std::f64::consts::PI * dof).ln();
        assert_relative_eq!(nll, expected, epsilon = 1.0e-13);
    }
}
