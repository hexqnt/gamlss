use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasCrps, HasQuantile, InitialEtaFromObservations, InitialEtaFromTheta, Link,
    Logit, ModelError, Mu, ObservationView, ParameterParts, PositiveLink, Sigma, UnitIntervalLink,
    ZeroProbability,
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

use crate::{
    domain::{ScalarObservationDomain, is_finite_location_scale, is_strict_probability},
    initial::{probability_floor, robust_location_scale, weighted_values},
};

use super::{StudentT, StudentTEta, StudentTTheta};

/// Student's t location-scale family with an additional point mass at zero.
///
/// Let `f_T` be the continuous Student-t density with fixed degrees of freedom
/// and let $\pi\in(0,1)$ be the zero-mass probability. The mixed distribution is
///
/// $$
/// \Pr(Y=0)=\pi,
/// \qquad
/// f_Y(y)=(1-\pi)f_T(y\mid\mu,\sigma),\quad y\ne0.
/// $$
///
/// `MuLink`, `SigmaLink`, and `ZeroProbabilityLink` control the location, scale,
/// and zero-probability links respectively. They default to `Identity`, `Log`,
/// and `Logit`. The continuous component uses the same fixed-DF kernel as
/// [`StudentT`], including its cached normalizing constant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZeroAdjustedStudentT<
    MuLink = gamlss_core::Identity,
    SigmaLink = gamlss_core::Log,
    ZeroProbabilityLink = Logit,
> {
    component: StudentT<MuLink, SigmaLink>,
    marker: PhantomData<ZeroProbabilityLink>,
}

impl<MuLink, SigmaLink, ZeroProbabilityLink>
    ZeroAdjustedStudentT<MuLink, SigmaLink, ZeroProbabilityLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    /// Creates a zero-adjusted Student-t family with finite positive degrees of freedom.
    pub fn try_new(degrees_of_freedom: f64) -> Result<Self, ModelError> {
        Ok(Self {
            component: StudentT::try_new(degrees_of_freedom)?,
            marker: PhantomData,
        })
    }

    /// Returns the fixed degrees of freedom of the continuous component.
    #[must_use]
    #[inline]
    pub const fn degrees_of_freedom(&self) -> f64 {
        self.component.degrees_of_freedom()
    }

    /// Returns the continuous Student-t component family.
    #[must_use]
    #[inline]
    pub const fn component(&self) -> &StudentT<MuLink, SigmaLink> {
        &self.component
    }

    #[inline]
    const fn component_eta(eta: ZeroAdjustedStudentTEta) -> StudentTEta {
        StudentTEta {
            mu: eta.mu,
            sigma: eta.sigma,
        }
    }

    #[inline]
    fn theta_from_eta(eta: ZeroAdjustedStudentTEta) -> ZeroAdjustedStudentTTheta {
        ZeroAdjustedStudentTTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
            zero_probability: ZeroProbabilityLink::inverse(eta.zero_probability),
        }
    }

    #[inline]
    fn valid_theta(theta: ZeroAdjustedStudentTTheta) -> bool {
        is_finite_location_scale(theta.mu, theta.sigma)
            && is_strict_probability(theta.zero_probability)
    }

    #[inline]
    fn invalid_gradient() -> ZeroAdjustedStudentTEta {
        ZeroAdjustedStudentTEta::from_array([f64::NAN; 3])
    }

    #[inline]
    fn nll_and_gradient_eta_values(
        &self,
        y: f64,
        eta: ZeroAdjustedStudentTEta,
    ) -> (f64, ZeroAdjustedStudentTEta) {
        let theta = Self::theta_from_eta(eta);
        if !y.is_finite() || !Self::valid_theta(theta) {
            return (f64::INFINITY, Self::invalid_gradient());
        }

        if y == 0.0 {
            return (
                -theta.zero_probability.ln(),
                ZeroAdjustedStudentTEta {
                    mu: 0.0,
                    sigma: 0.0,
                    zero_probability: -ZeroProbabilityLink::derivative_inverse(
                        eta.zero_probability,
                    ) / theta.zero_probability,
                },
            );
        }

        let (component_nll, component_gradient) = self.component.nll_and_gradient_eta(
            y,
            &Self::component_eta(eta),
            &mut self.component.workspace(),
        );
        let nll = component_nll - (-theta.zero_probability).ln_1p();
        if !nll.is_finite() {
            return (nll, Self::invalid_gradient());
        }

        (
            nll,
            ZeroAdjustedStudentTEta {
                mu: component_gradient.mu,
                sigma: component_gradient.sigma,
                zero_probability: ZeroProbabilityLink::derivative_inverse(eta.zero_probability)
                    / (1.0 - theta.zero_probability),
            },
        )
    }
}

impl<MuLink, SigmaLink, ZeroProbabilityLink> Default
    for ZeroAdjustedStudentT<MuLink, SigmaLink, ZeroProbabilityLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn default() -> Self {
        Self::try_new(5.0).expect("default degrees_of_freedom is valid")
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MuLink, SigmaLink, ZeroProbabilityLink> for ZeroAdjustedStudentT<MuLink, SigmaLink, ZeroProbabilityLink>;
    parameters = (Mu, Sigma, ZeroProbability);
    arity = 3;
);

impl<MuLink, SigmaLink, ZeroProbabilityLink> Family
    for ZeroAdjustedStudentT<MuLink, SigmaLink, ZeroProbabilityLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    type Eta = ZeroAdjustedStudentTEta;
    type Theta = ZeroAdjustedStudentTTheta;
    type GradientEta = ZeroAdjustedStudentTEta;
    type Observation<'obs> = f64;
    type Workspace = ();

    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    #[inline]
    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    #[inline]
    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        if !y.is_finite() || !Self::valid_theta(*theta) {
            return f64::INFINITY;
        }
        if y == 0.0 {
            -theta.zero_probability.ln()
        } else {
            self.component
                .nll(y, &theta.component(), &mut self.component.workspace())
                - (-theta.zero_probability).ln_1p()
        }
    }

    #[inline]
    fn nll_eta(&self, y: f64, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> f64 {
        let theta = Self::theta_from_eta(*eta);
        if !y.is_finite() || !Self::valid_theta(theta) {
            return f64::INFINITY;
        }
        if y == 0.0 {
            -theta.zero_probability.ln()
        } else {
            self.component.nll_eta(
                y,
                &Self::component_eta(*eta),
                &mut self.component.workspace(),
            ) - (-theta.zero_probability).ln_1p()
        }
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

impl<MuLink, SigmaLink, ZeroProbabilityLink> InitialEtaFromObservations<3>
    for ZeroAdjustedStudentT<MuLink, SigmaLink, ZeroProbabilityLink>
where
    MuLink: InitialEtaFromTheta<f64> + Link<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    ZeroProbabilityLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let mut values = weighted_values::<Self, _, _>(obs, |y| y.is_finite().then_some(y));
        let zero_weight = values
            .iter()
            .filter(|(value, _)| *value == 0.0)
            .map(|(_, weight)| *weight)
            .sum::<f64>();
        let total_weight = values.iter().map(|(_, weight)| *weight).sum::<f64>();
        values.retain(|(value, _)| *value != 0.0);
        let (mu, sigma) = robust_location_scale(&values).unwrap_or((0.0, 1.0));
        let zero_rate = if total_weight > 0.0 {
            zero_weight / total_weight
        } else {
            0.1
        };

        ZeroAdjustedStudentTEta {
            mu: MuLink::initial_eta_from_theta(mu),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
            zero_probability: ZeroProbabilityLink::initial_eta_from_theta(probability_floor(
                zero_rate,
            )),
        }
    }
}

impl<MuLink, SigmaLink, ZeroProbabilityLink> HasCdf
    for ZeroAdjustedStudentT<MuLink, SigmaLink, ZeroProbabilityLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn cdf(&self, y: f64, theta: &Self::Theta) -> f64 {
        if !y.is_finite() || !Self::valid_theta(*theta) {
            return f64::NAN;
        }

        let component_cdf = self.component.cdf(y, &theta.component());
        let component_probability = 1.0 - theta.zero_probability;
        if y < 0.0 {
            component_probability * component_cdf
        } else {
            component_probability.mul_add(component_cdf, theta.zero_probability)
        }
        .clamp(0.0, 1.0)
    }
}

impl<MuLink, SigmaLink, ZeroProbabilityLink> HasQuantile
    for ZeroAdjustedStudentT<MuLink, SigmaLink, ZeroProbabilityLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn quantile(&self, probability: f64, theta: &Self::Theta) -> f64 {
        if !(0.0..=1.0).contains(&probability) || !Self::valid_theta(*theta) {
            return f64::NAN;
        }

        let component_theta = theta.component();
        let component_probability = 1.0 - theta.zero_probability;
        let below_atom = component_probability * self.component.cdf(0.0, &component_theta);
        let above_atom = below_atom + theta.zero_probability;
        if probability < below_atom {
            self.component
                .quantile(probability / component_probability, &component_theta)
        } else if probability <= above_atom {
            0.0
        } else {
            self.component.quantile(
                (probability - theta.zero_probability) / component_probability,
                &component_theta,
            )
        }
    }
}

impl<MuLink, SigmaLink, ZeroProbabilityLink> HasCrps
    for ZeroAdjustedStudentT<MuLink, SigmaLink, ZeroProbabilityLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn crps(&self, y: f64, theta: &Self::Theta) -> f64 {
        if !y.is_finite() || !Self::valid_theta(*theta) {
            return f64::NAN;
        }

        let component_theta = theta.component();
        crate::crps::zero_inflated_crps(
            y,
            theta.zero_probability,
            self.component.crps(y, &component_theta),
            self.component.crps(0.0, &component_theta),
        )
    }
}

impl<MuLink, SigmaLink, ZeroProbabilityLink> ScalarObservationDomain
    for ZeroAdjustedStudentT<MuLink, SigmaLink, ZeroProbabilityLink>
{
    #[inline]
    fn observation_in_domain(&self, observation: f64) -> bool {
        observation.is_finite()
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink, SigmaLink, ZeroProbabilityLink> TrySimulate<Rng>
    for ZeroAdjustedStudentT<MuLink, SigmaLink, ZeroProbabilityLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    type Sample = f64;

    fn try_sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Result<f64, SimulationError> {
        if !Self::valid_theta(*theta) {
            return Err(SimulationError::InvalidParameters(
                "zero-adjusted Student-t theta",
            ));
        }
        if crate::simulation::open_unit(rng) <= theta.zero_probability {
            return Ok(0.0);
        }

        self.component.try_sample(rng, &theta.component())
    }
}

/// Predictors for zero-adjusted Student-t on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZeroAdjustedStudentTEta {
    /// Location predictor of the continuous component.
    pub mu: f64,
    /// Scale predictor of the continuous component.
    pub sigma: f64,
    /// Zero-mass probability predictor.
    pub zero_probability: f64,
}

impl ParameterParts<3> for ZeroAdjustedStudentTEta {
    #[inline]
    fn from_array(values: [f64; 3]) -> Self {
        Self {
            mu: values[0],
            sigma: values[1],
            zero_probability: values[2],
        }
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mu,
            1 => self.sigma,
            2 => self.zero_probability,
            _ => unreachable!("zero-adjusted student-t eta only has indices 0 through 2"),
        }
    }
}

/// Natural-scale parameters for zero-adjusted Student-t.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZeroAdjustedStudentTTheta {
    /// Location of the continuous component.
    pub mu: f64,
    /// Positive scale of the continuous component.
    pub sigma: f64,
    /// Zero-mass probability in `(0, 1)`.
    pub zero_probability: f64,
}

impl ZeroAdjustedStudentTTheta {
    /// Returns the natural parameters of the continuous Student-t component.
    #[must_use]
    #[inline]
    pub const fn component(self) -> StudentTTheta {
        StudentTTheta {
            mu: self.mu,
            sigma: self.sigma,
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    #[cfg(feature = "rand")]
    use gamlss_core::TrySimulate;
    use gamlss_core::{CompilableFamily, Family, HasCdf, HasCrps, HasQuantile};

    use super::{ZeroAdjustedStudentT, ZeroAdjustedStudentTEta, ZeroAdjustedStudentTTheta};
    use crate::{
        crps::zero_inflated_crps, test_support::assert_gradient_matches_finite_difference,
    };

    type FamilyUnderTest = ZeroAdjustedStudentT;

    #[test]
    fn gradient_matches_finite_difference_at_and_away_from_zero() {
        let family = FamilyUnderTest::try_new(5.0).unwrap();
        for observation in [0.0, -0.75, 0.75] {
            assert_gradient_matches_finite_difference::<_, 3>(
                &family,
                observation,
                [0.2, -0.3, 0.4],
            );
        }
    }

    #[test]
    fn likelihood_factorizes_into_zero_mass_and_continuous_component() {
        let family = FamilyUnderTest::try_new(5.0).unwrap();
        let eta = ZeroAdjustedStudentTEta {
            mu: 0.2,
            sigma: -0.3,
            zero_probability: 0.4,
        };
        let theta = family.theta(&eta, &mut ());
        let component_nll = family.component().nll(0.75, &theta.component(), &mut ());

        assert_relative_eq!(
            family.nll(0.0, &theta, &mut ()),
            -theta.zero_probability.ln(),
            epsilon = 1.0e-15,
        );
        assert_relative_eq!(
            family.nll(0.75, &theta, &mut ()),
            component_nll - (-theta.zero_probability).ln_1p(),
            epsilon = 1.0e-15,
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn cdf_and_quantile_preserve_the_jump_at_zero() {
        let family = FamilyUnderTest::try_new(5.0).unwrap();
        let theta = ZeroAdjustedStudentTTheta {
            mu: 0.2,
            sigma: 0.8,
            zero_probability: 0.3,
        };
        let component_probability = 1.0 - theta.zero_probability;
        let below_atom = component_probability * family.component().cdf(0.0, &theta.component());
        let above_atom = below_atom + theta.zero_probability;

        assert_relative_eq!(family.cdf(0.0, &theta), above_atom, epsilon = 2.0e-15,);
        for probability in [
            below_atom,
            f64::midpoint(below_atom, above_atom),
            above_atom,
        ] {
            assert_eq!(family.quantile(probability, &theta), 0.0);
        }
        for probability in [0.05, 0.95] {
            let quantile = family.quantile(probability, &theta);
            assert_relative_eq!(family.cdf(quantile, &theta), probability, epsilon = 2.0e-7);
        }
    }

    #[test]
    fn crps_uses_the_shared_zero_inflated_identity() {
        let family = FamilyUnderTest::try_new(5.0).unwrap();
        let theta = ZeroAdjustedStudentTTheta {
            mu: 0.2,
            sigma: 0.8,
            zero_probability: 0.3,
        };
        let y = -0.75;
        let component = theta.component();
        let expected = zero_inflated_crps(
            y,
            theta.zero_probability,
            family.component().crps(y, &component),
            family.component().crps(0.0, &component),
        );

        assert_relative_eq!(family.crps(y, &theta), expected, epsilon = 1.0e-15);
    }

    #[test]
    fn initialization_uses_only_nonzero_values_for_the_component() {
        let family = FamilyUnderTest::try_new(5.0).unwrap();
        let observations: &[f64] = &[0.0, 0.0, 2.0, 4.0];
        let eta = <FamilyUnderTest as CompilableFamily>::initial_shape(&family, &observations);

        assert_relative_eq!(eta[0], 2.0, epsilon = 1.0e-15);
        assert!(eta[1].is_finite());
        assert_relative_eq!(eta[2], 0.0, epsilon = 1.0e-15);
    }

    #[test]
    fn invalid_component_parameters_are_rejected_even_at_the_atom() {
        let family = FamilyUnderTest::try_new(5.0).unwrap();
        let invalid = ZeroAdjustedStudentTTheta {
            mu: 0.0,
            sigma: 0.0,
            zero_probability: 0.3,
        };

        assert!(family.nll(0.0, &invalid, &mut ()).is_infinite());
        assert!(family.cdf(0.0, &invalid).is_nan());
        assert!(family.quantile(0.5, &invalid).is_nan());
    }

    #[cfg(feature = "rand")]
    #[test]
    fn sampling_returns_finite_values_and_rejects_invalid_theta() {
        use rand::SeedableRng;

        let family = FamilyUnderTest::try_new(5.0).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let theta = ZeroAdjustedStudentTTheta {
            mu: 0.2,
            sigma: 0.8,
            zero_probability: 0.3,
        };
        assert!(family.try_sample(&mut rng, &theta).unwrap().is_finite());
        assert!(
            family
                .try_sample(
                    &mut rng,
                    &ZeroAdjustedStudentTTheta {
                        zero_probability: 1.0,
                        ..theta
                    },
                )
                .is_err(),
        );
    }
}
