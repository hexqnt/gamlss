use gamlss_core::{
    ComponentMean, Cv, Family, HasCdf, HasCrps, HasQuantile, InitialEtaFromObservations,
    InitialEtaFromTheta, Log, Logit, ObservationView, ParameterParts, PositiveLink,
    UnitIntervalLink, ZeroProbability,
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

use crate::initial::{positive_floor, probability_floor, weighted_summary, weighted_values};

use super::{Zaga, ZagaComponentMeanCvZeroProbabilityTheta, ZagaKernel};

/// ZAGA distribution parameterized by gamma component mean $\mu$, component CV $c$, and zero-mass probability $\pi$.
///
/// The default links give $\mu=\exp(\eta_\mu)$, $c=\exp(\eta_c)$, and $\pi=\operatorname{logit}^{-1}(\eta_\pi)$.
#[allow(clippy::doc_markdown)]
pub type ZagaComponentMeanCvZeroProbability = Zaga<Log, Log, Logit>;

/// Predictors for component-mean/CV/zero-probability ZAGA on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZagaComponentMeanCvZeroProbabilityEta {
    /// Positive mean predictor for the gamma component.
    pub component_mean: f64,
    /// Positive coefficient-of-variation predictor for the gamma component.
    pub cv: f64,
    /// Zero-mass probability predictor.
    pub zero_probability: f64,
}

impl ParameterParts<3> for ZagaComponentMeanCvZeroProbabilityEta {
    fn from_array(values: [f64; 3]) -> Self {
        Self {
            component_mean: values[0],
            cv: values[1],
            zero_probability: values[2],
        }
    }

    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.component_mean,
            1 => self.cv,
            2 => self.zero_probability,
            _ => unreachable!("zaga eta only has indices 0 through 2"),
        }
    }
}

impl<ComponentMeanLink, CvLink, ZeroProbabilityLink>
    Zaga<ComponentMeanLink, CvLink, ZeroProbabilityLink>
where
    ComponentMeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    #[inline]
    fn theta_from_eta(
        eta: ZagaComponentMeanCvZeroProbabilityEta,
    ) -> ZagaComponentMeanCvZeroProbabilityTheta {
        ZagaComponentMeanCvZeroProbabilityTheta {
            component_mean: ComponentMeanLink::inverse(eta.component_mean),
            cv: CvLink::inverse(eta.cv),
            zero_probability: ZeroProbabilityLink::inverse(eta.zero_probability),
        }
    }

    #[inline]
    fn nll_and_gradient_eta_values(
        y: f64,
        eta: ZagaComponentMeanCvZeroProbabilityEta,
    ) -> (f64, ZagaComponentMeanCvZeroProbabilityEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = ZagaKernel::nll_theta(y, theta);
        if !nll.is_finite() {
            return (
                nll,
                ZagaComponentMeanCvZeroProbabilityEta::from_array([f64::NAN; 3]),
            );
        }
        let gradient = ZagaKernel::gradient_component_theta(y, theta);
        (
            nll,
            ZagaComponentMeanCvZeroProbabilityEta {
                component_mean: gradient.component_mean
                    * ComponentMeanLink::derivative_inverse(eta.component_mean),
                cv: gradient.cv * CvLink::derivative_inverse(eta.cv),
                zero_probability: gradient.zero_probability
                    * ZeroProbabilityLink::derivative_inverse(eta.zero_probability),
            },
        )
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<ComponentMeanLink, CvLink, ZeroProbabilityLink> for Zaga<ComponentMeanLink, CvLink, ZeroProbabilityLink>;
    parameters = (ComponentMean, Cv, ZeroProbability);
    arity = 3;
);

impl<ComponentMeanLink, CvLink, ZeroProbabilityLink> Family
    for Zaga<ComponentMeanLink, CvLink, ZeroProbabilityLink>
where
    ComponentMeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    type Eta = ZagaComponentMeanCvZeroProbabilityEta;
    type Theta = ZagaComponentMeanCvZeroProbabilityTheta;
    type GradientEta = ZagaComponentMeanCvZeroProbabilityEta;
    type Observation<'obs> = f64;
    type Workspace = ();
    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        ZagaKernel::nll_theta(y, *theta)
    }

    fn nll_eta(&self, y: f64, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> f64 {
        ZagaKernel::nll_theta(y, Self::theta_from_eta(*eta))
    }

    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        Self::nll_and_gradient_eta_values(y, *eta)
    }
}

impl<ComponentMeanLink, CvLink, ZeroProbabilityLink> InitialEtaFromObservations<3>
    for Zaga<ComponentMeanLink, CvLink, ZeroProbabilityLink>
where
    ComponentMeanLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    CvLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    ZeroProbabilityLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values =
            weighted_values::<Self, _, _>(obs, |y| (y.is_finite() && y >= 0.0).then_some(y));
        let positives = values
            .iter()
            .copied()
            .filter(|(y, _)| *y > 0.0)
            .collect::<Vec<_>>();
        let summary = weighted_summary(&positives);
        let mu = positive_floor(summary.map_or(1.0, |s| s.mean));
        let sigma = positive_floor(summary.map_or(1.0, |s| s.variance.sqrt() / mu));
        let zero_weight = values
            .iter()
            .filter(|(y, _)| *y == 0.0)
            .map(|(_, w)| *w)
            .sum::<f64>();
        let total_weight = values.iter().map(|(_, w)| *w).sum::<f64>();
        let zero_rate = if total_weight > 0.0 {
            zero_weight / total_weight
        } else {
            0.1
        };

        ZagaComponentMeanCvZeroProbabilityEta {
            component_mean: ComponentMeanLink::initial_eta_from_theta(mu),
            cv: CvLink::initial_eta_from_theta(sigma),
            zero_probability: ZeroProbabilityLink::initial_eta_from_theta(probability_floor(
                zero_rate,
            )),
        }
    }
}

impl<ComponentMeanLink, CvLink, ZeroProbabilityLink> HasCdf
    for Zaga<ComponentMeanLink, CvLink, ZeroProbabilityLink>
where
    ComponentMeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        ZagaKernel::cdf_theta(y, *theta)
    }
}

impl<ComponentMeanLink, CvLink, ZeroProbabilityLink> HasQuantile
    for Zaga<ComponentMeanLink, CvLink, ZeroProbabilityLink>
where
    ComponentMeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        ZagaKernel::quantile_theta(p, *theta)
    }
}

impl<ComponentMeanLink, CvLink, ZeroProbabilityLink> HasCrps
    for Zaga<ComponentMeanLink, CvLink, ZeroProbabilityLink>
where
    ComponentMeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn crps(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        ZagaKernel::crps_theta(y, *theta)
    }
}

#[cfg(feature = "rand")]
impl<Rng, ComponentMeanLink, CvLink, ZeroProbabilityLink> TrySimulate<Rng>
    for Zaga<ComponentMeanLink, CvLink, ZeroProbabilityLink>
where
    Rng: rand::Rng,
    ComponentMeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    type Sample = f64;

    fn try_sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Result<f64, SimulationError> {
        ZagaKernel::try_sample_component_theta(rng, *theta)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "rand")]
    use gamlss_core::TrySimulate;

    #[cfg(feature = "rand")]
    use super::{ZagaComponentMeanCvZeroProbability, ZagaComponentMeanCvZeroProbabilityTheta};

    #[cfg(feature = "rand")]
    #[test]
    fn zaga_sampling_returns_nonnegative_values_and_errors_for_invalid_theta() {
        use rand::SeedableRng;

        let family = ZagaComponentMeanCvZeroProbability::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family
            .try_sample(
                &mut rng,
                &ZagaComponentMeanCvZeroProbabilityTheta {
                    component_mean: 1.5,
                    cv: 0.7,
                    zero_probability: 0.2,
                },
            )
            .unwrap();
        assert!(sample >= 0.0 && sample.is_finite());
        assert!(
            family
                .try_sample(
                    &mut rng,
                    &ZagaComponentMeanCvZeroProbabilityTheta {
                        component_mean: 1.5,
                        cv: 0.0,
                        zero_probability: 0.2,
                    }
                )
                .is_err()
        );
    }
}
