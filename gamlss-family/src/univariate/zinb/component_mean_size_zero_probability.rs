use gamlss_core::{
    ComponentMean, Family, HasCdf, HasQuantile, InitialEtaFromObservations, InitialEtaFromTheta,
    Log, Logit, ObservationView, ParameterParts, PositiveLink, Size, UnitIntervalLink,
    ZeroProbability,
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

use gamlss_special::{discrete_quantile, is_nonnegative_integer};

use crate::domain::{is_positive_finite, is_strict_probability};
use crate::initial::{
    LARGE_SHAPE, positive_floor, probability_floor, weighted_summary, weighted_values,
};

use super::{MAX_CDF_TERMS, Zinb, ZinbComponentMeanSizeZeroProbabilityTheta};

/// ZINB distribution parameterized by component mean $\mu$, size $r$, and structural-zero probability $\pi$.
///
/// The default links give $\mu=\exp(\eta_\mu)$, $r=\exp(\eta_r)$, and $\pi=\operatorname{logit}^{-1}(\eta_\pi)$.
#[allow(clippy::doc_markdown)]
pub type ZinbComponentMeanSizeZeroProbability = Zinb<Log, Log, Logit>;

/// Predictors for component-mean/size/zero-probability ZINB on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZinbComponentMeanSizeZeroProbabilityEta {
    /// Negative-binomial mean predictor.
    pub component_mean: f64,
    /// Negative-binomial size predictor.
    pub size: f64,
    /// Zero-inflation probability predictor.
    pub zero_probability: f64,
}

impl ParameterParts<3> for ZinbComponentMeanSizeZeroProbabilityEta {
    fn from_array(values: [f64; 3]) -> Self {
        Self {
            component_mean: values[0],
            size: values[1],
            zero_probability: values[2],
        }
    }

    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.component_mean,
            1 => self.size,
            2 => self.zero_probability,
            _ => unreachable!("zinb eta only has indices 0 through 2"),
        }
    }
}

impl<ComponentMeanLink, SizeLink, ZeroProbabilityLink>
    Zinb<ComponentMeanLink, SizeLink, ZeroProbabilityLink>
where
    ComponentMeanLink: PositiveLink<f64>,
    SizeLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    #[inline]
    fn theta_from_eta(
        eta: ZinbComponentMeanSizeZeroProbabilityEta,
    ) -> ZinbComponentMeanSizeZeroProbabilityTheta {
        ZinbComponentMeanSizeZeroProbabilityTheta {
            component_mean: ComponentMeanLink::inverse(eta.component_mean),
            size: SizeLink::inverse(eta.size),
            zero_probability: ZeroProbabilityLink::inverse(eta.zero_probability),
        }
    }

    #[inline]
    fn nll_and_gradient_eta_values(
        y: f64,
        eta: ZinbComponentMeanSizeZeroProbabilityEta,
    ) -> (f64, ZinbComponentMeanSizeZeroProbabilityEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (
                nll,
                ZinbComponentMeanSizeZeroProbabilityEta::from_array([f64::NAN; 3]),
            );
        }
        let gradient = Self::gradient_component_theta(y, theta);
        (
            nll,
            ZinbComponentMeanSizeZeroProbabilityEta {
                component_mean: gradient.component_mean
                    * ComponentMeanLink::derivative_inverse(eta.component_mean),
                size: gradient.size * SizeLink::derivative_inverse(eta.size),
                zero_probability: gradient.zero_probability
                    * ZeroProbabilityLink::derivative_inverse(eta.zero_probability),
            },
        )
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<ComponentMeanLink, SizeLink, ZeroProbabilityLink> for Zinb<ComponentMeanLink, SizeLink, ZeroProbabilityLink>;
    parameters = (ComponentMean, Size, ZeroProbability);
    arity = 3;
);

impl<ComponentMeanLink, SizeLink, ZeroProbabilityLink> Family
    for Zinb<ComponentMeanLink, SizeLink, ZeroProbabilityLink>
where
    ComponentMeanLink: PositiveLink<f64>,
    SizeLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    type Eta = ZinbComponentMeanSizeZeroProbabilityEta;
    type Theta = ZinbComponentMeanSizeZeroProbabilityTheta;
    type GradientEta = ZinbComponentMeanSizeZeroProbabilityEta;
    type Observation<'obs> = f64;
    type Workspace = ();
    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        Self::nll_theta(y, *theta)
    }

    fn nll_eta(&self, y: f64, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> f64 {
        Self::nll_theta(y, Self::theta_from_eta(*eta))
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

impl<ComponentMeanLink, SizeLink, ZeroProbabilityLink> InitialEtaFromObservations<3>
    for Zinb<ComponentMeanLink, SizeLink, ZeroProbabilityLink>
where
    ComponentMeanLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    SizeLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    ZeroProbabilityLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| is_nonnegative_integer(y).then_some(y));
        let Some(summary) = weighted_summary(&values) else {
            return ZinbComponentMeanSizeZeroProbabilityEta::from_array([0.0, 0.0, 0.0]);
        };
        let mu = positive_floor(summary.mean);
        let shape = if summary.variance <= mu {
            LARGE_SHAPE
        } else {
            positive_floor(mu * mu / (summary.variance - mu))
        };
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

        ZinbComponentMeanSizeZeroProbabilityEta {
            component_mean: ComponentMeanLink::initial_eta_from_theta(mu),
            size: SizeLink::initial_eta_from_theta(shape),
            zero_probability: ZeroProbabilityLink::initial_eta_from_theta(probability_floor(
                (zero_rate - (shape / (shape + mu)).powf(shape)).max(0.05),
            )),
        }
    }
}

impl<ComponentMeanLink, SizeLink, ZeroProbabilityLink> HasCdf
    for Zinb<ComponentMeanLink, SizeLink, ZeroProbabilityLink>
where
    ComponentMeanLink: PositiveLink<f64>,
    SizeLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        Self::cdf_theta(y, *theta)
    }
}

impl<ComponentMeanLink, SizeLink, ZeroProbabilityLink> HasQuantile
    for Zinb<ComponentMeanLink, SizeLink, ZeroProbabilityLink>
where
    ComponentMeanLink: PositiveLink<f64>,
    SizeLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        if !is_positive_finite(theta.component_mean)
            || !is_positive_finite(theta.size)
            || !is_strict_probability(theta.zero_probability)
        {
            return f64::NAN;
        }

        #[allow(clippy::cast_precision_loss)]
        discrete_quantile(p, MAX_CDF_TERMS, |count| {
            Self::cdf_theta(count as f64, *theta)
        })
    }
}

#[cfg(feature = "rand")]
impl<Rng, ComponentMeanLink, SizeLink, ZeroProbabilityLink> TrySimulate<Rng>
    for Zinb<ComponentMeanLink, SizeLink, ZeroProbabilityLink>
where
    Rng: rand::Rng,
    ComponentMeanLink: PositiveLink<f64>,
    SizeLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    type Sample = f64;

    fn try_sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Result<f64, SimulationError> {
        Self::try_sample_component_theta(rng, *theta)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "rand")]
    use gamlss_core::TrySimulate;

    #[cfg(feature = "rand")]
    use super::{ZinbComponentMeanSizeZeroProbability, ZinbComponentMeanSizeZeroProbabilityTheta};

    #[cfg(feature = "rand")]
    #[test]
    fn zinb_sampling_returns_counts_and_errors_for_invalid_theta() {
        use rand::SeedableRng;

        let family = ZinbComponentMeanSizeZeroProbability::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family
            .try_sample(
                &mut rng,
                &ZinbComponentMeanSizeZeroProbabilityTheta {
                    component_mean: 2.0,
                    size: 1.5,
                    zero_probability: 0.3,
                },
            )
            .unwrap();
        assert!(sample >= 0.0 && sample.fract() == 0.0);
        assert!(
            family
                .try_sample(
                    &mut rng,
                    &ZinbComponentMeanSizeZeroProbabilityTheta {
                        component_mean: 2.0,
                        size: 0.0,
                        zero_probability: 0.3,
                    }
                )
                .is_err()
        );
    }
}
