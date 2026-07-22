use gamlss_core::{
    Family, HasCdf, HasCrps, HasQuantile, InitialEtaFromObservations, InitialEtaFromTheta, Log,
    Logit, ObservationView, ParameterParts, PositiveLink, TotalMean, UnitIntervalLink,
    ZeroProbability,
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

use gamlss_special::is_nonnegative_integer;

use crate::initial::{positive_floor, probability_floor, weighted_mean, weighted_values};

use super::{Zip, ZipComponentMeanZeroProbabilityTheta, ZipKernel};

/// ZIP distribution parameterized by unconditional mean $m>0$ and structural-zero probability $\pi\in(0,1)$.
///
/// The Poisson component mean is derived as
///
/// $$
/// \lambda=\frac{m}{1-\pi},
/// \qquad
/// \mathbb{E}(Y)=m.
/// $$
///
/// The default links are $m=\exp(\eta_m)$ and $\pi=\operatorname{logit}^{-1}(\eta_\pi)$.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../../doc-assets/distributions/zip_total_mean.svg")
)]
#[allow(clippy::doc_markdown)]
pub type ZipTotalMeanZeroProbability = Zip<TotalMeanZeroProbability, Log, Logit>;

/// ZIP total-mean/zero-probability parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TotalMeanZeroProbability;

/// Predictors for total-mean ZIP on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZipTotalMeanZeroProbabilityEta {
    /// Unconditional-mean predictor.
    pub total_mean: f64,
    /// Zero-inflation probability predictor.
    pub zero_probability: f64,
}

impl ParameterParts<2> for ZipTotalMeanZeroProbabilityEta {
    #[inline]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            total_mean: values[0],
            zero_probability: values[1],
        }
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.total_mean,
            1 => self.zero_probability,
            _ => unreachable!("total-mean ZIP eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale ZIP total-mean/zero-probability parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZipTotalMeanZeroProbabilityTheta {
    /// Positive unconditional mean.
    pub total_mean: f64,
    /// Zero-inflation probability in `(0, 1)`.
    pub zero_probability: f64,
}

impl ZipTotalMeanZeroProbabilityTheta {
    #[inline]
    fn component(self) -> ZipComponentMeanZeroProbabilityTheta {
        ZipComponentMeanZeroProbabilityTheta {
            component_mean: self.total_mean / (1.0 - self.zero_probability),
            zero_probability: self.zero_probability,
        }
    }
}

impl<MeanLink, ZeroProbabilityLink> Zip<TotalMeanZeroProbability, MeanLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    #[inline]
    fn theta_from_eta(eta: ZipTotalMeanZeroProbabilityEta) -> ZipTotalMeanZeroProbabilityTheta {
        ZipTotalMeanZeroProbabilityTheta {
            total_mean: MeanLink::inverse(eta.total_mean),
            zero_probability: ZeroProbabilityLink::inverse(eta.zero_probability),
        }
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MeanLink, ZeroProbabilityLink> for Zip<TotalMeanZeroProbability, MeanLink, ZeroProbabilityLink>;
    parameters = (TotalMean, ZeroProbability);
    arity = 2;
);

impl<MeanLink, ZeroProbabilityLink> Family
    for Zip<TotalMeanZeroProbability, MeanLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    type Eta = ZipTotalMeanZeroProbabilityEta;
    type Theta = ZipTotalMeanZeroProbabilityTheta;
    type GradientEta = ZipTotalMeanZeroProbabilityEta;
    type Observation<'obs> = f64;
    type Workspace = ();
    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        ZipKernel::nll_theta(y, theta.component())
    }

    fn nll_eta(&self, y: f64, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> f64 {
        ZipKernel::nll_theta(y, Self::theta_from_eta(*eta).component())
    }

    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        let theta = Self::theta_from_eta(*eta);
        let component = theta.component();
        let nll = ZipKernel::nll_theta(y, component);
        if !nll.is_finite() {
            return (
                nll,
                ZipTotalMeanZeroProbabilityEta::from_array([f64::NAN; 2]),
            );
        }
        let component_gradient = ZipKernel::gradient_component_theta(y, component);
        let one_minus_zero = 1.0 - theta.zero_probability;
        let d_total_mean = component_gradient.component_mean / one_minus_zero;
        let d_zero_probability = component_gradient.component_mean * theta.total_mean
            / (one_minus_zero * one_minus_zero)
            + component_gradient.zero_probability;
        (
            nll,
            ZipTotalMeanZeroProbabilityEta {
                total_mean: d_total_mean * MeanLink::derivative_inverse(eta.total_mean),
                zero_probability: d_zero_probability
                    * ZeroProbabilityLink::derivative_inverse(eta.zero_probability),
            },
        )
    }
}

impl<MeanLink, ZeroProbabilityLink> InitialEtaFromObservations<2>
    for Zip<TotalMeanZeroProbability, MeanLink, ZeroProbabilityLink>
where
    MeanLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    ZeroProbabilityLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| is_nonnegative_integer(y).then_some(y));
        let mean = weighted_mean(&values).unwrap_or(1.0);
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

        ZipTotalMeanZeroProbabilityEta {
            total_mean: MeanLink::initial_eta_from_theta(positive_floor(mean)),
            zero_probability: ZeroProbabilityLink::initial_eta_from_theta(probability_floor(
                zero_rate,
            )),
        }
    }
}

impl<MeanLink, ZeroProbabilityLink> HasCdf
    for Zip<TotalMeanZeroProbability, MeanLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        ZipKernel::cdf_theta(y, theta.component())
    }
}

impl<MeanLink, ZeroProbabilityLink> HasQuantile
    for Zip<TotalMeanZeroProbability, MeanLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        ZipKernel::quantile_theta(p, theta.component())
    }
}

impl<MeanLink, ZeroProbabilityLink> HasCrps
    for Zip<TotalMeanZeroProbability, MeanLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn crps(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        ZipKernel::crps_theta(y, theta.component())
    }
}

#[cfg(feature = "rand")]
impl<Rng, MeanLink, ZeroProbabilityLink> TrySimulate<Rng>
    for Zip<TotalMeanZeroProbability, MeanLink, ZeroProbabilityLink>
where
    Rng: rand::Rng,
    MeanLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    type Sample = f64;

    fn try_sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Result<f64, SimulationError> {
        ZipKernel::try_sample_component_theta(rng, theta.component())
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "rand")]
    use gamlss_core::TrySimulate;

    #[cfg(feature = "rand")]
    use super::{ZipTotalMeanZeroProbability, ZipTotalMeanZeroProbabilityTheta};

    #[cfg(feature = "rand")]
    #[test]
    fn total_mean_zip_sampling_returns_counts_and_errors_for_invalid_theta() {
        use rand::SeedableRng;

        let family = ZipTotalMeanZeroProbability::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family
            .try_sample(
                &mut rng,
                &ZipTotalMeanZeroProbabilityTheta {
                    total_mean: 2.0,
                    zero_probability: 0.3,
                },
            )
            .unwrap();
        assert!(sample >= 0.0 && sample.fract() == 0.0);
        assert!(
            family
                .try_sample(
                    &mut rng,
                    &ZipTotalMeanZeroProbabilityTheta {
                        total_mean: 2.0,
                        zero_probability: 1.0,
                    }
                )
                .is_err()
        );
    }
}
