use gamlss_core::{
    ComponentMean, Family, HasCdf, HasQuantile, InitialEtaFromTheta, Log, Logit, ObservationView,
    ParameterParts, ParameterizedFamily, PositiveLink, UnitIntervalLink, ZeroProbability,
};

use crate::domain::{is_positive_finite, is_strict_probability};
use crate::initial::{positive_floor, probability_floor, weighted_mean, weighted_values};
use crate::special::{discrete_quantile, is_nonnegative_integer};

use super::{MAX_CDF_TERMS, Zip, ZipEta, ZipTheta};

/// ZIP distribution with log/logit links.
pub type ZipComponentMeanZeroProbability = Zip<ComponentMeanZeroProbability, Log, Logit>;

/// ZIP component-mean/zero-probability parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ComponentMeanZeroProbability;

impl<MeanLink, ZeroProbabilityLink> Zip<ComponentMeanZeroProbability, MeanLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    #[inline(always)]
    fn theta_from_eta(eta: ZipEta) -> ZipTheta {
        ZipTheta {
            mu: MeanLink::inverse(eta.mu),
            sigma: ZeroProbabilityLink::inverse(eta.sigma),
        }
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: ZipEta) -> (f64, ZipEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (nll, ZipEta::from_array([f64::NAN; 2]));
        }
        let gradient = Self::gradient_component_theta(y, theta);
        (
            nll,
            ZipEta {
                mu: gradient.mu * MeanLink::derivative_inverse(eta.mu),
                sigma: gradient.sigma * ZeroProbabilityLink::derivative_inverse(eta.sigma),
            },
        )
    }
}

impl<MeanLink, ZeroProbabilityLink> Family
    for Zip<ComponentMeanZeroProbability, MeanLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    type Eta = ZipEta;
    type Theta = ZipTheta;
    type NllGradientEta = ZipEta;
    type Observation<'obs> = f64;

    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::nll_theta(y, theta)
    }

    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        Self::nll_theta(y, Self::theta_from_eta(eta))
    }

    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        Self::nll_and_gradient_eta_values(y, eta)
    }
}

impl<MeanLink, ZeroProbabilityLink> ParameterizedFamily<2>
    for Zip<ComponentMeanZeroProbability, MeanLink, ZeroProbabilityLink>
where
    MeanLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    ZeroProbabilityLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
{
    type Params = (ComponentMean, ZeroProbability);
    type Links = (MeanLink, ZeroProbabilityLink);

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

        ZipEta {
            mu: MeanLink::initial_eta_from_theta(positive_floor(mean)),
            sigma: ZeroProbabilityLink::initial_eta_from_theta(probability_floor(
                (zero_rate - (-mean).exp()).max(0.05),
            )),
        }
    }
}

impl<MeanLink, ZeroProbabilityLink> HasCdf
    for Zip<ComponentMeanZeroProbability, MeanLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::cdf_theta(y, theta)
    }
}

impl<MeanLink, ZeroProbabilityLink> HasQuantile
    for Zip<ComponentMeanZeroProbability, MeanLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
        if !is_positive_finite(theta.mu) || !is_strict_probability(theta.sigma) {
            return f64::NAN;
        }

        discrete_quantile(p, MAX_CDF_TERMS, |count| {
            Self::cdf_theta(count as f64, theta)
        })
    }
}
