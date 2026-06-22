use gamlss_core::{
    Family, HasCdf, HasQuantile, InitialEtaFromTheta, Log, Logit, ObservationView, ParameterParts,
    ParameterizedFamily, PositiveLink, TotalMean, UnitIntervalLink, ZeroProbability,
};

use gamlss_special::{discrete_quantile, is_nonnegative_integer};

use crate::domain::{is_positive_finite, is_strict_probability};
use crate::initial::{positive_floor, probability_floor, weighted_mean, weighted_values};

use super::{MAX_CDF_TERMS, Zip, ZipEta, ZipTheta};

/// ZIP distribution parameterized by total mean and zero-inflation probability.
pub type ZipTotalMeanZeroProbability = Zip<TotalMeanZeroProbability, Log, Logit>;

/// ZIP total-mean/zero-probability parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TotalMeanZeroProbability;

/// Natural-scale ZIP total-mean/zero-probability parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZipTotalMeanZeroProbabilityTheta {
    /// Positive unconditional mean.
    pub total_mean: f64,
    /// Zero-inflation probability in `(0, 1)`.
    pub zero_probability: f64,
}

impl ZipTotalMeanZeroProbabilityTheta {
    #[inline(always)]
    fn component(self) -> ZipTheta {
        ZipTheta {
            mu: self.total_mean / (1.0 - self.zero_probability),
            sigma: self.zero_probability,
        }
    }
}

impl<MeanLink, ZeroProbabilityLink> Zip<TotalMeanZeroProbability, MeanLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    #[inline(always)]
    fn theta_from_eta(eta: ZipEta) -> ZipTotalMeanZeroProbabilityTheta {
        ZipTotalMeanZeroProbabilityTheta {
            total_mean: MeanLink::inverse(eta.mu),
            zero_probability: ZeroProbabilityLink::inverse(eta.sigma),
        }
    }
}

impl<MeanLink, ZeroProbabilityLink> Family
    for Zip<TotalMeanZeroProbability, MeanLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    type Eta = ZipEta;
    type Theta = ZipTotalMeanZeroProbabilityTheta;
    type NllGradientEta = ZipEta;
    type Observation<'obs> = f64;

    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::nll_theta(y, theta.component())
    }

    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        Self::nll_theta(y, Self::theta_from_eta(eta).component())
    }

    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        let theta = Self::theta_from_eta(eta);
        let component = theta.component();
        let nll = Self::nll_theta(y, component);
        if !nll.is_finite() {
            return (nll, ZipEta::from_array([f64::NAN; 2]));
        }
        let component_gradient = Self::gradient_component_theta(y, component);
        let one_minus_zero = 1.0 - theta.zero_probability;
        let d_total_mean = component_gradient.mu / one_minus_zero;
        let d_zero_probability = component_gradient.mu * theta.total_mean
            / (one_minus_zero * one_minus_zero)
            + component_gradient.sigma;
        (
            nll,
            ZipEta {
                mu: d_total_mean * MeanLink::derivative_inverse(eta.mu),
                sigma: d_zero_probability * ZeroProbabilityLink::derivative_inverse(eta.sigma),
            },
        )
    }
}

impl<MeanLink, ZeroProbabilityLink> ParameterizedFamily<2>
    for Zip<TotalMeanZeroProbability, MeanLink, ZeroProbabilityLink>
where
    MeanLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    ZeroProbabilityLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
{
    type Params = (TotalMean, ZeroProbability);
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
            sigma: ZeroProbabilityLink::initial_eta_from_theta(probability_floor(zero_rate)),
        }
    }
}

impl<MeanLink, ZeroProbabilityLink> HasCdf
    for Zip<TotalMeanZeroProbability, MeanLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::cdf_theta(y, theta.component())
    }
}

impl<MeanLink, ZeroProbabilityLink> HasQuantile
    for Zip<TotalMeanZeroProbability, MeanLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
        let theta = theta.component();
        if !is_positive_finite(theta.mu) || !is_strict_probability(theta.sigma) {
            return f64::NAN;
        }

        discrete_quantile(p, MAX_CDF_TERMS, |count| {
            Self::cdf_theta(count as f64, theta)
        })
    }
}
