use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasQuantile, InitialEtaFromTheta, Log, Logit, ObservationView, ParameterParts,
    ParameterizedFamily, PositiveLink, Size, TotalMean, UnitIntervalLink, ZeroProbability,
};

use crate::domain::{is_positive_finite, is_strict_probability};
use crate::initial::{
    LARGE_SHAPE, positive_floor, probability_floor, weighted_summary, weighted_values,
};
use crate::special::{discrete_quantile, is_nonnegative_integer};

use super::{MAX_CDF_TERMS, Zinb, ZinbTheta};

/// ZINB distribution parameterized by total mean, NB size, and zero probability.
pub type ZinbTotalMeanSizeZeroProbability = ZinbTotalMeanSize<Log, Log, Logit>;

/// Predictors for ZINB total-mean/size/zero-probability on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZinbTotalMeanSizeZeroProbabilityEta {
    /// Total mean predictor.
    pub total_mean: f64,
    /// Negative-binomial size predictor.
    pub size: f64,
    /// Zero-inflation probability predictor.
    pub zero_probability: f64,
}

impl ParameterParts<3> for ZinbTotalMeanSizeZeroProbabilityEta {
    #[inline(always)]
    fn from_array(values: [f64; 3]) -> Self {
        Self {
            total_mean: values[0],
            size: values[1],
            zero_probability: values[2],
        }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.total_mean,
            1 => self.size,
            2 => self.zero_probability,
            _ => unreachable!("ZINB total-mean/size eta only has indices 0 through 2"),
        }
    }
}

/// Natural-scale ZINB total-mean/size/zero-probability parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZinbTotalMeanSizeZeroProbabilityTheta {
    /// Positive unconditional mean.
    pub total_mean: f64,
    /// Positive negative-binomial size.
    pub size: f64,
    /// Zero-inflation probability in `(0, 1)`.
    pub zero_probability: f64,
}

impl ZinbTotalMeanSizeZeroProbabilityTheta {
    #[inline(always)]
    fn component(self) -> ZinbTheta {
        ZinbTheta {
            mu: self.total_mean / (1.0 - self.zero_probability),
            shape: self.size,
            nu: self.zero_probability,
        }
    }
}

/// ZINB total-mean/size/zero-probability implementation carrier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZinbTotalMeanSize<MeanLink = Log, SizeLink = Log, ZeroProbabilityLink = Logit> {
    marker: PhantomData<(MeanLink, SizeLink, ZeroProbabilityLink)>,
}

impl<MeanLink, SizeLink, ZeroProbabilityLink>
    ZinbTotalMeanSize<MeanLink, SizeLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    SizeLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    /// Creates a stateless ZINB total-mean/size family.
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn theta_from_eta(
        eta: ZinbTotalMeanSizeZeroProbabilityEta,
    ) -> ZinbTotalMeanSizeZeroProbabilityTheta {
        ZinbTotalMeanSizeZeroProbabilityTheta {
            total_mean: MeanLink::inverse(eta.total_mean),
            size: SizeLink::inverse(eta.size),
            zero_probability: ZeroProbabilityLink::inverse(eta.zero_probability),
        }
    }
}

impl<MeanLink, SizeLink, ZeroProbabilityLink> Default
    for ZinbTotalMeanSize<MeanLink, SizeLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    SizeLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<MeanLink, SizeLink, ZeroProbabilityLink> Family
    for ZinbTotalMeanSize<MeanLink, SizeLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    SizeLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    type Eta = ZinbTotalMeanSizeZeroProbabilityEta;
    type Theta = ZinbTotalMeanSizeZeroProbabilityTheta;
    type NllGradientEta = ZinbTotalMeanSizeZeroProbabilityEta;
    type Observation<'obs> = f64;

    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        Zinb::<Log, Log, Logit>::nll_theta(y, theta.component())
    }

    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        self.nll(y, Self::theta_from_eta(eta))
    }

    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        let theta = Self::theta_from_eta(eta);
        let component = theta.component();
        let nll = Zinb::<Log, Log, Logit>::nll_theta(y, component);
        if !nll.is_finite() {
            return (
                nll,
                ZinbTotalMeanSizeZeroProbabilityEta::from_array([f64::NAN; 3]),
            );
        }

        let component_gradient = Zinb::<Log, Log, Logit>::gradient_component_theta(y, component);
        let one_minus_zero = 1.0 - theta.zero_probability;
        let d_total_mean = component_gradient.mu / one_minus_zero;
        let d_zero_probability = component_gradient.mu * theta.total_mean
            / (one_minus_zero * one_minus_zero)
            + component_gradient.nu;
        (
            nll,
            ZinbTotalMeanSizeZeroProbabilityEta {
                total_mean: d_total_mean * MeanLink::derivative_inverse(eta.total_mean),
                size: component_gradient.shape * SizeLink::derivative_inverse(eta.size),
                zero_probability: d_zero_probability
                    * ZeroProbabilityLink::derivative_inverse(eta.zero_probability),
            },
        )
    }
}

impl<MeanLink, SizeLink, ZeroProbabilityLink> ParameterizedFamily<3>
    for ZinbTotalMeanSize<MeanLink, SizeLink, ZeroProbabilityLink>
where
    MeanLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    SizeLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    ZeroProbabilityLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
{
    type Params = (TotalMean, Size, ZeroProbability);
    type Links = (MeanLink, SizeLink, ZeroProbabilityLink);

    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| is_nonnegative_integer(y).then_some(y));
        let Some(summary) = weighted_summary(&values) else {
            return ZinbTotalMeanSizeZeroProbabilityEta::from_array([0.0, 0.0, 0.0]);
        };
        let total_mean = positive_floor(summary.mean);
        let size = if summary.variance <= total_mean {
            LARGE_SHAPE
        } else {
            positive_floor(total_mean * total_mean / (summary.variance - total_mean))
        };
        let zero_weight = values
            .iter()
            .filter(|(y, _)| *y == 0.0)
            .map(|(_, w)| *w)
            .sum::<f64>();
        let total_weight = values.iter().map(|(_, w)| *w).sum::<f64>();
        let zero_probability = if total_weight > 0.0 {
            zero_weight / total_weight
        } else {
            0.1
        };

        ZinbTotalMeanSizeZeroProbabilityEta {
            total_mean: MeanLink::initial_eta_from_theta(total_mean),
            size: SizeLink::initial_eta_from_theta(size),
            zero_probability: ZeroProbabilityLink::initial_eta_from_theta(probability_floor(
                zero_probability,
            )),
        }
    }
}

impl<MeanLink, SizeLink, ZeroProbabilityLink> HasCdf
    for ZinbTotalMeanSize<MeanLink, SizeLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    SizeLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
        Zinb::<Log, Log, Logit>::cdf_theta(y, theta.component())
    }
}

impl<MeanLink, SizeLink, ZeroProbabilityLink> HasQuantile
    for ZinbTotalMeanSize<MeanLink, SizeLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    SizeLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
        let theta = theta.component();
        if !is_positive_finite(theta.mu)
            || !is_positive_finite(theta.shape)
            || !is_strict_probability(theta.nu)
        {
            return f64::NAN;
        }

        discrete_quantile(p, MAX_CDF_TERMS, |count| {
            Zinb::<Log, Log, Logit>::cdf_theta(count as f64, theta)
        })
    }
}
