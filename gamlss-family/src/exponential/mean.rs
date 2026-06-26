use gamlss_core::{
    Family, InitialEtaFromTheta, Log, Mean, ObservationView, ParameterParts, ParameterizedFamily,
    PositiveLink,
};

use crate::initial::{positive_floor, weighted_mean, weighted_values};

use super::{Exponential, ExponentialRateTheta};

/// Exponential distribution parameterized by mean.
pub type ExponentialMean = Exponential<MeanParam, Log>;
/// Exponential mean parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeanParam;

/// Predictor for exponential mean on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExponentialMeanEta {
    /// Mean predictor.
    pub mean: f64,
}

impl ParameterParts<1> for ExponentialMeanEta {
    #[inline]
    fn from_array(values: [f64; 1]) -> Self {
        Self { mean: values[0] }
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mean,
            _ => unreachable!("exponential mean eta only has index 0"),
        }
    }
}

/// Natural-scale exponential mean parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExponentialMeanTheta {
    /// Positive mean.
    pub mean: f64,
}

impl ExponentialMeanTheta {
    #[inline]
    pub(super) fn rate(self) -> ExponentialRateTheta {
        ExponentialRateTheta {
            rate: 1.0 / self.mean,
        }
    }
}

impl<Link> Exponential<MeanParam, Link>
where
    Link: PositiveLink<f64>,
{
    #[inline]
    fn theta_from_eta(eta: ExponentialMeanEta) -> ExponentialMeanTheta {
        ExponentialMeanTheta {
            mean: Link::inverse(eta.mean),
        }
    }

    #[inline]
    fn nll_and_gradient_eta_values(y: f64, eta: ExponentialMeanEta) -> (f64, ExponentialMeanEta) {
        let theta = Self::theta_from_eta(eta);
        let rate = theta.rate();
        let nll = Self::nll_rate(y, rate);
        if !nll.is_finite() {
            return (nll, ExponentialMeanEta { mean: f64::NAN });
        }

        let d_rate = y - 1.0 / rate.rate;
        let d_mean = d_rate * (-1.0 / (theta.mean * theta.mean));
        (
            nll,
            ExponentialMeanEta {
                mean: d_mean * Link::derivative_inverse(eta.mean),
            },
        )
    }
}

impl<Link> Family for Exponential<MeanParam, Link>
where
    Link: PositiveLink<f64>,
{
    type Eta = ExponentialMeanEta;
    type Theta = ExponentialMeanTheta;
    type NllGradientEta = ExponentialMeanEta;
    type Observation<'obs> = f64;

    #[inline]
    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline]
    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::nll_rate(y, theta.rate())
    }

    #[inline]
    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        Self::nll_rate(y, Self::theta_from_eta(eta).rate())
    }

    #[inline]
    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        Self::nll_and_gradient_eta_values(y, eta)
    }
}

impl<Link> ParameterizedFamily<1> for Exponential<MeanParam, Link>
where
    Link: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Params = (Mean,);
    type Links = (Link,);

    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values =
            weighted_values::<Self, _, _>(obs, |y| (y.is_finite() && y >= 0.0).then_some(y));
        let Some(mean) = weighted_mean(&values) else {
            return ExponentialMeanEta::from_array([0.0]);
        };
        ExponentialMeanEta {
            mean: Link::initial_eta_from_theta(positive_floor(mean)),
        }
    }
}
