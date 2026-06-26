use gamlss_core::{
    Family, InitialEtaFromTheta, Log, ObservationView, ParameterParts, ParameterizedFamily,
    PositiveLink, Rate,
};

use crate::initial::{positive_floor, weighted_mean, weighted_values};

use super::{Exponential, ExponentialMeanTheta};

/// Exponential distribution parameterized by rate.
pub type ExponentialRate = Exponential<RateParam, Log>;
/// Exponential rate parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RateParam;

/// Predictor for exponential rate on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExponentialRateEta {
    /// Rate predictor.
    pub rate: f64,
}

impl ParameterParts<1> for ExponentialRateEta {
    #[inline]
    fn from_array(values: [f64; 1]) -> Self {
        Self { rate: values[0] }
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.rate,
            _ => unreachable!("exponential rate eta only has index 0"),
        }
    }
}

/// Natural-scale exponential rate parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExponentialRateTheta {
    /// Positive rate.
    pub rate: f64,
}

impl From<ExponentialMeanTheta> for ExponentialRateTheta {
    #[inline]
    fn from(theta: ExponentialMeanTheta) -> Self {
        Self {
            rate: 1.0 / theta.mean,
        }
    }
}
impl<Link> Exponential<RateParam, Link>
where
    Link: PositiveLink<f64>,
{
    #[inline]
    fn theta_from_eta(eta: ExponentialRateEta) -> ExponentialRateTheta {
        ExponentialRateTheta {
            rate: Link::inverse(eta.rate),
        }
    }

    #[inline]
    fn nll_and_gradient_eta_values(y: f64, eta: ExponentialRateEta) -> (f64, ExponentialRateEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_rate(y, theta);
        if !nll.is_finite() {
            return (nll, ExponentialRateEta { rate: f64::NAN });
        }

        let d_rate = y - 1.0 / theta.rate;
        (
            nll,
            ExponentialRateEta {
                rate: d_rate * Link::derivative_inverse(eta.rate),
            },
        )
    }
}

impl<Link> Family for Exponential<RateParam, Link>
where
    Link: PositiveLink<f64>,
{
    type Eta = ExponentialRateEta;
    type Theta = ExponentialRateTheta;
    type NllGradientEta = ExponentialRateEta;
    type Observation<'obs> = f64;

    #[inline]
    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline]
    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::nll_rate(y, theta)
    }

    #[inline]
    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        Self::nll_rate(y, Self::theta_from_eta(eta))
    }

    #[inline]
    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        Self::nll_and_gradient_eta_values(y, eta)
    }
}

impl<Link> ParameterizedFamily<1> for Exponential<RateParam, Link>
where
    Link: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Params = (Rate,);
    type Links = (Link,);

    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values =
            weighted_values::<Self, _, _>(obs, |y| (y.is_finite() && y >= 0.0).then_some(y));
        let Some(mean) = weighted_mean(&values) else {
            return ExponentialRateEta::from_array([0.0]);
        };
        ExponentialRateEta {
            rate: Link::initial_eta_from_theta(1.0 / positive_floor(mean)),
        }
    }
}
