use gamlss_core::{
    Family, HasCdf, HasQuantile, InitialEtaFromTheta, Log, Logit, Mu, Nu, ObservationView,
    ParameterParts, ParameterizedFamily, PositiveLink, Shape, UnitIntervalLink,
};

use gamlss_special::{discrete_quantile, is_nonnegative_integer};

use crate::domain::{is_positive_finite, is_strict_probability};
use crate::initial::{
    LARGE_SHAPE, positive_floor, probability_floor, weighted_summary, weighted_values,
};

use super::{MAX_CDF_TERMS, Zinb, ZinbTheta};

/// ZINB distribution with log/log/logit links.
pub type ZinbMeanSizeZeroProbability = Zinb<Log, Log, Logit>;
/// Explicit alias for the component-mean/size/zero-probability ZINB kernel parameterization.
pub type ZinbComponentMeanSizeZeroProbability = ZinbMeanSizeZeroProbability;

/// Predictors for ZINB on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZinbEta {
    /// Negative-binomial mean predictor.
    pub mu: f64,
    /// Negative-binomial shape predictor.
    pub shape: f64,
    /// Zero-inflation probability predictor.
    pub nu: f64,
}

impl ParameterParts<3> for ZinbEta {
    fn from_array(values: [f64; 3]) -> Self {
        Self {
            mu: values[0],
            shape: values[1],
            nu: values[2],
        }
    }

    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mu,
            1 => self.shape,
            2 => self.nu,
            _ => unreachable!("zinb eta only has indices 0 through 2"),
        }
    }
}

impl<MuLink, ShapeLink, NuLink> Zinb<MuLink, ShapeLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    #[inline]
    fn theta_from_eta(eta: ZinbEta) -> ZinbTheta {
        ZinbTheta {
            mu: MuLink::inverse(eta.mu),
            shape: ShapeLink::inverse(eta.shape),
            nu: NuLink::inverse(eta.nu),
        }
    }

    #[inline]
    fn nll_and_gradient_eta_values(y: f64, eta: ZinbEta) -> (f64, ZinbEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (nll, ZinbEta::from_array([f64::NAN; 3]));
        }
        let gradient = Self::gradient_component_theta(y, theta);
        (
            nll,
            ZinbEta {
                mu: gradient.mu * MuLink::derivative_inverse(eta.mu),
                shape: gradient.shape * ShapeLink::derivative_inverse(eta.shape),
                nu: gradient.nu * NuLink::derivative_inverse(eta.nu),
            },
        )
    }
}

impl<MuLink, ShapeLink, NuLink> Family for Zinb<MuLink, ShapeLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    type Eta = ZinbEta;
    type Theta = ZinbTheta;
    type NllGradientEta = ZinbEta;
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

impl<MuLink, ShapeLink, NuLink> ParameterizedFamily<3> for Zinb<MuLink, ShapeLink, NuLink>
where
    MuLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    ShapeLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    NuLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
{
    type Params = (Mu, Shape, Nu);
    type Links = (MuLink, ShapeLink, NuLink);

    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| is_nonnegative_integer(y).then_some(y));
        let Some(summary) = weighted_summary(&values) else {
            return ZinbEta::from_array([0.0, 0.0, 0.0]);
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

        ZinbEta {
            mu: MuLink::initial_eta_from_theta(mu),
            shape: ShapeLink::initial_eta_from_theta(shape),
            nu: NuLink::initial_eta_from_theta(probability_floor(
                (zero_rate - (shape / (shape + mu)).powf(shape)).max(0.05),
            )),
        }
    }
}

impl<MuLink, ShapeLink, NuLink> HasCdf for Zinb<MuLink, ShapeLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: Self::Theta) -> f64 {
        Self::cdf_theta(y, theta)
    }
}

impl<MuLink, ShapeLink, NuLink> HasQuantile for Zinb<MuLink, ShapeLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
        if !is_positive_finite(theta.mu)
            || !is_positive_finite(theta.shape)
            || !is_strict_probability(theta.nu)
        {
            return f64::NAN;
        }

        #[allow(clippy::cast_precision_loss)]
        discrete_quantile(p, MAX_CDF_TERMS, |count| {
            Self::cdf_theta(count as f64, theta)
        })
    }
}
