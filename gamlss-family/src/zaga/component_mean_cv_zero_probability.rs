use gamlss_core::{
    Family, HasCdf, HasQuantile, InitialEtaFromTheta, Log, Logit, Mu, Nu, ObservationView,
    ParameterParts, ParameterizedFamily, PositiveLink, Sigma, UnitIntervalLink,
};

use crate::initial::{positive_floor, probability_floor, weighted_summary, weighted_values};
use crate::special::{invert_positive_cdf, regularized_gamma_lower};

use super::{Zaga, ZagaTheta};

/// ZAGA distribution with log/log/logit links.
pub type ZagaMeanSigmaZeroProbability = Zaga<Log, Log, Logit>;
/// Explicit alias for the component-mean/CV/zero-probability ZAGA kernel parameterization.
pub type ZagaComponentMeanCvZeroProbability = ZagaMeanSigmaZeroProbability;

/// Predictors for ZAGA on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZagaEta {
    /// Positive mean predictor for the gamma component.
    pub mu: f64,
    /// Positive coefficient-of-variation predictor for the gamma component.
    pub sigma: f64,
    /// Zero-mass probability predictor.
    pub nu: f64,
}

impl ParameterParts<3> for ZagaEta {
    fn from_array(values: [f64; 3]) -> Self {
        Self {
            mu: values[0],
            sigma: values[1],
            nu: values[2],
        }
    }

    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mu,
            1 => self.sigma,
            2 => self.nu,
            _ => unreachable!("zaga eta only has indices 0 through 2"),
        }
    }
}

impl<MuLink, SigmaLink, NuLink> Zaga<MuLink, SigmaLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    #[inline(always)]
    fn theta_from_eta(eta: ZagaEta) -> ZagaTheta {
        ZagaTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
            nu: NuLink::inverse(eta.nu),
        }
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: ZagaEta) -> (f64, ZagaEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (nll, ZagaEta::from_array([f64::NAN; 3]));
        }
        let gradient = Self::gradient_component_theta(y, theta);
        (
            nll,
            ZagaEta {
                mu: gradient.mu * MuLink::derivative_inverse(eta.mu),
                sigma: gradient.sigma * SigmaLink::derivative_inverse(eta.sigma),
                nu: gradient.nu * NuLink::derivative_inverse(eta.nu),
            },
        )
    }
}

impl<MuLink, SigmaLink, NuLink> Family for Zaga<MuLink, SigmaLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    type Eta = ZagaEta;
    type Theta = ZagaTheta;
    type NllGradientEta = ZagaEta;
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

impl<MuLink, SigmaLink, NuLink> ParameterizedFamily<3> for Zaga<MuLink, SigmaLink, NuLink>
where
    MuLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    NuLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
{
    type Params = (Mu, Sigma, Nu);
    type Links = (MuLink, SigmaLink, NuLink);

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
        let mu = positive_floor(summary.map(|s| s.mean).unwrap_or(1.0));
        let sigma = positive_floor(summary.map(|s| s.variance.sqrt() / mu).unwrap_or(1.0));
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

        ZagaEta {
            mu: MuLink::initial_eta_from_theta(mu),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
            nu: NuLink::initial_eta_from_theta(probability_floor(zero_rate)),
        }
    }
}

impl<MuLink, SigmaLink, NuLink> HasCdf for Zaga<MuLink, SigmaLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::cdf_theta(y, theta)
    }
}

impl<MuLink, SigmaLink, NuLink> HasQuantile for Zaga<MuLink, SigmaLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
        if !(0.0..=1.0).contains(&p) || theta.nu <= 0.0 || theta.nu >= 1.0 {
            return f64::NAN;
        }
        if theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || !theta.sigma.is_finite()
        {
            return f64::NAN;
        }
        if p <= theta.nu {
            return 0.0;
        }
        let shape = 1.0 / (theta.sigma * theta.sigma);
        let rate = 1.0 / (theta.sigma * theta.sigma * theta.mu);
        let target = (p - theta.nu) / (1.0 - theta.nu);
        invert_positive_cdf(target, |y| regularized_gamma_lower(shape, rate * y))
    }
}
