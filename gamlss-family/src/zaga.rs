use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasQuantile, InitialEtaFromTheta, Log, Logit, Mu, Nu, ObservationView,
    ParameterParts, ParameterizedFamily, PositiveLink, Sigma, UnitIntervalLink,
};

use crate::initial::{positive_floor, probability_floor, weighted_summary, weighted_values};
use crate::numeric::finite_difference_gradient_eta;
use crate::special::{invert_positive_cdf, ln_gamma, regularized_gamma_lower};

/// ZAGA distribution with log/log/logit links.
pub type ZagaMeanSigmaZeroProbability = Zaga<Log, Log, Logit>;
/// Zero-adjusted gamma family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Zaga<MuLink = Log, SigmaLink = Log, NuLink = Logit> {
    marker: PhantomData<(MuLink, SigmaLink, NuLink)>,
}

impl<MuLink, SigmaLink, NuLink> Zaga<MuLink, SigmaLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    /// Creates a stateless ZAGA family.
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn theta_from_eta(eta: ZagaEta) -> ZagaTheta {
        ZagaTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
            nu: NuLink::inverse(eta.nu),
        }
    }

    #[inline(always)]
    fn gamma_shape_rate(theta: ZagaTheta) -> (f64, f64) {
        let shape = 1.0 / (theta.sigma * theta.sigma);
        let rate = 1.0 / (theta.sigma * theta.sigma * theta.mu);
        (shape, rate)
    }

    #[inline(always)]
    fn gamma_nll(y: f64, shape: f64, rate: f64) -> f64 {
        ln_gamma(shape) - shape * rate.ln() - (shape - 1.0) * y.ln() + rate * y
    }

    #[inline(always)]
    fn nll_theta(y: f64, theta: ZagaTheta) -> f64 {
        if y < 0.0
            || !y.is_finite()
            || theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || theta.nu <= 0.0
            || theta.nu >= 1.0
            || !theta.nu.is_finite()
        {
            return f64::INFINITY;
        }
        if y == 0.0 {
            return -theta.nu.ln();
        }
        let (shape, rate) = Self::gamma_shape_rate(theta);
        -(1.0 - theta.nu).ln() + Self::gamma_nll(y, shape, rate)
    }

    fn cdf_theta(y: f64, theta: ZagaTheta) -> f64 {
        if !y.is_finite()
            || theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || theta.nu <= 0.0
            || theta.nu >= 1.0
            || !theta.nu.is_finite()
        {
            return f64::NAN;
        }
        if y < 0.0 {
            return 0.0;
        }
        if y == 0.0 {
            return theta.nu;
        }
        let (shape, rate) = Self::gamma_shape_rate(theta);
        (theta.nu + (1.0 - theta.nu) * regularized_gamma_lower(shape, rate * y)).clamp(0.0, 1.0)
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: ZagaEta) -> (f64, ZagaEta) {
        let nll = Self::nll_theta(y, Self::theta_from_eta(eta));
        if !nll.is_finite() {
            return (nll, ZagaEta::from_array([f64::NAN; 3]));
        }
        let gradient = finite_difference_gradient_eta::<_, ZagaEta, 3>(eta, |probe| {
            Self::nll_theta(y, Self::theta_from_eta(probe))
        });
        (nll, ZagaEta::from_array(gradient))
    }
}

impl<MuLink, SigmaLink, NuLink> Default for Zaga<MuLink, SigmaLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    fn default() -> Self {
        Self::new()
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
        let (shape, rate) = Self::gamma_shape_rate(theta);
        let target = (p - theta.nu) / (1.0 - theta.nu);
        invert_positive_cdf(target, |y| regularized_gamma_lower(shape, rate * y))
    }
}

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

/// Natural-scale ZAGA parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZagaTheta {
    /// Positive mean for the gamma component.
    pub mu: f64,
    /// Positive coefficient of variation for the gamma component.
    pub sigma: f64,
    /// Zero-mass probability in `(0, 1)`.
    pub nu: f64,
}
