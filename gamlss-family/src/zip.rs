use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasQuantile, InitialEtaFromTheta, Log, Logit, Mu, ObservationView,
    ParameterParts, ParameterizedFamily, PositiveLink, Sigma, UnitIntervalLink,
};

use crate::domain::{is_positive_finite, is_strict_probability};
use crate::initial::{positive_floor, probability_floor, weighted_mean, weighted_values};
use crate::numeric::finite_difference_gradient_eta;
use crate::special::{
    discrete_quantile, included_count, is_nonnegative_integer, ln_gamma, log_add_exp,
};

const MAX_CDF_TERMS: u64 = 1_000_000;

/// Zero-inflated Poisson family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Zip<MuLink = Log, SigmaLink = Logit> {
    marker: PhantomData<(MuLink, SigmaLink)>,
}

impl<MuLink, SigmaLink> Zip<MuLink, SigmaLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: UnitIntervalLink<f64>,
{
    /// Creates a stateless zero-inflated Poisson family.
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn theta_from_eta(eta: ZipEta) -> ZipTheta {
        ZipTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
        }
    }

    #[inline(always)]
    fn poisson_log_pmf(y: f64, mu: f64) -> f64 {
        -mu + y * mu.ln() - ln_gamma(y + 1.0)
    }

    #[inline(always)]
    fn nll_theta(y: f64, theta: ZipTheta) -> f64 {
        if !is_nonnegative_integer(y)
            || theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || theta.sigma >= 1.0
            || !theta.sigma.is_finite()
        {
            return f64::INFINITY;
        }
        if y == 0.0 {
            -log_add_exp(theta.sigma.ln(), (1.0 - theta.sigma).ln() - theta.mu)
        } else {
            -((1.0 - theta.sigma).ln() + Self::poisson_log_pmf(y, theta.mu))
        }
    }

    fn cdf_theta(y: f64, theta: ZipTheta) -> f64 {
        if !y.is_finite()
            || theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || theta.sigma >= 1.0
            || !theta.sigma.is_finite()
        {
            return f64::NAN;
        }
        if y < 0.0 {
            return 0.0;
        }
        let Some(max_count) = included_count(y, MAX_CDF_TERMS) else {
            return f64::NAN;
        };
        let mut term = (-theta.mu).exp();
        let mut sum = term;
        for count in 1..=max_count {
            term *= theta.mu / count as f64;
            sum += term;
            if term <= f64::EPSILON * sum {
                break;
            }
        }
        (theta.sigma + (1.0 - theta.sigma) * sum).clamp(0.0, 1.0)
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: ZipEta) -> (f64, ZipEta) {
        let nll = Self::nll_theta(y, Self::theta_from_eta(eta));
        if !nll.is_finite() {
            return (nll, ZipEta::from_array([f64::NAN; 2]));
        }
        let gradient = finite_difference_gradient_eta::<_, ZipEta, 2>(eta, |probe| {
            Self::nll_theta(y, Self::theta_from_eta(probe))
        });
        (nll, ZipEta::from_array(gradient))
    }
}

impl<MuLink, SigmaLink> Default for Zip<MuLink, SigmaLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: UnitIntervalLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Predictors for ZIP on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZipEta {
    /// Poisson mean predictor.
    pub mu: f64,
    /// Zero-inflation probability predictor.
    pub sigma: f64,
}

impl ParameterParts<2> for ZipEta {
    #[inline(always)]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            mu: values[0],
            sigma: values[1],
        }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mu,
            1 => self.sigma,
            _ => unreachable!("zip eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale ZIP parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZipTheta {
    /// Positive Poisson mean.
    pub mu: f64,
    /// Zero-inflation probability in `(0, 1)`.
    pub sigma: f64,
}

impl<MuLink, SigmaLink> Family for Zip<MuLink, SigmaLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: UnitIntervalLink<f64>,
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

impl<MuLink, SigmaLink> ParameterizedFamily<2> for Zip<MuLink, SigmaLink>
where
    MuLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
{
    type Params = (Mu, Sigma);
    type Links = (MuLink, SigmaLink);

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
            mu: MuLink::initial_eta_from_theta(positive_floor(mean)),
            sigma: SigmaLink::initial_eta_from_theta(probability_floor(
                (zero_rate - (-mean).exp()).max(0.05),
            )),
        }
    }
}

impl<MuLink, SigmaLink> HasCdf for Zip<MuLink, SigmaLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: UnitIntervalLink<f64>,
{
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::cdf_theta(y, theta)
    }
}

impl<MuLink, SigmaLink> HasQuantile for Zip<MuLink, SigmaLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: UnitIntervalLink<f64>,
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

/// ZIP distribution with log/logit links.
pub type DefaultZip = Zip<Log, Logit>;
