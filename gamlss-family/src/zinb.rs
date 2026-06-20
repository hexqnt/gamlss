use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasQuantile, InitialEtaFromTheta, Log, Logit, Mu, Nu, ObservationView,
    ParameterParts, ParameterizedFamily, PositiveLink, Shape, UnitIntervalLink,
};

use crate::domain::{is_positive_finite, is_strict_probability};
use crate::initial::{
    LARGE_SHAPE, positive_floor, probability_floor, weighted_summary, weighted_values,
};
use crate::numeric::finite_difference_gradient_eta;
use crate::special::{
    discrete_quantile, included_count, is_nonnegative_integer, ln_gamma, log_add_exp,
};

const MAX_CDF_TERMS: u64 = 1_000_000;

/// ZINB distribution with log/log/logit links.
pub type ZinbMeanSizeZeroProbability = Zinb<Log, Log, Logit>;
/// Zero-inflated negative binomial family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Zinb<MuLink = Log, ShapeLink = Log, NuLink = Logit> {
    marker: PhantomData<(MuLink, ShapeLink, NuLink)>,
}

impl<MuLink, ShapeLink, NuLink> Zinb<MuLink, ShapeLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    /// Creates a stateless ZINB family.
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn theta_from_eta(eta: ZinbEta) -> ZinbTheta {
        ZinbTheta {
            mu: MuLink::inverse(eta.mu),
            shape: ShapeLink::inverse(eta.shape),
            nu: NuLink::inverse(eta.nu),
        }
    }

    #[inline(always)]
    fn nb_log_pmf(y: f64, mu: f64, shape: f64) -> f64 {
        ln_gamma(y + shape) - ln_gamma(shape) - ln_gamma(y + 1.0)
            + shape * (shape / (shape + mu)).ln()
            + y * (mu / (shape + mu)).ln()
    }

    #[inline(always)]
    fn nll_theta(y: f64, theta: ZinbTheta) -> f64 {
        if !is_nonnegative_integer(y)
            || theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
            || theta.nu <= 0.0
            || theta.nu >= 1.0
            || !theta.nu.is_finite()
        {
            return f64::INFINITY;
        }
        let log_nb = Self::nb_log_pmf(y, theta.mu, theta.shape);
        if y == 0.0 {
            -log_add_exp(theta.nu.ln(), (1.0 - theta.nu).ln() + log_nb)
        } else {
            -((1.0 - theta.nu).ln() + log_nb)
        }
    }

    fn nb_cdf(y: f64, mu: f64, shape: f64) -> f64 {
        if y < 0.0 {
            return 0.0;
        }
        let Some(max_count) = included_count(y, MAX_CDF_TERMS) else {
            return f64::NAN;
        };
        let success_probability = shape / (shape + mu);
        let failure_probability = mu / (shape + mu);
        let mut term = (shape * success_probability.ln()).exp();
        let mut sum = term;
        for count in 1..=max_count {
            let previous = (count - 1) as f64;
            term *= ((previous + shape) / count as f64) * failure_probability;
            sum += term;
            if term <= f64::EPSILON * sum {
                break;
            }
        }
        sum.clamp(0.0, 1.0)
    }

    fn cdf_theta(y: f64, theta: ZinbTheta) -> f64 {
        if !y.is_finite()
            || theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
            || theta.nu <= 0.0
            || theta.nu >= 1.0
            || !theta.nu.is_finite()
        {
            return f64::NAN;
        }
        if y < 0.0 {
            return 0.0;
        }

        (theta.nu + (1.0 - theta.nu) * Self::nb_cdf(y, theta.mu, theta.shape)).clamp(0.0, 1.0)
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: ZinbEta) -> (f64, ZinbEta) {
        let nll = Self::nll_theta(y, Self::theta_from_eta(eta));
        if !nll.is_finite() {
            return (nll, ZinbEta::from_array([f64::NAN; 3]));
        }
        let gradient = finite_difference_gradient_eta::<_, ZinbEta, 3>(eta, |probe| {
            Self::nll_theta(y, Self::theta_from_eta(probe))
        });
        (nll, ZinbEta::from_array(gradient))
    }
}

impl<MuLink, ShapeLink, NuLink> Default for Zinb<MuLink, ShapeLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    fn default() -> Self {
        Self::new()
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
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
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

        discrete_quantile(p, MAX_CDF_TERMS, |count| {
            Self::cdf_theta(count as f64, theta)
        })
    }
}

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

/// Natural-scale ZINB parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZinbTheta {
    /// Positive negative-binomial mean.
    pub mu: f64,
    /// Positive negative-binomial shape.
    pub shape: f64,
    /// Zero-inflation probability in `(0, 1)`.
    pub nu: f64,
}
