use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasQuantile, InitialEtaFromTheta, Log, Logit, Mu, Nu, ObservationView,
    ParameterParts, ParameterizedFamily, PositiveLink, Sigma, Tau, UnitIntervalLink,
};

use crate::initial::{
    POSITIVE_FLOOR, VARIANCE_FLOOR, positive_floor, probability_floor, weighted_summary,
    weighted_values,
};
use crate::numeric::finite_difference_gradient_eta;
use crate::special::{invert_bounded_cdf, ln_gamma, regularized_beta};

/// Beta distribution inflated at both zero and one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Beinf<MuLink = Logit, SigmaLink = Logit, NuLink = Log, TauLink = Log> {
    marker: PhantomData<(MuLink, SigmaLink, NuLink, TauLink)>,
}

#[derive(Debug, Clone, Copy)]
struct BeinfParts {
    alpha: f64,
    beta: f64,
    p0: f64,
    p1: f64,
    p_beta: f64,
}

impl<MuLink, SigmaLink, NuLink, TauLink> Beinf<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: UnitIntervalLink<f64>,
    SigmaLink: UnitIntervalLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    /// Creates a stateless BEINF family.
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn theta_from_eta(eta: BeinfEta) -> BeinfTheta {
        BeinfTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
            nu: NuLink::inverse(eta.nu),
            tau: TauLink::inverse(eta.tau),
        }
    }

    #[inline(always)]
    fn parts(theta: BeinfTheta) -> Option<BeinfParts> {
        if theta.mu <= 0.0
            || theta.mu >= 1.0
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || theta.sigma >= 1.0
            || !theta.sigma.is_finite()
            || theta.nu <= 0.0
            || !theta.nu.is_finite()
            || theta.tau <= 0.0
            || !theta.tau.is_finite()
        {
            return None;
        }

        let precision = 1.0 / theta.sigma - 1.0;
        if precision <= 0.0 || !precision.is_finite() {
            return None;
        }
        let denominator = 1.0 + theta.nu + theta.tau;
        Some(BeinfParts {
            alpha: theta.mu * precision,
            beta: (1.0 - theta.mu) * precision,
            p0: theta.nu / denominator,
            p1: theta.tau / denominator,
            p_beta: 1.0 / denominator,
        })
    }

    #[inline(always)]
    fn beta_log_density(y: f64, alpha: f64, beta: f64) -> f64 {
        ln_gamma(alpha + beta) - ln_gamma(alpha) - ln_gamma(beta)
            + (alpha - 1.0) * y.ln()
            + (beta - 1.0) * (1.0 - y).ln()
    }

    #[inline(always)]
    fn nll_theta(y: f64, theta: BeinfTheta) -> f64 {
        if !(0.0..=1.0).contains(&y) || !y.is_finite() {
            return f64::INFINITY;
        }
        let Some(parts) = Self::parts(theta) else {
            return f64::INFINITY;
        };
        if y == 0.0 {
            return -parts.p0.ln();
        }
        if y == 1.0 {
            return -parts.p1.ln();
        }

        -(parts.p_beta.ln() + Self::beta_log_density(y, parts.alpha, parts.beta))
    }

    fn cdf_theta(y: f64, theta: BeinfTheta) -> f64 {
        if !y.is_finite() {
            return f64::NAN;
        }
        let Some(parts) = Self::parts(theta) else {
            return f64::NAN;
        };
        if y < 0.0 {
            return 0.0;
        }
        if y == 0.0 {
            return parts.p0;
        }
        if y < 1.0 {
            return (parts.p0 + parts.p_beta * regularized_beta(parts.alpha, parts.beta, y))
                .clamp(0.0, 1.0);
        }
        1.0
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: BeinfEta) -> (f64, BeinfEta) {
        let nll = Self::nll_theta(y, Self::theta_from_eta(eta));
        if !nll.is_finite() {
            return (nll, BeinfEta::from_array([f64::NAN; 4]));
        }

        let gradient = finite_difference_gradient_eta::<_, BeinfEta, 4>(eta, |probe| {
            Self::nll_theta(y, Self::theta_from_eta(probe))
        });
        (nll, BeinfEta::from_array(gradient))
    }
}

impl<MuLink, SigmaLink, NuLink, TauLink> Default for Beinf<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: UnitIntervalLink<f64>,
    SigmaLink: UnitIntervalLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Predictors for BEINF on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeinfEta {
    /// Beta-component mean predictor.
    pub mu: f64,
    /// Beta-component dispersion predictor in `(0, 1)`.
    pub sigma: f64,
    /// Positive zero-mass odds predictor.
    pub nu: f64,
    /// Positive one-mass odds predictor.
    pub tau: f64,
}

impl ParameterParts<4> for BeinfEta {
    #[inline(always)]
    fn from_array(values: [f64; 4]) -> Self {
        Self {
            mu: values[0],
            sigma: values[1],
            nu: values[2],
            tau: values[3],
        }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mu,
            1 => self.sigma,
            2 => self.nu,
            3 => self.tau,
            _ => unreachable!("beinf eta only has indices 0 through 3"),
        }
    }
}

/// Natural-scale BEINF parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeinfTheta {
    /// Beta-component mean in `(0, 1)`.
    pub mu: f64,
    /// Beta-component dispersion in `(0, 1)`.
    pub sigma: f64,
    /// Positive zero-mass odds relative to the beta component.
    pub nu: f64,
    /// Positive one-mass odds relative to the beta component.
    pub tau: f64,
}

impl<MuLink, SigmaLink, NuLink, TauLink> Family for Beinf<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: UnitIntervalLink<f64>,
    SigmaLink: UnitIntervalLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    type Eta = BeinfEta;
    type Theta = BeinfTheta;
    type NllGradientEta = BeinfEta;
    type Observation<'obs> = f64;

    #[inline(always)]
    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline(always)]
    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::nll_theta(y, theta)
    }

    #[inline(always)]
    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        Self::nll_theta(y, Self::theta_from_eta(eta))
    }

    #[inline(always)]
    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        Self::nll_and_gradient_eta_values(y, eta)
    }
}

impl<MuLink, SigmaLink, NuLink, TauLink> ParameterizedFamily<4>
    for Beinf<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
    NuLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    TauLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Params = (Mu, Sigma, Nu, Tau);
    type Links = (MuLink, SigmaLink, NuLink, TauLink);

    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| {
            (y.is_finite() && (0.0..=1.0).contains(&y)).then_some(y)
        });
        let interior = values
            .iter()
            .copied()
            .filter(|(y, _)| *y > 0.0 && *y < 1.0)
            .collect::<Vec<_>>();
        let summary = weighted_summary(&interior);
        let mu = probability_floor(summary.map(|s| s.mean).unwrap_or(0.5));
        let max_variance = (mu * (1.0 - mu)).max(VARIANCE_FLOOR);
        let variance = summary
            .map(|s| s.variance.clamp(VARIANCE_FLOOR, max_variance * 0.99))
            .unwrap_or(max_variance * 0.5);
        let precision = positive_floor(max_variance / variance - 1.0);
        let sigma = probability_floor(1.0 / (precision + 1.0));

        let zero_weight = values
            .iter()
            .filter(|(y, _)| *y == 0.0)
            .map(|(_, w)| *w)
            .sum::<f64>();
        let one_weight = values
            .iter()
            .filter(|(y, _)| *y == 1.0)
            .map(|(_, w)| *w)
            .sum::<f64>();
        let interior_weight = interior
            .iter()
            .map(|(_, w)| *w)
            .sum::<f64>()
            .max(POSITIVE_FLOOR);

        BeinfEta {
            mu: MuLink::initial_eta_from_theta(mu),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
            nu: NuLink::initial_eta_from_theta(positive_floor(zero_weight / interior_weight)),
            tau: TauLink::initial_eta_from_theta(positive_floor(one_weight / interior_weight)),
        }
    }
}

impl<MuLink, SigmaLink, NuLink, TauLink> HasCdf for Beinf<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: UnitIntervalLink<f64>,
    SigmaLink: UnitIntervalLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::cdf_theta(y, theta)
    }
}

impl<MuLink, SigmaLink, NuLink, TauLink> HasQuantile for Beinf<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: UnitIntervalLink<f64>,
    SigmaLink: UnitIntervalLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
        if !(0.0..=1.0).contains(&p) {
            return f64::NAN;
        }
        let Some(parts) = Self::parts(theta) else {
            return f64::NAN;
        };
        if p <= parts.p0 {
            return 0.0;
        }
        if p >= parts.p0 + parts.p_beta {
            return 1.0;
        }

        let target = (p - parts.p0) / parts.p_beta;
        invert_bounded_cdf(target, 0.0, 1.0, |y| {
            regularized_beta(parts.alpha, parts.beta, y)
        })
    }
}

/// BEINF distribution with logit/logit/log/log links.
pub type DefaultBeinf = Beinf<Logit, Logit, Log, Log>;
