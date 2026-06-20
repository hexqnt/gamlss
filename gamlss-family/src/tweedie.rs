use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasQuantile, InitialEtaFromTheta, Log, Logit, Mu, Nu, ObservationView,
    ParameterParts, ParameterizedFamily, PositiveLink, Sigma, UnitIntervalLink,
};

use crate::initial::{positive_floor, weighted_summary, weighted_values};
use crate::numeric::finite_difference_gradient_eta;
use crate::special::{invert_positive_cdf, ln_gamma, log_add_exp, regularized_gamma_lower};

const MAX_SERIES_TERMS: usize = 2_000;
const SERIES_EPSILON: f64 = 1.0e-13;

/// Tweedie distribution with log/log/logit links.
pub type TweedieMeanDispersionPower = Tweedie<Log, Log, Logit>;
/// Tweedie compound Poisson-gamma family for `1 < nu < 2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tweedie<MuLink = Log, SigmaLink = Log, NuLink = Logit> {
    marker: PhantomData<(MuLink, SigmaLink, NuLink)>,
}

impl<MuLink, SigmaLink, NuLink> Tweedie<MuLink, SigmaLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    /// Creates a stateless Tweedie family.
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn theta_from_eta(eta: TweedieEta) -> TweedieTheta {
        TweedieTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
            nu: 1.0 + NuLink::inverse(eta.nu),
        }
    }

    #[inline(always)]
    fn compound(theta: TweedieTheta) -> Option<CompoundParams> {
        if theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || !(1.0..2.0).contains(&theta.nu)
        {
            return None;
        }

        let lambda = theta.mu.powf(2.0 - theta.nu) / (theta.sigma * (2.0 - theta.nu));
        let alpha = (2.0 - theta.nu) / (theta.nu - 1.0);
        let scale = theta.sigma * (theta.nu - 1.0) * theta.mu.powf(theta.nu - 1.0);
        if lambda <= 0.0
            || !lambda.is_finite()
            || alpha <= 0.0
            || !alpha.is_finite()
            || scale <= 0.0
            || !scale.is_finite()
        {
            return None;
        }

        Some(CompoundParams {
            lambda,
            alpha,
            rate: 1.0 / scale,
        })
    }

    #[inline(always)]
    fn nll_theta(y: f64, theta: TweedieTheta) -> f64 {
        if y < 0.0 || !y.is_finite() {
            return f64::INFINITY;
        }
        let Some(params) = Self::compound(theta) else {
            return f64::INFINITY;
        };
        if y == 0.0 {
            return params.lambda;
        }

        let log_density = Self::positive_log_density(y, params);
        if log_density.is_finite() {
            -log_density
        } else {
            f64::INFINITY
        }
    }

    fn positive_log_density(y: f64, params: CompoundParams) -> f64 {
        let mut log_sum = f64::NEG_INFINITY;
        let log_lambda = params.lambda.ln();
        for n in 1..=MAX_SERIES_TERMS {
            let n_f = n as f64;
            let shape = n_f * params.alpha;
            let log_weight = -params.lambda + n_f * log_lambda - ln_gamma(n_f + 1.0);
            let log_gamma = shape * params.rate.ln() - ln_gamma(shape) + (shape - 1.0) * y.ln()
                - params.rate * y;
            let log_term = log_weight + log_gamma;
            let next = log_add_exp(log_sum, log_term);
            if n > 5 && (next - log_sum).abs() <= SERIES_EPSILON {
                return next;
            }
            log_sum = next;
        }

        log_sum
    }

    fn cdf_theta(y: f64, theta: TweedieTheta) -> f64 {
        if y < 0.0 {
            return 0.0;
        }
        if !y.is_finite() {
            return f64::NAN;
        }
        let Some(params) = Self::compound(theta) else {
            return f64::NAN;
        };
        let p0 = (-params.lambda).exp();
        if y == 0.0 {
            return p0;
        }

        let mut cdf = p0;
        let log_lambda = params.lambda.ln();
        for n in 1..=MAX_SERIES_TERMS {
            let n_f = n as f64;
            let shape = n_f * params.alpha;
            let log_weight = -params.lambda + n_f * log_lambda - ln_gamma(n_f + 1.0);
            let term = log_weight.exp() * regularized_gamma_lower(shape, params.rate * y);
            cdf += term;
            if n > 5 && term.abs() <= SERIES_EPSILON * cdf.abs().max(1.0) {
                break;
            }
        }

        cdf.clamp(0.0, 1.0)
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: TweedieEta) -> (f64, TweedieEta) {
        let nll = Self::nll_theta(y, Self::theta_from_eta(eta));
        if !nll.is_finite() {
            return (nll, TweedieEta::from_array([f64::NAN; 3]));
        }

        let gradient = finite_difference_gradient_eta::<_, TweedieEta, 3>(eta, |probe| {
            Self::nll_theta(y, Self::theta_from_eta(probe))
        });
        (nll, TweedieEta::from_array(gradient))
    }
}

impl<MuLink, SigmaLink, NuLink> Default for Tweedie<MuLink, SigmaLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<MuLink, SigmaLink, NuLink> Family for Tweedie<MuLink, SigmaLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    type Eta = TweedieEta;
    type Theta = TweedieTheta;
    type NllGradientEta = TweedieEta;
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

impl<MuLink, SigmaLink, NuLink> ParameterizedFamily<3> for Tweedie<MuLink, SigmaLink, NuLink>
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
        let Some(summary) = weighted_summary(&values) else {
            return TweedieEta::from_array([0.0, 0.0, 0.0]);
        };
        let mu = positive_floor(summary.mean);
        let sigma = positive_floor(summary.variance / mu.powf(1.5));

        TweedieEta {
            mu: MuLink::initial_eta_from_theta(mu),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
            nu: NuLink::initial_eta_from_theta(0.5),
        }
    }
}

impl<MuLink, SigmaLink, NuLink> HasCdf for Tweedie<MuLink, SigmaLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::cdf_theta(y, theta)
    }
}

impl<MuLink, SigmaLink, NuLink> HasQuantile for Tweedie<MuLink, SigmaLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
        let Some(params) = Self::compound(theta) else {
            return f64::NAN;
        };
        if !(0.0..=1.0).contains(&p) {
            return f64::NAN;
        }
        if p <= (-params.lambda).exp() {
            return 0.0;
        }
        if p == 1.0 {
            return f64::INFINITY;
        }

        invert_positive_cdf(p, |y| Self::cdf_theta(y, theta))
    }
}

#[derive(Debug, Clone, Copy)]
struct CompoundParams {
    lambda: f64,
    alpha: f64,
    rate: f64,
}

/// Predictors for Tweedie on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TweedieEta {
    /// Mean predictor.
    pub mu: f64,
    /// Dispersion predictor.
    pub sigma: f64,
    /// Power predictor mapped into `(1, 2)`.
    pub nu: f64,
}

impl ParameterParts<3> for TweedieEta {
    #[inline(always)]
    fn from_array(values: [f64; 3]) -> Self {
        Self {
            mu: values[0],
            sigma: values[1],
            nu: values[2],
        }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mu,
            1 => self.sigma,
            2 => self.nu,
            _ => unreachable!("tweedie eta only has indices 0 through 2"),
        }
    }
}

/// Natural-scale Tweedie parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TweedieTheta {
    /// Positive mean parameter.
    pub mu: f64,
    /// Positive dispersion parameter.
    pub sigma: f64,
    /// Power parameter in `(1, 2)`.
    pub nu: f64,
}
