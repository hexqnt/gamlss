use std::marker::PhantomData;

use gamlss_core::{
    Cv, Dispersion, Family, HasCdf, HasQuantile, InitialEtaFromTheta, Log, Logit, Mu,
    ObservationView, ParameterParts, ParameterizedFamily, PositiveLink, Power, UnitIntervalLink,
};

use gamlss_special::{invert_positive_cdf, ln_gamma, log_add_exp, regularized_gamma_lower};

use crate::initial::{positive_floor, weighted_summary, weighted_values};
use crate::numeric::finite_difference_gradient_eta;

const MAX_SERIES_TERMS: usize = 2_000;
const SERIES_EPSILON: f64 = 1.0e-13;

/// Tweedie distribution parameterized by mean, dispersion, and power.
///
/// Its NLL gradient currently uses a finite-difference fallback and should be
/// treated as a training slow path until an analytic gradient is added.
pub type TweedieMeanDispersionPower = Tweedie<Log, Log, Logit>;
/// Tweedie distribution parameterized by mean, CV, and power.
///
/// Its NLL gradient currently uses a finite-difference fallback and should be
/// treated as a training slow path until an analytic gradient is added.
pub type TweedieMeanCvPower = TweedieCv<Log, Log, Logit>;
/// Tweedie compound Poisson-gamma family for `1 < power < 2`.
///
/// Its NLL gradient currently uses a finite-difference fallback and should be
/// treated as a training slow path until an analytic gradient is added.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tweedie<MeanLink = Log, DispersionLink = Log, PowerLink = Logit> {
    marker: PhantomData<(MeanLink, DispersionLink, PowerLink)>,
}

impl<MeanLink, DispersionLink, PowerLink> Tweedie<MeanLink, DispersionLink, PowerLink>
where
    MeanLink: PositiveLink<f64>,
    DispersionLink: PositiveLink<f64>,
    PowerLink: UnitIntervalLink<f64>,
{
    /// Creates a stateless Tweedie family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(eta: TweedieEta) -> TweedieTheta {
        TweedieTheta {
            mean: MeanLink::inverse(eta.mean),
            dispersion: DispersionLink::inverse(eta.dispersion),
            power: 1.0 + PowerLink::inverse(eta.power),
        }
    }

    #[inline]
    fn compound(theta: TweedieTheta) -> Option<CompoundParams> {
        if theta.mean <= 0.0
            || !theta.mean.is_finite()
            || theta.dispersion <= 0.0
            || !theta.dispersion.is_finite()
            || !(1.0..2.0).contains(&theta.power)
        {
            return None;
        }

        let lambda = theta.mean.powf(2.0 - theta.power) / (theta.dispersion * (2.0 - theta.power));
        let alpha = (2.0 - theta.power) / (theta.power - 1.0);
        let scale = theta.dispersion * (theta.power - 1.0) * theta.mean.powf(theta.power - 1.0);
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

    #[inline]
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

    #[inline]
    fn quantile_theta(p: f64, theta: TweedieTheta) -> f64 {
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

    #[inline]
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

impl<MeanLink, DispersionLink, PowerLink> Default for Tweedie<MeanLink, DispersionLink, PowerLink>
where
    MeanLink: PositiveLink<f64>,
    DispersionLink: PositiveLink<f64>,
    PowerLink: UnitIntervalLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<MeanLink, DispersionLink, PowerLink> Family for Tweedie<MeanLink, DispersionLink, PowerLink>
where
    MeanLink: PositiveLink<f64>,
    DispersionLink: PositiveLink<f64>,
    PowerLink: UnitIntervalLink<f64>,
{
    type Eta = TweedieEta;
    type Theta = TweedieTheta;
    type NllGradientEta = TweedieEta;
    type Observation<'obs> = f64;

    #[inline]
    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline]
    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::nll_theta(y, theta)
    }

    #[inline]
    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        Self::nll_theta(y, Self::theta_from_eta(eta))
    }

    #[inline]
    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        Self::nll_and_gradient_eta_values(y, eta)
    }
}

impl<MeanLink, DispersionLink, PowerLink> ParameterizedFamily<3>
    for Tweedie<MeanLink, DispersionLink, PowerLink>
where
    MeanLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    DispersionLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    PowerLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
{
    type Params = (Mu, Dispersion, Power);
    type Links = (MeanLink, DispersionLink, PowerLink);

    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values =
            weighted_values::<Self, _, _>(obs, |y| (y.is_finite() && y >= 0.0).then_some(y));
        let Some(summary) = weighted_summary(&values) else {
            return TweedieEta::from_array([0.0, 0.0, 0.0]);
        };
        let mean = positive_floor(summary.mean);
        let dispersion = positive_floor(summary.variance / mean.powf(1.5));

        TweedieEta {
            mean: MeanLink::initial_eta_from_theta(mean),
            dispersion: DispersionLink::initial_eta_from_theta(dispersion),
            power: PowerLink::initial_eta_from_theta(0.5),
        }
    }
}

impl<MeanLink, DispersionLink, PowerLink> HasCdf for Tweedie<MeanLink, DispersionLink, PowerLink>
where
    MeanLink: PositiveLink<f64>,
    DispersionLink: PositiveLink<f64>,
    PowerLink: UnitIntervalLink<f64>,
{
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::cdf_theta(y, theta)
    }
}

impl<MeanLink, DispersionLink, PowerLink> HasQuantile
    for Tweedie<MeanLink, DispersionLink, PowerLink>
where
    MeanLink: PositiveLink<f64>,
    DispersionLink: PositiveLink<f64>,
    PowerLink: UnitIntervalLink<f64>,
{
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
        Self::quantile_theta(p, theta)
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
    pub mean: f64,
    /// Dispersion predictor.
    pub dispersion: f64,
    /// Power predictor mapped into `(1, 2)`.
    pub power: f64,
}

impl ParameterParts<3> for TweedieEta {
    #[inline]
    fn from_array(values: [f64; 3]) -> Self {
        Self {
            mean: values[0],
            dispersion: values[1],
            power: values[2],
        }
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mean,
            1 => self.dispersion,
            2 => self.power,
            _ => unreachable!("tweedie eta only has indices 0 through 2"),
        }
    }
}

/// Natural-scale Tweedie parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TweedieTheta {
    /// Positive mean parameter.
    pub mean: f64,
    /// Positive dispersion parameter.
    pub dispersion: f64,
    /// Power parameter in `(1, 2)`.
    pub power: f64,
}

impl From<TweedieMeanCvPowerTheta> for TweedieTheta {
    #[inline]
    fn from(theta: TweedieMeanCvPowerTheta) -> Self {
        Self {
            mean: theta.mean,
            dispersion: theta.dispersion(),
            power: theta.power,
        }
    }
}

/// Tweedie compound Poisson-gamma family parameterized by mean, CV, and power.
///
/// Its NLL gradient currently uses a finite-difference fallback and should be
/// treated as a training slow path until an analytic gradient is added.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TweedieCv<MeanLink = Log, CvLink = Log, PowerLink = Logit> {
    marker: PhantomData<(MeanLink, CvLink, PowerLink)>,
}

impl<MeanLink, CvLink, PowerLink> TweedieCv<MeanLink, CvLink, PowerLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
    PowerLink: UnitIntervalLink<f64>,
{
    /// Creates a stateless Tweedie mean/CV/power family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(eta: TweedieMeanCvPowerEta) -> TweedieMeanCvPowerTheta {
        TweedieMeanCvPowerTheta {
            mean: MeanLink::inverse(eta.mean),
            cv: CvLink::inverse(eta.cv),
            power: 1.0 + PowerLink::inverse(eta.power),
        }
    }

    #[inline]
    fn nll_and_gradient_eta_values(
        y: f64,
        eta: TweedieMeanCvPowerEta,
    ) -> (f64, TweedieMeanCvPowerEta) {
        let nll = Tweedie::<Log, Log, Logit>::nll_theta(y, Self::theta_from_eta(eta).into());
        if !nll.is_finite() {
            return (nll, TweedieMeanCvPowerEta::from_array([f64::NAN; 3]));
        }

        let gradient =
            finite_difference_gradient_eta::<_, TweedieMeanCvPowerEta, 3>(eta, |probe| {
                Tweedie::<Log, Log, Logit>::nll_theta(y, Self::theta_from_eta(probe).into())
            });
        (nll, TweedieMeanCvPowerEta::from_array(gradient))
    }
}

impl<MeanLink, CvLink, PowerLink> Default for TweedieCv<MeanLink, CvLink, PowerLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
    PowerLink: UnitIntervalLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<MeanLink, CvLink, PowerLink> Family for TweedieCv<MeanLink, CvLink, PowerLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
    PowerLink: UnitIntervalLink<f64>,
{
    type Eta = TweedieMeanCvPowerEta;
    type Theta = TweedieMeanCvPowerTheta;
    type NllGradientEta = TweedieMeanCvPowerEta;
    type Observation<'obs> = f64;

    #[inline]
    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline]
    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        Tweedie::<Log, Log, Logit>::nll_theta(y, theta.into())
    }

    #[inline]
    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        Tweedie::<Log, Log, Logit>::nll_theta(y, Self::theta_from_eta(eta).into())
    }

    #[inline]
    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        Self::nll_and_gradient_eta_values(y, eta)
    }
}

impl<MeanLink, CvLink, PowerLink> ParameterizedFamily<3> for TweedieCv<MeanLink, CvLink, PowerLink>
where
    MeanLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    CvLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    PowerLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
{
    type Params = (Mu, Cv, Power);
    type Links = (MeanLink, CvLink, PowerLink);

    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values =
            weighted_values::<Self, _, _>(obs, |y| (y.is_finite() && y >= 0.0).then_some(y));
        let Some(summary) = weighted_summary(&values) else {
            return TweedieMeanCvPowerEta::from_array([0.0, 0.0, 0.0]);
        };
        let mean = positive_floor(summary.mean);
        let cv = positive_floor(summary.variance.sqrt() / mean);

        TweedieMeanCvPowerEta {
            mean: MeanLink::initial_eta_from_theta(mean),
            cv: CvLink::initial_eta_from_theta(cv),
            power: PowerLink::initial_eta_from_theta(0.5),
        }
    }
}

impl<MeanLink, CvLink, PowerLink> HasCdf for TweedieCv<MeanLink, CvLink, PowerLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
    PowerLink: UnitIntervalLink<f64>,
{
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
        Tweedie::<Log, Log, Logit>::cdf_theta(y, theta.into())
    }
}

impl<MeanLink, CvLink, PowerLink> HasQuantile for TweedieCv<MeanLink, CvLink, PowerLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
    PowerLink: UnitIntervalLink<f64>,
{
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
        Tweedie::<Log, Log, Logit>::quantile_theta(p, theta.into())
    }
}

/// Predictors for Tweedie mean/CV/power on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TweedieMeanCvPowerEta {
    /// Mean predictor.
    pub mean: f64,
    /// Coefficient-of-variation predictor.
    pub cv: f64,
    /// Power predictor mapped into `(1, 2)`.
    pub power: f64,
}

impl ParameterParts<3> for TweedieMeanCvPowerEta {
    #[inline]
    fn from_array(values: [f64; 3]) -> Self {
        Self {
            mean: values[0],
            cv: values[1],
            power: values[2],
        }
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mean,
            1 => self.cv,
            2 => self.power,
            _ => unreachable!("tweedie mean/CV/power eta only has indices 0 through 2"),
        }
    }
}

/// Natural-scale Tweedie mean/CV/power parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TweedieMeanCvPowerTheta {
    /// Positive mean parameter.
    pub mean: f64,
    /// Positive coefficient of variation.
    pub cv: f64,
    /// Power parameter in `(1, 2)`.
    pub power: f64,
}

impl TweedieMeanCvPowerTheta {
    #[inline]
    fn dispersion(self) -> f64 {
        self.cv * self.cv * self.mean.powf(2.0 - self.power)
    }
}
