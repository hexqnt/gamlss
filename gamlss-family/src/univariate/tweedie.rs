use std::marker::PhantomData;

use gamlss_core::{
    Cv, Dispersion, Family, HasCdf, HasCrps, HasQuantile, InitialEtaFromObservations,
    InitialEtaFromTheta, Log, Logit, Mu, ObservationView, ParameterParts, PositiveLink, Power,
    UnitIntervalLink,
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

use gamlss_special::{
    digamma, invert_positive_cdf, ln_gamma, log_add_exp, regularized_gamma_lower,
};

use crate::initial::{positive_floor, weighted_summary, weighted_values};

const MAX_SERIES_TERMS: usize = 2_000;
const SERIES_EPSILON: f64 = 1.0e-13;

/// Tweedie distribution parameterized by mean, dispersion, and power.
///
/// Its NLL gradient is analytic and differentiates the compound Poisson-gamma series.
pub type TweedieMeanDispersionPower = Tweedie<Log, Log, Logit>;
/// Tweedie distribution parameterized by mean, CV, and power.
///
/// Its NLL gradient is analytic and differentiates the compound Poisson-gamma series.
pub type TweedieMeanCvPower = TweedieCv<Log, Log, Logit>;
/// Tweedie compound Poisson-gamma family for `1 < power < 2`.
///
/// Its NLL gradient is analytic and differentiates the compound Poisson-gamma series.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/distributions/tweedie_mean_dispersion_power.svg")
)]
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
}

/// Link- and parameterization-independent Tweedie kernel.
#[derive(Debug, Clone, Copy)]
struct TweedieKernel;

impl TweedieKernel {
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

    #[allow(clippy::suboptimal_flops, clippy::cast_precision_loss)]
    fn positive_log_density(y: f64, params: CompoundParams) -> f64 {
        let mut log_sum = f64::NEG_INFINITY;
        let mut previous_log_term = f64::NEG_INFINITY;
        let mut past_mode = false;
        let log_lambda = params.lambda.ln();
        for n in 1..=MAX_SERIES_TERMS {
            let n_f = n as f64;
            let shape = n_f * params.alpha;
            let log_weight = -params.lambda + n_f * log_lambda - ln_gamma(n_f + 1.0);
            let log_gamma = shape * params.rate.ln() - ln_gamma(shape) + (shape - 1.0) * y.ln()
                - params.rate * y;
            let log_term = log_weight + log_gamma;
            past_mode |= log_term <= previous_log_term;
            let next = log_add_exp(log_sum, log_term);
            if n > 5 && past_mode && (next - log_sum).abs() <= SERIES_EPSILON {
                return next;
            }
            log_sum = next;
            previous_log_term = log_term;
        }

        f64::NAN
    }

    #[allow(clippy::suboptimal_flops, clippy::cast_precision_loss)]
    fn positive_log_density_gradient_theta(
        y: f64,
        theta: TweedieTheta,
        params: CompoundParams,
    ) -> (f64, TweedieGradient) {
        let mut log_sum = f64::NEG_INFINITY;
        let mut gradient = TweedieGradient::ZERO;
        let mut previous_log_term = f64::NEG_INFINITY;
        let mut past_mode = false;
        let log_lambda = params.lambda.ln();
        for n in 1..=MAX_SERIES_TERMS {
            let n_f = n as f64;
            let shape = n_f * params.alpha;
            let log_weight = -params.lambda + n_f * log_lambda - ln_gamma(n_f + 1.0);
            let log_gamma = shape * params.rate.ln() - ln_gamma(shape) + (shape - 1.0) * y.ln()
                - params.rate * y;
            let log_term = log_weight + log_gamma;
            past_mode |= log_term <= previous_log_term;
            let term_gradient = Self::positive_log_term_gradient(n_f, y, theta, params);
            let next = log_add_exp(log_sum, log_term);
            let old_weight = if log_sum.is_finite() {
                (log_sum - next).exp()
            } else {
                0.0
            };
            let term_weight = (log_term - next).exp();
            gradient = gradient.blend_with(term_gradient, old_weight, term_weight);
            if n > 5 && past_mode && (next - log_sum).abs() <= SERIES_EPSILON {
                return (next, gradient);
            }
            log_sum = next;
            previous_log_term = log_term;
        }

        (f64::NAN, TweedieGradient::NAN)
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn positive_log_term_gradient(
        n: f64,
        y: f64,
        theta: TweedieTheta,
        params: CompoundParams,
    ) -> TweedieGradient {
        let mean = theta.mean;
        let dispersion = theta.dispersion;
        let power_minus_one = theta.power - 1.0;
        let two_minus_power = 2.0 - theta.power;
        let log_mean = mean.ln();
        let shape = n * params.alpha;
        let log_rate = params.rate.ln();
        let y_log = y.ln();

        let d_log_lambda_d_mean = two_minus_power / mean;
        let d_lambda_d_mean = params.lambda * d_log_lambda_d_mean;
        let d_log_rate_d_mean = -power_minus_one / mean;
        let d_rate_d_mean = params.rate * d_log_rate_d_mean;

        let d_log_lambda_d_dispersion = -1.0 / dispersion;
        let d_lambda_d_dispersion = params.lambda * d_log_lambda_d_dispersion;
        let d_log_rate_d_dispersion = -1.0 / dispersion;
        let d_rate_d_dispersion = params.rate * d_log_rate_d_dispersion;

        let d_log_lambda_d_power = -log_mean + 1.0 / two_minus_power;
        let d_lambda_d_power = params.lambda * d_log_lambda_d_power;
        let d_alpha_d_power = -1.0 / (power_minus_one * power_minus_one);
        let d_shape_d_power = n * d_alpha_d_power;
        let d_log_rate_d_power = -1.0 / power_minus_one - log_mean;
        let d_rate_d_power = params.rate * d_log_rate_d_power;
        let d_shape_factor = log_rate - digamma(shape) + y_log;

        TweedieGradient {
            mean: -d_lambda_d_mean + n * d_log_lambda_d_mean + shape * d_log_rate_d_mean
                - y * d_rate_d_mean,
            dispersion: -d_lambda_d_dispersion
                + n * d_log_lambda_d_dispersion
                + shape * d_log_rate_d_dispersion
                - y * d_rate_d_dispersion,
            power: -d_lambda_d_power
                + n * d_log_lambda_d_power
                + d_shape_d_power * d_shape_factor
                + shape * d_log_rate_d_power
                - y * d_rate_d_power,
        }
    }

    #[allow(clippy::suboptimal_flops)]
    fn nll_and_gradient_theta(y: f64, theta: TweedieTheta) -> (f64, TweedieGradient) {
        if y < 0.0 || !y.is_finite() {
            return (f64::INFINITY, TweedieGradient::NAN);
        }
        let Some(params) = Self::compound(theta) else {
            return (f64::INFINITY, TweedieGradient::NAN);
        };
        if y == 0.0 {
            let two_minus_power = 2.0 - theta.power;
            let gradient = TweedieGradient {
                mean: params.lambda * two_minus_power / theta.mean,
                dispersion: -params.lambda / theta.dispersion,
                power: params.lambda * (-theta.mean.ln() + 1.0 / two_minus_power),
            };
            return (params.lambda, gradient);
        }

        let (log_density, log_density_gradient) =
            Self::positive_log_density_gradient_theta(y, theta, params);
        if !log_density.is_finite() {
            return (f64::INFINITY, TweedieGradient::NAN);
        }

        (-log_density, -log_density_gradient)
    }

    #[allow(clippy::suboptimal_flops, clippy::cast_precision_loss)]
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
        let log_p0 = -params.lambda;
        if y == 0.0 {
            return log_p0.exp();
        }

        let mut log_cdf = log_p0;
        let mut previous_log_term = log_p0;
        let mut past_mode = false;
        let log_lambda = params.lambda.ln();
        for n in 1..=MAX_SERIES_TERMS {
            let n_f = n as f64;
            let shape = n_f * params.alpha;
            let log_weight = -params.lambda + n_f * log_lambda - ln_gamma(n_f + 1.0);
            let gamma_cdf = regularized_gamma_lower(shape, params.rate * y);
            let log_term = log_weight + gamma_cdf.ln();
            past_mode |= log_term <= previous_log_term;
            let next = log_add_exp(log_cdf, log_term);
            if n > 5 && past_mode && (next - log_cdf).abs() <= SERIES_EPSILON {
                return next.exp().clamp(0.0, 1.0);
            }
            log_cdf = next;
            previous_log_term = log_term;
        }

        f64::NAN
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
        #[allow(clippy::float_cmp)]
        if p == 1.0 {
            return f64::INFINITY;
        }

        invert_positive_cdf(p, |y| Self::cdf_theta(y, theta))
    }

    fn crps_theta(y: f64, theta: TweedieTheta) -> f64 {
        if y < 0.0 || !y.is_finite() || Self::compound(theta).is_none() {
            return f64::NAN;
        }
        let standard_deviation = theta.dispersion.sqrt() * theta.mean.powf(0.5 * theta.power);
        let mapping_scale = if standard_deviation.is_finite() {
            theta.mean.max(standard_deviation)
        } else {
            // The mapping scale only conditions the quadrature. The mean is a
            // valid fallback even when the variance overflows representable f64.
            theta.mean
        };
        crate::crps::integrate_cdf_crps(y, mapping_scale, |x| Self::cdf_theta(x, theta))
    }

    #[cfg(feature = "rand")]
    fn try_sample_theta<Rng>(rng: &mut Rng, theta: TweedieTheta) -> Result<f64, SimulationError>
    where
        Rng: rand::Rng,
    {
        let Some(params) = Self::compound(theta) else {
            return Err(SimulationError::InvalidParameters("Tweedie theta"));
        };

        let poisson = rand_distr::Poisson::new(params.lambda)
            .map_err(|_| SimulationError::BackendRejected("Tweedie Poisson rate"))?;
        let count = rand_distr::Distribution::sample(&poisson, rng);
        if count == 0.0 {
            return Ok(0.0);
        }

        let gamma = rand_distr::Gamma::new(count * params.alpha, 1.0 / params.rate)
            .map_err(|_| SimulationError::BackendRejected("Tweedie gamma"))?;
        let sample = rand_distr::Distribution::sample(&gamma, rng);
        if sample.is_finite() {
            Ok(sample)
        } else {
            Err(SimulationError::NumericalFailure("Tweedie gamma sample"))
        }
    }
}

impl<MeanLink, DispersionLink, PowerLink> Tweedie<MeanLink, DispersionLink, PowerLink>
where
    MeanLink: PositiveLink<f64>,
    DispersionLink: PositiveLink<f64>,
    PowerLink: UnitIntervalLink<f64>,
{
    #[inline]
    fn nll_and_gradient_eta_values(y: f64, eta: TweedieEta) -> (f64, TweedieEta) {
        let (nll, gradient) = TweedieKernel::nll_and_gradient_theta(y, Self::theta_from_eta(eta));
        if !nll.is_finite() {
            return (nll, TweedieEta::from_array([f64::NAN; 3]));
        }

        (
            nll,
            TweedieEta {
                mean: gradient.mean * MeanLink::derivative_inverse(eta.mean),
                dispersion: gradient.dispersion
                    * DispersionLink::derivative_inverse(eta.dispersion),
                power: gradient.power * PowerLink::derivative_inverse(eta.power),
            },
        )
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

gamlss_core::impl_scalar_compilable_family!(
    impl<MeanLink, DispersionLink, PowerLink> for Tweedie<MeanLink, DispersionLink, PowerLink>;
    parameters = (Mu, Dispersion, Power);
    arity = 3;
);

impl<MeanLink, DispersionLink, PowerLink> Family for Tweedie<MeanLink, DispersionLink, PowerLink>
where
    MeanLink: PositiveLink<f64>,
    DispersionLink: PositiveLink<f64>,
    PowerLink: UnitIntervalLink<f64>,
{
    type Eta = TweedieEta;
    type Theta = TweedieTheta;
    type GradientEta = TweedieEta;
    type Observation<'obs> = f64;
    type Workspace = ();

    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    #[inline]
    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        TweedieKernel::nll_theta(y, *theta)
    }

    #[inline]
    fn nll_eta(&self, y: f64, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> f64 {
        TweedieKernel::nll_theta(y, Self::theta_from_eta(*eta))
    }

    #[inline]
    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        Self::nll_and_gradient_eta_values(y, *eta)
    }
}

impl<MeanLink, DispersionLink, PowerLink> InitialEtaFromObservations<3>
    for Tweedie<MeanLink, DispersionLink, PowerLink>
where
    MeanLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    DispersionLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    PowerLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
{
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
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        TweedieKernel::cdf_theta(y, *theta)
    }
}

impl<MeanLink, DispersionLink, PowerLink> HasQuantile
    for Tweedie<MeanLink, DispersionLink, PowerLink>
where
    MeanLink: PositiveLink<f64>,
    DispersionLink: PositiveLink<f64>,
    PowerLink: UnitIntervalLink<f64>,
{
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        TweedieKernel::quantile_theta(p, *theta)
    }
}

impl<MeanLink, DispersionLink, PowerLink> HasCrps for Tweedie<MeanLink, DispersionLink, PowerLink>
where
    MeanLink: PositiveLink<f64>,
    DispersionLink: PositiveLink<f64>,
    PowerLink: UnitIntervalLink<f64>,
{
    fn crps(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        TweedieKernel::crps_theta(y, *theta)
    }
}

#[cfg(feature = "rand")]
impl<Rng, MeanLink, DispersionLink, PowerLink> TrySimulate<Rng>
    for Tweedie<MeanLink, DispersionLink, PowerLink>
where
    Rng: rand::Rng,
    MeanLink: PositiveLink<f64>,
    DispersionLink: PositiveLink<f64>,
    PowerLink: UnitIntervalLink<f64>,
{
    type Sample = f64;

    fn try_sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Result<f64, SimulationError> {
        TweedieKernel::try_sample_theta(rng, *theta)
    }
}

#[derive(Debug, Clone, Copy)]
struct CompoundParams {
    lambda: f64,
    alpha: f64,
    rate: f64,
}

#[derive(Debug, Clone, Copy)]
struct TweedieGradient {
    mean: f64,
    dispersion: f64,
    power: f64,
}

impl TweedieGradient {
    const NAN: Self = Self {
        mean: f64::NAN,
        dispersion: f64::NAN,
        power: f64::NAN,
    };
    const ZERO: Self = Self {
        mean: 0.0,
        dispersion: 0.0,
        power: 0.0,
    };

    #[inline]
    fn blend_with(self, term: Self, self_weight: f64, term_weight: f64) -> Self {
        Self {
            mean: self_weight.mul_add(self.mean, term_weight * term.mean),
            dispersion: self_weight.mul_add(self.dispersion, term_weight * term.dispersion),
            power: self_weight.mul_add(self.power, term_weight * term.power),
        }
    }
}

impl std::ops::Neg for TweedieGradient {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            mean: -self.mean,
            dispersion: -self.dispersion,
            power: -self.power,
        }
    }
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
/// Its NLL gradient is analytic and differentiates the compound Poisson-gamma series.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/distributions/tweedie_mean_cv_power.svg")
)]
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
        let theta = Self::theta_from_eta(eta);
        let dispersion_theta: TweedieTheta = theta.into();
        let (nll, dispersion_gradient) = TweedieKernel::nll_and_gradient_theta(y, dispersion_theta);
        if !nll.is_finite() {
            return (nll, TweedieMeanCvPowerEta::from_array([f64::NAN; 3]));
        }

        let dispersion = dispersion_theta.dispersion;
        let d_mean = dispersion_gradient.mean
            + dispersion_gradient.dispersion * dispersion * (2.0 - theta.power) / theta.mean;
        let d_cv = dispersion_gradient.dispersion * 2.0 * dispersion / theta.cv;
        let d_power = (dispersion_gradient.dispersion * dispersion)
            .mul_add(-theta.mean.ln(), dispersion_gradient.power);

        (
            nll,
            TweedieMeanCvPowerEta {
                mean: d_mean * MeanLink::derivative_inverse(eta.mean),
                cv: d_cv * CvLink::derivative_inverse(eta.cv),
                power: d_power * PowerLink::derivative_inverse(eta.power),
            },
        )
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

gamlss_core::impl_scalar_compilable_family!(
    impl<MeanLink, CvLink, PowerLink> for TweedieCv<MeanLink, CvLink, PowerLink>;
    parameters = (Mu, Cv, Power);
    arity = 3;
);

impl<MeanLink, CvLink, PowerLink> Family for TweedieCv<MeanLink, CvLink, PowerLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
    PowerLink: UnitIntervalLink<f64>,
{
    type Eta = TweedieMeanCvPowerEta;
    type Theta = TweedieMeanCvPowerTheta;
    type GradientEta = TweedieMeanCvPowerEta;
    type Observation<'obs> = f64;
    type Workspace = ();

    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    #[inline]
    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        TweedieKernel::nll_theta(y, (*theta).into())
    }

    #[inline]
    fn nll_eta(&self, y: f64, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> f64 {
        TweedieKernel::nll_theta(y, Self::theta_from_eta(*eta).into())
    }

    #[inline]
    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        Self::nll_and_gradient_eta_values(y, *eta)
    }
}

impl<MeanLink, CvLink, PowerLink> InitialEtaFromObservations<3>
    for TweedieCv<MeanLink, CvLink, PowerLink>
where
    MeanLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    CvLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    PowerLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
{
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
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        TweedieKernel::cdf_theta(y, (*theta).into())
    }
}

impl<MeanLink, CvLink, PowerLink> HasQuantile for TweedieCv<MeanLink, CvLink, PowerLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
    PowerLink: UnitIntervalLink<f64>,
{
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        TweedieKernel::quantile_theta(p, (*theta).into())
    }
}

impl<MeanLink, CvLink, PowerLink> HasCrps for TweedieCv<MeanLink, CvLink, PowerLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
    PowerLink: UnitIntervalLink<f64>,
{
    fn crps(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        TweedieKernel::crps_theta(y, (*theta).into())
    }
}

#[cfg(feature = "rand")]
impl<Rng, MeanLink, CvLink, PowerLink> TrySimulate<Rng> for TweedieCv<MeanLink, CvLink, PowerLink>
where
    Rng: rand::Rng,
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
    PowerLink: UnitIntervalLink<f64>,
{
    type Sample = f64;

    fn try_sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Result<f64, SimulationError> {
        TweedieKernel::try_sample_theta(rng, (*theta).into())
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

#[cfg(test)]
mod tests {
    #[cfg(feature = "rand")]
    use gamlss_core::TrySimulate;
    use gamlss_core::{Family, HasCdf};

    #[cfg(feature = "rand")]
    use super::{TweedieMeanCvPower, TweedieMeanCvPowerTheta};
    use super::{TweedieMeanDispersionPower, TweedieTheta};

    #[test]
    fn tweedie_cdf_does_not_stop_on_underflow_before_the_poisson_mode() {
        let family = TweedieMeanDispersionPower::new();
        let theta = TweedieTheta {
            mean: 1.0,
            dispersion: 0.002,
            power: 1.5,
        };

        let cdf = family.cdf(1.0, &theta);
        assert!(cdf > 0.45 && cdf < 0.55, "large-lambda CDF was {cdf}");
        assert!(family.nll(1.0, &theta, &mut family.workspace()).is_finite());
    }

    #[cfg(feature = "rand")]
    #[test]
    fn tweedie_sampling_returns_nonnegative_values_and_errors_for_invalid_theta() {
        use rand::SeedableRng;

        let family = TweedieMeanDispersionPower::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family
            .try_sample(
                &mut rng,
                &TweedieTheta {
                    mean: 2.0,
                    dispersion: 0.5,
                    power: 1.5,
                },
            )
            .unwrap();
        assert!(sample >= 0.0 && sample.is_finite());
        assert!(
            family
                .try_sample(
                    &mut rng,
                    &TweedieTheta {
                        mean: 2.0,
                        dispersion: 0.5,
                        power: 2.0,
                    }
                )
                .is_err()
        );
    }

    #[cfg(feature = "rand")]
    #[test]
    fn tweedie_cv_sampling_returns_nonnegative_values_and_errors_for_invalid_theta() {
        use rand::SeedableRng;

        let family = TweedieMeanCvPower::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family
            .try_sample(
                &mut rng,
                &TweedieMeanCvPowerTheta {
                    mean: 2.0,
                    cv: 0.7,
                    power: 1.5,
                },
            )
            .unwrap();
        assert!(sample >= 0.0 && sample.is_finite());
        assert!(
            family
                .try_sample(
                    &mut rng,
                    &TweedieMeanCvPowerTheta {
                        mean: 2.0,
                        cv: 0.7,
                        power: 2.0,
                    }
                )
                .is_err()
        );
    }
}
