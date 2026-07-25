//! Two-piece skew power-exponential distributions.
//!
//! The symmetric generalized-error, exponential-power and power-exponential names describe the same density family. Prefixing any of them with "skew" does not identify a unique skewing construction: the literature also contains skew-symmetric/selection models and models with different powers in the two tails. This module deliberately implements the two-piece scale construction of Fernández and Steel, which is the class called SEPD by Zhu and Zinde-Walsh and corresponds, up to scale convention, to GAMLSS `SEP3`.
//!
//! For a unit-variance symmetric power-exponential density `f_power`, the standardized two-piece kernel is
//!
//! `2 / (skew_ratio + 1 / skew_ratio) * f_power(x * skew_ratio)` for `x < 0`, and
//! `2 / (skew_ratio + 1 / skew_ratio) * f_power(x / skew_ratio)` for `x >= 0`.
//!
//! `skew_ratio = 1` gives the symmetric [`super::PowerExponential`] exactly. Values above one stretch the right half and values below one stretch the left half. The construction is also a reparameterization of the standardized SGED of Theodossiou; the [`SkewPowerExponentialMeanSd`] parameterization exposes that useful mean/standard-deviation form directly.
//!
//! [`SkewPowerExponential`] and [`SkewPowerExponentialMeanSd`] span exactly the same four-parameter family; the latter is not an additional distribution or a special case. The mode/base-scale form evaluates the density kernel directly, while the mean/SD form pays for analytic recentering and rescaling in exchange for predictors with moment semantics. Alternative SGED and SEPD spellings remain documentation terms rather than public type aliases because those names do not uniquely determine the parameter conventions.
//!
//! References: [Fernández and Steel (1998)](https://doi.org/10.1080/01621459.1998.10474117), [Zhu and Zinde-Walsh (2009)](https://doi.org/10.1016/j.jeconom.2008.09.038), and [Theodossiou (2015)](https://doi.org/10.17578/19-4-1).

use gamlss_special::{digamma, invert_real_cdf, ln_gamma, ln_gamma_delta};

use super::power_exponential::{
    d_log_standardized_scale_d_power, log_standardized_scale, standardized_cdf,
};

pub use mean_sd::{
    SkewPowerExponentialMeanSd, SkewPowerExponentialMeanSdEta, SkewPowerExponentialMeanSdSkewPower,
    SkewPowerExponentialMeanSdTheta,
};
pub use mode_scale::{
    SkewPowerExponential, SkewPowerExponentialEta, SkewPowerExponentialMuSigmaSkewPower,
    SkewPowerExponentialTheta,
};

mod mean_sd;
mod mode_scale;

#[derive(Debug, Clone, Copy)]
pub(super) struct NllGradient {
    pub(super) location: f64,
    pub(super) scale: f64,
    pub(super) log_skew_ratio: f64,
    pub(super) power: f64,
}

impl NllGradient {
    #[inline]
    const fn nan() -> Self {
        Self {
            location: f64::NAN,
            scale: f64::NAN,
            log_skew_ratio: f64::NAN,
            power: f64::NAN,
        }
    }

    #[inline]
    const fn is_finite(self) -> bool {
        self.location.is_finite()
            && self.scale.is_finite()
            && self.log_skew_ratio.is_finite()
            && self.power.is_finite()
    }
}

#[derive(Debug, Clone, Copy)]
struct StandardizedScale {
    log_value: f64,
}

impl StandardizedScale {
    #[inline]
    fn from_power(power: f64) -> Option<Self> {
        let log_value = log_standardized_scale(power);
        log_value.is_finite().then_some(Self { log_value })
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ModeScaleLogValues {
    pub(super) sigma: f64,
    pub(super) skew_ratio: f64,
    pub(super) power: f64,
}

impl ModeScaleLogValues {
    #[inline]
    fn from_natural(sigma: f64, skew_ratio: f64, power: f64) -> Self {
        Self {
            sigma: sigma.ln(),
            skew_ratio: skew_ratio.ln(),
            power: power.ln(),
        }
    }

    #[inline]
    const fn is_finite(self) -> bool {
        self.sigma.is_finite() && self.skew_ratio.is_finite() && self.power.is_finite()
    }
}

#[derive(Debug, Clone, Copy)]
struct PointGeometry {
    residual: f64,
    log_distance: f64,
    radial_power: f64,
}

impl PointGeometry {
    #[inline]
    fn new_with_logs(
        y: f64,
        mu: f64,
        log_sigma: f64,
        log_skew_ratio: f64,
        power: f64,
        standardized_scale: StandardizedScale,
    ) -> Self {
        let residual = y - mu;
        if residual == 0.0 {
            return Self {
                residual,
                log_distance: f64::NEG_INFINITY,
                radial_power: 0.0,
            };
        }
        let log_side_scale = if residual < 0.0 {
            log_skew_ratio
        } else {
            -log_skew_ratio
        };
        let log_distance =
            residual.abs().ln() + log_side_scale - log_sigma - standardized_scale.log_value;
        Self {
            residual,
            log_distance,
            radial_power: (power * log_distance).exp(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ModeScaleGeometry {
    pub(super) mode: f64,
    pub(super) scale: f64,
    raw_mean: f64,
    raw_sd: f64,
    absolute_first_moment: f64,
    difference: f64,
    one_minus_moment_squared: f64,
    standardized_scale: StandardizedScale,
}

#[derive(Debug, Clone, Copy)]
struct MomentDerivatives {
    raw_mean_log_skew_ratio: f64,
    log_raw_sd_log_skew_ratio: f64,
    raw_mean_power: f64,
    log_raw_sd_power: f64,
}

#[inline]
pub(super) fn valid_mode_scale(mu: f64, sigma: f64, skew_ratio: f64, power: f64) -> bool {
    mu.is_finite()
        && sigma > 0.0
        && sigma.is_finite()
        && skew_ratio > 0.0
        && skew_ratio.is_finite()
        && power > 0.0
        && power.is_finite()
}

#[inline]
fn log_skew_normalizer_from_log(log_skew_ratio: f64) -> f64 {
    let magnitude = log_skew_ratio.abs();
    magnitude + (-2.0 * magnitude).exp().ln_1p()
}

#[inline]
fn skew_difference(skew_ratio: f64) -> f64 {
    let log_ratio = skew_ratio.ln();
    if log_ratio >= 0.0 {
        -skew_ratio * (-2.0 * log_ratio).exp_m1()
    } else {
        (1.0 / skew_ratio) * (2.0 * log_ratio).exp_m1()
    }
}

#[inline]
fn nll_mode_scale_unchecked(
    logs: ModeScaleLogValues,
    power: f64,
    standardized_scale: StandardizedScale,
    point: PointGeometry,
) -> f64 {
    logs.sigma
        + standardized_scale.log_value
        + ln_gamma(1.0 / power)
        + log_skew_normalizer_from_log(logs.skew_ratio)
        - logs.power
        + point.radial_power
}

#[inline]
fn gradient_mode_scale_unchecked(
    sigma: f64,
    logs: ModeScaleLogValues,
    power: f64,
    point: PointGeometry,
    d_log_standardized_scale: f64,
) -> NllGradient {
    if !point.radial_power.is_finite() {
        return NllGradient::nan();
    }
    let radial_score = power * point.radial_power;
    let d_radial_d_power = if point.residual == 0.0 {
        0.0
    } else {
        point.radial_power
            * point
                .log_distance
                .mul_add(1.0, -power * d_log_standardized_scale)
    };
    let side_log_score = if point.residual < 0.0 {
        radial_score
    } else {
        -radial_score
    };

    NllGradient {
        location: if point.residual == 0.0 {
            0.0
        } else {
            -radial_score / point.residual
        },
        scale: (1.0 - radial_score) / sigma,
        log_skew_ratio: logs.skew_ratio.tanh() + side_log_score,
        power: d_log_standardized_scale - digamma(1.0 / power) / (power * power) - 1.0 / power
            + d_radial_d_power,
    }
}

#[inline]
pub(super) fn nll_mode_scale(y: f64, mu: f64, sigma: f64, skew_ratio: f64, power: f64) -> f64 {
    if !y.is_finite() || !valid_mode_scale(mu, sigma, skew_ratio, power) {
        return f64::INFINITY;
    }
    let Some(standardized_scale) = StandardizedScale::from_power(power) else {
        return f64::INFINITY;
    };
    let logs = ModeScaleLogValues::from_natural(sigma, skew_ratio, power);
    let point = PointGeometry::new_with_logs(
        y,
        mu,
        logs.sigma,
        logs.skew_ratio,
        power,
        standardized_scale,
    );
    nll_mode_scale_unchecked(logs, power, standardized_scale, point)
}

#[inline]
pub(super) fn nll_mode_scale_with_logs(
    y: f64,
    mu: f64,
    sigma: f64,
    skew_ratio: f64,
    power: f64,
    logs: ModeScaleLogValues,
) -> f64 {
    if !y.is_finite() || !valid_mode_scale(mu, sigma, skew_ratio, power) || !logs.is_finite() {
        return f64::INFINITY;
    }
    let Some(standardized_scale) = StandardizedScale::from_power(power) else {
        return f64::INFINITY;
    };
    let point = PointGeometry::new_with_logs(
        y,
        mu,
        logs.sigma,
        logs.skew_ratio,
        power,
        standardized_scale,
    );
    nll_mode_scale_unchecked(logs, power, standardized_scale, point)
}

#[inline]
pub(super) fn nll_and_gradient_mode_scale_with_logs(
    y: f64,
    mu: f64,
    sigma: f64,
    skew_ratio: f64,
    power: f64,
    logs: ModeScaleLogValues,
) -> (f64, NllGradient) {
    if !y.is_finite() || !valid_mode_scale(mu, sigma, skew_ratio, power) || !logs.is_finite() {
        return (f64::INFINITY, NllGradient::nan());
    }
    let Some(standardized_scale) = StandardizedScale::from_power(power) else {
        return (f64::INFINITY, NllGradient::nan());
    };
    let point = PointGeometry::new_with_logs(
        y,
        mu,
        logs.sigma,
        logs.skew_ratio,
        power,
        standardized_scale,
    );
    let nll = nll_mode_scale_unchecked(logs, power, standardized_scale, point);
    if !nll.is_finite() {
        return (nll, NllGradient::nan());
    }
    let d_log_standardized_scale = d_log_standardized_scale_d_power(power);
    (
        nll,
        gradient_mode_scale_unchecked(sigma, logs, power, point, d_log_standardized_scale),
    )
}

#[inline]
fn side_probabilities(skew_ratio: f64) -> (f64, f64) {
    let norm = skew_ratio.hypot(1.0);
    let left = (1.0 / norm).powi(2);
    let right = (skew_ratio / norm).powi(2);
    (left, right)
}

#[inline]
pub(super) fn cdf_mode_scale(y: f64, mu: f64, sigma: f64, skew_ratio: f64, power: f64) -> f64 {
    if !y.is_finite() || !valid_mode_scale(mu, sigma, skew_ratio, power) {
        return f64::NAN;
    }

    let standardized = (y - mu) / sigma;
    let (left_probability, right_probability) = side_probabilities(skew_ratio);
    if standardized < 0.0 {
        2.0 * left_probability * standardized_cdf(skew_ratio * standardized, power)
    } else {
        let standardized_survival = standardized_cdf(-standardized / skew_ratio, power);
        (2.0 * right_probability).mul_add(-standardized_survival, 1.0)
    }
}

#[inline]
pub(super) fn quantile_mode_scale(
    probability: f64,
    mu: f64,
    sigma: f64,
    skew_ratio: f64,
    power: f64,
) -> f64 {
    if !valid_mode_scale(mu, sigma, skew_ratio, power) {
        return f64::NAN;
    }
    invert_real_cdf(probability, |value| {
        cdf_mode_scale(value, mu, sigma, skew_ratio, power)
    })
}

#[inline]
pub(super) fn crps_mode_scale(y: f64, mu: f64, sigma: f64, skew_ratio: f64, power: f64) -> f64 {
    if !y.is_finite() || !valid_mode_scale(mu, sigma, skew_ratio, power) {
        return f64::NAN;
    }
    crate::crps::integrate_cdf_crps(y, sigma, |x| {
        cdf_mode_scale(x, mu, sigma, skew_ratio, power)
    })
}

#[inline]
fn raw_moment_geometry(skew_ratio: f64, power: f64) -> Option<ModeScaleGeometry> {
    if skew_ratio <= 0.0 || !skew_ratio.is_finite() || power <= 0.0 || !power.is_finite() {
        return None;
    }

    let standardized_scale = StandardizedScale::from_power(power)?;
    let inverse_power = 1.0 / power;
    let absolute_first_moment =
        (standardized_scale.log_value + ln_gamma_delta(inverse_power, inverse_power)).exp();
    let difference = skew_difference(skew_ratio);
    let one_minus_moment_squared = absolute_first_moment.mul_add(-absolute_first_moment, 1.0);
    let raw_sd = (one_minus_moment_squared.sqrt() * difference).hypot(1.0);
    if !absolute_first_moment.is_finite()
        || !difference.is_finite()
        || one_minus_moment_squared <= 0.0
        || !one_minus_moment_squared.is_finite()
        || !raw_sd.is_finite()
    {
        return None;
    }

    let raw_mean = absolute_first_moment * difference;
    let geometry = ModeScaleGeometry {
        mode: 0.0,
        scale: 1.0,
        raw_mean,
        raw_sd,
        absolute_first_moment,
        difference,
        one_minus_moment_squared,
        standardized_scale,
    };
    geometry.raw_mean.is_finite().then_some(geometry)
}

#[inline]
fn raw_moment_derivatives(
    geometry: ModeScaleGeometry,
    skew_ratio: f64,
    power: f64,
    d_log_standardized_scale: f64,
) -> MomentDerivatives {
    let reciprocal_ratio = 1.0 / skew_ratio;
    let d_difference_d_log_skew_ratio = skew_ratio + reciprocal_ratio;
    let d_log_absolute_moment_d_power = d_log_standardized_scale
        + 2.0_f64.mul_add(-digamma(2.0 / power), digamma(1.0 / power)) / (power * power);
    let d_absolute_moment_d_power = geometry.absolute_first_moment * d_log_absolute_moment_d_power;
    let difference_over_sd = geometry.difference / geometry.raw_sd;
    let sum_over_sd = d_difference_d_log_skew_ratio / geometry.raw_sd;

    MomentDerivatives {
        raw_mean_log_skew_ratio: geometry.absolute_first_moment * d_difference_d_log_skew_ratio,
        log_raw_sd_log_skew_ratio: geometry.one_minus_moment_squared
            * difference_over_sd
            * sum_over_sd,
        raw_mean_power: d_absolute_moment_d_power * geometry.difference,
        log_raw_sd_power: -geometry.absolute_first_moment
            * d_absolute_moment_d_power
            * difference_over_sd
            * difference_over_sd,
    }
}

#[inline]
pub(super) fn mean_sd_to_mode_scale(
    mean: f64,
    sigma: f64,
    skew_ratio: f64,
    power: f64,
) -> Option<ModeScaleGeometry> {
    if !mean.is_finite() || sigma <= 0.0 || !sigma.is_finite() {
        return None;
    }
    let mut geometry = raw_moment_geometry(skew_ratio, power)?;
    geometry.scale = sigma / geometry.raw_sd;
    let standardized_mean = geometry.raw_mean / geometry.raw_sd;
    geometry.mode = sigma.mul_add(-standardized_mean, mean);
    valid_mode_scale(geometry.mode, geometry.scale, skew_ratio, power).then_some(geometry)
}

#[inline]
pub(super) fn mode_scale_to_mean_sd(
    mu: f64,
    sigma: f64,
    skew_ratio: f64,
    power: f64,
) -> Option<(f64, f64)> {
    if !valid_mode_scale(mu, sigma, skew_ratio, power) {
        return None;
    }
    let geometry = raw_moment_geometry(skew_ratio, power)?;
    let mean = sigma.mul_add(geometry.raw_mean, mu);
    let sd = sigma * geometry.raw_sd;
    (mean.is_finite() && sd > 0.0 && sd.is_finite()).then_some((mean, sd))
}

#[inline]
pub(super) fn nll_mean_sd(y: f64, mean: f64, sigma: f64, skew_ratio: f64, power: f64) -> f64 {
    if !y.is_finite() {
        return f64::INFINITY;
    }
    let Some(geometry) = mean_sd_to_mode_scale(mean, sigma, skew_ratio, power) else {
        return f64::INFINITY;
    };
    let logs = ModeScaleLogValues::from_natural(geometry.scale, skew_ratio, power);
    let point = PointGeometry::new_with_logs(
        y,
        geometry.mode,
        logs.sigma,
        logs.skew_ratio,
        power,
        geometry.standardized_scale,
    );
    nll_mode_scale_unchecked(logs, power, geometry.standardized_scale, point)
}

#[inline]
fn gradient_mean_sd_from_geometry(
    skew_ratio: f64,
    logs: ModeScaleLogValues,
    power: f64,
    geometry: ModeScaleGeometry,
    point: PointGeometry,
    d_log_standardized_scale: f64,
) -> NllGradient {
    let raw =
        gradient_mode_scale_unchecked(geometry.scale, logs, power, point, d_log_standardized_scale);
    if !raw.is_finite() {
        return NllGradient::nan();
    }
    let derivatives = raw_moment_derivatives(geometry, skew_ratio, power, d_log_standardized_scale);
    let response_sd = geometry.scale * geometry.raw_sd;
    let standardized_mean = geometry.raw_mean / geometry.raw_sd;

    let d_mode_d_log_skew_ratio = response_sd
        * standardized_mean.mul_add(
            derivatives.log_raw_sd_log_skew_ratio,
            -derivatives.raw_mean_log_skew_ratio / geometry.raw_sd,
        );
    let d_scale_d_log_skew_ratio = -geometry.scale * derivatives.log_raw_sd_log_skew_ratio;
    let d_mode_d_power = response_sd
        * standardized_mean.mul_add(
            derivatives.log_raw_sd_power,
            -derivatives.raw_mean_power / geometry.raw_sd,
        );
    let d_scale_d_power = -geometry.scale * derivatives.log_raw_sd_power;

    NllGradient {
        location: raw.location,
        scale: standardized_mean.mul_add(-raw.location, raw.scale / geometry.raw_sd),
        log_skew_ratio: raw.scale.mul_add(
            d_scale_d_log_skew_ratio,
            raw.location
                .mul_add(d_mode_d_log_skew_ratio, raw.log_skew_ratio),
        ),
        power: raw.scale.mul_add(
            d_scale_d_power,
            raw.location.mul_add(d_mode_d_power, raw.power),
        ),
    }
}

#[inline]
pub(super) fn nll_and_gradient_mean_sd(
    y: f64,
    mean: f64,
    sigma: f64,
    skew_ratio: f64,
    power: f64,
) -> (f64, NllGradient) {
    if !y.is_finite() {
        return (f64::INFINITY, NllGradient::nan());
    }
    let Some(geometry) = mean_sd_to_mode_scale(mean, sigma, skew_ratio, power) else {
        return (f64::INFINITY, NllGradient::nan());
    };
    let logs = ModeScaleLogValues::from_natural(geometry.scale, skew_ratio, power);
    let point = PointGeometry::new_with_logs(
        y,
        geometry.mode,
        logs.sigma,
        logs.skew_ratio,
        power,
        geometry.standardized_scale,
    );
    let nll = nll_mode_scale_unchecked(logs, power, geometry.standardized_scale, point);
    if !nll.is_finite() {
        return (nll, NllGradient::nan());
    }
    let d_log_standardized_scale = d_log_standardized_scale_d_power(power);
    (
        nll,
        gradient_mean_sd_from_geometry(
            skew_ratio,
            logs,
            power,
            geometry,
            point,
            d_log_standardized_scale,
        ),
    )
}

#[cfg(feature = "rand")]
pub(super) fn try_sample_mode_scale<Rng>(
    rng: &mut Rng,
    mu: f64,
    sigma: f64,
    skew_ratio: f64,
    power: f64,
) -> Result<f64, gamlss_core::SimulationError>
where
    Rng: rand::Rng,
{
    if !valid_mode_scale(mu, sigma, skew_ratio, power) {
        return Err(gamlss_core::SimulationError::InvalidParameters(
            "two-piece power-exponential mode/scale",
        ));
    }

    let standardized_scale = StandardizedScale::from_power(power).ok_or(
        gamlss_core::SimulationError::NumericalFailure("power-exponential standardized scale"),
    )?;
    let gamma = rand_distr::Gamma::new(1.0 / power, 1.0).map_err(|_| {
        gamlss_core::SimulationError::BackendRejected("power-exponential radial shape")
    })?;
    let radial_gamma = rand_distr::Distribution::sample(&gamma, rng);
    let radius = (standardized_scale.log_value + radial_gamma.ln() / power).exp();
    let (_, right_probability) = side_probabilities(skew_ratio);
    let standardized = if crate::simulation::open_unit(rng) < right_probability {
        skew_ratio * radius
    } else {
        -radius / skew_ratio
    };
    crate::simulation::ensure_finite(
        sigma.mul_add(standardized, mu),
        "two-piece power-exponential transform",
    )
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::{mean_sd_to_mode_scale, nll_and_gradient_mean_sd};

    #[test]
    fn mean_sd_geometry_has_requested_first_two_moments() {
        let geometry = mean_sd_to_mode_scale(2.5, 1.7, 2.3, 1.4).unwrap();
        assert_relative_eq!(
            geometry.scale.mul_add(geometry.raw_mean, geometry.mode),
            2.5,
            epsilon = 1.0e-14
        );
        assert_relative_eq!(geometry.scale * geometry.raw_sd, 1.7, epsilon = 1.0e-14);
    }

    #[test]
    fn mean_sd_gradient_is_finite_away_from_the_mode() {
        let (nll, gradient) = nll_and_gradient_mean_sd(0.8, 0.2, 1.3, 1.8, 1.4);
        assert!(nll.is_finite());
        assert!(gradient.location.is_finite());
        assert!(gradient.scale.is_finite());
        assert!(gradient.log_skew_ratio.is_finite());
        assert!(gradient.power.is_finite());
    }
}
