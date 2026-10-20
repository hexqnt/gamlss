#![allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]

//! Multivariate power-exponential distributions.

use gamlss_special::{digamma, ln_gamma};

use super::elliptical::{self, LowerTriangularMatrix};

pub use cholesky::{
    MvPowerExponentialCholesky, MvPowerExponentialCholeskyDefault, MvPowerExponentialCholeskyEta,
    MvPowerExponentialCholeskyTheta,
};
pub use mean_std_partial_corr::{
    MvPowerExponentialMeanStdPartialCorr, MvPowerExponentialMeanStdPartialCorrDefault,
    MvPowerExponentialMeanStdPartialCorrEta, MvPowerExponentialMeanStdPartialCorrTheta,
};

mod cholesky;
mod mean_std_partial_corr;

/// Radial values shared by the two scatter parameterizations.
pub(super) struct RadialTerms {
    pub(super) nll: f64,
    pub(super) quadratic: f64,
    pub(super) radial_power: f64,
}

impl RadialTerms {
    pub(super) fn location_score<const D: usize>(
        &self,
        cholesky: &impl LowerTriangularMatrix,
        standardized: [f64; D],
        power: f64,
    ) -> Option<([f64; D], f64)> {
        let radial_weight = if self.quadratic == 0.0 {
            0.0
        } else {
            0.5 * power * self.quadratic.powf(0.5 * power - 1.0)
        };
        let standardized_gradient = standardized.map(|value| radial_weight * value);
        let mut location_score = [0.0; D];
        elliptical::transpose_solve(D, cholesky, &standardized_gradient, &mut location_score)
            .then_some((location_score, radial_weight))
    }
}

pub(super) fn location_scale_terms<const D: usize>(
    observation: &[f64; D],
    location: &[f64; D],
    cholesky: &impl LowerTriangularMatrix,
    power: f64,
    standardized: &mut [f64; D],
) -> Option<RadialTerms> {
    let (quadratic, log_det_scale) =
        elliptical::standardize(D, observation, location, cholesky, standardized)?;
    let radial_power = quadratic.powf(0.5 * power);
    let nll = log_det_scale + radial_nll_constant(D as f64, power) + 0.5 * radial_power;
    nll.is_finite().then_some(RadialTerms {
        nll,
        quadratic,
        radial_power,
    })
}

#[cfg(feature = "rand")]
pub(super) fn try_sample_location_scale<Rng, const D: usize>(
    rng: &mut Rng,
    location: [f64; D],
    cholesky: &impl LowerTriangularMatrix,
    power: f64,
    numerical_context: &'static str,
) -> Result<[f64; D], gamlss_core::SimulationError>
where
    Rng: rand::Rng,
{
    use gamlss_core::SimulationError;

    let dimension = D as f64;
    let gamma = rand_distr::Gamma::new(dimension / power, 1.0)
        .map_err(|_| SimulationError::BackendRejected("power-exponential radial shape"))?;
    let radial_gamma = rand_distr::Distribution::sample(&gamma, rng);
    let radius = (2.0 * radial_gamma).powf(1.0 / power);
    let normal = rand_distr::StandardNormal;
    let direction: [f64; D] =
        std::array::from_fn(|_| rand_distr::Distribution::sample(&normal, rng));
    let direction_norm = direction
        .iter()
        .fold(0.0_f64, |norm, value| norm.hypot(*value));
    if !radius.is_finite() || direction_norm == 0.0 || !direction_norm.is_finite() {
        return Err(SimulationError::NumericalFailure(
            "power-exponential radial transform",
        ));
    }
    let standardized = direction.map(|value| radius * value / direction_norm);
    let mut out = location;
    for (row, out) in out.iter_mut().enumerate() {
        for (col, value) in standardized.iter().copied().take(row + 1).enumerate() {
            *out += cholesky.lower(row, col) * value;
        }
    }
    if out.iter().all(|value| value.is_finite()) {
        Ok(out)
    } else {
        Err(SimulationError::NumericalFailure(numerical_context))
    }
}

pub(super) fn radial_nll_constant(dimension: f64, power: f64) -> f64 {
    -power.ln() - ln_gamma(0.5 * dimension)
        + (1.0 + dimension / power) * std::f64::consts::LN_2
        + 0.5 * dimension * std::f64::consts::PI.ln()
        + ln_gamma(dimension / power)
}

pub(super) fn direct_power_score(
    dimension: f64,
    power: f64,
    quadratic: f64,
    radial_power: f64,
) -> f64 {
    let radial_derivative = if quadratic == 0.0 {
        0.0
    } else {
        0.25 * radial_power * quadratic.ln()
    };
    -1.0 / power
        - dimension * (std::f64::consts::LN_2 + digamma(dimension / power)) / (power * power)
        + radial_derivative
}

pub(super) fn log_covariance_multiplier(dimension: f64, power: f64) -> f64 {
    2.0 * std::f64::consts::LN_2 / power + ln_gamma((dimension + 2.0) / power)
        - ln_gamma(dimension / power)
        - dimension.ln()
}

pub(super) fn d_log_covariance_multiplier_d_power(dimension: f64, power: f64) -> f64 {
    (-2.0 * std::f64::consts::LN_2 - (dimension + 2.0) * digamma((dimension + 2.0) / power)
        + dimension * digamma(dimension / power))
        / (power * power)
}
