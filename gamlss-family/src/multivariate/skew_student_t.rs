#![allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]

//! Multivariate skew-Student-t distributions.

use gamlss_special::{student_t_log_cdf_standardized, student_t_log_pdf_standardized};

use crate::constants::LOG_2;

pub use fixed_tau_cholesky::{
    MvSkewStudentTFixedTauCholesky, MvSkewStudentTFixedTauCholeskyDefault,
    MvSkewStudentTFixedTauCholeskyEta, MvSkewStudentTFixedTauCholeskyTheta,
};

mod fixed_tau_cholesky;

pub(super) struct SkewStudentTGradientTerms<const D: usize> {
    pub nll: f64,
    pub standardized: [f64; D],
    pub shape: [f64; D],
}

pub(super) fn skew_student_t_nll<const D: usize>(
    base_nll: f64,
    standardized: &[f64; D],
    shape: &[f64; D],
    tau: f64,
) -> f64 {
    if !base_nll.is_finite() {
        return f64::INFINITY;
    }
    let quadratic = standardized.iter().map(|value| value * value).sum::<f64>();
    let projection = shape
        .iter()
        .zip(standardized)
        .map(|(shape, standardized)| shape * standardized)
        .sum::<f64>();
    let argument = projection * ((tau + D as f64) / (tau + quadratic)).sqrt();
    let nll = base_nll - LOG_2 - student_t_log_cdf_standardized(argument, tau + D as f64);
    if nll.is_finite() { nll } else { f64::INFINITY }
}

pub(super) fn skew_student_t_gradient_terms<const D: usize>(
    base_nll: f64,
    standardized: &[f64; D],
    shape: &[f64; D],
    tau: f64,
) -> Option<SkewStudentTGradientTerms<D>> {
    if !base_nll.is_finite() {
        return None;
    }
    let dimension = D as f64;
    let quadratic = standardized.iter().map(|value| value * value).sum::<f64>();
    let denominator = tau + quadratic;
    let projection = shape
        .iter()
        .zip(standardized)
        .map(|(shape, standardized)| shape * standardized)
        .sum::<f64>();
    let argument_scale = ((tau + dimension) / denominator).sqrt();
    let argument = projection * argument_scale;
    let skew_log_probability = student_t_log_cdf_standardized(argument, tau + dimension);
    let skew_log_density = student_t_log_pdf_standardized(argument, tau + dimension);
    let cdf_score = (skew_log_density - skew_log_probability).exp();
    let nll = base_nll - LOG_2 - skew_log_probability;
    if !nll.is_finite() || !cdf_score.is_finite() || !argument_scale.is_finite() {
        return None;
    }
    let base_weight = crate::multivariate::student_t::robust_weight(dimension, tau, quadratic);
    Some(SkewStudentTGradientTerms {
        nll,
        standardized: std::array::from_fn(|component| {
            base_weight * standardized[component]
                - cdf_score
                    * argument_scale
                    * (shape[component] - projection * standardized[component] / denominator)
        }),
        shape: std::array::from_fn(|component| {
            -cdf_score * argument_scale * standardized[component]
        }),
    })
}

#[cfg(feature = "rand")]
pub(super) fn try_sample_location_scale<Rng, const D: usize>(
    rng: &mut Rng,
    location: [f64; D],
    cholesky: &impl crate::multivariate::elliptical::LowerTriangularMatrix,
    shape: &[f64; D],
    tau: f64,
) -> Result<[f64; D], gamlss_core::SimulationError>
where
    Rng: rand::Rng,
{
    let skew_normal = crate::multivariate::skew_normal::try_sample_location_scale(
        rng, [0.0; D], cholesky, shape,
    )?;
    let chi_squared = rand_distr::ChiSquared::new(tau)
        .map_err(|_| gamlss_core::SimulationError::BackendRejected("skew-Student-t tau"))?;
    let mixture = (tau / rand_distr::Distribution::sample(&chi_squared, rng)).sqrt();
    if !mixture.is_finite() {
        return Err(gamlss_core::SimulationError::NumericalFailure(
            "multivariate skew-Student-t scale mixture",
        ));
    }
    let out =
        std::array::from_fn(|component| location[component] + mixture * skew_normal[component]);
    if out.iter().all(|value| value.is_finite()) {
        Ok(out)
    } else {
        Err(gamlss_core::SimulationError::NumericalFailure(
            "multivariate skew-Student-t transform",
        ))
    }
}
