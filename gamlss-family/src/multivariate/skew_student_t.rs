#![allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]

//! Multivariate skew-Student-t distributions.

use gamlss_special::{StandardStudentTKernel, student_t_log_cdf_standardized};

use crate::constants::LOG_2;
use crate::multivariate::student_t::MvStudentTKernel;

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

/// Prepared fixed-`tau` multivariate skew-Student-t density.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct MvSkewStudentTKernel<const D: usize> {
    base: MvStudentTKernel<D>,
    skew_density: StandardStudentTKernel,
}

impl<const D: usize> MvSkewStudentTKernel<D> {
    pub(super) fn try_new(tau: f64) -> Option<Self> {
        let base = MvStudentTKernel::try_new(tau)?;
        let skew_density = StandardStudentTKernel::try_new(tau + D as f64)?;
        Some(Self { base, skew_density })
    }

    #[inline]
    pub(super) const fn tau(self) -> f64 {
        self.base.tau()
    }

    pub(super) fn base_nll_location_scale(
        self,
        observation: [f64; D],
        location: &[f64; D],
        cholesky: &impl crate::multivariate::elliptical::LowerTriangularMatrix,
        standardized: &mut [f64; D],
    ) -> f64 {
        self.base
            .nll_location_scale(observation, location, cholesky, standardized)
    }

    pub(super) fn nll(self, base_nll: f64, standardized: &[f64; D], shape: &[f64; D]) -> f64 {
        if !base_nll.is_finite() {
            return f64::INFINITY;
        }
        let quadratic = standardized.iter().map(|value| value * value).sum::<f64>();
        let projection = shape
            .iter()
            .zip(standardized)
            .map(|(shape, standardized)| shape * standardized)
            .sum::<f64>();
        let skew_df = self.skew_density.degrees_of_freedom();
        let argument = projection * (skew_df / (self.tau() + quadratic)).sqrt();
        let nll = base_nll - LOG_2 - student_t_log_cdf_standardized(argument, skew_df);
        if nll.is_finite() { nll } else { f64::INFINITY }
    }

    pub(super) fn gradient_terms(
        self,
        base_nll: f64,
        standardized: &[f64; D],
        shape: &[f64; D],
    ) -> Option<SkewStudentTGradientTerms<D>> {
        if !base_nll.is_finite() {
            return None;
        }
        let dimension = D as f64;
        let tau = self.tau();
        let quadratic = standardized.iter().map(|value| value * value).sum::<f64>();
        let denominator = tau + quadratic;
        let projection = shape
            .iter()
            .zip(standardized)
            .map(|(shape, standardized)| shape * standardized)
            .sum::<f64>();
        let skew_df = self.skew_density.degrees_of_freedom();
        let argument_scale = (skew_df / denominator).sqrt();
        let argument = projection * argument_scale;
        let skew_log_probability = student_t_log_cdf_standardized(argument, skew_df);
        let skew_log_density = self.skew_density.log_pdf(argument);
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
