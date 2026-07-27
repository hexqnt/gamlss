#![allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]

//! Multivariate Student-t distributions and shared elliptical kernel.

use gamlss_special::{digamma_delta, ln_gamma_delta};

#[cfg(feature = "rand")]
use gamlss_core::SimulationError;

use crate::multivariate::elliptical::{self, LowerTriangularMatrix};

pub use cholesky::{
    MvStudentTCholesky, MvStudentTCholeskyDefault, MvStudentTCholeskyEta, MvStudentTCholeskyTheta,
};
pub use mean_std_partial_corr::{
    MvStudentTMeanStdPartialCorr, MvStudentTMeanStdPartialCorrDefault,
    MvStudentTMeanStdPartialCorrEta, MvStudentTMeanStdPartialCorrTheta,
};

mod cholesky;
mod mean_std_partial_corr;

pub(super) fn valid_degrees_of_freedom(tau: f64) -> bool {
    tau > 0.0 && tau.is_finite()
}

pub(super) fn robust_weight(dimension: f64, tau: f64, quadratic: f64) -> f64 {
    let scale = tau.max(dimension).max(quadratic);
    (tau / scale + dimension / scale) / (tau / scale + quadratic / scale)
}

/// Prepared multivariate Student-t density for one dimension and fixed `tau`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct MvStudentTKernel<const D: usize> {
    tau: f64,
    nll_constant: f64,
}

impl<const D: usize> MvStudentTKernel<D> {
    #[inline]
    pub(super) fn try_new(tau: f64) -> Option<Self> {
        if D == 0 || !valid_degrees_of_freedom(tau) {
            return None;
        }
        let dimension = D as f64;
        Some(Self {
            tau,
            nll_constant: -ln_gamma_delta(0.5 * tau, 0.5 * dimension)
                + 0.5 * dimension * (tau.ln() + std::f64::consts::PI.ln()),
        })
    }

    #[inline]
    pub(super) const fn tau(self) -> f64 {
        self.tau
    }

    pub(super) fn nll_location_scale(
        self,
        observation: [f64; D],
        location: &[f64; D],
        cholesky: &impl LowerTriangularMatrix,
        standardized: &mut [f64; D],
    ) -> f64 {
        let Some((quadratic, log_det_scale)) =
            elliptical::standardize(D, &observation, location, cholesky, standardized)
        else {
            return f64::INFINITY;
        };
        log_det_scale
            + f64::midpoint(self.tau, D as f64) * (quadratic / self.tau).ln_1p()
            + self.nll_constant
    }
}

fn direct_tau_score(dimension: f64, tau: f64, quadratic: f64) -> f64 {
    let quadratic_fraction = if quadratic == 0.0 {
        0.0
    } else if quadratic < tau {
        let ratio = quadratic / tau;
        ratio / (1.0 + ratio)
    } else {
        1.0 / (1.0 + tau / quadratic)
    };
    let tail_derivative = if quadratic_fraction == 0.0 {
        0.0
    } else {
        f64::midpoint(1.0, dimension / tau) * quadratic_fraction
    };
    -0.5 * digamma_delta(0.5 * tau, 0.5 * dimension)
        + 0.5 * dimension / tau
        + 0.5 * (quadratic / tau).ln_1p()
        - tail_derivative
}

pub(super) fn nll_location_scale<const D: usize>(
    observation: [f64; D],
    location: &[f64; D],
    cholesky: &impl LowerTriangularMatrix,
    tau: f64,
    standardized: &mut [f64; D],
) -> f64 {
    let Some(kernel) = MvStudentTKernel::try_new(tau) else {
        return f64::INFINITY;
    };
    kernel.nll_location_scale(observation, location, cholesky, standardized)
}

#[cfg(feature = "rand")]
fn try_sample_location_scale<Rng, const D: usize>(
    rng: &mut Rng,
    location: [f64; D],
    cholesky: &impl LowerTriangularMatrix,
    tau: f64,
    parameter_context: &'static str,
    numerical_context: &'static str,
) -> Result<[f64; D], SimulationError>
where
    Rng: rand::Rng,
{
    let chi_squared = rand_distr::ChiSquared::new(tau)
        .map_err(|_| SimulationError::BackendRejected(parameter_context))?;
    let mixture = (tau / rand_distr::Distribution::sample(&chi_squared, rng)).sqrt();
    if !mixture.is_finite() {
        return Err(SimulationError::NumericalFailure(numerical_context));
    }

    let normal = rand_distr::StandardNormal;
    let z: [f64; D] = std::array::from_fn(|_| rand_distr::Distribution::sample(&normal, rng));
    let mut out = location;
    for (row, out) in out.iter_mut().enumerate() {
        for (col, z) in z.iter().copied().take(row + 1).enumerate() {
            *out += mixture * cholesky.lower(row, col) * z;
        }
    }
    if out.iter().all(|value| value.is_finite()) {
        Ok(out)
    } else {
        Err(SimulationError::NumericalFailure(numerical_context))
    }
}
