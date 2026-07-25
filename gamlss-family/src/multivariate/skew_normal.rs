#![allow(clippy::needless_range_loop, clippy::suboptimal_flops)]

//! Multivariate skew-normal distributions.

use gamlss_special::{log_ndtr, normal_mills_ratio};

use crate::constants::LOG_2;
#[cfg(feature = "rand")]
use crate::multivariate::elliptical::LowerTriangularMatrix;

pub use cholesky::{
    MvSkewNormalCholesky, MvSkewNormalCholeskyDefault, MvSkewNormalCholeskyEta,
    MvSkewNormalCholeskyTheta,
};
pub use location_kernel_std_partial_corr::{
    MvSkewNormalLocationKernelStdPartialCorr, MvSkewNormalLocationKernelStdPartialCorrDefault,
    MvSkewNormalLocationKernelStdPartialCorrEta, MvSkewNormalLocationKernelStdPartialCorrTheta,
};

mod cholesky;
mod location_kernel_std_partial_corr;

pub(super) struct SkewGradientTerms<const D: usize> {
    pub nll: f64,
    pub standardized: [f64; D],
    pub shape: [f64; D],
}

pub(super) fn skew_nll<const D: usize>(
    gaussian_nll: f64,
    standardized: &[f64; D],
    shape: &[f64; D],
) -> f64 {
    if !gaussian_nll.is_finite() {
        return f64::INFINITY;
    }
    let argument = shape
        .iter()
        .zip(standardized)
        .map(|(shape, standardized)| shape * standardized)
        .sum::<f64>();
    let nll = gaussian_nll - LOG_2 - log_ndtr(argument);
    if nll.is_finite() { nll } else { f64::INFINITY }
}

pub(super) fn skew_gradient_terms<const D: usize>(
    gaussian_nll: f64,
    standardized: &[f64; D],
    shape: &[f64; D],
) -> Option<SkewGradientTerms<D>> {
    let argument = shape
        .iter()
        .zip(standardized)
        .map(|(shape, standardized)| shape * standardized)
        .sum::<f64>();
    let mills = normal_mills_ratio(argument);
    let nll = gaussian_nll - LOG_2 - log_ndtr(argument);
    if !nll.is_finite() || !mills.is_finite() {
        return None;
    }
    Some(SkewGradientTerms {
        nll,
        standardized: std::array::from_fn(|component| {
            standardized[component] - mills * shape[component]
        }),
        shape: std::array::from_fn(|component| -mills * standardized[component]),
    })
}

#[cfg(feature = "rand")]
pub(super) fn try_sample_location_scale<Rng, const D: usize>(
    rng: &mut Rng,
    location: [f64; D],
    cholesky: &impl LowerTriangularMatrix,
    shape: &[f64; D],
) -> Result<[f64; D], gamlss_core::SimulationError>
where
    Rng: rand::Rng,
{
    let normalization = shape.iter().fold(1.0_f64, |norm, shape| norm.hypot(*shape));
    let delta = shape.map(|shape| shape / normalization);
    let rank_one_scale = 1.0 / (1.0 + normalization.recip());
    let normal = rand_distr::StandardNormal;
    let latent: f64 = rand_distr::Distribution::sample(&normal, rng);
    let noise: [f64; D] = std::array::from_fn(|_| rand_distr::Distribution::sample(&normal, rng));
    let projection = delta
        .iter()
        .zip(noise)
        .map(|(delta, noise)| delta * noise)
        .sum::<f64>();
    let standardized: [f64; D] = std::array::from_fn(|component| {
        delta[component] * latent.abs() + noise[component]
            - rank_one_scale * delta[component] * projection
    });

    let mut out = location;
    for row in 0..D {
        for col in 0..=row {
            out[row] += cholesky.lower(row, col) * standardized[col];
        }
    }
    if out.iter().all(|value| value.is_finite()) {
        Ok(out)
    } else {
        Err(gamlss_core::SimulationError::NumericalFailure(
            "multivariate skew-normal transform",
        ))
    }
}
