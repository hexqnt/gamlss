use std::marker::PhantomData;

use gamlss_core::{Log, Logit, PositiveLink, UnitIntervalLink};

use crate::special::{included_count, is_nonnegative_integer, ln_gamma, log_add_exp};

pub use component_mean_size_zero_probability::{
    ZinbComponentMeanSizeZeroProbability, ZinbEta, ZinbMeanSizeZeroProbability,
};
pub use total_mean_size::{
    ZinbTotalMeanSizeZeroProbability, ZinbTotalMeanSizeZeroProbabilityEta,
    ZinbTotalMeanSizeZeroProbabilityTheta,
};

mod component_mean_size_zero_probability;
mod total_mean_size;

const MAX_CDF_TERMS: u64 = 1_000_000;

/// Zero-inflated negative binomial family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Zinb<MuLink = Log, ShapeLink = Log, NuLink = Logit> {
    marker: PhantomData<(MuLink, ShapeLink, NuLink)>,
}

impl<MuLink, ShapeLink, NuLink> Zinb<MuLink, ShapeLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    /// Creates a stateless ZINB family.
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn nb_log_pmf(y: f64, mu: f64, shape: f64) -> f64 {
        ln_gamma(y + shape) - ln_gamma(shape) - ln_gamma(y + 1.0)
            + shape * (shape / (shape + mu)).ln()
            + y * (mu / (shape + mu)).ln()
    }

    #[inline(always)]
    pub(super) fn nll_theta(y: f64, theta: ZinbTheta) -> f64 {
        if !is_nonnegative_integer(y)
            || theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
            || theta.nu <= 0.0
            || theta.nu >= 1.0
            || !theta.nu.is_finite()
        {
            return f64::INFINITY;
        }
        let log_nb = Self::nb_log_pmf(y, theta.mu, theta.shape);
        if y == 0.0 {
            -log_add_exp(theta.nu.ln(), (1.0 - theta.nu).ln() + log_nb)
        } else {
            -((1.0 - theta.nu).ln() + log_nb)
        }
    }

    fn nb_cdf(y: f64, mu: f64, shape: f64) -> f64 {
        if y < 0.0 {
            return 0.0;
        }
        let Some(max_count) = included_count(y, MAX_CDF_TERMS) else {
            return f64::NAN;
        };
        let success_probability = shape / (shape + mu);
        let failure_probability = mu / (shape + mu);
        let mut term = (shape * success_probability.ln()).exp();
        let mut sum = term;
        for count in 1..=max_count {
            let previous = (count - 1) as f64;
            term *= ((previous + shape) / count as f64) * failure_probability;
            sum += term;
            if term <= f64::EPSILON * sum {
                break;
            }
        }
        sum.clamp(0.0, 1.0)
    }

    pub(super) fn cdf_theta(y: f64, theta: ZinbTheta) -> f64 {
        if !y.is_finite()
            || theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
            || theta.nu <= 0.0
            || theta.nu >= 1.0
            || !theta.nu.is_finite()
        {
            return f64::NAN;
        }
        if y < 0.0 {
            return 0.0;
        }

        (theta.nu + (1.0 - theta.nu) * Self::nb_cdf(y, theta.mu, theta.shape)).clamp(0.0, 1.0)
    }
}

impl<MuLink, ShapeLink, NuLink> Default for Zinb<MuLink, ShapeLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Natural-scale ZINB parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZinbTheta {
    /// Positive negative-binomial mean.
    pub mu: f64,
    /// Positive negative-binomial shape.
    pub shape: f64,
    /// Zero-inflation probability in `(0, 1)`.
    pub nu: f64,
}
