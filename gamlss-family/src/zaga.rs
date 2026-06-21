use std::marker::PhantomData;

use gamlss_core::{Log, Logit, PositiveLink, UnitIntervalLink};

use crate::special::{ln_gamma, regularized_gamma_lower};

pub use component_mean_cv_zero_probability::{
    ZagaComponentMeanCvZeroProbability, ZagaEta, ZagaMeanSigmaZeroProbability,
};
pub use total_mean_cv::{
    ZagaTotalMeanCvZeroProbability, ZagaTotalMeanCvZeroProbabilityEta,
    ZagaTotalMeanCvZeroProbabilityTheta,
};

mod component_mean_cv_zero_probability;
mod total_mean_cv;

/// Zero-adjusted gamma family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Zaga<MuLink = Log, SigmaLink = Log, NuLink = Logit> {
    marker: PhantomData<(MuLink, SigmaLink, NuLink)>,
}

impl<MuLink, SigmaLink, NuLink> Zaga<MuLink, SigmaLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    /// Creates a stateless ZAGA family.
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn gamma_shape_rate(theta: ZagaTheta) -> (f64, f64) {
        let shape = 1.0 / (theta.sigma * theta.sigma);
        let rate = 1.0 / (theta.sigma * theta.sigma * theta.mu);
        (shape, rate)
    }

    #[inline(always)]
    fn gamma_nll(y: f64, shape: f64, rate: f64) -> f64 {
        ln_gamma(shape) - shape * rate.ln() - (shape - 1.0) * y.ln() + rate * y
    }

    #[inline(always)]
    pub(super) fn nll_theta(y: f64, theta: ZagaTheta) -> f64 {
        if y < 0.0
            || !y.is_finite()
            || theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || theta.nu <= 0.0
            || theta.nu >= 1.0
            || !theta.nu.is_finite()
        {
            return f64::INFINITY;
        }
        if y == 0.0 {
            return -theta.nu.ln();
        }
        let (shape, rate) = Self::gamma_shape_rate(theta);
        -(1.0 - theta.nu).ln() + Self::gamma_nll(y, shape, rate)
    }

    pub(super) fn cdf_theta(y: f64, theta: ZagaTheta) -> f64 {
        if !y.is_finite()
            || theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || theta.nu <= 0.0
            || theta.nu >= 1.0
            || !theta.nu.is_finite()
        {
            return f64::NAN;
        }
        if y < 0.0 {
            return 0.0;
        }
        if y == 0.0 {
            return theta.nu;
        }
        let (shape, rate) = Self::gamma_shape_rate(theta);
        (theta.nu + (1.0 - theta.nu) * regularized_gamma_lower(shape, rate * y)).clamp(0.0, 1.0)
    }
}

impl<MuLink, SigmaLink, NuLink> Default for Zaga<MuLink, SigmaLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Natural-scale ZAGA parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZagaTheta {
    /// Positive mean for the gamma component.
    pub mu: f64,
    /// Positive coefficient of variation for the gamma component.
    pub sigma: f64,
    /// Zero-mass probability in `(0, 1)`.
    pub nu: f64,
}
