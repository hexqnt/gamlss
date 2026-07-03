use std::marker::PhantomData;

use gamlss_core::{Log, Logit, PositiveLink, UnitIntervalLink};

use gamlss_special::{digamma, is_nonnegative_integer, ln_gamma, log_add_exp};

use super::negative_binomial::{NegativeBinomial, NegativeBinomialTheta};

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
///
/// The default parameterization models the negative-binomial component mean,
/// component size, and zero-inflation probability. Use
/// [`ZinbTotalMeanSizeZeroProbability`] for the derived unconditional-mean
/// parameterization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn nb_log_pmf(y: f64, mu: f64, shape: f64) -> f64 {
        ln_gamma(y + shape) - ln_gamma(shape) - ln_gamma(y + 1.0)
            + shape * (shape / (shape + mu)).ln()
            + y * (mu / (shape + mu)).ln()
    }

    #[inline]
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

    #[inline]
    fn negative_binomial_gradient_theta(y: f64, mu: f64, shape: f64) -> (f64, f64) {
        let total = shape + mu;
        let d_mu = (y + shape) / total - y / mu;
        let d_shape = -digamma(y + shape) + digamma(shape) - shape.ln() - 1.0
            + total.ln()
            + (y + shape) / total;
        (d_mu, d_shape)
    }

    #[inline]
    pub(super) fn gradient_component_theta(y: f64, theta: ZinbTheta) -> ZinbTheta {
        let (d_mu, d_shape) = Self::negative_binomial_gradient_theta(y, theta.mu, theta.shape);
        if y == 0.0 {
            let q0 = (theta.shape / (theta.shape + theta.mu)).powf(theta.shape);
            let p0 = (1.0 - theta.nu).mul_add(q0, theta.nu);
            let responsibility = (1.0 - theta.nu) * q0 / p0;
            ZinbTheta {
                mu: responsibility * d_mu,
                shape: responsibility * d_shape,
                nu: -(1.0 - q0) / p0,
            }
        } else {
            ZinbTheta {
                mu: d_mu,
                shape: d_shape,
                nu: 1.0 / (1.0 - theta.nu),
            }
        }
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

        let base_cdf = NegativeBinomial::<Log, Log>::cdf_theta(
            y,
            NegativeBinomialTheta {
                mu: theta.mu,
                shape: theta.shape,
            },
        );
        (1.0 - theta.nu).mul_add(base_cdf, theta.nu).clamp(0.0, 1.0)
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
