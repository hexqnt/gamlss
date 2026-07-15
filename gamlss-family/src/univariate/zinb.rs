use std::marker::PhantomData;

use gamlss_core::{Log, Logit, PositiveLink, UnitIntervalLink};

use gamlss_special::{is_nonnegative_integer, log_add_exp};

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
        -NegativeBinomial::<Log, Log>::nll_theta(y, NegativeBinomialTheta { mu, shape })
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
    pub(super) fn gradient_component_theta(y: f64, theta: ZinbTheta) -> ZinbTheta {
        let gradient = NegativeBinomial::<Log, Log>::gradient_theta(
            y,
            NegativeBinomialTheta {
                mu: theta.mu,
                shape: theta.shape,
            },
        );
        if y == 0.0 {
            let log_q0 = -theta.shape * (theta.mu / theta.shape).ln_1p();
            let q0 = log_q0.exp();
            let one_minus_q0 = -log_q0.exp_m1();
            let p0 = theta.nu.mul_add(one_minus_q0, q0);
            let responsibility = (1.0 - theta.nu) * q0 / p0;
            ZinbTheta {
                mu: responsibility * gradient.mu,
                shape: responsibility * gradient.shape,
                nu: -one_minus_q0 / p0,
            }
        } else {
            ZinbTheta {
                mu: gradient.mu,
                shape: gradient.shape,
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

    #[cfg(feature = "rand")]
    pub(super) fn sample_component_theta<Rng>(rng: &mut Rng, theta: ZinbTheta) -> f64
    where
        Rng: rand::Rng,
    {
        if theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
            || theta.nu <= 0.0
            || theta.nu >= 1.0
            || !theta.nu.is_finite()
        {
            return f64::NAN;
        }
        if crate::simulation::open_unit(rng) <= theta.nu {
            return 0.0;
        }

        let lambda = rand_distr::Distribution::sample(
            &rand_distr::Gamma::new(theta.shape, theta.mu / theta.shape)
                .expect("validated ZINB gamma-poisson parameters must construct"),
            rng,
        );
        rand_distr::Distribution::sample(
            &rand_distr::Poisson::new(lambda).expect("validated ZINB poisson mean must construct"),
            rng,
        )
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
