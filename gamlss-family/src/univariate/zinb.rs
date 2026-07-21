use std::marker::PhantomData;

use gamlss_core::{Log, Logit, PositiveLink, UnitIntervalLink};

use gamlss_special::{discrete_quantile, is_nonnegative_integer, log_add_exp};

use super::negative_binomial::{NegativeBinomialKernel, NegativeBinomialTheta};

pub use component_mean_size_zero_probability::{
    ZinbComponentMeanSizeZeroProbability, ZinbComponentMeanSizeZeroProbabilityEta,
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
/// Let $q_r(y\mid\mu)$ denote the negative-binomial probability mass with component mean $\mu>0$ and size $r>0$, and let $\pi\in(0,1)$ be the structural-zero probability. Then
///
/// $$
/// \Pr(Y=0)=\pi+(1-\pi)q_r(0\mid\mu),
/// $$
///
/// and, for $y\in\\{1,2,\ldots\\}$,
///
/// $$
/// \Pr(Y=y)=(1-\pi)q_r(y\mid\mu).
/// $$
///
/// Thus $\mathbb{E}(Y)=(1-\pi)\mu$. The default parameterization models the component mean, component size, and structural-zero probability; use [`ZinbTotalMeanSizeZeroProbability`] to model the unconditional mean instead.
///
/// The default component parameterization uses [`ZinbComponentMeanSizeZeroProbabilityTheta`]; use [`ZinbTotalMeanSizeZeroProbability`] when the first modeled parameter should be the unconditional mean.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/distributions/zinb.svg")
)]
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Zinb<ComponentMeanLink = Log, SizeLink = Log, ZeroProbabilityLink = Logit> {
    marker: PhantomData<(ComponentMeanLink, SizeLink, ZeroProbabilityLink)>,
}

impl<ComponentMeanLink, SizeLink, ZeroProbabilityLink>
    Zinb<ComponentMeanLink, SizeLink, ZeroProbabilityLink>
where
    ComponentMeanLink: PositiveLink<f64>,
    SizeLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    /// Creates a stateless ZINB family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

/// Link-independent zero-inflated negative-binomial kernel.
#[derive(Debug, Clone, Copy)]
pub(super) struct ZinbKernel;

impl ZinbKernel {
    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn nb_log_pmf(y: f64, component_mean: f64, size: f64) -> f64 {
        -NegativeBinomialKernel::nll_theta(
            y,
            NegativeBinomialTheta {
                mu: component_mean,
                shape: size,
            },
        )
    }

    #[inline]
    pub(super) fn nll_theta(y: f64, theta: ZinbComponentMeanSizeZeroProbabilityTheta) -> f64 {
        if !is_nonnegative_integer(y)
            || theta.component_mean <= 0.0
            || !theta.component_mean.is_finite()
            || theta.size <= 0.0
            || !theta.size.is_finite()
            || theta.zero_probability <= 0.0
            || theta.zero_probability >= 1.0
            || !theta.zero_probability.is_finite()
        {
            return f64::INFINITY;
        }
        let log_nb = Self::nb_log_pmf(y, theta.component_mean, theta.size);
        if y == 0.0 {
            -log_add_exp(
                theta.zero_probability.ln(),
                (1.0 - theta.zero_probability).ln() + log_nb,
            )
        } else {
            -((1.0 - theta.zero_probability).ln() + log_nb)
        }
    }

    #[inline]
    pub(super) fn gradient_component_theta(
        y: f64,
        theta: ZinbComponentMeanSizeZeroProbabilityTheta,
    ) -> ZinbComponentMeanSizeZeroProbabilityTheta {
        let gradient = NegativeBinomialKernel::gradient_theta(
            y,
            NegativeBinomialTheta {
                mu: theta.component_mean,
                shape: theta.size,
            },
        );
        if y == 0.0 {
            let log_q0 = -theta.size * (theta.component_mean / theta.size).ln_1p();
            let q0 = log_q0.exp();
            let one_minus_q0 = -log_q0.exp_m1();
            let p0 = theta.zero_probability.mul_add(one_minus_q0, q0);
            let responsibility = (1.0 - theta.zero_probability) * q0 / p0;
            ZinbComponentMeanSizeZeroProbabilityTheta {
                component_mean: responsibility * gradient.mu,
                size: responsibility * gradient.shape,
                zero_probability: -one_minus_q0 / p0,
            }
        } else {
            ZinbComponentMeanSizeZeroProbabilityTheta {
                component_mean: gradient.mu,
                size: gradient.shape,
                zero_probability: 1.0 / (1.0 - theta.zero_probability),
            }
        }
    }

    pub(super) fn cdf_theta(y: f64, theta: ZinbComponentMeanSizeZeroProbabilityTheta) -> f64 {
        if !y.is_finite()
            || theta.component_mean <= 0.0
            || !theta.component_mean.is_finite()
            || theta.size <= 0.0
            || !theta.size.is_finite()
            || theta.zero_probability <= 0.0
            || theta.zero_probability >= 1.0
            || !theta.zero_probability.is_finite()
        {
            return f64::NAN;
        }
        if y < 0.0 {
            return 0.0;
        }

        let base_cdf = NegativeBinomialKernel::cdf_theta(
            y,
            NegativeBinomialTheta {
                mu: theta.component_mean,
                shape: theta.size,
            },
        );
        (1.0 - theta.zero_probability)
            .mul_add(base_cdf, theta.zero_probability)
            .clamp(0.0, 1.0)
    }

    pub(super) fn quantile_theta(p: f64, theta: ZinbComponentMeanSizeZeroProbabilityTheta) -> f64 {
        if theta.component_mean <= 0.0
            || !theta.component_mean.is_finite()
            || theta.size <= 0.0
            || !theta.size.is_finite()
            || theta.zero_probability <= 0.0
            || theta.zero_probability >= 1.0
            || !theta.zero_probability.is_finite()
        {
            return f64::NAN;
        }

        #[allow(clippy::cast_precision_loss)]
        discrete_quantile(p, MAX_CDF_TERMS, |count| {
            Self::cdf_theta(count as f64, theta)
        })
    }

    #[cfg(feature = "rand")]
    pub(super) fn try_sample_component_theta<Rng>(
        rng: &mut Rng,
        theta: ZinbComponentMeanSizeZeroProbabilityTheta,
    ) -> Result<f64, gamlss_core::SimulationError>
    where
        Rng: rand::Rng,
    {
        if theta.component_mean <= 0.0
            || !theta.component_mean.is_finite()
            || theta.size <= 0.0
            || !theta.size.is_finite()
            || theta.zero_probability <= 0.0
            || theta.zero_probability >= 1.0
            || !theta.zero_probability.is_finite()
        {
            return Err(gamlss_core::SimulationError::InvalidParameters(
                "ZINB theta",
            ));
        }
        if crate::simulation::open_unit(rng) <= theta.zero_probability {
            return Ok(0.0);
        }

        let mixing = rand_distr::Gamma::new(theta.size, theta.component_mean / theta.size)
            .map_err(|_| gamlss_core::SimulationError::BackendRejected("ZINB gamma mixture"))?;
        let lambda = rand_distr::Distribution::sample(&mixing, rng);
        let count = rand_distr::Poisson::new(lambda)
            .map_err(|_| gamlss_core::SimulationError::BackendRejected("ZINB Poisson mean"))?;
        Ok(rand_distr::Distribution::sample(&count, rng))
    }
}

impl<ComponentMeanLink, SizeLink, ZeroProbabilityLink> Default
    for Zinb<ComponentMeanLink, SizeLink, ZeroProbabilityLink>
where
    ComponentMeanLink: PositiveLink<f64>,
    SizeLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Natural-scale component-mean/size/zero-probability ZINB parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZinbComponentMeanSizeZeroProbabilityTheta {
    /// Positive negative-binomial mean.
    pub component_mean: f64,
    /// Positive negative-binomial size.
    pub size: f64,
    /// Zero-inflation probability in `(0, 1)`.
    pub zero_probability: f64,
}
