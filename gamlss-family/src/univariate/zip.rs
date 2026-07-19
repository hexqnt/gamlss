use std::marker::PhantomData;

use gamlss_core::{Log, Logit, ParameterParts, PositiveLink, UnitIntervalLink};

use gamlss_special::{is_nonnegative_integer, ln_gamma, log_add_exp};

use super::poisson::{Poisson, PoissonTheta};

pub use component_mean_zero_probability::{
    ComponentMeanZeroProbability, ZipComponentMeanZeroProbability,
};
pub use total_mean_zero_probability::{
    TotalMeanZeroProbability, ZipTotalMeanZeroProbability, ZipTotalMeanZeroProbabilityEta,
    ZipTotalMeanZeroProbabilityTheta,
};

mod component_mean_zero_probability;
mod total_mean_zero_probability;

const MAX_CDF_TERMS: u64 = 1_000_000;

/// Zero-inflated Poisson family.
///
/// Let $\lambda>0$ be the Poisson component mean and $\pi\in(0,1)$ the structural-zero probability. Then
///
/// $$
/// \Pr(Y=0)=\pi+(1-\pi)e^{-\lambda},
/// $$
///
/// and, for $y\in\\{1,2,\ldots\\}$,
///
/// $$
/// \Pr(Y=y)=(1-\pi)\frac{e^{-\lambda}\lambda^y}{y!}.
/// $$
///
/// Therefore $\mathbb{E}(Y)=(1-\pi)\lambda$. The default parameterization models $\lambda$ and $\pi$ directly; use [`ZipTotalMeanZeroProbability`] to model the unconditional mean instead.
///
/// The component-mean parameterization uses [`ZipComponentMeanZeroProbabilityTheta`]; use [`ZipTotalMeanZeroProbability`] when the first modeled parameter should be the unconditional mean.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/distributions/zip.svg")
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Zip<Param = ComponentMeanZeroProbability, MeanLink = Log, ZeroProbabilityLink = Logit> {
    marker: PhantomData<(Param, MeanLink, ZeroProbabilityLink)>,
}

impl<Param, MeanLink, ZeroProbabilityLink> Zip<Param, MeanLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    /// Creates a stateless zero-inflated Poisson family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn poisson_log_pmf(y: f64, mu: f64) -> f64 {
        y.mul_add(mu.ln(), -mu) - ln_gamma(y + 1.0)
    }

    #[inline]
    pub(super) fn nll_theta(y: f64, theta: ZipComponentMeanZeroProbabilityTheta) -> f64 {
        if !is_nonnegative_integer(y)
            || theta.component_mean <= 0.0
            || !theta.component_mean.is_finite()
            || theta.zero_probability <= 0.0
            || theta.zero_probability >= 1.0
            || !theta.zero_probability.is_finite()
        {
            return f64::INFINITY;
        }
        if y == 0.0 {
            -log_add_exp(
                theta.zero_probability.ln(),
                (1.0 - theta.zero_probability).ln() - theta.component_mean,
            )
        } else {
            -((1.0 - theta.zero_probability).ln() + Self::poisson_log_pmf(y, theta.component_mean))
        }
    }

    #[inline]
    pub(super) fn gradient_component_theta(
        y: f64,
        theta: ZipComponentMeanZeroProbabilityTheta,
    ) -> ZipComponentMeanZeroProbabilityTheta {
        if y == 0.0 {
            let q0 = (-theta.component_mean).exp();
            let one_minus_q0 = -(-theta.component_mean).exp_m1();
            let p0 = theta.zero_probability.mul_add(one_minus_q0, q0);
            ZipComponentMeanZeroProbabilityTheta {
                component_mean: (1.0 - theta.zero_probability) * q0 / p0,
                zero_probability: -one_minus_q0 / p0,
            }
        } else {
            ZipComponentMeanZeroProbabilityTheta {
                component_mean: 1.0 - y / theta.component_mean,
                zero_probability: 1.0 / (1.0 - theta.zero_probability),
            }
        }
    }

    pub(super) fn cdf_theta(y: f64, theta: ZipComponentMeanZeroProbabilityTheta) -> f64 {
        if !y.is_finite()
            || theta.component_mean <= 0.0
            || !theta.component_mean.is_finite()
            || theta.zero_probability <= 0.0
            || theta.zero_probability >= 1.0
            || !theta.zero_probability.is_finite()
        {
            return f64::NAN;
        }
        if y < 0.0 {
            return 0.0;
        }
        let base_cdf = Poisson::<Log>::cdf_theta(
            y,
            PoissonTheta {
                mu: theta.component_mean,
            },
        );
        (1.0 - theta.zero_probability)
            .mul_add(base_cdf, theta.zero_probability)
            .clamp(0.0, 1.0)
    }

    #[cfg(feature = "rand")]
    pub(super) fn try_sample_component_theta<Rng>(
        rng: &mut Rng,
        theta: ZipComponentMeanZeroProbabilityTheta,
    ) -> Result<f64, gamlss_core::SimulationError>
    where
        Rng: rand::Rng,
    {
        if theta.component_mean <= 0.0
            || !theta.component_mean.is_finite()
            || theta.zero_probability <= 0.0
            || theta.zero_probability >= 1.0
            || !theta.zero_probability.is_finite()
        {
            return Err(gamlss_core::SimulationError::InvalidParameters("ZIP theta"));
        }
        if crate::simulation::open_unit(rng) <= theta.zero_probability {
            return Ok(0.0);
        }

        let distribution = rand_distr::Poisson::new(theta.component_mean)
            .map_err(|_| gamlss_core::SimulationError::BackendRejected("ZIP Poisson mean"))?;
        Ok(rand_distr::Distribution::sample(&distribution, rng))
    }
}

impl<Param, MeanLink, ZeroProbabilityLink> Default for Zip<Param, MeanLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Predictors for component-mean ZIP on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZipComponentMeanZeroProbabilityEta {
    /// Poisson mean predictor.
    pub component_mean: f64,
    /// Zero-inflation probability predictor.
    pub zero_probability: f64,
}

impl ParameterParts<2> for ZipComponentMeanZeroProbabilityEta {
    #[inline]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            component_mean: values[0],
            zero_probability: values[1],
        }
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.component_mean,
            1 => self.zero_probability,
            _ => unreachable!("zip eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale component-mean ZIP parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZipComponentMeanZeroProbabilityTheta {
    /// Positive Poisson mean.
    pub component_mean: f64,
    /// Zero-inflation probability in `(0, 1)`.
    pub zero_probability: f64,
}
