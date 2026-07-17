use std::marker::PhantomData;

use gamlss_core::{Log, Logit, ParameterParts, PositiveLink, UnitIntervalLink};

use gamlss_special::{is_nonnegative_integer, ln_gamma, log_add_exp};

use super::poisson::{Poisson, PoissonTheta};

pub use component_mean_zero_probability::{
    ComponentMeanZeroProbability, ZipComponentMeanZeroProbability,
};
pub use total_mean_zero_probability::{
    TotalMeanZeroProbability, ZipTotalMeanZeroProbability, ZipTotalMeanZeroProbabilityTheta,
};

mod component_mean_zero_probability;
mod total_mean_zero_probability;

const MAX_CDF_TERMS: u64 = 1_000_000;

/// ZIP distribution with log/logit links.
///
/// Backward-compatible alias for the component-mean ZIP parameterization.
pub type ZipMeanZeroProbability = ZipComponentMeanZeroProbability;

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
/// The canonical carrier retains historical field names: [`ZipTheta::mu`] stores the Poisson component mean $\lambda$, [`ZipTheta::sigma`] stores the zero probability $\pi$, and [`ZipEta`] uses `mu` and `sigma` for their predictors.
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
    pub(super) fn nll_theta(y: f64, theta: ZipTheta) -> f64 {
        if !is_nonnegative_integer(y)
            || theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || theta.sigma >= 1.0
            || !theta.sigma.is_finite()
        {
            return f64::INFINITY;
        }
        if y == 0.0 {
            -log_add_exp(theta.sigma.ln(), (1.0 - theta.sigma).ln() - theta.mu)
        } else {
            -((1.0 - theta.sigma).ln() + Self::poisson_log_pmf(y, theta.mu))
        }
    }

    #[inline]
    pub(super) fn gradient_component_theta(y: f64, theta: ZipTheta) -> ZipTheta {
        if y == 0.0 {
            let q0 = (-theta.mu).exp();
            let one_minus_q0 = -(-theta.mu).exp_m1();
            let p0 = theta.sigma.mul_add(one_minus_q0, q0);
            ZipTheta {
                mu: (1.0 - theta.sigma) * q0 / p0,
                sigma: -one_minus_q0 / p0,
            }
        } else {
            ZipTheta {
                mu: 1.0 - y / theta.mu,
                sigma: 1.0 / (1.0 - theta.sigma),
            }
        }
    }

    pub(super) fn cdf_theta(y: f64, theta: ZipTheta) -> f64 {
        if !y.is_finite()
            || theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || theta.sigma >= 1.0
            || !theta.sigma.is_finite()
        {
            return f64::NAN;
        }
        if y < 0.0 {
            return 0.0;
        }
        let base_cdf = Poisson::<Log>::cdf_theta(y, PoissonTheta { mu: theta.mu });
        (1.0 - theta.sigma)
            .mul_add(base_cdf, theta.sigma)
            .clamp(0.0, 1.0)
    }

    #[cfg(feature = "rand")]
    pub(super) fn sample_component_theta<Rng>(rng: &mut Rng, theta: ZipTheta) -> f64
    where
        Rng: rand::Rng,
    {
        if theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || theta.sigma >= 1.0
            || !theta.sigma.is_finite()
        {
            return f64::NAN;
        }
        if crate::simulation::open_unit(rng) <= theta.sigma {
            return 0.0;
        }

        rand_distr::Distribution::sample(
            &rand_distr::Poisson::new(theta.mu).expect("validated ZIP mean must construct"),
            rng,
        )
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

/// Predictors for ZIP on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZipEta {
    /// Poisson mean predictor.
    pub mu: f64,
    /// Zero-inflation probability predictor.
    ///
    /// The field name is retained for compatibility with existing code.
    pub sigma: f64,
}

impl ParameterParts<2> for ZipEta {
    #[inline]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            mu: values[0],
            sigma: values[1],
        }
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mu,
            1 => self.sigma,
            _ => unreachable!("zip eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale ZIP parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZipTheta {
    /// Positive Poisson mean.
    pub mu: f64,
    /// Zero-inflation probability in `(0, 1)`.
    ///
    /// The field name is retained for compatibility with existing code.
    pub sigma: f64,
}
