use std::marker::PhantomData;

use gamlss_core::{Log, Logit, ParameterParts, PositiveLink, UnitIntervalLink};

use crate::poisson::{Poisson, PoissonTheta};
use crate::special::{is_nonnegative_integer, ln_gamma, log_add_exp};

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
/// Backward-compatible alias for the component-mean ZIP parameterization.
pub type ZipMeanZeroProbability = ZipComponentMeanZeroProbability;

/// Zero-inflated Poisson family.
#[derive(Debug, Clone, Copy, PartialEq)]
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
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn poisson_log_pmf(y: f64, mu: f64) -> f64 {
        -mu + y * mu.ln() - ln_gamma(y + 1.0)
    }

    #[inline(always)]
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
        (theta.sigma + (1.0 - theta.sigma) * base_cdf).clamp(0.0, 1.0)
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
    pub sigma: f64,
}

impl ParameterParts<2> for ZipEta {
    #[inline(always)]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            mu: values[0],
            sigma: values[1],
        }
    }

    #[inline(always)]
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
    pub sigma: f64,
}
