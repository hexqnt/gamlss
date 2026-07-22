use std::marker::PhantomData;

use gamlss_core::{Log, Logit, PositiveLink, UnitIntervalLink};

use crate::domain::{is_positive_finite, is_strict_probability};

pub use component_mean_cv_zero_probability::{
    ZagaComponentMeanCvZeroProbability, ZagaComponentMeanCvZeroProbabilityEta,
};
pub use total_mean_cv::{
    ZagaTotalMeanCvZeroProbability, ZagaTotalMeanCvZeroProbabilityEta,
    ZagaTotalMeanCvZeroProbabilityTheta,
};

mod component_mean_cv_zero_probability;
mod total_mean_cv;

use super::gamma::{GammaKernel, GammaShapeRateTheta};

/// Zero-adjusted gamma family.
///
/// Let $\mu>0$ be the mean of the positive gamma component, let $c=\sqrt{\operatorname{Var}(Y\mid Y>0)}/\mu>0$ be its coefficient of variation, and let $\pi\in(0,1)$ be the zero-mass probability. The mixture is
///
/// $$
/// \Pr(Y=0)=\pi,
/// \qquad
/// f_Y(y)=(1-\pi)f_\Gamma(y\mid\alpha,\beta),\quad y>0,
/// $$
///
/// where
///
/// $$
/// \alpha=c^{-2},
/// \qquad
/// \beta=\frac{1}{\mu c^2}.
/// $$
///
/// Here $f_\Gamma(\\,\cdot\mid\alpha,\beta)$ is the shape/rate gamma density documented by [`crate::Gamma`]. Therefore $\mathbb{E}(Y)=(1-\pi)\mu$. The default parameterization models the component mean, component CV, and zero-mass probability; use [`ZagaTotalMeanCvZeroProbability`] to model the unconditional mean instead.
///
/// The default component parameterization uses [`ZagaComponentMeanCvZeroProbabilityTheta`]; use [`ZagaTotalMeanCvZeroProbability`] when the first modeled parameter should be the unconditional mean.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/distributions/zaga_component_mean_cv.svg")
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Zaga<ComponentMeanLink = Log, CvLink = Log, ZeroProbabilityLink = Logit> {
    marker: PhantomData<(ComponentMeanLink, CvLink, ZeroProbabilityLink)>,
}

impl<ComponentMeanLink, CvLink, ZeroProbabilityLink>
    Zaga<ComponentMeanLink, CvLink, ZeroProbabilityLink>
where
    ComponentMeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    /// Creates a stateless ZAGA family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

/// Link-independent zero-adjusted gamma kernel shared by its parameterizations.
#[derive(Debug, Clone, Copy)]
pub(super) struct ZagaKernel;

impl ZagaKernel {
    #[inline]
    fn valid_theta(theta: ZagaComponentMeanCvZeroProbabilityTheta) -> bool {
        is_positive_finite(theta.component_mean)
            && is_positive_finite(theta.cv)
            && is_strict_probability(theta.zero_probability)
    }

    #[inline]
    fn gamma_shape_rate(theta: ZagaComponentMeanCvZeroProbabilityTheta) -> GammaShapeRateTheta {
        let shape = 1.0 / (theta.cv * theta.cv);
        let rate = 1.0 / (theta.cv * theta.cv * theta.component_mean);
        GammaShapeRateTheta { shape, rate }
    }

    #[inline]
    pub(super) fn nll_theta(y: f64, theta: ZagaComponentMeanCvZeroProbabilityTheta) -> f64 {
        if y < 0.0 || !y.is_finite() || !Self::valid_theta(theta) {
            return f64::INFINITY;
        }
        if y == 0.0 {
            return -theta.zero_probability.ln();
        }
        let shape_rate = Self::gamma_shape_rate(theta);
        -(1.0 - theta.zero_probability).ln() + GammaKernel::nll_shape_rate(y, shape_rate)
    }

    #[inline]
    pub(super) fn gradient_component_theta(
        y: f64,
        theta: ZagaComponentMeanCvZeroProbabilityTheta,
    ) -> ZagaComponentMeanCvZeroProbabilityTheta {
        if y == 0.0 {
            return ZagaComponentMeanCvZeroProbabilityTheta {
                component_mean: 0.0,
                cv: 0.0,
                zero_probability: -1.0 / theta.zero_probability,
            };
        }

        let shape_rate = Self::gamma_shape_rate(theta);
        let (d_shape, d_rate) = GammaKernel::gradient_shape_rate(y, shape_rate);

        #[allow(clippy::suboptimal_flops)]
        ZagaComponentMeanCvZeroProbabilityTheta {
            component_mean: d_rate * (-shape_rate.rate / theta.component_mean),
            cv: d_shape * (-2.0 * shape_rate.shape / theta.cv)
                + d_rate * (-2.0 * shape_rate.rate / theta.cv),
            zero_probability: 1.0 / (1.0 - theta.zero_probability),
        }
    }

    #[allow(clippy::suboptimal_flops)]
    pub(super) fn cdf_theta(y: f64, theta: ZagaComponentMeanCvZeroProbabilityTheta) -> f64 {
        if !y.is_finite() || !Self::valid_theta(theta) {
            return f64::NAN;
        }
        if y < 0.0 {
            return 0.0;
        }
        if y == 0.0 {
            return theta.zero_probability;
        }
        let shape_rate = Self::gamma_shape_rate(theta);
        (theta.zero_probability
            + (1.0 - theta.zero_probability) * GammaKernel::cdf_shape_rate(y, shape_rate))
        .clamp(0.0, 1.0)
    }

    pub(super) fn quantile_theta(p: f64, theta: ZagaComponentMeanCvZeroProbabilityTheta) -> f64 {
        if !(0.0..=1.0).contains(&p) || !Self::valid_theta(theta) {
            return f64::NAN;
        }
        if p <= theta.zero_probability {
            return 0.0;
        }

        let target = (p - theta.zero_probability) / (1.0 - theta.zero_probability);
        GammaKernel::quantile_shape_rate(target, Self::gamma_shape_rate(theta))
    }

    pub(super) fn crps_theta(y: f64, theta: ZagaComponentMeanCvZeroProbabilityTheta) -> f64 {
        if y < 0.0 || !y.is_finite() || !Self::valid_theta(theta) {
            return f64::NAN;
        }
        let gamma = Self::gamma_shape_rate(theta);
        crate::crps::zero_inflated_crps(
            y,
            theta.zero_probability,
            GammaKernel::crps_shape_rate(y, gamma),
            GammaKernel::crps_shape_rate(0.0, gamma),
        )
    }

    #[cfg(feature = "rand")]
    pub(super) fn try_sample_component_theta<Rng>(
        rng: &mut Rng,
        theta: ZagaComponentMeanCvZeroProbabilityTheta,
    ) -> Result<f64, gamlss_core::SimulationError>
    where
        Rng: rand::Rng,
    {
        if !Self::valid_theta(theta) {
            return Err(gamlss_core::SimulationError::InvalidParameters(
                "ZAGA theta",
            ));
        }
        if crate::simulation::open_unit(rng) <= theta.zero_probability {
            return Ok(0.0);
        }

        let shape_rate = Self::gamma_shape_rate(theta);
        let distribution = rand_distr::Gamma::new(shape_rate.shape, 1.0 / shape_rate.rate)
            .map_err(|_| gamlss_core::SimulationError::BackendRejected("ZAGA gamma"))?;
        Ok(rand_distr::Distribution::sample(&distribution, rng))
    }
}

impl<ComponentMeanLink, CvLink, ZeroProbabilityLink> Default
    for Zaga<ComponentMeanLink, CvLink, ZeroProbabilityLink>
where
    ComponentMeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Natural-scale component-mean/CV/zero-probability ZAGA parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZagaComponentMeanCvZeroProbabilityTheta {
    /// Positive mean for the gamma component.
    pub component_mean: f64,
    /// Positive coefficient of variation for the gamma component.
    pub cv: f64,
    /// Zero-mass probability in `(0, 1)`.
    pub zero_probability: f64,
}
