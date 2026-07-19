use std::marker::PhantomData;

use gamlss_core::{Log, Logit, PositiveLink, UnitIntervalLink};

use gamlss_special::{digamma, ln_gamma, regularized_gamma_lower};

pub use component_mean_cv_zero_probability::{
    ZagaComponentMeanCvZeroProbability, ZagaComponentMeanCvZeroProbabilityEta,
};
pub use total_mean_cv::{
    ZagaTotalMeanCvZeroProbability, ZagaTotalMeanCvZeroProbabilityEta,
    ZagaTotalMeanCvZeroProbabilityTheta,
};

mod component_mean_cv_zero_probability;
mod total_mean_cv;

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

    #[inline]
    fn gamma_shape_rate(theta: ZagaComponentMeanCvZeroProbabilityTheta) -> (f64, f64) {
        let shape = 1.0 / (theta.cv * theta.cv);
        let rate = 1.0 / (theta.cv * theta.cv * theta.component_mean);
        (shape, rate)
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn gamma_nll(y: f64, shape: f64, rate: f64) -> f64 {
        ln_gamma(shape) - shape * rate.ln() - (shape - 1.0) * y.ln() + rate * y
    }

    #[inline]
    pub(super) fn nll_theta(y: f64, theta: ZagaComponentMeanCvZeroProbabilityTheta) -> f64 {
        if y < 0.0
            || !y.is_finite()
            || theta.component_mean <= 0.0
            || !theta.component_mean.is_finite()
            || theta.cv <= 0.0
            || !theta.cv.is_finite()
            || theta.zero_probability <= 0.0
            || theta.zero_probability >= 1.0
            || !theta.zero_probability.is_finite()
        {
            return f64::INFINITY;
        }
        if y == 0.0 {
            return -theta.zero_probability.ln();
        }
        let (shape, rate) = Self::gamma_shape_rate(theta);
        -(1.0 - theta.zero_probability).ln() + Self::gamma_nll(y, shape, rate)
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

        let (shape, rate) = Self::gamma_shape_rate(theta);
        let d_shape = digamma(shape) - rate.ln() - y.ln();
        let d_rate = y - shape / rate;

        #[allow(clippy::suboptimal_flops)]
        ZagaComponentMeanCvZeroProbabilityTheta {
            component_mean: d_rate * (-rate / theta.component_mean),
            cv: d_shape * (-2.0 * shape / theta.cv) + d_rate * (-2.0 * rate / theta.cv),
            zero_probability: 1.0 / (1.0 - theta.zero_probability),
        }
    }

    #[allow(clippy::suboptimal_flops)]
    pub(super) fn cdf_theta(y: f64, theta: ZagaComponentMeanCvZeroProbabilityTheta) -> f64 {
        if !y.is_finite()
            || theta.component_mean <= 0.0
            || !theta.component_mean.is_finite()
            || theta.cv <= 0.0
            || !theta.cv.is_finite()
            || theta.zero_probability <= 0.0
            || theta.zero_probability >= 1.0
            || !theta.zero_probability.is_finite()
        {
            return f64::NAN;
        }
        if y < 0.0 {
            return 0.0;
        }
        if y == 0.0 {
            return theta.zero_probability;
        }
        let (shape, rate) = Self::gamma_shape_rate(theta);
        (theta.zero_probability
            + (1.0 - theta.zero_probability) * regularized_gamma_lower(shape, rate * y))
        .clamp(0.0, 1.0)
    }

    #[cfg(feature = "rand")]
    pub(super) fn try_sample_component_theta<Rng>(
        rng: &mut Rng,
        theta: ZagaComponentMeanCvZeroProbabilityTheta,
    ) -> Result<f64, gamlss_core::SimulationError>
    where
        Rng: rand::Rng,
    {
        if theta.component_mean <= 0.0
            || !theta.component_mean.is_finite()
            || theta.cv <= 0.0
            || !theta.cv.is_finite()
            || theta.zero_probability <= 0.0
            || theta.zero_probability >= 1.0
            || !theta.zero_probability.is_finite()
        {
            return Err(gamlss_core::SimulationError::InvalidParameters(
                "ZAGA theta",
            ));
        }
        if crate::simulation::open_unit(rng) <= theta.zero_probability {
            return Ok(0.0);
        }

        let (shape, rate) = Self::gamma_shape_rate(theta);
        let distribution = rand_distr::Gamma::new(shape, 1.0 / rate)
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
