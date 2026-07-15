use std::marker::PhantomData;

use gamlss_core::{Log, Logit, PositiveLink, UnitIntervalLink};

use gamlss_special::{digamma, ln_gamma, regularized_gamma_lower};

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
/// The natural-scale carrier retains historical field names: [`ZagaTheta::mu`] is $\mu$, [`ZagaTheta::sigma`] is the component CV $c$, and [`ZagaTheta::nu`] is the zero probability $\pi$. [`ZagaEta`] uses the same field-to-symbol mapping for their predictors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn gamma_shape_rate(theta: ZagaTheta) -> (f64, f64) {
        let shape = 1.0 / (theta.sigma * theta.sigma);
        let rate = 1.0 / (theta.sigma * theta.sigma * theta.mu);
        (shape, rate)
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn gamma_nll(y: f64, shape: f64, rate: f64) -> f64 {
        ln_gamma(shape) - shape * rate.ln() - (shape - 1.0) * y.ln() + rate * y
    }

    #[inline]
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

    #[inline]
    pub(super) fn gradient_component_theta(y: f64, theta: ZagaTheta) -> ZagaTheta {
        if y == 0.0 {
            return ZagaTheta {
                mu: 0.0,
                sigma: 0.0,
                nu: -1.0 / theta.nu,
            };
        }

        let (shape, rate) = Self::gamma_shape_rate(theta);
        let d_shape = digamma(shape) - rate.ln() - y.ln();
        let d_rate = y - shape / rate;

        #[allow(clippy::suboptimal_flops)]
        ZagaTheta {
            mu: d_rate * (-rate / theta.mu),
            sigma: d_shape * (-2.0 * shape / theta.sigma) + d_rate * (-2.0 * rate / theta.sigma),
            nu: 1.0 / (1.0 - theta.nu),
        }
    }

    #[allow(clippy::suboptimal_flops)]
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

    #[cfg(feature = "rand")]
    pub(super) fn sample_component_theta<Rng>(rng: &mut Rng, theta: ZagaTheta) -> f64
    where
        Rng: rand::Rng,
    {
        if theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || theta.nu <= 0.0
            || theta.nu >= 1.0
            || !theta.nu.is_finite()
        {
            return f64::NAN;
        }
        if crate::simulation::open_unit(rng) <= theta.nu {
            return 0.0;
        }

        let (shape, rate) = Self::gamma_shape_rate(theta);
        rand_distr::Distribution::sample(
            &rand_distr::Gamma::new(shape, 1.0 / rate)
                .expect("validated ZAGA gamma parameters must construct"),
            rng,
        )
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
