use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Family, HasCdf, HasQuantile, InitialEtaFromObservations, InitialEtaFromTheta, Log, Logit,
    ObservationView, ParameterParts, PositiveLink, Size, TotalMean, UnitIntervalLink,
    ZeroProbability,
};

use gamlss_special::{discrete_quantile, is_nonnegative_integer};

use crate::domain::{is_positive_finite, is_strict_probability};
use crate::initial::{
    LARGE_SHAPE, positive_floor, probability_floor, weighted_summary, weighted_values,
};

use super::{MAX_CDF_TERMS, Zinb, ZinbTheta};

/// ZINB distribution parameterized by unconditional mean $m$, NB size $r$, and structural-zero probability $\pi$.
///
/// The negative-binomial component mean is derived as
///
/// $$
/// \mu=\frac{m}{1-\pi},
/// \qquad
/// \mathbb{E}(Y)=m.
/// $$
///
/// The default links are $m=\exp(\eta_m)$, $r=\exp(\eta_r)$, and $\pi=\operatorname{logit}^{-1}(\eta_\pi)$.
///
/// These symbols correspond to the `total_mean`, `size`, and `zero_probability` fields of [`ZinbTotalMeanSizeZeroProbabilityTheta`] and [`ZinbTotalMeanSizeZeroProbabilityEta`].
#[allow(clippy::doc_markdown)]
pub type ZinbTotalMeanSizeZeroProbability = ZinbTotalMeanSize<Log, Log, Logit>;

/// Predictors for ZINB total-mean/size/zero-probability on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZinbTotalMeanSizeZeroProbabilityEta {
    /// Total mean predictor.
    pub total_mean: f64,
    /// Negative-binomial size predictor.
    pub size: f64,
    /// Zero-inflation probability predictor.
    pub zero_probability: f64,
}

impl ParameterParts<3> for ZinbTotalMeanSizeZeroProbabilityEta {
    #[inline]
    fn from_array(values: [f64; 3]) -> Self {
        Self {
            total_mean: values[0],
            size: values[1],
            zero_probability: values[2],
        }
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.total_mean,
            1 => self.size,
            2 => self.zero_probability,
            _ => unreachable!("ZINB total-mean/size eta only has indices 0 through 2"),
        }
    }
}

/// Natural-scale ZINB total-mean/size/zero-probability parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZinbTotalMeanSizeZeroProbabilityTheta {
    /// Positive unconditional mean.
    pub total_mean: f64,
    /// Positive negative-binomial size.
    pub size: f64,
    /// Zero-inflation probability in `(0, 1)`.
    pub zero_probability: f64,
}

impl ZinbTotalMeanSizeZeroProbabilityTheta {
    #[inline]
    fn component(self) -> ZinbTheta {
        ZinbTheta {
            mu: self.total_mean / (1.0 - self.zero_probability),
            shape: self.size,
            nu: self.zero_probability,
        }
    }
}

/// ZINB total-mean/size/zero-probability implementation carrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZinbTotalMeanSize<MeanLink = Log, SizeLink = Log, ZeroProbabilityLink = Logit> {
    marker: PhantomData<(MeanLink, SizeLink, ZeroProbabilityLink)>,
}

impl<MeanLink, SizeLink, ZeroProbabilityLink>
    ZinbTotalMeanSize<MeanLink, SizeLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    SizeLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    /// Creates a stateless ZINB total-mean/size family.
    #[inline]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(
        eta: ZinbTotalMeanSizeZeroProbabilityEta,
    ) -> ZinbTotalMeanSizeZeroProbabilityTheta {
        ZinbTotalMeanSizeZeroProbabilityTheta {
            total_mean: MeanLink::inverse(eta.total_mean),
            size: SizeLink::inverse(eta.size),
            zero_probability: ZeroProbabilityLink::inverse(eta.zero_probability),
        }
    }
}

impl<MeanLink, SizeLink, ZeroProbabilityLink> Default
    for ZinbTotalMeanSize<MeanLink, SizeLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    SizeLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MeanLink, SizeLink, ZeroProbabilityLink> for ZinbTotalMeanSize<MeanLink, SizeLink, ZeroProbabilityLink>;
    parameters = (TotalMean, Size, ZeroProbability);
    arity = 3;
);

impl<MeanLink, SizeLink, ZeroProbabilityLink> Family
    for ZinbTotalMeanSize<MeanLink, SizeLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    SizeLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    type Eta = ZinbTotalMeanSizeZeroProbabilityEta;
    type Theta = ZinbTotalMeanSizeZeroProbabilityTheta;
    type GradientEta = ZinbTotalMeanSizeZeroProbabilityEta;
    type Observation<'obs> = f64;
    type Workspace = ();
    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        Zinb::<Log, Log, Logit>::nll_theta(y, theta.component())
    }

    fn nll_eta(&self, y: f64, eta: &Self::Eta, workspace: &mut Self::Workspace) -> f64 {
        let theta = Self::theta_from_eta(*eta);
        self.nll(y, &theta, workspace)
    }

    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        let theta = Self::theta_from_eta(*eta);
        let component = theta.component();
        let nll = Zinb::<Log, Log, Logit>::nll_theta(y, component);
        if !nll.is_finite() {
            return (
                nll,
                ZinbTotalMeanSizeZeroProbabilityEta::from_array([f64::NAN; 3]),
            );
        }

        let component_gradient = Zinb::<Log, Log, Logit>::gradient_component_theta(y, component);
        let one_minus_zero = 1.0 - theta.zero_probability;
        let d_total_mean = component_gradient.mu / one_minus_zero;
        let d_zero_probability = component_gradient.mu * theta.total_mean
            / (one_minus_zero * one_minus_zero)
            + component_gradient.nu;
        (
            nll,
            ZinbTotalMeanSizeZeroProbabilityEta {
                total_mean: d_total_mean * MeanLink::derivative_inverse(eta.total_mean),
                size: component_gradient.shape * SizeLink::derivative_inverse(eta.size),
                zero_probability: d_zero_probability
                    * ZeroProbabilityLink::derivative_inverse(eta.zero_probability),
            },
        )
    }
}

impl<MeanLink, SizeLink, ZeroProbabilityLink> InitialEtaFromObservations<3>
    for ZinbTotalMeanSize<MeanLink, SizeLink, ZeroProbabilityLink>
where
    MeanLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    SizeLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    ZeroProbabilityLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| is_nonnegative_integer(y).then_some(y));
        let Some(summary) = weighted_summary(&values) else {
            return ZinbTotalMeanSizeZeroProbabilityEta::from_array([0.0, 0.0, 0.0]);
        };
        let total_mean = positive_floor(summary.mean);
        let size = if summary.variance <= total_mean {
            LARGE_SHAPE
        } else {
            positive_floor(total_mean * total_mean / (summary.variance - total_mean))
        };
        let zero_weight = values
            .iter()
            .filter(|(y, _)| *y == 0.0)
            .map(|(_, w)| *w)
            .sum::<f64>();
        let total_weight = values.iter().map(|(_, w)| *w).sum::<f64>();
        let zero_probability = if total_weight > 0.0 {
            zero_weight / total_weight
        } else {
            0.1
        };

        ZinbTotalMeanSizeZeroProbabilityEta {
            total_mean: MeanLink::initial_eta_from_theta(total_mean),
            size: SizeLink::initial_eta_from_theta(size),
            zero_probability: ZeroProbabilityLink::initial_eta_from_theta(probability_floor(
                zero_probability,
            )),
        }
    }
}

impl<MeanLink, SizeLink, ZeroProbabilityLink> HasCdf
    for ZinbTotalMeanSize<MeanLink, SizeLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    SizeLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        Zinb::<Log, Log, Logit>::cdf_theta(y, theta.component())
    }
}

impl<MeanLink, SizeLink, ZeroProbabilityLink> HasQuantile
    for ZinbTotalMeanSize<MeanLink, SizeLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    SizeLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        let theta = theta.component();
        if !is_positive_finite(theta.mu)
            || !is_positive_finite(theta.shape)
            || !is_strict_probability(theta.nu)
        {
            return f64::NAN;
        }

        #[allow(clippy::cast_precision_loss)]
        discrete_quantile(p, MAX_CDF_TERMS, |count| {
            Zinb::<Log, Log, Logit>::cdf_theta(count as f64, theta)
        })
    }
}

#[cfg(feature = "rand")]
impl<Rng, MeanLink, SizeLink, ZeroProbabilityLink> CanSimulate<Rng>
    for ZinbTotalMeanSize<MeanLink, SizeLink, ZeroProbabilityLink>
where
    Rng: rand::Rng,
    MeanLink: PositiveLink<f64>,
    SizeLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    type Sample = f64;

    fn sample(&self, rng: &mut Rng, theta: &Self::Theta) -> f64 {
        Zinb::<Log, Log, Logit>::sample_component_theta(rng, theta.component())
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;

    #[cfg(feature = "rand")]
    use super::{ZinbTotalMeanSizeZeroProbability, ZinbTotalMeanSizeZeroProbabilityTheta};

    #[cfg(feature = "rand")]
    #[test]
    fn total_mean_zinb_sampling_returns_counts_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = ZinbTotalMeanSizeZeroProbability::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family.sample(
            &mut rng,
            &ZinbTotalMeanSizeZeroProbabilityTheta {
                total_mean: 2.0,
                size: 1.5,
                zero_probability: 0.3,
            },
        );
        assert!(sample >= 0.0 && sample.fract() == 0.0);
        assert!(
            family
                .sample(
                    &mut rng,
                    &ZinbTotalMeanSizeZeroProbabilityTheta {
                        total_mean: 2.0,
                        size: 0.0,
                        zero_probability: 0.3,
                    }
                )
                .is_nan()
        );
    }
}
