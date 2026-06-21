use std::marker::PhantomData;

use gamlss_core::{
    Dispersion, Family, HasCdf, HasQuantile, InitialEtaFromTheta, Log, Mean, ObservationView,
    ParameterParts, ParameterizedFamily, PositiveLink,
};

use crate::initial::{LARGE_SHAPE, positive_floor, weighted_summary, weighted_values};
use crate::numeric::finite_difference_gradient_eta;
use crate::special::{discrete_quantile, is_nonnegative_integer};

use super::{MAX_CDF_TERMS, NegativeBinomial, NegativeBinomialTheta};

/// Negative binomial distribution parameterized by mean and dispersion.
///
/// The variance is `mean + dispersion * mean^2`; this is equivalent to the
/// mean/size parameterization with `size = 1 / dispersion`.
pub type NegativeBinomialMeanDispersion = NegativeBinomialDispersion<Log, Log>;

/// Predictors for negative-binomial mean/dispersion on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NegativeBinomialMeanDispersionEta {
    /// Mean predictor.
    pub mean: f64,
    /// Dispersion predictor.
    pub dispersion: f64,
}

impl ParameterParts<2> for NegativeBinomialMeanDispersionEta {
    #[inline(always)]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            mean: values[0],
            dispersion: values[1],
        }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mean,
            1 => self.dispersion,
            _ => unreachable!("negative binomial mean/dispersion eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale negative-binomial mean/dispersion parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NegativeBinomialMeanDispersionTheta {
    /// Positive mean.
    pub mean: f64,
    /// Positive dispersion.
    pub dispersion: f64,
}

impl NegativeBinomialMeanDispersionTheta {
    #[inline(always)]
    fn mean_size(self) -> NegativeBinomialTheta {
        NegativeBinomialTheta {
            mu: self.mean,
            shape: 1.0 / self.dispersion,
        }
    }
}

/// Negative-binomial mean/dispersion implementation carrier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NegativeBinomialDispersion<MeanLink = Log, DispersionLink = Log> {
    marker: PhantomData<(MeanLink, DispersionLink)>,
}

impl<MeanLink, DispersionLink> NegativeBinomialDispersion<MeanLink, DispersionLink>
where
    MeanLink: PositiveLink<f64>,
    DispersionLink: PositiveLink<f64>,
{
    /// Creates a stateless negative-binomial mean/dispersion family.
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn theta_from_eta(
        eta: NegativeBinomialMeanDispersionEta,
    ) -> NegativeBinomialMeanDispersionTheta {
        NegativeBinomialMeanDispersionTheta {
            mean: MeanLink::inverse(eta.mean),
            dispersion: DispersionLink::inverse(eta.dispersion),
        }
    }
}

impl<MeanLink, DispersionLink> Default for NegativeBinomialDispersion<MeanLink, DispersionLink>
where
    MeanLink: PositiveLink<f64>,
    DispersionLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<MeanLink, DispersionLink> Family for NegativeBinomialDispersion<MeanLink, DispersionLink>
where
    MeanLink: PositiveLink<f64>,
    DispersionLink: PositiveLink<f64>,
{
    type Eta = NegativeBinomialMeanDispersionEta;
    type Theta = NegativeBinomialMeanDispersionTheta;
    type NllGradientEta = NegativeBinomialMeanDispersionEta;
    type Observation<'obs> = f64;

    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        NegativeBinomial::<Log, Log>::nll_theta(y, theta.mean_size())
    }

    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        self.nll(y, Self::theta_from_eta(eta))
    }

    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        let nll = self.nll_eta(y, eta);
        if !nll.is_finite() {
            return (
                nll,
                NegativeBinomialMeanDispersionEta::from_array([f64::NAN; 2]),
            );
        }

        let gradient = finite_difference_gradient_eta::<_, NegativeBinomialMeanDispersionEta, 2>(
            eta,
            |probe| self.nll_eta(y, probe),
        );
        (nll, NegativeBinomialMeanDispersionEta::from_array(gradient))
    }
}

impl<MeanLink, DispersionLink> ParameterizedFamily<2>
    for NegativeBinomialDispersion<MeanLink, DispersionLink>
where
    MeanLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    DispersionLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Params = (Mean, Dispersion);
    type Links = (MeanLink, DispersionLink);

    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| is_nonnegative_integer(y).then_some(y));
        let Some(summary) = weighted_summary(&values) else {
            return NegativeBinomialMeanDispersionEta::from_array([0.0, 0.0]);
        };

        let mean = positive_floor(summary.mean);
        let dispersion = if summary.variance <= mean {
            positive_floor(1.0 / LARGE_SHAPE)
        } else {
            positive_floor((summary.variance - mean) / (mean * mean))
        };

        NegativeBinomialMeanDispersionEta {
            mean: MeanLink::initial_eta_from_theta(mean),
            dispersion: DispersionLink::initial_eta_from_theta(dispersion),
        }
    }
}

impl<MeanLink, DispersionLink> HasCdf for NegativeBinomialDispersion<MeanLink, DispersionLink>
where
    MeanLink: PositiveLink<f64>,
    DispersionLink: PositiveLink<f64>,
{
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
        NegativeBinomial::<Log, Log>::cdf_theta(y, theta.mean_size())
    }
}

impl<MeanLink, DispersionLink> HasQuantile for NegativeBinomialDispersion<MeanLink, DispersionLink>
where
    MeanLink: PositiveLink<f64>,
    DispersionLink: PositiveLink<f64>,
{
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
        let theta = theta.mean_size();
        if theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
        {
            return f64::NAN;
        }

        discrete_quantile(p, MAX_CDF_TERMS, |count| {
            NegativeBinomial::<Log, Log>::cdf_theta(count as f64, theta)
        })
    }
}
