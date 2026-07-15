use std::marker::PhantomData;

use gamlss_core::{
    Dispersion, Family, HasCdf, HasQuantile, InitialEtaFromObservations, InitialEtaFromTheta, Log,
    Mean, ObservationView, ParameterParts, PositiveLink,
};

use gamlss_special::{discrete_quantile, is_nonnegative_integer};

use crate::initial::{LARGE_SHAPE, positive_floor, weighted_summary, weighted_values};

use super::{MAX_CDF_TERMS, NegativeBinomial, NegativeBinomialTheta};

/// Negative binomial distribution parameterized by mean and dispersion.
///
/// The variance is `mean + dispersion * mean^2`; this is equivalent to the
/// mean/size parameterization with `size = 1 / dispersion`.
pub type NegativeBinomialMeanDispersion = NegativeBinomialDispersion<Log, Log>;

define_two_positive_parameter_blocks! {
    eta:
    /// Predictors for negative-binomial mean/dispersion on the link scale.
    NegativeBinomialMeanDispersionEta {
        /// Mean predictor.
        mean,
        /// Dispersion predictor.
        dispersion,
    }
    theta:
    /// Natural-scale negative-binomial mean/dispersion parameters.
    NegativeBinomialMeanDispersionTheta {
        /// Positive mean.
        mean,
        /// Positive dispersion.
        dispersion,
    }
}

impl NegativeBinomialMeanDispersionTheta {
    #[inline]
    fn mean_size(self) -> NegativeBinomialTheta {
        NegativeBinomialTheta {
            mu: self.mean,
            shape: 1.0 / self.dispersion,
        }
    }
}

/// Negative-binomial mean/dispersion implementation carrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(
        eta: NegativeBinomialMeanDispersionEta,
    ) -> NegativeBinomialMeanDispersionTheta {
        eta.theta_from_links::<MeanLink, DispersionLink>()
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

gamlss_core::impl_scalar_compilable_family!(
    impl<MeanLink, DispersionLink> for NegativeBinomialDispersion<MeanLink, DispersionLink>;
    parameters = (Mean, Dispersion);
    arity = 2;
);

impl<MeanLink, DispersionLink> Family for NegativeBinomialDispersion<MeanLink, DispersionLink>
where
    MeanLink: PositiveLink<f64>,
    DispersionLink: PositiveLink<f64>,
{
    type Eta = NegativeBinomialMeanDispersionEta;
    type Theta = NegativeBinomialMeanDispersionTheta;
    type GradientEta = NegativeBinomialMeanDispersionEta;
    type Observation<'obs> = f64;
    type Workspace = ();
    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        NegativeBinomial::<Log, Log>::nll_theta(y, theta.mean_size())
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
        let mean_size = theta.mean_size();
        let nll = NegativeBinomial::<Log, Log>::nll_theta(y, mean_size);
        if !nll.is_finite() {
            return (
                nll,
                NegativeBinomialMeanDispersionEta::from_array([f64::NAN; 2]),
            );
        }

        let gradient_theta = NegativeBinomial::<Log, Log>::gradient_theta(y, mean_size);
        let d_dispersion = gradient_theta.shape * (-1.0 / (theta.dispersion * theta.dispersion));

        (
            nll,
            eta.chain_gradient::<MeanLink, DispersionLink>(gradient_theta.mu, d_dispersion),
        )
    }
}

impl<MeanLink, DispersionLink> InitialEtaFromObservations<2>
    for NegativeBinomialDispersion<MeanLink, DispersionLink>
where
    MeanLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    DispersionLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
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
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        NegativeBinomial::<Log, Log>::cdf_theta(y, theta.mean_size())
    }
}

impl<MeanLink, DispersionLink> HasQuantile for NegativeBinomialDispersion<MeanLink, DispersionLink>
where
    MeanLink: PositiveLink<f64>,
    DispersionLink: PositiveLink<f64>,
{
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        let theta = theta.mean_size();
        if theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
        {
            return f64::NAN;
        }

        #[allow(clippy::cast_precision_loss)]
        discrete_quantile(p, MAX_CDF_TERMS, |count| {
            NegativeBinomial::<Log, Log>::cdf_theta(count as f64, theta)
        })
    }
}
