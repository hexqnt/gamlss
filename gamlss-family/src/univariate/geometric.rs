use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasQuantile, InitialEtaFromObservations, InitialEtaFromTheta, Log, Mean,
    ObservationView, ParameterParts, PositiveLink,
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};
use gamlss_special::is_nonnegative_integer;

use crate::domain::is_probability;
use crate::initial::{positive_floor, weighted_values};

/// Geometric distribution parameterized by its mean with a log link.
pub type GeometricMean = Geometric<Log>;

/// Geometric family for the number of failures before the first success.
///
/// The natural parameter is the positive mean $\mu=(1-p)/p$, where $p$ is the
/// success probability. Its support is $0,1,2,\ldots$.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/distributions/geometric.svg")
)]
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometric<MeanLink = Log> {
    marker: PhantomData<MeanLink>,
}

impl<MeanLink> Geometric<MeanLink>
where
    MeanLink: PositiveLink<f64>,
{
    /// Creates a stateless geometric family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(eta: GeometricEta) -> GeometricTheta {
        GeometricTheta {
            mean: MeanLink::inverse(eta.mean),
        }
    }

    #[inline]
    fn valid_theta(theta: GeometricTheta) -> bool {
        theta.mean > 0.0 && theta.mean.is_finite()
    }

    #[inline]
    fn log_one_plus_inverse_mean(mean: f64) -> f64 {
        if mean >= 1.0 {
            (1.0 / mean).ln_1p()
        } else {
            mean.ln_1p() - mean.ln()
        }
    }

    #[inline]
    fn success_probability(mean: f64) -> f64 {
        (-mean.ln_1p()).exp()
    }

    #[inline]
    fn nll_theta(y: f64, theta: GeometricTheta) -> f64 {
        if !is_nonnegative_integer(y) || !Self::valid_theta(theta) {
            return f64::INFINITY;
        }
        y.mul_add(
            Self::log_one_plus_inverse_mean(theta.mean),
            theta.mean.ln_1p(),
        )
    }

    #[inline]
    fn nll_and_gradient_eta_values(y: f64, eta: GeometricEta) -> (f64, GeometricEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (nll, GeometricEta { mean: f64::NAN });
        }
        let d_mean = Self::success_probability(theta.mean) * (1.0 - y / theta.mean);
        (
            nll,
            GeometricEta {
                mean: d_mean * MeanLink::derivative_inverse(eta.mean),
            },
        )
    }
}

impl<MeanLink> Default for Geometric<MeanLink>
where
    MeanLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MeanLink> for Geometric<MeanLink>;
    parameters = (Mean,);
    arity = 1;
);

impl<MeanLink> Family for Geometric<MeanLink>
where
    MeanLink: PositiveLink<f64>,
{
    type Eta = GeometricEta;
    type Theta = GeometricTheta;
    type GradientEta = GeometricEta;
    type Observation<'obs> = f64;
    type Workspace = ();

    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut ()) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut ()) -> f64 {
        Self::nll_theta(y, *theta)
    }

    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut (),
    ) -> (f64, Self::GradientEta) {
        Self::nll_and_gradient_eta_values(y, *eta)
    }
}

impl<MeanLink> InitialEtaFromObservations<1> for Geometric<MeanLink>
where
    MeanLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = f64> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| is_nonnegative_integer(y).then_some(y));
        let mean = crate::initial::weighted_mean(&values).map_or(1.0, positive_floor);
        GeometricEta {
            mean: MeanLink::initial_eta_from_theta(mean),
        }
    }
}

impl<MeanLink> HasCdf for Geometric<MeanLink>
where
    MeanLink: PositiveLink<f64>,
{
    fn cdf(&self, y: f64, theta: &Self::Theta) -> f64 {
        if !y.is_finite() || !Self::valid_theta(*theta) {
            return f64::NAN;
        }
        if y < 0.0 {
            return 0.0;
        }
        let log_failure_probability = -Self::log_one_plus_inverse_mean(theta.mean);
        -((y.floor() + 1.0) * log_failure_probability).exp_m1()
    }
}

impl<MeanLink> HasQuantile for Geometric<MeanLink>
where
    MeanLink: PositiveLink<f64>,
{
    #[allow(clippy::float_cmp)]
    fn quantile(&self, probability: f64, theta: &Self::Theta) -> f64 {
        if !is_probability(probability) || !Self::valid_theta(*theta) {
            return f64::NAN;
        }
        if probability == 1.0 {
            return f64::INFINITY;
        }
        if probability == 0.0 {
            return 0.0;
        }
        let log_failure_probability = -Self::log_one_plus_inverse_mean(theta.mean);
        let mut candidate =
            (((-probability).ln_1p() / log_failure_probability).ceil() - 1.0).max(0.0);
        loop {
            if candidate <= 0.0 {
                break;
            }
            let previous = candidate - 1.0;
            if previous == candidate || self.cdf(previous, theta) < probability {
                break;
            }
            candidate = previous;
        }
        loop {
            if self.cdf(candidate, theta) >= probability {
                break;
            }
            let next = candidate + 1.0;
            if next == candidate {
                break;
            }
            candidate = next;
        }
        candidate
    }
}

#[cfg(feature = "rand")]
impl<Rng, MeanLink> TrySimulate<Rng> for Geometric<MeanLink>
where
    Rng: rand::Rng,
    MeanLink: PositiveLink<f64>,
{
    type Sample = f64;

    #[allow(clippy::cast_precision_loss)]
    fn try_sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Result<f64, SimulationError> {
        if !Self::valid_theta(*theta) {
            return Err(SimulationError::InvalidParameters("Geometric theta"));
        }
        let probability = Self::success_probability(theta.mean);
        let distribution = rand_distr::Geometric::new(probability)
            .map_err(|_| SimulationError::BackendRejected("Geometric mean"))?;
        Ok(rand_distr::Distribution::sample(&distribution, rng) as f64)
    }
}

/// Link-scale predictor for [`Geometric`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometricEta {
    /// Mean predictor.
    pub mean: f64,
}

impl ParameterParts<1> for GeometricEta {
    fn from_array(values: [f64; 1]) -> Self {
        Self { mean: values[0] }
    }

    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mean,
            _ => unreachable!("geometric eta only has index 0"),
        }
    }
}

/// Natural-scale parameter for [`Geometric`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometricTheta {
    /// Positive mean number of failures.
    pub mean: f64,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use approx::assert_relative_eq;
    use gamlss_core::{Family, HasCdf, HasQuantile};

    use super::{GeometricMean, GeometricTheta};
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn gradient_matches_finite_difference() {
        assert_gradient_matches_finite_difference::<_, 1>(&GeometricMean::new(), 3.0, [0.4]);
    }

    #[test]
    fn cdf_and_quantile_are_consistent() {
        let family = GeometricMean::new();
        let theta = GeometricTheta { mean: 2.0 };
        assert_relative_eq!(family.cdf(0.0, &theta), 1.0 / 3.0, epsilon = 1.0e-14);
        assert_eq!(family.quantile(1.0 / 3.0, &theta), 0.0);
        assert_eq!(family.quantile(0.8, &theta), 3.0);
        assert!(family.nll(0.5, &theta, &mut ()).is_infinite());
    }

    #[test]
    fn large_mean_preserves_likelihood_cdf_and_quantile() {
        let family = GeometricMean::new();
        let mean = 1.0e16;
        let theta = GeometricTheta { mean };
        let expected_nll = mean.ln_1p() + mean * (1.0 / mean).ln_1p();
        assert_relative_eq!(
            family.nll(mean, &theta, &mut ()),
            expected_nll,
            epsilon = 1.0e-14
        );
        assert_relative_eq!(
            family.cdf(0.0, &theta),
            (-mean.ln_1p()).exp(),
            max_relative = 2.0e-15
        );
        let quantile = family.quantile(0.75, &theta);
        assert!(quantile.is_finite());
        assert!(family.cdf(quantile, &theta) >= 0.75);
    }
}
