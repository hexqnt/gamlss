use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Family, HasCdf, HasCrps, HasQuantile, InitialEtaFromTheta, Log, Mean, ObservationView,
    ParameterParts, ParameterizedFamily, PositiveLink, Rate,
};

use crate::domain::{is_positive_finite, is_probability};
use crate::initial::{positive_floor, weighted_mean, weighted_values};

/// Exponential mean parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeanParam;

/// Exponential rate parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RateParam;

/// Exponential family implementation carrier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exponential<Param = RateParam, Link = Log> {
    marker: PhantomData<(Param, Link)>,
}

impl<Param, Link> Exponential<Param, Link> {
    /// Creates a stateless exponential family.
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn valid_rate(theta: ExponentialRateTheta) -> bool {
        is_positive_finite(theta.rate)
    }

    #[inline(always)]
    fn nll_rate(y: f64, theta: ExponentialRateTheta) -> f64 {
        if y < 0.0 || !y.is_finite() || !Self::valid_rate(theta) {
            return f64::INFINITY;
        }

        -theta.rate.ln() + theta.rate * y
    }

    #[inline(always)]
    fn cdf_rate(y: f64, theta: ExponentialRateTheta) -> f64 {
        if !y.is_finite() || !Self::valid_rate(theta) {
            return f64::NAN;
        }
        if y < 0.0 {
            return 0.0;
        }

        -(-theta.rate * y).exp_m1()
    }

    #[inline(always)]
    fn quantile_rate(p: f64, theta: ExponentialRateTheta) -> f64 {
        if !is_probability(p) || !Self::valid_rate(theta) {
            return f64::NAN;
        }

        -(-p).ln_1p() / theta.rate
    }

    #[inline(always)]
    fn crps_rate(y: f64, theta: ExponentialRateTheta) -> f64 {
        if y < 0.0 || !y.is_finite() || !Self::valid_rate(theta) {
            return f64::NAN;
        }

        y + 2.0 * (-theta.rate * y).exp() / theta.rate - 1.5 / theta.rate
    }
}

impl<Param, Link> Default for Exponential<Param, Link> {
    fn default() -> Self {
        Self::new()
    }
}

/// Predictor for exponential mean on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExponentialMeanEta {
    /// Mean predictor.
    pub mean: f64,
}

impl ParameterParts<1> for ExponentialMeanEta {
    #[inline(always)]
    fn from_array(values: [f64; 1]) -> Self {
        Self { mean: values[0] }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mean,
            _ => unreachable!("exponential mean eta only has index 0"),
        }
    }
}

/// Natural-scale exponential mean parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExponentialMeanTheta {
    /// Positive mean.
    pub mean: f64,
}

impl ExponentialMeanTheta {
    #[inline(always)]
    fn rate(self) -> ExponentialRateTheta {
        ExponentialRateTheta {
            rate: 1.0 / self.mean,
        }
    }
}

/// Predictor for exponential rate on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExponentialRateEta {
    /// Rate predictor.
    pub rate: f64,
}

impl ParameterParts<1> for ExponentialRateEta {
    #[inline(always)]
    fn from_array(values: [f64; 1]) -> Self {
        Self { rate: values[0] }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.rate,
            _ => unreachable!("exponential rate eta only has index 0"),
        }
    }
}

/// Natural-scale exponential rate parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExponentialRateTheta {
    /// Positive rate.
    pub rate: f64,
}

impl<Link> Exponential<MeanParam, Link>
where
    Link: PositiveLink<f64>,
{
    #[inline(always)]
    fn theta_from_eta(eta: ExponentialMeanEta) -> ExponentialMeanTheta {
        ExponentialMeanTheta {
            mean: Link::inverse(eta.mean),
        }
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: ExponentialMeanEta) -> (f64, ExponentialMeanEta) {
        let theta = Self::theta_from_eta(eta);
        let rate = theta.rate();
        let nll = Self::nll_rate(y, rate);
        if !nll.is_finite() {
            return (nll, ExponentialMeanEta { mean: f64::NAN });
        }

        let d_rate = y - 1.0 / rate.rate;
        let d_mean = d_rate * (-1.0 / (theta.mean * theta.mean));
        (
            nll,
            ExponentialMeanEta {
                mean: d_mean * Link::derivative_inverse(eta.mean),
            },
        )
    }
}

impl<Link> Family for Exponential<MeanParam, Link>
where
    Link: PositiveLink<f64>,
{
    type Eta = ExponentialMeanEta;
    type Theta = ExponentialMeanTheta;
    type NllGradientEta = ExponentialMeanEta;
    type Observation<'obs> = f64;

    #[inline(always)]
    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline(always)]
    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::nll_rate(y, theta.rate())
    }

    #[inline(always)]
    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        Self::nll_rate(y, Self::theta_from_eta(eta).rate())
    }

    #[inline(always)]
    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        Self::nll_and_gradient_eta_values(y, eta)
    }
}

impl<Link> ParameterizedFamily<1> for Exponential<MeanParam, Link>
where
    Link: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Params = (Mean,);
    type Links = (Link,);

    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values =
            weighted_values::<Self, _, _>(obs, |y| (y.is_finite() && y >= 0.0).then_some(y));
        let Some(mean) = weighted_mean(&values) else {
            return ExponentialMeanEta::from_array([0.0]);
        };
        ExponentialMeanEta {
            mean: Link::initial_eta_from_theta(positive_floor(mean)),
        }
    }
}

impl<Link> Exponential<RateParam, Link>
where
    Link: PositiveLink<f64>,
{
    #[inline(always)]
    fn theta_from_eta(eta: ExponentialRateEta) -> ExponentialRateTheta {
        ExponentialRateTheta {
            rate: Link::inverse(eta.rate),
        }
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: ExponentialRateEta) -> (f64, ExponentialRateEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_rate(y, theta);
        if !nll.is_finite() {
            return (nll, ExponentialRateEta { rate: f64::NAN });
        }

        let d_rate = y - 1.0 / theta.rate;
        (
            nll,
            ExponentialRateEta {
                rate: d_rate * Link::derivative_inverse(eta.rate),
            },
        )
    }
}

impl<Link> Family for Exponential<RateParam, Link>
where
    Link: PositiveLink<f64>,
{
    type Eta = ExponentialRateEta;
    type Theta = ExponentialRateTheta;
    type NllGradientEta = ExponentialRateEta;
    type Observation<'obs> = f64;

    #[inline(always)]
    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline(always)]
    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::nll_rate(y, theta)
    }

    #[inline(always)]
    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        Self::nll_rate(y, Self::theta_from_eta(eta))
    }

    #[inline(always)]
    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        Self::nll_and_gradient_eta_values(y, eta)
    }
}

impl<Link> ParameterizedFamily<1> for Exponential<RateParam, Link>
where
    Link: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Params = (Rate,);
    type Links = (Link,);

    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values =
            weighted_values::<Self, _, _>(obs, |y| (y.is_finite() && y >= 0.0).then_some(y));
        let Some(mean) = weighted_mean(&values) else {
            return ExponentialRateEta::from_array([0.0]);
        };
        ExponentialRateEta {
            rate: Link::initial_eta_from_theta(1.0 / positive_floor(mean)),
        }
    }
}

impl From<ExponentialMeanTheta> for ExponentialRateTheta {
    #[inline(always)]
    fn from(theta: ExponentialMeanTheta) -> Self {
        theta.rate()
    }
}

macro_rules! impl_exponential_helpers {
    ($param:ty) => {
        impl<Link> HasCdf for Exponential<$param, Link>
        where
            Exponential<$param, Link>: for<'obs> Family<Observation<'obs> = f64>,
            <Exponential<$param, Link> as Family>::Theta: Copy + Into<ExponentialRateTheta>,
        {
            fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
                Self::cdf_rate(y, theta.into())
            }
        }

        impl<Link> HasQuantile for Exponential<$param, Link>
        where
            Exponential<$param, Link>: for<'obs> Family<Observation<'obs> = f64>,
            <Exponential<$param, Link> as Family>::Theta: Copy + Into<ExponentialRateTheta>,
        {
            fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
                Self::quantile_rate(p, theta.into())
            }
        }

        impl<Link> HasCrps for Exponential<$param, Link>
        where
            Exponential<$param, Link>: for<'obs> Family<Observation<'obs> = f64>,
            <Exponential<$param, Link> as Family>::Theta: Copy + Into<ExponentialRateTheta>,
        {
            fn crps<'obs>(&self, y: Self::Observation<'obs>, theta: Self::Theta) -> f64 {
                Self::crps_rate(y, theta.into())
            }
        }

        #[cfg(feature = "rand")]
        impl<Rng, Link> CanSimulate<Rng> for Exponential<$param, Link>
        where
            Rng: rand::Rng,
            Exponential<$param, Link>: for<'obs> Family<Observation<'obs> = f64>,
            <Exponential<$param, Link> as Family>::Theta: Copy + Into<ExponentialRateTheta>,
        {
            fn sample(&self, rng: &mut Rng, theta: Self::Theta) -> f64 {
                let theta = theta.into();
                if !Self::valid_rate(theta) {
                    return f64::NAN;
                }

                rand_distr::Distribution::sample(
                    &rand_distr::Exp::new(theta.rate)
                        .expect("validated exponential rate must construct"),
                    rng,
                )
            }
        }
    };
}

impl_exponential_helpers!(MeanParam);
impl_exponential_helpers!(RateParam);

/// Exponential distribution parameterized by mean.
pub type ExponentialMean = Exponential<MeanParam, Log>;
/// Exponential distribution parameterized by rate.
pub type ExponentialRate = Exponential<RateParam, Log>;

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;
    use gamlss_core::{Family, HasCdf, HasCrps, HasQuantile};

    use super::{ExponentialMean, ExponentialMeanTheta, ExponentialRate, ExponentialRateTheta};
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn exponential_parameterization_gradients_match_finite_difference() {
        assert_gradient_matches_finite_difference::<_, 1>(&ExponentialRate::new(), 1.7, [0.4]);
        assert_gradient_matches_finite_difference::<_, 1>(
            &ExponentialMean::new(),
            1.7,
            [0.7_f64.ln()],
        );
    }

    #[test]
    fn exponential_mean_matches_rate_equivalent() {
        let mean = ExponentialMean::new();
        let rate = ExponentialRate::new();
        let theta = ExponentialMeanTheta { mean: 0.7 };
        let canonical = theta.rate();

        assert_relative_eq!(
            mean.nll(1.7, theta),
            rate.nll(1.7, canonical),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean.cdf(1.7, theta),
            rate.cdf(1.7, canonical),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean.quantile(0.4, theta),
            rate.quantile(0.4, canonical),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean.crps(1.7, theta),
            rate.crps(1.7, canonical),
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn exponential_rejects_invalid_domains_and_handles_boundaries() {
        let family = ExponentialMean::new();
        let theta = ExponentialMeanTheta { mean: 0.5 };

        assert!(family.nll(0.0, theta).is_finite());
        assert!(family.nll(1.7, theta).is_finite());
        assert!(family.nll(-1.0, theta).is_infinite());
        assert!(
            family
                .nll(1.7, ExponentialMeanTheta { mean: 0.0 })
                .is_infinite()
        );
        assert_eq!(family.cdf(-1.0, theta), 0.0);
        assert_eq!(family.quantile(0.0, theta), 0.0);
        assert!(family.quantile(1.0, theta).is_infinite());
        assert!(family.quantile(f64::NAN, theta).is_nan());
    }

    #[test]
    fn exponential_crps_matches_fixed_values() {
        let family = ExponentialRate::new();

        assert_relative_eq!(
            family.crps(1.0, ExponentialRateTheta { rate: 2.0 }),
            0.385_335_283_236_612_7,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            family.crps(0.0, ExponentialRateTheta { rate: 2.0 }),
            0.25,
            epsilon = 1.0e-12
        );
    }

    #[cfg(feature = "rand")]
    #[test]
    fn exponential_sampling_returns_finite_values_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = ExponentialMean::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family.sample(&mut rng, ExponentialMeanTheta { mean: 0.5 });
        assert!(sample >= 0.0 && sample.is_finite());
        assert!(
            family
                .sample(&mut rng, ExponentialMeanTheta { mean: 0.0 })
                .is_nan()
        );
    }
}
