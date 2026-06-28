use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{Family, HasCdf, HasCrps, HasQuantile, Log};

use crate::domain::{is_positive_finite, is_probability};

pub use mean::{ExponentialMean, ExponentialMeanEta, ExponentialMeanTheta, MeanParam};
pub use rate::{ExponentialRate, ExponentialRateEta, ExponentialRateTheta, RateParam};

mod mean;
mod rate;

/// Exponential family implementation carrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exponential<Param = RateParam, Link = Log> {
    marker: PhantomData<(Param, Link)>,
}

impl<Param, Link> Exponential<Param, Link> {
    /// Creates a stateless exponential family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn valid_rate(theta: ExponentialRateTheta) -> bool {
        is_positive_finite(theta.rate)
    }

    #[inline]
    fn nll_rate(y: f64, theta: ExponentialRateTheta) -> f64 {
        if y < 0.0 || !y.is_finite() || !Self::valid_rate(theta) {
            return f64::INFINITY;
        }

        theta.rate.mul_add(y, -theta.rate.ln())
    }

    #[inline]
    fn cdf_rate(y: f64, theta: ExponentialRateTheta) -> f64 {
        if !y.is_finite() || !Self::valid_rate(theta) {
            return f64::NAN;
        }
        if y < 0.0 {
            return 0.0;
        }

        -(-theta.rate * y).exp_m1()
    }

    #[inline]
    fn quantile_rate(p: f64, theta: ExponentialRateTheta) -> f64 {
        if !is_probability(p) || !Self::valid_rate(theta) {
            return f64::NAN;
        }

        -(-p).ln_1p() / theta.rate
    }

    #[inline]
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

macro_rules! impl_exponential_helpers {
    ($param:ty) => {
        impl<Link> HasCdf for Exponential<$param, Link>
        where
            Exponential<$param, Link>: for<'obs> Family<Observation<'obs> = f64>,
            <Exponential<$param, Link> as Family>::Theta: Copy + Into<ExponentialRateTheta>,
        {
            fn cdf(&self, y: Self::Observation<'_>, theta: Self::Theta) -> f64 {
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
            fn crps(&self, y: Self::Observation<'_>, theta: Self::Theta) -> f64 {
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
            type Sample = f64;

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
    #[allow(clippy::float_cmp)]
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
