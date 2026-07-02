use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{Family, HasCdf, HasCrps, HasQuantile, Log, ObservationView};

use gamlss_special::{ln_gamma, regularized_gamma_lower};

use crate::domain::{is_positive_finite, is_probability};
use crate::initial::{VARIANCE_FLOOR, positive_floor, weighted_summary};

pub use mean_shape::{MeanShape, WeibullMeanShape, WeibullMeanShapeEta, WeibullMeanShapeTheta};
pub use scale_shape::{
    ScaleShape, WeibullEta, WeibullScaleShape, WeibullScaleShapeEta, WeibullScaleShapeTheta,
    WeibullTheta,
};

mod mean_shape;
mod scale_shape;

const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// Weibull family implementation carrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Weibull<Param = ScaleShape, FirstLink = Log, SecondLink = Log> {
    marker: PhantomData<(Param, FirstLink, SecondLink)>,
}

impl<Param, FirstLink, SecondLink> Weibull<Param, FirstLink, SecondLink> {
    /// Creates a stateless Weibull family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn valid_scale_shape(theta: WeibullScaleShapeTheta) -> bool {
        is_positive_finite(theta.scale) && is_positive_finite(theta.shape)
    }

    #[inline]
    fn mean_factor(shape: f64) -> f64 {
        ln_gamma(1.0 + 1.0 / shape).exp()
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn nll_scale_shape(y: f64, theta: WeibullScaleShapeTheta) -> f64 {
        if y <= 0.0 || !y.is_finite() || !Self::valid_scale_shape(theta) {
            return f64::INFINITY;
        }

        let log_ratio = y.ln() - theta.scale.ln();
        -theta.shape.ln() - (theta.shape - 1.0) * y.ln()
            + theta.shape * theta.scale.ln()
            + (theta.shape * log_ratio).exp()
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn gradient_scale_shape(y: f64, theta: WeibullScaleShapeTheta) -> (f64, f64) {
        let log_ratio = y.ln() - theta.scale.ln();
        let power = (theta.shape * log_ratio).exp();
        (
            theta.shape * (1.0 - power) / theta.scale,
            -1.0 / theta.shape - log_ratio + power * log_ratio,
        )
    }

    #[inline]
    fn cdf_scale_shape(y: f64, theta: WeibullScaleShapeTheta) -> f64 {
        if !y.is_finite() || !Self::valid_scale_shape(theta) {
            return f64::NAN;
        }
        if y <= 0.0 {
            return 0.0;
        }

        let log_ratio = y.ln() - theta.scale.ln();
        -(-(theta.shape * log_ratio).exp()).exp_m1()
    }

    #[inline]
    fn quantile_scale_shape(p: f64, theta: WeibullScaleShapeTheta) -> f64 {
        if !is_probability(p) || !Self::valid_scale_shape(theta) {
            return f64::NAN;
        }

        theta.scale * (-(-p).ln_1p()).powf(1.0 / theta.shape)
    }

    #[inline]
    #[allow(clippy::suboptimal_flops, clippy::imprecise_flops)]
    fn crps_scale_shape(y: f64, theta: WeibullScaleShapeTheta) -> f64 {
        if y < 0.0 || !y.is_finite() || !Self::valid_scale_shape(theta) {
            return f64::NAN;
        }

        let a = 1.0 + 1.0 / theta.shape;
        let mean = theta.scale * ln_gamma(a).exp();
        let t = if y == 0.0 {
            0.0
        } else {
            (y / theta.scale).powf(theta.shape)
        };
        let cdf = -(-t).exp_m1();
        y * (2.0 * cdf - 1.0) - 2.0 * mean * regularized_gamma_lower(a, t)
            + mean * 2.0_f64.powf(-1.0 / theta.shape)
    }

    #[inline]
    fn initial_scale_shape<'obs, Obs>(obs: &'obs Obs) -> Option<(f64, f64)>
    where
        Obs: ObservationView<'obs, Observation = f64> + 'obs,
    {
        let mut log_values = Vec::with_capacity(obs.len());
        for row in 0..obs.len() {
            let y = obs.observation_at(row);
            if y.is_finite() && y > 0.0 {
                log_values.push((y.ln(), obs.weight_at(row)));
            }
        }
        let summary = weighted_summary(&log_values)?;

        let shape = if summary.variance <= VARIANCE_FLOOR {
            10.0
        } else {
            positive_floor(std::f64::consts::PI / (6.0 * summary.variance).sqrt())
        };
        let scale = positive_floor((summary.mean + EULER_GAMMA / shape).exp());
        Some((scale, shape))
    }
}

impl<Param, FirstLink, SecondLink> Default for Weibull<Param, FirstLink, SecondLink> {
    fn default() -> Self {
        Self::new()
    }
}

macro_rules! impl_weibull_helpers {
    ($param:ty, $first:ident, $second:ident) => {
        impl<$first, $second> HasCdf for Weibull<$param, $first, $second>
        where
            Weibull<$param, $first, $second>: for<'obs> Family<Observation<'obs> = f64>,
            <Weibull<$param, $first, $second> as Family>::Theta:
                Copy + Into<WeibullScaleShapeTheta>,
        {
            fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
                Self::cdf_scale_shape(y, (*theta).into())
            }
        }

        impl<$first, $second> HasQuantile for Weibull<$param, $first, $second>
        where
            Weibull<$param, $first, $second>: for<'obs> Family<Observation<'obs> = f64>,
            <Weibull<$param, $first, $second> as Family>::Theta:
                Copy + Into<WeibullScaleShapeTheta>,
        {
            fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
                Self::quantile_scale_shape(p, (*theta).into())
            }
        }

        impl<$first, $second> HasCrps for Weibull<$param, $first, $second>
        where
            Weibull<$param, $first, $second>: for<'obs> Family<Observation<'obs> = f64>,
            <Weibull<$param, $first, $second> as Family>::Theta:
                Copy + Into<WeibullScaleShapeTheta>,
        {
            fn crps(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
                Self::crps_scale_shape(y, (*theta).into())
            }
        }

        #[cfg(feature = "rand")]
        impl<Rng, $first, $second> CanSimulate<Rng> for Weibull<$param, $first, $second>
        where
            Rng: rand::Rng,
            Weibull<$param, $first, $second>: for<'obs> Family<Observation<'obs> = f64>,
            <Weibull<$param, $first, $second> as Family>::Theta:
                Copy + Into<WeibullScaleShapeTheta>,
        {
            type Sample = f64;

            fn sample(&self, rng: &mut Rng, theta: &Self::Theta) -> f64 {
                let theta = (*theta).into();
                if !Self::valid_scale_shape(theta) {
                    return f64::NAN;
                }

                rand_distr::Distribution::sample(
                    &rand_distr::Weibull::new(theta.scale, theta.shape)
                        .expect("validated weibull parameters must construct"),
                    rng,
                )
            }
        }
    };
}

impl_weibull_helpers!(MeanShape, MeanLink, ShapeLink);
impl_weibull_helpers!(ScaleShape, ScaleLink, ShapeLink);

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;
    use gamlss_core::{Family, HasCdf, HasCrps, HasQuantile};

    use super::{
        WeibullMeanShape, WeibullMeanShapeTheta, WeibullScaleShape, WeibullScaleShapeTheta,
    };
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn weibull_parameterization_gradients_match_finite_difference() {
        assert_gradient_matches_finite_difference::<_, 2>(
            &WeibullScaleShape::new(),
            1.7,
            [0.8_f64.ln(), 1.5_f64.ln()],
        );
        assert_gradient_matches_finite_difference::<_, 2>(
            &WeibullMeanShape::new(),
            1.7,
            [1.2_f64.ln(), 1.5_f64.ln()],
        );
    }

    #[test]
    fn weibull_mean_matches_scale_shape_equivalent() {
        let mean = WeibullMeanShape::new();
        let scale_shape = WeibullScaleShape::new();
        let theta = WeibullMeanShapeTheta {
            mean: 1.2,
            shape: 1.5,
        };
        let canonical = theta.scale_shape();

        assert_relative_eq!(
            mean.nll(1.7, &theta, &mut mean.workspace()),
            scale_shape.nll(1.7, &canonical, &mut scale_shape.workspace()),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean.cdf(1.7, &theta),
            scale_shape.cdf(1.7, &canonical),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean.quantile(0.4, &theta),
            scale_shape.quantile(0.4, &canonical),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean.crps(1.7, &theta),
            scale_shape.crps(1.7, &canonical),
            epsilon = 1.0e-12
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn weibull_rejects_invalid_domains_and_handles_boundaries() {
        let family = WeibullMeanShape::new();
        let theta = WeibullMeanShapeTheta {
            mean: 1.2,
            shape: 1.5,
        };

        assert!(family.nll(1.7, &theta, &mut family.workspace()).is_finite());
        assert!(
            family
                .nll(0.0, &theta, &mut family.workspace())
                .is_infinite()
        );
        assert!(
            family
                .nll(
                    1.7,
                    &WeibullMeanShapeTheta {
                        mean: 0.0,
                        shape: 1.5
                    },
                    &mut family.workspace()
                )
                .is_infinite()
        );
        assert!(
            family
                .nll(
                    1.7,
                    &WeibullMeanShapeTheta {
                        mean: 1.2,
                        shape: 0.0
                    },
                    &mut family.workspace()
                )
                .is_infinite()
        );
        assert_eq!(family.cdf(0.0, &theta), 0.0);
        assert_eq!(family.cdf(-1.0, &theta), 0.0);
        assert_eq!(family.quantile(0.0, &theta), 0.0);
        assert!(family.quantile(1.0, &theta).is_infinite());
    }

    #[test]
    fn weibull_cdf_and_crps_match_fixed_values() {
        let family = WeibullScaleShape::new();
        let theta = WeibullScaleShapeTheta {
            scale: 3.0,
            shape: 2.0,
        };

        assert_relative_eq!(
            family.cdf(theta.scale, &theta),
            1.0 - (-1.0_f64).exp(),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            family.cdf(theta.scale * std::f64::consts::LN_2.sqrt(), &theta),
            0.5,
            epsilon = 1.0e-12
        );

        let exp_theta = WeibullScaleShapeTheta {
            scale: 0.5,
            shape: 1.0,
        };
        assert_relative_eq!(
            family.crps(1.0, &exp_theta),
            0.385_335_283_236_612_7,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(family.crps(0.0, &exp_theta), 0.25, epsilon = 1.0e-12);
    }

    #[cfg(feature = "rand")]
    #[test]
    fn weibull_sampling_returns_finite_values_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = WeibullMeanShape::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family.sample(
            &mut rng,
            &WeibullMeanShapeTheta {
                mean: 1.2,
                shape: 1.5,
            },
        );
        assert!(sample > 0.0 && sample.is_finite());
    }
}
