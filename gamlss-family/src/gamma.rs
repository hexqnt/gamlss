use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{Family, HasCdf, HasCrps, HasQuantile, Log, ObservationView};

use gamlss_special::{digamma, invert_positive_cdf, ln_beta, ln_gamma, regularized_gamma_lower};

use crate::initial::{LARGE_SHAPE, VARIANCE_FLOOR, positive_floor, weighted_summary};

pub use mean_cv::{GammaMeanCv, GammaMeanCvEta, GammaMeanCvTheta, MeanCv};
pub use mean_shape::{GammaMeanShape, GammaMeanShapeEta, GammaMeanShapeTheta, MeanShape};
pub use shape_rate::{
    GammaEta, GammaShapeRate, GammaShapeRateEta, GammaShapeRateTheta, GammaTheta, ShapeRate,
};

mod mean_cv;
mod mean_shape;
mod shape_rate;

/// Gamma family implementation carrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gamma<Param = ShapeRate, FirstLink = Log, SecondLink = Log> {
    marker: PhantomData<(Param, FirstLink, SecondLink)>,
}

impl<Param, FirstLink, SecondLink> Gamma<Param, FirstLink, SecondLink> {
    /// Creates a stateless gamma family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn valid_shape_rate(theta: GammaShapeRateTheta) -> bool {
        theta.shape > 0.0 && theta.shape.is_finite() && theta.rate > 0.0 && theta.rate.is_finite()
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn nll_shape_rate(y: f64, theta: GammaShapeRateTheta) -> f64 {
        if y <= 0.0 || !y.is_finite() || !Self::valid_shape_rate(theta) {
            return f64::INFINITY;
        }

        ln_gamma(theta.shape) - theta.shape * theta.rate.ln() - (theta.shape - 1.0) * y.ln()
            + theta.rate * y
    }

    #[inline]
    fn gradient_shape_rate(y: f64, theta: GammaShapeRateTheta) -> (f64, f64) {
        (
            digamma(theta.shape) - theta.rate.ln() - y.ln(),
            y - theta.shape / theta.rate,
        )
    }

    #[inline]
    fn cdf_shape_rate(y: f64, theta: GammaShapeRateTheta) -> f64 {
        if !y.is_finite() || !Self::valid_shape_rate(theta) {
            return f64::NAN;
        }
        if y <= 0.0 {
            return 0.0;
        }

        regularized_gamma_lower(theta.shape, theta.rate * y)
    }

    #[inline]
    fn quantile_shape_rate(p: f64, theta: GammaShapeRateTheta) -> f64 {
        if !Self::valid_shape_rate(theta) {
            return f64::NAN;
        }

        invert_positive_cdf(p, |y| Self::cdf_shape_rate(y, theta))
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn crps_shape_rate(y: f64, theta: GammaShapeRateTheta) -> f64 {
        if y < 0.0 || !y.is_finite() || !Self::valid_shape_rate(theta) {
            return f64::NAN;
        }

        let f_shape = regularized_gamma_lower(theta.shape, theta.rate * y);
        let f_next_shape = regularized_gamma_lower(theta.shape + 1.0, theta.rate * y);
        let mean = theta.shape / theta.rate;
        let beta_term = ln_beta(theta.shape + 0.5, 0.5).exp() / (std::f64::consts::PI * theta.rate);

        y * (2.0 * f_shape - 1.0) - mean * (2.0 * f_next_shape - 1.0) - beta_term
    }

    #[inline]
    fn initial_mean_shape<'obs, Obs>(obs: &'obs Obs) -> Option<(f64, f64)>
    where
        Obs: ObservationView<'obs, Observation = f64> + 'obs,
    {
        let mut values = Vec::new();
        for row in 0..obs.len() {
            let y = obs.observation_at(row);
            if y.is_finite() && y > 0.0 {
                values.push((y, obs.weight_at(row)));
            }
        }
        let summary = weighted_summary(&values)?;

        let mean = positive_floor(summary.mean);
        let shape = if summary.variance <= VARIANCE_FLOOR {
            LARGE_SHAPE
        } else {
            positive_floor(mean * mean / summary.variance)
        };
        Some((mean, shape))
    }
}

impl<Param, FirstLink, SecondLink> Default for Gamma<Param, FirstLink, SecondLink> {
    fn default() -> Self {
        Self::new()
    }
}

macro_rules! impl_gamma_helpers {
    ($param:ty, $first:ident, $second:ident) => {
        impl<$first, $second> HasCdf for Gamma<$param, $first, $second>
        where
            Gamma<$param, $first, $second>: for<'obs> Family<Observation<'obs> = f64>,
            <Gamma<$param, $first, $second> as Family>::Theta: Copy + Into<GammaShapeRateTheta>,
        {
            fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
                Self::cdf_shape_rate(y, theta.into())
            }
        }

        impl<$first, $second> HasQuantile for Gamma<$param, $first, $second>
        where
            Gamma<$param, $first, $second>: for<'obs> Family<Observation<'obs> = f64>,
            <Gamma<$param, $first, $second> as Family>::Theta: Copy + Into<GammaShapeRateTheta>,
        {
            fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
                Self::quantile_shape_rate(p, theta.into())
            }
        }

        impl<$first, $second> HasCrps for Gamma<$param, $first, $second>
        where
            Gamma<$param, $first, $second>: for<'obs> Family<Observation<'obs> = f64>,
            <Gamma<$param, $first, $second> as Family>::Theta: Copy + Into<GammaShapeRateTheta>,
        {
            fn crps(&self, y: Self::Observation<'_>, theta: Self::Theta) -> f64 {
                Self::crps_shape_rate(y, theta.into())
            }
        }

        #[cfg(feature = "rand")]
        impl<Rng, $first, $second> CanSimulate<Rng> for Gamma<$param, $first, $second>
        where
            Rng: rand::Rng,
            Gamma<$param, $first, $second>: for<'obs> Family<Observation<'obs> = f64>,
            <Gamma<$param, $first, $second> as Family>::Theta: Copy + Into<GammaShapeRateTheta>,
        {
            fn sample(&self, rng: &mut Rng, theta: Self::Theta) -> f64 {
                let theta = theta.into();
                if !Self::valid_shape_rate(theta) {
                    return f64::NAN;
                }

                rand_distr::Distribution::sample(
                    &rand_distr::Gamma::new(theta.shape, 1.0 / theta.rate)
                        .expect("validated gamma parameters must construct"),
                    rng,
                )
            }
        }
    };
}

impl_gamma_helpers!(MeanCv, MeanLink, CvLink);
impl_gamma_helpers!(MeanShape, MeanLink, ShapeLink);
impl_gamma_helpers!(ShapeRate, ShapeLink, RateLink);

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;
    use gamlss_core::{Family, HasCdf, HasCrps, HasQuantile};
    use statrs::distribution::{ContinuousCDF, Gamma as StatrsGamma};

    use super::{
        GammaMeanCv, GammaMeanCvTheta, GammaMeanShape, GammaShapeRate, GammaShapeRateTheta,
    };
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn gamma_parameterization_gradients_match_finite_difference() {
        assert_gradient_matches_finite_difference::<_, 2>(&GammaShapeRate::new(), 1.7, [0.4, -0.2]);
        assert_gradient_matches_finite_difference::<_, 2>(
            &GammaMeanShape::new(),
            1.7,
            [1.4_f64.ln(), 2.5_f64.ln()],
        );
        assert_gradient_matches_finite_difference::<_, 2>(
            &GammaMeanCv::new(),
            1.7,
            [1.4_f64.ln(), 0.6_f64.ln()],
        );
    }

    #[test]
    fn gamma_mean_cv_matches_shape_rate_equivalent() {
        let mean_cv = GammaMeanCv::new();
        let shape_rate = GammaShapeRate::new();
        let theta = GammaMeanCvTheta { mean: 1.4, cv: 0.6 };
        let canonical = theta.shape_rate();

        assert_relative_eq!(
            mean_cv.nll(1.7, theta),
            shape_rate.nll(1.7, canonical),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean_cv.cdf(1.7, theta),
            shape_rate.cdf(1.7, canonical),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean_cv.quantile(0.4, theta),
            shape_rate.quantile(0.4, canonical),
            epsilon = 1.0e-8
        );
        assert_relative_eq!(
            mean_cv.crps(1.7, theta),
            shape_rate.crps(1.7, canonical),
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn gamma_rejects_invalid_domains() {
        let family = GammaMeanCv::new();

        assert!(
            family
                .nll(1.7, GammaMeanCvTheta { mean: 1.0, cv: 0.5 })
                .is_finite()
        );
        assert!(
            family
                .nll(0.0, GammaMeanCvTheta { mean: 1.0, cv: 0.5 })
                .is_infinite()
        );
        assert!(
            family
                .nll(1.7, GammaMeanCvTheta { mean: 0.0, cv: 0.5 })
                .is_infinite()
        );
        assert!(
            family
                .nll(1.7, GammaMeanCvTheta { mean: 1.0, cv: 0.0 })
                .is_infinite()
        );
        assert!(
            family
                .nll(
                    1.7,
                    GammaMeanCvTheta {
                        mean: f64::NAN,
                        cv: 0.5
                    }
                )
                .is_infinite()
        );
    }

    #[test]
    fn gamma_cdf_and_quantile_match_statrs_reference() {
        let family = GammaShapeRate::new();
        let theta = GammaShapeRateTheta {
            shape: 2.5,
            rate: 1.7,
        };
        let reference = StatrsGamma::new(theta.shape, theta.rate).unwrap();

        for y in [0.05, 0.25, 1.0, 2.0, 8.0] {
            assert_relative_eq!(family.cdf(y, theta), reference.cdf(y), epsilon = 1.0e-11);
        }

        for p in [0.01, 0.1, 0.5, 0.9, 0.99] {
            assert_relative_eq!(
                family.quantile(p, theta),
                reference.inverse_cdf(p),
                epsilon = 1.0e-8
            );
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn gamma_boundaries_and_crps_behave_like_shape_rate_kernel() {
        let family = GammaShapeRate::new();
        let theta = GammaShapeRateTheta {
            shape: 1.0,
            rate: 2.0,
        };

        assert_eq!(family.cdf(0.0, theta), 0.0);
        assert_eq!(family.quantile(0.0, theta), 0.0);
        assert_eq!(family.quantile(1.0, theta), f64::INFINITY);
        assert!(family.quantile(f64::NAN, theta).is_nan());
        assert_relative_eq!(
            family.crps(1.0, theta),
            0.385_335_283_236_612_7,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(family.crps(0.0, theta), 0.25, epsilon = 1.0e-12);
        assert!(family.crps(-1.0, theta).is_nan());
    }

    #[cfg(feature = "rand")]
    #[test]
    fn gamma_sampling_returns_finite_values_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = GammaMeanCv::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family.sample(&mut rng, GammaMeanCvTheta { mean: 1.5, cv: 0.7 });
        assert!(sample > 0.0 && sample.is_finite());
        assert!(
            family
                .sample(&mut rng, GammaMeanCvTheta { mean: 1.5, cv: 0.0 })
                .is_nan()
        );
    }
}
