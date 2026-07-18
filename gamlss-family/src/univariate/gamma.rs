use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{Family, HasCdf, HasCrps, HasQuantile, Log, ObservationView};

use gamlss_special::{
    digamma_minus_ln, invert_positive_cdf, ln_beta, ln_gamma_stirling_residual,
    regularized_gamma_lower, regularized_gamma_upper,
};

use crate::initial::{LARGE_SHAPE, VARIANCE_FLOOR, positive_floor, weighted_summary};

pub use mean_cv::{GammaMeanCv, GammaMeanCvEta, GammaMeanCvTheta, MeanCv};
pub use mean_shape::{GammaMeanShape, GammaMeanShapeEta, GammaMeanShapeTheta, MeanShape};
pub use shape_rate::{GammaShapeRate, GammaShapeRateEta, GammaShapeRateTheta, ShapeRate};

mod mean_cv;
mod mean_shape;
mod shape_rate;

/// Gamma family implementation carrier using the shape/rate kernel.
///
/// For shape $\alpha>0$, rate $\beta>0$, and $y>0$, the density is
///
/// $$
/// f(y\mid\alpha,\beta)
/// = \frac{\beta^\alpha}{\Gamma(\alpha)}
///   y^{\alpha-1}\exp(-\beta y).
/// $$
///
/// Here $\Gamma$ is the gamma function.
///
/// Consequently,
///
/// $$
/// \mathbb{E}(Y)=\frac{\alpha}{\beta},
/// \qquad
/// \operatorname{Var}(Y)=\frac{\alpha}{\beta^2}.
/// $$
///
/// [`ShapeRate`], [`MeanShape`], and [`MeanCv`] select the public natural-scale parameterization while sharing this kernel.
///
/// In the canonical carrier, [`GammaShapeRateTheta::shape`] is $\alpha$ and [`GammaShapeRateTheta::rate`] is $\beta$. The other parameterizations document their exact conversion to these two fields.
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

        ln_gamma_stirling_residual(theta.shape)
            + Self::scaled_gamma_ratio_deviance(y, theta)
            + y.ln()
    }

    #[inline]
    fn gradient_shape_rate(y: f64, theta: GammaShapeRateTheta) -> (f64, f64) {
        (
            digamma_minus_ln(theta.shape) - Self::log_rate_y_over_shape(y, theta),
            y - theta.shape / theta.rate,
        )
    }

    #[inline]
    fn log_rate_y_over_shape(y: f64, theta: GammaShapeRateTheta) -> f64 {
        let ratio = (theta.rate / theta.shape) * y;
        if ratio.is_finite() && ratio > 0.0 {
            let centered = ratio - 1.0;
            if centered.abs() <= 0.5 {
                return centered.ln_1p();
            }
        }
        theta.rate.ln() + y.ln() - theta.shape.ln()
    }

    #[inline]
    fn scaled_gamma_ratio_deviance(y: f64, theta: GammaShapeRateTheta) -> f64 {
        let ratio = (theta.rate / theta.shape) * y;
        if ratio.is_finite() && ratio > 0.0 {
            let centered = ratio - 1.0;
            if centered.abs() <= 0.5 {
                return theta.shape * (centered - centered.ln_1p());
            }
        }
        theta.shape.mul_add(
            -Self::log_rate_y_over_shape(y, theta),
            theta.rate.mul_add(y, -theta.shape),
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

        let scaled_y = theta.rate * y;
        let f_shape = regularized_gamma_lower(theta.shape, scaled_y);
        let s_shape = regularized_gamma_upper(theta.shape, scaled_y);
        let f_next_shape = regularized_gamma_lower(theta.shape + 1.0, scaled_y);
        let s_next_shape = regularized_gamma_upper(theta.shape + 1.0, scaled_y);
        let mean = theta.shape / theta.rate;
        let beta_term = ln_beta(theta.shape + 0.5, 0.5).exp() / (std::f64::consts::PI * theta.rate);

        y * (f_shape - s_shape) - mean * (f_next_shape - s_next_shape) - beta_term
    }

    #[inline]
    fn initial_mean_shape<'obs, Obs>(obs: &'obs Obs) -> Option<(f64, f64)>
    where
        Obs: ObservationView<'obs, Observation = f64> + 'obs,
    {
        let mut values = Vec::with_capacity(obs.len());
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
            fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
                Self::cdf_shape_rate(y, (*theta).into())
            }
        }

        impl<$first, $second> HasQuantile for Gamma<$param, $first, $second>
        where
            Gamma<$param, $first, $second>: for<'obs> Family<Observation<'obs> = f64>,
            <Gamma<$param, $first, $second> as Family>::Theta: Copy + Into<GammaShapeRateTheta>,
        {
            fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
                Self::quantile_shape_rate(p, (*theta).into())
            }
        }

        impl<$first, $second> HasCrps for Gamma<$param, $first, $second>
        where
            Gamma<$param, $first, $second>: for<'obs> Family<Observation<'obs> = f64>,
            <Gamma<$param, $first, $second> as Family>::Theta: Copy + Into<GammaShapeRateTheta>,
        {
            fn crps(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
                Self::crps_shape_rate(y, (*theta).into())
            }
        }

        #[cfg(feature = "rand")]
        impl<Rng, $first, $second> CanSimulate<Rng> for Gamma<$param, $first, $second>
        where
            Rng: rand::Rng,
            Gamma<$param, $first, $second>: for<'obs> Family<Observation<'obs> = f64>,
            <Gamma<$param, $first, $second> as Family>::Theta: Copy + Into<GammaShapeRateTheta>,
        {
            type Sample = f64;

            fn sample(&self, rng: &mut Rng, theta: &Self::Theta) -> f64 {
                let theta = (*theta).into();
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
        GammaMeanCv, GammaMeanCvEta, GammaMeanCvTheta, GammaMeanShape, GammaShapeRate,
        GammaShapeRateEta, GammaShapeRateTheta,
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
            mean_cv.nll(1.7, &theta, &mut mean_cv.workspace()),
            shape_rate.nll(1.7, &canonical, &mut shape_rate.workspace()),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean_cv.cdf(1.7, &theta),
            shape_rate.cdf(1.7, &canonical),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean_cv.quantile(0.4, &theta),
            shape_rate.quantile(0.4, &canonical),
            epsilon = 1.0e-8
        );
        assert_relative_eq!(
            mean_cv.crps(1.7, &theta),
            shape_rate.crps(1.7, &canonical),
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn gamma_mean_cv_accepts_representable_shape_beyond_squared_cv_range() {
        let family = GammaMeanCv::new();
        let theta = GammaMeanCvTheta {
            mean: 1.0,
            cv: 1.0e155,
        };
        let eta = GammaMeanCvEta {
            mean: 0.0,
            cv: theta.cv.ln(),
        };

        let natural_nll = family.nll(1.0, &theta, &mut ());
        let (eta_nll, gradient) = family.nll_and_gradient_eta(1.0, &eta, &mut ());

        assert!(
            natural_nll.is_finite(),
            "natural-scale nll was {natural_nll}"
        );
        assert!(eta_nll.is_finite(), "eta-scale nll was {eta_nll}");
        assert!(
            gradient.mean.abs() < 1.0e-14,
            "mean gradient was {}",
            gradient.mean
        );
        assert!(
            (gradient.cv - 2.0).abs() < 1.0e-12,
            "cv gradient was {}",
            gradient.cv
        );
    }

    #[test]
    fn concentrated_gamma_preserves_normalizer_and_shape_gradient() {
        let family = GammaShapeRate::new();
        let eta = GammaShapeRateEta {
            shape: 1.0e16_f64.ln(),
            rate: 1.0e16_f64.ln(),
        };
        let shape = eta.shape.exp();
        let expected_nll = 0.5f64.mul_add(
            -shape.ln(),
            crate::constants::HALF_LOG_2_PI + 1.0 / (12.0 * shape),
        );
        let (nll, gradient) = family.nll_and_gradient_eta(1.0, &eta, &mut ());

        assert!((nll - expected_nll).abs() < 1.0e-13, "nll was {nll}");
        assert!(
            (gradient.shape + 0.5).abs() < 1.0e-14,
            "shape gradient was {}",
            gradient.shape
        );
        assert!(
            gradient.rate.abs() < 1.0e-14,
            "rate gradient was {}",
            gradient.rate
        );
    }

    #[test]
    fn gamma_rejects_invalid_domains() {
        let family = GammaMeanCv::new();

        assert!(
            family
                .nll(
                    1.7,
                    &GammaMeanCvTheta { mean: 1.0, cv: 0.5 },
                    &mut family.workspace()
                )
                .is_finite()
        );
        assert!(
            family
                .nll(
                    0.0,
                    &GammaMeanCvTheta { mean: 1.0, cv: 0.5 },
                    &mut family.workspace()
                )
                .is_infinite()
        );
        assert!(
            family
                .nll(
                    1.7,
                    &GammaMeanCvTheta { mean: 0.0, cv: 0.5 },
                    &mut family.workspace()
                )
                .is_infinite()
        );
        assert!(
            family
                .nll(
                    1.7,
                    &GammaMeanCvTheta { mean: 1.0, cv: 0.0 },
                    &mut family.workspace()
                )
                .is_infinite()
        );
        assert!(
            family
                .nll(
                    1.7,
                    &GammaMeanCvTheta {
                        mean: f64::NAN,
                        cv: 0.5
                    },
                    &mut family.workspace()
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
            assert_relative_eq!(family.cdf(y, &theta), reference.cdf(y), epsilon = 1.0e-11);
        }

        for p in [0.01, 0.1, 0.5, 0.9, 0.99] {
            assert_relative_eq!(
                family.quantile(p, &theta),
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

        assert_eq!(family.cdf(0.0, &theta), 0.0);
        assert_eq!(family.quantile(0.0, &theta), 0.0);
        assert_eq!(family.quantile(1.0, &theta), f64::INFINITY);
        assert!(family.quantile(f64::NAN, &theta).is_nan());
        assert_relative_eq!(
            family.crps(1.0, &theta),
            0.385_335_283_236_612_7,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(family.crps(0.0, &theta), 0.25, epsilon = 1.0e-12);
        assert!(family.crps(-1.0, &theta).is_nan());
    }

    #[cfg(feature = "rand")]
    #[test]
    fn gamma_sampling_returns_finite_values_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = GammaMeanCv::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family.sample(&mut rng, &GammaMeanCvTheta { mean: 1.5, cv: 0.7 });
        assert!(sample > 0.0 && sample.is_finite());
        assert!(
            family
                .sample(&mut rng, &GammaMeanCvTheta { mean: 1.5, cv: 0.0 })
                .is_nan()
        );
    }
}
