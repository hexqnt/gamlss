use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Family, HasCdf, HasCrps, HasQuantile, InitialEtaFromTheta, Log, ObservationView,
    ParameterParts, ParameterizedFamily, PositiveLink, Scale, Shape,
};

use crate::initial::{VARIANCE_FLOOR, positive_floor, weighted_summary, weighted_values};
use crate::special::{ln_gamma, regularized_gamma_lower};

const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// Weibull family parameterized by positive shape and scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Weibull<ShapeLink = Log, ScaleLink = Log> {
    marker: PhantomData<(ShapeLink, ScaleLink)>,
}

impl<ShapeLink, ScaleLink> Weibull<ShapeLink, ScaleLink>
where
    ShapeLink: PositiveLink<f64>,
    ScaleLink: PositiveLink<f64>,
{
    /// Creates a stateless Weibull family.
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn theta_from_eta(eta: WeibullEta) -> WeibullTheta {
        WeibullTheta {
            shape: ShapeLink::inverse(eta.shape),
            scale: ScaleLink::inverse(eta.scale),
        }
    }

    #[inline(always)]
    fn nll_theta(y: f64, theta: WeibullTheta) -> f64 {
        if y <= 0.0
            || !y.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
            || theta.scale <= 0.0
            || !theta.scale.is_finite()
        {
            return f64::INFINITY;
        }

        let log_ratio = y.ln() - theta.scale.ln();
        -theta.shape.ln() - (theta.shape - 1.0) * y.ln()
            + theta.shape * theta.scale.ln()
            + (theta.shape * log_ratio).exp()
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: WeibullEta) -> (f64, WeibullEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (
                nll,
                WeibullEta {
                    shape: f64::NAN,
                    scale: f64::NAN,
                },
            );
        }

        let log_ratio = y.ln() - theta.scale.ln();
        let power = (theta.shape * log_ratio).exp();
        let d_shape = -1.0 / theta.shape - log_ratio + power * log_ratio;
        let d_scale = theta.shape * (1.0 - power) / theta.scale;
        let gradient_eta = WeibullEta {
            shape: d_shape * ShapeLink::derivative_inverse(eta.shape),
            scale: d_scale * ScaleLink::derivative_inverse(eta.scale),
        };

        (nll, gradient_eta)
    }
}

impl<ShapeLink, ScaleLink> Default for Weibull<ShapeLink, ScaleLink>
where
    ShapeLink: PositiveLink<f64>,
    ScaleLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Predictors for the Weibull family on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeibullEta {
    /// Shape predictor.
    pub shape: f64,
    /// Scale predictor.
    pub scale: f64,
}

impl ParameterParts<2> for WeibullEta {
    #[inline(always)]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            shape: values[0],
            scale: values[1],
        }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.shape,
            1 => self.scale,
            _ => unreachable!("weibull eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale Weibull parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeibullTheta {
    /// Positive shape parameter.
    pub shape: f64,
    /// Positive scale parameter.
    pub scale: f64,
}

impl<ShapeLink, ScaleLink> Family for Weibull<ShapeLink, ScaleLink>
where
    ShapeLink: PositiveLink<f64>,
    ScaleLink: PositiveLink<f64>,
{
    type Eta = WeibullEta;
    type Theta = WeibullTheta;
    type NllGradientEta = WeibullEta;
    type Observation<'obs> = f64;

    #[inline(always)]
    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline(always)]
    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::nll_theta(y, theta)
    }

    #[inline(always)]
    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        Self::nll_theta(y, Self::theta_from_eta(eta))
    }

    #[inline(always)]
    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        Self::nll_and_gradient_eta_values(y, eta)
    }
}

impl<ShapeLink, ScaleLink> ParameterizedFamily<2> for Weibull<ShapeLink, ScaleLink>
where
    ShapeLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    ScaleLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Params = (Shape, Scale);
    type Links = (ShapeLink, ScaleLink);

    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let log_values =
            weighted_values::<Self, _, _>(obs, |y| (y.is_finite() && y > 0.0).then_some(y.ln()));
        let Some(summary) = weighted_summary(&log_values) else {
            return WeibullEta::from_array([0.0, 0.0]);
        };

        let shape = if summary.variance <= VARIANCE_FLOOR {
            10.0
        } else {
            positive_floor(std::f64::consts::PI / (6.0 * summary.variance).sqrt())
        };
        let scale = positive_floor((summary.mean + EULER_GAMMA / shape).exp());

        WeibullEta {
            shape: ShapeLink::initial_eta_from_theta(shape),
            scale: ScaleLink::initial_eta_from_theta(scale),
        }
    }
}

impl<ShapeLink, ScaleLink> HasCdf for Weibull<ShapeLink, ScaleLink>
where
    ShapeLink: PositiveLink<f64>,
    ScaleLink: PositiveLink<f64>,
{
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
        if !y.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
            || theta.scale <= 0.0
            || !theta.scale.is_finite()
        {
            return f64::NAN;
        }
        if y <= 0.0 {
            return 0.0;
        }

        -(-(y / theta.scale).powf(theta.shape)).exp_m1()
    }
}

impl<ShapeLink, ScaleLink> HasQuantile for Weibull<ShapeLink, ScaleLink>
where
    ShapeLink: PositiveLink<f64>,
    ScaleLink: PositiveLink<f64>,
{
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
        if !(0.0..=1.0).contains(&p)
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
            || theta.scale <= 0.0
            || !theta.scale.is_finite()
        {
            return f64::NAN;
        }

        theta.scale * (-(-p).ln_1p()).powf(1.0 / theta.shape)
    }
}

impl<ShapeLink, ScaleLink> HasCrps for Weibull<ShapeLink, ScaleLink>
where
    ShapeLink: PositiveLink<f64>,
    ScaleLink: PositiveLink<f64>,
{
    fn crps<'obs>(&self, y: Self::Observation<'obs>, theta: Self::Theta) -> f64 {
        if y < 0.0
            || !y.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
            || theta.scale <= 0.0
            || !theta.scale.is_finite()
        {
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
}

#[cfg(feature = "rand")]
impl<Rng, ShapeLink, ScaleLink> CanSimulate<Rng> for Weibull<ShapeLink, ScaleLink>
where
    Rng: rand::Rng,
    ShapeLink: PositiveLink<f64>,
    ScaleLink: PositiveLink<f64>,
{
    fn sample(&self, rng: &mut Rng, theta: Self::Theta) -> f64 {
        if theta.shape <= 0.0
            || !theta.shape.is_finite()
            || theta.scale <= 0.0
            || !theta.scale.is_finite()
        {
            return f64::NAN;
        }

        rand_distr::Distribution::sample(
            &rand_distr::Weibull::new(theta.scale, theta.shape)
                .expect("validated weibull parameters must construct"),
            rng,
        )
    }
}

/// Weibull distribution with log links for shape and scale.
pub type DefaultWeibull = Weibull<Log, Log>;

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;
    use gamlss_core::{Family, HasCdf, HasCrps, HasQuantile};

    use super::{DefaultWeibull, WeibullTheta};
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn weibull_gradient_matches_finite_difference() {
        let family = DefaultWeibull::new();
        assert_gradient_matches_finite_difference::<_, 2>(&family, 1.7, [0.4, -0.2]);
    }

    #[test]
    fn weibull_rejects_invalid_domain_and_has_finite_nll_inside_domain() {
        let family = DefaultWeibull::new();
        let theta = WeibullTheta {
            shape: 1.5,
            scale: 0.8,
        };

        assert!(family.nll(1.7, theta).is_finite());
        assert!(family.nll(0.0, theta).is_infinite());
        assert!(
            family
                .nll(
                    1.7,
                    WeibullTheta {
                        shape: 0.0,
                        scale: theta.scale,
                    },
                )
                .is_infinite()
        );
    }

    #[test]
    fn weibull_cdf_matches_reference_points() {
        let family = DefaultWeibull::new();
        let theta = WeibullTheta {
            shape: 2.0,
            scale: 3.0,
        };

        assert_relative_eq!(
            family.cdf(theta.scale, theta),
            1.0 - (-1.0_f64).exp(),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            family.cdf(theta.scale * std::f64::consts::LN_2.sqrt(), theta),
            0.5,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn weibull_quantile_inverts_cdf() {
        let family = DefaultWeibull::new();
        let theta = WeibullTheta {
            shape: 2.0,
            scale: 3.0,
        };

        assert_relative_eq!(family.quantile(0.0, theta), 0.0, epsilon = 1.0e-12);
        assert!(family.quantile(1.0, theta).is_infinite());

        let y = family.quantile(0.75, theta);
        assert_relative_eq!(family.cdf(y, theta), 0.75, epsilon = 1.0e-12);
        assert!(family.quantile(f64::NAN, theta).is_nan());
        assert!(
            family
                .quantile(
                    0.5,
                    WeibullTheta {
                        shape: 0.0,
                        scale: 3.0
                    }
                )
                .is_nan()
        );
    }

    #[test]
    fn weibull_cdf_returns_nan_for_invalid_domains() {
        let family = DefaultWeibull::new();

        assert_relative_eq!(
            family.cdf(
                0.0,
                WeibullTheta {
                    shape: 2.0,
                    scale: 3.0
                }
            ),
            0.0,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            family.cdf(
                -1.0,
                WeibullTheta {
                    shape: 2.0,
                    scale: 3.0
                }
            ),
            0.0,
            epsilon = 1.0e-12
        );
        assert!(
            family
                .cdf(
                    f64::NAN,
                    WeibullTheta {
                        shape: 2.0,
                        scale: 3.0
                    }
                )
                .is_nan()
        );
        assert!(
            family
                .cdf(
                    1.0,
                    WeibullTheta {
                        shape: 0.0,
                        scale: 3.0
                    }
                )
                .is_nan()
        );
    }

    #[test]
    fn weibull_crps_matches_fixed_values() {
        let family = DefaultWeibull::new();

        assert_relative_eq!(
            family.crps(
                1.0,
                WeibullTheta {
                    shape: 1.0,
                    scale: 0.5,
                },
            ),
            0.385_335_283_236_612_7,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            family.crps(
                0.0,
                WeibullTheta {
                    shape: 1.0,
                    scale: 0.5,
                },
            ),
            0.25,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn weibull_crps_returns_nan_for_invalid_domains() {
        let family = DefaultWeibull::new();

        assert!(
            family
                .crps(
                    -1.0,
                    WeibullTheta {
                        shape: 2.0,
                        scale: 3.0,
                    },
                )
                .is_nan()
        );
        assert!(
            family
                .crps(
                    1.0,
                    WeibullTheta {
                        shape: 0.0,
                        scale: 3.0,
                    },
                )
                .is_nan()
        );
    }

    #[test]
    fn weibull_crps_is_nonnegative_for_valid_domains() {
        let family = DefaultWeibull::new();

        assert!(
            family.crps(
                1.0,
                WeibullTheta {
                    shape: 2.0,
                    scale: 3.0,
                },
            ) >= 0.0
        );
    }

    #[cfg(feature = "rand")]
    #[test]
    fn weibull_sampling_returns_finite_values_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = DefaultWeibull::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family.sample(
            &mut rng,
            WeibullTheta {
                shape: 1.5,
                scale: 0.8,
            },
        );
        assert!(sample >= 0.0 && sample.is_finite());
        assert!(
            family
                .sample(
                    &mut rng,
                    WeibullTheta {
                        shape: 0.0,
                        scale: 0.8
                    }
                )
                .is_nan()
        );
    }
}
