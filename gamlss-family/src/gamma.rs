use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Family, HasCdf, HasCrps, HasQuantile, Log, ParameterParts, ParameterizedFamily, PositiveLink,
    Rate, Shape,
};

use crate::special::{digamma, invert_positive_cdf, ln_beta, ln_gamma, regularized_gamma_lower};

/// Gamma family parameterized by positive shape and rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gamma<ShapeLink = Log, RateLink = Log> {
    marker: PhantomData<(ShapeLink, RateLink)>,
}

impl<ShapeLink, RateLink> Gamma<ShapeLink, RateLink>
where
    ShapeLink: PositiveLink<f64>,
    RateLink: PositiveLink<f64>,
{
    /// Creates a stateless gamma family.
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn theta_from_eta(eta: GammaEta) -> GammaTheta {
        GammaTheta {
            shape: ShapeLink::inverse(eta.shape),
            rate: RateLink::inverse(eta.rate),
        }
    }

    #[inline(always)]
    fn nll_theta(y: f64, theta: GammaTheta) -> f64 {
        if y <= 0.0
            || !y.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
            || theta.rate <= 0.0
            || !theta.rate.is_finite()
        {
            return f64::INFINITY;
        }

        ln_gamma(theta.shape) - theta.shape * theta.rate.ln() - (theta.shape - 1.0) * y.ln()
            + theta.rate * y
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: GammaEta) -> (f64, GammaEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (
                nll,
                GammaEta {
                    shape: f64::NAN,
                    rate: f64::NAN,
                },
            );
        }

        let d_shape = digamma(theta.shape) - theta.rate.ln() - y.ln();
        let d_rate = y - theta.shape / theta.rate;
        let gradient_eta = GammaEta {
            shape: d_shape * ShapeLink::derivative_inverse(eta.shape),
            rate: d_rate * RateLink::derivative_inverse(eta.rate),
        };

        (nll, gradient_eta)
    }
}

impl<ShapeLink, RateLink> Default for Gamma<ShapeLink, RateLink>
where
    ShapeLink: PositiveLink<f64>,
    RateLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Predictors for the gamma family on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaEta {
    /// Shape predictor.
    pub shape: f64,
    /// Rate predictor.
    pub rate: f64,
}

impl ParameterParts<2> for GammaEta {
    #[inline(always)]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            shape: values[0],
            rate: values[1],
        }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.shape,
            1 => self.rate,
            _ => unreachable!("gamma eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale gamma parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaTheta {
    /// Positive shape parameter.
    pub shape: f64,
    /// Positive rate parameter.
    pub rate: f64,
}

impl<ShapeLink, RateLink> Family for Gamma<ShapeLink, RateLink>
where
    ShapeLink: PositiveLink<f64>,
    RateLink: PositiveLink<f64>,
{
    type Eta = GammaEta;
    type Theta = GammaTheta;
    type NllGradientEta = GammaEta;
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

impl<ShapeLink, RateLink> ParameterizedFamily<2> for Gamma<ShapeLink, RateLink>
where
    ShapeLink: PositiveLink<f64>,
    RateLink: PositiveLink<f64>,
{
    type Params = (Shape, Rate);
    type Links = (ShapeLink, RateLink);
}

impl<ShapeLink, RateLink> HasCdf for Gamma<ShapeLink, RateLink>
where
    ShapeLink: PositiveLink<f64>,
    RateLink: PositiveLink<f64>,
{
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
        if !y.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
            || theta.rate <= 0.0
            || !theta.rate.is_finite()
        {
            return f64::NAN;
        }
        if y <= 0.0 {
            return 0.0;
        }

        regularized_gamma_lower(theta.shape, theta.rate * y)
    }
}

impl<ShapeLink, RateLink> HasQuantile for Gamma<ShapeLink, RateLink>
where
    ShapeLink: PositiveLink<f64>,
    RateLink: PositiveLink<f64>,
{
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
        if theta.shape <= 0.0
            || !theta.shape.is_finite()
            || theta.rate <= 0.0
            || !theta.rate.is_finite()
        {
            return f64::NAN;
        }

        invert_positive_cdf(p, |y| self.cdf(y, theta))
    }
}

impl<ShapeLink, RateLink> HasCrps for Gamma<ShapeLink, RateLink>
where
    ShapeLink: PositiveLink<f64>,
    RateLink: PositiveLink<f64>,
{
    fn crps<'obs>(&self, y: Self::Observation<'obs>, theta: Self::Theta) -> f64 {
        if y < 0.0
            || !y.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
            || theta.rate <= 0.0
            || !theta.rate.is_finite()
        {
            return f64::NAN;
        }

        let f_shape = regularized_gamma_lower(theta.shape, theta.rate * y);
        let f_next_shape = regularized_gamma_lower(theta.shape + 1.0, theta.rate * y);
        let mean = theta.shape / theta.rate;
        let beta_term = ln_beta(theta.shape + 0.5, 0.5).exp() / (std::f64::consts::PI * theta.rate);

        y * (2.0 * f_shape - 1.0) - mean * (2.0 * f_next_shape - 1.0) - beta_term
    }
}

#[cfg(feature = "rand")]
impl<Rng, ShapeLink, RateLink> CanSimulate<Rng> for Gamma<ShapeLink, RateLink>
where
    Rng: rand::Rng,
    ShapeLink: PositiveLink<f64>,
    RateLink: PositiveLink<f64>,
{
    fn sample(&self, rng: &mut Rng, theta: Self::Theta) -> f64 {
        if theta.shape <= 0.0
            || !theta.shape.is_finite()
            || theta.rate <= 0.0
            || !theta.rate.is_finite()
        {
            return f64::NAN;
        }

        rand_distr::Distribution::sample(
            &rand_distr::Gamma::new(theta.shape, 1.0 / theta.rate)
                .expect("validated gamma parameters must construct"),
            rng,
        )
    }
}

/// Gamma distribution with log links for shape and rate.
pub type DefaultGamma = Gamma<Log, Log>;

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;
    use gamlss_core::{Family, HasCdf, HasCrps, HasQuantile, Link, Log};
    use statrs::distribution::{ContinuousCDF, Gamma as StatrsGamma};

    use super::{DefaultGamma, GammaTheta};
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn gamma_gradient_matches_finite_difference() {
        let family = DefaultGamma::new();
        assert_gradient_matches_finite_difference::<_, 2>(&family, 1.7, [0.4, -0.2]);
    }

    #[test]
    fn gamma_rejects_invalid_domain_and_has_finite_nll_inside_domain() {
        let family = DefaultGamma::new();
        let theta = GammaTheta {
            shape: Log::inverse(0.4),
            rate: Log::inverse(-0.2),
        };

        assert!(family.nll(1.7, theta).is_finite());
        assert!(family.nll(0.0, theta).is_infinite());
        assert!(
            family
                .nll(
                    1.7,
                    GammaTheta {
                        shape: 0.0,
                        rate: theta.rate,
                    },
                )
                .is_infinite()
        );
    }

    #[test]
    fn gamma_cdf_and_quantile_match_statrs_reference() {
        let family = DefaultGamma::new();
        let theta = GammaTheta {
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
    fn gamma_cdf_and_quantile_handle_boundaries_and_invalid_domains() {
        let family = DefaultGamma::new();
        let theta = GammaTheta {
            shape: 2.0,
            rate: 3.0,
        };

        assert_eq!(family.cdf(0.0, theta), 0.0);
        assert_eq!(family.quantile(0.0, theta), 0.0);
        assert_eq!(family.quantile(1.0, theta), f64::INFINITY);
        assert!(family.quantile(f64::NAN, theta).is_nan());
        assert!(
            family
                .cdf(
                    1.0,
                    GammaTheta {
                        shape: 0.0,
                        rate: 1.0,
                    },
                )
                .is_nan()
        );
    }

    #[test]
    fn gamma_crps_matches_fixed_values() {
        let family = DefaultGamma::new();

        assert_relative_eq!(
            family.crps(
                1.0,
                GammaTheta {
                    shape: 1.0,
                    rate: 2.0,
                },
            ),
            0.385_335_283_236_612_7,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            family.crps(
                0.0,
                GammaTheta {
                    shape: 1.0,
                    rate: 2.0,
                },
            ),
            0.25,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn gamma_crps_returns_nan_for_invalid_domains() {
        let family = DefaultGamma::new();

        assert!(
            family
                .crps(
                    -1.0,
                    GammaTheta {
                        shape: 2.0,
                        rate: 3.0,
                    },
                )
                .is_nan()
        );
        assert!(
            family
                .crps(
                    1.0,
                    GammaTheta {
                        shape: 0.0,
                        rate: 3.0,
                    },
                )
                .is_nan()
        );
    }

    #[test]
    fn gamma_crps_is_nonnegative_for_valid_domains() {
        let family = DefaultGamma::new();

        assert!(
            family.crps(
                1.0,
                GammaTheta {
                    shape: 2.0,
                    rate: 3.0,
                },
            ) >= 0.0
        );
    }

    #[cfg(feature = "rand")]
    #[test]
    fn gamma_sampling_returns_finite_values_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = DefaultGamma::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        assert!(
            family
                .sample(
                    &mut rng,
                    GammaTheta {
                        shape: 2.0,
                        rate: 3.0
                    }
                )
                .is_finite()
        );
        assert!(
            family
                .sample(
                    &mut rng,
                    GammaTheta {
                        shape: 0.0,
                        rate: 3.0
                    }
                )
                .is_nan()
        );
    }
}
