use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Family, HasCdf, HasQuantile, Log, Mu, ParameterParts, ParameterizedFamily, PositiveLink, Shape,
};

use crate::special::{invert_positive_cdf, unit_normal_cdf};

const HALF_LOG_2_PI: f64 = 0.918_938_533_204_672_7;

/// Inverse Gaussian family parameterized by positive mean and shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InverseGaussian<MuLink = Log, ShapeLink = Log> {
    marker: PhantomData<(MuLink, ShapeLink)>,
}

impl<MuLink, ShapeLink> InverseGaussian<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    /// Creates a stateless inverse Gaussian family.
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn theta_from_eta(eta: InverseGaussianEta) -> InverseGaussianTheta {
        InverseGaussianTheta {
            mu: MuLink::inverse(eta.mu),
            shape: ShapeLink::inverse(eta.shape),
        }
    }

    #[inline(always)]
    fn nll_theta(y: f64, theta: InverseGaussianTheta) -> f64 {
        if y <= 0.0
            || !y.is_finite()
            || theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
        {
            return f64::INFINITY;
        }

        let residual = y - theta.mu;
        HALF_LOG_2_PI + 1.5 * y.ln() - 0.5 * theta.shape.ln()
            + theta.shape * residual * residual / (2.0 * theta.mu * theta.mu * y)
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: InverseGaussianEta) -> (f64, InverseGaussianEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (
                nll,
                InverseGaussianEta {
                    mu: f64::NAN,
                    shape: f64::NAN,
                },
            );
        }

        let residual = y - theta.mu;
        let d_mu = -theta.shape * residual / (theta.mu * theta.mu * theta.mu);
        let d_shape = -0.5 / theta.shape + residual * residual / (2.0 * theta.mu * theta.mu * y);
        let gradient_eta = InverseGaussianEta {
            mu: d_mu * MuLink::derivative_inverse(eta.mu),
            shape: d_shape * ShapeLink::derivative_inverse(eta.shape),
        };

        (nll, gradient_eta)
    }
}

impl<MuLink, ShapeLink> Default for InverseGaussian<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Predictors for the inverse Gaussian family on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InverseGaussianEta {
    /// Mean predictor.
    pub mu: f64,
    /// Shape predictor.
    pub shape: f64,
}

impl ParameterParts<2> for InverseGaussianEta {
    #[inline(always)]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            mu: values[0],
            shape: values[1],
        }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mu,
            1 => self.shape,
            _ => unreachable!("inverse Gaussian eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale inverse Gaussian parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InverseGaussianTheta {
    /// Positive mean parameter.
    pub mu: f64,
    /// Positive shape parameter.
    pub shape: f64,
}

impl<MuLink, ShapeLink> Family for InverseGaussian<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    type Eta = InverseGaussianEta;
    type Theta = InverseGaussianTheta;
    type NllGradientEta = InverseGaussianEta;
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

impl<MuLink, ShapeLink> ParameterizedFamily<2> for InverseGaussian<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    type Params = (Mu, Shape);
    type Links = (MuLink, ShapeLink);
}

impl<MuLink, ShapeLink> HasCdf for InverseGaussian<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
        if !y.is_finite()
            || theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
        {
            return f64::NAN;
        }
        if y <= 0.0 {
            return 0.0;
        }

        let scale = (theta.shape / y).sqrt();
        let ratio = y / theta.mu;
        let first = unit_normal_cdf(scale * (ratio - 1.0));
        let log_multiplier = 2.0 * theta.shape / theta.mu;
        let tail = unit_normal_cdf(-scale * (ratio + 1.0));
        let second = if tail == 0.0 {
            0.0
        } else {
            (log_multiplier + tail.ln()).exp()
        };

        (first + second).clamp(0.0, 1.0)
    }
}

impl<MuLink, ShapeLink> HasQuantile for InverseGaussian<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
        if theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
        {
            return f64::NAN;
        }

        invert_positive_cdf(p, |y| self.cdf(y, theta))
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink, ShapeLink> CanSimulate<Rng> for InverseGaussian<MuLink, ShapeLink>
where
    Rng: rand::Rng,
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    fn sample(&self, rng: &mut Rng, theta: Self::Theta) -> f64 {
        if theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
        {
            return f64::NAN;
        }

        rand_distr::Distribution::sample(
            &rand_distr::InverseGaussian::new(theta.mu, theta.shape)
                .expect("validated inverse Gaussian parameters must construct"),
            rng,
        )
    }
}

/// Inverse Gaussian distribution with log links for mean and shape.
pub type DefaultInverseGaussian = InverseGaussian<Log, Log>;

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;
    use gamlss_core::{Family, HasCdf, HasQuantile};

    use super::{DefaultInverseGaussian, InverseGaussianTheta};
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn inverse_gaussian_gradient_matches_finite_difference() {
        let family = DefaultInverseGaussian::new();
        assert_gradient_matches_finite_difference::<_, 2>(&family, 1.7, [0.4, -0.2]);
    }

    #[test]
    fn inverse_gaussian_rejects_invalid_domain_and_has_finite_nll_inside_domain() {
        let family = DefaultInverseGaussian::new();
        let theta = InverseGaussianTheta {
            mu: 1.5,
            shape: 0.8,
        };

        assert!(family.nll(1.7, theta).is_finite());
        assert!(family.nll(0.0, theta).is_infinite());
        assert!(
            family
                .nll(
                    1.7,
                    InverseGaussianTheta {
                        mu: 0.0,
                        shape: theta.shape,
                    },
                )
                .is_infinite()
        );
    }

    #[test]
    fn inverse_gaussian_cdf_matches_reference_points() {
        let family = DefaultInverseGaussian::new();
        let theta = InverseGaussianTheta {
            mu: 1.0,
            shape: 1.0,
        };

        assert_relative_eq!(family.cdf(1.0, theta), 0.668_102, epsilon = 1.0e-6);
        assert_relative_eq!(family.cdf(0.5, theta), 0.364_975, epsilon = 1.0e-6);
    }

    #[test]
    fn inverse_gaussian_cdf_returns_nan_for_invalid_domains() {
        let family = DefaultInverseGaussian::new();

        assert_eq!(
            family.cdf(
                0.0,
                InverseGaussianTheta {
                    mu: 1.0,
                    shape: 1.0
                }
            ),
            0.0
        );
        assert_eq!(
            family.cdf(
                -1.0,
                InverseGaussianTheta {
                    mu: 1.0,
                    shape: 1.0
                }
            ),
            0.0
        );
        assert!(
            family
                .cdf(
                    f64::NAN,
                    InverseGaussianTheta {
                        mu: 1.0,
                        shape: 1.0
                    }
                )
                .is_nan()
        );
        assert!(
            family
                .cdf(
                    1.0,
                    InverseGaussianTheta {
                        mu: 0.0,
                        shape: 1.0
                    }
                )
                .is_nan()
        );
    }

    #[test]
    fn inverse_gaussian_cdf_is_finite_for_extreme_shape_ratio() {
        let family = DefaultInverseGaussian::new();
        let cdf = family.cdf(
            1.0,
            InverseGaussianTheta {
                mu: 1.0,
                shape: 1000.0,
            },
        );

        assert!(cdf.is_finite());
        assert!((0.0..=1.0).contains(&cdf));
    }

    #[test]
    fn inverse_gaussian_quantile_inverts_cdf() {
        let family = DefaultInverseGaussian::new();
        let theta = InverseGaussianTheta {
            mu: 1.5,
            shape: 0.8,
        };

        for p in [0.01, 0.1, 0.5, 0.9, 0.99] {
            let y = family.quantile(p, theta);
            assert_relative_eq!(family.cdf(y, theta), p, epsilon = 1.0e-10);
        }

        assert_eq!(family.quantile(0.0, theta), 0.0);
        assert_eq!(family.quantile(1.0, theta), f64::INFINITY);
        assert!(family.quantile(f64::NAN, theta).is_nan());
        assert!(
            family
                .quantile(
                    0.5,
                    InverseGaussianTheta {
                        mu: 0.0,
                        shape: 1.0,
                    },
                )
                .is_nan()
        );
    }

    #[cfg(feature = "rand")]
    #[test]
    fn inverse_gaussian_sampling_returns_positive_values_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = DefaultInverseGaussian::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family.sample(
            &mut rng,
            InverseGaussianTheta {
                mu: 1.5,
                shape: 0.8,
            },
        );

        assert!(sample > 0.0 && sample.is_finite());
        assert!(
            family
                .sample(
                    &mut rng,
                    InverseGaussianTheta {
                        mu: 0.0,
                        shape: 1.0,
                    },
                )
                .is_nan()
        );
    }
}
