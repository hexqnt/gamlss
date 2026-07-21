use std::marker::PhantomData;

use gamlss_core::{Log, PositiveLink};
use gamlss_special::{
    integrate_finite, invert_positive_cdf, unit_normal_cdf, unit_normal_log_sf, unit_normal_sf,
};

use crate::constants::HALF_LOG_2_PI;
use crate::domain::{ScalarObservationDomain, is_positive_finite};

pub use mean_cv::{
    InverseGaussianCv, InverseGaussianMeanCv, InverseGaussianMeanCvEta, InverseGaussianMeanCvTheta,
};
pub use mean_shape::{
    InverseGaussianEta, InverseGaussianMeanShape, InverseGaussianMuShape, InverseGaussianTheta,
};

mod mean_cv;
mod mean_shape;

/// Inverse Gaussian family parameterized by positive mean and shape.
///
/// For mean $\mu>0$ and shape $\lambda>0$, the density is
///
/// $$
/// f(y\mid\mu,\lambda)=
/// \sqrt{\frac{\lambda}{2\pi y^3}}
/// \exp\\!\left[-\frac{\lambda(y-\mu)^2}{2\mu^2y}\right],
/// \qquad y>0.
/// $$
///
/// The moments are $\mathbb{E}(Y)=\mu$ and $\operatorname{Var}(Y)=\mu^3/\lambda$.
///
/// The canonical Rust carrier retains the generic name `shape` for $\lambda$: [`InverseGaussianTheta::mu`] stores $\mu$ and [`InverseGaussianTheta::shape`] stores $\lambda$.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/distributions/inverse_gaussian.svg")
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InverseGaussian<MuLink = Log, ShapeLink = Log> {
    marker: PhantomData<(MuLink, ShapeLink)>,
}

impl<MuLink, ShapeLink> ScalarObservationDomain for InverseGaussian<MuLink, ShapeLink> {
    #[inline]
    fn observation_in_domain(&self, observation: f64) -> bool {
        is_positive_finite(observation)
    }
}

impl<MeanLink, CvLink> ScalarObservationDomain for InverseGaussianCv<MeanLink, CvLink> {
    #[inline]
    fn observation_in_domain(&self, observation: f64) -> bool {
        is_positive_finite(observation)
    }
}

impl<MuLink, ShapeLink> InverseGaussian<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    /// Creates a stateless inverse Gaussian family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

/// Link- and parameterization-independent inverse-Gaussian mean/shape kernel.
#[derive(Debug, Clone, Copy)]
pub(super) struct InverseGaussianKernel;

impl InverseGaussianKernel {
    #[inline]
    fn valid_theta(theta: InverseGaussianTheta) -> bool {
        theta.mu > 0.0 && theta.mu.is_finite() && theta.shape > 0.0 && theta.shape.is_finite()
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    pub(super) fn nll_theta(y: f64, theta: InverseGaussianTheta) -> f64 {
        if y <= 0.0 || !y.is_finite() || !Self::valid_theta(theta) {
            return f64::INFINITY;
        }

        let residual = y - theta.mu;
        HALF_LOG_2_PI + 1.5 * y.ln() - 0.5 * theta.shape.ln()
            + theta.shape * residual * residual / (2.0 * theta.mu * theta.mu * y)
    }

    pub(super) fn cdf_theta(y: f64, theta: InverseGaussianTheta) -> f64 {
        if !y.is_finite() || !Self::valid_theta(theta) {
            return f64::NAN;
        }
        if y <= 0.0 {
            return 0.0;
        }

        let scale = (theta.shape / y).sqrt();
        let ratio = y / theta.mu;
        let first_argument = scale * (ratio - 1.0);
        let first = unit_normal_cdf(first_argument);
        let log_multiplier = 2.0 * theta.shape / theta.mu;
        let second = (log_multiplier + unit_normal_log_sf(scale * (ratio + 1.0))).exp();

        (first + second).clamp(0.0, 1.0)
    }

    pub(super) fn quantile_theta(p: f64, theta: InverseGaussianTheta) -> f64 {
        if !Self::valid_theta(theta) {
            return f64::NAN;
        }

        invert_positive_cdf(p, |y| Self::cdf_theta(y, theta))
    }

    pub(super) fn crps_theta(y: f64, theta: InverseGaussianTheta) -> f64 {
        if y < 0.0 || !y.is_finite() || !Self::valid_theta(theta) {
            return f64::NAN;
        }

        let left = integrate_finite(0.0, y, |x| {
            let cdf = Self::cdf_theta(x, theta);
            cdf * cdf
        });
        let right = integrate_finite(0.0, 1.0, |u| {
            #[allow(clippy::float_cmp)]
            if u == 1.0 {
                return 0.0;
            }

            let one_minus_u = 1.0 - u;
            let x = y + u / one_minus_u;
            let scale = (theta.shape / x).sqrt();
            let ratio = x / theta.mu;
            let first_survival = unit_normal_sf(scale * (ratio - 1.0));
            let second =
                (2.0 * theta.shape / theta.mu + unit_normal_log_sf(scale * (ratio + 1.0))).exp();
            let survival = (first_survival - second).max(0.0);
            survival * survival / (one_minus_u * one_minus_u)
        });

        left + right
    }

    #[cfg(feature = "rand")]
    pub(super) fn try_sample<Rng>(
        rng: &mut Rng,
        theta: InverseGaussianTheta,
    ) -> Result<f64, gamlss_core::SimulationError>
    where
        Rng: rand::Rng,
    {
        if !Self::valid_theta(theta) {
            return Err(gamlss_core::SimulationError::InvalidParameters(
                "Inverse Gaussian theta",
            ));
        }

        let distribution =
            rand_distr::InverseGaussian::new(theta.mu, theta.shape).map_err(|_| {
                gamlss_core::SimulationError::BackendRejected("Inverse Gaussian mean/shape")
            })?;
        crate::simulation::ensure_finite(
            rand_distr::Distribution::sample(&distribution, rng),
            "Inverse Gaussian sample",
        )
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

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    #[cfg(feature = "rand")]
    use gamlss_core::TrySimulate;
    use gamlss_core::{Family, HasCdf, HasCrps, HasQuantile};

    use super::{
        InverseGaussianMeanCv, InverseGaussianMeanCvEta, InverseGaussianMeanCvTheta,
        InverseGaussianMuShape, InverseGaussianTheta,
    };
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn inverse_gaussian_gradient_matches_finite_difference() {
        let family = InverseGaussianMuShape::new();
        assert_gradient_matches_finite_difference::<_, 2>(&family, 1.7, [0.4, -0.2]);

        let mean_cv = InverseGaussianMeanCv::new();
        assert_gradient_matches_finite_difference::<_, 2>(
            &mean_cv,
            1.7,
            [1.5_f64.ln(), 0.5_f64.ln()],
        );
    }

    #[test]
    fn inverse_gaussian_mean_cv_matches_mean_shape_equivalent() {
        let mean_shape = InverseGaussianMuShape::new();
        let mean_cv = InverseGaussianMeanCv::new();
        let mean_shape_theta = InverseGaussianTheta {
            mu: 1.5,
            shape: 6.0,
        };
        let mean_cv_theta = InverseGaussianMeanCvTheta {
            mean: mean_shape_theta.mu,
            cv: (mean_shape_theta.mu / mean_shape_theta.shape).sqrt(),
        };

        assert_relative_eq!(
            mean_cv.nll(1.7, &mean_cv_theta, &mut mean_cv.workspace()),
            mean_shape.nll(1.7, &mean_shape_theta, &mut mean_shape.workspace()),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean_cv.cdf(1.7, &mean_cv_theta),
            mean_shape.cdf(1.7, &mean_shape_theta),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean_cv.quantile(0.4, &mean_cv_theta),
            mean_shape.quantile(0.4, &mean_shape_theta),
            epsilon = 1.0e-10
        );
    }

    #[test]
    fn inverse_gaussian_mean_cv_accepts_shape_beyond_squared_cv_range() {
        let family = InverseGaussianMeanCv::new();
        let theta = InverseGaussianMeanCvTheta {
            mean: 1.0,
            cv: 1.0e155,
        };
        let eta = InverseGaussianMeanCvEta {
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
            (gradient.mean + 0.5).abs() < 1.0e-14,
            "mean gradient was {}",
            gradient.mean
        );
        assert!(
            (gradient.cv - 1.0).abs() < 1.0e-14,
            "cv gradient was {}",
            gradient.cv
        );
    }

    #[test]
    fn inverse_gaussian_rejects_invalid_domain_and_has_finite_nll_inside_domain() {
        let family = InverseGaussianMuShape::new();
        let theta = InverseGaussianTheta {
            mu: 1.5,
            shape: 0.8,
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
                    &InverseGaussianTheta {
                        mu: 0.0,
                        shape: theta.shape,
                    },
                    &mut family.workspace(),
                )
                .is_infinite()
        );
    }

    #[test]
    fn inverse_gaussian_cdf_matches_reference_points() {
        let family = InverseGaussianMuShape::new();
        let theta = InverseGaussianTheta {
            mu: 1.0,
            shape: 1.0,
        };

        assert_relative_eq!(family.cdf(1.0, &theta), 0.668_102, epsilon = 1.0e-6);
        assert_relative_eq!(family.cdf(0.5, &theta), 0.364_975, epsilon = 1.0e-6);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn inverse_gaussian_cdf_returns_nan_for_invalid_domains() {
        let family = InverseGaussianMuShape::new();

        assert_eq!(
            family.cdf(
                0.0,
                &InverseGaussianTheta {
                    mu: 1.0,
                    shape: 1.0
                }
            ),
            0.0
        );
        assert_eq!(
            family.cdf(
                -1.0,
                &InverseGaussianTheta {
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
                    &InverseGaussianTheta {
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
                    &InverseGaussianTheta {
                        mu: 0.0,
                        shape: 1.0
                    }
                )
                .is_nan()
        );
    }

    #[test]
    fn inverse_gaussian_cdf_is_finite_for_extreme_shape_ratio() {
        let family = InverseGaussianMuShape::new();
        let cdf = family.cdf(
            1.0,
            &InverseGaussianTheta {
                mu: 1.0,
                shape: 1000.0,
            },
        );

        assert!(cdf.is_finite());
        assert!((0.0..=1.0).contains(&cdf));
        assert!(cdf > 0.506 && cdf < 0.507, "extreme-ratio CDF was {cdf}");
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn inverse_gaussian_quantile_inverts_cdf() {
        let family = InverseGaussianMuShape::new();
        let theta = InverseGaussianTheta {
            mu: 1.5,
            shape: 0.8,
        };

        for p in [0.01, 0.1, 0.5, 0.9, 0.99] {
            let y = family.quantile(p, &theta);
            assert_relative_eq!(family.cdf(y, &theta), p, epsilon = 1.0e-10);
        }

        assert_eq!(family.quantile(0.0, &theta), 0.0);
        assert_eq!(family.quantile(1.0, &theta), f64::INFINITY);
        assert!(family.quantile(f64::NAN, &theta).is_nan());
        assert!(
            family
                .quantile(
                    0.5,
                    &InverseGaussianTheta {
                        mu: 0.0,
                        shape: 1.0,
                    },
                )
                .is_nan()
        );
    }

    #[test]
    fn inverse_gaussian_crps_matches_fixed_values() {
        let family = InverseGaussianMuShape::new();

        assert_relative_eq!(
            family.crps(
                1.0,
                &InverseGaussianTheta {
                    mu: 1.0,
                    shape: 1.0,
                },
            ),
            0.215_550_872_022_949_15,
            epsilon = 1.0e-7
        );
        assert_relative_eq!(
            family.crps(
                0.0,
                &InverseGaussianTheta {
                    mu: 1.0,
                    shape: 1.0,
                },
            ),
            0.543_142_867_130_266_3,
            epsilon = 1.0e-6
        );
    }

    #[test]
    fn inverse_gaussian_crps_returns_nan_for_invalid_domains() {
        let family = InverseGaussianMuShape::new();

        assert!(
            family
                .crps(
                    -1.0,
                    &InverseGaussianTheta {
                        mu: 1.0,
                        shape: 1.0,
                    },
                )
                .is_nan()
        );
        assert!(
            family
                .crps(
                    1.0,
                    &InverseGaussianTheta {
                        mu: 0.0,
                        shape: 1.0,
                    },
                )
                .is_nan()
        );
    }

    #[test]
    fn inverse_gaussian_crps_is_nonnegative_for_valid_domains() {
        let family = InverseGaussianMuShape::new();

        assert!(
            family.crps(
                1.0,
                &InverseGaussianTheta {
                    mu: 1.0,
                    shape: 1.0,
                },
            ) >= 0.0
        );
    }

    #[cfg(feature = "rand")]
    #[test]
    fn inverse_gaussian_sampling_returns_positive_values_and_errors_for_invalid_theta() {
        use rand::SeedableRng;

        let family = InverseGaussianMuShape::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family
            .try_sample(
                &mut rng,
                &InverseGaussianTheta {
                    mu: 1.5,
                    shape: 0.8,
                },
            )
            .unwrap();

        assert!(sample > 0.0 && sample.is_finite());
        assert!(
            family
                .try_sample(
                    &mut rng,
                    &InverseGaussianTheta {
                        mu: 0.0,
                        shape: 1.0,
                    },
                )
                .is_err()
        );
    }
}
