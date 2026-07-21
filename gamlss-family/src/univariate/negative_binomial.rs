use std::marker::PhantomData;

use gamlss_core::{Log, PositiveLink};

use gamlss_special::{
    bernoulli_kl, digamma_minus_ln, included_count, is_nonnegative_integer, ln_gamma,
    ln_gamma_delta, ln_gamma_stirling_residual, log_add_exp,
};

pub use mean_dispersion::{
    NegativeBinomialMeanDispersion, NegativeBinomialMeanDispersionEta,
    NegativeBinomialMeanDispersionTheta,
};
pub use mean_size::{NegativeBinomialEta, NegativeBinomialMeanSize};

mod mean_dispersion;
mod mean_size;

const MAX_CDF_TERMS: u64 = 1_000_000;

/// Negative binomial family parameterized by mean $\mu>0$ and shape $r>0$.
///
/// For a count $y\in\\{0,1,2,\ldots\\}$, its probability mass is
///
/// $$
/// \Pr(Y=y\mid\mu,r)
/// = \frac{\Gamma(y+r)}{\Gamma(r)\\,\Gamma(y+1)}
///   \left(\frac{r}{r+\mu}\right)^r
///   \left(\frac{\mu}{r+\mu}\right)^y.
/// $$
///
/// Here $\Gamma$ is the gamma function and $r$ is commonly called the size parameter.
///
/// The natural-scale moments are
///
/// $$
/// \mathbb{E}(Y)=\mu,
/// \qquad
/// \operatorname{Var}(Y)=\mu+\frac{\mu^2}{r}.
/// $$
///
/// The default [`NegativeBinomialMeanSize`] alias uses log links for both parameters.
///
/// The implementation calls $r$ “shape”: [`NegativeBinomialTheta::mu`] stores $\mu$, [`NegativeBinomialTheta::shape`] stores $r$, and the matching [`NegativeBinomialEta`] fields hold $\eta_\mu,\eta_r$.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/distributions/negative_binomial.svg")
)]
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NegativeBinomial<MuLink = Log, ShapeLink = Log> {
    marker: PhantomData<(MuLink, ShapeLink)>,
}

impl<MuLink, ShapeLink> NegativeBinomial<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    /// Creates a stateless negative binomial family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

/// Link-independent mean/size kernel shared by negative-binomial parameterizations.
#[derive(Debug, Clone, Copy)]
pub(super) struct NegativeBinomialKernel;

impl NegativeBinomialKernel {
    #[inline]
    #[allow(clippy::suboptimal_flops)]
    pub(super) fn nll_theta(y: f64, theta: NegativeBinomialTheta) -> f64 {
        if !is_nonnegative_integer(y)
            || theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
        {
            return f64::INFINITY;
        }

        if y == 0.0 {
            return theta.shape * Self::log1p_ratio(theta.mu, theta.shape);
        }

        let total = theta.shape + y;
        if total.is_finite() {
            let count_fraction = theta.shape / total;
            let success_probability = (-Self::log1p_ratio(theta.mu, theta.shape)).exp();
            if (0.0..1.0).contains(&count_fraction) && (0.0..1.0).contains(&success_probability) {
                return ln_gamma_stirling_residual(theta.shape) + ln_gamma_stirling_residual(y)
                    - ln_gamma_stirling_residual(total)
                    + y.ln()
                    + total * bernoulli_kl(count_fraction, success_probability);
            }
        }

        -ln_gamma_delta(theta.shape, y)
            + ln_gamma(y + 1.0)
            + theta.shape * Self::log1p_ratio(theta.mu, theta.shape)
            + y * Self::log1p_ratio(theta.shape, theta.mu)
    }

    #[inline]
    fn log1p_ratio(numerator: f64, denominator: f64) -> f64 {
        log_add_exp(0.0, numerator.ln() - denominator.ln())
    }

    #[inline]
    fn reciprocal_sum(left: f64, right: f64) -> f64 {
        let scale = left.max(right);
        (1.0 / scale) / (left / scale + right / scale)
    }

    fn log1p_ratio_minus_fraction(numerator: f64, denominator: f64) -> f64 {
        const LOG_ONE_QUARTER: f64 = -1.386_294_361_119_890_6;

        let log_ratio = numerator.ln() - denominator.ln();
        if log_ratio <= LOG_ONE_QUARTER {
            let ratio = log_ratio.exp();
            let mut power = ratio * ratio;
            let mut sum = 0.0;
            for order in 2..=128 {
                let order_f = f64::from(order);
                let term = power * (order_f - 1.0) / order_f;
                sum += term;
                if term.abs() <= f64::EPSILON * sum.abs() {
                    break;
                }
                power *= -ratio;
            }
            sum
        } else {
            Self::log1p_ratio(numerator, denominator) - 1.0 / (1.0 + (-log_ratio).exp())
        }
    }

    #[inline]
    pub(super) fn gradient_theta(y: f64, theta: NegativeBinomialTheta) -> NegativeBinomialTheta {
        let success_probability = (-Self::log1p_ratio(theta.mu, theta.shape)).exp();
        let d_mu = (1.0 - y / theta.mu) * success_probability;
        let d_shape = if y == 0.0 {
            Self::log1p_ratio_minus_fraction(theta.mu, theta.shape)
        } else {
            let total = theta.shape + y;
            let residual_difference = digamma_minus_ln(theta.shape) - digamma_minus_ln(total);
            let log_sum_difference =
                Self::log1p_ratio(theta.mu, theta.shape) - Self::log1p_ratio(y, theta.shape);
            let ratio_difference = (y - theta.mu) * Self::reciprocal_sum(theta.shape, theta.mu);
            residual_difference + log_sum_difference + ratio_difference
        };

        NegativeBinomialTheta {
            mu: d_mu,
            shape: d_shape,
        }
    }

    #[inline]
    pub(crate) fn cdf_theta(y: f64, theta: NegativeBinomialTheta) -> f64 {
        if !y.is_finite()
            || theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
        {
            return f64::NAN;
        }
        if y < 0.0 {
            return 0.0;
        }

        let Some(max_count) = included_count(y, MAX_CDF_TERMS) else {
            return f64::NAN;
        };
        let log_success = -Self::log1p_ratio(theta.mu, theta.shape);
        let log_failure = -Self::log1p_ratio(theta.shape, theta.mu);
        let failure_probability = log_failure.exp();
        let term = (theta.shape * log_success).exp();
        if term.is_finite() && term > 0.0 {
            return Self::cdf_by_recurrence(theta.shape, failure_probability, max_count, term);
        }

        Self::cdf_by_log_sum(theta.shape, log_success, log_failure, max_count)
    }

    #[allow(clippy::cast_precision_loss)]
    fn cdf_by_recurrence(
        shape: f64,
        failure_probability: f64,
        max_count: u64,
        mut term: f64,
    ) -> f64 {
        let mut sum = term;
        for count in 1..=max_count {
            let previous_count = (count - 1) as f64;
            term *= ((previous_count + shape) / count as f64) * failure_probability;
            sum += term;
            if term <= f64::EPSILON * sum {
                break;
            }
        }

        sum.clamp(0.0, 1.0)
    }

    #[allow(clippy::suboptimal_flops, clippy::cast_precision_loss)]
    fn cdf_by_log_sum(shape: f64, log_success: f64, log_failure: f64, max_count: u64) -> f64 {
        let mut log_sum = f64::NEG_INFINITY;

        for count in 0..=max_count {
            let count_f = count as f64;
            let log_term = ln_gamma_delta(shape, count_f) - ln_gamma(count_f + 1.0)
                + shape * log_success
                + count_f * log_failure;
            log_sum = log_add_exp(log_sum, log_term);
        }

        log_sum.exp().clamp(0.0, 1.0)
    }

    #[cfg(feature = "rand")]
    pub(super) fn try_sample<Rng>(
        rng: &mut Rng,
        theta: NegativeBinomialTheta,
    ) -> Result<f64, gamlss_core::SimulationError>
    where
        Rng: rand::Rng,
    {
        if theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
        {
            return Err(gamlss_core::SimulationError::InvalidParameters(
                "Negative binomial theta",
            ));
        }

        let mixing = rand_distr::Gamma::new(theta.shape, theta.mu / theta.shape).map_err(|_| {
            gamlss_core::SimulationError::BackendRejected("Negative binomial gamma mixture")
        })?;
        let lambda = rand_distr::Distribution::sample(&mixing, rng);
        let count = rand_distr::Poisson::new(lambda).map_err(|_| {
            gamlss_core::SimulationError::BackendRejected("Negative binomial Poisson mean")
        })?;
        Ok(rand_distr::Distribution::sample(&count, rng))
    }
}

impl<MuLink, ShapeLink> Default for NegativeBinomial<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Natural-scale negative binomial parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NegativeBinomialTheta {
    /// Positive mean parameter.
    pub mu: f64,
    /// Positive shape parameter.
    pub shape: f64,
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "rand")]
    use gamlss_core::TrySimulate;
    use gamlss_core::{Family, HasCdf, HasQuantile};
    use statrs::distribution::{DiscreteCDF, NegativeBinomial as StatrsNegativeBinomial};

    use super::{
        NegativeBinomialEta, NegativeBinomialMeanDispersion, NegativeBinomialMeanDispersionTheta,
        NegativeBinomialMeanSize, NegativeBinomialTheta,
    };
    use crate::test_support::{
        assert_gradient_matches_finite_difference, statrs_discrete_quantile,
    };

    #[test]
    fn negative_binomial_gradient_matches_finite_difference() {
        let family = NegativeBinomialMeanSize::new();
        assert_gradient_matches_finite_difference::<_, 2>(&family, 3.0, [0.4, -0.2]);

        let mean_dispersion = NegativeBinomialMeanDispersion::new();
        assert_gradient_matches_finite_difference::<_, 2>(
            &mean_dispersion,
            3.0,
            [2.0_f64.ln(), 0.4_f64.ln()],
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn negative_binomial_mean_dispersion_matches_mean_size_equivalent() {
        let mean_size = NegativeBinomialMeanSize::new();
        let mean_dispersion = NegativeBinomialMeanDispersion::new();
        let mean_size_theta = NegativeBinomialTheta {
            mu: 2.0,
            shape: 4.0,
        };
        let mean_dispersion_theta = NegativeBinomialMeanDispersionTheta {
            mean: mean_size_theta.mu,
            dispersion: 1.0 / mean_size_theta.shape,
        };

        assert_eq!(
            mean_dispersion.nll(
                3.0,
                &mean_dispersion_theta,
                &mut mean_dispersion.workspace()
            ),
            mean_size.nll(3.0, &mean_size_theta, &mut mean_size.workspace())
        );
        assert_eq!(
            mean_dispersion.cdf(3.0, &mean_dispersion_theta),
            mean_size.cdf(3.0, &mean_size_theta)
        );
        assert_eq!(
            mean_dispersion.quantile(0.5, &mean_dispersion_theta),
            mean_size.quantile(0.5, &mean_size_theta)
        );
    }

    #[test]
    fn negative_binomial_rejects_invalid_domain_and_has_finite_nll_inside_domain() {
        let family = NegativeBinomialMeanSize::new();
        let theta = NegativeBinomialTheta {
            mu: 2.0,
            shape: 1.5,
        };

        assert!(family.nll(3.0, &theta, &mut family.workspace()).is_finite());
        assert!(
            family
                .nll(-1.0, &theta, &mut family.workspace())
                .is_infinite()
        );
        assert!(
            family
                .nll(1.5, &theta, &mut family.workspace())
                .is_infinite()
        );
        assert!(
            family
                .nll(
                    3.0,
                    &NegativeBinomialTheta {
                        mu: 0.0,
                        shape: theta.shape,
                    },
                    &mut family.workspace(),
                )
                .is_infinite()
        );
    }

    #[test]
    #[allow(clippy::float_cmp, clippy::cast_precision_loss)]
    fn negative_binomial_cdf_matches_reference_points() {
        let family = NegativeBinomialMeanSize::new();
        let theta = NegativeBinomialTheta {
            mu: 2.0,
            shape: 1.5,
        };
        let p = theta.shape / (theta.shape + theta.mu);
        let q = theta.mu / (theta.shape + theta.mu);
        let p0 = p.powf(theta.shape);
        let p1 = p0 * theta.shape * q;

        assert_eq!(family.cdf(-1.0, &theta), 0.0);
        assert!((family.cdf(0.0, &theta) - p0).abs() < 1.0e-12);
        assert!((family.cdf(1.5, &theta) - (p0 + p1)).abs() < 1.0e-12);
        assert!(
            family
                .cdf(
                    1.0,
                    &NegativeBinomialTheta {
                        mu: 0.0,
                        shape: theta.shape,
                    },
                )
                .is_nan()
        );
        assert!(
            family
                .cdf((super::MAX_CDF_TERMS + 1) as f64, &theta)
                .is_nan()
        );
    }

    #[test]
    fn negative_binomial_cdf_is_stable_for_large_parameters() {
        let family = NegativeBinomialMeanSize::new();
        let cdf = family.cdf(
            1000.0,
            &NegativeBinomialTheta {
                mu: 1000.0,
                shape: 1000.0,
            },
        );

        assert!(cdf.is_finite());
        assert!(cdf > 0.45 && cdf < 0.55, "cdf was {cdf}");
    }

    #[test]
    fn negative_binomial_preserves_poisson_limit_for_huge_shape() {
        let family = NegativeBinomialMeanSize::new();
        let theta = NegativeBinomialTheta {
            mu: 1.0,
            shape: 1.0e16,
        };

        let nll_at_zero = family.nll(0.0, &theta, &mut family.workspace());
        let nll_at_one = family.nll(1.0, &theta, &mut family.workspace());
        let cdf_at_zero = family.cdf(0.0, &theta);
        let cdf_at_one = family.cdf(1.0, &theta);
        let (_, gradient) = family.nll_and_gradient_eta(
            0.0,
            &NegativeBinomialEta {
                mu: theta.mu.ln(),
                shape: theta.shape.ln(),
            },
            &mut family.workspace(),
        );

        assert!(
            (nll_at_zero - 1.0).abs() < 1.0e-14,
            "nll(0) was {nll_at_zero}"
        );
        assert!(
            (nll_at_one - 1.0).abs() < 1.0e-14,
            "nll(1) was {nll_at_one}"
        );
        assert!(
            (cdf_at_zero - (-1.0_f64).exp()).abs() < 1.0e-14,
            "cdf(0) was {cdf_at_zero}"
        );
        let poisson_cdf_at_one = 2.0 * (-1.0_f64).exp();
        assert!(
            (cdf_at_one - poisson_cdf_at_one).abs() < 1.0e-14,
            "cdf(1) was {cdf_at_one}"
        );
        assert!(
            (gradient.mu - 1.0).abs() < 1.0e-14,
            "mu gradient was {}",
            gradient.mu
        );
        assert!(
            gradient.shape.abs() < 1.0e-14,
            "shape gradient was {}",
            gradient.shape
        );
    }

    #[test]
    fn concentrated_negative_binomial_preserves_normalizer_and_shape_gradient() {
        let family = NegativeBinomialMeanSize::new();
        let eta = NegativeBinomialEta {
            mu: 1.0e16_f64.ln(),
            shape: 1.0e16_f64.ln(),
        };
        let size = eta.shape.exp();
        let expected_nll = 2.0_f64.mul_add(
            gamlss_special::ln_gamma_stirling_residual(size),
            size.ln() - gamlss_special::ln_gamma_stirling_residual(2.0 * size),
        );
        let (nll, gradient) = family.nll_and_gradient_eta(size, &eta, &mut ());

        assert!((nll - expected_nll).abs() < 1.0e-13, "nll was {nll}");
        assert!(
            gradient.mu.abs() < 1.0e-14,
            "mu gradient was {}",
            gradient.mu
        );
        assert!(
            (gradient.shape + 0.25).abs() < 1.0e-14,
            "shape gradient was {}",
            gradient.shape
        );
    }

    #[test]
    #[allow(clippy::float_cmp, clippy::cast_precision_loss)]
    fn negative_binomial_quantile_matches_statrs_reference() {
        let family = NegativeBinomialMeanSize::new();
        let theta = NegativeBinomialTheta {
            mu: 2.0,
            shape: 1.5,
        };
        let success_probability = theta.shape / (theta.shape + theta.mu);
        let reference = StatrsNegativeBinomial::new(theta.shape, success_probability).unwrap();

        for p in [0.01, 0.1, 0.5, 0.9, 0.99] {
            assert_eq!(
                family.quantile(p, &theta),
                statrs_discrete_quantile(p, |count| reference.cdf(count)) as f64
            );
        }

        assert_eq!(family.quantile(0.0, &theta), 0.0);
        assert!(family.quantile(f64::NAN, &theta).is_nan());
        assert!(
            family
                .quantile(
                    0.5,
                    &NegativeBinomialTheta {
                        mu: 0.0,
                        shape: 1.5,
                    },
                )
                .is_nan()
        );
    }

    #[test]
    fn negative_binomial_quantile_is_generalized_inverse_cdf() {
        let family = NegativeBinomialMeanSize::new();
        let theta = NegativeBinomialTheta {
            mu: 6.0,
            shape: 2.5,
        };

        for p in [0.01, 0.1, 0.5, 0.9, 0.99] {
            let q = family.quantile(p, &theta);
            assert!(family.cdf(q, &theta) >= p);
            if q > 0.0 {
                assert!(family.cdf(q - 1.0, &theta) < p);
            }
        }
    }

    #[cfg(feature = "rand")]
    #[test]
    fn negative_binomial_sampling_returns_counts_and_errors_for_invalid_theta() {
        use rand::SeedableRng;

        let family = NegativeBinomialMeanSize::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family
            .try_sample(
                &mut rng,
                &NegativeBinomialTheta {
                    mu: 2.0,
                    shape: 1.5,
                },
            )
            .unwrap();
        assert!(sample >= 0.0 && sample.fract() == 0.0);
        assert!(
            family
                .try_sample(
                    &mut rng,
                    &NegativeBinomialTheta {
                        mu: 0.0,
                        shape: 1.5
                    }
                )
                .is_err()
        );
    }
}
