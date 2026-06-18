use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Family, HasCdf, Log, Mu, ParameterParts, ParameterizedFamily, PositiveLink, Shape,
};

use crate::special::{digamma, included_count, is_nonnegative_integer, ln_gamma, log_add_exp};

const MAX_CDF_TERMS: u64 = 1_000_000;

/// Negative binomial family parameterized by positive mean and shape.
///
/// The variance is `mu + mu^2 / shape`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NegativeBinomial<MuLink = Log, ShapeLink = Log> {
    marker: PhantomData<(MuLink, ShapeLink)>,
}

impl<MuLink, ShapeLink> NegativeBinomial<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    /// Creates a stateless negative binomial family.
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn theta_from_eta(eta: NegativeBinomialEta) -> NegativeBinomialTheta {
        NegativeBinomialTheta {
            mu: MuLink::inverse(eta.mu),
            shape: ShapeLink::inverse(eta.shape),
        }
    }

    #[inline(always)]
    fn nll_theta(y: f64, theta: NegativeBinomialTheta) -> f64 {
        if !is_nonnegative_integer(y)
            || theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
        {
            return f64::INFINITY;
        }

        let total = theta.shape + theta.mu;
        -ln_gamma(y + theta.shape) + ln_gamma(theta.shape) + ln_gamma(y + 1.0)
            - theta.shape * theta.shape.ln()
            - y * theta.mu.ln()
            + (y + theta.shape) * total.ln()
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: NegativeBinomialEta) -> (f64, NegativeBinomialEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (
                nll,
                NegativeBinomialEta {
                    mu: f64::NAN,
                    shape: f64::NAN,
                },
            );
        }

        let total = theta.shape + theta.mu;
        let d_mu = (y + theta.shape) / total - y / theta.mu;
        let d_shape = -digamma(y + theta.shape) + digamma(theta.shape) - theta.shape.ln() - 1.0
            + total.ln()
            + (y + theta.shape) / total;
        let gradient_eta = NegativeBinomialEta {
            mu: d_mu * MuLink::derivative_inverse(eta.mu),
            shape: d_shape * ShapeLink::derivative_inverse(eta.shape),
        };

        (nll, gradient_eta)
    }

    #[inline]
    fn cdf_theta(y: f64, theta: NegativeBinomialTheta) -> f64 {
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
        let success_probability = theta.shape / (theta.shape + theta.mu);
        let failure_probability = theta.mu / (theta.shape + theta.mu);
        let term = (theta.shape * success_probability.ln()).exp();
        if term.is_finite() && term > 0.0 {
            return Self::cdf_by_recurrence(theta.shape, failure_probability, max_count, term);
        }

        Self::cdf_by_log_sum(
            theta.shape,
            success_probability,
            failure_probability,
            max_count,
        )
    }

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

    fn cdf_by_log_sum(
        shape: f64,
        success_probability: f64,
        failure_probability: f64,
        max_count: u64,
    ) -> f64 {
        let log_success = success_probability.ln();
        let log_failure = failure_probability.ln();
        let log_shape_gamma = ln_gamma(shape);
        let mut log_sum = f64::NEG_INFINITY;

        for count in 0..=max_count {
            let count_f = count as f64;
            let log_term = ln_gamma(count_f + shape) - log_shape_gamma - ln_gamma(count_f + 1.0)
                + shape * log_success
                + count_f * log_failure;
            log_sum = log_add_exp(log_sum, log_term);
        }

        log_sum.exp().clamp(0.0, 1.0)
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

/// Predictors for the negative binomial family on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NegativeBinomialEta {
    /// Mean predictor.
    pub mu: f64,
    /// Shape predictor.
    pub shape: f64,
}

impl ParameterParts<2> for NegativeBinomialEta {
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
            _ => unreachable!("negative binomial eta only has indices 0 and 1"),
        }
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

impl<MuLink, ShapeLink> Family for NegativeBinomial<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    type Eta = NegativeBinomialEta;
    type Theta = NegativeBinomialTheta;
    type NllGradientEta = NegativeBinomialEta;
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

impl<MuLink, ShapeLink> ParameterizedFamily<2> for NegativeBinomial<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    type Params = (Mu, Shape);
    type Links = (MuLink, ShapeLink);
}

impl<MuLink, ShapeLink> HasCdf for NegativeBinomial<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::cdf_theta(y, theta)
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink, ShapeLink> CanSimulate<Rng> for NegativeBinomial<MuLink, ShapeLink>
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

        let lambda = rand_distr::Distribution::sample(
            &rand_distr::Gamma::new(theta.shape, theta.mu / theta.shape)
                .expect("validated gamma-poisson parameters must construct"),
            rng,
        );
        rand_distr::Distribution::sample(
            &rand_distr::Poisson::new(lambda).expect("validated poisson mean must construct"),
            rng,
        )
    }
}

/// Negative binomial distribution with log links for mean and shape.
pub type DefaultNegativeBinomial = NegativeBinomial<Log, Log>;

#[cfg(test)]
mod tests {
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;
    use gamlss_core::{Family, HasCdf};

    use super::{DefaultNegativeBinomial, NegativeBinomialTheta};
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn negative_binomial_gradient_matches_finite_difference() {
        let family = DefaultNegativeBinomial::new();
        assert_gradient_matches_finite_difference::<_, 2>(&family, 3.0, [0.4, -0.2]);
    }

    #[test]
    fn negative_binomial_rejects_invalid_domain_and_has_finite_nll_inside_domain() {
        let family = DefaultNegativeBinomial::new();
        let theta = NegativeBinomialTheta {
            mu: 2.0,
            shape: 1.5,
        };

        assert!(family.nll(3.0, theta).is_finite());
        assert!(family.nll(-1.0, theta).is_infinite());
        assert!(family.nll(1.5, theta).is_infinite());
        assert!(
            family
                .nll(
                    3.0,
                    NegativeBinomialTheta {
                        mu: 0.0,
                        shape: theta.shape,
                    },
                )
                .is_infinite()
        );
    }

    #[test]
    fn negative_binomial_cdf_matches_reference_points() {
        let family = DefaultNegativeBinomial::new();
        let theta = NegativeBinomialTheta {
            mu: 2.0,
            shape: 1.5,
        };
        let p = theta.shape / (theta.shape + theta.mu);
        let q = theta.mu / (theta.shape + theta.mu);
        let p0 = p.powf(theta.shape);
        let p1 = p0 * theta.shape * q;

        assert_eq!(family.cdf(-1.0, theta), 0.0);
        assert!((family.cdf(0.0, theta) - p0).abs() < 1.0e-12);
        assert!((family.cdf(1.5, theta) - (p0 + p1)).abs() < 1.0e-12);
        assert!(
            family
                .cdf(
                    1.0,
                    NegativeBinomialTheta {
                        mu: 0.0,
                        shape: theta.shape,
                    },
                )
                .is_nan()
        );
        assert!(
            family
                .cdf((super::MAX_CDF_TERMS + 1) as f64, theta)
                .is_nan()
        );
    }

    #[test]
    fn negative_binomial_cdf_is_stable_for_large_parameters() {
        let family = DefaultNegativeBinomial::new();
        let cdf = family.cdf(
            1000.0,
            NegativeBinomialTheta {
                mu: 1000.0,
                shape: 1000.0,
            },
        );

        assert!(cdf.is_finite());
        assert!(cdf > 0.45 && cdf < 0.55, "cdf was {cdf}");
    }

    #[cfg(feature = "rand")]
    #[test]
    fn negative_binomial_sampling_returns_counts_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = DefaultNegativeBinomial::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family.sample(
            &mut rng,
            NegativeBinomialTheta {
                mu: 2.0,
                shape: 1.5,
            },
        );
        assert!(sample >= 0.0 && sample.fract() == 0.0);
        assert!(
            family
                .sample(
                    &mut rng,
                    NegativeBinomialTheta {
                        mu: 0.0,
                        shape: 1.5
                    }
                )
                .is_nan()
        );
    }
}
