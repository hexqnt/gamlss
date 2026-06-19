use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Family, HasCdf, HasQuantile, Log, Mu, ParameterParts, ParameterizedFamily, PositiveLink,
};

use crate::special::{
    discrete_quantile, included_count, is_nonnegative_integer, ln_gamma, log_add_exp,
};

const MAX_CDF_TERMS: u64 = 1_000_000;

/// Poisson family parameterized by positive mean.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Poisson<MuLink = Log> {
    marker: PhantomData<MuLink>,
}

impl<MuLink> Poisson<MuLink>
where
    MuLink: PositiveLink<f64>,
{
    /// Creates a stateless Poisson family.
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn theta_from_eta(eta: PoissonEta) -> PoissonTheta {
        PoissonTheta {
            mu: MuLink::inverse(eta.mu),
        }
    }

    #[inline(always)]
    fn nll_theta(y: f64, theta: PoissonTheta) -> f64 {
        if !is_nonnegative_integer(y) || theta.mu <= 0.0 || !theta.mu.is_finite() {
            return f64::INFINITY;
        }

        theta.mu - y * theta.mu.ln() + ln_gamma(y + 1.0)
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: PoissonEta) -> (f64, PoissonEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (nll, PoissonEta { mu: f64::NAN });
        }

        let d_mu = 1.0 - y / theta.mu;
        let gradient_eta = PoissonEta {
            mu: d_mu * MuLink::derivative_inverse(eta.mu),
        };

        (nll, gradient_eta)
    }

    #[inline]
    fn cdf_theta(y: f64, theta: PoissonTheta) -> f64 {
        if !y.is_finite() || theta.mu <= 0.0 || !theta.mu.is_finite() {
            return f64::NAN;
        }
        if y < 0.0 {
            return 0.0;
        }

        let Some(max_count) = included_count(y, MAX_CDF_TERMS) else {
            return f64::NAN;
        };
        let term = (-theta.mu).exp();
        if term.is_finite() && term > 0.0 {
            return Self::cdf_by_recurrence(theta.mu, max_count, term);
        }

        Self::cdf_by_log_sum(theta.mu, max_count)
    }

    fn cdf_by_recurrence(mu: f64, max_count: u64, mut term: f64) -> f64 {
        let mut sum = term;
        for count in 1..=max_count {
            term *= mu / count as f64;
            sum += term;
            if term <= f64::EPSILON * sum {
                break;
            }
        }

        sum.clamp(0.0, 1.0)
    }

    fn cdf_by_log_sum(mu: f64, max_count: u64) -> f64 {
        let log_mu = mu.ln();
        let mut log_sum = f64::NEG_INFINITY;
        for count in 0..=max_count {
            let count_f = count as f64;
            let log_term = -mu + count_f * log_mu - ln_gamma(count_f + 1.0);
            log_sum = log_add_exp(log_sum, log_term);
        }

        log_sum.exp().clamp(0.0, 1.0)
    }
}

impl<MuLink> Default for Poisson<MuLink>
where
    MuLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Predictor for the Poisson family on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoissonEta {
    /// Mean predictor.
    pub mu: f64,
}

impl ParameterParts<1> for PoissonEta {
    #[inline(always)]
    fn from_array(values: [f64; 1]) -> Self {
        Self { mu: values[0] }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mu,
            _ => unreachable!("poisson eta only has index 0"),
        }
    }
}

/// Natural-scale Poisson parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoissonTheta {
    /// Positive mean parameter.
    pub mu: f64,
}

impl<MuLink> Family for Poisson<MuLink>
where
    MuLink: PositiveLink<f64>,
{
    type Eta = PoissonEta;
    type Theta = PoissonTheta;
    type NllGradientEta = PoissonEta;
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

impl<MuLink> ParameterizedFamily<1> for Poisson<MuLink>
where
    MuLink: PositiveLink<f64>,
{
    type Params = (Mu,);
    type Links = (MuLink,);
}

impl<MuLink> HasCdf for Poisson<MuLink>
where
    MuLink: PositiveLink<f64>,
{
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::cdf_theta(y, theta)
    }
}

impl<MuLink> HasQuantile for Poisson<MuLink>
where
    MuLink: PositiveLink<f64>,
{
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
        if theta.mu <= 0.0 || !theta.mu.is_finite() {
            return f64::NAN;
        }

        discrete_quantile(p, MAX_CDF_TERMS, |count| {
            Self::cdf_theta(count as f64, theta)
        })
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink> CanSimulate<Rng> for Poisson<MuLink>
where
    Rng: rand::Rng,
    MuLink: PositiveLink<f64>,
{
    fn sample(&self, rng: &mut Rng, theta: Self::Theta) -> f64 {
        if theta.mu <= 0.0 || !theta.mu.is_finite() {
            return f64::NAN;
        }

        rand_distr::Distribution::sample(
            &rand_distr::Poisson::new(theta.mu).expect("validated poisson mean must construct"),
            rng,
        )
    }
}

/// Poisson distribution with log link for mean.
pub type DefaultPoisson = Poisson<Log>;

#[cfg(test)]
mod tests {
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;
    use gamlss_core::{Family, HasCdf, HasQuantile};
    use statrs::distribution::{DiscreteCDF, Poisson as StatrsPoisson};

    use super::{DefaultPoisson, PoissonTheta};
    use crate::test_support::{
        assert_gradient_matches_finite_difference, statrs_discrete_quantile,
    };

    #[test]
    fn poisson_gradient_matches_finite_difference() {
        let family = DefaultPoisson::new();
        assert_gradient_matches_finite_difference::<_, 1>(&family, 3.0, [0.4]);
    }

    #[test]
    fn poisson_rejects_invalid_domain_and_has_finite_nll_inside_domain() {
        let family = DefaultPoisson::new();
        let theta = PoissonTheta { mu: 2.0 };

        assert!(family.nll(3.0, theta).is_finite());
        assert!(family.nll(-1.0, theta).is_infinite());
        assert!(family.nll(1.5, theta).is_infinite());
        assert!(family.nll(3.0, PoissonTheta { mu: 0.0 }).is_infinite());
    }

    #[test]
    fn poisson_cdf_matches_reference_points() {
        let family = DefaultPoisson::new();
        let theta = PoissonTheta { mu: 2.0 };

        assert_eq!(family.cdf(-1.0, theta), 0.0);
        assert!((family.cdf(0.0, theta) - (-2.0_f64).exp()).abs() < 1.0e-12);
        assert!((family.cdf(1.5, theta) - 3.0 * (-2.0_f64).exp()).abs() < 1.0e-12);
        assert!(family.cdf(3.0, PoissonTheta { mu: 0.0 }).is_nan());
        assert!(
            family
                .cdf((super::MAX_CDF_TERMS + 1) as f64, theta)
                .is_nan()
        );
    }

    #[test]
    fn poisson_cdf_is_stable_for_large_mean() {
        let family = DefaultPoisson::new();
        let cdf = family.cdf(1000.0, PoissonTheta { mu: 1000.0 });

        assert!(cdf.is_finite());
        assert!(cdf > 0.45 && cdf < 0.55, "cdf was {cdf}");
    }

    #[test]
    fn poisson_quantile_matches_statrs_reference() {
        let family = DefaultPoisson::new();
        let theta = PoissonTheta { mu: 2.0 };
        let reference = StatrsPoisson::new(theta.mu).unwrap();

        for p in [0.01, 0.1, 0.5, 0.9, 0.99] {
            assert_eq!(
                family.quantile(p, theta),
                statrs_discrete_quantile(p, |count| reference.cdf(count)) as f64
            );
        }

        assert_eq!(family.quantile(0.0, theta), 0.0);
        assert!(family.quantile(f64::NAN, theta).is_nan());
        assert!(family.quantile(0.5, PoissonTheta { mu: 0.0 }).is_nan());
    }

    #[test]
    fn poisson_quantile_is_generalized_inverse_cdf() {
        let family = DefaultPoisson::new();
        let theta = PoissonTheta { mu: 6.0 };

        for p in [0.01, 0.1, 0.5, 0.9, 0.99] {
            let q = family.quantile(p, theta);
            assert!(family.cdf(q, theta) >= p);
            if q > 0.0 {
                assert!(family.cdf(q - 1.0, theta) < p);
            }
        }
    }

    #[cfg(feature = "rand")]
    #[test]
    fn poisson_sampling_returns_counts_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = DefaultPoisson::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family.sample(&mut rng, PoissonTheta { mu: 2.0 });
        assert!(sample >= 0.0 && sample.fract() == 0.0);
        assert!(family.sample(&mut rng, PoissonTheta { mu: 0.0 }).is_nan());
    }
}
