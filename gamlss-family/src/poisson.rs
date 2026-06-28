use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Family, HasCdf, HasCrps, HasQuantile, InitialEtaFromTheta, Log, Mu, ObservationView,
    ParameterParts, ParameterizedFamily, PositiveLink,
};

use gamlss_special::{
    discrete_quantile, included_count, is_nonnegative_integer, ln_gamma, log_add_exp,
};

use crate::initial::{positive_floor, weighted_mean, weighted_values};

const MAX_CDF_TERMS: u64 = 1_000_000;
const MAX_BESSEL_SERIES_TERMS: usize = 10_000;
const BESSEL_SERIES_EPSILON: f64 = 1.0e-15;
const DIRECT_BESSEL_MU_LIMIT: f64 = 350.0;

/// Poisson distribution with log link for mean.
pub type PoissonMean = Poisson<Log>;

/// Poisson family parameterized by positive mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Poisson<MuLink = Log> {
    marker: PhantomData<MuLink>,
}

impl<MuLink> Poisson<MuLink>
where
    MuLink: PositiveLink<f64>,
{
    /// Creates a stateless Poisson family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(eta: PoissonEta) -> PoissonTheta {
        PoissonTheta {
            mu: MuLink::inverse(eta.mu),
        }
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn nll_theta(y: f64, theta: PoissonTheta) -> f64 {
        if !is_nonnegative_integer(y) || theta.mu <= 0.0 || !theta.mu.is_finite() {
            return f64::INFINITY;
        }

        theta.mu - y * theta.mu.ln() + ln_gamma(y + 1.0)
    }

    #[inline]
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
    pub(crate) fn cdf_theta(y: f64, theta: PoissonTheta) -> f64 {
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

    #[allow(clippy::cast_precision_loss)]
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

    #[allow(clippy::suboptimal_flops, clippy::cast_precision_loss)]
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

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn pmf_theta(y: f64, theta: PoissonTheta) -> f64 {
        if !is_nonnegative_integer(y) || theta.mu <= 0.0 || !theta.mu.is_finite() {
            return f64::NAN;
        }

        (-theta.mu + y * theta.mu.ln() - ln_gamma(y + 1.0)).exp()
    }

    #[inline]
    fn half_gini_mean_difference(mu: f64) -> f64 {
        mu * Self::scaled_bessel_i0_plus_i1_two_mu(mu)
    }

    fn scaled_bessel_i0_plus_i1_two_mu(mu: f64) -> f64 {
        if mu <= DIRECT_BESSEL_MU_LIMIT {
            return Self::scaled_bessel_i0_plus_i1_by_series(mu);
        }

        Self::scaled_bessel_i0_plus_i1_asymptotic(2.0 * mu)
    }

    #[allow(clippy::cast_precision_loss)]
    fn scaled_bessel_i0_plus_i1_by_series(mu: f64) -> f64 {
        let mu2 = mu * mu;
        let scale = (-2.0 * mu).exp();

        let mut i0_term = scale;
        let mut i0_sum = i0_term;
        for count in 1..=MAX_BESSEL_SERIES_TERMS {
            let count_f = count as f64;
            i0_term *= mu2 / (count_f * count_f);
            i0_sum += i0_term;
            if i0_term.abs() <= BESSEL_SERIES_EPSILON * i0_sum.abs() {
                break;
            }
        }

        let mut i1_term = scale * mu;
        let mut i1_sum = i1_term;
        for count in 1..=MAX_BESSEL_SERIES_TERMS {
            let count_f = count as f64;
            i1_term *= mu2 / (count_f * (count_f + 1.0));
            i1_sum += i1_term;
            if i1_term.abs() <= BESSEL_SERIES_EPSILON * i1_sum.abs() {
                break;
            }
        }

        i0_sum + i1_sum
    }

    #[allow(clippy::suboptimal_flops)]
    fn scaled_bessel_i0_plus_i1_asymptotic(x: f64) -> f64 {
        let inv = 1.0 / (8.0 * x);
        let inv2 = inv * inv;
        let inv3 = inv2 * inv;
        let inv4 = inv2 * inv2;
        let i0 = 1.0 + inv + 9.0 * inv2 / 2.0 + 225.0 * inv3 / 6.0 + 11_025.0 * inv4 / 24.0;
        let i1 = 1.0 - 3.0 * inv - 15.0 * inv2 / 2.0 - 315.0 * inv3 / 6.0 - 14_175.0 * inv4 / 24.0;

        (i0 + i1) / (2.0 * std::f64::consts::PI * x).sqrt()
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

impl<MuLink> Family for Poisson<MuLink>
where
    MuLink: PositiveLink<f64>,
{
    type Eta = PoissonEta;
    type Theta = PoissonTheta;
    type NllGradientEta = PoissonEta;
    type Observation<'obs> = f64;

    #[inline]
    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline]
    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::nll_theta(y, theta)
    }

    #[inline]
    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        Self::nll_theta(y, Self::theta_from_eta(eta))
    }

    #[inline]
    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        Self::nll_and_gradient_eta_values(y, eta)
    }
}

impl<MuLink> ParameterizedFamily<1> for Poisson<MuLink>
where
    MuLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Params = (Mu,);
    type Links = (MuLink,);

    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| is_nonnegative_integer(y).then_some(y));
        let Some(mean) = weighted_mean(&values) else {
            return PoissonEta::from_array([0.0]);
        };

        PoissonEta {
            mu: MuLink::initial_eta_from_theta(positive_floor(mean)),
        }
    }
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
    #[allow(clippy::cast_precision_loss)]
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
        if theta.mu <= 0.0 || !theta.mu.is_finite() {
            return f64::NAN;
        }

        discrete_quantile(p, MAX_CDF_TERMS, |count| {
            Self::cdf_theta(count as f64, theta)
        })
    }
}

impl<MuLink> HasCrps for Poisson<MuLink>
where
    MuLink: PositiveLink<f64>,
{
    #[allow(clippy::suboptimal_flops)]
    fn crps(&self, y: Self::Observation<'_>, theta: Self::Theta) -> f64 {
        if !is_nonnegative_integer(y) || theta.mu <= 0.0 || !theta.mu.is_finite() {
            return f64::NAN;
        }

        let cdf = Self::cdf_theta(y, theta);
        let pmf = Self::pmf_theta(y, theta);
        let half_gini = Self::half_gini_mean_difference(theta.mu);
        if !cdf.is_finite() || !pmf.is_finite() || !half_gini.is_finite() {
            return f64::NAN;
        }

        (y - theta.mu) * (2.0 * cdf - 1.0) + 2.0 * theta.mu * pmf - half_gini
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink> CanSimulate<Rng> for Poisson<MuLink>
where
    Rng: rand::Rng,
    MuLink: PositiveLink<f64>,
{
    type Sample = f64;

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

/// Predictor for the Poisson family on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoissonEta {
    /// Mean predictor.
    pub mu: f64,
}

impl ParameterParts<1> for PoissonEta {
    #[inline]
    fn from_array(values: [f64; 1]) -> Self {
        Self { mu: values[0] }
    }

    #[inline]
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

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;
    use gamlss_core::{Family, HasCdf, HasCrps, HasQuantile};
    use statrs::distribution::{DiscreteCDF, Poisson as StatrsPoisson};

    use super::{PoissonMean, PoissonTheta};
    use crate::test_support::{
        assert_gradient_matches_finite_difference, statrs_discrete_quantile,
    };

    #[test]
    fn poisson_gradient_matches_finite_difference() {
        let family = PoissonMean::new();
        assert_gradient_matches_finite_difference::<_, 1>(&family, 3.0, [0.4]);
    }

    #[test]
    fn poisson_rejects_invalid_domain_and_has_finite_nll_inside_domain() {
        let family = PoissonMean::new();
        let theta = PoissonTheta { mu: 2.0 };

        assert!(family.nll(3.0, theta).is_finite());
        assert!(family.nll(-1.0, theta).is_infinite());
        assert!(family.nll(1.5, theta).is_infinite());
        assert!(family.nll(3.0, PoissonTheta { mu: 0.0 }).is_infinite());
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        clippy::suboptimal_flops,
        clippy::cast_precision_loss
    )]
    fn poisson_cdf_matches_reference_points() {
        let family = PoissonMean::new();
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
        let family = PoissonMean::new();
        let cdf = family.cdf(1000.0, PoissonTheta { mu: 1000.0 });

        assert!(cdf.is_finite());
        assert!(cdf > 0.45 && cdf < 0.55, "cdf was {cdf}");
    }

    #[test]
    #[allow(clippy::float_cmp, clippy::cast_precision_loss)]
    fn poisson_quantile_matches_statrs_reference() {
        let family = PoissonMean::new();
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
        let family = PoissonMean::new();
        let theta = PoissonTheta { mu: 6.0 };

        for p in [0.01, 0.1, 0.5, 0.9, 0.99] {
            let q = family.quantile(p, theta);
            assert!(family.cdf(q, theta) >= p);
            if q > 0.0 {
                assert!(family.cdf(q - 1.0, theta) < p);
            }
        }
    }

    #[test]
    fn poisson_crps_matches_fixed_values() {
        let family = PoissonMean::new();
        let theta = PoissonTheta { mu: 2.0 };

        assert_relative_eq!(
            family.crps(3.0, theta),
            0.664_529_576_806_184_1,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            family.crps(0.0, theta),
            1.228_494_478_547_156,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn poisson_crps_matches_truncated_expectation_identity() {
        let family = PoissonMean::new();
        let theta = PoissonTheta { mu: 6.0 };

        assert_relative_eq!(
            family.crps(5.0, theta),
            poisson_crps_by_truncated_expectations(5, theta.mu),
            epsilon = 1.0e-12
        );
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn poisson_crps_returns_nan_for_invalid_domains() {
        let family = PoissonMean::new();
        let theta = PoissonTheta { mu: 2.0 };

        assert!(family.crps(-1.0, theta).is_nan());
        assert!(family.crps(1.5, theta).is_nan());
        assert!(family.crps(3.0, PoissonTheta { mu: 0.0 }).is_nan());
        assert!(
            family
                .crps((super::MAX_CDF_TERMS + 1) as f64, theta)
                .is_nan()
        );
    }

    #[test]
    fn poisson_crps_is_nonnegative_for_valid_domains() {
        let family = PoissonMean::new();

        assert!(family.crps(3.0, PoissonTheta { mu: 2.0 }) >= 0.0);
        assert!(family.crps(1000.0, PoissonTheta { mu: 1000.0 }) >= 0.0);
    }

    #[allow(clippy::suboptimal_flops, clippy::cast_precision_loss)]
    fn poisson_crps_by_truncated_expectations(y: u64, mu: f64) -> f64 {
        let mut term = (-mu).exp();
        let mut cdf = term;
        let mut expected_absolute_error = y as f64 * term;
        let mut half_gini = cdf * (1.0 - cdf);

        for count in 1_u64..=10_000 {
            term *= mu / count as f64;
            cdf += term;
            expected_absolute_error += count.abs_diff(y) as f64 * term;
            half_gini += cdf * (1.0 - cdf);
            if term <= 1.0e-15 && 1.0 - cdf <= 1.0e-15 {
                break;
            }
        }

        expected_absolute_error - half_gini
    }

    #[cfg(feature = "rand")]
    #[test]
    fn poisson_sampling_returns_counts_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = PoissonMean::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family.sample(&mut rng, PoissonTheta { mu: 2.0 });
        assert!(sample >= 0.0 && sample.fract() == 0.0);
        assert!(family.sample(&mut rng, PoissonTheta { mu: 0.0 }).is_nan());
    }
}
