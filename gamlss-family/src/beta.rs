use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Family, HasCdf, HasCrps, HasQuantile, InitialEtaFromObservations, InitialEtaFromTheta, Log,
    Logit, Mu, ObservationView, ParameterParts, PositiveLink, Precision, ScalarParams,
    UnitIntervalLink,
};

use gamlss_special::{digamma, integrate_finite, invert_bounded_cdf, ln_gamma, regularized_beta};

use crate::initial::{
    VARIANCE_FLOOR, positive_floor, probability_floor, weighted_summary, weighted_values,
};

/// Beta distribution with logit link for mean and log link for precision.
pub type BetaMeanPrecision = Beta<Logit, Log>;

/// Beta family parameterized by mean in `(0, 1)` and positive precision.
///
/// The mean link must guarantee values in `(0, 1)`.
///
/// ```compile_fail
/// use gamlss_core::{ScalarParams, Identity, Log};
/// use gamlss_family::Beta;
///
/// let _ = Beta::<Identity, Log>::new();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Beta<MuLink = Logit, PrecisionLink = Log> {
    marker: PhantomData<(MuLink, PrecisionLink)>,
}

impl<MuLink, PrecisionLink> Beta<MuLink, PrecisionLink>
where
    MuLink: UnitIntervalLink<f64>,
    PrecisionLink: PositiveLink<f64>,
{
    /// Creates a stateless beta family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(eta: BetaEta) -> BetaTheta {
        BetaTheta {
            mu: MuLink::inverse(eta.mu),
            precision: PrecisionLink::inverse(eta.precision),
        }
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn nll_theta(y: f64, theta: BetaTheta) -> f64 {
        if y <= 0.0
            || y >= 1.0
            || !y.is_finite()
            || theta.mu <= 0.0
            || theta.mu >= 1.0
            || !theta.mu.is_finite()
            || theta.precision <= 0.0
            || !theta.precision.is_finite()
        {
            return f64::INFINITY;
        }

        let alpha = theta.mu * theta.precision;
        let beta = (1.0 - theta.mu) * theta.precision;
        ln_gamma(alpha) + ln_gamma(beta)
            - ln_gamma(theta.precision)
            - (alpha - 1.0) * y.ln()
            - (beta - 1.0) * (1.0 - y).ln()
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn nll_and_gradient_eta_values(y: f64, eta: BetaEta) -> (f64, BetaEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (
                nll,
                BetaEta {
                    mu: f64::NAN,
                    precision: f64::NAN,
                },
            );
        }

        let alpha = theta.mu * theta.precision;
        let beta = (1.0 - theta.mu) * theta.precision;
        let common = digamma(theta.precision);
        let d_alpha = digamma(alpha) - common - y.ln();
        let d_beta = digamma(beta) - common - (1.0 - y).ln();
        let d_mu = theta.precision * (d_alpha - d_beta);
        let d_precision = theta.mu * d_alpha + (1.0 - theta.mu) * d_beta;
        let gradient_eta = BetaEta {
            mu: d_mu * MuLink::derivative_inverse(eta.mu),
            precision: d_precision * PrecisionLink::derivative_inverse(eta.precision),
        };

        (nll, gradient_eta)
    }
}

impl<MuLink, PrecisionLink> Default for Beta<MuLink, PrecisionLink>
where
    MuLink: UnitIntervalLink<f64>,
    PrecisionLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<MuLink, PrecisionLink> Family for Beta<MuLink, PrecisionLink>
where
    MuLink: UnitIntervalLink<f64>,
    PrecisionLink: PositiveLink<f64>,
{
    type Eta = BetaEta;
    type Theta = BetaTheta;
    type GradientEta = BetaEta;
    type Observation<'obs> = f64;
    type Workspace = ();
    type ParamSpec = ScalarParams<(Mu, Precision), (MuLink, PrecisionLink), 2>;

    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    #[inline]
    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        Self::nll_theta(y, *theta)
    }

    #[inline]
    fn nll_eta(&self, y: f64, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> f64 {
        Self::nll_theta(y, Self::theta_from_eta(*eta))
    }

    #[inline]
    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        Self::nll_and_gradient_eta_values(y, *eta)
    }
}

impl<MuLink, PrecisionLink> InitialEtaFromObservations<2> for Beta<MuLink, PrecisionLink>
where
    MuLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
    PrecisionLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values =
            weighted_values::<Self, _, _>(obs, |y| y.is_finite().then_some(probability_floor(y)));
        let Some(summary) = weighted_summary(&values) else {
            return BetaEta::from_array([0.0, 0.0]);
        };

        let mu = probability_floor(summary.mean);
        let max_variance = (mu * (1.0 - mu)).max(VARIANCE_FLOOR);
        let variance = summary.variance.clamp(VARIANCE_FLOOR, max_variance * 0.99);
        let precision = positive_floor(max_variance / variance - 1.0);

        BetaEta {
            mu: MuLink::initial_eta_from_theta(mu),
            precision: PrecisionLink::initial_eta_from_theta(precision),
        }
    }
}

impl<MuLink, PrecisionLink> HasCdf for Beta<MuLink, PrecisionLink>
where
    MuLink: UnitIntervalLink<f64>,
    PrecisionLink: PositiveLink<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        if !y.is_finite()
            || theta.mu <= 0.0
            || theta.mu >= 1.0
            || !theta.mu.is_finite()
            || theta.precision <= 0.0
            || !theta.precision.is_finite()
        {
            return f64::NAN;
        }
        if y <= 0.0 {
            return 0.0;
        }
        if y >= 1.0 {
            return 1.0;
        }

        let alpha = theta.mu * theta.precision;
        let beta = (1.0 - theta.mu) * theta.precision;
        regularized_beta(alpha, beta, y)
    }
}

impl<MuLink, PrecisionLink> HasQuantile for Beta<MuLink, PrecisionLink>
where
    MuLink: UnitIntervalLink<f64>,
    PrecisionLink: PositiveLink<f64>,
{
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        if theta.mu <= 0.0
            || theta.mu >= 1.0
            || !theta.mu.is_finite()
            || theta.precision <= 0.0
            || !theta.precision.is_finite()
        {
            return f64::NAN;
        }

        let alpha = theta.mu * theta.precision;
        let beta = (1.0 - theta.mu) * theta.precision;
        if p <= 0.5 {
            invert_bounded_cdf(p, 0.0, 1.0, |y| regularized_beta(alpha, beta, y))
        } else {
            1.0 - invert_bounded_cdf(1.0 - p, 0.0, 1.0, |tail| {
                regularized_beta(beta, alpha, tail)
            })
        }
    }
}

impl<MuLink, PrecisionLink> HasCrps for Beta<MuLink, PrecisionLink>
where
    MuLink: UnitIntervalLink<f64>,
    PrecisionLink: PositiveLink<f64>,
{
    fn crps(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        if !(0.0..=1.0).contains(&y)
            || !y.is_finite()
            || theta.mu <= 0.0
            || theta.mu >= 1.0
            || !theta.mu.is_finite()
            || theta.precision <= 0.0
            || !theta.precision.is_finite()
        {
            return f64::NAN;
        }

        let left = integrate_finite(0.0, y, |x| {
            let cdf = self.cdf(x, theta);
            cdf * cdf
        });
        let right = integrate_finite(y, 1.0, |x| {
            let survival = 1.0 - self.cdf(x, theta);
            survival * survival
        });

        left + right
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink, PrecisionLink> CanSimulate<Rng> for Beta<MuLink, PrecisionLink>
where
    Rng: rand::Rng,
    MuLink: UnitIntervalLink<f64>,
    PrecisionLink: PositiveLink<f64>,
{
    type Sample = f64;

    fn sample(&self, rng: &mut Rng, theta: &Self::Theta) -> f64 {
        if theta.mu <= 0.0
            || theta.mu >= 1.0
            || !theta.mu.is_finite()
            || theta.precision <= 0.0
            || !theta.precision.is_finite()
        {
            return f64::NAN;
        }

        let alpha = theta.mu * theta.precision;
        let beta = (1.0 - theta.mu) * theta.precision;
        rand_distr::Distribution::sample(
            &rand_distr::Beta::new(alpha, beta).expect("validated beta parameters must construct"),
            rng,
        )
    }
}

/// Predictors for the beta family on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BetaEta {
    /// Mean predictor.
    pub mu: f64,
    /// Precision predictor.
    pub precision: f64,
}

impl ParameterParts<2> for BetaEta {
    #[inline]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            mu: values[0],
            precision: values[1],
        }
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mu,
            1 => self.precision,
            _ => unreachable!("beta eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale beta parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BetaTheta {
    /// Mean parameter in `(0, 1)`.
    pub mu: f64,
    /// Positive precision parameter.
    pub precision: f64,
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;
    use gamlss_core::{Family, HasCdf, HasCrps, HasQuantile};
    use statrs::distribution::{Beta as StatrsBeta, ContinuousCDF};

    use super::{BetaMeanPrecision, BetaTheta};
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn beta_gradient_matches_finite_difference() {
        let family = BetaMeanPrecision::new();
        assert_gradient_matches_finite_difference::<_, 2>(&family, 0.4, [0.2, 1.0]);
    }

    #[test]
    fn beta_rejects_invalid_domain_and_has_finite_nll_inside_domain() {
        let family = BetaMeanPrecision::new();
        let theta = BetaTheta {
            mu: 0.4,
            precision: 3.0,
        };

        assert!(family.nll(0.4, &theta, &mut family.workspace()).is_finite());
        assert!(
            family
                .nll(0.0, &theta, &mut family.workspace())
                .is_infinite()
        );
        assert!(
            family
                .nll(1.0, &theta, &mut family.workspace())
                .is_infinite()
        );
        assert!(
            family
                .nll(
                    0.4,
                    &BetaTheta {
                        mu: 1.0,
                        precision: theta.precision,
                    },
                    &mut family.workspace(),
                )
                .is_infinite()
        );
    }

    #[test]
    fn beta_cdf_and_quantile_match_statrs_reference() {
        let family = BetaMeanPrecision::new();
        let theta = BetaTheta {
            mu: 0.4,
            precision: 3.0,
        };
        let alpha = theta.mu * theta.precision;
        let beta = (1.0 - theta.mu) * theta.precision;
        let reference = StatrsBeta::new(alpha, beta).unwrap();

        for y in [0.01, 0.2, 0.4, 0.8, 0.99] {
            assert_relative_eq!(family.cdf(y, &theta), reference.cdf(y), epsilon = 1.0e-11);
        }

        for p in [0.01, 0.1, 0.5, 0.9, 0.99] {
            assert_relative_eq!(
                family.quantile(p, &theta),
                reference.inverse_cdf(p),
                epsilon = 1.0e-10
            );
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn beta_cdf_and_quantile_handle_boundaries_and_invalid_domains() {
        let family = BetaMeanPrecision::new();
        let theta = BetaTheta {
            mu: 0.4,
            precision: 3.0,
        };

        assert_eq!(family.cdf(0.0, &theta), 0.0);
        assert_eq!(family.cdf(1.0, &theta), 1.0);
        assert_eq!(family.quantile(0.0, &theta), 0.0);
        assert_eq!(family.quantile(1.0, &theta), 1.0);
        assert!(family.quantile(f64::NAN, &theta).is_nan());
        assert!(
            family
                .cdf(
                    0.5,
                    &BetaTheta {
                        mu: 0.0,
                        precision: 1.0,
                    },
                )
                .is_nan()
        );
    }

    #[test]
    fn beta_crps_matches_fixed_values() {
        let family = BetaMeanPrecision::new();

        assert_relative_eq!(
            family.crps(
                0.4,
                &BetaTheta {
                    mu: 0.5,
                    precision: 2.0,
                },
            ),
            0.093_333_333_333_333_34,
            epsilon = 1.0e-10
        );
        assert_relative_eq!(
            family.crps(
                0.0,
                &BetaTheta {
                    mu: 0.5,
                    precision: 2.0,
                },
            ),
            0.333_333_333_333_333_3,
            epsilon = 1.0e-10
        );
    }

    #[test]
    fn beta_crps_returns_nan_for_invalid_domains() {
        let family = BetaMeanPrecision::new();

        assert!(
            family
                .crps(
                    -0.1,
                    &BetaTheta {
                        mu: 0.4,
                        precision: 3.0,
                    },
                )
                .is_nan()
        );
        assert!(
            family
                .crps(
                    0.4,
                    &BetaTheta {
                        mu: 1.0,
                        precision: 3.0,
                    },
                )
                .is_nan()
        );
    }

    #[test]
    fn beta_crps_is_nonnegative_for_valid_domains() {
        let family = BetaMeanPrecision::new();

        assert!(
            family.crps(
                0.4,
                &BetaTheta {
                    mu: 0.4,
                    precision: 3.0,
                },
            ) >= 0.0
        );
    }

    #[cfg(feature = "rand")]
    #[test]
    fn beta_sampling_returns_unit_interval_values_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = BetaMeanPrecision::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family.sample(
            &mut rng,
            &BetaTheta {
                mu: 0.4,
                precision: 3.0,
            },
        );

        assert!(sample > 0.0 && sample < 1.0);
        assert!(
            family
                .sample(
                    &mut rng,
                    &BetaTheta {
                        mu: 0.0,
                        precision: 3.0,
                    },
                )
                .is_nan()
        );
    }
}
