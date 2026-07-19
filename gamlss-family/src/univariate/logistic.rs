use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasCrps, HasQuantile, Identity, InitialEtaFromObservations,
    InitialEtaFromTheta, Link, Log, Mu, ObservationView, ParameterParts, PositiveLink, Sigma,
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

use crate::domain::{is_finite_location_scale, is_probability};
use crate::initial::{robust_location_scale, weighted_values};

/// Logistic distribution with identity link for location and log link for scale.
pub type LogisticMuSigma = Logistic<Identity, Log>;

/// Logistic family parameterized by location $\mu\in\mathbb{R}$ and scale $\sigma>0$.
///
/// With $z=(y-\mu)/\sigma$, its density is
///
/// $$
/// f(y\mid\mu,\sigma)
/// =\frac{e^{-z}}{\sigma(1+e^{-z})^2},
/// \qquad y\in\mathbb{R}.
/// $$
///
/// The natural-scale moments are $\mathbb{E}(Y)=\mu$ and $\operatorname{Var}(Y)=\pi^2\sigma^2/3$. The default [`LogisticMuSigma`] alias uses $\mu=\eta_\mu$ and $\sigma=\exp(\eta_\sigma)$.
///
/// The symbols $\mu,\sigma$ and predictors $\eta_\mu,\eta_\sigma$ correspond to the same-named fields of [`LogisticTheta`] and [`LogisticEta`].
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/distributions/logistic.svg")
)]
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Logistic<MuLink = Identity, SigmaLink = Log> {
    marker: PhantomData<(MuLink, SigmaLink)>,
}

impl<MuLink, SigmaLink> Logistic<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    /// Creates a stateless logistic family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(eta: LogisticEta) -> LogisticTheta {
        LogisticTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
        }
    }

    #[inline]
    fn valid_theta(theta: LogisticTheta) -> bool {
        is_finite_location_scale(theta.mu, theta.sigma)
    }

    #[inline]
    fn log_one_plus_exp(value: f64) -> f64 {
        if value > 0.0 {
            value + (-value).exp().ln_1p()
        } else {
            value.exp().ln_1p()
        }
    }

    #[inline]
    fn logistic(value: f64) -> f64 {
        if value >= 0.0 {
            let z = (-value).exp();
            1.0 / (1.0 + z)
        } else {
            let z = value.exp();
            z / (1.0 + z)
        }
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn nll_theta(y: f64, theta: LogisticTheta) -> f64 {
        if !y.is_finite() || !Self::valid_theta(theta) {
            return f64::INFINITY;
        }

        let z = (y - theta.mu) / theta.sigma;
        theta.sigma.ln() + z + 2.0 * Self::log_one_plus_exp(-z)
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn nll_and_gradient_eta_values(y: f64, eta: LogisticEta) -> (f64, LogisticEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (
                nll,
                LogisticEta {
                    mu: f64::NAN,
                    sigma: f64::NAN,
                },
            );
        }

        let z = (y - theta.mu) / theta.sigma;
        let d_z = 2.0 * Self::logistic(z) - 1.0;
        let d_mu = -d_z / theta.sigma;
        let d_sigma = z.mul_add(-d_z, 1.0) / theta.sigma;
        let gradient_eta = LogisticEta {
            mu: d_mu * MuLink::derivative_inverse(eta.mu),
            sigma: d_sigma * SigmaLink::derivative_inverse(eta.sigma),
        };

        (nll, gradient_eta)
    }
}

impl<MuLink, SigmaLink> Default for Logistic<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MuLink, SigmaLink> for Logistic<MuLink, SigmaLink>;
    parameters = (Mu, Sigma);
    arity = 2;
);

impl<MuLink, SigmaLink> Family for Logistic<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    type Eta = LogisticEta;
    type Theta = LogisticTheta;
    type GradientEta = LogisticEta;
    type Observation<'obs> = f64;
    type Workspace = ();

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

impl<MuLink, SigmaLink> InitialEtaFromObservations<2> for Logistic<MuLink, SigmaLink>
where
    MuLink: InitialEtaFromTheta<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| y.is_finite().then_some(y));
        let Some((mu, sigma)) = robust_location_scale(&values) else {
            return LogisticEta::from_array([0.0, 0.0]);
        };

        LogisticEta {
            mu: MuLink::initial_eta_from_theta(mu),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
        }
    }
}

impl<MuLink, SigmaLink> HasCdf for Logistic<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        if !y.is_finite() || !Self::valid_theta(*theta) {
            return f64::NAN;
        }

        Self::logistic((y - theta.mu) / theta.sigma)
    }
}

impl<MuLink, SigmaLink> HasQuantile for Logistic<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    #[allow(clippy::suboptimal_flops)]
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        if !is_probability(p) || !Self::valid_theta(*theta) {
            return f64::NAN;
        }

        theta.mu + theta.sigma * (p.ln() - (-p).ln_1p())
    }
}

impl<MuLink, SigmaLink> HasCrps for Logistic<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    #[allow(clippy::suboptimal_flops)]
    fn crps(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        if !y.is_finite() || !Self::valid_theta(*theta) {
            return f64::NAN;
        }

        let z = (y - theta.mu) / theta.sigma;
        let log_cdf = -Self::log_one_plus_exp(-z);
        theta.sigma * (z - 2.0 * log_cdf - 1.0)
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink, SigmaLink> TrySimulate<Rng> for Logistic<MuLink, SigmaLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    type Sample = f64;

    #[allow(clippy::suboptimal_flops)]
    fn try_sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Result<f64, SimulationError> {
        if !Self::valid_theta(*theta) {
            return Err(SimulationError::InvalidParameters("Logistic theta"));
        }

        let uniform: f64 = rand_distr::Distribution::sample(&rand_distr::Open01, rng);
        crate::simulation::ensure_finite(
            theta.mu + theta.sigma * (uniform / (1.0_f64 - uniform)).ln(),
            "Logistic transform",
        )
    }
}

/// Predictors for the logistic family on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogisticEta {
    /// Location predictor.
    pub mu: f64,
    /// Scale predictor.
    pub sigma: f64,
}

impl ParameterParts<2> for LogisticEta {
    #[inline]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            mu: values[0],
            sigma: values[1],
        }
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mu,
            1 => self.sigma,
            _ => unreachable!("logistic eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale logistic parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogisticTheta {
    /// Location parameter.
    pub mu: f64,
    /// Positive scale parameter.
    pub sigma: f64,
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    #[cfg(feature = "rand")]
    use gamlss_core::TrySimulate;
    use gamlss_core::{Family, HasCdf, HasCrps, HasQuantile};

    use super::{LogisticMuSigma, LogisticTheta};
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn logistic_gradient_matches_finite_difference() {
        let family = LogisticMuSigma::new();
        assert_gradient_matches_finite_difference::<_, 2>(&family, 1.7, [0.4, -0.2]);
    }

    #[test]
    fn logistic_rejects_invalid_domain_and_has_finite_nll_inside_domain() {
        let family = LogisticMuSigma::new();
        let theta = LogisticTheta {
            mu: 0.4,
            sigma: 1.5,
        };

        assert!(family.nll(1.7, &theta, &mut family.workspace()).is_finite());
        assert!(
            family
                .nll(
                    1.7,
                    &LogisticTheta {
                        mu: theta.mu,
                        sigma: 0.0,
                    },
                    &mut family.workspace(),
                )
                .is_infinite()
        );
    }

    #[test]
    fn logistic_cdf_matches_reference_points() {
        let family = LogisticMuSigma::new();
        let theta = LogisticTheta {
            mu: 0.4,
            sigma: 1.5,
        };

        assert_relative_eq!(family.cdf(theta.mu, &theta), 0.5, epsilon = 1.0e-12);
        assert!(family.cdf(f64::NAN, &theta).is_nan());
    }

    #[test]
    fn logistic_quantile_inverts_cdf() {
        let family = LogisticMuSigma::new();
        let theta = LogisticTheta {
            mu: 0.4,
            sigma: 1.5,
        };

        assert!(family.quantile(0.0, &theta).is_infinite());
        assert!(family.quantile(0.0, &theta).is_sign_negative());
        assert!(family.quantile(1.0, &theta).is_infinite());
        assert!(family.quantile(1.0, &theta).is_sign_positive());

        let y = family.quantile(0.75, &theta);
        assert_relative_eq!(family.cdf(y, &theta), 0.75, epsilon = 1.0e-12);
        assert!(family.quantile(f64::NAN, &theta).is_nan());
        assert!(
            family
                .quantile(
                    0.5,
                    &LogisticTheta {
                        mu: 0.4,
                        sigma: 0.0
                    }
                )
                .is_nan()
        );
    }

    #[test]
    fn logistic_crps_matches_fixed_values() {
        let family = LogisticMuSigma::new();

        assert_relative_eq!(
            family.crps(
                1.0,
                &LogisticTheta {
                    mu: 0.0,
                    sigma: 2.0,
                },
            ),
            0.896_307_936_720_426_7,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn logistic_crps_returns_nan_for_invalid_domains() {
        let family = LogisticMuSigma::new();

        assert!(
            family
                .crps(
                    f64::NAN,
                    &LogisticTheta {
                        mu: 0.0,
                        sigma: 1.0,
                    },
                )
                .is_nan()
        );
        assert!(
            family
                .crps(
                    1.0,
                    &LogisticTheta {
                        mu: 0.0,
                        sigma: 0.0,
                    },
                )
                .is_nan()
        );
    }

    #[test]
    fn logistic_crps_is_nonnegative_for_valid_domains() {
        let family = LogisticMuSigma::new();

        assert!(
            family.crps(
                1.0,
                &LogisticTheta {
                    mu: 0.0,
                    sigma: 2.0,
                },
            ) >= 0.0
        );
    }

    #[cfg(feature = "rand")]
    #[test]
    fn logistic_sampling_returns_finite_values_and_errors_for_invalid_theta() {
        use rand::SeedableRng;

        let family = LogisticMuSigma::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        assert!(
            family
                .try_sample(
                    &mut rng,
                    &LogisticTheta {
                        mu: 0.4,
                        sigma: 1.5
                    }
                )
                .is_ok_and(f64::is_finite)
        );
        assert!(
            family
                .try_sample(
                    &mut rng,
                    &LogisticTheta {
                        mu: 0.4,
                        sigma: 0.0
                    }
                )
                .is_err()
        );
    }
}
