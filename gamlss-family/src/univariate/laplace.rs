use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Family, HasCdf, HasCrps, HasQuantile, Identity, InitialEtaFromObservations,
    InitialEtaFromTheta, Link, Log, Mu, ObservationView, ParameterParts, PositiveLink, Sigma,
};

use crate::constants::LOG_2;
use crate::domain::{is_finite_location_scale, is_probability};
use crate::initial::{robust_location_scale, weighted_values};

/// Laplace distribution with `Identity` link for `mu` and `Log` link for `sigma`.
pub type LaplaceMuSigma = Laplace<Identity, Log>;

/// Laplace family parameterized by location $\mu\in\mathbb{R}$ and scale $\sigma>0$.
///
/// Its density is
///
/// $$
/// f(y\mid\mu,\sigma)
/// =\frac{1}{2\sigma}\exp\left(-\frac{|y-\mu|}{\sigma}\right),
/// \qquad y\in\mathbb{R}.
/// $$
///
/// The natural-scale moments are $\mathbb{E}(Y)=\mu$ and $\operatorname{Var}(Y)=2\sigma^2$.
///
/// The symbols $\mu$ and $\sigma$ correspond to the same-named fields of [`LaplaceTheta`]. `MuLink` and `SigmaLink` control their links; the default [`LaplaceMuSigma`] alias gives $\mu=\eta_\mu$ and $\sigma=\exp(\eta_\sigma)$.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/laplace.svg")
)]
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Laplace<MuLink = Identity, SigmaLink = Log> {
    marker: PhantomData<(MuLink, SigmaLink)>,
}

impl<MuLink, SigmaLink> Laplace<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    /// Creates a stateless Laplace family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    /// Converts link-scale predictors to natural-scale parameters.
    #[inline]
    fn theta_from_eta(eta: LaplaceEta) -> LaplaceTheta {
        LaplaceTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
        }
    }

    #[inline]
    fn valid_theta(theta: LaplaceTheta) -> bool {
        is_finite_location_scale(theta.mu, theta.sigma)
    }

    /// Negative log-likelihood for a single observation on the natural scale.
    ///
    /// Returns `INFINITY` for non-finite observation/location or non-positive
    /// sigma.
    #[inline]
    fn nll_theta(y: f64, theta: LaplaceTheta) -> f64 {
        if !y.is_finite() || !Self::valid_theta(theta) {
            return f64::INFINITY;
        }

        LOG_2 + theta.sigma.ln() + (y - theta.mu).abs() / theta.sigma
    }

    /// Computes NLL and gradient with respect to eta for one observation.
    ///
    /// The gradient with respect to `mu` uses the sign subgradient (0 when
    /// `residual == 0`).
    #[inline]
    fn nll_and_gradient_eta_values(y: f64, eta: LaplaceEta) -> (f64, LaplaceEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (
                nll,
                LaplaceEta {
                    mu: f64::NAN,
                    sigma: f64::NAN,
                },
            );
        }

        let residual = y - theta.mu;
        let d_nll_d_mu = if residual > 0.0 {
            -1.0 / theta.sigma
        } else if residual < 0.0 {
            1.0 / theta.sigma
        } else {
            0.0
        };
        let d_nll_d_sigma = 1.0 / theta.sigma - residual.abs() / (theta.sigma * theta.sigma);

        let gradient_eta = LaplaceEta {
            mu: d_nll_d_mu * MuLink::derivative_inverse(eta.mu),
            sigma: d_nll_d_sigma * SigmaLink::derivative_inverse(eta.sigma),
        };

        (nll, gradient_eta)
    }
}

impl<MuLink, SigmaLink> Default for Laplace<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MuLink, SigmaLink> for Laplace<MuLink, SigmaLink>;
    parameters = (Mu, Sigma);
    arity = 2;
);

impl<MuLink, SigmaLink> Family for Laplace<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    type Eta = LaplaceEta;
    type Theta = LaplaceTheta;
    type GradientEta = LaplaceEta;
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

impl<MuLink, SigmaLink> InitialEtaFromObservations<2> for Laplace<MuLink, SigmaLink>
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
            return LaplaceEta::from_array([0.0, 0.0]);
        };

        LaplaceEta {
            mu: MuLink::initial_eta_from_theta(mu),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
        }
    }
}

impl<MuLink, SigmaLink> HasCdf for Laplace<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        if !y.is_finite() || !Self::valid_theta(*theta) {
            return f64::NAN;
        }

        let standardized = (y - theta.mu) / theta.sigma;
        if standardized < 0.0 {
            0.5 * standardized.exp()
        } else {
            0.5f64.mul_add(-(-standardized).exp(), 1.0)
        }
    }
}

impl<MuLink, SigmaLink> HasQuantile for Laplace<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    #[allow(clippy::suboptimal_flops)]
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        if !is_probability(p) || !Self::valid_theta(*theta) {
            return f64::NAN;
        }

        if p < 0.5 {
            theta.mu + theta.sigma * (2.0 * p).ln()
        } else {
            theta.mu - theta.sigma * (2.0 * (1.0 - p)).ln()
        }
    }
}

impl<MuLink, SigmaLink> HasCrps for Laplace<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    #[allow(clippy::suboptimal_flops)]
    fn crps(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        if !y.is_finite() || !Self::valid_theta(*theta) {
            return f64::NAN;
        }

        let abs_residual = (y - theta.mu).abs();
        abs_residual + theta.sigma * (-abs_residual / theta.sigma).exp() - 0.75 * theta.sigma
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink, SigmaLink> CanSimulate<Rng> for Laplace<MuLink, SigmaLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    type Sample = f64;

    #[allow(clippy::suboptimal_flops)]
    fn sample(&self, rng: &mut Rng, theta: &Self::Theta) -> f64 {
        if !Self::valid_theta(*theta) {
            return f64::NAN;
        }

        let centered = crate::simulation::open_unit(rng) - 0.5;
        let tail_probability: f64 = 1.0 - 2.0 * centered.abs();
        theta.mu - theta.sigma * centered.signum() * tail_probability.ln()
    }
}

/// Predictors for the Laplace distribution on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaplaceEta {
    /// Location predictor.
    pub mu: f64,
    /// Scale predictor.
    pub sigma: f64,
}

impl ParameterParts<2> for LaplaceEta {
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
            _ => unreachable!("laplace eta only has indices 0 and 1"),
        }
    }
}

/// Laplace distribution parameters on the natural scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaplaceTheta {
    /// Location parameter.
    pub mu: f64,
    /// Positive scale parameter.
    pub sigma: f64,
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;
    use gamlss_core::{Family, HasCdf, HasCrps, HasQuantile};

    use super::{LaplaceEta, LaplaceMuSigma, LaplaceTheta};
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn laplace_gradient_matches_finite_difference() {
        let family = LaplaceMuSigma::new();
        assert_gradient_matches_finite_difference::<_, 2>(&family, 1.7, [0.4, -0.2]);
    }

    #[test]
    fn laplace_rejects_non_finite_domain_and_returns_nan_gradient() {
        let family = LaplaceMuSigma::new();
        let theta = LaplaceTheta {
            mu: 0.4,
            sigma: 0.8,
        };

        assert!(family.nll(1.7, &theta, &mut family.workspace()).is_finite());
        assert!(
            family
                .nll(f64::INFINITY, &theta, &mut family.workspace())
                .is_infinite()
        );
        assert!(
            family
                .nll(
                    1.7,
                    &LaplaceTheta {
                        mu: f64::NAN,
                        sigma: theta.sigma,
                    },
                    &mut family.workspace(),
                )
                .is_infinite()
        );
        assert!(
            family
                .nll(
                    1.7,
                    &LaplaceTheta {
                        mu: theta.mu,
                        sigma: 0.0,
                    },
                    &mut family.workspace(),
                )
                .is_infinite()
        );

        let (nll, gradient) = family.nll_and_gradient_eta(
            1.7,
            &LaplaceEta {
                mu: 0.4,
                sigma: f64::NEG_INFINITY,
            },
            &mut family.workspace(),
        );
        assert!(nll.is_infinite());
        assert!(gradient.mu.is_nan());
        assert!(gradient.sigma.is_nan());
    }

    #[test]
    fn laplace_cdf_matches_reference_points() {
        let family = LaplaceMuSigma::new();
        let theta = LaplaceTheta {
            mu: 2.0,
            sigma: 0.5,
        };

        assert_relative_eq!(family.cdf(theta.mu, &theta), 0.5);
        assert_relative_eq!(
            family.cdf(theta.mu + theta.sigma, &theta),
            1.0 - 0.5 / std::f64::consts::E,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            family.cdf(theta.mu - theta.sigma, &theta),
            0.5 / std::f64::consts::E,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn laplace_cdf_returns_nan_for_invalid_domains() {
        let family = LaplaceMuSigma::new();

        assert!(
            family
                .cdf(
                    f64::NAN,
                    &LaplaceTheta {
                        mu: 0.0,
                        sigma: 1.0
                    }
                )
                .is_nan()
        );
        assert!(
            family
                .cdf(
                    0.0,
                    &LaplaceTheta {
                        mu: 0.0,
                        sigma: 0.0
                    }
                )
                .is_nan()
        );
    }

    #[test]
    fn laplace_quantile_inverts_cdf() {
        let family = LaplaceMuSigma::new();
        let theta = LaplaceTheta {
            mu: 2.0,
            sigma: 0.5,
        };

        assert_relative_eq!(family.quantile(0.5, &theta), theta.mu, epsilon = 1.0e-12);
        assert!(family.quantile(0.0, &theta).is_infinite());
        assert!(family.quantile(0.0, &theta).is_sign_negative());
        assert!(family.quantile(1.0, &theta).is_infinite());
        assert!(family.quantile(1.0, &theta).is_sign_positive());

        let y = family.quantile(0.25, &theta);
        assert_relative_eq!(family.cdf(y, &theta), 0.25, epsilon = 1.0e-12);
        assert!(family.quantile(f64::NAN, &theta).is_nan());
        assert!(
            family
                .quantile(
                    0.5,
                    &LaplaceTheta {
                        mu: 0.0,
                        sigma: 0.0
                    }
                )
                .is_nan()
        );
    }

    #[test]
    fn laplace_crps_matches_fixed_values() {
        let family = LaplaceMuSigma::new();

        assert_relative_eq!(
            family.crps(
                1.0,
                &LaplaceTheta {
                    mu: 0.0,
                    sigma: 2.0,
                },
            ),
            0.713_061_319_425_266_8,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn laplace_crps_returns_nan_for_invalid_domains() {
        let family = LaplaceMuSigma::new();

        assert!(
            family
                .crps(
                    1.0,
                    &LaplaceTheta {
                        mu: 0.0,
                        sigma: 0.0,
                    },
                )
                .is_nan()
        );
        assert!(
            family
                .crps(
                    f64::NAN,
                    &LaplaceTheta {
                        mu: 0.0,
                        sigma: 1.0,
                    },
                )
                .is_nan()
        );
    }

    #[cfg(feature = "rand")]
    #[test]
    fn laplace_sampling_returns_finite_values_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = LaplaceMuSigma::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        assert!(
            family
                .sample(
                    &mut rng,
                    &LaplaceTheta {
                        mu: 0.0,
                        sigma: 1.0
                    }
                )
                .is_finite()
        );
        assert!(
            family
                .sample(
                    &mut rng,
                    &LaplaceTheta {
                        mu: 0.0,
                        sigma: 0.0
                    }
                )
                .is_nan()
        );
    }
}
