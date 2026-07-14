use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Family, HasCdf, HasCrps, HasQuantile, Identity, InitialEtaFromObservations,
    InitialEtaFromTheta, Link, Log, Mu, ObservationView, ParameterParts, PositiveLink, Sigma,
};
use gamlss_special::exponential_integral_e1;

use crate::constants::{EULER_MASCHERONI, LOG_2};
use crate::domain::{is_finite_location_scale, is_probability};
use crate::initial::{positive_floor, weighted_quantile, weighted_values};

/// Gumbel distribution with identity link for location and log link for scale.
pub type GumbelMuSigma = Gumbel<Identity, Log>;

/// Maximum-type Gumbel family parameterized by location and positive scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gumbel<MuLink = Identity, SigmaLink = Log> {
    marker: PhantomData<(MuLink, SigmaLink)>,
}

impl<MuLink, SigmaLink> Gumbel<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    /// Creates a stateless Gumbel family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(eta: GumbelEta) -> GumbelTheta {
        GumbelTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
        }
    }

    #[inline]
    fn valid_theta(theta: GumbelTheta) -> bool {
        is_finite_location_scale(theta.mu, theta.sigma)
    }

    #[inline]
    fn nll_theta(y: f64, theta: GumbelTheta) -> f64 {
        if !y.is_finite() || !Self::valid_theta(theta) {
            return f64::INFINITY;
        }

        let z = (y - theta.mu) / theta.sigma;
        theta.sigma.ln() + z + (-z).exp()
    }

    #[inline]
    fn nll_and_gradient_eta_values(y: f64, eta: GumbelEta) -> (f64, GumbelEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (
                nll,
                GumbelEta {
                    mu: f64::NAN,
                    sigma: f64::NAN,
                },
            );
        }

        let z = (y - theta.mu) / theta.sigma;
        let exp_neg_z = (-z).exp();
        let d_z = 1.0 - exp_neg_z;
        let d_mu = -d_z / theta.sigma;
        let d_sigma = z.mul_add(-d_z, 1.0) / theta.sigma;
        let gradient_eta = GumbelEta {
            mu: d_mu * MuLink::derivative_inverse(eta.mu),
            sigma: d_sigma * SigmaLink::derivative_inverse(eta.sigma),
        };

        (nll, gradient_eta)
    }
}

impl<MuLink, SigmaLink> Default for Gumbel<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MuLink, SigmaLink> for Gumbel<MuLink, SigmaLink>;
    parameters = (Mu, Sigma);
    arity = 2;
);

impl<MuLink, SigmaLink> Family for Gumbel<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    type Eta = GumbelEta;
    type Theta = GumbelTheta;
    type GradientEta = GumbelEta;
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

impl<MuLink, SigmaLink> InitialEtaFromObservations<2> for Gumbel<MuLink, SigmaLink>
where
    MuLink: InitialEtaFromTheta<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| y.is_finite().then_some(y));
        let Some(median) = weighted_quantile(&values, 0.5) else {
            return GumbelEta::from_array([0.0, 0.0]);
        };
        let q1 = weighted_quantile(&values, 0.25).unwrap_or(median);
        let q3 = weighted_quantile(&values, 0.75).unwrap_or(median);
        let standard_q1 = -(-0.25_f64.ln()).ln();
        let standard_q3 = -(-0.75_f64.ln()).ln();
        let standard_median = -(-0.5_f64.ln()).ln();
        let sigma = positive_floor((q3 - q1).abs() / (standard_q3 - standard_q1));
        let mu = median - sigma * standard_median;

        GumbelEta {
            mu: MuLink::initial_eta_from_theta(mu),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
        }
    }
}

impl<MuLink, SigmaLink> HasCdf for Gumbel<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        if !y.is_finite() || !Self::valid_theta(*theta) {
            return f64::NAN;
        }

        let z = (y - theta.mu) / theta.sigma;
        (-(-z).exp()).exp()
    }
}

impl<MuLink, SigmaLink> HasQuantile for Gumbel<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    #[allow(clippy::suboptimal_flops)]
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        if !is_probability(p) || !Self::valid_theta(*theta) {
            return f64::NAN;
        }

        theta.mu - theta.sigma * (-p.ln()).ln()
    }
}

impl<MuLink, SigmaLink> HasCrps for Gumbel<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn crps(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        if !y.is_finite() || !Self::valid_theta(*theta) {
            return f64::NAN;
        }

        let z = (y - theta.mu) / theta.sigma;
        let exp_neg_z = (-z).exp();
        let standard = if exp_neg_z == 0.0 {
            z - EULER_MASCHERONI - LOG_2
        } else {
            2.0_f64.mul_add(
                exponential_integral_e1(exp_neg_z),
                EULER_MASCHERONI - z - LOG_2,
            )
        };
        theta.sigma * standard.max(0.0)
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink, SigmaLink> CanSimulate<Rng> for Gumbel<MuLink, SigmaLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    type Sample = f64;

    fn sample(&self, rng: &mut Rng, theta: &Self::Theta) -> f64 {
        if !Self::valid_theta(*theta) {
            return f64::NAN;
        }

        rand_distr::Distribution::sample(
            &rand_distr::Gumbel::new(theta.mu, theta.sigma)
                .expect("validated gumbel parameters must construct"),
            rng,
        )
    }
}

/// Predictors for the Gumbel family on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GumbelEta {
    /// Location predictor.
    pub mu: f64,
    /// Scale predictor.
    pub sigma: f64,
}

impl ParameterParts<2> for GumbelEta {
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
            _ => unreachable!("gumbel eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale Gumbel parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GumbelTheta {
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

    use super::{GumbelMuSigma, GumbelTheta};
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn gumbel_gradient_matches_finite_difference() {
        let family = GumbelMuSigma::new();
        assert_gradient_matches_finite_difference::<_, 2>(&family, 1.7, [0.4, -0.2]);
    }

    #[test]
    fn gumbel_rejects_invalid_domain_and_has_finite_nll_inside_domain() {
        let family = GumbelMuSigma::new();
        let theta = GumbelTheta {
            mu: 0.4,
            sigma: 1.5,
        };

        assert!(family.nll(1.7, &theta, &mut family.workspace()).is_finite());
        assert!(
            family
                .nll(
                    1.7,
                    &GumbelTheta {
                        mu: theta.mu,
                        sigma: 0.0,
                    },
                    &mut family.workspace(),
                )
                .is_infinite()
        );
    }

    #[test]
    fn gumbel_cdf_matches_reference_points() {
        let family = GumbelMuSigma::new();
        let theta = GumbelTheta {
            mu: 0.4,
            sigma: 1.5,
        };

        assert_relative_eq!(
            family.cdf(theta.mu, &theta),
            (-1.0_f64).exp(),
            epsilon = 1.0e-12
        );
        assert!(family.cdf(f64::NAN, &theta).is_nan());
    }

    #[test]
    fn gumbel_quantile_inverts_cdf() {
        let family = GumbelMuSigma::new();
        let theta = GumbelTheta {
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
                    &GumbelTheta {
                        mu: 0.4,
                        sigma: 0.0
                    }
                )
                .is_nan()
        );
    }

    #[test]
    fn gumbel_crps_matches_fixed_values() {
        let family = GumbelMuSigma::new();
        let theta = GumbelTheta {
            mu: 0.4,
            sigma: 1.5,
        };

        assert_relative_eq!(
            family.crps(theta.mu, &theta),
            0.484_254_529_698_942_9,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            family.crps(3.0, &theta),
            1.202_013_180_764_635_3,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn gumbel_crps_returns_nan_for_invalid_domains() {
        let family = GumbelMuSigma::new();
        let theta = GumbelTheta {
            mu: 0.4,
            sigma: 1.5,
        };

        assert!(family.crps(f64::NAN, &theta).is_nan());
        assert!(
            family
                .crps(
                    1.0,
                    &GumbelTheta {
                        mu: 0.4,
                        sigma: 0.0
                    }
                )
                .is_nan()
        );
    }

    #[test]
    fn gumbel_crps_is_nonnegative_for_valid_domains() {
        let family = GumbelMuSigma::new();
        let theta = GumbelTheta {
            mu: 0.4,
            sigma: 1.5,
        };

        assert!(family.crps(-10.0, &theta) >= 0.0);
        assert!(family.crps(theta.mu, &theta) >= 0.0);
        assert!(family.crps(10.0, &theta) >= 0.0);
    }

    #[cfg(feature = "rand")]
    #[test]
    fn gumbel_sampling_returns_finite_values_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = GumbelMuSigma::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        assert!(
            family
                .sample(
                    &mut rng,
                    &GumbelTheta {
                        mu: 0.4,
                        sigma: 1.5
                    }
                )
                .is_finite()
        );
        assert!(
            family
                .sample(
                    &mut rng,
                    &GumbelTheta {
                        mu: 0.4,
                        sigma: 0.0
                    }
                )
                .is_nan()
        );
    }
}
