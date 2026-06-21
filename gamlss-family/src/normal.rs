use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    DesignMatrix, Family, Gamlss, HasCdf, HasCrps, HasDeviance, HasInitialEta, HasQuantile,
    Identity, InitialEtaFromTheta, LinearPredictorBlock, Link, Log, ModelError, Mu, NoPenalty,
    ObservationView, ParameterBlock, ParameterBlocks, ParameterParts, ParameterizedFamily, Penalty,
    PositiveLink, Sigma,
};

use crate::domain::is_finite_location_scale;
use crate::initial::{robust_location_scale, weighted_values};
use crate::special::{unit_normal_cdf, unit_normal_quantile};

const HALF_LOG_2_PI: f64 = 0.918_938_533_204_672_7;
const INV_SQRT_2_PI: f64 = 0.398_942_280_401_432_7;
const INV_SQRT_PI: f64 = 0.564_189_583_547_756_3;
const DEFAULT_INITIAL_LOG_SIGMA: f64 = 0.0;

/// Normal distribution with `Identity` link for `mu` and `Log` link for `sigma`.
pub type NormalMuSigma = Normal<Identity, Log>;

/// Typed GAMLSS model for the default normal family.
///
/// The lifetime tracks the borrowed response slice.
pub type NormalGamlss<'a, XMu, XSigma, PMu = NoPenalty, PSigma = NoPenalty> = Gamlss<
    NormalMuSigma,
    (
        ParameterBlock<Mu, Identity, LinearPredictorBlock<XMu>, PMu>,
        ParameterBlock<Sigma, Log, LinearPredictorBlock<XSigma>, PSigma>,
    ),
    &'a [f64],
>;

/// Normal distribution with typed link functions for `mu` and `sigma`.
///
/// `SigmaLink` must be a positive link so that the scale parameter stays
/// positive at the type level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normal<MuLink = Identity, SigmaLink = Log> {
    marker: PhantomData<(MuLink, SigmaLink)>,
}

impl<MuLink, SigmaLink> Normal<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    /// Creates a stateless family value.
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    /// Converts link-scale predictors to natural-scale parameters.
    #[inline(always)]
    fn theta_from_eta(eta: NormalEta) -> NormalTheta {
        NormalTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
        }
    }

    #[inline(always)]
    fn valid_theta(theta: NormalTheta) -> bool {
        is_finite_location_scale(theta.mu, theta.sigma)
    }

    /// Negative log-likelihood for one observation on the natural scale.
    ///
    /// Returns `INFINITY` for non-finite observation/location or non-positive
    /// sigma.
    #[inline(always)]
    fn nll_theta(y: f64, theta: NormalTheta) -> f64 {
        if !y.is_finite() || !Self::valid_theta(theta) {
            return f64::INFINITY;
        }

        let residual = y - theta.mu;
        let z = residual / theta.sigma;
        HALF_LOG_2_PI + theta.sigma.ln() + 0.5 * z * z
    }

    /// Computes NLL and gradient w.r.t. eta for one observation.
    ///
    /// Uses analytic NLL derivatives w.r.t. `mu` and `sigma` and multiplies
    /// by the link function derivatives (chain rule).
    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: NormalEta) -> (f64, NormalEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (
                nll,
                NormalEta {
                    mu: f64::NAN,
                    sigma: f64::NAN,
                },
            );
        }

        let residual = y - theta.mu;
        let sigma2 = theta.sigma * theta.sigma;
        let d_nll_d_mu = (theta.mu - y) / sigma2;
        let d_nll_d_sigma = (1.0 / theta.sigma) - (residual * residual / (sigma2 * theta.sigma));

        let gradient_eta = NormalEta {
            mu: d_nll_d_mu * MuLink::derivative_inverse(eta.mu),
            sigma: d_nll_d_sigma * SigmaLink::derivative_inverse(eta.sigma),
        };

        (nll, gradient_eta)
    }
}

impl<MuLink, SigmaLink> Default for Normal<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<MuLink, SigmaLink> Family for Normal<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    type Eta = NormalEta;
    type Theta = NormalTheta;
    type NllGradientEta = NormalEta;
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

impl<MuLink, SigmaLink> ParameterizedFamily<2> for Normal<MuLink, SigmaLink>
where
    MuLink: InitialEtaFromTheta<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Params = (Mu, Sigma);
    type Links = (MuLink, SigmaLink);

    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| y.is_finite().then_some(y));
        let Some((mu, sigma)) = robust_location_scale(&values) else {
            return NormalEta::from_array([0.0, 0.0]);
        };

        NormalEta {
            mu: MuLink::initial_eta_from_theta(mu),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
        }
    }
}

impl<MuLink, SigmaLink> HasDeviance for Normal<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn deviance<'obs>(&self, y: Self::Observation<'obs>, theta: Self::Theta) -> f64 {
        if !y.is_finite() || !Self::valid_theta(theta) {
            return f64::INFINITY;
        }

        let z = (y - theta.mu) / theta.sigma;
        z * z
    }
}

impl<MuLink, SigmaLink> HasCdf for Normal<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
        if !y.is_finite() || !Self::valid_theta(theta) {
            return f64::NAN;
        }

        unit_normal_cdf((y - theta.mu) / theta.sigma)
    }
}

impl<MuLink, SigmaLink> HasQuantile for Normal<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
        if !Self::valid_theta(theta) {
            return f64::NAN;
        }

        theta.mu + theta.sigma * unit_normal_quantile(p)
    }
}

impl<MuLink, SigmaLink> HasCrps for Normal<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn crps<'obs>(&self, y: Self::Observation<'obs>, theta: Self::Theta) -> f64 {
        if !y.is_finite() || !Self::valid_theta(theta) {
            return f64::NAN;
        }

        let z = (y - theta.mu) / theta.sigma;
        let cdf = unit_normal_cdf(z);
        let pdf = INV_SQRT_2_PI * (-0.5 * z * z).exp();
        theta.sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - INV_SQRT_PI)
    }
}

impl HasInitialEta for Normal<Identity, Log> {
    fn initial_eta<'obs>(&self, y: Self::Observation<'obs>) -> Self::Eta {
        NormalEta {
            mu: y,
            sigma: DEFAULT_INITIAL_LOG_SIGMA,
        }
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink, SigmaLink> CanSimulate<Rng> for Normal<MuLink, SigmaLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn sample(&self, rng: &mut Rng, theta: Self::Theta) -> f64 {
        if !Self::valid_theta(theta) {
            return f64::NAN;
        }

        rand_distr::Distribution::sample(
            &rand_distr::Normal::new(theta.mu, theta.sigma)
                .expect("validated normal parameters must construct"),
            rng,
        )
    }
}

/// Normal distribution predictors on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalEta {
    /// Predictor for `mu`.
    pub mu: f64,
    /// Predictor for `sigma`.
    pub sigma: f64,
}

impl ParameterParts<2> for NormalEta {
    #[inline(always)]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            mu: values[0],
            sigma: values[1],
        }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mu,
            1 => self.sigma,
            _ => unreachable!("normal eta only has indices 0 and 1"),
        }
    }
}

/// Normal distribution parameters on the natural scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalTheta {
    /// Location parameter.
    pub mu: f64,
    /// Positive scale parameter.
    pub sigma: f64,
}

/// Creates a normal GAMLSS model from a response, two design matrices and
/// penalties.
///
/// The returned model borrows `y` and owns the design matrices and penalties.
pub fn normal_gamlss<'a, XMu, XSigma, PMu, PSigma>(
    y: &'a [f64],
    mu_x: XMu,
    sigma_x: XSigma,
    mu_penalty: PMu,
    sigma_penalty: PSigma,
) -> Result<NormalGamlss<'a, XMu, XSigma, PMu, PSigma>, ModelError>
where
    XMu: DesignMatrix,
    XSigma: DesignMatrix,
    PMu: Penalty,
    PSigma: Penalty,
{
    let blocks = ParameterBlocks::new((
        ParameterBlock::<Mu, Identity, _, _>::linear(mu_x, mu_penalty, 0),
        ParameterBlock::<Sigma, Log, _, _>::linear(sigma_x, sigma_penalty, 0),
    ));

    Gamlss::try_new(NormalMuSigma::new(), blocks, y)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;
    use gamlss_core::{
        DenseDesign, Family, HasCdf, HasCrps, HasDensity, HasDeviance, HasInitialEta,
        HasLogDensity, HasQuantile, NoPenalty, Objective,
    };
    use statrs::distribution::{ContinuousCDF, Normal as StatrsNormal};

    use super::{DEFAULT_INITIAL_LOG_SIGMA, NormalEta, NormalMuSigma, NormalTheta, normal_gamlss};
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn normal_gradient_matches_finite_difference() {
        let family = NormalMuSigma::new();
        assert_gradient_matches_finite_difference::<_, 2>(&family, 1.7, [0.4, -0.2]);
    }

    #[test]
    fn normal_rejects_non_finite_domain_and_returns_nan_gradient() {
        let family = NormalMuSigma::new();
        let theta = NormalTheta {
            mu: 0.4,
            sigma: 0.8,
        };

        assert!(family.nll(1.7, theta).is_finite());
        assert!(family.nll(f64::NAN, theta).is_infinite());
        assert!(
            family
                .nll(
                    1.7,
                    NormalTheta {
                        mu: f64::INFINITY,
                        sigma: theta.sigma,
                    },
                )
                .is_infinite()
        );
        assert!(
            family
                .nll(
                    1.7,
                    NormalTheta {
                        mu: theta.mu,
                        sigma: 0.0,
                    },
                )
                .is_infinite()
        );

        let (nll, gradient) = family.nll_and_gradient_eta(
            1.7,
            NormalEta {
                mu: 0.4,
                sigma: f64::NEG_INFINITY,
            },
        );
        assert!(nll.is_infinite());
        assert!(gradient.mu.is_nan());
        assert!(gradient.sigma.is_nan());
    }

    #[test]
    fn normal_initial_eta_starts_inside_domain_for_valid_observation() {
        let family = NormalMuSigma::new();
        let eta = family.initial_eta(1.7);

        assert_relative_eq!(eta.mu, 1.7);
        assert_relative_eq!(eta.sigma, DEFAULT_INITIAL_LOG_SIGMA);
        assert!(family.nll_eta(1.7, eta).is_finite());
    }

    #[test]
    fn normal_initial_eta_propagates_invalid_observation_without_panic() {
        let family = NormalMuSigma::new();
        let invalid_eta = family.initial_eta(f64::NAN);

        assert!(invalid_eta.mu.is_nan());
        assert!(invalid_eta.sigma.is_finite());
    }

    #[test]
    fn normal_deviance_returns_non_finite_for_invalid_domains() {
        let family = NormalMuSigma::new();

        assert!(
            family
                .deviance(
                    1.7,
                    NormalTheta {
                        mu: 1.7,
                        sigma: 0.0,
                    },
                )
                .is_infinite()
        );
        assert!(
            family
                .deviance(
                    f64::NAN,
                    NormalTheta {
                        mu: 1.7,
                        sigma: 1.0,
                    },
                )
                .is_infinite()
        );
    }

    #[test]
    fn normal_deviance_is_standardized_squared_residual() {
        let family = NormalMuSigma::new();
        let deviance = family.deviance(
            2.5,
            NormalTheta {
                mu: 1.5,
                sigma: 0.5,
            },
        );

        assert_relative_eq!(deviance, 4.0);
    }

    #[test]
    fn normal_cdf_matches_standard_normal_reference_points() {
        let family = NormalMuSigma::new();
        let theta = NormalTheta {
            mu: 2.0,
            sigma: 0.5,
        };

        assert_relative_eq!(family.cdf(theta.mu, theta), 0.5, epsilon = 1.0e-7);
        assert_relative_eq!(
            family.cdf(theta.mu + theta.sigma, theta),
            0.841_344_746,
            epsilon = 1.0e-7
        );
        assert_relative_eq!(
            family.cdf(theta.mu - theta.sigma, theta),
            0.158_655_254,
            epsilon = 1.0e-7
        );
    }

    #[test]
    fn normal_cdf_returns_nan_for_invalid_domains() {
        let family = NormalMuSigma::new();

        assert!(
            family
                .cdf(
                    1.0,
                    NormalTheta {
                        mu: 0.0,
                        sigma: 0.0
                    }
                )
                .is_nan()
        );
        assert!(
            family
                .cdf(
                    f64::NAN,
                    NormalTheta {
                        mu: 0.0,
                        sigma: 1.0
                    }
                )
                .is_nan()
        );
    }

    #[test]
    fn density_helpers_reuse_nll() {
        let family = NormalMuSigma::new();
        let theta = NormalTheta {
            mu: 0.4,
            sigma: 0.8,
        };
        let nll = family.nll(1.7, theta);

        assert_relative_eq!(family.log_density(1.7, theta), -nll, epsilon = 1.0e-12);
        assert_relative_eq!(family.density(1.7, theta), (-nll).exp(), epsilon = 1.0e-12);
    }

    #[test]
    fn normal_quantile_inverts_cdf() {
        let family = NormalMuSigma::new();
        let theta = NormalTheta {
            mu: 2.0,
            sigma: 0.5,
        };

        let y = family.quantile(0.75, theta);

        assert_relative_eq!(family.cdf(y, theta), 0.75, epsilon = 1.0e-7);
        assert_eq!(family.quantile(0.0, theta), f64::NEG_INFINITY);
        assert_eq!(family.quantile(1.0, theta), f64::INFINITY);
        assert!(family.quantile(f64::NAN, theta).is_nan());
    }

    #[test]
    fn normal_quantile_matches_statrs_reference() {
        let family = NormalMuSigma::new();
        let theta = NormalTheta {
            mu: 2.0,
            sigma: 0.5,
        };
        let reference = StatrsNormal::new(theta.mu, theta.sigma).unwrap();

        for p in [0.01, 0.1, 0.5, 0.9, 0.99] {
            assert_relative_eq!(
                family.quantile(p, theta),
                reference.inverse_cdf(p),
                epsilon = 1.0e-6
            );
        }
    }

    #[test]
    fn normal_crps_matches_fixed_values() {
        let family = NormalMuSigma::new();

        assert_relative_eq!(
            family.crps(
                1.0,
                NormalTheta {
                    mu: 0.0,
                    sigma: 2.0,
                },
            ),
            0.662_807_075_536_335_3,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn normal_crps_returns_nan_for_invalid_domains() {
        let family = NormalMuSigma::new();

        assert!(
            family
                .crps(
                    1.0,
                    NormalTheta {
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
                    NormalTheta {
                        mu: 0.0,
                        sigma: 1.0,
                    },
                )
                .is_nan()
        );
    }

    #[test]
    fn normal_model_gradient_matches_finite_difference() {
        let y = vec![0.2, 1.1, 1.8];
        let mu_x = DenseDesign::from_rows(&[[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]);
        let sigma_x = DenseDesign::intercept(y.len());
        let mut model = normal_gamlss(&y, mu_x, sigma_x, NoPenalty, NoPenalty).unwrap();
        let beta = vec![0.1, 0.8, -0.3];
        let eps = 1.0e-6;
        let mut grad = vec![0.0; model.dim()];

        model.gradient(&beta, &mut grad).unwrap();

        for index in 0..beta.len() {
            let mut plus = beta.clone();
            plus[index] += eps;
            let mut minus = beta.clone();
            minus[index] -= eps;
            let finite_difference =
                (model.value(&plus).unwrap() - model.value(&minus).unwrap()) / (2.0 * eps);

            assert_relative_eq!(grad[index], finite_difference, epsilon = 1.0e-6);
        }
    }

    #[cfg(feature = "rand")]
    #[test]
    fn normal_sampling_returns_finite_values_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = NormalMuSigma::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        assert!(
            family
                .sample(
                    &mut rng,
                    super::NormalTheta {
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
                    super::NormalTheta {
                        mu: 0.0,
                        sigma: 0.0
                    }
                )
                .is_nan()
        );
    }
}
