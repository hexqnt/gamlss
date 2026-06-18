use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    DesignMatrix, Family, Gamlss, HasCdf, HasCrps, HasDeviance, HasInitialEta, Identity,
    LinearPredictorBlock, Link, Log, ModelError, Mu, NoPenalty, ParameterBlock, ParameterBlocks,
    ParameterParts, ParameterizedFamily, Penalty, PositiveLink, Sigma,
};

use crate::special::unit_normal_cdf;

const HALF_LOG_2_PI: f64 = 0.918_938_533_204_672_7;
const INV_SQRT_2_PI: f64 = 0.398_942_280_401_432_7;
const INV_SQRT_PI: f64 = 0.564_189_583_547_756_3;
const DEFAULT_INITIAL_LOG_SIGMA: f64 = 0.0;

/// Нормальное распределение с типизированными link-функциями для `mu` и `sigma`.
///
/// `SigmaLink` обязан быть positive link, чтобы scale-параметр оставался
/// положительным на уровне типов.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normal<MuLink = Identity, SigmaLink = Log> {
    marker: PhantomData<(MuLink, SigmaLink)>,
}

impl<MuLink, SigmaLink> Normal<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    /// Создаёт stateless значение family.
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    /// Преобразует предикторы с link-шкалы в параметры на естественной шкале.
    #[inline(always)]
    fn theta_from_eta(eta: NormalEta) -> NormalTheta {
        NormalTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
        }
    }

    /// Negative log-likelihood одного наблюдения на естественной шкале.
    ///
    /// Возвращает `INFINITY` при non-finite observation/location или
    /// неположительном sigma.
    #[inline(always)]
    fn nll_theta(y: f64, theta: NormalTheta) -> f64 {
        if !y.is_finite() || !theta.mu.is_finite() || theta.sigma <= 0.0 || !theta.sigma.is_finite()
        {
            return f64::INFINITY;
        }

        let residual = y - theta.mu;
        let z = residual / theta.sigma;
        HALF_LOG_2_PI + theta.sigma.ln() + 0.5 * z * z
    }

    /// Вычисляет NLL и gradient по eta для одного наблюдения.
    ///
    /// Использует аналитические производные NLL по `mu` и `sigma` и
    /// домножает на производные link-функций (chain rule).
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

/// Предикторы нормального распределения на link-шкале.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalEta {
    /// Предиктор для `mu`.
    pub mu: f64,
    /// Предиктор для `sigma`.
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

/// Параметры нормального распределения на естественной шкале.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalTheta {
    /// Location-параметр.
    pub mu: f64,
    /// Положительный scale-параметр.
    pub sigma: f64,
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
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    type Params = (Mu, Sigma);
    type Links = (MuLink, SigmaLink);
}

impl<MuLink, SigmaLink> HasDeviance for Normal<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn deviance<'obs>(&self, y: Self::Observation<'obs>, theta: Self::Theta) -> f64 {
        if !y.is_finite() || !theta.mu.is_finite() || theta.sigma <= 0.0 || !theta.sigma.is_finite()
        {
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
        if !y.is_finite() || !theta.mu.is_finite() || theta.sigma <= 0.0 || !theta.sigma.is_finite()
        {
            return f64::NAN;
        }

        unit_normal_cdf((y - theta.mu) / theta.sigma)
    }
}

impl<MuLink, SigmaLink> HasCrps for Normal<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn crps<'obs>(&self, y: Self::Observation<'obs>, theta: Self::Theta) -> f64 {
        if !y.is_finite() || !theta.mu.is_finite() || theta.sigma <= 0.0 || !theta.sigma.is_finite()
        {
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
        if theta.sigma <= 0.0 || !theta.sigma.is_finite() || !theta.mu.is_finite() {
            return f64::NAN;
        }

        rand_distr::Distribution::sample(
            &rand_distr::Normal::new(theta.mu, theta.sigma)
                .expect("validated normal parameters must construct"),
            rng,
        )
    }
}

/// Нормальное распределение с `Identity` link для `mu` и `Log` link для `sigma`.
pub type DefaultNormal = Normal<Identity, Log>;

/// Типизированная GAMLSS-модель для normal family по умолчанию.
///
/// The lifetime tracks the borrowed response slice.
pub type NormalGamlss<'a, XMu, XSigma, PMu = NoPenalty, PSigma = NoPenalty> = Gamlss<
    DefaultNormal,
    (
        ParameterBlock<Mu, Identity, LinearPredictorBlock<XMu>, PMu>,
        ParameterBlock<Sigma, Log, LinearPredictorBlock<XSigma>, PSigma>,
    ),
    &'a [f64],
>;

/// Создаёт normal GAMLSS-модель из response, двух design matrices и штрафов.
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

    Gamlss::try_new(DefaultNormal::new(), blocks, y)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;
    use gamlss_core::{
        DenseDesign, Family, HasCdf, HasCrps, HasDeviance, HasInitialEta, NoPenalty, Objective,
    };

    use super::{DEFAULT_INITIAL_LOG_SIGMA, DefaultNormal, NormalEta, NormalTheta, normal_gamlss};
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn normal_gradient_matches_finite_difference() {
        let family = DefaultNormal::new();
        assert_gradient_matches_finite_difference::<_, 2>(&family, 1.7, [0.4, -0.2]);
    }

    #[test]
    fn normal_rejects_non_finite_domain_and_returns_nan_gradient() {
        let family = DefaultNormal::new();
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
        let family = DefaultNormal::new();
        let eta = family.initial_eta(1.7);

        assert_relative_eq!(eta.mu, 1.7);
        assert_relative_eq!(eta.sigma, DEFAULT_INITIAL_LOG_SIGMA);
        assert!(family.nll_eta(1.7, eta).is_finite());
    }

    #[test]
    fn normal_initial_eta_propagates_invalid_observation_without_panic() {
        let family = DefaultNormal::new();
        let invalid_eta = family.initial_eta(f64::NAN);

        assert!(invalid_eta.mu.is_nan());
        assert!(invalid_eta.sigma.is_finite());
    }

    #[test]
    fn normal_deviance_returns_non_finite_for_invalid_domains() {
        let family = DefaultNormal::new();

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
        let family = DefaultNormal::new();
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
        let family = DefaultNormal::new();
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
        let family = DefaultNormal::new();

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
    fn normal_crps_matches_fixed_values() {
        let family = DefaultNormal::new();

        assert_relative_eq!(
            family.crps(
                1.0,
                NormalTheta {
                    mu: 0.0,
                    sigma: 2.0,
                },
            ),
            0.662_807_065_409_673_2,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn normal_crps_returns_nan_for_invalid_domains() {
        let family = DefaultNormal::new();

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

        let family = DefaultNormal::new();
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
