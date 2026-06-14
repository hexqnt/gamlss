use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    DesignMatrix, Family, Gamlss, Identity, LinearPredictorBlock, Link, Log, ModelError, Mu,
    NoPenalty, ParameterBlock, ParameterBlocks, ParameterParts, ParameterizedFamily, Penalty,
    PositiveLink, Sigma,
};

const HALF_LOG_2_PI: f64 = 0.918_938_533_204_672_7;

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
    /// Возвращает `INFINITY` при неположительном sigma.
    #[inline(always)]
    fn nll_theta(y: f64, theta: NormalTheta) -> f64 {
        if theta.sigma <= 0.0 || !theta.sigma.is_finite() {
            return f64::INFINITY;
        }

        let residual = y - theta.mu;
        let z = residual / theta.sigma;
        HALF_LOG_2_PI + theta.sigma.ln() + 0.5 * z * z
    }

    /// Вычисляет NLL и score по eta для одного наблюдения.
    ///
    /// Использует аналитические производные NLL по `mu` и `sigma` и
    /// домножает на производные link-функций (chain rule).
    #[inline(always)]
    fn nll_and_score_eta_values(y: f64, eta: NormalEta) -> (f64, NormalEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);

        let residual = y - theta.mu;
        let sigma2 = theta.sigma * theta.sigma;
        let d_nll_d_mu = (theta.mu - y) / sigma2;
        let d_nll_d_sigma = (1.0 / theta.sigma) - (residual * residual / (sigma2 * theta.sigma));

        let score_eta = NormalEta {
            mu: d_nll_d_mu * MuLink::derivative_inverse(eta.mu),
            sigma: d_nll_d_sigma * SigmaLink::derivative_inverse(eta.sigma),
        };

        (nll, score_eta)
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
    type ScoreEta = NormalEta;

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
    fn nll_and_score_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::ScoreEta) {
        Self::nll_and_score_eta_values(y, eta)
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
    'a,
    DefaultNormal,
    (
        ParameterBlock<Mu, Identity, LinearPredictorBlock<XMu>, PMu>,
        ParameterBlock<Sigma, Log, LinearPredictorBlock<XSigma>, PSigma>,
    ),
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
    use gamlss_core::{DenseDesign, NoPenalty, Objective};

    use super::{DefaultNormal, normal_gamlss};
    use crate::test_support::assert_score_matches_finite_difference;

    #[test]
    fn normal_score_matches_finite_difference() {
        let family = DefaultNormal::new();
        assert_score_matches_finite_difference::<_, 2>(&family, 1.7, [0.4, -0.2]);
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
