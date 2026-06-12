use std::marker::PhantomData;

use gamlss_core::{
    Family, Identity, Link, Log, Mu, ParameterParts, ParameterizedFamily, PositiveLink, Sigma,
};

const LOG_2: f64 = std::f64::consts::LN_2;

/// Laplace location-scale family.
///
/// `MuLink` и `SigmaLink` управляют link-функциями. По умолчанию
/// `Identity` для `mu` и `Log` для `sigma` (positive link).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Laplace<MuLink = Identity, SigmaLink = Log> {
    marker: PhantomData<(MuLink, SigmaLink)>,
}

impl<MuLink, SigmaLink> Laplace<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    /// Creates a stateless Laplace family.
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    /// Преобразует предикторы с link-шкалы в параметры на естественной шкале.
    #[inline(always)]
    fn theta_from_eta(eta: LaplaceEta) -> LaplaceTheta {
        LaplaceTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
        }
    }

    /// Negative log-likelihood одного наблюдения на естественной шкале.
    ///
    /// Возвращает `INFINITY` при неположительном sigma.
    #[inline(always)]
    fn nll_theta(y: f64, theta: LaplaceTheta) -> f64 {
        if theta.sigma <= 0.0 || !theta.sigma.is_finite() {
            return f64::INFINITY;
        }

        LOG_2 + theta.sigma.ln() + (y - theta.mu).abs() / theta.sigma
    }

    /// Вычисляет NLL и score по eta для одного наблюдения.
    ///
    /// Градиент по `mu` использует субградиент sign (0 при `residual == 0`).
    #[inline(always)]
    fn nll_and_score_eta_values(y: f64, eta: LaplaceEta) -> (f64, LaplaceEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);

        let residual = y - theta.mu;
        let d_nll_d_mu = if residual > 0.0 {
            -1.0 / theta.sigma
        } else if residual < 0.0 {
            1.0 / theta.sigma
        } else {
            0.0
        };
        let d_nll_d_sigma = 1.0 / theta.sigma - residual.abs() / (theta.sigma * theta.sigma);

        let score_eta = LaplaceEta {
            mu: d_nll_d_mu * MuLink::derivative_inverse(eta.mu),
            sigma: d_nll_d_sigma * SigmaLink::derivative_inverse(eta.sigma),
        };

        (nll, score_eta)
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

/// Predictors для распределения Лапласа на link-шкале.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaplaceEta {
    /// Location predictor.
    pub mu: f64,
    /// Scale predictor.
    pub sigma: f64,
}

impl ParameterParts<2> for LaplaceEta {
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
            _ => unreachable!("laplace eta only has indices 0 and 1"),
        }
    }
}

/// Параметры распределения Лапласа на естественной шкале.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaplaceTheta {
    /// Location parameter.
    pub mu: f64,
    /// Positive scale parameter.
    pub sigma: f64,
}

impl<MuLink, SigmaLink> Family for Laplace<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    type Eta = LaplaceEta;
    type Theta = LaplaceTheta;
    type ScoreEta = LaplaceEta;

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

impl<MuLink, SigmaLink> ParameterizedFamily<2> for Laplace<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    type Params = (Mu, Sigma);
    type Links = (MuLink, SigmaLink);
}

/// Распределение Лапласа с `Identity` link для `mu` и `Log` link для `sigma`.
pub type DefaultLaplace = Laplace<Identity, Log>;

#[cfg(test)]
mod tests {
    use super::DefaultLaplace;
    use crate::test_support::assert_score_matches_finite_difference;

    #[test]
    fn laplace_score_matches_finite_difference() {
        let family = DefaultLaplace::new();
        assert_score_matches_finite_difference::<_, 2>(&family, 1.7, [0.4, -0.2]);
    }
}
