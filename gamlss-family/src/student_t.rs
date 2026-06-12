use std::marker::PhantomData;

use gamlss_core::{
    Family, Identity, Link, Log, ModelError, Mu, ParameterParts, ParameterizedFamily, PositiveLink,
    Sigma,
};

/// Student's t location-scale family с фиксированным числом степеней свободы.
///
/// `MuLink` и `SigmaLink` управляют link-функциями для параметров
/// расположения и масштаба соответственно. По умолчанию используются
/// `Identity` для `mu` и `Log` для `sigma`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudentT<MuLink = Identity, SigmaLink = Log> {
    degrees_of_freedom: f64,
    marker: PhantomData<(MuLink, SigmaLink)>,
}

impl<MuLink, SigmaLink> StudentT<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    /// Creates a Student's t family with finite positive degrees of freedom.
    pub fn try_new(degrees_of_freedom: f64) -> Result<Self, ModelError> {
        if !degrees_of_freedom.is_finite() || degrees_of_freedom <= 0.0 {
            return Err(ModelError::InvalidParameter {
                parameter: "degrees_of_freedom",
                expected: "finite and > 0",
            });
        }

        Ok(Self {
            degrees_of_freedom,
            marker: PhantomData,
        })
    }

    /// Returns the fixed degrees of freedom.
    pub fn degrees_of_freedom(&self) -> f64 {
        self.degrees_of_freedom
    }

    /// Преобразует предикторы с link-шкалы в параметры на естественной шкале.
    #[inline(always)]
    fn theta_from_eta(eta: StudentTEta) -> StudentTTheta {
        StudentTTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
        }
    }

    /// Negative log-likelihood одного наблюдения на естественной шкале.
    ///
    /// Возвращает `INFINITY` при неположительном sigma.
    #[inline(always)]
    fn nll_theta(&self, y: f64, theta: StudentTTheta) -> f64 {
        if theta.sigma <= 0.0 || !theta.sigma.is_finite() {
            return f64::INFINITY;
        }

        let nu = self.degrees_of_freedom;
        let z = (y - theta.mu) / theta.sigma;
        student_t_constant(nu) + theta.sigma.ln() + 0.5 * (nu + 1.0) * (z * z / nu).ln_1p()
    }

    /// Вычисляет NLL и score по eta для одного наблюдения.
    ///
    /// Использует аналитические производные с учётом фиксированного `nu`
    /// и домножает на производные link-функций (chain rule).
    #[inline(always)]
    fn nll_and_score_eta_values(&self, y: f64, eta: StudentTEta) -> (f64, StudentTEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = self.nll_theta(y, theta);

        let nu = self.degrees_of_freedom;
        let sigma = theta.sigma;
        let z = (y - theta.mu) / sigma;
        let slope = (nu + 1.0) * z / (nu + z * z);
        let d_nll_d_mu = -slope / sigma;
        let d_nll_d_sigma = (1.0 - slope * z) / sigma;

        let score_eta = StudentTEta {
            mu: d_nll_d_mu * MuLink::derivative_inverse(eta.mu),
            sigma: d_nll_d_sigma * SigmaLink::derivative_inverse(eta.sigma),
        };

        (nll, score_eta)
    }
}

impl<MuLink, SigmaLink> Default for StudentT<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::try_new(5.0).expect("default degrees_of_freedom is valid")
    }
}

/// Predictors для распределения Стьюдента на link-шкале.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudentTEta {
    /// Location predictor.
    pub mu: f64,
    /// Scale predictor.
    pub sigma: f64,
}

impl ParameterParts<2> for StudentTEta {
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
            _ => unreachable!("student-t eta only has indices 0 and 1"),
        }
    }
}

/// Параметры распределения Стьюдента на естественной шкале.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudentTTheta {
    /// Location parameter.
    pub mu: f64,
    /// Positive scale parameter.
    pub sigma: f64,
}

impl<MuLink, SigmaLink> Family for StudentT<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    type Eta = StudentTEta;
    type Theta = StudentTTheta;
    type ScoreEta = StudentTEta;

    #[inline(always)]
    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline(always)]
    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        self.nll_theta(y, theta)
    }

    #[inline(always)]
    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        self.nll_theta(y, Self::theta_from_eta(eta))
    }

    #[inline(always)]
    fn nll_and_score_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::ScoreEta) {
        self.nll_and_score_eta_values(y, eta)
    }
}

impl<MuLink, SigmaLink> ParameterizedFamily<2> for StudentT<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    type Params = (Mu, Sigma);
    type Links = (MuLink, SigmaLink);
}

/// Распределение Стьюдента с `Identity` link для `mu` и `Log` link для `sigma`.
pub type DefaultStudentT = StudentT<Identity, Log>;

/// Нормировочная константа логарифма плотности распределения Стьюдента.
fn student_t_constant(nu: f64) -> f64 {
    0.5 * (nu.ln() + std::f64::consts::PI.ln()) + ln_gamma(0.5 * nu) - ln_gamma(0.5 * (nu + 1.0))
}

/// Приближение логарифма гамма-функции (алгоритм Lanczos).
///
/// Используется для вычисления нормировочной константы плотности
/// распределения Стьюдента. Точность достаточна для типовых приложений.
fn ln_gamma(value: f64) -> f64 {
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if value < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * value).sin().ln()
            - ln_gamma(1.0 - value);
    }

    let shifted = value - 1.0;
    let mut x = COEFFICIENTS[0];
    for (index, coefficient) in COEFFICIENTS.iter().copied().enumerate().skip(1) {
        x += coefficient / (shifted + index as f64);
    }
    let t = shifted + 7.5;

    0.5 * (2.0 * std::f64::consts::PI).ln() + (shifted + 0.5) * t.ln() - t + x.ln()
}

#[cfg(test)]
mod tests {
    use super::DefaultStudentT;
    use crate::test_support::assert_score_matches_finite_difference;

    #[test]
    fn student_t_rejects_invalid_degrees_of_freedom() {
        assert!(DefaultStudentT::try_new(0.0).is_err());
        assert!(DefaultStudentT::try_new(f64::INFINITY).is_err());
    }

    #[test]
    fn student_t_score_matches_finite_difference() {
        let family = DefaultStudentT::try_new(5.0).unwrap();
        assert_score_matches_finite_difference::<_, 2>(&family, 1.7, [0.4, -0.2]);
    }
}
