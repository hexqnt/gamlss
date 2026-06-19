use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Family, HasCdf, HasCrps, HasQuantile, Identity, InitialEtaFromTheta, Link, Log, ModelError, Mu,
    ObservationView, ParameterParts, ParameterizedFamily, PositiveLink, Sigma,
};

use crate::initial::{robust_location_scale, weighted_values};
use crate::special::{invert_real_cdf, ln_beta, ln_gamma, regularized_beta};

/// Student's t location-scale family с фиксированным числом степеней свободы.
///
/// `MuLink` и `SigmaLink` управляют link-функциями для параметров
/// расположения и масштаба соответственно. По умолчанию используются
/// `Identity` для `mu` и `Log` для `sigma`.
///
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
    /// Возвращает `INFINITY` при non-finite observation/location или
    /// неположительном sigma.
    #[inline(always)]
    fn nll_theta(&self, y: f64, theta: StudentTTheta) -> f64 {
        if !y.is_finite() || !theta.mu.is_finite() || theta.sigma <= 0.0 || !theta.sigma.is_finite()
        {
            return f64::INFINITY;
        }

        let nu = self.degrees_of_freedom;
        let z = (y - theta.mu) / theta.sigma;
        student_t_constant(nu) + theta.sigma.ln() + 0.5 * (nu + 1.0) * (z * z / nu).ln_1p()
    }

    /// Вычисляет NLL и gradient по eta для одного наблюдения.
    ///
    /// Использует аналитические производные с учётом фиксированного `nu`
    /// и домножает на производные link-функций (chain rule).
    #[inline(always)]
    fn nll_and_gradient_eta_values(&self, y: f64, eta: StudentTEta) -> (f64, StudentTEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = self.nll_theta(y, theta);
        if !nll.is_finite() {
            return (
                nll,
                StudentTEta {
                    mu: f64::NAN,
                    sigma: f64::NAN,
                },
            );
        }

        let nu = self.degrees_of_freedom;
        let sigma = theta.sigma;
        let z = (y - theta.mu) / sigma;
        let slope = (nu + 1.0) * z / (nu + z * z);
        let d_nll_d_mu = -slope / sigma;
        let d_nll_d_sigma = (1.0 - slope * z) / sigma;

        let gradient_eta = StudentTEta {
            mu: d_nll_d_mu * MuLink::derivative_inverse(eta.mu),
            sigma: d_nll_d_sigma * SigmaLink::derivative_inverse(eta.sigma),
        };

        (nll, gradient_eta)
    }

    fn standard_cdf(&self, t: f64) -> f64 {
        if !t.is_finite() {
            return if t.is_sign_negative() { 0.0 } else { 1.0 };
        }
        if t == 0.0 {
            return 0.5;
        }

        let nu = self.degrees_of_freedom;
        let beta = regularized_beta(0.5 * nu, 0.5, nu / (nu + t * t));
        if t < 0.0 {
            0.5 * beta
        } else {
            1.0 - 0.5 * beta
        }
    }

    fn standard_quantile(&self, p: f64) -> f64 {
        if p < 0.0 || !p.is_finite() || p > 1.0 {
            return f64::NAN;
        }
        if p == 0.0 {
            return f64::NEG_INFINITY;
        }
        if p == 1.0 {
            return f64::INFINITY;
        }
        if p == 0.5 {
            return 0.0;
        }

        invert_real_cdf(p, |t| self.standard_cdf(t))
    }

    fn standard_density(&self, t: f64) -> f64 {
        if !t.is_finite() {
            return 0.0;
        }

        let nu = self.degrees_of_freedom;
        (-student_t_constant(nu) - 0.5 * (nu + 1.0) * (t * t / nu).ln_1p()).exp()
    }

    fn standard_crps_constant(&self) -> f64 {
        let nu = self.degrees_of_freedom;
        let log_beta_half_nu_minus_half = ln_beta(0.5, nu - 0.5);
        let log_beta_half_nu_half = ln_beta(0.5, 0.5 * nu);

        2.0 * nu.sqrt() / (nu - 1.0)
            * (log_beta_half_nu_minus_half - 2.0 * log_beta_half_nu_half).exp()
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
    type NllGradientEta = StudentTEta;
    type Observation<'obs> = f64;

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
    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        self.nll_and_gradient_eta_values(y, eta)
    }
}

impl<MuLink, SigmaLink> ParameterizedFamily<2> for StudentT<MuLink, SigmaLink>
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
            return StudentTEta::from_array([0.0, 0.0]);
        };

        StudentTEta {
            mu: MuLink::initial_eta_from_theta(mu),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
        }
    }
}

impl<MuLink, SigmaLink> HasCdf for StudentT<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
        if !y.is_finite() || !theta.mu.is_finite() || theta.sigma <= 0.0 || !theta.sigma.is_finite()
        {
            return f64::NAN;
        }

        self.standard_cdf((y - theta.mu) / theta.sigma)
    }
}

impl<MuLink, SigmaLink> HasQuantile for StudentT<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
        if theta.sigma <= 0.0 || !theta.sigma.is_finite() || !theta.mu.is_finite() {
            return f64::NAN;
        }

        theta.mu + theta.sigma * self.standard_quantile(p)
    }
}

impl<MuLink, SigmaLink> HasCrps for StudentT<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn crps<'obs>(&self, y: Self::Observation<'obs>, theta: Self::Theta) -> f64 {
        if !y.is_finite()
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || !theta.sigma.is_finite()
            || self.degrees_of_freedom <= 1.0
        {
            return f64::NAN;
        }

        let nu = self.degrees_of_freedom;
        let z = (y - theta.mu) / theta.sigma;
        let cdf = self.standard_cdf(z);
        let density = self.standard_density(z);
        let tail_moment = 2.0 * density * (nu + z * z) / (nu - 1.0);

        theta.sigma * (z * (2.0 * cdf - 1.0) + tail_moment - self.standard_crps_constant())
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink, SigmaLink> CanSimulate<Rng> for StudentT<MuLink, SigmaLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn sample(&self, rng: &mut Rng, theta: Self::Theta) -> f64 {
        if theta.sigma <= 0.0 || !theta.sigma.is_finite() || !theta.mu.is_finite() {
            return f64::NAN;
        }

        let z = rand_distr::Distribution::sample(
            &rand_distr::StudentT::new(self.degrees_of_freedom)
                .expect("validated degrees_of_freedom must construct"),
            rng,
        );
        theta.mu + theta.sigma * z
    }
}

/// Распределение Стьюдента с `Identity` link для `mu` и `Log` link для `sigma`.
pub type DefaultStudentT = StudentT<Identity, Log>;

/// Нормировочная константа логарифма плотности распределения Стьюдента.
fn student_t_constant(nu: f64) -> f64 {
    0.5 * (nu.ln() + std::f64::consts::PI.ln()) + ln_gamma(0.5 * nu) - ln_gamma(0.5 * (nu + 1.0))
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;
    use gamlss_core::{Family, HasCdf, HasCrps, HasQuantile};
    use statrs::distribution::{ContinuousCDF, StudentsT};

    use super::{DefaultStudentT, StudentTEta, StudentTTheta};
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn student_t_rejects_invalid_degrees_of_freedom() {
        assert!(DefaultStudentT::try_new(0.0).is_err());
        assert!(DefaultStudentT::try_new(f64::INFINITY).is_err());
    }

    #[test]
    fn student_t_gradient_matches_finite_difference() {
        let family = DefaultStudentT::try_new(5.0).unwrap();
        assert_gradient_matches_finite_difference::<_, 2>(&family, 1.7, [0.4, -0.2]);
    }

    #[test]
    fn student_t_rejects_non_finite_domain_and_returns_nan_gradient() {
        let family = DefaultStudentT::try_new(5.0).unwrap();
        let theta = StudentTTheta {
            mu: 0.4,
            sigma: 0.8,
        };

        assert!(family.nll(1.7, theta).is_finite());
        assert!(family.nll(f64::NEG_INFINITY, theta).is_infinite());
        assert!(
            family
                .nll(
                    1.7,
                    StudentTTheta {
                        mu: f64::NAN,
                        sigma: theta.sigma,
                    },
                )
                .is_infinite()
        );
        assert!(
            family
                .nll(
                    1.7,
                    StudentTTheta {
                        mu: theta.mu,
                        sigma: 0.0,
                    },
                )
                .is_infinite()
        );

        let (nll, gradient) = family.nll_and_gradient_eta(
            1.7,
            StudentTEta {
                mu: 0.4,
                sigma: f64::NEG_INFINITY,
            },
        );
        assert!(nll.is_infinite());
        assert!(gradient.mu.is_nan());
        assert!(gradient.sigma.is_nan());
    }

    #[test]
    fn student_t_cdf_matches_reference_points() {
        let family = DefaultStudentT::try_new(5.0).unwrap();
        let theta = StudentTTheta {
            mu: 0.4,
            sigma: 0.8,
        };

        assert_relative_eq!(family.cdf(theta.mu, theta), 0.5, epsilon = 1.0e-12);
        assert_relative_eq!(
            family.cdf(theta.mu + theta.sigma, theta),
            0.818_391_266_175_438_7,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            family.cdf(theta.mu - theta.sigma, theta),
            0.181_608_733_824_561_27,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn student_t_cdf_and_quantile_match_statrs_reference() {
        let family = DefaultStudentT::try_new(7.0).unwrap();
        let theta = StudentTTheta {
            mu: 0.4,
            sigma: 0.8,
        };
        let reference = StudentsT::new(theta.mu, theta.sigma, 7.0).unwrap();

        for y in [-2.0, -0.3, 0.4, 1.2, 3.0] {
            assert_relative_eq!(family.cdf(y, theta), reference.cdf(y), epsilon = 1.0e-11);
        }

        for p in [0.01, 0.1, 0.5, 0.9, 0.99] {
            assert_relative_eq!(
                family.quantile(p, theta),
                reference.inverse_cdf(p),
                epsilon = 1.0e-10
            );
        }
    }

    #[test]
    fn student_t_quantile_inverts_cdf() {
        let family = DefaultStudentT::try_new(5.0).unwrap();
        let theta = StudentTTheta {
            mu: 0.4,
            sigma: 0.8,
        };

        let y = family.quantile(0.75, theta);

        assert_relative_eq!(family.cdf(y, theta), 0.75, epsilon = 1.0e-12);
        assert_eq!(family.quantile(0.0, theta), f64::NEG_INFINITY);
        assert_eq!(family.quantile(1.0, theta), f64::INFINITY);
        assert!(family.quantile(f64::NAN, theta).is_nan());
    }

    #[test]
    fn student_t_cdf_returns_nan_for_invalid_domains() {
        let family = DefaultStudentT::try_new(5.0).unwrap();

        assert!(
            family
                .cdf(
                    1.0,
                    StudentTTheta {
                        mu: 0.0,
                        sigma: 0.0,
                    },
                )
                .is_nan()
        );
        assert!(
            family
                .cdf(
                    f64::NAN,
                    StudentTTheta {
                        mu: 0.0,
                        sigma: 1.0,
                    },
                )
                .is_nan()
        );
    }

    #[test]
    fn student_t_crps_matches_fixed_values() {
        let family = DefaultStudentT::try_new(5.0).unwrap();

        assert_relative_eq!(
            family.crps(
                1.0,
                StudentTTheta {
                    mu: 0.0,
                    sigma: 1.0,
                },
            ),
            0.603_830_562_748_23,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            family.crps(
                0.0,
                StudentTTheta {
                    mu: 0.0,
                    sigma: 1.0,
                },
            ),
            0.257_025_362_900_647_5,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn student_t_crps_scales_with_sigma() {
        let family = DefaultStudentT::try_new(5.0).unwrap();

        assert_relative_eq!(
            family.crps(
                2.0,
                StudentTTheta {
                    mu: 0.0,
                    sigma: 2.0,
                },
            ),
            2.0 * family.crps(
                1.0,
                StudentTTheta {
                    mu: 0.0,
                    sigma: 1.0,
                },
            ),
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn student_t_crps_returns_nan_for_invalid_domains() {
        let family = DefaultStudentT::try_new(5.0).unwrap();

        assert!(
            family
                .crps(
                    f64::NAN,
                    StudentTTheta {
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
                    StudentTTheta {
                        mu: 0.0,
                        sigma: 0.0,
                    },
                )
                .is_nan()
        );
        assert!(
            DefaultStudentT::try_new(1.0)
                .unwrap()
                .crps(
                    1.0,
                    StudentTTheta {
                        mu: 0.0,
                        sigma: 1.0,
                    },
                )
                .is_nan()
        );
    }

    #[test]
    fn student_t_crps_is_nonnegative_for_valid_domains() {
        let family = DefaultStudentT::try_new(5.0).unwrap();

        assert!(
            family.crps(
                1.0,
                StudentTTheta {
                    mu: 0.0,
                    sigma: 1.0,
                },
            ) >= 0.0
        );
    }

    #[cfg(feature = "rand")]
    #[test]
    fn student_t_sampling_returns_finite_values_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = DefaultStudentT::try_new(5.0).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        assert!(
            family
                .sample(
                    &mut rng,
                    StudentTTheta {
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
                    StudentTTheta {
                        mu: 0.0,
                        sigma: 0.0
                    }
                )
                .is_nan()
        );
    }
}
