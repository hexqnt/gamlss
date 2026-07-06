#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Family, HasCdf, HasQuantile, InitialEtaFromObservations, InitialEtaFromTheta, Log, Logit, Mu,
    Nu, ObservationView, ParameterParts, PositiveLink, ScalarParams, Sigma, UnitIntervalLink,
};

use gamlss_special::{invert_positive_cdf, regularized_gamma_lower};

use crate::initial::{positive_floor, probability_floor, weighted_summary, weighted_values};

use super::{Zaga, ZagaTheta};

/// ZAGA distribution with log/log/logit links.
pub type ZagaMeanSigmaZeroProbability = Zaga<Log, Log, Logit>;
/// Explicit alias for the component-mean/CV/zero-probability ZAGA kernel parameterization.
pub type ZagaComponentMeanCvZeroProbability = ZagaMeanSigmaZeroProbability;

/// Predictors for ZAGA on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZagaEta {
    /// Positive mean predictor for the gamma component.
    pub mu: f64,
    /// Positive coefficient-of-variation predictor for the gamma component.
    pub sigma: f64,
    /// Zero-mass probability predictor.
    pub nu: f64,
}

impl ParameterParts<3> for ZagaEta {
    fn from_array(values: [f64; 3]) -> Self {
        Self {
            mu: values[0],
            sigma: values[1],
            nu: values[2],
        }
    }

    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mu,
            1 => self.sigma,
            2 => self.nu,
            _ => unreachable!("zaga eta only has indices 0 through 2"),
        }
    }
}

impl<MuLink, SigmaLink, NuLink> Zaga<MuLink, SigmaLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    #[inline]
    fn theta_from_eta(eta: ZagaEta) -> ZagaTheta {
        ZagaTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
            nu: NuLink::inverse(eta.nu),
        }
    }

    #[inline]
    fn nll_and_gradient_eta_values(y: f64, eta: ZagaEta) -> (f64, ZagaEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (nll, ZagaEta::from_array([f64::NAN; 3]));
        }
        let gradient = Self::gradient_component_theta(y, theta);
        (
            nll,
            ZagaEta {
                mu: gradient.mu * MuLink::derivative_inverse(eta.mu),
                sigma: gradient.sigma * SigmaLink::derivative_inverse(eta.sigma),
                nu: gradient.nu * NuLink::derivative_inverse(eta.nu),
            },
        )
    }
}

impl<MuLink, SigmaLink, NuLink> Family for Zaga<MuLink, SigmaLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    type Eta = ZagaEta;
    type Theta = ZagaTheta;
    type GradientEta = ZagaEta;
    type Observation<'obs> = f64;
    type Workspace = ();
    type ParamSpec = ScalarParams<(Mu, Sigma, Nu), (MuLink, SigmaLink, NuLink), 3>;
    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        Self::nll_theta(y, *theta)
    }

    fn nll_eta(&self, y: f64, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> f64 {
        Self::nll_theta(y, Self::theta_from_eta(*eta))
    }

    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        Self::nll_and_gradient_eta_values(y, *eta)
    }
}

impl<MuLink, SigmaLink, NuLink> InitialEtaFromObservations<3> for Zaga<MuLink, SigmaLink, NuLink>
where
    MuLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    NuLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values =
            weighted_values::<Self, _, _>(obs, |y| (y.is_finite() && y >= 0.0).then_some(y));
        let positives = values
            .iter()
            .copied()
            .filter(|(y, _)| *y > 0.0)
            .collect::<Vec<_>>();
        let summary = weighted_summary(&positives);
        let mu = positive_floor(summary.map_or(1.0, |s| s.mean));
        let sigma = positive_floor(summary.map_or(1.0, |s| s.variance.sqrt() / mu));
        let zero_weight = values
            .iter()
            .filter(|(y, _)| *y == 0.0)
            .map(|(_, w)| *w)
            .sum::<f64>();
        let total_weight = values.iter().map(|(_, w)| *w).sum::<f64>();
        let zero_rate = if total_weight > 0.0 {
            zero_weight / total_weight
        } else {
            0.1
        };

        ZagaEta {
            mu: MuLink::initial_eta_from_theta(mu),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
            nu: NuLink::initial_eta_from_theta(probability_floor(zero_rate)),
        }
    }
}

impl<MuLink, SigmaLink, NuLink> HasCdf for Zaga<MuLink, SigmaLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        Self::cdf_theta(y, *theta)
    }
}

impl<MuLink, SigmaLink, NuLink> HasQuantile for Zaga<MuLink, SigmaLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        if !(0.0..=1.0).contains(&p) || theta.nu <= 0.0 || theta.nu >= 1.0 {
            return f64::NAN;
        }
        if theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || !theta.sigma.is_finite()
        {
            return f64::NAN;
        }
        if p <= theta.nu {
            return 0.0;
        }
        let shape = 1.0 / (theta.sigma * theta.sigma);
        let rate = 1.0 / (theta.sigma * theta.sigma * theta.mu);
        let target = (p - theta.nu) / (1.0 - theta.nu);
        invert_positive_cdf(target, |y| regularized_gamma_lower(shape, rate * y))
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink, SigmaLink, NuLink> CanSimulate<Rng> for Zaga<MuLink, SigmaLink, NuLink>
where
    Rng: rand::Rng,
    MuLink: PositiveLink<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    type Sample = f64;

    fn sample(&self, rng: &mut Rng, theta: &Self::Theta) -> f64 {
        Self::sample_component_theta(rng, *theta)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;

    use super::{ZagaMeanSigmaZeroProbability, ZagaTheta};

    #[cfg(feature = "rand")]
    #[test]
    fn zaga_sampling_returns_nonnegative_values_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = ZagaMeanSigmaZeroProbability::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family.sample(
            &mut rng,
            &ZagaTheta {
                mu: 1.5,
                sigma: 0.7,
                nu: 0.2,
            },
        );
        assert!(sample >= 0.0 && sample.is_finite());
        assert!(
            family
                .sample(
                    &mut rng,
                    &ZagaTheta {
                        mu: 1.5,
                        sigma: 0.0,
                        nu: 0.2,
                    }
                )
                .is_nan()
        );
    }
}
