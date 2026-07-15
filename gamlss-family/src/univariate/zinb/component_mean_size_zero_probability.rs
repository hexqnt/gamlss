#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Family, HasCdf, HasQuantile, InitialEtaFromObservations, InitialEtaFromTheta, Log, Logit, Mu,
    Nu, ObservationView, ParameterParts, PositiveLink, Shape, UnitIntervalLink,
};

use gamlss_special::{discrete_quantile, is_nonnegative_integer};

use crate::domain::{is_positive_finite, is_strict_probability};
use crate::initial::{
    LARGE_SHAPE, positive_floor, probability_floor, weighted_summary, weighted_values,
};

use super::{MAX_CDF_TERMS, Zinb, ZinbTheta};

/// ZINB distribution parameterized by component mean $\mu$, size $r$, and structural-zero probability $\pi$.
///
/// The eta fields `mu`, `shape`, and `nu` represent $\eta_\mu,\eta_r,\eta_\pi$, respectively. The default links give $\mu=\exp(\eta_\mu)$, $r=\exp(\eta_r)$, and $\pi=\operatorname{logit}^{-1}(\eta_\pi)$.
#[allow(clippy::doc_markdown)]
pub type ZinbMeanSizeZeroProbability = Zinb<Log, Log, Logit>;
/// Explicit alias for the component-mean/size/zero-probability ZINB kernel parameterization.
pub type ZinbComponentMeanSizeZeroProbability = ZinbMeanSizeZeroProbability;

/// Predictors for ZINB on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZinbEta {
    /// Negative-binomial mean predictor.
    pub mu: f64,
    /// Negative-binomial shape predictor.
    pub shape: f64,
    /// Zero-inflation probability predictor.
    pub nu: f64,
}

impl ParameterParts<3> for ZinbEta {
    fn from_array(values: [f64; 3]) -> Self {
        Self {
            mu: values[0],
            shape: values[1],
            nu: values[2],
        }
    }

    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mu,
            1 => self.shape,
            2 => self.nu,
            _ => unreachable!("zinb eta only has indices 0 through 2"),
        }
    }
}

impl<MuLink, ShapeLink, NuLink> Zinb<MuLink, ShapeLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    #[inline]
    fn theta_from_eta(eta: ZinbEta) -> ZinbTheta {
        ZinbTheta {
            mu: MuLink::inverse(eta.mu),
            shape: ShapeLink::inverse(eta.shape),
            nu: NuLink::inverse(eta.nu),
        }
    }

    #[inline]
    fn nll_and_gradient_eta_values(y: f64, eta: ZinbEta) -> (f64, ZinbEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (nll, ZinbEta::from_array([f64::NAN; 3]));
        }
        let gradient = Self::gradient_component_theta(y, theta);
        (
            nll,
            ZinbEta {
                mu: gradient.mu * MuLink::derivative_inverse(eta.mu),
                shape: gradient.shape * ShapeLink::derivative_inverse(eta.shape),
                nu: gradient.nu * NuLink::derivative_inverse(eta.nu),
            },
        )
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MuLink, ShapeLink, NuLink> for Zinb<MuLink, ShapeLink, NuLink>;
    parameters = (Mu, Shape, Nu);
    arity = 3;
);

impl<MuLink, ShapeLink, NuLink> Family for Zinb<MuLink, ShapeLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    type Eta = ZinbEta;
    type Theta = ZinbTheta;
    type GradientEta = ZinbEta;
    type Observation<'obs> = f64;
    type Workspace = ();
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

impl<MuLink, ShapeLink, NuLink> InitialEtaFromObservations<3> for Zinb<MuLink, ShapeLink, NuLink>
where
    MuLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    ShapeLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    NuLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| is_nonnegative_integer(y).then_some(y));
        let Some(summary) = weighted_summary(&values) else {
            return ZinbEta::from_array([0.0, 0.0, 0.0]);
        };
        let mu = positive_floor(summary.mean);
        let shape = if summary.variance <= mu {
            LARGE_SHAPE
        } else {
            positive_floor(mu * mu / (summary.variance - mu))
        };
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

        ZinbEta {
            mu: MuLink::initial_eta_from_theta(mu),
            shape: ShapeLink::initial_eta_from_theta(shape),
            nu: NuLink::initial_eta_from_theta(probability_floor(
                (zero_rate - (shape / (shape + mu)).powf(shape)).max(0.05),
            )),
        }
    }
}

impl<MuLink, ShapeLink, NuLink> HasCdf for Zinb<MuLink, ShapeLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        Self::cdf_theta(y, *theta)
    }
}

impl<MuLink, ShapeLink, NuLink> HasQuantile for Zinb<MuLink, ShapeLink, NuLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
    NuLink: UnitIntervalLink<f64>,
{
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        if !is_positive_finite(theta.mu)
            || !is_positive_finite(theta.shape)
            || !is_strict_probability(theta.nu)
        {
            return f64::NAN;
        }

        #[allow(clippy::cast_precision_loss)]
        discrete_quantile(p, MAX_CDF_TERMS, |count| {
            Self::cdf_theta(count as f64, *theta)
        })
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink, ShapeLink, NuLink> CanSimulate<Rng> for Zinb<MuLink, ShapeLink, NuLink>
where
    Rng: rand::Rng,
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
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

    #[cfg(feature = "rand")]
    use super::{ZinbMeanSizeZeroProbability, ZinbTheta};

    #[cfg(feature = "rand")]
    #[test]
    fn zinb_sampling_returns_counts_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = ZinbMeanSizeZeroProbability::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family.sample(
            &mut rng,
            &ZinbTheta {
                mu: 2.0,
                shape: 1.5,
                nu: 0.3,
            },
        );
        assert!(sample >= 0.0 && sample.fract() == 0.0);
        assert!(
            family
                .sample(
                    &mut rng,
                    &ZinbTheta {
                        mu: 2.0,
                        shape: 0.0,
                        nu: 0.3,
                    }
                )
                .is_nan()
        );
    }
}
