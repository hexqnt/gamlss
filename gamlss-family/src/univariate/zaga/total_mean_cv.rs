use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Cv, Family, HasCdf, HasQuantile, InitialEtaFromObservations, InitialEtaFromTheta, Log, Logit,
    ObservationView, ParameterParts, PositiveLink, ScalarParams, TotalMean, UnitIntervalLink,
    ZeroProbability,
};

use gamlss_special::{invert_positive_cdf, regularized_gamma_lower};

use crate::initial::{positive_floor, probability_floor, weighted_summary, weighted_values};

use super::{Zaga, ZagaTheta};

/// ZAGA distribution parameterized by total mean, component CV, and zero probability.
pub type ZagaTotalMeanCvZeroProbability = ZagaTotalMeanCv<Log, Log, Logit>;

/// Predictors for ZAGA total-mean/CV/zero-probability on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZagaTotalMeanCvZeroProbabilityEta {
    /// Total mean predictor.
    pub total_mean: f64,
    /// Gamma component coefficient-of-variation predictor.
    pub cv: f64,
    /// Zero-mass probability predictor.
    pub zero_probability: f64,
}

impl ParameterParts<3> for ZagaTotalMeanCvZeroProbabilityEta {
    #[inline]
    fn from_array(values: [f64; 3]) -> Self {
        Self {
            total_mean: values[0],
            cv: values[1],
            zero_probability: values[2],
        }
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.total_mean,
            1 => self.cv,
            2 => self.zero_probability,
            _ => unreachable!("ZAGA total-mean/CV eta only has indices 0 through 2"),
        }
    }
}

/// Natural-scale ZAGA total-mean/CV/zero-probability parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZagaTotalMeanCvZeroProbabilityTheta {
    /// Positive unconditional mean.
    pub total_mean: f64,
    /// Positive gamma component coefficient of variation.
    pub cv: f64,
    /// Zero-mass probability in `(0, 1)`.
    pub zero_probability: f64,
}

impl ZagaTotalMeanCvZeroProbabilityTheta {
    #[inline]
    fn component(self) -> ZagaTheta {
        ZagaTheta {
            mu: self.total_mean / (1.0 - self.zero_probability),
            sigma: self.cv,
            nu: self.zero_probability,
        }
    }
}

/// ZAGA total-mean/CV/zero-probability implementation carrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZagaTotalMeanCv<MeanLink = Log, CvLink = Log, ZeroProbabilityLink = Logit> {
    marker: PhantomData<(MeanLink, CvLink, ZeroProbabilityLink)>,
}

impl<MeanLink, CvLink, ZeroProbabilityLink> ZagaTotalMeanCv<MeanLink, CvLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    /// Creates a stateless ZAGA total-mean/CV family.
    #[inline]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(
        eta: ZagaTotalMeanCvZeroProbabilityEta,
    ) -> ZagaTotalMeanCvZeroProbabilityTheta {
        ZagaTotalMeanCvZeroProbabilityTheta {
            total_mean: MeanLink::inverse(eta.total_mean),
            cv: CvLink::inverse(eta.cv),
            zero_probability: ZeroProbabilityLink::inverse(eta.zero_probability),
        }
    }
}

impl<MeanLink, CvLink, ZeroProbabilityLink> Default
    for ZagaTotalMeanCv<MeanLink, CvLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<MeanLink, CvLink, ZeroProbabilityLink> Family
    for ZagaTotalMeanCv<MeanLink, CvLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    type Eta = ZagaTotalMeanCvZeroProbabilityEta;
    type Theta = ZagaTotalMeanCvZeroProbabilityTheta;
    type GradientEta = ZagaTotalMeanCvZeroProbabilityEta;
    type Observation<'obs> = f64;
    type Workspace = ();
    type ParamSpec =
        ScalarParams<(TotalMean, Cv, ZeroProbability), (MeanLink, CvLink, ZeroProbabilityLink), 3>;
    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        Zaga::<Log, Log, Logit>::nll_theta(y, theta.component())
    }

    fn nll_eta(&self, y: f64, eta: &Self::Eta, workspace: &mut Self::Workspace) -> f64 {
        let theta = Self::theta_from_eta(*eta);
        self.nll(y, &theta, workspace)
    }

    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        let theta = Self::theta_from_eta(*eta);
        let component = theta.component();
        let nll = Zaga::<Log, Log, Logit>::nll_theta(y, component);
        if !nll.is_finite() {
            return (
                nll,
                ZagaTotalMeanCvZeroProbabilityEta::from_array([f64::NAN; 3]),
            );
        }

        let component_gradient = Zaga::<Log, Log, Logit>::gradient_component_theta(y, component);
        let one_minus_zero = 1.0 - theta.zero_probability;
        let d_total_mean = component_gradient.mu / one_minus_zero;
        let d_zero_probability = component_gradient.mu * theta.total_mean
            / (one_minus_zero * one_minus_zero)
            + component_gradient.nu;
        (
            nll,
            ZagaTotalMeanCvZeroProbabilityEta {
                total_mean: d_total_mean * MeanLink::derivative_inverse(eta.total_mean),
                cv: component_gradient.sigma * CvLink::derivative_inverse(eta.cv),
                zero_probability: d_zero_probability
                    * ZeroProbabilityLink::derivative_inverse(eta.zero_probability),
            },
        )
    }
}

impl<MeanLink, CvLink, ZeroProbabilityLink> InitialEtaFromObservations<3>
    for ZagaTotalMeanCv<MeanLink, CvLink, ZeroProbabilityLink>
where
    MeanLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    CvLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    ZeroProbabilityLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
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
        let total_mean = positive_floor(
            weighted_summary(&values).map_or_else(|| summary.map_or(1.0, |s| s.mean), |s| s.mean),
        );
        let positive_mean = positive_floor(summary.map_or(total_mean, |s| s.mean));
        let cv = positive_floor(summary.map_or(1.0, |s| s.variance.sqrt() / positive_mean));
        let zero_weight = values
            .iter()
            .filter(|(y, _)| *y == 0.0)
            .map(|(_, w)| *w)
            .sum::<f64>();
        let total_weight = values.iter().map(|(_, w)| *w).sum::<f64>();
        let zero_probability = if total_weight > 0.0 {
            zero_weight / total_weight
        } else {
            0.1
        };

        ZagaTotalMeanCvZeroProbabilityEta {
            total_mean: MeanLink::initial_eta_from_theta(total_mean),
            cv: CvLink::initial_eta_from_theta(cv),
            zero_probability: ZeroProbabilityLink::initial_eta_from_theta(probability_floor(
                zero_probability,
            )),
        }
    }
}

impl<MeanLink, CvLink, ZeroProbabilityLink> HasCdf
    for ZagaTotalMeanCv<MeanLink, CvLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        Zaga::<Log, Log, Logit>::cdf_theta(y, theta.component())
    }
}

impl<MeanLink, CvLink, ZeroProbabilityLink> HasQuantile
    for ZagaTotalMeanCv<MeanLink, CvLink, ZeroProbabilityLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        if !(0.0..=1.0).contains(&p)
            || theta.zero_probability <= 0.0
            || theta.zero_probability >= 1.0
        {
            return f64::NAN;
        }
        let theta = theta.component();
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
impl<Rng, MeanLink, CvLink, ZeroProbabilityLink> CanSimulate<Rng>
    for ZagaTotalMeanCv<MeanLink, CvLink, ZeroProbabilityLink>
where
    Rng: rand::Rng,
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
    ZeroProbabilityLink: UnitIntervalLink<f64>,
{
    type Sample = f64;

    fn sample(&self, rng: &mut Rng, theta: &Self::Theta) -> f64 {
        Zaga::<Log, Log, Logit>::sample_component_theta(rng, theta.component())
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;

    use super::{ZagaTotalMeanCvZeroProbability, ZagaTotalMeanCvZeroProbabilityTheta};

    #[cfg(feature = "rand")]
    #[test]
    fn total_mean_zaga_sampling_returns_nonnegative_values_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = ZagaTotalMeanCvZeroProbability::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family.sample(
            &mut rng,
            &ZagaTotalMeanCvZeroProbabilityTheta {
                total_mean: 1.5,
                cv: 0.7,
                zero_probability: 0.2,
            },
        );
        assert!(sample >= 0.0 && sample.is_finite());
        assert!(
            family
                .sample(
                    &mut rng,
                    &ZagaTotalMeanCvZeroProbabilityTheta {
                        total_mean: 1.5,
                        cv: 0.0,
                        zero_probability: 0.2,
                    }
                )
                .is_nan()
        );
    }
}
