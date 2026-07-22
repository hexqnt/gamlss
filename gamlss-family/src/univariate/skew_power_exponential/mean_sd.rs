use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasQuantile, Identity, InitialEtaFromObservations, InitialEtaFromTheta, Link,
    Log, Mean, ObservationView, ParameterParts, PositiveLink, Power, Sigma, SkewRatio,
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

use crate::initial::{robust_location_scale, weighted_values};

#[cfg(feature = "rand")]
use super::try_sample_mode_scale;
use super::{
    cdf_mode_scale, mean_sd_to_mode_scale, nll_and_gradient_mean_sd, nll_mean_sd,
    quantile_mode_scale,
};

/// Mean/SD two-piece skew power-exponential with identity/log/log/log links.
pub type SkewPowerExponentialMeanSdSkewPower = SkewPowerExponentialMeanSd<Identity, Log, Log, Log>;

/// Fernández--Steel two-piece skew power-exponential parameterized by mathematical mean and standard deviation.
///
/// The positive `skew_ratio` and `power` have the same meaning as in [`super::SkewPowerExponential`], but every change to either shape parameter is accompanied by an analytic recentering and rescaling so that `mean` and `sigma` remain the response mean and standard deviation. This is the same density family, not an additional distribution or a special case, and is provided for interpretable distributional-regression predictors familiar from standardized SGED/SEPD formulations. Prefer the direct mode/base-scale form when moment semantics are unnecessary on a performance-sensitive path.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!(
        "../../../doc-assets/distributions/skew_power_exponential_mean_sd.svg"
    )
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkewPowerExponentialMeanSd<
    MeanLink = Identity,
    SigmaLink = Log,
    SkewLink = Log,
    PowerLink = Log,
> {
    marker: PhantomData<(MeanLink, SigmaLink, SkewLink, PowerLink)>,
}

impl<MeanLink, SigmaLink, SkewLink, PowerLink>
    SkewPowerExponentialMeanSd<MeanLink, SigmaLink, SkewLink, PowerLink>
where
    MeanLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    SkewLink: PositiveLink<f64>,
    PowerLink: PositiveLink<f64>,
{
    /// Creates a stateless mean/SD two-piece skew power-exponential family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(eta: SkewPowerExponentialMeanSdEta) -> SkewPowerExponentialMeanSdTheta {
        SkewPowerExponentialMeanSdTheta {
            mean: MeanLink::inverse(eta.mean),
            sigma: SigmaLink::inverse(eta.sigma),
            skew_ratio: SkewLink::inverse(eta.skew_ratio),
            power: PowerLink::inverse(eta.power),
        }
    }

    #[inline]
    fn nll_and_gradient_eta_values(
        y: f64,
        eta: SkewPowerExponentialMeanSdEta,
    ) -> (f64, SkewPowerExponentialMeanSdEta) {
        let theta = Self::theta_from_eta(eta);
        let (nll, gradient) =
            nll_and_gradient_mean_sd(y, theta.mean, theta.sigma, theta.skew_ratio, theta.power);
        if !nll.is_finite() || !gradient.is_finite() {
            return (
                nll,
                SkewPowerExponentialMeanSdEta::from_array([f64::NAN; 4]),
            );
        }
        (
            nll,
            SkewPowerExponentialMeanSdEta {
                mean: gradient.location * MeanLink::derivative_inverse(eta.mean),
                sigma: gradient.scale * SigmaLink::derivative_inverse(eta.sigma),
                skew_ratio: gradient.log_skew_ratio * SkewLink::derivative_inverse(eta.skew_ratio)
                    / theta.skew_ratio,
                power: gradient.power * PowerLink::derivative_inverse(eta.power),
            },
        )
    }
}

impl<MeanLink, SigmaLink, SkewLink, PowerLink> Default
    for SkewPowerExponentialMeanSd<MeanLink, SigmaLink, SkewLink, PowerLink>
where
    MeanLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    SkewLink: PositiveLink<f64>,
    PowerLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MeanLink, SigmaLink, SkewLink, PowerLink> for SkewPowerExponentialMeanSd<MeanLink, SigmaLink, SkewLink, PowerLink>;
    parameters = (Mean, Sigma, SkewRatio, Power);
    arity = 4;
);

impl<MeanLink, SigmaLink, SkewLink, PowerLink> Family
    for SkewPowerExponentialMeanSd<MeanLink, SigmaLink, SkewLink, PowerLink>
where
    MeanLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    SkewLink: PositiveLink<f64>,
    PowerLink: PositiveLink<f64>,
{
    type Eta = SkewPowerExponentialMeanSdEta;
    type Theta = SkewPowerExponentialMeanSdTheta;
    type GradientEta = SkewPowerExponentialMeanSdEta;
    type Observation<'obs> = f64;
    type Workspace = ();

    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    #[inline]
    fn theta(&self, eta: &Self::Eta, _workspace: &mut ()) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    #[inline]
    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut ()) -> f64 {
        nll_mean_sd(y, theta.mean, theta.sigma, theta.skew_ratio, theta.power)
    }

    #[inline]
    fn nll_eta(&self, y: f64, eta: &Self::Eta, _workspace: &mut ()) -> f64 {
        let theta = Self::theta_from_eta(*eta);
        nll_mean_sd(y, theta.mean, theta.sigma, theta.skew_ratio, theta.power)
    }

    #[inline]
    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut (),
    ) -> (f64, Self::GradientEta) {
        Self::nll_and_gradient_eta_values(y, *eta)
    }
}

impl<MeanLink, SigmaLink, SkewLink, PowerLink> InitialEtaFromObservations<4>
    for SkewPowerExponentialMeanSd<MeanLink, SigmaLink, SkewLink, PowerLink>
where
    MeanLink: InitialEtaFromTheta<f64> + Link<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    SkewLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    PowerLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = f64> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| y.is_finite().then_some(y));
        let Some((mean, sigma)) = robust_location_scale(&values) else {
            return SkewPowerExponentialMeanSdEta {
                mean: MeanLink::initial_eta_from_theta(0.0),
                sigma: SigmaLink::initial_eta_from_theta(1.0),
                skew_ratio: SkewLink::initial_eta_from_theta(1.0),
                power: PowerLink::initial_eta_from_theta(2.0),
            };
        };
        SkewPowerExponentialMeanSdEta {
            mean: MeanLink::initial_eta_from_theta(mean),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
            skew_ratio: SkewLink::initial_eta_from_theta(1.0),
            power: PowerLink::initial_eta_from_theta(2.0),
        }
    }
}

impl<MeanLink, SigmaLink, SkewLink, PowerLink> HasCdf
    for SkewPowerExponentialMeanSd<MeanLink, SigmaLink, SkewLink, PowerLink>
where
    MeanLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    SkewLink: PositiveLink<f64>,
    PowerLink: PositiveLink<f64>,
{
    fn cdf(&self, y: f64, theta: &Self::Theta) -> f64 {
        let Some(geometry) =
            mean_sd_to_mode_scale(theta.mean, theta.sigma, theta.skew_ratio, theta.power)
        else {
            return f64::NAN;
        };
        cdf_mode_scale(
            y,
            geometry.mode,
            geometry.scale,
            theta.skew_ratio,
            theta.power,
        )
    }
}

impl<MeanLink, SigmaLink, SkewLink, PowerLink> HasQuantile
    for SkewPowerExponentialMeanSd<MeanLink, SigmaLink, SkewLink, PowerLink>
where
    MeanLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    SkewLink: PositiveLink<f64>,
    PowerLink: PositiveLink<f64>,
{
    fn quantile(&self, probability: f64, theta: &Self::Theta) -> f64 {
        let Some(geometry) =
            mean_sd_to_mode_scale(theta.mean, theta.sigma, theta.skew_ratio, theta.power)
        else {
            return f64::NAN;
        };
        quantile_mode_scale(
            probability,
            geometry.mode,
            geometry.scale,
            theta.skew_ratio,
            theta.power,
        )
    }
}

#[cfg(feature = "rand")]
impl<Rng, MeanLink, SigmaLink, SkewLink, PowerLink> TrySimulate<Rng>
    for SkewPowerExponentialMeanSd<MeanLink, SigmaLink, SkewLink, PowerLink>
where
    Rng: rand::Rng,
    MeanLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    SkewLink: PositiveLink<f64>,
    PowerLink: PositiveLink<f64>,
{
    type Sample = f64;

    fn try_sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Result<f64, SimulationError> {
        let Some(geometry) =
            mean_sd_to_mode_scale(theta.mean, theta.sigma, theta.skew_ratio, theta.power)
        else {
            return Err(SimulationError::InvalidParameters(
                "mean/SD two-piece power-exponential theta",
            ));
        };
        try_sample_mode_scale(
            rng,
            geometry.mode,
            geometry.scale,
            theta.skew_ratio,
            theta.power,
        )
    }
}

/// Link-scale predictors for the mean/SD two-piece power-exponential family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkewPowerExponentialMeanSdEta {
    /// Mathematical-mean predictor.
    pub mean: f64,
    /// Standard-deviation predictor.
    pub sigma: f64,
    /// Positive Fernández--Steel skew-ratio predictor.
    pub skew_ratio: f64,
    /// Positive tail-power predictor.
    pub power: f64,
}

impl ParameterParts<4> for SkewPowerExponentialMeanSdEta {
    #[inline]
    fn from_array(values: [f64; 4]) -> Self {
        Self {
            mean: values[0],
            sigma: values[1],
            skew_ratio: values[2],
            power: values[3],
        }
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mean,
            1 => self.sigma,
            2 => self.skew_ratio,
            3 => self.power,
            _ => unreachable!("mean/SD skew power-exponential eta only has indices 0 through 3"),
        }
    }
}

/// Natural-scale mean/SD two-piece power-exponential parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkewPowerExponentialMeanSdTheta {
    /// Mathematical mean.
    pub mean: f64,
    /// Positive mathematical standard deviation.
    pub sigma: f64,
    /// Positive Fernández--Steel skew ratio; one gives symmetry.
    pub skew_ratio: f64,
    /// Positive tail power; two gives split normal and one gives split Laplace.
    pub power: f64,
}

impl SkewPowerExponentialMeanSdTheta {
    /// Returns whether every parameter can be represented by the shared density kernel.
    #[inline]
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.mode_scale().is_some()
    }

    /// Returns the equivalent mode/base-scale parameters when they are representable.
    #[inline]
    #[must_use]
    pub fn mode_scale(self) -> Option<super::SkewPowerExponentialTheta> {
        let geometry = mean_sd_to_mode_scale(self.mean, self.sigma, self.skew_ratio, self.power)?;
        Some(super::SkewPowerExponentialTheta {
            mu: geometry.mode,
            sigma: geometry.scale,
            skew_ratio: self.skew_ratio,
            power: self.power,
        })
    }
}

#[cfg(test)]
mod tests {
    use gamlss_core::Family;

    use super::SkewPowerExponentialMeanSdEta;
    use super::SkewPowerExponentialMeanSdSkewPower;
    use crate::test_support::assert_gradient_matches_finite_difference_with_tolerance;

    #[test]
    fn analytic_gradient_matches_finite_difference() {
        let family = SkewPowerExponentialMeanSdSkewPower::new();
        assert_gradient_matches_finite_difference_with_tolerance::<_, 4>(
            &family,
            1.2,
            [0.2, 0.3, 1.7_f64.ln(), 1.3_f64.ln()],
            2.0e-5,
            2.0e-5,
        );
        assert_gradient_matches_finite_difference_with_tolerance::<_, 4>(
            &family,
            -1.0,
            [0.2, 0.3, 1.7_f64.ln(), 1.3_f64.ln()],
            2.0e-5,
            2.0e-5,
        );
    }

    #[test]
    fn standardized_form_handles_extreme_representable_skew_ratios() {
        let family = SkewPowerExponentialMeanSdSkewPower::new();
        for skew_ratio in [-700.0, 700.0] {
            let eta = SkewPowerExponentialMeanSdEta {
                mean: 0.0,
                sigma: 0.0,
                skew_ratio,
                power: 2.0_f64.ln(),
            };
            let (nll, gradient) = family.nll_and_gradient_eta(0.0, &eta, &mut ());
            assert!(nll.is_finite());
            assert!(gradient.mean.is_finite());
            assert!(gradient.sigma.is_finite());
            assert!(gradient.skew_ratio.is_finite());
            assert!(gradient.power.is_finite());
        }
    }
}
