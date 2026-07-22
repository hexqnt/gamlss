use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasCrps, HasQuantile, Identity, InitialEtaFromObservations,
    InitialEtaFromTheta, Link, Log, Mean, Nu, ObservationView, ParameterParts, PositiveLink, Sigma,
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

use crate::initial::{robust_location_scale, weighted_values};

use super::mu_sigma_nu::SkewNormalTheta;
use super::{
    SQRT_2_OVER_PI, cdf_location_scale, crps_location_scale, mean_sd_to_location_scale,
    nll_gradient_location_scale, nll_location_scale, quantile_location_scale,
};

/// Skew-normal distribution parameterized by mean, standard deviation and skewness.
pub type SkewNormalMeanSdNu = SkewNormalMeanSd<Identity, Log, Identity>;

/// Azzalini/SN1-style skew-normal family parameterized by mean and standard deviation.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../../doc-assets/distributions/skew_normal_mean_sd.svg")
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkewNormalMeanSd<MeanLink = Identity, SigmaLink = Log, NuLink = Identity> {
    marker: PhantomData<(MeanLink, SigmaLink, NuLink)>,
}

impl<MeanLink, SigmaLink, NuLink> SkewNormalMeanSd<MeanLink, SigmaLink, NuLink>
where
    MeanLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    /// Creates a stateless mean/SD skew-normal family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(eta: SkewNormalMeanSdEta) -> SkewNormalMeanSdTheta {
        SkewNormalMeanSdTheta {
            mean: MeanLink::inverse(eta.mean),
            sigma: SigmaLink::inverse(eta.sigma),
            nu: NuLink::inverse(eta.nu),
        }
    }

    #[inline]
    fn nll_theta(y: f64, theta: SkewNormalMeanSdTheta) -> f64 {
        let Some(location_scale) = theta.location_scale() else {
            return f64::INFINITY;
        };

        nll_location_scale(
            y,
            location_scale.mu,
            location_scale.sigma,
            location_scale.nu,
        )
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn nll_and_gradient_eta_values(y: f64, eta: SkewNormalMeanSdEta) -> (f64, SkewNormalMeanSdEta) {
        let theta = Self::theta_from_eta(eta);
        let Some(location_scale) = theta.location_scale() else {
            return (
                f64::INFINITY,
                SkewNormalMeanSdEta::from_array([f64::NAN; 3]),
            );
        };
        let nll = nll_location_scale(
            y,
            location_scale.mu,
            location_scale.sigma,
            location_scale.nu,
        );
        if !nll.is_finite() {
            return (nll, SkewNormalMeanSdEta::from_array([f64::NAN; 3]));
        }

        let gradient =
            nll_gradient_location_scale(y, location_scale.mu, location_scale.sigma, theta.nu);
        let hypot = theta.nu.hypot(1.0);
        let inverse_hypot = 1.0 / hypot;
        let delta_derivative = inverse_hypot * inverse_hypot * inverse_hypot;
        let standardized_mean = SQRT_2_OVER_PI * theta.nu / hypot;
        let standardized_mean_derivative = SQRT_2_OVER_PI * delta_derivative;
        let standardized_variance = 1.0 - standardized_mean * standardized_mean;
        let scale_per_sd = location_scale.sigma / theta.sigma;
        let scale_per_nu = location_scale.sigma * standardized_mean * standardized_mean_derivative
            / standardized_variance;
        let location_per_sd = -standardized_mean * scale_per_sd;
        let location_per_nu =
            -standardized_mean * scale_per_nu - location_scale.sigma * standardized_mean_derivative;

        let d_mean = gradient.mu;
        let d_sigma = gradient.mu * location_per_sd + gradient.sigma * scale_per_sd;
        let d_nu = gradient.mu * location_per_nu + gradient.sigma * scale_per_nu + gradient.nu;

        (
            nll,
            SkewNormalMeanSdEta {
                mean: d_mean * MeanLink::derivative_inverse(eta.mean),
                sigma: d_sigma * SigmaLink::derivative_inverse(eta.sigma),
                nu: d_nu * NuLink::derivative_inverse(eta.nu),
            },
        )
    }
}

impl<MeanLink, SigmaLink, NuLink> Default for SkewNormalMeanSd<MeanLink, SigmaLink, NuLink>
where
    MeanLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MeanLink, SigmaLink, NuLink> for SkewNormalMeanSd<MeanLink, SigmaLink, NuLink>;
    parameters = (Mean, Sigma, Nu);
    arity = 3;
);

impl<MeanLink, SigmaLink, NuLink> Family for SkewNormalMeanSd<MeanLink, SigmaLink, NuLink>
where
    MeanLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    type Eta = SkewNormalMeanSdEta;
    type Theta = SkewNormalMeanSdTheta;
    type GradientEta = SkewNormalMeanSdEta;
    type Observation<'obs> = f64;
    type Workspace = ();

    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    #[inline]
    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        Self::nll_theta(y, *theta)
    }

    #[inline]
    fn nll_eta(&self, y: f64, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> f64 {
        Self::nll_theta(y, Self::theta_from_eta(*eta))
    }

    #[inline]
    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        Self::nll_and_gradient_eta_values(y, *eta)
    }
}

impl<MeanLink, SigmaLink, NuLink> InitialEtaFromObservations<3>
    for SkewNormalMeanSd<MeanLink, SigmaLink, NuLink>
where
    MeanLink: InitialEtaFromTheta<f64> + Link<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    NuLink: InitialEtaFromTheta<f64> + Link<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| y.is_finite().then_some(y));
        let Some((mean, sigma)) = robust_location_scale(&values) else {
            return SkewNormalMeanSdEta::from_array([0.0, 0.0, 0.0]);
        };

        SkewNormalMeanSdEta {
            mean: MeanLink::initial_eta_from_theta(mean),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
            nu: NuLink::initial_eta_from_theta(0.0),
        }
    }
}

impl<MeanLink, SigmaLink, NuLink> HasCdf for SkewNormalMeanSd<MeanLink, SigmaLink, NuLink>
where
    MeanLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        let Some(location_scale) = theta.location_scale() else {
            return f64::NAN;
        };

        cdf_location_scale(
            y,
            location_scale.mu,
            location_scale.sigma,
            location_scale.nu,
        )
    }
}

impl<MeanLink, SigmaLink, NuLink> HasQuantile for SkewNormalMeanSd<MeanLink, SigmaLink, NuLink>
where
    MeanLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        let Some(location_scale) = theta.location_scale() else {
            return f64::NAN;
        };

        quantile_location_scale(
            p,
            location_scale.mu,
            location_scale.sigma,
            location_scale.nu,
        )
    }
}

impl<MeanLink, SigmaLink, NuLink> HasCrps for SkewNormalMeanSd<MeanLink, SigmaLink, NuLink>
where
    MeanLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    fn crps(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        let Some(location_scale) = theta.location_scale() else {
            return f64::NAN;
        };

        crps_location_scale(
            y,
            location_scale.mu,
            location_scale.sigma,
            location_scale.nu,
        )
    }
}

#[cfg(feature = "rand")]
impl<Rng, MeanLink, SigmaLink, NuLink> TrySimulate<Rng>
    for SkewNormalMeanSd<MeanLink, SigmaLink, NuLink>
where
    Rng: rand::Rng,
    MeanLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    type Sample = f64;

    fn try_sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Result<f64, SimulationError> {
        let Some(location_scale) = theta.location_scale() else {
            return Err(SimulationError::InvalidParameters(
                "mean/SD skew-normal theta",
            ));
        };

        super::try_sample_location_scale(
            rng,
            location_scale.mu,
            location_scale.sigma,
            location_scale.nu,
        )
    }
}

/// Predictors for mean/SD skew-normal on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkewNormalMeanSdEta {
    /// Mean predictor.
    pub mean: f64,
    /// Standard-deviation predictor.
    pub sigma: f64,
    /// Skewness predictor.
    pub nu: f64,
}

impl ParameterParts<3> for SkewNormalMeanSdEta {
    #[inline]
    fn from_array(values: [f64; 3]) -> Self {
        Self {
            mean: values[0],
            sigma: values[1],
            nu: values[2],
        }
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mean,
            1 => self.sigma,
            2 => self.nu,
            _ => unreachable!("mean/SD skew-normal eta only has indices 0 through 2"),
        }
    }
}

/// Natural-scale mean/SD skew-normal parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkewNormalMeanSdTheta {
    /// Mathematical mean.
    pub mean: f64,
    /// Positive standard deviation.
    pub sigma: f64,
    /// Skewness parameter.
    pub nu: f64,
}

impl SkewNormalMeanSdTheta {
    #[inline]
    fn location_scale(self) -> Option<SkewNormalTheta> {
        let (mu, sigma) = mean_sd_to_location_scale(self.mean, self.sigma, self.nu)?;
        Some(SkewNormalTheta {
            mu,
            sigma,
            nu: self.nu,
        })
    }
}

#[cfg(test)]
mod tests {
    use gamlss_core::Family;
    #[cfg(feature = "rand")]
    use gamlss_core::TrySimulate;

    #[cfg(feature = "rand")]
    use super::SkewNormalMeanSdTheta;
    use super::{SkewNormalMeanSdEta, SkewNormalMeanSdNu};

    #[test]
    fn extreme_finite_skewness_keeps_mean_sd_gradient_consistent() {
        let family = SkewNormalMeanSdNu::new();
        let eta = SkewNormalMeanSdEta {
            mean: 0.0,
            sigma: 0.0,
            nu: 1.0e200,
        };
        let (nll, gradient) = family.nll_and_gradient_eta(0.0, &eta, &mut ());

        assert!(nll.is_finite());
        assert!(
            (gradient.sigma - 1.0).abs() < 1.0e-14,
            "sigma gradient was {}",
            gradient.sigma
        );
        assert!(gradient.mean.is_finite());
        assert!(gradient.nu.is_finite());
    }

    #[cfg(feature = "rand")]
    #[test]
    fn mean_sd_skew_normal_sampling_returns_finite_values_and_errors_for_invalid_theta() {
        use rand::SeedableRng;

        let family = SkewNormalMeanSdNu::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        assert!(
            family
                .try_sample(
                    &mut rng,
                    &SkewNormalMeanSdTheta {
                        mean: 0.4,
                        sigma: 1.5,
                        nu: 2.0,
                    }
                )
                .is_ok_and(f64::is_finite)
        );
        assert!(
            family
                .try_sample(
                    &mut rng,
                    &SkewNormalMeanSdTheta {
                        mean: 0.4,
                        sigma: 0.0,
                        nu: 2.0,
                    }
                )
                .is_err()
        );
    }
}
