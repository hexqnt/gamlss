use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasQuantile, Identity, InitialEtaFromTheta, Link, Log, Mean, Nu,
    ObservationView, ParameterParts, ParameterizedFamily, PositiveLink, Sigma,
};

use crate::initial::{robust_location_scale, weighted_values};
use crate::numeric::finite_difference_gradient_eta;

use super::mu_sigma_nu::SkewNormalTheta;
use super::{
    cdf_location_scale, mean_sd_to_location_scale, nll_location_scale, quantile_location_scale,
};

/// Skew-normal distribution parameterized by mean, standard deviation and skewness.
pub type SkewNormalMeanSdNu = SkewNormalMeanSd<Identity, Log, Identity>;

/// Azzalini/SN1-style skew-normal family parameterized by mean and standard deviation.
#[derive(Debug, Clone, Copy, PartialEq)]
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
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn theta_from_eta(eta: SkewNormalMeanSdEta) -> SkewNormalMeanSdTheta {
        SkewNormalMeanSdTheta {
            mean: MeanLink::inverse(eta.mean),
            sigma: SigmaLink::inverse(eta.sigma),
            nu: NuLink::inverse(eta.nu),
        }
    }

    #[inline(always)]
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

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: SkewNormalMeanSdEta) -> (f64, SkewNormalMeanSdEta) {
        let nll = Self::nll_theta(y, Self::theta_from_eta(eta));
        if !nll.is_finite() {
            return (nll, SkewNormalMeanSdEta::from_array([f64::NAN; 3]));
        }

        let gradient = finite_difference_gradient_eta::<_, SkewNormalMeanSdEta, 3>(eta, |probe| {
            Self::nll_theta(y, Self::theta_from_eta(probe))
        });
        (nll, SkewNormalMeanSdEta::from_array(gradient))
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

impl<MeanLink, SigmaLink, NuLink> Family for SkewNormalMeanSd<MeanLink, SigmaLink, NuLink>
where
    MeanLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
{
    type Eta = SkewNormalMeanSdEta;
    type Theta = SkewNormalMeanSdTheta;
    type NllGradientEta = SkewNormalMeanSdEta;
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

impl<MeanLink, SigmaLink, NuLink> ParameterizedFamily<3>
    for SkewNormalMeanSd<MeanLink, SigmaLink, NuLink>
where
    MeanLink: InitialEtaFromTheta<f64> + Link<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    NuLink: InitialEtaFromTheta<f64> + Link<f64>,
{
    type Params = (Mean, Sigma, Nu);
    type Links = (MeanLink, SigmaLink, NuLink);

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
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
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
    fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
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
    #[inline(always)]
    fn from_array(values: [f64; 3]) -> Self {
        Self {
            mean: values[0],
            sigma: values[1],
            nu: values[2],
        }
    }

    #[inline(always)]
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
    #[inline(always)]
    fn location_scale(self) -> Option<SkewNormalTheta> {
        let (mu, sigma) = mean_sd_to_location_scale(self.mean, self.sigma, self.nu)?;
        Some(SkewNormalTheta {
            mu,
            sigma,
            nu: self.nu,
        })
    }
}
