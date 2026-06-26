use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasQuantile, Identity, InitialEtaFromTheta, Link, Log, LogPlus, Mean, Nu,
    ObservationView, ParameterParts, ParameterizedFamily, PositiveLink, Sigma, Tau,
};

use crate::initial::{robust_location_scale, weighted_values};
use crate::numeric::finite_difference_gradient_eta;

use super::mu_sigma_nu_tau::SkewStudentTTheta;
use super::{
    cdf_location_scale, mean_sd_to_location_scale, nll_location_scale, quantile_location_scale,
};

/// Skew Student-t distribution parameterized by mean, standard deviation, skewness and `tau > 2`.
///
/// Its NLL gradient currently uses a finite-difference fallback and should be
/// treated as a training slow path until an analytic gradient is added.
pub type SkewStudentTMeanSdNuTau = SkewStudentTMeanSd<Identity, Log, Identity, LogPlus<2>>;

/// Azzalini/ST1-style skew Student-t family parameterized by mean and standard deviation.
///
/// Its NLL gradient currently uses a finite-difference fallback and should be
/// treated as a training slow path until an analytic gradient is added.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkewStudentTMeanSd<
    MeanLink = Identity,
    SigmaLink = Log,
    NuLink = Identity,
    TauLink = LogPlus<2>,
> {
    marker: PhantomData<(MeanLink, SigmaLink, NuLink, TauLink)>,
}

impl<MeanLink, SigmaLink, NuLink, TauLink> SkewStudentTMeanSd<MeanLink, SigmaLink, NuLink, TauLink>
where
    MeanLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
    TauLink: Link<f64>,
{
    /// Creates a stateless mean/SD skew Student-t family.
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn theta_from_eta(eta: SkewStudentTMeanSdEta) -> SkewStudentTMeanSdTheta {
        SkewStudentTMeanSdTheta {
            mean: MeanLink::inverse(eta.mean),
            sigma: SigmaLink::inverse(eta.sigma),
            nu: NuLink::inverse(eta.nu),
            tau: TauLink::inverse(eta.tau),
        }
    }

    #[inline(always)]
    fn nll_theta(y: f64, theta: SkewStudentTMeanSdTheta) -> f64 {
        let Some(location_scale) = theta.location_scale() else {
            return f64::INFINITY;
        };

        nll_location_scale(
            y,
            location_scale.mu,
            location_scale.sigma,
            location_scale.nu,
            location_scale.tau,
        )
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(
        y: f64,
        eta: SkewStudentTMeanSdEta,
    ) -> (f64, SkewStudentTMeanSdEta) {
        let nll = Self::nll_theta(y, Self::theta_from_eta(eta));
        if !nll.is_finite() {
            return (nll, SkewStudentTMeanSdEta::from_array([f64::NAN; 4]));
        }

        let gradient =
            finite_difference_gradient_eta::<_, SkewStudentTMeanSdEta, 4>(eta, |probe| {
                Self::nll_theta(y, Self::theta_from_eta(probe))
            });
        (nll, SkewStudentTMeanSdEta::from_array(gradient))
    }
}

impl<MeanLink, SigmaLink, NuLink, TauLink> Default
    for SkewStudentTMeanSd<MeanLink, SigmaLink, NuLink, TauLink>
where
    MeanLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
    TauLink: Link<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<MeanLink, SigmaLink, NuLink, TauLink> Family
    for SkewStudentTMeanSd<MeanLink, SigmaLink, NuLink, TauLink>
where
    MeanLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
    TauLink: Link<f64>,
{
    type Eta = SkewStudentTMeanSdEta;
    type Theta = SkewStudentTMeanSdTheta;
    type NllGradientEta = SkewStudentTMeanSdEta;
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

impl<MeanLink, SigmaLink, NuLink, TauLink> ParameterizedFamily<4>
    for SkewStudentTMeanSd<MeanLink, SigmaLink, NuLink, TauLink>
where
    MeanLink: InitialEtaFromTheta<f64> + Link<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    NuLink: InitialEtaFromTheta<f64> + Link<f64>,
    TauLink: InitialEtaFromTheta<f64> + Link<f64>,
{
    type Params = (Mean, Sigma, Nu, Tau);
    type Links = (MeanLink, SigmaLink, NuLink, TauLink);

    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| y.is_finite().then_some(y));
        let Some((mean, sigma)) = robust_location_scale(&values) else {
            return SkewStudentTMeanSdEta {
                mean: MeanLink::initial_eta_from_theta(0.0),
                sigma: SigmaLink::initial_eta_from_theta(1.0),
                nu: NuLink::initial_eta_from_theta(0.0),
                tau: TauLink::initial_eta_from_theta(5.0),
            };
        };

        SkewStudentTMeanSdEta {
            mean: MeanLink::initial_eta_from_theta(mean),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
            nu: NuLink::initial_eta_from_theta(0.0),
            tau: TauLink::initial_eta_from_theta(5.0),
        }
    }
}

impl<MeanLink, SigmaLink, NuLink, TauLink> HasCdf
    for SkewStudentTMeanSd<MeanLink, SigmaLink, NuLink, TauLink>
where
    MeanLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
    TauLink: Link<f64>,
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
            location_scale.tau,
        )
    }
}

impl<MeanLink, SigmaLink, NuLink, TauLink> HasQuantile
    for SkewStudentTMeanSd<MeanLink, SigmaLink, NuLink, TauLink>
where
    MeanLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: Link<f64>,
    TauLink: Link<f64>,
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
            location_scale.tau,
        )
    }
}

/// Predictors for mean/SD skew Student-t on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkewStudentTMeanSdEta {
    /// Mean predictor.
    pub mean: f64,
    /// Standard-deviation predictor.
    pub sigma: f64,
    /// Skewness predictor.
    pub nu: f64,
    /// Degrees-of-freedom predictor.
    pub tau: f64,
}

impl ParameterParts<4> for SkewStudentTMeanSdEta {
    #[inline(always)]
    fn from_array(values: [f64; 4]) -> Self {
        Self {
            mean: values[0],
            sigma: values[1],
            nu: values[2],
            tau: values[3],
        }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mean,
            1 => self.sigma,
            2 => self.nu,
            3 => self.tau,
            _ => unreachable!("mean/SD skew student-t eta only has indices 0 through 3"),
        }
    }
}

/// Natural-scale mean/SD skew Student-t parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkewStudentTMeanSdTheta {
    /// Mathematical mean.
    pub mean: f64,
    /// Positive standard deviation.
    pub sigma: f64,
    /// Skewness parameter.
    pub nu: f64,
    /// Degrees of freedom, greater than two.
    pub tau: f64,
}

impl SkewStudentTMeanSdTheta {
    #[inline(always)]
    fn location_scale(self) -> Option<SkewStudentTTheta> {
        let (mu, sigma) = mean_sd_to_location_scale(self.mean, self.sigma, self.nu, self.tau)?;
        Some(SkewStudentTTheta {
            mu,
            sigma,
            nu: self.nu,
            tau: self.tau,
        })
    }
}
