use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasQuantile, Identity, InitialEtaFromObservations, InitialEtaFromTheta, Link,
    Log, Mu, ObservationView, ParameterParts, PositiveLink, Power, Sigma, SkewRatio,
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

use crate::initial::{robust_location_scale, weighted_values};

#[cfg(feature = "rand")]
use super::try_sample_mode_scale;
use super::{
    cdf_mode_scale, mode_scale_to_mean_sd, nll_and_gradient_mode_scale, nll_mode_scale,
    quantile_mode_scale, valid_mode_scale,
};

/// Two-piece skew power-exponential with identity/log/log/log links.
pub type SkewPowerExponentialMuSigmaSkewPower = SkewPowerExponential<Identity, Log, Log, Log>;

/// Fernández--Steel two-piece skew power-exponential parameterized by mode and base scale.
///
/// `mu` is the mode and join point of the two density pieces. `sigma` is the scale of the symmetric power-exponential kernel before skewing; except when `skew_ratio = 1`, it is not the response standard deviation. Use [`super::SkewPowerExponentialMeanSd`] when regression predictors should target the mathematical mean and standard deviation.
///
/// `skew_ratio = 1` recovers [`crate::PowerExponential`] with the same `mu`, `sigma`, and `power`. Values above one stretch the right half of the density; reciprocal ratios produce reflected densities.
///
/// At an observation exactly equal to `mu`, the location score is defined as zero. This is the ordinary derivative for `power > 1` and an explicit optimization convention at the non-smooth join for `power <= 1`.
///
/// This is the direct kernel parameterization and avoids the moment conversion required by [`super::SkewPowerExponentialMeanSd`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkewPowerExponential<MuLink = Identity, SigmaLink = Log, SkewLink = Log, PowerLink = Log>
{
    marker: PhantomData<(MuLink, SigmaLink, SkewLink, PowerLink)>,
}

impl<MuLink, SigmaLink, SkewLink, PowerLink>
    SkewPowerExponential<MuLink, SigmaLink, SkewLink, PowerLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    SkewLink: PositiveLink<f64>,
    PowerLink: PositiveLink<f64>,
{
    /// Creates a stateless two-piece skew power-exponential family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(eta: SkewPowerExponentialEta) -> SkewPowerExponentialTheta {
        SkewPowerExponentialTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
            skew_ratio: SkewLink::inverse(eta.skew_ratio),
            power: PowerLink::inverse(eta.power),
        }
    }

    #[inline]
    fn nll_and_gradient_eta_values(
        y: f64,
        eta: SkewPowerExponentialEta,
    ) -> (f64, SkewPowerExponentialEta) {
        let theta = Self::theta_from_eta(eta);
        let (nll, gradient) =
            nll_and_gradient_mode_scale(y, theta.mu, theta.sigma, theta.skew_ratio, theta.power);
        if !nll.is_finite() || !gradient.is_finite() {
            return (nll, SkewPowerExponentialEta::from_array([f64::NAN; 4]));
        }
        (
            nll,
            SkewPowerExponentialEta {
                mu: gradient.location * MuLink::derivative_inverse(eta.mu),
                sigma: gradient.scale * SigmaLink::derivative_inverse(eta.sigma),
                skew_ratio: gradient.log_skew_ratio * SkewLink::derivative_inverse(eta.skew_ratio)
                    / theta.skew_ratio,
                power: gradient.power * PowerLink::derivative_inverse(eta.power),
            },
        )
    }
}

impl<MuLink, SigmaLink, SkewLink, PowerLink> Default
    for SkewPowerExponential<MuLink, SigmaLink, SkewLink, PowerLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    SkewLink: PositiveLink<f64>,
    PowerLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MuLink, SigmaLink, SkewLink, PowerLink> for SkewPowerExponential<MuLink, SigmaLink, SkewLink, PowerLink>;
    parameters = (Mu, Sigma, SkewRatio, Power);
    arity = 4;
);

impl<MuLink, SigmaLink, SkewLink, PowerLink> Family
    for SkewPowerExponential<MuLink, SigmaLink, SkewLink, PowerLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    SkewLink: PositiveLink<f64>,
    PowerLink: PositiveLink<f64>,
{
    type Eta = SkewPowerExponentialEta;
    type Theta = SkewPowerExponentialTheta;
    type GradientEta = SkewPowerExponentialEta;
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
        nll_mode_scale(y, theta.mu, theta.sigma, theta.skew_ratio, theta.power)
    }

    #[inline]
    fn nll_eta(&self, y: f64, eta: &Self::Eta, _workspace: &mut ()) -> f64 {
        let theta = Self::theta_from_eta(*eta);
        nll_mode_scale(y, theta.mu, theta.sigma, theta.skew_ratio, theta.power)
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

impl<MuLink, SigmaLink, SkewLink, PowerLink> InitialEtaFromObservations<4>
    for SkewPowerExponential<MuLink, SigmaLink, SkewLink, PowerLink>
where
    MuLink: InitialEtaFromTheta<f64> + Link<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    SkewLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    PowerLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = f64> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| y.is_finite().then_some(y));
        let Some((mu, sigma)) = robust_location_scale(&values) else {
            return SkewPowerExponentialEta {
                mu: MuLink::initial_eta_from_theta(0.0),
                sigma: SigmaLink::initial_eta_from_theta(1.0),
                skew_ratio: SkewLink::initial_eta_from_theta(1.0),
                power: PowerLink::initial_eta_from_theta(2.0),
            };
        };
        SkewPowerExponentialEta {
            mu: MuLink::initial_eta_from_theta(mu),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
            skew_ratio: SkewLink::initial_eta_from_theta(1.0),
            power: PowerLink::initial_eta_from_theta(2.0),
        }
    }
}

impl<MuLink, SigmaLink, SkewLink, PowerLink> HasCdf
    for SkewPowerExponential<MuLink, SigmaLink, SkewLink, PowerLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    SkewLink: PositiveLink<f64>,
    PowerLink: PositiveLink<f64>,
{
    fn cdf(&self, y: f64, theta: &Self::Theta) -> f64 {
        cdf_mode_scale(y, theta.mu, theta.sigma, theta.skew_ratio, theta.power)
    }
}

impl<MuLink, SigmaLink, SkewLink, PowerLink> HasQuantile
    for SkewPowerExponential<MuLink, SigmaLink, SkewLink, PowerLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    SkewLink: PositiveLink<f64>,
    PowerLink: PositiveLink<f64>,
{
    fn quantile(&self, probability: f64, theta: &Self::Theta) -> f64 {
        quantile_mode_scale(
            probability,
            theta.mu,
            theta.sigma,
            theta.skew_ratio,
            theta.power,
        )
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink, SigmaLink, SkewLink, PowerLink> TrySimulate<Rng>
    for SkewPowerExponential<MuLink, SigmaLink, SkewLink, PowerLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    SkewLink: PositiveLink<f64>,
    PowerLink: PositiveLink<f64>,
{
    type Sample = f64;

    fn try_sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Result<f64, SimulationError> {
        try_sample_mode_scale(rng, theta.mu, theta.sigma, theta.skew_ratio, theta.power)
    }
}

/// Link-scale predictors for the mode/base-scale two-piece power-exponential family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkewPowerExponentialEta {
    /// Mode predictor.
    pub mu: f64,
    /// Base-scale predictor.
    pub sigma: f64,
    /// Positive Fernández--Steel skew-ratio predictor.
    pub skew_ratio: f64,
    /// Positive tail-power predictor.
    pub power: f64,
}

impl ParameterParts<4> for SkewPowerExponentialEta {
    #[inline]
    fn from_array(values: [f64; 4]) -> Self {
        Self {
            mu: values[0],
            sigma: values[1],
            skew_ratio: values[2],
            power: values[3],
        }
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mu,
            1 => self.sigma,
            2 => self.skew_ratio,
            3 => self.power,
            _ => unreachable!("skew power-exponential eta only has indices 0 through 3"),
        }
    }
}

/// Natural-scale mode/base-scale two-piece power-exponential parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkewPowerExponentialTheta {
    /// Mode and join point of the two pieces.
    pub mu: f64,
    /// Positive base scale; this is the SD only when `skew_ratio = 1`.
    pub sigma: f64,
    /// Positive Fernández--Steel skew ratio; one gives symmetry.
    pub skew_ratio: f64,
    /// Positive tail power; two gives split normal and one gives split Laplace.
    pub power: f64,
}

impl SkewPowerExponentialTheta {
    /// Returns whether every parameter satisfies the family domain.
    #[inline]
    #[must_use]
    pub fn is_valid(self) -> bool {
        valid_mode_scale(self.mu, self.sigma, self.skew_ratio, self.power)
    }

    /// Returns the equivalent mathematical mean/SD parameters when they are representable.
    #[inline]
    #[must_use]
    pub fn mean_sd(self) -> Option<super::SkewPowerExponentialMeanSdTheta> {
        let (mean, sigma) =
            mode_scale_to_mean_sd(self.mu, self.sigma, self.skew_ratio, self.power)?;
        Some(super::SkewPowerExponentialMeanSdTheta {
            mean,
            sigma,
            skew_ratio: self.skew_ratio,
            power: self.power,
        })
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use gamlss_core::Family;

    use super::{SkewPowerExponentialEta, SkewPowerExponentialMuSigmaSkewPower};
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn analytic_gradient_matches_finite_difference_on_both_sides() {
        let family = SkewPowerExponentialMuSigmaSkewPower::new();
        assert_gradient_matches_finite_difference::<_, 4>(
            &family,
            -1.1,
            [0.2, 0.3, 1.7_f64.ln(), 1.3_f64.ln()],
        );
        assert_gradient_matches_finite_difference::<_, 4>(
            &family,
            1.4,
            [0.2, 0.3, 1.7_f64.ln(), 1.3_f64.ln()],
        );
    }

    #[test]
    fn center_uses_documented_zero_location_score() {
        let family = SkewPowerExponentialMuSigmaSkewPower::new();
        let eta = SkewPowerExponentialEta {
            mu: 0.3,
            sigma: 0.2,
            skew_ratio: 1.6_f64.ln(),
            power: 0.8_f64.ln(),
        };
        let (_, gradient) = family.nll_and_gradient_eta(0.3, &eta, &mut ());
        assert_abs_diff_eq!(gradient.mu, 0.0, epsilon = 0.0);
        assert!(gradient.sigma.is_finite());
        assert!(gradient.skew_ratio.is_finite());
        assert!(gradient.power.is_finite());
    }

    #[test]
    fn log_skew_score_stays_finite_for_extreme_representable_ratios() {
        let family = SkewPowerExponentialMuSigmaSkewPower::new();
        for skew_ratio in [-700.0, 700.0] {
            let eta = SkewPowerExponentialEta {
                mu: 0.0,
                sigma: 0.0,
                skew_ratio,
                power: 2.0_f64.ln(),
            };
            let (nll, gradient) = family.nll_and_gradient_eta(0.0, &eta, &mut ());
            assert!(nll.is_finite());
            assert!(gradient.mu.is_finite());
            assert!(gradient.sigma.is_finite());
            assert!(gradient.skew_ratio.is_finite());
            assert!(gradient.power.is_finite());
        }
    }
}
