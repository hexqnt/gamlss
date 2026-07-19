use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasQuantile, InitialEtaFromObservations, InitialEtaFromTheta, Log, Logit, Mu,
    Nu, ObservationView, ParameterParts, PositiveLink, Sigma, Tau, UnitIntervalLink,
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

use gamlss_special::{
    bernoulli_kl, digamma_minus_ln, invert_bounded_cdf, ln_gamma_stirling_residual,
    regularized_beta,
};

use crate::initial::{
    POSITIVE_FLOOR, VARIANCE_FLOOR, positive_floor, probability_floor, weighted_summary,
    weighted_values,
};

/// BEINF distribution with logit/logit/log/log links.
pub type BeinfMuSigmaNuTau = Beinf<Logit, Logit, Log, Log>;
/// Beta distribution inflated at both zero and one.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/distributions/beinf.svg")
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Beinf<MuLink = Logit, SigmaLink = Logit, NuLink = Log, TauLink = Log> {
    marker: PhantomData<(MuLink, SigmaLink, NuLink, TauLink)>,
}

impl<MuLink, SigmaLink, NuLink, TauLink> Beinf<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: UnitIntervalLink<f64>,
    SigmaLink: UnitIntervalLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    /// Creates a stateless BEINF family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(eta: BeinfEta) -> BeinfTheta {
        BeinfTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
            nu: NuLink::inverse(eta.nu),
            tau: TauLink::inverse(eta.tau),
        }
    }

    #[inline]
    fn parts(theta: BeinfTheta) -> Option<BeinfParts> {
        if theta.mu <= 0.0
            || theta.mu >= 1.0
            || !theta.mu.is_finite()
            || theta.sigma <= 0.0
            || theta.sigma >= 1.0
            || !theta.sigma.is_finite()
            || theta.nu <= 0.0
            || !theta.nu.is_finite()
            || theta.tau <= 0.0
            || !theta.tau.is_finite()
        {
            return None;
        }

        let precision = 1.0 / theta.sigma - 1.0;
        if precision <= 0.0 || !precision.is_finite() {
            return None;
        }
        let alpha = theta.mu * precision;
        let beta = (1.0 - theta.mu) * precision;
        if alpha <= 0.0 || !alpha.is_finite() || beta <= 0.0 || !beta.is_finite() {
            return None;
        }
        let scale = 1.0_f64.max(theta.nu).max(theta.tau);
        let scaled_beta = 1.0 / scale;
        let scaled_zero = theta.nu / scale;
        let scaled_one = theta.tau / scale;
        let scaled_denominator = scaled_beta + scaled_zero + scaled_one;
        let log_scaled_denominator = scaled_denominator.ln();
        let log_p0 = scaled_zero.ln() - log_scaled_denominator;
        let log_p_beta = scaled_beta.ln() - log_scaled_denominator;
        Some(BeinfParts {
            alpha,
            beta,
            precision,
            inverse_denominator: scaled_beta / scaled_denominator,
            log_p0,
            log_p1: scaled_one.ln() - log_scaled_denominator,
            log_p_beta,
            p0: log_p0.exp(),
            p_beta: log_p_beta.exp(),
        })
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn beta_log_density(y: f64, mean: f64, parts: BeinfParts) -> f64 {
        -(ln_gamma_stirling_residual(parts.alpha) + ln_gamma_stirling_residual(parts.beta)
            - ln_gamma_stirling_residual(parts.precision)
            + parts.precision * bernoulli_kl(mean, y)
            + y.ln()
            + (-y).ln_1p())
    }

    #[inline]
    fn log_ratio(numerator: f64, denominator: f64) -> f64 {
        let centered = (numerator - denominator) / denominator;
        if centered.abs() <= 0.5 {
            centered.ln_1p()
        } else {
            numerator.ln() - denominator.ln()
        }
    }

    #[inline]
    #[allow(clippy::float_cmp)]
    fn nll_theta(y: f64, theta: BeinfTheta) -> f64 {
        if !(0.0..=1.0).contains(&y) || !y.is_finite() {
            return f64::INFINITY;
        }
        let Some(parts) = Self::parts(theta) else {
            return f64::INFINITY;
        };
        if y == 0.0 {
            return -parts.log_p0;
        }
        if y == 1.0 {
            return -parts.log_p1;
        }

        -parts.log_p_beta - Self::beta_log_density(y, theta.mu, parts)
    }

    #[inline]
    #[allow(clippy::float_cmp, clippy::suboptimal_flops)]
    fn gradient_theta(y: f64, theta: BeinfTheta, parts: BeinfParts) -> BeinfTheta {
        let d_log_denominator = parts.inverse_denominator;

        if y == 0.0 {
            return BeinfTheta {
                mu: 0.0,
                sigma: 0.0,
                nu: d_log_denominator - 1.0 / theta.nu,
                tau: d_log_denominator,
            };
        }
        if y == 1.0 {
            return BeinfTheta {
                mu: 0.0,
                sigma: 0.0,
                nu: d_log_denominator,
                tau: d_log_denominator - 1.0 / theta.tau,
            };
        }

        let precision_residual = digamma_minus_ln(parts.precision);
        let d_alpha =
            digamma_minus_ln(parts.alpha) - precision_residual + Self::log_ratio(theta.mu, y);
        let d_beta = digamma_minus_ln(parts.beta) - precision_residual
            + Self::log_ratio(1.0 - theta.mu, 1.0 - y);
        let d_mu = parts.precision * (d_alpha - d_beta);
        let d_precision = theta.mu * d_alpha + (1.0 - theta.mu) * d_beta;

        BeinfTheta {
            mu: d_mu,
            sigma: d_precision * (-1.0 / (theta.sigma * theta.sigma)),
            nu: d_log_denominator,
            tau: d_log_denominator,
        }
    }

    fn cdf_theta(y: f64, theta: BeinfTheta) -> f64 {
        if !y.is_finite() {
            return f64::NAN;
        }
        let Some(parts) = Self::parts(theta) else {
            return f64::NAN;
        };
        if y < 0.0 {
            return 0.0;
        }
        if y == 0.0 {
            return parts.p0;
        }
        if y < 1.0 {
            #[allow(clippy::suboptimal_flops)]
            return (parts.p0 + parts.p_beta * regularized_beta(parts.alpha, parts.beta, y))
                .clamp(0.0, 1.0);
        }
        1.0
    }

    #[inline]
    fn nll_and_gradient_eta_values(y: f64, eta: BeinfEta) -> (f64, BeinfEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (nll, BeinfEta::from_array([f64::NAN; 4]));
        }

        let Some(parts) = Self::parts(theta) else {
            return (nll, BeinfEta::from_array([f64::NAN; 4]));
        };
        let gradient = Self::gradient_theta(y, theta, parts);
        (
            nll,
            BeinfEta {
                mu: gradient.mu * MuLink::derivative_inverse(eta.mu),
                sigma: gradient.sigma * SigmaLink::derivative_inverse(eta.sigma),
                nu: gradient.nu * NuLink::derivative_inverse(eta.nu),
                tau: gradient.tau * TauLink::derivative_inverse(eta.tau),
            },
        )
    }
}

impl<MuLink, SigmaLink, NuLink, TauLink> Default for Beinf<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: UnitIntervalLink<f64>,
    SigmaLink: UnitIntervalLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MuLink, SigmaLink, NuLink, TauLink> for Beinf<MuLink, SigmaLink, NuLink, TauLink>;
    parameters = (Mu, Sigma, Nu, Tau);
    arity = 4;
);

impl<MuLink, SigmaLink, NuLink, TauLink> Family for Beinf<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: UnitIntervalLink<f64>,
    SigmaLink: UnitIntervalLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    type Eta = BeinfEta;
    type Theta = BeinfTheta;
    type GradientEta = BeinfEta;
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

impl<MuLink, SigmaLink, NuLink, TauLink> InitialEtaFromObservations<4>
    for Beinf<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
    NuLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    TauLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values = weighted_values::<Self, _, _>(obs, |y| {
            (y.is_finite() && (0.0..=1.0).contains(&y)).then_some(y)
        });

        let interior = values
            .iter()
            .copied()
            .filter(|(y, _)| *y > 0.0 && *y < 1.0)
            .collect::<Vec<_>>();

        let summary = weighted_summary(&interior);
        let mu = probability_floor(summary.map_or(0.5, |s| s.mean));
        let max_variance = (mu * (1.0 - mu)).max(VARIANCE_FLOOR);
        let variance = summary.map_or(max_variance * 0.5, |s| {
            s.variance.clamp(VARIANCE_FLOOR, max_variance * 0.99)
        });
        let precision = positive_floor(max_variance / variance - 1.0);
        let sigma = probability_floor(1.0 / (precision + 1.0));

        let zero_weight = values
            .iter()
            .filter(|(y, _)| *y == 0.0)
            .map(|(_, w)| *w)
            .sum::<f64>();

        #[allow(clippy::float_cmp)]
        let one_weight = values
            .iter()
            .filter(|(y, _)| *y == 1.0)
            .map(|(_, w)| *w)
            .sum::<f64>();

        let interior_weight = interior
            .iter()
            .map(|(_, w)| *w)
            .sum::<f64>()
            .max(POSITIVE_FLOOR);

        BeinfEta {
            mu: MuLink::initial_eta_from_theta(mu),
            sigma: SigmaLink::initial_eta_from_theta(sigma),
            nu: NuLink::initial_eta_from_theta(positive_floor(zero_weight / interior_weight)),
            tau: TauLink::initial_eta_from_theta(positive_floor(one_weight / interior_weight)),
        }
    }
}

impl<MuLink, SigmaLink, NuLink, TauLink> HasCdf for Beinf<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: UnitIntervalLink<f64>,
    SigmaLink: UnitIntervalLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        Self::cdf_theta(y, *theta)
    }
}

impl<MuLink, SigmaLink, NuLink, TauLink> HasQuantile for Beinf<MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: UnitIntervalLink<f64>,
    SigmaLink: UnitIntervalLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        if !(0.0..=1.0).contains(&p) {
            return f64::NAN;
        }
        let Some(parts) = Self::parts(*theta) else {
            return f64::NAN;
        };
        if p <= parts.p0 {
            return 0.0;
        }
        if p >= parts.p0 + parts.p_beta {
            return 1.0;
        }

        let target = (p - parts.p0) / parts.p_beta;
        invert_bounded_cdf(target, 0.0, 1.0, |y| {
            regularized_beta(parts.alpha, parts.beta, y)
        })
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink, SigmaLink, NuLink, TauLink> TrySimulate<Rng>
    for Beinf<MuLink, SigmaLink, NuLink, TauLink>
where
    Rng: rand::Rng,
    MuLink: UnitIntervalLink<f64>,
    SigmaLink: UnitIntervalLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    type Sample = f64;

    fn try_sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Result<f64, SimulationError> {
        if Self::parts(*theta).is_none() {
            return Err(SimulationError::InvalidParameters("BEINF theta"));
        }
        crate::simulation::try_sample_quantile(rng, self, theta, "BEINF quantile")
    }
}

#[derive(Debug, Clone, Copy)]
struct BeinfParts {
    alpha: f64,
    beta: f64,
    precision: f64,
    inverse_denominator: f64,
    log_p0: f64,
    log_p1: f64,
    log_p_beta: f64,
    p0: f64,
    p_beta: f64,
}

/// Predictors for BEINF on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeinfEta {
    /// Beta-component mean predictor.
    pub mu: f64,
    /// Beta-component dispersion predictor in `(0, 1)`.
    pub sigma: f64,
    /// Positive zero-mass odds predictor.
    pub nu: f64,
    /// Positive one-mass odds predictor.
    pub tau: f64,
}

impl ParameterParts<4> for BeinfEta {
    #[inline]
    fn from_array(values: [f64; 4]) -> Self {
        Self {
            mu: values[0],
            sigma: values[1],
            nu: values[2],
            tau: values[3],
        }
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mu,
            1 => self.sigma,
            2 => self.nu,
            3 => self.tau,
            _ => unreachable!("beinf eta only has indices 0 through 3"),
        }
    }
}

/// Natural-scale BEINF parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeinfTheta {
    /// Beta-component mean in `(0, 1)`.
    pub mu: f64,
    /// Beta-component dispersion in `(0, 1)`.
    pub sigma: f64,
    /// Positive zero-mass odds relative to the beta component.
    pub nu: f64,
    /// Positive one-mass odds relative to the beta component.
    pub tau: f64,
}

#[cfg(test)]
mod tests {
    use gamlss_core::Family;
    #[cfg(feature = "rand")]
    use gamlss_core::TrySimulate;

    use super::{BeinfMuSigmaNuTau, BeinfTheta};
    use crate::{BetaMeanPrecision, BetaTheta};

    #[test]
    fn beinf_preserves_concentrated_beta_component_and_large_odds() {
        let family = BeinfMuSigmaNuTau::new();
        let beta = BetaMeanPrecision::new();
        let precision = 1.0e16;
        let theta = BeinfTheta {
            mu: 0.5,
            sigma: 1.0 / (precision + 1.0),
            nu: 0.3,
            tau: 0.4,
        };
        let expected = (1.0_f64 + theta.nu + theta.tau).ln()
            + beta.nll(
                0.5,
                &BetaTheta {
                    mu: theta.mu,
                    precision,
                },
                &mut (),
            );
        let nll = family.nll(0.5, &theta, &mut ());

        assert!((nll - expected).abs() < 1.0e-13, "nll was {nll}");

        let atom_nll = family.nll(
            0.0,
            &BeinfTheta {
                mu: 0.5,
                sigma: 0.2,
                nu: f64::MAX,
                tau: f64::MAX,
            },
            &mut (),
        );
        assert!(
            (atom_nll - std::f64::consts::LN_2).abs() < 1.0e-14,
            "atom nll was {atom_nll}"
        );
    }

    #[cfg(feature = "rand")]
    #[test]
    fn beinf_sampling_returns_unit_interval_values_and_errors_for_invalid_theta() {
        use rand::SeedableRng;

        let family = BeinfMuSigmaNuTau::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family
            .try_sample(
                &mut rng,
                &BeinfTheta {
                    mu: 0.4,
                    sigma: 0.2,
                    nu: 0.3,
                    tau: 0.4,
                },
            )
            .unwrap();
        assert!((0.0..=1.0).contains(&sample));
        assert_eq!(
            family.try_sample(
                &mut rng,
                &BeinfTheta {
                    mu: 0.4,
                    sigma: 0.0,
                    nu: 0.3,
                    tau: 0.4,
                }
            ),
            Err(gamlss_core::SimulationError::InvalidParameters(
                "BEINF theta"
            ))
        );
    }
}
