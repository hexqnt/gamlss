use std::marker::PhantomData;

use gamlss_core::{
    Family, Identity, Link, Log, Mu, ParameterParts, ParameterizedFamily, PositiveLink, Sigma,
};

const HALF_LOG_2_PI: f64 = 0.918_938_533_204_672_7;

/// Log-normal location-scale family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogNormal<MuLink = Identity, SigmaLink = Log> {
    marker: PhantomData<(MuLink, SigmaLink)>,
}

impl<MuLink, SigmaLink> LogNormal<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    /// Creates a stateless log-normal family.
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn theta_from_eta(eta: LogNormalEta) -> LogNormalTheta {
        LogNormalTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
        }
    }

    #[inline(always)]
    fn nll_theta(y: f64, theta: LogNormalTheta) -> f64 {
        if y <= 0.0 || !y.is_finite() || theta.sigma <= 0.0 || !theta.sigma.is_finite() {
            return f64::INFINITY;
        }

        let log_y = y.ln();
        let residual = log_y - theta.mu;
        let z = residual / theta.sigma;
        log_y + HALF_LOG_2_PI + theta.sigma.ln() + 0.5 * z * z
    }

    #[inline(always)]
    fn nll_and_score_eta_values(y: f64, eta: LogNormalEta) -> (f64, LogNormalEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (
                nll,
                LogNormalEta {
                    mu: f64::NAN,
                    sigma: f64::NAN,
                },
            );
        }

        let residual = y.ln() - theta.mu;
        let sigma2 = theta.sigma * theta.sigma;
        let d_mu = (theta.mu - y.ln()) / sigma2;
        let d_sigma = 1.0 / theta.sigma - residual * residual / (sigma2 * theta.sigma);
        let score_eta = LogNormalEta {
            mu: d_mu * MuLink::derivative_inverse(eta.mu),
            sigma: d_sigma * SigmaLink::derivative_inverse(eta.sigma),
        };

        (nll, score_eta)
    }
}

impl<MuLink, SigmaLink> Default for LogNormal<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Predictors for the log-normal family on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogNormalEta {
    /// Location predictor for `log(Y)`.
    pub mu: f64,
    /// Scale predictor.
    pub sigma: f64,
}

impl ParameterParts<2> for LogNormalEta {
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
            _ => unreachable!("log-normal eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale log-normal parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogNormalTheta {
    /// Location parameter for `log(Y)`.
    pub mu: f64,
    /// Positive scale parameter.
    pub sigma: f64,
}

impl<MuLink, SigmaLink> Family for LogNormal<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    type Eta = LogNormalEta;
    type Theta = LogNormalTheta;
    type ScoreEta = LogNormalEta;

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
    fn nll_and_score_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::ScoreEta) {
        Self::nll_and_score_eta_values(y, eta)
    }
}

impl<MuLink, SigmaLink> ParameterizedFamily<2> for LogNormal<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    type Params = (Mu, Sigma);
    type Links = (MuLink, SigmaLink);
}

/// Log-normal distribution with identity link for `mu` and log link for `sigma`.
pub type DefaultLogNormal = LogNormal<Identity, Log>;

#[cfg(test)]
mod tests {
    use gamlss_core::Family;

    use super::{DefaultLogNormal, LogNormalTheta};
    use crate::test_support::assert_score_matches_finite_difference;

    #[test]
    fn log_normal_score_matches_finite_difference() {
        let family = DefaultLogNormal::new();
        assert_score_matches_finite_difference::<_, 2>(&family, 1.7, [0.4, -0.2]);
    }

    #[test]
    fn log_normal_rejects_invalid_domain_and_has_finite_nll_inside_domain() {
        let family = DefaultLogNormal::new();
        let theta = LogNormalTheta {
            mu: 0.4,
            sigma: 0.8,
        };

        assert!(family.nll(1.7, theta).is_finite());
        assert!(family.nll(0.0, theta).is_infinite());
        assert!(
            family
                .nll(
                    1.7,
                    LogNormalTheta {
                        mu: theta.mu,
                        sigma: 0.0,
                    },
                )
                .is_infinite()
        );
    }
}
