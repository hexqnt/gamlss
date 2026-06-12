use std::marker::PhantomData;

use gamlss_core::{
    Family, Link, Log, Logit, Mu, ParameterParts, ParameterizedFamily, PositiveLink, Precision,
};

use crate::special::{digamma, ln_gamma};

/// Beta family parameterized by mean in `(0, 1)` and positive precision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Beta<MuLink = Logit, PrecisionLink = Log> {
    marker: PhantomData<(MuLink, PrecisionLink)>,
}

impl<MuLink, PrecisionLink> Beta<MuLink, PrecisionLink>
where
    MuLink: Link<f64>,
    PrecisionLink: PositiveLink<f64>,
{
    /// Creates a stateless beta family.
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn theta_from_eta(eta: BetaEta) -> BetaTheta {
        BetaTheta {
            mu: MuLink::inverse(eta.mu),
            precision: PrecisionLink::inverse(eta.precision),
        }
    }

    #[inline(always)]
    fn nll_theta(y: f64, theta: BetaTheta) -> f64 {
        if y <= 0.0
            || y >= 1.0
            || !y.is_finite()
            || theta.mu <= 0.0
            || theta.mu >= 1.0
            || !theta.mu.is_finite()
            || theta.precision <= 0.0
            || !theta.precision.is_finite()
        {
            return f64::INFINITY;
        }

        let alpha = theta.mu * theta.precision;
        let beta = (1.0 - theta.mu) * theta.precision;
        ln_gamma(alpha) + ln_gamma(beta)
            - ln_gamma(theta.precision)
            - (alpha - 1.0) * y.ln()
            - (beta - 1.0) * (1.0 - y).ln()
    }

    #[inline(always)]
    fn nll_and_score_eta_values(y: f64, eta: BetaEta) -> (f64, BetaEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (
                nll,
                BetaEta {
                    mu: f64::NAN,
                    precision: f64::NAN,
                },
            );
        }

        let alpha = theta.mu * theta.precision;
        let beta = (1.0 - theta.mu) * theta.precision;
        let common = digamma(theta.precision);
        let d_alpha = digamma(alpha) - common - y.ln();
        let d_beta = digamma(beta) - common - (1.0 - y).ln();
        let d_mu = theta.precision * (d_alpha - d_beta);
        let d_precision = theta.mu * d_alpha + (1.0 - theta.mu) * d_beta;
        let score_eta = BetaEta {
            mu: d_mu * MuLink::derivative_inverse(eta.mu),
            precision: d_precision * PrecisionLink::derivative_inverse(eta.precision),
        };

        (nll, score_eta)
    }
}

impl<MuLink, PrecisionLink> Default for Beta<MuLink, PrecisionLink>
where
    MuLink: Link<f64>,
    PrecisionLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Predictors for the beta family on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BetaEta {
    /// Mean predictor.
    pub mu: f64,
    /// Precision predictor.
    pub precision: f64,
}

impl ParameterParts<2> for BetaEta {
    #[inline(always)]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            mu: values[0],
            precision: values[1],
        }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mu,
            1 => self.precision,
            _ => unreachable!("beta eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale beta parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BetaTheta {
    /// Mean parameter in `(0, 1)`.
    pub mu: f64,
    /// Positive precision parameter.
    pub precision: f64,
}

impl<MuLink, PrecisionLink> Family for Beta<MuLink, PrecisionLink>
where
    MuLink: Link<f64>,
    PrecisionLink: PositiveLink<f64>,
{
    type Eta = BetaEta;
    type Theta = BetaTheta;
    type ScoreEta = BetaEta;

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

impl<MuLink, PrecisionLink> ParameterizedFamily<2> for Beta<MuLink, PrecisionLink>
where
    MuLink: Link<f64>,
    PrecisionLink: PositiveLink<f64>,
{
    type Params = (Mu, Precision);
    type Links = (MuLink, PrecisionLink);
}

/// Beta distribution with logit link for mean and log link for precision.
pub type DefaultBeta = Beta<Logit, Log>;

#[cfg(test)]
mod tests {
    use gamlss_core::Family;

    use super::{BetaTheta, DefaultBeta};
    use crate::test_support::assert_score_matches_finite_difference;

    #[test]
    fn beta_score_matches_finite_difference() {
        let family = DefaultBeta::new();
        assert_score_matches_finite_difference::<_, 2>(&family, 0.4, [0.2, 1.0]);
    }

    #[test]
    fn beta_rejects_invalid_domain_and_has_finite_nll_inside_domain() {
        let family = DefaultBeta::new();
        let theta = BetaTheta {
            mu: 0.4,
            precision: 3.0,
        };

        assert!(family.nll(0.4, theta).is_finite());
        assert!(family.nll(0.0, theta).is_infinite());
        assert!(family.nll(1.0, theta).is_infinite());
        assert!(
            family
                .nll(
                    0.4,
                    BetaTheta {
                        mu: 1.0,
                        precision: theta.precision,
                    },
                )
                .is_infinite()
        );
    }
}
