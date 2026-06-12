use std::marker::PhantomData;

use gamlss_core::{Family, Log, ParameterParts, ParameterizedFamily, PositiveLink, Rate, Shape};

use crate::special::{digamma, ln_gamma};

/// Gamma family parameterized by positive shape and rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gamma<ShapeLink = Log, RateLink = Log> {
    marker: PhantomData<(ShapeLink, RateLink)>,
}

impl<ShapeLink, RateLink> Gamma<ShapeLink, RateLink>
where
    ShapeLink: PositiveLink<f64>,
    RateLink: PositiveLink<f64>,
{
    /// Creates a stateless gamma family.
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn theta_from_eta(eta: GammaEta) -> GammaTheta {
        GammaTheta {
            shape: ShapeLink::inverse(eta.shape),
            rate: RateLink::inverse(eta.rate),
        }
    }

    #[inline(always)]
    fn nll_theta(y: f64, theta: GammaTheta) -> f64 {
        if y <= 0.0
            || !y.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
            || theta.rate <= 0.0
            || !theta.rate.is_finite()
        {
            return f64::INFINITY;
        }

        ln_gamma(theta.shape) - theta.shape * theta.rate.ln() - (theta.shape - 1.0) * y.ln()
            + theta.rate * y
    }

    #[inline(always)]
    fn nll_and_score_eta_values(y: f64, eta: GammaEta) -> (f64, GammaEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (
                nll,
                GammaEta {
                    shape: f64::NAN,
                    rate: f64::NAN,
                },
            );
        }

        let d_shape = digamma(theta.shape) - theta.rate.ln() - y.ln();
        let d_rate = y - theta.shape / theta.rate;
        let score_eta = GammaEta {
            shape: d_shape * ShapeLink::derivative_inverse(eta.shape),
            rate: d_rate * RateLink::derivative_inverse(eta.rate),
        };

        (nll, score_eta)
    }
}

impl<ShapeLink, RateLink> Default for Gamma<ShapeLink, RateLink>
where
    ShapeLink: PositiveLink<f64>,
    RateLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Predictors for the gamma family on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaEta {
    /// Shape predictor.
    pub shape: f64,
    /// Rate predictor.
    pub rate: f64,
}

impl ParameterParts<2> for GammaEta {
    #[inline(always)]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            shape: values[0],
            rate: values[1],
        }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.shape,
            1 => self.rate,
            _ => unreachable!("gamma eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale gamma parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaTheta {
    /// Positive shape parameter.
    pub shape: f64,
    /// Positive rate parameter.
    pub rate: f64,
}

impl<ShapeLink, RateLink> Family for Gamma<ShapeLink, RateLink>
where
    ShapeLink: PositiveLink<f64>,
    RateLink: PositiveLink<f64>,
{
    type Eta = GammaEta;
    type Theta = GammaTheta;
    type ScoreEta = GammaEta;

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

impl<ShapeLink, RateLink> ParameterizedFamily<2> for Gamma<ShapeLink, RateLink>
where
    ShapeLink: PositiveLink<f64>,
    RateLink: PositiveLink<f64>,
{
    type Params = (Shape, Rate);
    type Links = (ShapeLink, RateLink);
}

/// Gamma distribution with log links for shape and rate.
pub type DefaultGamma = Gamma<Log, Log>;

#[cfg(test)]
mod tests {
    use gamlss_core::{Family, Link, Log};

    use super::{DefaultGamma, GammaTheta};
    use crate::test_support::assert_score_matches_finite_difference;

    #[test]
    fn gamma_score_matches_finite_difference() {
        let family = DefaultGamma::new();
        assert_score_matches_finite_difference::<_, 2>(&family, 1.7, [0.4, -0.2]);
    }

    #[test]
    fn gamma_rejects_invalid_domain_and_has_finite_nll_inside_domain() {
        let family = DefaultGamma::new();
        let theta = GammaTheta {
            shape: Log::inverse(0.4),
            rate: Log::inverse(-0.2),
        };

        assert!(family.nll(1.7, theta).is_finite());
        assert!(family.nll(0.0, theta).is_infinite());
        assert!(
            family
                .nll(
                    1.7,
                    GammaTheta {
                        shape: 0.0,
                        rate: theta.rate,
                    },
                )
                .is_infinite()
        );
    }
}
