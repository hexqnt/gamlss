use std::marker::PhantomData;

use gamlss_core::{Family, Log, ParameterParts, ParameterizedFamily, PositiveLink, Scale, Shape};

/// Weibull family parameterized by positive shape and scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Weibull<ShapeLink = Log, ScaleLink = Log> {
    marker: PhantomData<(ShapeLink, ScaleLink)>,
}

impl<ShapeLink, ScaleLink> Weibull<ShapeLink, ScaleLink>
where
    ShapeLink: PositiveLink<f64>,
    ScaleLink: PositiveLink<f64>,
{
    /// Creates a stateless Weibull family.
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn theta_from_eta(eta: WeibullEta) -> WeibullTheta {
        WeibullTheta {
            shape: ShapeLink::inverse(eta.shape),
            scale: ScaleLink::inverse(eta.scale),
        }
    }

    #[inline(always)]
    fn nll_theta(y: f64, theta: WeibullTheta) -> f64 {
        if y <= 0.0
            || !y.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
            || theta.scale <= 0.0
            || !theta.scale.is_finite()
        {
            return f64::INFINITY;
        }

        let log_ratio = y.ln() - theta.scale.ln();
        -theta.shape.ln() - (theta.shape - 1.0) * y.ln()
            + theta.shape * theta.scale.ln()
            + (theta.shape * log_ratio).exp()
    }

    #[inline(always)]
    fn nll_and_score_eta_values(y: f64, eta: WeibullEta) -> (f64, WeibullEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (
                nll,
                WeibullEta {
                    shape: f64::NAN,
                    scale: f64::NAN,
                },
            );
        }

        let log_ratio = y.ln() - theta.scale.ln();
        let power = (theta.shape * log_ratio).exp();
        let d_shape = -1.0 / theta.shape - log_ratio + power * log_ratio;
        let d_scale = theta.shape * (1.0 - power) / theta.scale;
        let score_eta = WeibullEta {
            shape: d_shape * ShapeLink::derivative_inverse(eta.shape),
            scale: d_scale * ScaleLink::derivative_inverse(eta.scale),
        };

        (nll, score_eta)
    }
}

impl<ShapeLink, ScaleLink> Default for Weibull<ShapeLink, ScaleLink>
where
    ShapeLink: PositiveLink<f64>,
    ScaleLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Predictors for the Weibull family on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeibullEta {
    /// Shape predictor.
    pub shape: f64,
    /// Scale predictor.
    pub scale: f64,
}

impl ParameterParts<2> for WeibullEta {
    #[inline(always)]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            shape: values[0],
            scale: values[1],
        }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.shape,
            1 => self.scale,
            _ => unreachable!("weibull eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale Weibull parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeibullTheta {
    /// Positive shape parameter.
    pub shape: f64,
    /// Positive scale parameter.
    pub scale: f64,
}

impl<ShapeLink, ScaleLink> Family for Weibull<ShapeLink, ScaleLink>
where
    ShapeLink: PositiveLink<f64>,
    ScaleLink: PositiveLink<f64>,
{
    type Eta = WeibullEta;
    type Theta = WeibullTheta;
    type ScoreEta = WeibullEta;

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

impl<ShapeLink, ScaleLink> ParameterizedFamily<2> for Weibull<ShapeLink, ScaleLink>
where
    ShapeLink: PositiveLink<f64>,
    ScaleLink: PositiveLink<f64>,
{
    type Params = (Shape, Scale);
    type Links = (ShapeLink, ScaleLink);
}

/// Weibull distribution with log links for shape and scale.
pub type DefaultWeibull = Weibull<Log, Log>;

#[cfg(test)]
mod tests {
    use gamlss_core::Family;

    use super::{DefaultWeibull, WeibullTheta};
    use crate::test_support::assert_score_matches_finite_difference;

    #[test]
    fn weibull_score_matches_finite_difference() {
        let family = DefaultWeibull::new();
        assert_score_matches_finite_difference::<_, 2>(&family, 1.7, [0.4, -0.2]);
    }

    #[test]
    fn weibull_rejects_invalid_domain_and_has_finite_nll_inside_domain() {
        let family = DefaultWeibull::new();
        let theta = WeibullTheta {
            shape: 1.5,
            scale: 0.8,
        };

        assert!(family.nll(1.7, theta).is_finite());
        assert!(family.nll(0.0, theta).is_infinite());
        assert!(
            family
                .nll(
                    1.7,
                    WeibullTheta {
                        shape: 0.0,
                        scale: theta.scale,
                    },
                )
                .is_infinite()
        );
    }
}
