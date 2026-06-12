use std::marker::PhantomData;

use gamlss_core::{Family, Log, Mu, ParameterParts, ParameterizedFamily, PositiveLink, Shape};

const HALF_LOG_2_PI: f64 = 0.918_938_533_204_672_7;

/// Inverse Gaussian family parameterized by positive mean and shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InverseGaussian<MuLink = Log, ShapeLink = Log> {
    marker: PhantomData<(MuLink, ShapeLink)>,
}

impl<MuLink, ShapeLink> InverseGaussian<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    /// Creates a stateless inverse Gaussian family.
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn theta_from_eta(eta: InverseGaussianEta) -> InverseGaussianTheta {
        InverseGaussianTheta {
            mu: MuLink::inverse(eta.mu),
            shape: ShapeLink::inverse(eta.shape),
        }
    }

    #[inline(always)]
    fn nll_theta(y: f64, theta: InverseGaussianTheta) -> f64 {
        if y <= 0.0
            || !y.is_finite()
            || theta.mu <= 0.0
            || !theta.mu.is_finite()
            || theta.shape <= 0.0
            || !theta.shape.is_finite()
        {
            return f64::INFINITY;
        }

        let residual = y - theta.mu;
        HALF_LOG_2_PI + 1.5 * y.ln() - 0.5 * theta.shape.ln()
            + theta.shape * residual * residual / (2.0 * theta.mu * theta.mu * y)
    }

    #[inline(always)]
    fn nll_and_score_eta_values(y: f64, eta: InverseGaussianEta) -> (f64, InverseGaussianEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (
                nll,
                InverseGaussianEta {
                    mu: f64::NAN,
                    shape: f64::NAN,
                },
            );
        }

        let residual = y - theta.mu;
        let d_mu = -theta.shape * residual / (theta.mu * theta.mu * theta.mu);
        let d_shape = -0.5 / theta.shape + residual * residual / (2.0 * theta.mu * theta.mu * y);
        let score_eta = InverseGaussianEta {
            mu: d_mu * MuLink::derivative_inverse(eta.mu),
            shape: d_shape * ShapeLink::derivative_inverse(eta.shape),
        };

        (nll, score_eta)
    }
}

impl<MuLink, ShapeLink> Default for InverseGaussian<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Predictors for the inverse Gaussian family on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InverseGaussianEta {
    /// Mean predictor.
    pub mu: f64,
    /// Shape predictor.
    pub shape: f64,
}

impl ParameterParts<2> for InverseGaussianEta {
    #[inline(always)]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            mu: values[0],
            shape: values[1],
        }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mu,
            1 => self.shape,
            _ => unreachable!("inverse Gaussian eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale inverse Gaussian parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InverseGaussianTheta {
    /// Positive mean parameter.
    pub mu: f64,
    /// Positive shape parameter.
    pub shape: f64,
}

impl<MuLink, ShapeLink> Family for InverseGaussian<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    type Eta = InverseGaussianEta;
    type Theta = InverseGaussianTheta;
    type ScoreEta = InverseGaussianEta;

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

impl<MuLink, ShapeLink> ParameterizedFamily<2> for InverseGaussian<MuLink, ShapeLink>
where
    MuLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    type Params = (Mu, Shape);
    type Links = (MuLink, ShapeLink);
}

/// Inverse Gaussian distribution with log links for mean and shape.
pub type DefaultInverseGaussian = InverseGaussian<Log, Log>;

#[cfg(test)]
mod tests {
    use gamlss_core::Family;

    use super::{DefaultInverseGaussian, InverseGaussianTheta};
    use crate::test_support::assert_score_matches_finite_difference;

    #[test]
    fn inverse_gaussian_score_matches_finite_difference() {
        let family = DefaultInverseGaussian::new();
        assert_score_matches_finite_difference::<_, 2>(&family, 1.7, [0.4, -0.2]);
    }

    #[test]
    fn inverse_gaussian_rejects_invalid_domain_and_has_finite_nll_inside_domain() {
        let family = DefaultInverseGaussian::new();
        let theta = InverseGaussianTheta {
            mu: 1.5,
            shape: 0.8,
        };

        assert!(family.nll(1.7, theta).is_finite());
        assert!(family.nll(0.0, theta).is_infinite());
        assert!(
            family
                .nll(
                    1.7,
                    InverseGaussianTheta {
                        mu: 0.0,
                        shape: theta.shape,
                    },
                )
                .is_infinite()
        );
    }
}
