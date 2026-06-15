use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{Family, HasCdf, Log, ParameterParts, ParameterizedFamily, PositiveLink, Rate};

/// Exponential family parameterized by positive rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exponential<RateLink = Log> {
    marker: PhantomData<RateLink>,
}

impl<RateLink> Exponential<RateLink>
where
    RateLink: PositiveLink<f64>,
{
    /// Creates a stateless exponential family.
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn theta_from_eta(eta: ExponentialEta) -> ExponentialTheta {
        ExponentialTheta {
            rate: RateLink::inverse(eta.rate),
        }
    }

    #[inline(always)]
    fn nll_theta(y: f64, theta: ExponentialTheta) -> f64 {
        if y < 0.0 || !y.is_finite() || theta.rate <= 0.0 || !theta.rate.is_finite() {
            return f64::INFINITY;
        }

        -theta.rate.ln() + theta.rate * y
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: ExponentialEta) -> (f64, ExponentialEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (nll, ExponentialEta { rate: f64::NAN });
        }

        let d_rate = y - 1.0 / theta.rate;
        let gradient_eta = ExponentialEta {
            rate: d_rate * RateLink::derivative_inverse(eta.rate),
        };

        (nll, gradient_eta)
    }
}

impl<RateLink> Default for Exponential<RateLink>
where
    RateLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Predictor for the exponential family on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExponentialEta {
    /// Rate predictor.
    pub rate: f64,
}

impl ParameterParts<1> for ExponentialEta {
    #[inline(always)]
    fn from_array(values: [f64; 1]) -> Self {
        Self { rate: values[0] }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.rate,
            _ => unreachable!("exponential eta only has index 0"),
        }
    }
}

/// Natural-scale exponential parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExponentialTheta {
    /// Positive rate parameter.
    pub rate: f64,
}

impl<RateLink> Family for Exponential<RateLink>
where
    RateLink: PositiveLink<f64>,
{
    type Eta = ExponentialEta;
    type Theta = ExponentialTheta;
    type NllGradientEta = ExponentialEta;
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

impl<RateLink> ParameterizedFamily<1> for Exponential<RateLink>
where
    RateLink: PositiveLink<f64>,
{
    type Params = (Rate,);
    type Links = (RateLink,);
}

impl<RateLink> HasCdf for Exponential<RateLink>
where
    RateLink: PositiveLink<f64>,
{
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
        if y < 0.0 || !y.is_finite() || theta.rate <= 0.0 || !theta.rate.is_finite() {
            return f64::NAN;
        }

        -(-theta.rate * y).exp_m1()
    }
}

#[cfg(feature = "rand")]
impl<Rng, RateLink> CanSimulate<Rng> for Exponential<RateLink>
where
    Rng: rand::Rng,
    RateLink: PositiveLink<f64>,
{
    fn sample(&self, rng: &mut Rng, theta: Self::Theta) -> f64 {
        if theta.rate <= 0.0 || !theta.rate.is_finite() {
            return f64::NAN;
        }

        rand_distr::Distribution::sample(
            &rand_distr::Exp::new(theta.rate).expect("validated exponential rate must construct"),
            rng,
        )
    }
}

/// Exponential distribution with log link for rate.
pub type DefaultExponential = Exponential<Log>;

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;
    use gamlss_core::{Family, HasCdf};

    use super::{DefaultExponential, ExponentialTheta};
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn exponential_gradient_matches_finite_difference() {
        let family = DefaultExponential::new();
        assert_gradient_matches_finite_difference::<_, 1>(&family, 1.7, [0.4]);
    }

    #[test]
    fn exponential_rejects_invalid_domain_and_has_finite_nll_inside_domain() {
        let family = DefaultExponential::new();
        let theta = ExponentialTheta { rate: 2.0 };

        assert!(family.nll(0.0, theta).is_finite());
        assert!(family.nll(1.7, theta).is_finite());
        assert!(family.nll(-1.0, theta).is_infinite());
        assert!(
            family
                .nll(1.7, ExponentialTheta { rate: 0.0 })
                .is_infinite()
        );
    }

    #[test]
    fn exponential_cdf_matches_reference_points() {
        let family = DefaultExponential::new();
        let theta = ExponentialTheta { rate: 2.0 };

        assert_relative_eq!(family.cdf(0.0, theta), 0.0, epsilon = 1.0e-12);
        assert_relative_eq!(
            family.cdf(std::f64::consts::LN_2 / theta.rate, theta),
            0.5,
            epsilon = 1.0e-12
        );
        assert!(family.cdf(-1.0, theta).is_nan());
    }

    #[cfg(feature = "rand")]
    #[test]
    fn exponential_sampling_returns_finite_values_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = DefaultExponential::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family.sample(&mut rng, ExponentialTheta { rate: 2.0 });
        assert!(sample >= 0.0 && sample.is_finite());
        assert!(
            family
                .sample(&mut rng, ExponentialTheta { rate: 0.0 })
                .is_nan()
        );
    }
}
