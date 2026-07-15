use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Family, HasCdf, HasQuantile, InitialEtaFromObservations, InitialEtaFromTheta, Log,
    ObservationView, ParameterParts, PositiveLink, Scale, Shape,
};

use crate::domain::{is_positive_finite, is_probability};
use crate::initial::{positive_floor, weighted_quantile, weighted_values};

/// Lomax distribution with log links for shape and scale.
pub type LomaxShapeScale = Lomax<Log, Log>;

/// Lomax (Pareto type II) family parameterized by positive shape and scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lomax<ShapeLink = Log, ScaleLink = Log> {
    marker: PhantomData<(ShapeLink, ScaleLink)>,
}

impl<ShapeLink, ScaleLink> Lomax<ShapeLink, ScaleLink>
where
    ShapeLink: PositiveLink<f64>,
    ScaleLink: PositiveLink<f64>,
{
    /// Creates a stateless Lomax family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(eta: LomaxEta) -> LomaxTheta {
        eta.theta_from_links::<ShapeLink, ScaleLink>()
    }

    #[inline]
    fn valid_theta(theta: LomaxTheta) -> bool {
        is_positive_finite(theta.shape) && is_positive_finite(theta.scale)
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn nll_theta(y: f64, theta: LomaxTheta) -> f64 {
        if y < 0.0 || !y.is_finite() || !Self::valid_theta(theta) {
            return f64::INFINITY;
        }

        -theta.shape.ln() + theta.scale.ln() + (theta.shape + 1.0) * (y / theta.scale).ln_1p()
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn nll_and_gradient_eta_values(y: f64, eta: LomaxEta) -> (f64, LomaxEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (
                nll,
                LomaxEta {
                    shape: f64::NAN,
                    scale: f64::NAN,
                },
            );
        }

        let ratio = y / theta.scale;
        let d_shape = -1.0 / theta.shape + ratio.ln_1p();
        let d_scale = (1.0 - theta.shape * ratio) / (theta.scale * (1.0 + ratio));
        let gradient_eta = eta.chain_gradient::<ShapeLink, ScaleLink>(d_shape, d_scale);

        (nll, gradient_eta)
    }
}

impl<ShapeLink, ScaleLink> Default for Lomax<ShapeLink, ScaleLink>
where
    ShapeLink: PositiveLink<f64>,
    ScaleLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<ShapeLink, ScaleLink> for Lomax<ShapeLink, ScaleLink>;
    parameters = (Shape, Scale);
    arity = 2;
);

impl<ShapeLink, ScaleLink> Family for Lomax<ShapeLink, ScaleLink>
where
    ShapeLink: PositiveLink<f64>,
    ScaleLink: PositiveLink<f64>,
{
    type Eta = LomaxEta;
    type Theta = LomaxTheta;
    type GradientEta = LomaxEta;
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

impl<ShapeLink, ScaleLink> InitialEtaFromObservations<2> for Lomax<ShapeLink, ScaleLink>
where
    ShapeLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    ScaleLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    #[allow(clippy::suboptimal_flops)]
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let values =
            weighted_values::<Self, _, _>(obs, |y| (y.is_finite() && y >= 0.0).then_some(y));
        let Some(q50) = weighted_quantile(&values, 0.5) else {
            return LomaxEta::from_array([0.0, 0.0]);
        };
        let q75 = weighted_quantile(&values, 0.75).unwrap_or(q50);

        let (shape, scale) = if q50 > 0.0 && q75 > q50 {
            let ratio = (q75 / q50).max(2.0 + 1.0e-6);
            let shape = positive_floor(std::f64::consts::LN_2 / (ratio - 1.0).ln());
            let scale = positive_floor(q50 / (2.0_f64.powf(1.0 / shape) - 1.0));
            (shape, scale)
        } else {
            (2.0, positive_floor(q50.max(1.0)))
        };

        LomaxEta {
            shape: ShapeLink::initial_eta_from_theta(shape),
            scale: ScaleLink::initial_eta_from_theta(scale),
        }
    }
}

impl<ShapeLink, ScaleLink> HasCdf for Lomax<ShapeLink, ScaleLink>
where
    ShapeLink: PositiveLink<f64>,
    ScaleLink: PositiveLink<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        if !y.is_finite() || !Self::valid_theta(*theta) {
            return f64::NAN;
        }
        if y < 0.0 {
            return 0.0;
        }

        -(-theta.shape * (y / theta.scale).ln_1p()).exp_m1()
    }
}

impl<ShapeLink, ScaleLink> HasQuantile for Lomax<ShapeLink, ScaleLink>
where
    ShapeLink: PositiveLink<f64>,
    ScaleLink: PositiveLink<f64>,
{
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        if !is_probability(p) || !Self::valid_theta(*theta) {
            return f64::NAN;
        }

        theta.scale * ((-p).ln_1p() / -theta.shape).exp_m1()
    }
}

#[cfg(feature = "rand")]
impl<Rng, ShapeLink, ScaleLink> CanSimulate<Rng> for Lomax<ShapeLink, ScaleLink>
where
    Rng: rand::Rng,
    ShapeLink: PositiveLink<f64>,
    ScaleLink: PositiveLink<f64>,
{
    type Sample = f64;

    fn sample(&self, rng: &mut Rng, theta: &Self::Theta) -> f64 {
        if !Self::valid_theta(*theta) {
            return f64::NAN;
        }

        let uniform: f64 = rand_distr::Distribution::sample(&rand_distr::Open01, rng);
        theta.scale * ((-uniform).ln_1p() / -theta.shape).exp_m1()
    }
}

define_two_positive_parameter_blocks! {
    eta:
    /// Predictors for the Lomax family on the link scale.
    LomaxEta {
        /// Shape predictor.
        shape,
        /// Scale predictor.
        scale,
    }
    theta:
    /// Natural-scale Lomax parameters.
    LomaxTheta {
        /// Positive shape parameter.
        shape,
        /// Positive scale parameter.
        scale,
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;
    use gamlss_core::{Family, HasCdf, HasQuantile};

    use super::{LomaxShapeScale, LomaxTheta};
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn lomax_gradient_matches_finite_difference() {
        let family = LomaxShapeScale::new();
        assert_gradient_matches_finite_difference::<_, 2>(&family, 1.7, [0.4, -0.2]);
    }

    #[test]
    fn lomax_rejects_invalid_domain_and_has_finite_nll_inside_domain() {
        let family = LomaxShapeScale::new();
        let theta = LomaxTheta {
            shape: 1.5,
            scale: 0.8,
        };

        assert!(family.nll(0.0, &theta, &mut family.workspace()).is_finite());
        assert!(family.nll(1.7, &theta, &mut family.workspace()).is_finite());
        assert!(
            family
                .nll(-1.0, &theta, &mut family.workspace())
                .is_infinite()
        );
        assert!(
            family
                .nll(
                    1.7,
                    &LomaxTheta {
                        shape: 0.0,
                        scale: theta.scale,
                    },
                    &mut family.workspace(),
                )
                .is_infinite()
        );
    }

    #[test]
    #[allow(clippy::float_cmp, clippy::suboptimal_flops, clippy::imprecise_flops)]
    fn lomax_cdf_matches_reference_points() {
        let family = LomaxShapeScale::new();
        let theta = LomaxTheta {
            shape: 2.0,
            scale: 3.0,
        };
        let median = theta.scale * 2.0_f64.powf(1.0 / theta.shape) - theta.scale;

        assert_relative_eq!(family.cdf(0.0, &theta), 0.0, epsilon = 1.0e-12);
        assert_relative_eq!(family.cdf(median, &theta), 0.5, epsilon = 1.0e-12);
        assert_eq!(family.cdf(-1.0, &theta), 0.0);
        assert!(family.cdf(f64::NAN, &theta).is_nan());
    }

    #[test]
    fn lomax_quantile_inverts_cdf() {
        let family = LomaxShapeScale::new();
        let theta = LomaxTheta {
            shape: 2.0,
            scale: 3.0,
        };

        assert_relative_eq!(family.quantile(0.0, &theta), 0.0, epsilon = 1.0e-12);
        assert!(family.quantile(1.0, &theta).is_infinite());

        let y = family.quantile(0.75, &theta);
        assert_relative_eq!(family.cdf(y, &theta), 0.75, epsilon = 1.0e-12);
        assert!(family.quantile(f64::NAN, &theta).is_nan());
        assert!(
            family
                .quantile(
                    0.5,
                    &LomaxTheta {
                        shape: 0.0,
                        scale: 3.0
                    }
                )
                .is_nan()
        );
    }

    #[cfg(feature = "rand")]
    #[test]
    fn lomax_sampling_returns_finite_values_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = LomaxShapeScale::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family.sample(
            &mut rng,
            &LomaxTheta {
                shape: 1.5,
                scale: 0.8,
            },
        );
        assert!(sample >= 0.0 && sample.is_finite());
        assert!(
            family
                .sample(
                    &mut rng,
                    &LomaxTheta {
                        shape: 0.0,
                        scale: 0.8
                    }
                )
                .is_nan()
        );
    }
}
