use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasCrps, HasQuantile, Identity, InitialEtaFromObservations,
    InitialEtaFromTheta, Link, Log, ObservationView, ParameterParts, PositiveLink, Scale, Shape,
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

use crate::domain::{is_positive_finite, is_probability};
use crate::initial::{positive_floor, weighted_values};

/// Generalized Pareto excess distribution with log/identity links.
pub type GeneralizedParetoScaleShape = GeneralizedPareto<Log, Identity>;

/// Generalized Pareto family for non-negative threshold excesses.
///
/// The location/threshold is fixed at zero. A positive `shape` gives an
/// unbounded heavy tail, zero is the exponential limit, and a negative value
/// gives the finite upper endpoint `-scale / shape`.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/distributions/generalized_pareto.svg")
)]
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneralizedPareto<ScaleLink = Log, ShapeLink = Identity> {
    marker: PhantomData<(ScaleLink, ShapeLink)>,
}

impl<ScaleLink, ShapeLink> GeneralizedPareto<ScaleLink, ShapeLink>
where
    ScaleLink: PositiveLink<f64>,
    ShapeLink: Link<f64>,
{
    /// Creates a stateless generalized Pareto family.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    fn theta_from_eta(eta: GeneralizedParetoEta) -> GeneralizedParetoTheta {
        GeneralizedParetoTheta {
            scale: ScaleLink::inverse(eta.scale),
            shape: ShapeLink::inverse(eta.shape),
        }
    }

    fn valid_theta(theta: GeneralizedParetoTheta) -> bool {
        is_positive_finite(theta.scale) && theta.shape.is_finite()
    }

    fn valid_support(y: f64, theta: GeneralizedParetoTheta) -> bool {
        y >= 0.0
            && y.is_finite()
            && Self::valid_theta(theta)
            && (theta.shape >= 0.0 || theta.shape.mul_add(y / theta.scale, 1.0) > 0.0)
    }

    #[inline]
    fn log1p_ratio(value: f64) -> f64 {
        if value.abs() > 1.0e-4 {
            return value.ln_1p() / value;
        }
        let mut power = 1.0;
        let mut sum = 0.0;
        for order in 0..=8 {
            let term = power / f64::from(order + 1);
            sum += if order % 2 == 0 { term } else { -term };
            power *= value;
        }
        sum
    }

    #[inline]
    fn d_log1p_ratio(value: f64) -> f64 {
        if value.abs() > 1.0e-4 {
            return (value / (1.0 + value) - value.ln_1p()) / (value * value);
        }
        let mut power = 1.0;
        let mut sum = 0.0;
        for order in 0..=7 {
            let coefficient = f64::from(order + 1) / f64::from(order + 2);
            sum += if order % 2 == 0 {
                -coefficient * power
            } else {
                coefficient * power
            };
            power *= value;
        }
        sum
    }

    #[allow(clippy::suboptimal_flops)]
    fn nll_theta(y: f64, theta: GeneralizedParetoTheta) -> f64 {
        if !Self::valid_support(y, theta) {
            return f64::INFINITY;
        }
        let scaled = y / theta.scale;
        let product = theta.shape * scaled;
        let tail = if product.is_finite() {
            (1.0 + theta.shape) * scaled * Self::log1p_ratio(product)
        } else {
            (1.0 + 1.0 / theta.shape) * (theta.shape.ln() + scaled.ln())
        };
        theta.scale.ln() + tail
    }

    #[allow(clippy::suboptimal_flops)]
    fn gradient_theta(y: f64, theta: GeneralizedParetoTheta) -> (f64, f64) {
        let scaled = y / theta.scale;
        let product = theta.shape * scaled;
        if product.is_infinite() && product.is_sign_positive() {
            let log_product = theta.shape.ln() + scaled.ln();
            return (
                -1.0 / (theta.shape * theta.scale),
                ((1.0 + theta.shape) - log_product) / (theta.shape * theta.shape),
            );
        }
        let denominator = 1.0 + product;
        let d_scale = (1.0 - (1.0 + theta.shape) * scaled / denominator) / theta.scale;
        let ratio = Self::log1p_ratio(product);
        let d_shape =
            scaled * ratio + (1.0 + theta.shape) * scaled * scaled * Self::d_log1p_ratio(product);
        (d_scale, d_shape)
    }

    fn cdf_theta(y: f64, theta: GeneralizedParetoTheta) -> f64 {
        if y < 0.0 {
            return 0.0;
        }
        if theta.shape < 0.0 && y >= -theta.scale / theta.shape {
            return 1.0;
        }

        -Self::log_survival_theta(y, theta).exp_m1()
    }

    #[inline]
    fn log_survival_theta(y: f64, theta: GeneralizedParetoTheta) -> f64 {
        let scaled = y / theta.scale;
        if theta.shape == 0.0 {
            return -scaled;
        }
        let product = theta.shape * scaled;
        if product.is_finite() {
            -scaled * Self::log1p_ratio(product)
        } else {
            -(theta.shape.ln() + scaled.ln()) / theta.shape
        }
    }
}

impl<ScaleLink, ShapeLink> Default for GeneralizedPareto<ScaleLink, ShapeLink>
where
    ScaleLink: PositiveLink<f64>,
    ShapeLink: Link<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<ScaleLink, ShapeLink> for GeneralizedPareto<ScaleLink, ShapeLink>;
    parameters = (Scale, Shape);
    arity = 2;
);

impl<ScaleLink, ShapeLink> Family for GeneralizedPareto<ScaleLink, ShapeLink>
where
    ScaleLink: PositiveLink<f64>,
    ShapeLink: Link<f64>,
{
    type Eta = GeneralizedParetoEta;
    type Theta = GeneralizedParetoTheta;
    type GradientEta = GeneralizedParetoEta;
    type Observation<'obs> = f64;
    type Workspace = ();

    fn workspace(&self) -> Self::Workspace {}
    fn theta(&self, eta: &Self::Eta, _workspace: &mut ()) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }
    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut ()) -> f64 {
        Self::nll_theta(y, *theta)
    }
    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut (),
    ) -> (f64, Self::GradientEta) {
        let theta = Self::theta_from_eta(*eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (nll, GeneralizedParetoEta::from_array([f64::NAN; 2]));
        }
        let (d_scale, d_shape) = Self::gradient_theta(y, theta);
        (
            nll,
            GeneralizedParetoEta {
                scale: d_scale * ScaleLink::derivative_inverse(eta.scale),
                shape: d_shape * ShapeLink::derivative_inverse(eta.shape),
            },
        )
    }
}

impl<ScaleLink, ShapeLink> InitialEtaFromObservations<2> for GeneralizedPareto<ScaleLink, ShapeLink>
where
    ScaleLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    ShapeLink: InitialEtaFromTheta<f64> + Link<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = f64> + 'obs,
    {
        let values =
            weighted_values::<Self, _, _>(obs, |y| (y >= 0.0 && y.is_finite()).then_some(y));
        let scale = crate::initial::weighted_mean(&values).map_or(1.0, positive_floor);
        GeneralizedParetoEta {
            scale: ScaleLink::initial_eta_from_theta(scale),
            shape: ShapeLink::initial_eta_from_theta(0.0),
        }
    }
}

impl<ScaleLink, ShapeLink> HasCdf for GeneralizedPareto<ScaleLink, ShapeLink>
where
    ScaleLink: PositiveLink<f64>,
    ShapeLink: Link<f64>,
{
    fn cdf(&self, y: f64, theta: &Self::Theta) -> f64 {
        if !y.is_finite() || !Self::valid_theta(*theta) {
            return f64::NAN;
        }
        Self::cdf_theta(y, *theta)
    }
}

impl<ScaleLink, ShapeLink> HasQuantile for GeneralizedPareto<ScaleLink, ShapeLink>
where
    ScaleLink: PositiveLink<f64>,
    ShapeLink: Link<f64>,
{
    #[allow(clippy::float_cmp)]
    fn quantile(&self, probability: f64, theta: &Self::Theta) -> f64 {
        if !is_probability(probability) || !Self::valid_theta(*theta) {
            return f64::NAN;
        }
        if probability == 1.0 {
            return if theta.shape < 0.0 {
                -theta.scale / theta.shape
            } else {
                f64::INFINITY
            };
        }
        let negative_log_survival = -(-probability).ln_1p();
        let value = theta.shape * negative_log_survival;
        let exponential_ratio = if value.abs() <= 1.0e-8 {
            1.0 + value.mul_add(0.5, value * value / 6.0)
        } else {
            value.exp_m1() / value
        };
        theta.scale * negative_log_survival * exponential_ratio
    }
}

impl<ScaleLink, ShapeLink> HasCrps for GeneralizedPareto<ScaleLink, ShapeLink>
where
    ScaleLink: PositiveLink<f64>,
    ShapeLink: Link<f64>,
{
    #[allow(clippy::suboptimal_flops)]
    fn crps(&self, y: f64, theta: &Self::Theta) -> f64 {
        if theta.shape >= 1.0 || !Self::valid_support(y, *theta) {
            return f64::NAN;
        }

        let log_survival = Self::log_survival_theta(y, *theta);
        let integrated_survival =
            theta.scale / (1.0 - theta.shape) * -((1.0 - theta.shape) * log_survival).exp_m1();
        (y - 2.0 * integrated_survival + theta.scale / (2.0 - theta.shape)).max(0.0)
    }
}

#[cfg(feature = "rand")]
impl<Rng, ScaleLink, ShapeLink> TrySimulate<Rng> for GeneralizedPareto<ScaleLink, ShapeLink>
where
    Rng: rand::Rng,
    ScaleLink: PositiveLink<f64>,
    ShapeLink: Link<f64>,
{
    type Sample = f64;

    fn try_sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Result<f64, SimulationError> {
        if !Self::valid_theta(*theta) {
            return Err(SimulationError::InvalidParameters(
                "Generalized Pareto theta",
            ));
        }
        crate::simulation::try_sample_quantile(rng, self, theta, "Generalized Pareto sample")
    }
}

/// Link-scale predictors for [`GeneralizedPareto`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneralizedParetoEta {
    /// Scale predictor.
    pub scale: f64,
    /// Shape predictor.
    pub shape: f64,
}

impl ParameterParts<2> for GeneralizedParetoEta {
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            scale: values[0],
            shape: values[1],
        }
    }
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.scale,
            1 => self.shape,
            _ => unreachable!("generalized Pareto eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale parameters for [`GeneralizedPareto`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneralizedParetoTheta {
    /// Positive scale.
    pub scale: f64,
    /// Real shape, with zero representing the exponential limit.
    pub shape: f64,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use approx::assert_relative_eq;
    use gamlss_core::{Family, HasCdf, HasCrps, HasQuantile};

    use super::{GeneralizedParetoScaleShape, GeneralizedParetoTheta};
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn gradient_matches_finite_difference_including_exponential_limit() {
        let family = GeneralizedParetoScaleShape::new();
        assert_gradient_matches_finite_difference::<_, 2>(&family, 1.7, [0.2, 0.3]);
        assert_gradient_matches_finite_difference::<_, 2>(&family, 1.7, [0.2, 0.0]);
        assert_gradient_matches_finite_difference::<_, 2>(&family, 1.7, [0.2, -0.2]);
    }

    #[test]
    fn exponential_limit_and_finite_endpoint_are_correct() {
        let family = GeneralizedParetoScaleShape::new();
        let exponential = GeneralizedParetoTheta {
            scale: 1.3,
            shape: 0.0,
        };
        let y = 1.7;
        assert_relative_eq!(
            family.nll(y, &exponential, &mut ()),
            exponential.scale.ln() + y / exponential.scale,
            epsilon = 1.0e-14
        );
        assert_relative_eq!(
            family.quantile(family.cdf(y, &exponential), &exponential),
            y,
            epsilon = 1.0e-12
        );

        let bounded = GeneralizedParetoTheta {
            scale: 2.0,
            shape: -0.5,
        };
        assert_eq!(family.cdf(4.0, &bounded), 1.0);
        assert!(family.nll(4.0, &bounded, &mut ()).is_infinite());
        assert_eq!(family.quantile(1.0, &bounded), 4.0);
    }

    #[test]
    fn tiny_shape_and_overflowing_scaled_value_remain_stable() {
        let family = GeneralizedParetoScaleShape::new();
        let y = 1.7;
        let exponential_nll = 1.3_f64.ln() + y / 1.3;
        for shape in [-1.0e-12, 1.0e-12] {
            let theta = GeneralizedParetoTheta { scale: 1.3, shape };
            assert_relative_eq!(
                family.nll(y, &theta, &mut ()),
                exponential_nll,
                epsilon = 2.0e-12
            );
        }

        let extreme = GeneralizedParetoTheta {
            scale: 1.0e-308,
            shape: 2.0,
        };
        assert!(family.nll(1.0, &extreme, &mut ()).is_finite());
        assert!(family.cdf(1.0, &extreme).is_finite());

        let exponential = GeneralizedParetoTheta {
            scale: 1.0e-308,
            shape: 0.0,
        };
        assert_relative_eq!(family.cdf(1.0, &exponential), 1.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn generalized_pareto_crps_matches_reference_values() {
        let family = GeneralizedParetoScaleShape::new();
        for (shape, expected) in [
            (0.0, 0.267_478_243_173_292_55),
            (0.3, 0.369_388_535_807_232_63),
            (-0.4, 0.179_115_795_155_739_5),
        ] {
            assert_relative_eq!(
                family.crps(0.7, &GeneralizedParetoTheta { scale: 1.3, shape }),
                expected,
                epsilon = 1.0e-12
            );
        }
        assert!(
            family
                .crps(
                    0.7,
                    &GeneralizedParetoTheta {
                        scale: 1.3,
                        shape: 1.0,
                    },
                )
                .is_nan()
        );

        for shape in [0.0, 0.5] {
            let score = family.crps(
                f64::MAX,
                &GeneralizedParetoTheta {
                    scale: f64::MIN_POSITIVE,
                    shape,
                },
            );
            assert!(score.is_finite(), "shape={shape}, score={score}");
        }
    }
}
