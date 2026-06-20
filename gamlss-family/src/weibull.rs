use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Family, HasCdf, HasCrps, HasQuantile, InitialEtaFromTheta, Log, Mean, ObservationView,
    ParameterParts, ParameterizedFamily, PositiveLink, Scale, Shape,
};

use crate::domain::{is_positive_finite, is_probability};
use crate::initial::{VARIANCE_FLOOR, positive_floor, weighted_summary};
use crate::special::{digamma, ln_gamma, regularized_gamma_lower};

const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// Weibull mean/shape parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeanShape;

/// Weibull scale/shape parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScaleShape;

/// Weibull family implementation carrier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Weibull<Param = ScaleShape, FirstLink = Log, SecondLink = Log> {
    marker: PhantomData<(Param, FirstLink, SecondLink)>,
}

impl<Param, FirstLink, SecondLink> Weibull<Param, FirstLink, SecondLink> {
    /// Creates a stateless Weibull family.
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn valid_scale_shape(theta: WeibullScaleShapeTheta) -> bool {
        is_positive_finite(theta.scale) && is_positive_finite(theta.shape)
    }

    #[inline(always)]
    fn mean_factor(shape: f64) -> f64 {
        ln_gamma(1.0 + 1.0 / shape).exp()
    }

    #[inline(always)]
    fn nll_scale_shape(y: f64, theta: WeibullScaleShapeTheta) -> f64 {
        if y <= 0.0 || !y.is_finite() || !Self::valid_scale_shape(theta) {
            return f64::INFINITY;
        }

        let log_ratio = y.ln() - theta.scale.ln();
        -theta.shape.ln() - (theta.shape - 1.0) * y.ln()
            + theta.shape * theta.scale.ln()
            + (theta.shape * log_ratio).exp()
    }

    #[inline(always)]
    fn gradient_scale_shape(y: f64, theta: WeibullScaleShapeTheta) -> (f64, f64) {
        let log_ratio = y.ln() - theta.scale.ln();
        let power = (theta.shape * log_ratio).exp();
        (
            theta.shape * (1.0 - power) / theta.scale,
            -1.0 / theta.shape - log_ratio + power * log_ratio,
        )
    }

    #[inline(always)]
    fn cdf_scale_shape(y: f64, theta: WeibullScaleShapeTheta) -> f64 {
        if !y.is_finite() || !Self::valid_scale_shape(theta) {
            return f64::NAN;
        }
        if y <= 0.0 {
            return 0.0;
        }

        -(-(y / theta.scale).powf(theta.shape)).exp_m1()
    }

    #[inline(always)]
    fn quantile_scale_shape(p: f64, theta: WeibullScaleShapeTheta) -> f64 {
        if !is_probability(p) || !Self::valid_scale_shape(theta) {
            return f64::NAN;
        }

        theta.scale * (-(-p).ln_1p()).powf(1.0 / theta.shape)
    }

    #[inline(always)]
    fn crps_scale_shape(y: f64, theta: WeibullScaleShapeTheta) -> f64 {
        if y < 0.0 || !y.is_finite() || !Self::valid_scale_shape(theta) {
            return f64::NAN;
        }

        let a = 1.0 + 1.0 / theta.shape;
        let mean = theta.scale * ln_gamma(a).exp();
        let t = if y == 0.0 {
            0.0
        } else {
            (y / theta.scale).powf(theta.shape)
        };
        let cdf = -(-t).exp_m1();
        y * (2.0 * cdf - 1.0) - 2.0 * mean * regularized_gamma_lower(a, t)
            + mean * 2.0_f64.powf(-1.0 / theta.shape)
    }

    #[inline(always)]
    fn initial_scale_shape<'obs, Obs>(obs: &'obs Obs) -> Option<(f64, f64)>
    where
        Obs: ObservationView<'obs, Observation = f64> + 'obs,
    {
        let mut log_values = Vec::new();
        for row in 0..obs.len() {
            let y = obs.observation_at(row);
            if y.is_finite() && y > 0.0 {
                log_values.push((y.ln(), obs.weight_at(row)));
            }
        }
        let summary = weighted_summary(&log_values)?;

        let shape = if summary.variance <= VARIANCE_FLOOR {
            10.0
        } else {
            positive_floor(std::f64::consts::PI / (6.0 * summary.variance).sqrt())
        };
        let scale = positive_floor((summary.mean + EULER_GAMMA / shape).exp());
        Some((scale, shape))
    }
}

impl<Param, FirstLink, SecondLink> Default for Weibull<Param, FirstLink, SecondLink> {
    fn default() -> Self {
        Self::new()
    }
}

/// Predictors for Weibull mean/shape on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeibullMeanShapeEta {
    /// Mean predictor.
    pub mean: f64,
    /// Shape predictor.
    pub shape: f64,
}

impl ParameterParts<2> for WeibullMeanShapeEta {
    #[inline(always)]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            mean: values[0],
            shape: values[1],
        }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mean,
            1 => self.shape,
            _ => unreachable!("weibull mean/shape eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale Weibull mean/shape parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeibullMeanShapeTheta {
    /// Positive mean.
    pub mean: f64,
    /// Positive shape.
    pub shape: f64,
}

impl WeibullMeanShapeTheta {
    #[inline(always)]
    fn scale_shape(self) -> WeibullScaleShapeTheta {
        WeibullScaleShapeTheta {
            scale: self.mean / Weibull::<MeanShape>::mean_factor(self.shape),
            shape: self.shape,
        }
    }
}

/// Predictors for Weibull scale/shape on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeibullScaleShapeEta {
    /// Scale predictor.
    pub scale: f64,
    /// Shape predictor.
    pub shape: f64,
}

impl ParameterParts<2> for WeibullScaleShapeEta {
    #[inline(always)]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            scale: values[0],
            shape: values[1],
        }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.scale,
            1 => self.shape,
            _ => unreachable!("weibull scale/shape eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale Weibull scale/shape parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeibullScaleShapeTheta {
    /// Positive scale.
    pub scale: f64,
    /// Positive shape.
    pub shape: f64,
}

impl<MeanLink, ShapeLink> Weibull<MeanShape, MeanLink, ShapeLink>
where
    MeanLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    #[inline(always)]
    fn theta_from_eta(eta: WeibullMeanShapeEta) -> WeibullMeanShapeTheta {
        WeibullMeanShapeTheta {
            mean: MeanLink::inverse(eta.mean),
            shape: ShapeLink::inverse(eta.shape),
        }
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: WeibullMeanShapeEta) -> (f64, WeibullMeanShapeEta) {
        let theta = Self::theta_from_eta(eta);
        let scale_shape = theta.scale_shape();
        let nll = Self::nll_scale_shape(y, scale_shape);
        if !nll.is_finite() {
            return (nll, WeibullMeanShapeEta::from_array([f64::NAN; 2]));
        }

        let (d_scale, d_shape_kernel) = Self::gradient_scale_shape(y, scale_shape);
        let d_mean = d_scale * scale_shape.scale / theta.mean;
        let a = 1.0 + 1.0 / theta.shape;
        let d_scale_d_shape = scale_shape.scale * digamma(a) / (theta.shape * theta.shape);
        let d_shape = d_shape_kernel + d_scale * d_scale_d_shape;

        (
            nll,
            WeibullMeanShapeEta {
                mean: d_mean * MeanLink::derivative_inverse(eta.mean),
                shape: d_shape * ShapeLink::derivative_inverse(eta.shape),
            },
        )
    }
}

impl<MeanLink, ShapeLink> Family for Weibull<MeanShape, MeanLink, ShapeLink>
where
    MeanLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    type Eta = WeibullMeanShapeEta;
    type Theta = WeibullMeanShapeTheta;
    type NllGradientEta = WeibullMeanShapeEta;
    type Observation<'obs> = f64;

    #[inline(always)]
    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline(always)]
    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::nll_scale_shape(y, theta.scale_shape())
    }

    #[inline(always)]
    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        Self::nll_scale_shape(y, Self::theta_from_eta(eta).scale_shape())
    }

    #[inline(always)]
    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        Self::nll_and_gradient_eta_values(y, eta)
    }
}

impl<MeanLink, ShapeLink> ParameterizedFamily<2> for Weibull<MeanShape, MeanLink, ShapeLink>
where
    MeanLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    ShapeLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Params = (Mean, Shape);
    type Links = (MeanLink, ShapeLink);

    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let Some((scale, shape)) = Self::initial_scale_shape(obs) else {
            return WeibullMeanShapeEta::from_array([0.0, 0.0]);
        };
        let mean = scale * Self::mean_factor(shape);

        WeibullMeanShapeEta {
            mean: MeanLink::initial_eta_from_theta(mean),
            shape: ShapeLink::initial_eta_from_theta(shape),
        }
    }
}

impl<ScaleLink, ShapeLink> Weibull<ScaleShape, ScaleLink, ShapeLink>
where
    ScaleLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    #[inline(always)]
    fn theta_from_eta(eta: WeibullScaleShapeEta) -> WeibullScaleShapeTheta {
        WeibullScaleShapeTheta {
            scale: ScaleLink::inverse(eta.scale),
            shape: ShapeLink::inverse(eta.shape),
        }
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(
        y: f64,
        eta: WeibullScaleShapeEta,
    ) -> (f64, WeibullScaleShapeEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_scale_shape(y, theta);
        if !nll.is_finite() {
            return (nll, WeibullScaleShapeEta::from_array([f64::NAN; 2]));
        }

        let (d_scale, d_shape) = Self::gradient_scale_shape(y, theta);
        (
            nll,
            WeibullScaleShapeEta {
                scale: d_scale * ScaleLink::derivative_inverse(eta.scale),
                shape: d_shape * ShapeLink::derivative_inverse(eta.shape),
            },
        )
    }
}

impl<ScaleLink, ShapeLink> Family for Weibull<ScaleShape, ScaleLink, ShapeLink>
where
    ScaleLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    type Eta = WeibullScaleShapeEta;
    type Theta = WeibullScaleShapeTheta;
    type NllGradientEta = WeibullScaleShapeEta;
    type Observation<'obs> = f64;

    #[inline(always)]
    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline(always)]
    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::nll_scale_shape(y, theta)
    }

    #[inline(always)]
    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        Self::nll_scale_shape(y, Self::theta_from_eta(eta))
    }

    #[inline(always)]
    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        Self::nll_and_gradient_eta_values(y, eta)
    }
}

impl<ScaleLink, ShapeLink> ParameterizedFamily<2> for Weibull<ScaleShape, ScaleLink, ShapeLink>
where
    ScaleLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    ShapeLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Params = (Scale, Shape);
    type Links = (ScaleLink, ShapeLink);

    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let Some((scale, shape)) = Self::initial_scale_shape(obs) else {
            return WeibullScaleShapeEta::from_array([0.0, 0.0]);
        };

        WeibullScaleShapeEta {
            scale: ScaleLink::initial_eta_from_theta(scale),
            shape: ShapeLink::initial_eta_from_theta(shape),
        }
    }
}

impl From<WeibullMeanShapeTheta> for WeibullScaleShapeTheta {
    #[inline(always)]
    fn from(theta: WeibullMeanShapeTheta) -> Self {
        theta.scale_shape()
    }
}

macro_rules! impl_weibull_helpers {
    ($param:ty, $first:ident, $second:ident) => {
        impl<$first, $second> HasCdf for Weibull<$param, $first, $second>
        where
            Weibull<$param, $first, $second>: for<'obs> Family<Observation<'obs> = f64>,
            <Weibull<$param, $first, $second> as Family>::Theta:
                Copy + Into<WeibullScaleShapeTheta>,
        {
            fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
                Self::cdf_scale_shape(y, theta.into())
            }
        }

        impl<$first, $second> HasQuantile for Weibull<$param, $first, $second>
        where
            Weibull<$param, $first, $second>: for<'obs> Family<Observation<'obs> = f64>,
            <Weibull<$param, $first, $second> as Family>::Theta:
                Copy + Into<WeibullScaleShapeTheta>,
        {
            fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
                Self::quantile_scale_shape(p, theta.into())
            }
        }

        impl<$first, $second> HasCrps for Weibull<$param, $first, $second>
        where
            Weibull<$param, $first, $second>: for<'obs> Family<Observation<'obs> = f64>,
            <Weibull<$param, $first, $second> as Family>::Theta:
                Copy + Into<WeibullScaleShapeTheta>,
        {
            fn crps<'obs>(&self, y: Self::Observation<'obs>, theta: Self::Theta) -> f64 {
                Self::crps_scale_shape(y, theta.into())
            }
        }

        #[cfg(feature = "rand")]
        impl<Rng, $first, $second> CanSimulate<Rng> for Weibull<$param, $first, $second>
        where
            Rng: rand::Rng,
            Weibull<$param, $first, $second>: for<'obs> Family<Observation<'obs> = f64>,
            <Weibull<$param, $first, $second> as Family>::Theta:
                Copy + Into<WeibullScaleShapeTheta>,
        {
            fn sample(&self, rng: &mut Rng, theta: Self::Theta) -> f64 {
                let theta = theta.into();
                if !Self::valid_scale_shape(theta) {
                    return f64::NAN;
                }

                rand_distr::Distribution::sample(
                    &rand_distr::Weibull::new(theta.scale, theta.shape)
                        .expect("validated weibull parameters must construct"),
                    rng,
                )
            }
        }
    };
}

impl_weibull_helpers!(MeanShape, MeanLink, ShapeLink);
impl_weibull_helpers!(ScaleShape, ScaleLink, ShapeLink);

/// Weibull distribution parameterized by mean and shape.
pub type WeibullMeanShape = Weibull<MeanShape, Log, Log>;
/// Weibull distribution parameterized by scale and shape.
pub type WeibullScaleShape = Weibull<ScaleShape, Log, Log>;

/// Backward-compatible eta alias for scale/shape Weibull.
pub type WeibullEta = WeibullScaleShapeEta;
/// Backward-compatible theta alias for scale/shape Weibull.
pub type WeibullTheta = WeibullScaleShapeTheta;

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;
    use gamlss_core::{Family, HasCdf, HasCrps, HasQuantile};

    use super::{
        WeibullMeanShape, WeibullMeanShapeTheta, WeibullScaleShape, WeibullScaleShapeTheta,
    };
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn weibull_parameterization_gradients_match_finite_difference() {
        assert_gradient_matches_finite_difference::<_, 2>(
            &WeibullScaleShape::new(),
            1.7,
            [0.8_f64.ln(), 1.5_f64.ln()],
        );
        assert_gradient_matches_finite_difference::<_, 2>(
            &WeibullMeanShape::new(),
            1.7,
            [1.2_f64.ln(), 1.5_f64.ln()],
        );
    }

    #[test]
    fn weibull_mean_matches_scale_shape_equivalent() {
        let mean = WeibullMeanShape::new();
        let scale_shape = WeibullScaleShape::new();
        let theta = WeibullMeanShapeTheta {
            mean: 1.2,
            shape: 1.5,
        };
        let canonical = theta.scale_shape();

        assert_relative_eq!(
            mean.nll(1.7, theta),
            scale_shape.nll(1.7, canonical),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean.cdf(1.7, theta),
            scale_shape.cdf(1.7, canonical),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean.quantile(0.4, theta),
            scale_shape.quantile(0.4, canonical),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean.crps(1.7, theta),
            scale_shape.crps(1.7, canonical),
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn weibull_rejects_invalid_domains_and_handles_boundaries() {
        let family = WeibullMeanShape::new();
        let theta = WeibullMeanShapeTheta {
            mean: 1.2,
            shape: 1.5,
        };

        assert!(family.nll(1.7, theta).is_finite());
        assert!(family.nll(0.0, theta).is_infinite());
        assert!(
            family
                .nll(
                    1.7,
                    WeibullMeanShapeTheta {
                        mean: 0.0,
                        shape: 1.5
                    }
                )
                .is_infinite()
        );
        assert!(
            family
                .nll(
                    1.7,
                    WeibullMeanShapeTheta {
                        mean: 1.2,
                        shape: 0.0
                    }
                )
                .is_infinite()
        );
        assert_eq!(family.cdf(0.0, theta), 0.0);
        assert_eq!(family.cdf(-1.0, theta), 0.0);
        assert_eq!(family.quantile(0.0, theta), 0.0);
        assert!(family.quantile(1.0, theta).is_infinite());
    }

    #[test]
    fn weibull_cdf_and_crps_match_fixed_values() {
        let family = WeibullScaleShape::new();
        let theta = WeibullScaleShapeTheta {
            scale: 3.0,
            shape: 2.0,
        };

        assert_relative_eq!(
            family.cdf(theta.scale, theta),
            1.0 - (-1.0_f64).exp(),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            family.cdf(theta.scale * std::f64::consts::LN_2.sqrt(), theta),
            0.5,
            epsilon = 1.0e-12
        );

        let exp_theta = WeibullScaleShapeTheta {
            scale: 0.5,
            shape: 1.0,
        };
        assert_relative_eq!(
            family.crps(1.0, exp_theta),
            0.385_335_283_236_612_7,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(family.crps(0.0, exp_theta), 0.25, epsilon = 1.0e-12);
    }

    #[cfg(feature = "rand")]
    #[test]
    fn weibull_sampling_returns_finite_values_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = WeibullMeanShape::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family.sample(
            &mut rng,
            WeibullMeanShapeTheta {
                mean: 1.2,
                shape: 1.5,
            },
        );
        assert!(sample > 0.0 && sample.is_finite());
    }
}
