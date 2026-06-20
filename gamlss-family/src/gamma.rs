use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Cv, Family, HasCdf, HasCrps, HasQuantile, InitialEtaFromTheta, Log, Mean, ObservationView,
    ParameterParts, ParameterizedFamily, PositiveLink, Rate, Shape,
};

use crate::initial::{LARGE_SHAPE, VARIANCE_FLOOR, positive_floor, weighted_summary};
use crate::special::{digamma, invert_positive_cdf, ln_beta, ln_gamma, regularized_gamma_lower};

/// Gamma mean/coefficient-of-variation parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeanCv;

/// Gamma mean/shape parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeanShape;

/// Gamma shape/rate parameterization marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShapeRate;

/// Gamma family implementation carrier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gamma<Param = ShapeRate, FirstLink = Log, SecondLink = Log> {
    marker: PhantomData<(Param, FirstLink, SecondLink)>,
}

impl<Param, FirstLink, SecondLink> Gamma<Param, FirstLink, SecondLink> {
    /// Creates a stateless gamma family.
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn valid_shape_rate(theta: GammaShapeRateTheta) -> bool {
        theta.shape > 0.0 && theta.shape.is_finite() && theta.rate > 0.0 && theta.rate.is_finite()
    }

    #[inline(always)]
    fn nll_shape_rate(y: f64, theta: GammaShapeRateTheta) -> f64 {
        if y <= 0.0 || !y.is_finite() || !Self::valid_shape_rate(theta) {
            return f64::INFINITY;
        }

        ln_gamma(theta.shape) - theta.shape * theta.rate.ln() - (theta.shape - 1.0) * y.ln()
            + theta.rate * y
    }

    #[inline(always)]
    fn gradient_shape_rate(y: f64, theta: GammaShapeRateTheta) -> (f64, f64) {
        (
            digamma(theta.shape) - theta.rate.ln() - y.ln(),
            y - theta.shape / theta.rate,
        )
    }

    #[inline(always)]
    fn cdf_shape_rate(y: f64, theta: GammaShapeRateTheta) -> f64 {
        if !y.is_finite() || !Self::valid_shape_rate(theta) {
            return f64::NAN;
        }
        if y <= 0.0 {
            return 0.0;
        }

        regularized_gamma_lower(theta.shape, theta.rate * y)
    }

    #[inline(always)]
    fn quantile_shape_rate(p: f64, theta: GammaShapeRateTheta) -> f64 {
        if !Self::valid_shape_rate(theta) {
            return f64::NAN;
        }

        invert_positive_cdf(p, |y| Self::cdf_shape_rate(y, theta))
    }

    #[inline(always)]
    fn crps_shape_rate(y: f64, theta: GammaShapeRateTheta) -> f64 {
        if y < 0.0 || !y.is_finite() || !Self::valid_shape_rate(theta) {
            return f64::NAN;
        }

        let f_shape = regularized_gamma_lower(theta.shape, theta.rate * y);
        let f_next_shape = regularized_gamma_lower(theta.shape + 1.0, theta.rate * y);
        let mean = theta.shape / theta.rate;
        let beta_term = ln_beta(theta.shape + 0.5, 0.5).exp() / (std::f64::consts::PI * theta.rate);

        y * (2.0 * f_shape - 1.0) - mean * (2.0 * f_next_shape - 1.0) - beta_term
    }

    #[inline(always)]
    fn initial_mean_shape<'obs, Obs>(obs: &'obs Obs) -> Option<(f64, f64)>
    where
        Obs: ObservationView<'obs, Observation = f64> + 'obs,
    {
        let mut values = Vec::new();
        for row in 0..obs.len() {
            let y = obs.observation_at(row);
            if y.is_finite() && y > 0.0 {
                values.push((y, obs.weight_at(row)));
            }
        }
        let summary = weighted_summary(&values)?;

        let mean = positive_floor(summary.mean);
        let shape = if summary.variance <= VARIANCE_FLOOR {
            LARGE_SHAPE
        } else {
            positive_floor(mean * mean / summary.variance)
        };
        Some((mean, shape))
    }
}

impl<Param, FirstLink, SecondLink> Default for Gamma<Param, FirstLink, SecondLink> {
    fn default() -> Self {
        Self::new()
    }
}

/// Predictors for gamma mean/CV on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaMeanCvEta {
    /// Mean predictor.
    pub mean: f64,
    /// Coefficient-of-variation predictor.
    pub cv: f64,
}

impl ParameterParts<2> for GammaMeanCvEta {
    #[inline(always)]
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            mean: values[0],
            cv: values[1],
        }
    }

    #[inline(always)]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mean,
            1 => self.cv,
            _ => unreachable!("gamma mean/CV eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale gamma mean/CV parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaMeanCvTheta {
    /// Positive mean.
    pub mean: f64,
    /// Positive coefficient of variation.
    pub cv: f64,
}

impl GammaMeanCvTheta {
    #[inline(always)]
    fn shape_rate(self) -> GammaShapeRateTheta {
        let shape = 1.0 / (self.cv * self.cv);
        GammaShapeRateTheta {
            shape,
            rate: shape / self.mean,
        }
    }
}

/// Predictors for gamma mean/shape on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaMeanShapeEta {
    /// Mean predictor.
    pub mean: f64,
    /// Shape predictor.
    pub shape: f64,
}

impl ParameterParts<2> for GammaMeanShapeEta {
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
            _ => unreachable!("gamma mean/shape eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale gamma mean/shape parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaMeanShapeTheta {
    /// Positive mean.
    pub mean: f64,
    /// Positive shape.
    pub shape: f64,
}

impl GammaMeanShapeTheta {
    #[inline(always)]
    fn shape_rate(self) -> GammaShapeRateTheta {
        GammaShapeRateTheta {
            shape: self.shape,
            rate: self.shape / self.mean,
        }
    }
}

/// Predictors for gamma shape/rate on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaShapeRateEta {
    /// Shape predictor.
    pub shape: f64,
    /// Rate predictor.
    pub rate: f64,
}

impl ParameterParts<2> for GammaShapeRateEta {
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
            _ => unreachable!("gamma shape/rate eta only has indices 0 and 1"),
        }
    }
}

/// Natural-scale gamma shape/rate parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaShapeRateTheta {
    /// Positive shape.
    pub shape: f64,
    /// Positive rate.
    pub rate: f64,
}

impl<MeanLink, CvLink> Gamma<MeanCv, MeanLink, CvLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
{
    #[inline(always)]
    fn theta_from_eta(eta: GammaMeanCvEta) -> GammaMeanCvTheta {
        GammaMeanCvTheta {
            mean: MeanLink::inverse(eta.mean),
            cv: CvLink::inverse(eta.cv),
        }
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: GammaMeanCvEta) -> (f64, GammaMeanCvEta) {
        let theta = Self::theta_from_eta(eta);
        let shape_rate = theta.shape_rate();
        let nll = Self::nll_shape_rate(y, shape_rate);
        if !nll.is_finite() {
            return (nll, GammaMeanCvEta::from_array([f64::NAN; 2]));
        }

        let (d_shape, d_rate) = Self::gradient_shape_rate(y, shape_rate);
        let d_mean = d_rate * (-shape_rate.rate / theta.mean);
        let d_cv = d_shape * (-2.0 * shape_rate.shape / theta.cv)
            + d_rate * (-2.0 * shape_rate.rate / theta.cv);

        (
            nll,
            GammaMeanCvEta {
                mean: d_mean * MeanLink::derivative_inverse(eta.mean),
                cv: d_cv * CvLink::derivative_inverse(eta.cv),
            },
        )
    }
}

impl<MeanLink, CvLink> Family for Gamma<MeanCv, MeanLink, CvLink>
where
    MeanLink: PositiveLink<f64>,
    CvLink: PositiveLink<f64>,
{
    type Eta = GammaMeanCvEta;
    type Theta = GammaMeanCvTheta;
    type NllGradientEta = GammaMeanCvEta;
    type Observation<'obs> = f64;

    #[inline(always)]
    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline(always)]
    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::nll_shape_rate(y, theta.shape_rate())
    }

    #[inline(always)]
    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        Self::nll_shape_rate(y, Self::theta_from_eta(eta).shape_rate())
    }

    #[inline(always)]
    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        Self::nll_and_gradient_eta_values(y, eta)
    }
}

impl<MeanLink, CvLink> ParameterizedFamily<2> for Gamma<MeanCv, MeanLink, CvLink>
where
    MeanLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    CvLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Params = (Mean, Cv);
    type Links = (MeanLink, CvLink);

    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let Some((mean, shape)) = Self::initial_mean_shape(obs) else {
            return GammaMeanCvEta::from_array([0.0, 0.0]);
        };
        let cv = positive_floor(1.0 / shape.sqrt());

        GammaMeanCvEta {
            mean: MeanLink::initial_eta_from_theta(mean),
            cv: CvLink::initial_eta_from_theta(cv),
        }
    }
}

impl<MeanLink, ShapeLink> Gamma<MeanShape, MeanLink, ShapeLink>
where
    MeanLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    #[inline(always)]
    fn theta_from_eta(eta: GammaMeanShapeEta) -> GammaMeanShapeTheta {
        GammaMeanShapeTheta {
            mean: MeanLink::inverse(eta.mean),
            shape: ShapeLink::inverse(eta.shape),
        }
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: GammaMeanShapeEta) -> (f64, GammaMeanShapeEta) {
        let theta = Self::theta_from_eta(eta);
        let shape_rate = theta.shape_rate();
        let nll = Self::nll_shape_rate(y, shape_rate);
        if !nll.is_finite() {
            return (nll, GammaMeanShapeEta::from_array([f64::NAN; 2]));
        }

        let (d_shape, d_rate) = Self::gradient_shape_rate(y, shape_rate);
        let d_mean = d_rate * (-theta.shape / (theta.mean * theta.mean));
        let d_shape_param = d_shape + d_rate / theta.mean;

        (
            nll,
            GammaMeanShapeEta {
                mean: d_mean * MeanLink::derivative_inverse(eta.mean),
                shape: d_shape_param * ShapeLink::derivative_inverse(eta.shape),
            },
        )
    }
}

impl<MeanLink, ShapeLink> Family for Gamma<MeanShape, MeanLink, ShapeLink>
where
    MeanLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    type Eta = GammaMeanShapeEta;
    type Theta = GammaMeanShapeTheta;
    type NllGradientEta = GammaMeanShapeEta;
    type Observation<'obs> = f64;

    #[inline(always)]
    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline(always)]
    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::nll_shape_rate(y, theta.shape_rate())
    }

    #[inline(always)]
    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        Self::nll_shape_rate(y, Self::theta_from_eta(eta).shape_rate())
    }

    #[inline(always)]
    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        Self::nll_and_gradient_eta_values(y, eta)
    }
}

impl<MeanLink, ShapeLink> ParameterizedFamily<2> for Gamma<MeanShape, MeanLink, ShapeLink>
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
        let Some((mean, shape)) = Self::initial_mean_shape(obs) else {
            return GammaMeanShapeEta::from_array([0.0, 0.0]);
        };

        GammaMeanShapeEta {
            mean: MeanLink::initial_eta_from_theta(mean),
            shape: ShapeLink::initial_eta_from_theta(shape),
        }
    }
}

impl<ShapeLink, RateLink> Gamma<ShapeRate, ShapeLink, RateLink>
where
    ShapeLink: PositiveLink<f64>,
    RateLink: PositiveLink<f64>,
{
    #[inline(always)]
    fn theta_from_eta(eta: GammaShapeRateEta) -> GammaShapeRateTheta {
        GammaShapeRateTheta {
            shape: ShapeLink::inverse(eta.shape),
            rate: RateLink::inverse(eta.rate),
        }
    }

    #[inline(always)]
    fn nll_and_gradient_eta_values(y: f64, eta: GammaShapeRateEta) -> (f64, GammaShapeRateEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_shape_rate(y, theta);
        if !nll.is_finite() {
            return (nll, GammaShapeRateEta::from_array([f64::NAN; 2]));
        }

        let (d_shape, d_rate) = Self::gradient_shape_rate(y, theta);
        (
            nll,
            GammaShapeRateEta {
                shape: d_shape * ShapeLink::derivative_inverse(eta.shape),
                rate: d_rate * RateLink::derivative_inverse(eta.rate),
            },
        )
    }
}

impl<ShapeLink, RateLink> Family for Gamma<ShapeRate, ShapeLink, RateLink>
where
    ShapeLink: PositiveLink<f64>,
    RateLink: PositiveLink<f64>,
{
    type Eta = GammaShapeRateEta;
    type Theta = GammaShapeRateTheta;
    type NllGradientEta = GammaShapeRateEta;
    type Observation<'obs> = f64;

    #[inline(always)]
    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline(always)]
    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        Self::nll_shape_rate(y, theta)
    }

    #[inline(always)]
    fn nll_eta(&self, y: f64, eta: Self::Eta) -> f64 {
        Self::nll_shape_rate(y, Self::theta_from_eta(eta))
    }

    #[inline(always)]
    fn nll_and_gradient_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::NllGradientEta) {
        Self::nll_and_gradient_eta_values(y, eta)
    }
}

impl<ShapeLink, RateLink> ParameterizedFamily<2> for Gamma<ShapeRate, ShapeLink, RateLink>
where
    ShapeLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    RateLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Params = (Shape, Rate);
    type Links = (ShapeLink, RateLink);

    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let Some((mean, shape)) = Self::initial_mean_shape(obs) else {
            return GammaShapeRateEta::from_array([0.0, 0.0]);
        };
        let rate = positive_floor(shape / mean);

        GammaShapeRateEta {
            shape: ShapeLink::initial_eta_from_theta(shape),
            rate: RateLink::initial_eta_from_theta(rate),
        }
    }
}

macro_rules! impl_gamma_helpers {
    ($param:ty, $first:ident, $second:ident) => {
        impl<$first, $second> HasCdf for Gamma<$param, $first, $second>
        where
            Gamma<$param, $first, $second>: for<'obs> Family<Observation<'obs> = f64>,
            <Gamma<$param, $first, $second> as Family>::Theta: Copy + Into<GammaShapeRateTheta>,
        {
            fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
                Self::cdf_shape_rate(y, theta.into())
            }
        }

        impl<$first, $second> HasQuantile for Gamma<$param, $first, $second>
        where
            Gamma<$param, $first, $second>: for<'obs> Family<Observation<'obs> = f64>,
            <Gamma<$param, $first, $second> as Family>::Theta: Copy + Into<GammaShapeRateTheta>,
        {
            fn quantile(&self, p: f64, theta: Self::Theta) -> f64 {
                Self::quantile_shape_rate(p, theta.into())
            }
        }

        impl<$first, $second> HasCrps for Gamma<$param, $first, $second>
        where
            Gamma<$param, $first, $second>: for<'obs> Family<Observation<'obs> = f64>,
            <Gamma<$param, $first, $second> as Family>::Theta: Copy + Into<GammaShapeRateTheta>,
        {
            fn crps<'obs>(&self, y: Self::Observation<'obs>, theta: Self::Theta) -> f64 {
                Self::crps_shape_rate(y, theta.into())
            }
        }

        #[cfg(feature = "rand")]
        impl<Rng, $first, $second> CanSimulate<Rng> for Gamma<$param, $first, $second>
        where
            Rng: rand::Rng,
            Gamma<$param, $first, $second>: for<'obs> Family<Observation<'obs> = f64>,
            <Gamma<$param, $first, $second> as Family>::Theta: Copy + Into<GammaShapeRateTheta>,
        {
            fn sample(&self, rng: &mut Rng, theta: Self::Theta) -> f64 {
                let theta = theta.into();
                if !Self::valid_shape_rate(theta) {
                    return f64::NAN;
                }

                rand_distr::Distribution::sample(
                    &rand_distr::Gamma::new(theta.shape, 1.0 / theta.rate)
                        .expect("validated gamma parameters must construct"),
                    rng,
                )
            }
        }
    };
}

impl From<GammaMeanCvTheta> for GammaShapeRateTheta {
    #[inline(always)]
    fn from(theta: GammaMeanCvTheta) -> Self {
        theta.shape_rate()
    }
}

impl From<GammaMeanShapeTheta> for GammaShapeRateTheta {
    #[inline(always)]
    fn from(theta: GammaMeanShapeTheta) -> Self {
        theta.shape_rate()
    }
}

impl_gamma_helpers!(MeanCv, MeanLink, CvLink);
impl_gamma_helpers!(MeanShape, MeanLink, ShapeLink);
impl_gamma_helpers!(ShapeRate, ShapeLink, RateLink);

/// Gamma distribution parameterized by mean and coefficient of variation.
pub type GammaMeanCv = Gamma<MeanCv, Log, Log>;
/// Gamma distribution parameterized by mean and shape.
pub type GammaMeanShape = Gamma<MeanShape, Log, Log>;
/// Gamma distribution parameterized by shape and rate.
pub type GammaShapeRate = Gamma<ShapeRate, Log, Log>;

/// Backward-compatible eta alias for the shape/rate gamma.
pub type GammaEta = GammaShapeRateEta;
/// Backward-compatible theta alias for the shape/rate gamma.
pub type GammaTheta = GammaShapeRateTheta;

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;
    use gamlss_core::{Family, HasCdf, HasCrps, HasQuantile};
    use statrs::distribution::{ContinuousCDF, Gamma as StatrsGamma};

    use super::{
        GammaMeanCv, GammaMeanCvTheta, GammaMeanShape, GammaShapeRate, GammaShapeRateTheta,
    };
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn gamma_parameterization_gradients_match_finite_difference() {
        assert_gradient_matches_finite_difference::<_, 2>(&GammaShapeRate::new(), 1.7, [0.4, -0.2]);
        assert_gradient_matches_finite_difference::<_, 2>(
            &GammaMeanShape::new(),
            1.7,
            [1.4_f64.ln(), 2.5_f64.ln()],
        );
        assert_gradient_matches_finite_difference::<_, 2>(
            &GammaMeanCv::new(),
            1.7,
            [1.4_f64.ln(), 0.6_f64.ln()],
        );
    }

    #[test]
    fn gamma_mean_cv_matches_shape_rate_equivalent() {
        let mean_cv = GammaMeanCv::new();
        let shape_rate = GammaShapeRate::new();
        let theta = GammaMeanCvTheta { mean: 1.4, cv: 0.6 };
        let canonical = theta.shape_rate();

        assert_relative_eq!(
            mean_cv.nll(1.7, theta),
            shape_rate.nll(1.7, canonical),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean_cv.cdf(1.7, theta),
            shape_rate.cdf(1.7, canonical),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            mean_cv.quantile(0.4, theta),
            shape_rate.quantile(0.4, canonical),
            epsilon = 1.0e-8
        );
        assert_relative_eq!(
            mean_cv.crps(1.7, theta),
            shape_rate.crps(1.7, canonical),
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn gamma_rejects_invalid_domains() {
        let family = GammaMeanCv::new();

        assert!(
            family
                .nll(1.7, GammaMeanCvTheta { mean: 1.0, cv: 0.5 })
                .is_finite()
        );
        assert!(
            family
                .nll(0.0, GammaMeanCvTheta { mean: 1.0, cv: 0.5 })
                .is_infinite()
        );
        assert!(
            family
                .nll(1.7, GammaMeanCvTheta { mean: 0.0, cv: 0.5 })
                .is_infinite()
        );
        assert!(
            family
                .nll(1.7, GammaMeanCvTheta { mean: 1.0, cv: 0.0 })
                .is_infinite()
        );
        assert!(
            family
                .nll(
                    1.7,
                    GammaMeanCvTheta {
                        mean: f64::NAN,
                        cv: 0.5
                    }
                )
                .is_infinite()
        );
    }

    #[test]
    fn gamma_cdf_and_quantile_match_statrs_reference() {
        let family = GammaShapeRate::new();
        let theta = GammaShapeRateTheta {
            shape: 2.5,
            rate: 1.7,
        };
        let reference = StatrsGamma::new(theta.shape, theta.rate).unwrap();

        for y in [0.05, 0.25, 1.0, 2.0, 8.0] {
            assert_relative_eq!(family.cdf(y, theta), reference.cdf(y), epsilon = 1.0e-11);
        }

        for p in [0.01, 0.1, 0.5, 0.9, 0.99] {
            assert_relative_eq!(
                family.quantile(p, theta),
                reference.inverse_cdf(p),
                epsilon = 1.0e-8
            );
        }
    }

    #[test]
    fn gamma_boundaries_and_crps_behave_like_shape_rate_kernel() {
        let family = GammaShapeRate::new();
        let theta = GammaShapeRateTheta {
            shape: 1.0,
            rate: 2.0,
        };

        assert_eq!(family.cdf(0.0, theta), 0.0);
        assert_eq!(family.quantile(0.0, theta), 0.0);
        assert_eq!(family.quantile(1.0, theta), f64::INFINITY);
        assert!(family.quantile(f64::NAN, theta).is_nan());
        assert_relative_eq!(
            family.crps(1.0, theta),
            0.385_335_283_236_612_7,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(family.crps(0.0, theta), 0.25, epsilon = 1.0e-12);
        assert!(family.crps(-1.0, theta).is_nan());
    }

    #[cfg(feature = "rand")]
    #[test]
    fn gamma_sampling_returns_finite_values_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = GammaMeanCv::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family.sample(&mut rng, GammaMeanCvTheta { mean: 1.5, cv: 0.7 });
        assert!(sample > 0.0 && sample.is_finite());
        assert!(
            family
                .sample(&mut rng, GammaMeanCvTheta { mean: 1.5, cv: 0.0 })
                .is_nan()
        );
    }
}
