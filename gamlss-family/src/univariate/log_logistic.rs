use std::marker::PhantomData;

use gamlss_core::{
    Family, HasCdf, HasCrps, HasQuantile, InitialEtaFromObservations, InitialEtaFromTheta, Log,
    ObservationView, ParameterParts, PositiveLink, Scale, Shape,
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};
use gamlss_special::{ln_beta, log_add_exp, regularized_beta};

use crate::domain::{is_positive_finite, is_probability};
use crate::initial::{positive_floor, weighted_summary, weighted_values};

/// Log-logistic distribution with scale/shape parameterization and log links.
pub type LogLogisticScaleShape = LogLogistic<Log, Log>;

/// Log-logistic family parameterized by positive scale and shape.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/distributions/log_logistic.svg")
)]
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogLogistic<ScaleLink = Log, ShapeLink = Log> {
    marker: PhantomData<(ScaleLink, ShapeLink)>,
}

impl<ScaleLink, ShapeLink> LogLogistic<ScaleLink, ShapeLink>
where
    ScaleLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    /// Creates a stateless log-logistic family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    fn theta_from_eta(eta: LogLogisticEta) -> LogLogisticTheta {
        eta.theta_from_links::<ScaleLink, ShapeLink>()
    }

    fn valid_theta(theta: LogLogisticTheta) -> bool {
        is_positive_finite(theta.scale) && is_positive_finite(theta.shape)
    }

    fn cdf_theta(y: f64, theta: LogLogisticTheta) -> f64 {
        if y <= 0.0 {
            return 0.0;
        }
        let value = theta.shape * (y.ln() - theta.scale.ln());
        if value >= 0.0 {
            1.0 / (1.0 + (-value).exp())
        } else {
            let exponential = value.exp();
            exponential / (1.0 + exponential)
        }
    }

    #[allow(clippy::suboptimal_flops)]
    fn nll_theta(y: f64, theta: LogLogisticTheta) -> f64 {
        if !is_positive_finite(y) || !Self::valid_theta(theta) {
            return f64::INFINITY;
        }
        let log_ratio = y.ln() - theta.scale.ln();
        -theta.shape.ln() + theta.scale.ln() - (theta.shape - 1.0) * log_ratio
            + 2.0 * log_add_exp(0.0, theta.shape * log_ratio)
    }

    #[allow(clippy::suboptimal_flops)]
    fn nll_and_gradient_eta_values(y: f64, eta: LogLogisticEta) -> (f64, LogLogisticEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (nll, LogLogisticEta::from_array([f64::NAN; 2]));
        }
        let log_ratio = y.ln() - theta.scale.ln();
        let cdf = Self::cdf_theta(y, theta);
        let d_scale = theta.shape * (1.0 - 2.0 * cdf) / theta.scale;
        let d_shape = -1.0 / theta.shape + (2.0 * cdf - 1.0) * log_ratio;
        (
            nll,
            eta.chain_gradient::<ScaleLink, ShapeLink>(d_scale, d_shape),
        )
    }
}

impl<ScaleLink, ShapeLink> Default for LogLogistic<ScaleLink, ShapeLink>
where
    ScaleLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<ScaleLink, ShapeLink> for LogLogistic<ScaleLink, ShapeLink>;
    parameters = (Scale, Shape);
    arity = 2;
);

impl<ScaleLink, ShapeLink> Family for LogLogistic<ScaleLink, ShapeLink>
where
    ScaleLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    type Eta = LogLogisticEta;
    type Theta = LogLogisticTheta;
    type GradientEta = LogLogisticEta;
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
        Self::nll_and_gradient_eta_values(y, *eta)
    }
}

impl<ScaleLink, ShapeLink> InitialEtaFromObservations<2> for LogLogistic<ScaleLink, ShapeLink>
where
    ScaleLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    ShapeLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = f64> + 'obs,
    {
        let values =
            weighted_values::<Self, _, _>(obs, |y| is_positive_finite(y).then_some(y.ln()));
        let summary = weighted_summary(&values);
        let (log_scale, shape) = summary.map_or((0.0, 1.0), |summary| {
            let shape = if summary.variance > 1.0e-12 {
                std::f64::consts::PI / (3.0 * summary.variance).sqrt()
            } else {
                1.0e6
            };
            (summary.mean, positive_floor(shape))
        });
        LogLogisticEta {
            scale: ScaleLink::initial_eta_from_theta(positive_floor(log_scale.exp())),
            shape: ShapeLink::initial_eta_from_theta(shape),
        }
    }
}

impl<ScaleLink, ShapeLink> HasCdf for LogLogistic<ScaleLink, ShapeLink>
where
    ScaleLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    fn cdf(&self, y: f64, theta: &Self::Theta) -> f64 {
        if !y.is_finite() || !Self::valid_theta(*theta) {
            return f64::NAN;
        }
        Self::cdf_theta(y, *theta)
    }
}

impl<ScaleLink, ShapeLink> HasQuantile for LogLogistic<ScaleLink, ShapeLink>
where
    ScaleLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    fn quantile(&self, probability: f64, theta: &Self::Theta) -> f64 {
        if !is_probability(probability) || !Self::valid_theta(*theta) {
            return f64::NAN;
        }
        theta.scale * ((probability.ln() - (-probability).ln_1p()) / theta.shape).exp()
    }
}

impl<ScaleLink, ShapeLink> HasCrps for LogLogistic<ScaleLink, ShapeLink>
where
    ScaleLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    #[allow(clippy::suboptimal_flops)]
    fn crps(&self, y: f64, theta: &Self::Theta) -> f64 {
        if y < 0.0 || !y.is_finite() || !Self::valid_theta(*theta) || theta.shape <= 1.0 {
            return f64::NAN;
        }

        let cdf = Self::cdf_theta(y, *theta);
        let inverse_shape = 1.0 / theta.shape;
        let first_beta_shape = 1.0 + inverse_shape;
        let second_beta_shape = 1.0 - inverse_shape;
        let beta = ln_beta(first_beta_shape, second_beta_shape).exp();
        let partial_first_moment =
            theta.scale * regularized_beta(first_beta_shape, second_beta_shape, cdf) * beta;
        let half_gini = theta.scale * second_beta_shape * beta;
        (y * (2.0 * cdf - 1.0) - 2.0 * partial_first_moment + half_gini).max(0.0)
    }
}

#[cfg(feature = "rand")]
impl<Rng, ScaleLink, ShapeLink> TrySimulate<Rng> for LogLogistic<ScaleLink, ShapeLink>
where
    Rng: rand::Rng,
    ScaleLink: PositiveLink<f64>,
    ShapeLink: PositiveLink<f64>,
{
    type Sample = f64;

    fn try_sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Result<f64, SimulationError> {
        if !Self::valid_theta(*theta) {
            return Err(SimulationError::InvalidParameters("Log-logistic theta"));
        }
        crate::simulation::try_sample_quantile(rng, self, theta, "Log-logistic sample")
    }
}

define_two_positive_parameter_blocks! {
    eta:
    /// Link-scale predictors for [`LogLogistic`].
    LogLogisticEta {
        /// Scale predictor.
        scale,
        /// Shape predictor.
        shape,
    }
    theta:
    /// Natural-scale parameters for [`LogLogistic`].
    LogLogisticTheta {
        /// Positive scale, equal to the median.
        scale,
        /// Positive shape.
        shape,
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{HasCdf, HasCrps, HasQuantile};

    use super::{LogLogisticScaleShape, LogLogisticTheta};
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn gradient_matches_finite_difference() {
        assert_gradient_matches_finite_difference::<_, 2>(
            &LogLogisticScaleShape::new(),
            1.7,
            [0.2, 0.4],
        );
    }

    #[test]
    fn scale_is_median_and_quantile_inverts_cdf() {
        let family = LogLogisticScaleShape::new();
        let theta = LogLogisticTheta {
            scale: 1.3,
            shape: 2.4,
        };
        assert_relative_eq!(family.cdf(theta.scale, &theta), 0.5, epsilon = 1.0e-14);
        assert_relative_eq!(
            family.quantile(family.cdf(1.7, &theta), &theta),
            1.7,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn log_logistic_crps_matches_reference_value_and_requires_finite_mean() {
        let family = LogLogisticScaleShape::new();
        assert_relative_eq!(
            family.crps(
                0.7,
                &LogLogisticTheta {
                    scale: 1.3,
                    shape: 2.4,
                },
            ),
            0.410_194_621_302_024_66,
            epsilon = 1.0e-12
        );
        assert!(
            family
                .crps(
                    0.7,
                    &LogLogisticTheta {
                        scale: 1.3,
                        shape: 1.0,
                    },
                )
                .is_nan()
        );
    }
}
