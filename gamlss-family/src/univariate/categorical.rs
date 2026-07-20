use gamlss_core::{
    CompilableFamily, Family, HasCdf, HasQuantile, ModelError, ObservationView, Probability,
    shape::{ShapeValues, Simplex},
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};
use gamlss_special::baseline_softmax;
#[cfg(feature = "rand")]
use rand::RngExt as _;

use crate::domain::{is_interior_simplex, is_probability};

/// Categorical family with `K` classes and `K - 1` baseline-softmax predictors.
///
/// Scalar observations encode categories as integers from `0` through `K - 1`.
/// CDF and quantile operations follow that numeric order; for nominal outcomes,
/// the ordering is therefore chosen by the caller.
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/distributions/categorical.svg")
)]
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Categorical<const K: usize> {
    private: (),
}

impl<const K: usize> Categorical<K> {
    /// Creates a family after checking that at least two categories exist.
    pub const fn try_new() -> Result<Self, ModelError> {
        if K < 2 {
            return Err(ModelError::InvalidParameter {
                parameter: "categorical category count",
                expected: "at least two",
            });
        }
        Ok(Self { private: () })
    }

    /// Creates a categorical family.
    ///
    /// # Panics
    ///
    /// Panics when `K < 2`.
    #[must_use]
    pub const fn new() -> Self {
        assert!(
            K >= 2,
            "categorical family requires at least two categories"
        );
        Self { private: () }
    }

    #[inline]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn category(observation: f64) -> Option<usize> {
        if observation >= 0.0 && observation.is_finite() && observation.fract() == 0.0 {
            let category = observation as usize;
            (category < K).then_some(category)
        } else {
            None
        }
    }

    #[inline]
    fn theta_from_eta(eta: &CategoricalEta<K>) -> CategoricalTheta<K> {
        CategoricalTheta {
            probabilities: baseline_softmax(eta.logits),
        }
    }

    #[inline]
    fn valid_theta(theta: &CategoricalTheta<K>) -> bool {
        is_interior_simplex(&theta.probabilities)
    }

    fn nll_theta(observation: f64, theta: &CategoricalTheta<K>) -> f64 {
        let Some(category) = Self::category(observation) else {
            return f64::INFINITY;
        };
        if !Self::valid_theta(theta) {
            return f64::INFINITY;
        }
        -theta.probabilities[category].ln()
    }
}

impl<const K: usize> Default for Categorical<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const K: usize> Family for Categorical<K> {
    type Eta = CategoricalEta<K>;
    type Theta = CategoricalTheta<K>;
    type GradientEta = CategoricalEta<K>;
    type Observation<'obs> = f64;
    type Workspace = ();

    fn workspace(&self) -> Self::Workspace {}
    fn theta(&self, eta: &Self::Eta, _workspace: &mut ()) -> Self::Theta {
        Self::theta_from_eta(eta)
    }
    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut ()) -> f64 {
        Self::nll_theta(y, theta)
    }
    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut (),
    ) -> (f64, Self::GradientEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, &theta);
        let Some(category) = Self::category(y) else {
            return (
                nll,
                CategoricalEta {
                    logits: [f64::NAN; K],
                },
            );
        };
        if !nll.is_finite() {
            return (
                nll,
                CategoricalEta {
                    logits: [f64::NAN; K],
                },
            );
        }
        let mut logits = theta.probabilities;
        if category < K - 1 {
            logits[category] -= 1.0;
        }
        logits[K - 1] = 0.0;
        (nll, CategoricalEta { logits })
    }
}

impl<const K: usize> CompilableFamily for Categorical<K> {
    type Shape = Simplex<Probability, K>;

    fn eta_from_shape(values: ShapeValues<Self::Shape>) -> Self::Eta {
        CategoricalEta { logits: values }
    }

    fn gradient_to_shape(gradient: &Self::GradientEta) -> ShapeValues<Self::Shape> {
        gradient.logits
    }

    fn initial_shape<'obs, Obs>(&self, obs: &'obs Obs) -> ShapeValues<Self::Shape>
    where
        Obs: ObservationView<'obs, Observation = f64> + 'obs,
    {
        let mut counts = [0.5; K];
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            if weight > 0.0
                && let Some(category) = Self::category(obs.observation_at(row))
            {
                counts[category] += weight;
            }
        }
        let baseline = counts[K - 1].ln();
        let mut logits = counts.map(|count| count.ln() - baseline);
        logits[K - 1] = 0.0;
        logits
    }

    fn validate_compiled(&self) -> Result<(), ModelError> {
        Self::try_new().map(|_| ())
    }
}

impl<const K: usize> HasCdf for Categorical<K> {
    #[allow(clippy::cast_precision_loss)]
    fn cdf(&self, observation: f64, theta: &Self::Theta) -> f64 {
        if !observation.is_finite() || !Self::valid_theta(theta) {
            return f64::NAN;
        }
        if observation < 0.0 {
            return 0.0;
        }
        theta
            .probabilities
            .iter()
            .copied()
            .enumerate()
            .take_while(|(category, _)| *category as f64 <= observation)
            .map(|(_, probability)| probability)
            .sum::<f64>()
            .min(1.0)
    }
}

impl<const K: usize> HasQuantile for Categorical<K> {
    #[allow(clippy::cast_precision_loss)]
    fn quantile(&self, probability: f64, theta: &Self::Theta) -> f64 {
        if !is_probability(probability) || !Self::valid_theta(theta) {
            return f64::NAN;
        }
        let mut cumulative = 0.0;
        for (category, mass) in theta.probabilities.iter().copied().enumerate() {
            cumulative += mass;
            if cumulative >= probability || category == K - 1 {
                return category as f64;
            }
        }
        f64::NAN
    }
}

#[cfg(feature = "rand")]
impl<Rng, const K: usize> TrySimulate<Rng> for Categorical<K>
where
    Rng: rand::Rng,
{
    type Sample = f64;

    #[allow(clippy::cast_precision_loss)]
    fn try_sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Result<f64, SimulationError> {
        if !Self::valid_theta(theta) {
            return Err(SimulationError::InvalidParameters("Categorical theta"));
        }
        let draw: f64 = rng.random();
        let mut cumulative = 0.0;
        for (category, probability) in theta.probabilities.iter().copied().enumerate() {
            cumulative += probability;
            if draw < cumulative || category == K - 1 {
                return Ok(category as f64);
            }
        }
        Err(SimulationError::NumericalFailure(
            "Categorical cumulative probability",
        ))
    }
}

/// Baseline-softmax predictors for [`Categorical`].
///
/// The last array entry is a structural zero retained by the fixed-size carrier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CategoricalEta<const K: usize> {
    /// `K - 1` free logits followed by the zero baseline slot.
    pub logits: [f64; K],
}

impl<const K: usize> CategoricalEta<K> {
    /// Creates a predictor carrier and enforces a zero baseline slot.
    #[must_use]
    pub const fn new(mut logits: [f64; K]) -> Self {
        if K > 0 {
            logits[K - 1] = 0.0;
        }
        Self { logits }
    }
}

/// Natural-scale categorical probabilities for observations encoded as `0..K - 1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CategoricalTheta<const K: usize> {
    /// Strictly positive probabilities summing to one.
    pub probabilities: [f64; K],
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use approx::assert_relative_eq;
    use gamlss_core::{
        CompilableFamily, DenseDesign, Family, Gamlss, HasCdf, HasQuantile, LinearPredictorBlock,
        NoPenalty, ParameterBlocks, Probability, SimplexLogitParameterBlock,
    };

    use super::{Categorical, CategoricalEta, CategoricalTheta};

    #[test]
    fn gradient_matches_finite_difference() {
        let family = Categorical::<3>::new();
        let eta = CategoricalEta::new([0.3, -0.4, 0.0]);
        let (_, gradient) = family.nll_and_gradient_eta(1.0, &eta, &mut ());
        for index in 0..2 {
            let epsilon = 1.0e-6;
            let mut lower = eta;
            let mut upper = eta;
            lower.logits[index] -= epsilon;
            upper.logits[index] += epsilon;
            let numeric = (family.nll_eta(1.0, &upper, &mut ())
                - family.nll_eta(1.0, &lower, &mut ()))
                / (2.0 * epsilon);
            assert_relative_eq!(gradient.logits[index], numeric, epsilon = 1.0e-8);
        }
        assert_eq!(gradient.logits[2], 0.0);
    }

    #[test]
    fn extreme_finite_logits_keep_likelihood_and_gradient_finite() {
        let family = Categorical::<3>::new();
        let eta = CategoricalEta::new([f64::MAX, -f64::MAX, 0.0]);
        let (nll, gradient) = family.nll_and_gradient_eta(1.0, &eta, &mut ());
        assert!(nll.is_finite());
        assert!(gradient.logits.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn initialization_uses_empirical_class_probabilities() {
        let family = Categorical::<3>::new();
        let observations: &[f64] = &[0.0, 0.0, 1.0, 2.0];
        let logits = family.initial_shape(&observations);
        assert!(logits[0] > logits[1]);
        assert_relative_eq!(logits[1], logits[2], epsilon = 1.0e-14);
        let theta = family.theta(&CategoricalEta::new(logits), &mut ());
        assert!(family.nll(3.0, &theta, &mut ()).is_infinite());
    }

    #[test]
    fn cdf_and_quantile_follow_category_order() {
        let family = Categorical::<3>::new();
        let theta = CategoricalTheta {
            probabilities: [0.2, 0.3, 0.5],
        };
        assert_relative_eq!(family.cdf(0.0, &theta), 0.2, epsilon = 1.0e-14);
        assert_relative_eq!(family.cdf(1.7, &theta), 0.5, epsilon = 1.0e-14);
        assert_relative_eq!(family.cdf(2.0, &theta), 1.0, epsilon = 1.0e-14);
        assert_eq!(family.quantile(0.2, &theta), 0.0);
        assert_eq!(family.quantile(0.21, &theta), 1.0);
        assert_eq!(family.quantile(1.0, &theta), 2.0);
    }

    #[test]
    fn compiled_family_uses_two_predictors_for_three_categories() {
        let observations = [0.0, 1.0, 2.0, 1.0];
        let logits = SimplexLogitParameterBlock::<Probability, 3, _, _>::new(
            vec![
                LinearPredictorBlock::new(DenseDesign::intercept(observations.len())),
                LinearPredictorBlock::new(DenseDesign::intercept(observations.len())),
            ],
            NoPenalty,
            0,
        );
        let model = Gamlss::try_new(
            Categorical::<3>::new(),
            ParameterBlocks::new(logits),
            &observations,
        )
        .unwrap();
        let beta = [0.2, -0.1];
        let mut gradient = [0.0; 2];
        model.try_value_gradient_into(&beta, &mut gradient).unwrap();
        assert_eq!(model.nparams(), 2);
        for index in 0..2 {
            let mut lower = beta;
            let mut upper = beta;
            lower[index] -= 1.0e-6;
            upper[index] += 1.0e-6;
            let numeric =
                (model.try_value(&upper).unwrap() - model.try_value(&lower).unwrap()) / 2.0e-6;
            assert_relative_eq!(gradient[index], numeric, epsilon = 1.0e-8);
        }
    }
}
