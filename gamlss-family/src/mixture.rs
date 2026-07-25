//! Homogeneous finite-mixture families.

use gamlss_core::{
    CompilableFamily, Family, MixtureWeight, ModelError, ObservationView,
    shape::{ParameterShape, Product, Repeated, ShapeValues, Simplex},
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};
#[cfg(feature = "rand")]
use rand::RngExt as _;

/// Homogeneous fixed-size mixture of one component family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mixture<F, const C: usize> {
    component: F,
}

impl<F, const C: usize> Mixture<F, C> {
    /// Creates a homogeneous mixture family.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `C < 2`.
    #[inline]
    pub fn try_new(component: F) -> Result<Self, ModelError> {
        if C < 2 {
            return Err(ModelError::InvalidParameter {
                parameter: "components",
                expected: "at least two mixture components",
            });
        }
        Ok(Self { component })
    }

    /// Returns the shared component family.
    #[must_use]
    #[inline]
    pub const fn component(&self) -> &F {
        &self.component
    }
}

#[cfg(feature = "rand")]
impl<Rng, F, const C: usize> TrySimulate<Rng> for Mixture<F, C>
where
    Rng: rand::Rng,
    F: TrySimulate<Rng>,
    for<'obs> F::Observation<'obs>: Clone,
{
    type Sample = F::Sample;

    fn try_sample(
        &self,
        rng: &mut Rng,
        theta: &Self::Theta,
    ) -> Result<Self::Sample, SimulationError> {
        let selected = try_select_component(rng, &theta.weights)?;
        self.component.try_sample(rng, &theta.components[selected])
    }

    fn try_sample_into(
        &self,
        rng: &mut Rng,
        theta: &Self::Theta,
        out: &mut Self::Sample,
    ) -> Result<(), SimulationError> {
        let selected = try_select_component(rng, &theta.weights)?;
        self.component
            .try_sample_into(rng, &theta.components[selected], out)
    }
}

impl<F, const C: usize> Family for Mixture<F, C>
where
    F: Family,
    for<'obs> F::Observation<'obs>: Clone,
{
    type Eta = MixtureEta<F::Eta, C>;
    type Theta = MixtureTheta<F::Theta, C>;
    type GradientEta = MixtureGradient<F::GradientEta, C>;
    type Observation<'obs> = F::Observation<'obs>;
    type Workspace = MixtureWorkspace<F::Workspace, C>;

    fn workspace(&self) -> Self::Workspace {
        MixtureWorkspace {
            components: std::array::from_fn(|_| self.component.workspace()),
        }
    }

    fn theta(&self, eta: &Self::Eta, workspace: &mut Self::Workspace) -> Self::Theta {
        MixtureTheta {
            weights: softmax_baseline(eta.logits),
            components: std::array::from_fn(|index| {
                self.component
                    .theta(&eta.components[index], &mut workspace.components[index])
            }),
        }
    }

    fn nll(
        &self,
        observation: Self::Observation<'_>,
        theta: &Self::Theta,
        workspace: &mut Self::Workspace,
    ) -> f64 {
        if C < 2 || !valid_weights(&theta.weights) {
            return f64::INFINITY;
        }

        let terms: [f64; C] = std::array::from_fn(|index| {
            if theta.weights[index] == 0.0 {
                return f64::NEG_INFINITY;
            }
            let nll = self.component.nll(
                observation.clone(),
                &theta.components[index],
                &mut workspace.components[index],
            );
            theta.weights[index].ln() - nll
        });
        -log_sum_exp(&terms)
    }

    fn nll_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        workspace: &mut Self::Workspace,
    ) -> f64 {
        if C < 2 {
            return f64::INFINITY;
        }

        let log_weights = log_softmax_baseline(eta.logits);
        let terms: [f64; C] = std::array::from_fn(|index| {
            if log_weights[index] == f64::NEG_INFINITY {
                return f64::NEG_INFINITY;
            }
            let nll = self.component.nll_eta(
                observation.clone(),
                &eta.components[index],
                &mut workspace.components[index],
            );
            log_weights[index] - nll
        });
        -log_sum_exp(&terms)
    }

    fn nll_and_gradient_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        if C < 2 {
            return (
                f64::INFINITY,
                MixtureGradient::from_parts(
                    [f64::NAN; C],
                    std::array::from_fn(|index| {
                        let (_, gradient) = self.component.nll_and_gradient_eta(
                            observation.clone(),
                            &eta.components[index],
                            &mut workspace.components[index],
                        );
                        FactorizedComponentGradient::from_parts(f64::NAN, gradient)
                    }),
                ),
            );
        }

        let log_weights = log_softmax_baseline(eta.logits);
        let weights = log_weights.map(f64::exp);
        let mut component_nll = [0.0; C];
        let component_gradients: [F::GradientEta; C] = std::array::from_fn(|index| {
            let (nll, gradient) = self.component.nll_and_gradient_eta(
                observation.clone(),
                &eta.components[index],
                &mut workspace.components[index],
            );
            component_nll[index] = nll;
            gradient
        });

        let terms: [f64; C] =
            std::array::from_fn(|index| log_weights[index] - component_nll[index]);
        let log_mix = log_sum_exp(&terms);
        let nll = -log_mix;
        let responsibilities = terms.map(|term| (term - log_mix).exp());

        let mut logits = [0.0; C];
        for ((logit, weight), responsibility) in logits
            .iter_mut()
            .zip(weights.iter().copied())
            .zip(responsibilities.iter().copied())
            .take(C.saturating_sub(1))
        {
            *logit = weight - responsibility;
        }

        let mut responsibilities = responsibilities.into_iter();
        (
            nll,
            MixtureGradient::from_parts(
                logits,
                component_gradients.map(|gradient| {
                    FactorizedComponentGradient::from_parts(
                        responsibilities
                            .next()
                            .expect("responsibility count matches component count"),
                        gradient,
                    )
                }),
            ),
        )
    }
}

impl<F, const C: usize> CompilableFamily for Mixture<F, C>
where
    F: CompilableFamily,
    for<'obs> F::Observation<'obs>: Clone,
{
    type Shape = Product<Simplex<MixtureWeight, C>, Repeated<F::Shape, C>>;

    fn eta_from_shape(values: ShapeValues<Self::Shape>) -> <Self as Family>::Eta {
        MixtureEta::new(values.0, values.1.map(F::eta_from_shape))
    }

    fn gradient_to_shape(gradient: &<Self as Family>::GradientEta) -> ShapeValues<Self::Shape> {
        let components = std::array::from_fn(|index| {
            let component = &gradient.components()[index];
            let mut values = F::gradient_to_shape(component.conditional_gradient());
            F::Shape::scale(&mut values, component.responsibility());
            values
        });
        (*gradient.logits(), components)
    }

    fn initial_shape<'obs, Obs>(&self, obs: &'obs Obs) -> ShapeValues<Self::Shape>
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        #[allow(clippy::cast_precision_loss)]
        let center = (C.saturating_sub(1) as f64) * 0.5;
        let components = std::array::from_fn(|index| {
            let mut values = self.component.initial_shape(obs);
            #[allow(clippy::cast_precision_loss)]
            F::Shape::add_to_first(&mut values, (index as f64 - center) * 0.5);
            values
        });
        ([0.0; C], components)
    }

    fn validate_compiled(&self) -> Result<(), ModelError> {
        if C < 2 {
            return Err(ModelError::InvalidParameter {
                parameter: "components",
                expected: "at least two mixture components",
            });
        }
        self.component.validate_compiled()
    }
}

/// Link-scale predictors for a homogeneous fixed-size mixture.
#[derive(Debug, Clone, PartialEq)]
pub struct MixtureEta<Eta, const C: usize> {
    /// Baseline-softmax logits. The last entry is normalized to zero.
    pub logits: [f64; C],
    /// Component link-scale predictors.
    pub components: [Eta; C],
}

impl<Eta, const C: usize> MixtureEta<Eta, C> {
    /// Creates mixture predictors and normalizes the baseline logit to zero.
    #[must_use]
    pub const fn new(mut logits: [f64; C], components: [Eta; C]) -> Self {
        if C > 0 {
            logits[C - 1] = 0.0;
        }
        Self { logits, components }
    }
}

/// Natural-scale parameters for a homogeneous fixed-size mixture.
#[derive(Debug, Clone, PartialEq)]
pub struct MixtureTheta<Theta, const C: usize> {
    /// Normalized mixture weights.
    weights: [f64; C],
    /// Component natural-scale parameters.
    components: [Theta; C],
}

impl<Theta, const C: usize> MixtureTheta<Theta, C> {
    /// Creates checked natural-scale mixture parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] unless there are at least two
    /// components and `weights` form a finite non-negative simplex.
    pub fn try_new(weights: [f64; C], components: [Theta; C]) -> Result<Self, ModelError> {
        if C < 2 || !valid_weights(&weights) {
            return Err(ModelError::InvalidParameter {
                parameter: "mixture weights",
                expected: "a finite non-negative simplex for at least two components",
            });
        }
        Ok(Self {
            weights,
            components,
        })
    }

    /// Normalized mixture weights.
    #[must_use]
    pub const fn weights(&self) -> &[f64; C] {
        &self.weights
    }

    /// Natural-scale parameters in component order.
    #[must_use]
    pub const fn components(&self) -> &[Theta; C] {
        &self.components
    }

    /// Consumes the carrier and returns weights and component parameters.
    #[must_use]
    pub fn into_parts(self) -> ([f64; C], [Theta; C]) {
        (self.weights, self.components)
    }
}

/// Factorized exact gradient for one mixture component.
///
/// For a component parameter `eta_c`, the mixture NLL derivative is the
/// posterior responsibility multiplied by the conditional component
/// derivative. Keeping those factors separate avoids imposing an algebraic
/// scaling trait on every family-specific gradient carrier. The
/// [`CompilableFamily`] codec for [`Mixture`] materializes their product before
/// the score reaches predictor blocks.
#[derive(Debug, Clone, PartialEq)]
pub struct FactorizedComponentGradient<Gradient> {
    responsibility: f64,
    conditional_gradient: Gradient,
}

impl<Gradient> FactorizedComponentGradient<Gradient> {
    #[inline]
    const fn from_parts(responsibility: f64, conditional_gradient: Gradient) -> Self {
        Self {
            responsibility,
            conditional_gradient,
        }
    }

    /// Posterior component responsibility for the current observation.
    #[must_use]
    #[inline]
    pub const fn responsibility(&self) -> f64 {
        self.responsibility
    }

    /// Conditional component NLL gradient before responsibility scaling.
    #[must_use]
    #[inline]
    pub const fn conditional_gradient(&self) -> &Gradient {
        &self.conditional_gradient
    }

    /// Consumes the carrier and returns its responsibility and conditional gradient.
    #[must_use]
    #[inline]
    pub fn into_parts(self) -> (f64, Gradient) {
        (self.responsibility, self.conditional_gradient)
    }
}

/// Link-scale NLL gradient for a homogeneous fixed-size mixture.
#[derive(Debug, Clone, PartialEq)]
pub struct MixtureGradient<Gradient, const C: usize> {
    /// Gradients for baseline-softmax logits. The last baseline entry is zero.
    logits: [f64; C],
    /// Exact component gradients represented as responsibility/conditional-gradient factors.
    components: [FactorizedComponentGradient<Gradient>; C],
}

impl<Gradient, const C: usize> MixtureGradient<Gradient, C> {
    #[inline]
    const fn from_parts(
        logits: [f64; C],
        components: [FactorizedComponentGradient<Gradient>; C],
    ) -> Self {
        Self { logits, components }
    }

    /// Gradients for baseline-softmax logits in component order.
    #[must_use]
    #[inline]
    pub const fn logits(&self) -> &[f64; C] {
        &self.logits
    }

    /// Factorized exact gradients in component order.
    #[must_use]
    #[inline]
    pub const fn components(&self) -> &[FactorizedComponentGradient<Gradient>; C] {
        &self.components
    }

    /// Consumes the carrier and returns its logit and component gradients.
    #[must_use]
    #[inline]
    pub fn into_parts(self) -> ([f64; C], [FactorizedComponentGradient<Gradient>; C]) {
        (self.logits, self.components)
    }
}

/// Reusable per-component family workspaces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixtureWorkspace<W, const C: usize> {
    components: [W; C],
}

fn softmax_baseline<const C: usize>(logits: [f64; C]) -> [f64; C] {
    log_softmax_baseline(logits).map(f64::exp)
}

fn log_softmax_baseline<const C: usize>(mut logits: [f64; C]) -> [f64; C] {
    if C == 0 {
        return logits;
    }
    logits[C - 1] = 0.0;
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for logit in logits {
        sum += (logit - max).exp();
    }
    let log_normalizer = max + sum.ln();
    logits.map(|logit| logit - log_normalizer)
}

fn valid_weights(weights: &[f64]) -> bool {
    let sum = weights.iter().sum::<f64>();
    let tolerance = 16.0 * f64::EPSILON * weights.iter().fold(0.0, |count, _| count + 1.0);
    weights
        .iter()
        .all(|weight| weight.is_finite() && *weight >= 0.0)
        && sum.is_finite()
        && sum > 0.0
        && (sum - 1.0).abs() <= tolerance
}

#[cfg(feature = "rand")]
fn try_select_component<Rng, const C: usize>(
    rng: &mut Rng,
    weights: &[f64; C],
) -> Result<usize, SimulationError>
where
    Rng: rand::Rng,
{
    if C < 2 || !valid_weights(weights) {
        return Err(SimulationError::InvalidParameters("mixture weights"));
    }

    let draw = rng.random::<f64>();
    let mut cumulative = 0.0;
    for (index, weight) in weights.iter().copied().enumerate() {
        cumulative += weight;
        if draw < cumulative {
            return Ok(index);
        }
    }

    // Valid weights sum to one within rounding tolerance. Assign the tiny
    // residual interval, if any, to the final component.
    Ok(C - 1)
}

fn log_sum_exp(values: &[f64]) -> f64 {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return max;
    }
    max + values
        .iter()
        .map(|value| (*value - max).exp())
        .sum::<f64>()
        .ln()
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{
        CompilableFamily, DenseDesign, Family, Gamlss, MixtureWeight, ModelError, Mu, NoPenalty,
        Objective, ParameterAxis, ParameterBlock, ParameterBlocks, ParameterPath, Sigma,
        SimplexLogitParameterBlock,
    };

    use super::{Mixture, MixtureEta, MixtureTheta};
    #[cfg(feature = "rand")]
    use crate::{BernoulliProbability, BernoulliTheta};
    use crate::{ExponentialRate, ExponentialRateEta, NormalEta, NormalMuSigma, NormalTheta};

    #[test]
    fn rejects_less_than_two_components() {
        assert!(Mixture::<_, 1>::try_new(NormalMuSigma::new()).is_err());
    }

    #[test]
    fn nll_matches_manual_log_sum_exp() {
        let family = Mixture::<_, 2>::try_new(NormalMuSigma::new()).unwrap();
        let eta = mixture_eta();
        let y = 0.4;
        let mut workspace = family.workspace();
        let nll = family.nll_eta(y, &eta, &mut workspace);

        let normal = NormalMuSigma::new();
        let weights = [
            eta.logits[0].exp() / (eta.logits[0].exp() + 1.0),
            1.0 / (eta.logits[0].exp() + 1.0),
        ];
        let terms = [
            weights[0].ln() - normal.nll_eta(y, &eta.components[0], &mut ()),
            weights[1].ln() - normal.nll_eta(y, &eta.components[1], &mut ()),
        ];
        let manual = -(terms[0].max(terms[1])
            + ((terms[0] - terms[0].max(terms[1])).exp()
                + (terms[1] - terms[0].max(terms[1])).exp())
            .ln());

        assert_relative_eq!(nll, manual, epsilon = 1.0e-12);
    }

    #[test]
    fn natural_weights_must_be_a_simplex_and_allow_zero_weight_components() {
        let family = Mixture::<_, 2>::try_new(NormalMuSigma::new()).unwrap();
        let valid_component = NormalTheta {
            mu: 0.0,
            sigma: 1.0,
        };
        let invalid_component = NormalTheta {
            mu: f64::NAN,
            sigma: f64::NAN,
        };
        let unnormalized = MixtureTheta {
            weights: [1.0, 1.0],
            components: [valid_component, invalid_component],
        };
        let zero_weight = MixtureTheta {
            weights: [1.0, 0.0],
            components: [valid_component, invalid_component],
        };

        assert!(
            family
                .nll(0.25, &unnormalized, &mut family.workspace())
                .is_infinite()
        );
        assert!(
            family
                .nll(0.25, &zero_weight, &mut family.workspace())
                .is_finite()
        );

        let checked = MixtureTheta::try_new([0.25, 0.75], [valid_component; 2]).unwrap();
        assert_relative_eq!(checked.weights()[0], 0.25);
        assert_eq!(checked.components().len(), 2);
        assert!(MixtureTheta::try_new([1.0, 1.0], [valid_component; 2]).is_err());
    }

    #[cfg(feature = "rand")]
    #[test]
    fn fallible_mixture_sampling_rejects_invalid_weights_without_panicking() {
        use gamlss_core::TrySimulate;
        use rand::SeedableRng;

        let family = Mixture::<_, 2>::try_new(NormalMuSigma::new()).unwrap();
        let components = [
            NormalTheta {
                mu: -1.0,
                sigma: 1.0,
            },
            NormalTheta {
                mu: 1.0,
                sigma: 1.0,
            },
        ];
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        assert!(
            family
                .try_sample(
                    &mut rng,
                    &MixtureTheta {
                        weights: [0.4, 0.6],
                        components,
                    },
                )
                .unwrap()
                .is_finite()
        );
        assert!(
            family
                .try_sample(
                    &mut rng,
                    &MixtureTheta {
                        weights: [f64::NAN, 0.0],
                        components,
                    },
                )
                .is_err()
        );
    }

    #[cfg(feature = "rand")]
    #[test]
    #[allow(clippy::float_cmp)]
    fn bernoulli_mixture_samples_and_fills_caller_storage() {
        use gamlss_core::TrySimulate;
        use rand::SeedableRng;

        let family = Mixture::<_, 2>::try_new(BernoulliProbability::new()).unwrap();
        let theta = MixtureTheta {
            weights: [0.4, 0.6],
            components: [BernoulliTheta { mu: 0.2 }, BernoulliTheta { mu: 0.8 }],
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(19);
        let mut samples = [f64::NAN; 32];

        family.try_fill(&mut rng, &theta, &mut samples).unwrap();

        assert!(
            samples
                .iter()
                .all(|sample| *sample == 0.0 || *sample == 1.0)
        );

        let invalid_theta = MixtureTheta {
            weights: [0.4, 0.5],
            components: theta.components,
        };
        let mut out = 7.0;
        assert_eq!(
            family.try_sample_into(&mut rng, &invalid_theta, &mut out),
            Err(gamlss_core::SimulationError::InvalidParameters(
                "mixture weights"
            ))
        );
        assert_eq!(out, 7.0);
    }

    #[test]
    fn extreme_logits_preserve_natural_scale_nll_parity() {
        let family = Mixture::<_, 2>::try_new(NormalMuSigma::new()).unwrap();
        let eta = MixtureEta::new(
            [1_000.0, 0.0],
            [
                NormalEta {
                    mu: 0.0,
                    sigma: 0.0,
                },
                NormalEta {
                    mu: 2.0,
                    sigma: 0.0,
                },
            ],
        );
        let theta = family.theta(&eta, &mut family.workspace());

        assert!((theta.weights[0] - 1.0).abs() <= f64::EPSILON);
        assert!(theta.weights[1].abs() <= f64::EPSILON);
        assert_relative_eq!(
            family.nll_eta(0.25, &eta, &mut family.workspace()),
            family.nll(0.25, &theta, &mut family.workspace()),
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn eta_gradient_keeps_underflowed_log_weight_when_likelihood_compensates_for_it() {
        let family = Mixture::<_, 2>::try_new(ExponentialRate::new()).unwrap();
        let eta = MixtureEta::new(
            [-750.0, 0.0],
            [
                ExponentialRateEta { rate: 100.0 },
                ExponentialRateEta { rate: -700.0 },
            ],
        );

        let expected = family.nll_eta(0.0, &eta, &mut family.workspace());
        let (actual, gradient) = family.nll_and_gradient_eta(0.0, &eta, &mut family.workspace());

        assert_relative_eq!(actual, expected, epsilon = 1.0e-12);
        assert!(gradient.components()[0].responsibility() > 0.999);
        assert!(gradient.logits()[0] < -0.999);
    }

    #[test]
    fn gradients_match_finite_differences() {
        let family = Mixture::<_, 2>::try_new(NormalMuSigma::new()).unwrap();
        let eta = mixture_eta();
        let y = 0.4;
        let mut workspace = family.workspace();
        let (_, gradient) = family.nll_and_gradient_eta(y, &eta, &mut workspace);

        let fd_logit = finite_difference(
            family,
            &eta,
            y,
            |probe, value| {
                probe.logits[0] = value;
            },
            eta.logits[0],
        );
        assert_relative_eq!(gradient.logits()[0], fd_logit, epsilon = 1.0e-6);
        assert_relative_eq!(gradient.logits()[1], 0.0, epsilon = 1.0e-12);

        let fd_mu0 = finite_difference(
            family,
            &eta,
            y,
            |probe, value| {
                probe.components[0].mu = value;
            },
            eta.components[0].mu,
        );
        let factorized = &gradient.components()[0];
        assert_relative_eq!(
            factorized.responsibility() * factorized.conditional_gradient().mu,
            fd_mu0,
            epsilon = 1.0e-6
        );
        let materialized =
            <Mixture<NormalMuSigma, 2> as CompilableFamily>::gradient_to_shape(&gradient);
        assert_relative_eq!(materialized.1[0][0], fd_mu0, epsilon = 1.0e-6);
    }

    #[test]
    fn compiled_normal_mixture_has_nested_layout_and_beta_gradient() {
        let y = [-1.0, -0.2, 0.5, 1.7];
        let n = y.len();
        let weights = SimplexLogitParameterBlock::<MixtureWeight, 2, _, _>::new(
            vec![gamlss_core::LinearPredictorBlock::new(
                DenseDesign::from_rows(&[[1.0, -1.0], [1.0, -0.25], [1.0, 0.25], [1.0, 1.0]]),
            )],
            NoPenalty,
            99,
        );
        let components = std::array::from_fn::<_, 2, _>(|_| {
            (
                ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(n), NoPenalty, 99),
                ParameterBlock::<Sigma, _, _>::linear(DenseDesign::intercept(n), NoPenalty, 99),
            )
        });
        let blocks = ParameterBlocks::new((weights, components));
        let family = Mixture::<_, 2>::try_new(NormalMuSigma::new()).unwrap();
        let mut model = Gamlss::try_new(family, blocks, &y).unwrap();
        let beta = [0.2, 0.7, -0.8, -0.1, 0.9, 0.2];
        let mut gradient = [0.0; 6];

        model.try_value_gradient_into(&beta, &mut gradient).unwrap();
        for index in 0..beta.len() {
            let mut plus = beta;
            let mut minus = beta;
            plus[index] += 1.0e-6;
            minus[index] -= 1.0e-6;
            let finite_difference =
                (model.try_value(&plus).unwrap() - model.try_value(&minus).unwrap()) / 2.0e-6;
            assert_relative_eq!(gradient[index], finite_difference, epsilon = 2.0e-5);
        }

        let descriptors = model.parameter_descriptors();
        assert_eq!(
            descriptors[0].path.axes(),
            &[ParameterAxis::SimplexLogit { class: 0 }]
        );
        assert_eq!(
            descriptors[1].path.axes(),
            &[ParameterAxis::Component { index: 0 }]
        );
        assert_eq!(descriptors[1].role, "mu");
        assert_eq!(
            descriptors[3].path.axes(),
            &[ParameterAxis::Component { index: 1 }]
        );

        assert_eq!(model.parameter_layout().ranges_of::<Mu>(), vec![2..3, 4..5]);
        assert_eq!(
            model.block_objective_for::<Mu>(beta.to_vec()).unwrap_err(),
            ModelError::AmbiguousParameter {
                name: "mu".to_owned(),
                matches: 2,
            }
        );
        let component_one_path = ParameterPath::from_axis(ParameterAxis::Component { index: 1 });
        let (descriptor_index, component_one_mu) = model
            .unique_parameter_descriptor_at_path::<Mu>(&component_one_path)
            .unwrap()
            .unwrap();
        assert_eq!(descriptor_index, 3);
        assert_eq!(component_one_mu, descriptors[3]);

        let unpacked = model.unpack_parameters(&beta).unwrap();
        assert_eq!(unpacked.blocks_of::<Mu>().count(), 2);
        assert_eq!(
            unpacked.block_at(descriptor_index).unwrap().descriptor,
            component_one_mu
        );
        assert_eq!(
            unpacked.block_at(descriptor_index).unwrap().coefficients,
            beta[4..5]
        );
        assert_eq!(
            unpacked.unique_block_of::<Mu>().unwrap_err(),
            ModelError::AmbiguousParameter {
                name: "mu".to_owned(),
                matches: 2,
            }
        );
        {
            let mut component_objective = model
                .block_objective_at(descriptor_index, beta.to_vec())
                .unwrap();
            let mut component_gradient = [0.0];
            component_objective
                .value_gradient(&beta[4..5], &mut component_gradient)
                .unwrap();
            assert_relative_eq!(component_gradient[0], gradient[4], epsilon = 1.0e-12);
        }

        let starts = model.initial_parameters().unwrap();
        assert!((starts[2] - starts[4]).abs() > f64::EPSILON);
    }

    #[cfg(feature = "multivariate")]
    #[test]
    fn compiled_mvn_mixture_reuses_repeated_product_executor() {
        use gamlss_core::{
            CholeskyScale, LinearPredictorBlock, LowerTriangularParameterBlock,
            VectorParameterBlock,
        };

        let y = [[-1.0, 0.2], [0.1, -0.4], [1.2, 0.8]];
        let n = y.len();
        let weights = SimplexLogitParameterBlock::<MixtureWeight, 2, _, _>::new(
            vec![LinearPredictorBlock::new(DenseDesign::from_rows(&[
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]))],
            NoPenalty,
            99,
        );
        let components = std::array::from_fn::<_, 2, _>(|_| {
            (
                VectorParameterBlock::<Mu, 2, _, _>::new(
                    [
                        LinearPredictorBlock::new(DenseDesign::intercept(n)),
                        LinearPredictorBlock::new(DenseDesign::intercept(n)),
                    ],
                    NoPenalty,
                    99,
                ),
                LowerTriangularParameterBlock::<CholeskyScale, 2, _, _>::new(
                    vec![
                        LinearPredictorBlock::new(DenseDesign::intercept(n)),
                        LinearPredictorBlock::new(DenseDesign::intercept(n)),
                        LinearPredictorBlock::new(DenseDesign::intercept(n)),
                    ],
                    NoPenalty,
                    99,
                ),
            )
        });
        let family = Mixture::<_, 2>::try_new(crate::MvNormalCholeskyDefault::<2>::new()).unwrap();
        let model = Gamlss::try_new_with_observations(
            family,
            ParameterBlocks::new((weights, components)),
            y.as_slice(),
        )
        .unwrap();
        let beta = [
            0.2, 0.4, -0.3, -0.1, 0.15, 0.2, -0.5, 0.6, 0.1, -0.2, -0.1, 0.25,
        ];
        let mut gradient = [0.0; 12];

        model
            .try_likelihood_value_gradient_into(&beta, &mut gradient)
            .unwrap();
        for index in 0..beta.len() {
            let mut plus = beta;
            let mut minus = beta;
            plus[index] += 1.0e-6;
            minus[index] -= 1.0e-6;
            let finite_difference = (model.try_likelihood_value(&plus).unwrap()
                - model.try_likelihood_value(&minus).unwrap())
                / 2.0e-6;
            assert_relative_eq!(gradient[index], finite_difference, epsilon = 3.0e-5);
        }

        let descriptors = model.parameter_descriptors();
        assert_eq!(
            descriptors[1].path.axes(),
            &[
                ParameterAxis::Component { index: 0 },
                ParameterAxis::Vector { component: 0 },
            ]
        );
        assert_eq!(
            descriptors[3].path.axes(),
            &[
                ParameterAxis::Component { index: 0 },
                ParameterAxis::Lower { row: 0, col: 0 },
            ]
        );
    }

    fn mixture_eta() -> MixtureEta<NormalEta, 2> {
        MixtureEta::new(
            [0.7, 12.0],
            [
                NormalEta {
                    mu: -0.3,
                    sigma: -0.2,
                },
                NormalEta {
                    mu: 1.1,
                    sigma: 0.4,
                },
            ],
        )
    }

    fn finite_difference(
        family: Mixture<NormalMuSigma, 2>,
        eta: &MixtureEta<NormalEta, 2>,
        y: f64,
        mut set: impl FnMut(&mut MixtureEta<NormalEta, 2>, f64),
        center: f64,
    ) -> f64 {
        let step = 1.0e-6;
        let mut plus = eta.clone();
        let mut minus = eta.clone();
        set(&mut plus, center + step);
        set(&mut minus, center - step);
        let mut workspace = family.workspace();
        let plus_nll = family.nll_eta(y, &plus, &mut workspace);
        let minus_nll = family.nll_eta(y, &minus, &mut workspace);
        (plus_nll - minus_nll) / (2.0 * step)
    }
}
