#![allow(clippy::needless_range_loop, clippy::suboptimal_flops)]

use std::marker::PhantomData;

use gamlss_core::{
    CompilableFamily, Family, FixedDimensionalFamily, HasObservationDimension, InitialEtaFromTheta,
    Log, Mean, ModelError, ObservationView, PositiveLink, Precision,
    shape::{Product, Scalar, ShapeValues, Simplex},
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};
use gamlss_special::{baseline_softmax, digamma_minus_ln, ln_gamma_stirling_residual};

const SIMPLEX_TOLERANCE: f64 = 1.0e-8;

/// Dirichlet distribution parameterized by simplex mean and precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirichletMeanPrecision<const D: usize, PrecisionLink = Log> {
    marker: PhantomData<PrecisionLink>,
}

impl<const D: usize, PrecisionLink> DirichletMeanPrecision<D, PrecisionLink>
where
    PrecisionLink: PositiveLink<f64>,
{
    /// Creates a stateless family value after checking the compile-time dimension.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `D < 2`.
    #[inline]
    pub const fn try_new() -> Result<Self, ModelError> {
        if D < 2 {
            return Err(ModelError::InvalidParameter {
                parameter: "dimension",
                expected: "at least two",
            });
        }
        Ok(Self {
            marker: PhantomData,
        })
    }

    /// Creates a stateless Dirichlet mean/precision family.
    ///
    /// # Panics
    ///
    /// Panics when the compile-time dimension is less than two.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        assert!(D >= 2, "Dirichlet dimension must be at least two");
        Self {
            marker: PhantomData,
        }
    }

    fn theta_from_eta(eta: &DirichletMeanPrecisionEta<D>) -> DirichletMeanPrecisionTheta<D> {
        DirichletMeanPrecisionTheta {
            mean: baseline_softmax(eta.logits),
            precision: PrecisionLink::inverse(eta.precision),
        }
    }

    fn nll_theta(y: [f64; D], theta: &DirichletMeanPrecisionTheta<D>) -> f64 {
        if !valid_observation(&y) || !valid_theta(theta) {
            return f64::INFINITY;
        }

        let alpha: [f64; D] = std::array::from_fn(|component| theta.alpha_unchecked(component));
        let alpha_sum = alpha.iter().sum::<f64>();
        if !alpha_sum.is_finite() {
            return f64::INFINITY;
        }
        let concentration_mean = alpha.map(|component| component / alpha_sum);
        let mut nll = -ln_gamma_stirling_residual(alpha_sum)
            + alpha_sum * Self::categorical_kl(&concentration_mean, &y);
        for component in 0..D {
            nll += ln_gamma_stirling_residual(alpha[component]) + y[component].ln();
        }
        nll
    }

    fn categorical_kl(probability: &[f64; D], reference: &[f64; D]) -> f64 {
        let relative: [f64; D] = std::array::from_fn(|component| {
            (reference[component] - probability[component]) / probability[component]
        });
        if relative.iter().all(|value| value.abs() <= 0.25) {
            let mut powers = relative;
            let mut sum = 0.0;
            for order in 1..=128 {
                let weighted_power = probability
                    .iter()
                    .zip(powers.iter())
                    .map(|(weight, power)| weight * power)
                    .sum::<f64>();
                let magnitude = probability
                    .iter()
                    .zip(powers.iter())
                    .map(|(weight, power)| weight * power.abs())
                    .sum::<f64>()
                    / f64::from(order);
                let sign = if order % 2 == 0 { 1.0 } else { -1.0 };
                let term = sign * weighted_power / f64::from(order);
                sum += term;
                if order > 1 && magnitude <= f64::EPSILON * sum.abs() {
                    break;
                }
                for component in 0..D {
                    powers[component] *= relative[component];
                }
            }
            sum
        } else {
            probability
                .iter()
                .zip(reference.iter())
                .map(|(probability, reference)| probability * (probability.ln() - reference.ln()))
                .sum()
        }
    }

    #[inline]
    fn log_ratio(numerator: f64, denominator: f64) -> f64 {
        let centered = (numerator - denominator) / denominator;
        if centered.abs() <= 0.5 {
            centered.ln_1p()
        } else {
            numerator.ln() - denominator.ln()
        }
    }

    fn nll_and_gradient_eta_values(
        y: [f64; D],
        eta: &DirichletMeanPrecisionEta<D>,
    ) -> (f64, DirichletMeanPrecisionEta<D>) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, &theta);
        if !nll.is_finite() {
            return (
                nll,
                DirichletMeanPrecisionEta {
                    logits: [f64::NAN; D],
                    precision: f64::NAN,
                },
            );
        }

        let mut d_nll_d_alpha = [0.0; D];
        let alpha_sum = (0..D)
            .map(|component| theta.alpha_unchecked(component))
            .sum::<f64>();
        let precision_residual = digamma_minus_ln(alpha_sum);
        for component in 0..D {
            let alpha = theta.alpha_unchecked(component);
            d_nll_d_alpha[component] = digamma_minus_ln(alpha) - precision_residual
                + Self::log_ratio(alpha / alpha_sum, y[component]);
        }
        let d_nll_d_precision = theta
            .mean
            .iter()
            .zip(d_nll_d_alpha.iter())
            .map(|(mean, gradient)| mean * gradient)
            .sum::<f64>();

        let mut logits = [0.0; D];
        for component in 0..D.saturating_sub(1) {
            logits[component] = theta.precision
                * theta.mean[component]
                * (d_nll_d_alpha[component] - d_nll_d_precision);
        }

        (
            nll,
            DirichletMeanPrecisionEta {
                logits,
                precision: d_nll_d_precision * PrecisionLink::derivative_inverse(eta.precision),
            },
        )
    }
}

impl<const D: usize, PrecisionLink> Default for DirichletMeanPrecision<D, PrecisionLink>
where
    PrecisionLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize, PrecisionLink> Family for DirichletMeanPrecision<D, PrecisionLink>
where
    PrecisionLink: PositiveLink<f64>,
{
    type Eta = DirichletMeanPrecisionEta<D>;
    type Theta = DirichletMeanPrecisionTheta<D>;
    type GradientEta = DirichletMeanPrecisionEta<D>;
    type Observation<'obs> = [f64; D];
    type Workspace = ();

    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline]
    fn nll(
        &self,
        observation: Self::Observation<'_>,
        theta: &Self::Theta,
        _workspace: &mut Self::Workspace,
    ) -> f64 {
        Self::nll_theta(observation, theta)
    }

    #[inline]
    fn nll_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> f64 {
        Self::nll_theta(observation, &Self::theta_from_eta(eta))
    }

    #[inline]
    fn nll_and_gradient_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        Self::nll_and_gradient_eta_values(observation, eta)
    }
}

impl<const D: usize, PrecisionLink> FixedDimensionalFamily<D>
    for DirichletMeanPrecision<D, PrecisionLink>
where
    PrecisionLink: PositiveLink<f64>,
{
}

impl<const D: usize, PrecisionLink> HasObservationDimension
    for DirichletMeanPrecision<D, PrecisionLink>
where
    PrecisionLink: PositiveLink<f64>,
{
    fn observation_dimension(&self) -> usize {
        D
    }
}

#[cfg(feature = "rand")]
impl<Rng, const D: usize, PrecisionLink> TrySimulate<Rng>
    for DirichletMeanPrecision<D, PrecisionLink>
where
    Rng: rand::Rng,
    PrecisionLink: PositiveLink<f64>,
{
    type Sample = [f64; D];

    fn try_sample(
        &self,
        rng: &mut Rng,
        theta: &Self::Theta,
    ) -> Result<Self::Sample, SimulationError> {
        if !valid_theta(theta) {
            return Err(SimulationError::InvalidParameters("Dirichlet theta"));
        }
        let mut out = [0.0; D];
        let mut sum = 0.0;
        for component in 0..D {
            let Ok(gamma) = rand_distr::Gamma::new(theta.alpha_unchecked(component), 1.0) else {
                return Err(SimulationError::BackendRejected("Dirichlet concentration"));
            };
            out[component] = rand_distr::Distribution::sample(&gamma, rng);
            sum += out[component];
        }
        if !sum.is_finite() || sum <= 0.0 {
            return Err(SimulationError::NumericalFailure(
                "Dirichlet gamma normalization",
            ));
        }
        for value in &mut out {
            *value /= sum;
        }
        Ok(out)
    }
}

impl<const D: usize, PrecisionLink> CompilableFamily for DirichletMeanPrecision<D, PrecisionLink>
where
    PrecisionLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Shape = Product<Simplex<Mean, D>, Scalar<Precision>>;

    fn eta_from_shape(values: ShapeValues<Self::Shape>) -> DirichletMeanPrecisionEta<D> {
        DirichletMeanPrecisionEta::new(values.0, values.1)
    }

    fn gradient_to_shape(gradient: &DirichletMeanPrecisionEta<D>) -> ShapeValues<Self::Shape> {
        (gradient.logits, gradient.precision)
    }

    fn initial_shape<'obs, Obs>(&self, obs: &'obs Obs) -> ShapeValues<Self::Shape>
    where
        Obs: ObservationView<'obs, Observation = [f64; D]> + 'obs,
    {
        let mut weight_sum = 0.0;
        let mut mean = [0.0; D];
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            let value = obs.observation_at(row);
            if weight > 0.0 && valid_observation(&value) {
                weight_sum += weight;
                for component in 0..D {
                    mean[component] += weight * value[component];
                }
            }
        }
        if weight_sum > 0.0 {
            for value in &mut mean {
                *value /= weight_sum;
            }
        } else if D > 0 {
            #[allow(clippy::cast_precision_loss)]
            let uniform = 1.0 / D as f64;
            mean.fill(uniform);
        }
        let floor = f64::MIN_POSITIVE.sqrt();
        for value in &mut mean {
            *value = value.max(floor);
        }
        let total = mean.iter().sum::<f64>();
        for value in &mut mean {
            *value /= total;
        }

        let mut variance = [0.0; D];
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            let value = obs.observation_at(row);
            if weight > 0.0 && valid_observation(&value) {
                for component in 0..D {
                    variance[component] += weight * (value[component] - mean[component]).powi(2);
                }
            }
        }
        let mut precision_sum = 0.0;
        let mut precision_count = 0.0;
        if weight_sum > 0.0 {
            for component in 0..D {
                let variance = variance[component] / weight_sum;
                if variance > 0.0 {
                    let estimate = mean[component] * (1.0 - mean[component]) / variance - 1.0;
                    if estimate.is_finite() && estimate > 0.0 {
                        precision_sum += estimate.clamp(1.0, 1.0e4);
                        precision_count += 1.0;
                    }
                }
            }
        }
        let precision = if precision_count > 0.0 {
            precision_sum / precision_count
        } else {
            10.0
        };
        let baseline = mean[D - 1].ln();
        let logits = std::array::from_fn(|component| {
            if component + 1 == D {
                0.0
            } else {
                mean[component].ln() - baseline
            }
        });
        (logits, PrecisionLink::initial_eta_from_theta(precision))
    }
}

/// Link-scale predictors for Dirichlet mean/precision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirichletMeanPrecisionEta<const D: usize> {
    /// Baseline-softmax logits. The last entry is normalized to zero.
    pub logits: [f64; D],
    /// Precision predictor.
    pub precision: f64,
}

impl<const D: usize> DirichletMeanPrecisionEta<D> {
    /// Creates predictors and normalizes the baseline logit to zero.
    #[must_use]
    pub const fn new(mut logits: [f64; D], precision: f64) -> Self {
        if D > 0 {
            logits[D - 1] = 0.0;
        }
        Self { logits, precision }
    }
}

/// Natural-scale Dirichlet mean/precision parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirichletMeanPrecisionTheta<const D: usize> {
    /// Simplex mean weights.
    mean: [f64; D],
    /// Positive concentration precision. Component `alpha_i = mean_i * precision`.
    precision: f64,
}

impl<const D: usize> DirichletMeanPrecisionTheta<D> {
    /// Creates valid natural-scale parameters.
    pub fn try_new(mean: [f64; D], precision: f64) -> Result<Self, ModelError> {
        let theta = Self { mean, precision };
        if valid_theta(&theta) {
            Ok(theta)
        } else {
            Err(ModelError::InvalidParameter {
                parameter: "dirichlet theta",
                expected: "D >= 2, an interior simplex mean, and finite positive alpha values",
            })
        }
    }

    /// Simplex mean weights.
    #[must_use]
    pub const fn mean(&self) -> &[f64; D] {
        &self.mean
    }

    /// Concentration precision.
    #[must_use]
    pub const fn precision(&self) -> f64 {
        self.precision
    }

    /// Returns one Dirichlet concentration parameter.
    #[must_use]
    #[inline]
    pub fn alpha(&self, component: usize) -> Option<f64> {
        self.mean.get(component).map(|mean| mean * self.precision)
    }

    #[inline]
    fn alpha_unchecked(&self, component: usize) -> f64 {
        self.mean[component] * self.precision
    }
}

fn valid_observation<const D: usize>(y: &[f64; D]) -> bool {
    D >= 2
        && y.iter().all(|value| value.is_finite() && *value > 0.0)
        && (y.iter().sum::<f64>() - 1.0).abs() <= SIMPLEX_TOLERANCE
}

fn valid_theta<const D: usize>(theta: &DirichletMeanPrecisionTheta<D>) -> bool {
    D >= 2
        && theta.precision > 0.0
        && theta.precision.is_finite()
        && theta
            .mean
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        && theta.mean.iter().all(|value| {
            let alpha = *value * theta.precision;
            alpha.is_finite() && alpha > 0.0
        })
        && (theta.mean.iter().sum::<f64>() - 1.0).abs() <= SIMPLEX_TOLERANCE
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{
        DenseDesign, Family, Gamlss, LinearPredictorBlock, Mean, ModelError, NoPenalty,
        ParameterBlock, ParameterBlocks, Precision, SimplexLogitParameterBlock,
    };

    use super::{DirichletMeanPrecision, DirichletMeanPrecisionEta, DirichletMeanPrecisionTheta};

    #[test]
    fn softmax_eta_constructs_valid_simplex_theta() {
        let family = DirichletMeanPrecision::<4>::new();
        let eta = DirichletMeanPrecisionEta::new([2.0, -1.0, 0.5, 9.0], 3.0_f64.ln());
        let theta = family.theta(&eta, &mut family.workspace());
        assert_relative_eq!(theta.mean.iter().sum::<f64>(), 1.0, epsilon = 1.0e-12);
        assert!(theta.mean.iter().all(|value| *value > 0.0));
        assert_relative_eq!(theta.precision, 3.0, epsilon = 1.0e-12);
    }

    #[test]
    fn checked_constructor_rejects_dimensions_below_two() {
        assert_eq!(
            DirichletMeanPrecision::<1>::try_new(),
            Err(ModelError::InvalidParameter {
                parameter: "dimension",
                expected: "at least two",
            })
        );
        assert!(DirichletMeanPrecision::<2>::try_new().is_ok());
    }

    #[test]
    fn extreme_finite_logits_preserve_interior_simplex() {
        let family = DirichletMeanPrecision::<3>::new();
        let eta = DirichletMeanPrecisionEta::new([1.0e6, -1.0e6, 0.0], 2.0_f64.ln());
        let theta = family.theta(&eta, &mut family.workspace());

        assert!(
            theta
                .mean()
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
        );
        assert_relative_eq!(theta.mean().iter().sum::<f64>(), 1.0, epsilon = 1.0e-12);
        assert!(
            family
                .nll([0.2, 0.3, 0.5], &theta, &mut family.workspace())
                .is_finite()
        );
    }

    #[test]
    fn checked_theta_rejects_nonrepresentable_alpha() {
        assert!(DirichletMeanPrecisionTheta::<1>::try_new([1.0], 1.0).is_err());
        assert!(DirichletMeanPrecisionTheta::<2>::try_new([0.0, 1.0], 1.0).is_err());
        assert!(
            DirichletMeanPrecisionTheta::<2>::try_new([f64::MIN_POSITIVE, 1.0], f64::MIN_POSITIVE,)
                .is_err()
        );
    }

    #[test]
    fn nll_matches_hand_computed_value() {
        let family = DirichletMeanPrecision::<3>::new();
        let theta = DirichletMeanPrecisionTheta::try_new([0.2, 0.3, 0.5], 10.0).unwrap();
        let y = [0.1_f64, 0.4, 0.5];
        let expected = -((gamlss_special::ln_gamma(10.0)
            - gamlss_special::ln_gamma(2.0)
            - gamlss_special::ln_gamma(3.0)
            - gamlss_special::ln_gamma(5.0))
            + (2.0 - 1.0) * y[0].ln()
            + (3.0 - 1.0) * y[1].ln()
            + (5.0 - 1.0) * y[2].ln());
        assert_relative_eq!(
            family.nll(y, &theta, &mut family.workspace()),
            expected,
            epsilon = 1.0e-12
        );
        assert_eq!(theta.alpha(0), Some(2.0));
        assert_eq!(theta.alpha(3), None);
    }

    #[test]
    fn concentrated_dirichlet_preserves_normalizer_and_precision_gradient() {
        let family = DirichletMeanPrecision::<3>::new();
        let mean = [0.25, 0.25, 0.5];
        let eta =
            DirichletMeanPrecisionEta::new([0.5_f64.ln(), 0.5_f64.ln(), 0.0], 1.0e16_f64.ln());
        let theta = family.theta(&eta, &mut ());
        let alpha = theta.mean.map(|component| component * theta.precision);
        let alpha_sum = alpha.iter().sum::<f64>();
        let expected = alpha
            .iter()
            .copied()
            .map(gamlss_special::ln_gamma_stirling_residual)
            .sum::<f64>()
            - gamlss_special::ln_gamma_stirling_residual(alpha_sum)
            + mean.iter().copied().map(f64::ln).sum::<f64>();
        let (nll, gradient) = family.nll_and_gradient_eta(mean, &eta, &mut ());

        assert!((nll - expected).abs() < 1.0e-13, "nll was {nll}");
        assert!(
            (gradient.precision + 1.0).abs() < 1.0e-14,
            "precision gradient was {}",
            gradient.precision
        );
    }

    #[test]
    fn eta_gradient_matches_finite_difference() {
        let family = DirichletMeanPrecision::<3>::new();
        let eta = DirichletMeanPrecisionEta::new([0.2, -0.3, 0.0], 2.0_f64.ln());
        let y = [0.2, 0.3, 0.5];
        let (_, gradient) = family.nll_and_gradient_eta(y, &eta, &mut family.workspace());

        for index in 0..2 {
            let mut plus = eta;
            plus.logits[index] += 1.0e-6;
            let mut minus = eta;
            minus.logits[index] -= 1.0e-6;
            let fd = (family.nll_eta(y, &plus, &mut family.workspace())
                - family.nll_eta(y, &minus, &mut family.workspace()))
                / 2.0e-6;
            assert_relative_eq!(gradient.logits[index], fd, epsilon = 1.0e-6);
        }

        let mut plus = eta;
        plus.precision += 1.0e-6;
        let mut minus = eta;
        minus.precision -= 1.0e-6;
        let fd = (family.nll_eta(y, &plus, &mut family.workspace())
            - family.nll_eta(y, &minus, &mut family.workspace()))
            / 2.0e-6;
        assert_relative_eq!(gradient.precision, fd, epsilon = 1.0e-6);
    }

    #[test]
    fn invalid_domains_return_infinite_nll_and_nan_gradient() {
        let family = DirichletMeanPrecision::<3>::new();
        let eta = DirichletMeanPrecisionEta::new([0.0, 0.0, 0.0], 0.0);
        assert!(
            family
                .nll_eta([0.0, 0.5, 0.5], &eta, &mut family.workspace())
                .is_infinite()
        );
        assert!(
            family
                .nll(
                    [0.2, 0.3, 0.6],
                    &DirichletMeanPrecisionTheta::try_new([0.2, 0.3, 0.5], 3.0).unwrap(),
                    &mut family.workspace()
                )
                .is_infinite()
        );
        let (nll, gradient) =
            family.nll_and_gradient_eta([f64::NAN, 0.5, 0.5], &eta, &mut family.workspace());
        assert!(nll.is_infinite());
        assert!(gradient.logits.iter().all(|value| value.is_nan()));
        assert!(gradient.precision.is_nan());
    }

    #[test]
    fn compiled_blocks_are_fit_ready_with_baseline_logits() {
        let y = [[0.2, 0.3, 0.5], [0.1, 0.7, 0.2], [0.4, 0.2, 0.4]];
        let n = y.len();
        let mean = SimplexLogitParameterBlock::<Mean, 3, _, _>::new(
            vec![
                LinearPredictorBlock::new(DenseDesign::intercept(n)),
                LinearPredictorBlock::new(DenseDesign::intercept(n)),
            ],
            NoPenalty,
            99,
        );
        let precision =
            ParameterBlock::<Precision, _, _>::linear(DenseDesign::intercept(n), NoPenalty, 99);
        let blocks = ParameterBlocks::new((mean, precision));
        let model = Gamlss::try_new_with_observations(
            DirichletMeanPrecision::<3>::new(),
            blocks,
            y.as_slice(),
        )
        .unwrap();
        let beta = vec![0.2, -0.1, 2.0_f64.ln()];
        let eta = model.predict_eta_row(&beta, 0).unwrap();

        assert_eq!(model.nparams(), 3);
        assert_relative_eq!(eta.logits[0], 0.2);
        assert_relative_eq!(eta.logits[1], -0.1);
        assert_relative_eq!(eta.logits[2], 0.0);

        let mut gradient = vec![0.0; beta.len()];
        model.try_value_gradient_into(&beta, &mut gradient).unwrap();
        for index in 0..beta.len() {
            let mut plus = beta.clone();
            plus[index] += 1.0e-6;
            let mut minus = beta.clone();
            minus[index] -= 1.0e-6;
            let fd = (model.try_value(&plus).unwrap() - model.try_value(&minus).unwrap()) / 2.0e-6;
            assert_relative_eq!(gradient[index], fd, epsilon = 1.0e-6);
        }
    }
}
