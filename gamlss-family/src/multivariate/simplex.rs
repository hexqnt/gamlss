#![allow(clippy::needless_range_loop, clippy::suboptimal_flops)]

use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Family, FixedDimensionalFamily, Log, Mean, MeanPrecisionSimplex, MeanPrecisionSimplexSpec,
    PositiveLink, Precision,
};
use gamlss_special::{digamma, ln_gamma};

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
    /// Creates a stateless Dirichlet mean/precision family.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    fn theta_from_eta(eta: &DirichletMeanPrecisionEta<D>) -> DirichletMeanPrecisionTheta<D> {
        DirichletMeanPrecisionTheta {
            mean: softmax_baseline(eta.logits),
            precision: PrecisionLink::inverse(eta.precision),
        }
    }

    fn nll_theta(y: [f64; D], theta: &DirichletMeanPrecisionTheta<D>) -> f64 {
        if !valid_observation(&y) || !valid_theta(theta) {
            return f64::INFINITY;
        }

        let mut sum = ln_gamma(theta.precision);
        for component in 0..D {
            let alpha = theta.alpha_unchecked(component);
            sum -= ln_gamma(alpha);
            sum += (alpha - 1.0) * y[component].ln();
        }
        -sum
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
        let mut weighted_alpha_gradient_sum = 0.0;
        let mut d_nll_d_precision = -digamma(theta.precision);
        for component in 0..D {
            let alpha = theta.alpha_unchecked(component);
            d_nll_d_alpha[component] = digamma(alpha) - y[component].ln();
            d_nll_d_precision += theta.mean[component] * d_nll_d_alpha[component];
            weighted_alpha_gradient_sum += theta.mean[component] * d_nll_d_alpha[component];
        }

        let mut logits = [0.0; D];
        for component in 0..D.saturating_sub(1) {
            logits[component] = theta.precision
                * theta.mean[component]
                * (d_nll_d_alpha[component] - weighted_alpha_gradient_sum);
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
    PrecisionLink: PositiveLink<f64>,
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
    type ParamSpec = MeanPrecisionSimplex<Mean, Precision, D>;

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

#[cfg(feature = "rand")]
impl<Rng, const D: usize, PrecisionLink> CanSimulate<Rng>
    for DirichletMeanPrecision<D, PrecisionLink>
where
    Rng: rand::Rng,
    PrecisionLink: PositiveLink<f64>,
{
    type Sample = [f64; D];

    fn sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Self::Sample {
        if !valid_theta(theta) {
            return [f64::NAN; D];
        }
        let mut out = [0.0; D];
        let mut sum = 0.0;
        for component in 0..D {
            let gamma = rand_distr::Gamma::new(theta.alpha_unchecked(component), 1.0).unwrap();
            out[component] = rand_distr::Distribution::sample(&gamma, rng);
            sum += out[component];
        }
        for value in &mut out {
            *value /= sum;
        }
        out
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
    pub mean: [f64; D],
    /// Positive concentration precision. Component `alpha_i = mean_i * precision`.
    pub precision: f64,
}

impl<const D: usize> DirichletMeanPrecisionTheta<D> {
    /// Creates natural-scale parameters.
    #[must_use]
    #[inline]
    pub const fn new(mean: [f64; D], precision: f64) -> Self {
        Self { mean, precision }
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

impl<const D: usize, PrecisionLink>
    MeanPrecisionSimplexSpec<DirichletMeanPrecision<D, PrecisionLink>, D>
    for MeanPrecisionSimplex<Mean, Precision, D>
where
    PrecisionLink: PositiveLink<f64>,
{
    type MeanParameter = Mean;
    type PrecisionParameter = Precision;
    type PrecisionLink = PrecisionLink;

    fn eta_from_simplex_logits_precision(
        logits: [f64; D],
        precision: f64,
    ) -> DirichletMeanPrecisionEta<D> {
        DirichletMeanPrecisionEta::new(logits, precision)
    }

    fn simplex_logit_gradient_part(
        gradient: &DirichletMeanPrecisionEta<D>,
        component: usize,
    ) -> f64 {
        gradient.logits[component]
    }

    fn precision_gradient_part(gradient: &DirichletMeanPrecisionEta<D>) -> f64 {
        gradient.precision
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
        && (theta.mean.iter().sum::<f64>() - 1.0).abs() <= SIMPLEX_TOLERANCE
}

fn softmax_baseline<const D: usize>(mut logits: [f64; D]) -> [f64; D] {
    if D == 0 {
        return logits;
    }
    logits[D - 1] = 0.0;
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut weights = [0.0; D];
    let mut sum = 0.0;
    for component in 0..D {
        weights[component] = (logits[component] - max).exp();
        sum += weights[component];
    }
    for weight in &mut weights {
        *weight /= sum;
    }
    weights
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{
        DenseDesign, Family, Gamlss, LinearPredictorBlock, Mean, NoPenalty, ParameterBlock,
        ParameterBlocks, Precision, SimplexLogitParameterBlock,
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
    fn nll_matches_hand_computed_value() {
        let family = DirichletMeanPrecision::<3>::new();
        let theta = DirichletMeanPrecisionTheta::new([0.2, 0.3, 0.5], 10.0);
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
                    &DirichletMeanPrecisionTheta::new([0.2, 0.3, 0.5], 3.0),
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
        let precision = ParameterBlock::<Precision, gamlss_core::Log, _, _>::linear(
            DenseDesign::intercept(n),
            NoPenalty,
            99,
        );
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
