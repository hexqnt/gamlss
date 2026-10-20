use approx::assert_relative_eq;
use gamlss_core::{
    DenseDesign, Gamlss, ModelError, Mu, NoPenalty, Objective, ParameterBlock, ParameterBlocks,
    Sigma,
};
use gamlss_family::NormalMuSigma;

use super::{CoefficientPrior, GaussianCoefficientPrior, PosteriorPotential};

#[test]
fn normalized_gaussian_prior_includes_constant_and_gradient() {
    let prior = GaussianCoefficientPrior::try_new(vec![0.0, 1.0], vec![1.0, 2.0]).unwrap();
    let coefficients = [0.5, -1.0];
    let mut gradient = [0.0; 2];
    prior.add_gradient(&coefficients, &mut gradient);
    let mut fused_gradient = [0.0; 2];
    let fused_value = prior
        .try_negative_log_density_and_add_gradient(&coefficients, &mut fused_gradient)
        .unwrap();

    assert_relative_eq!(gradient[0], 0.5);
    assert_relative_eq!(gradient[1], -0.5);
    for (actual, expected) in fused_gradient.iter().zip(gradient) {
        assert_relative_eq!(*actual, expected);
    }
    let expected_value = 2.0_f64.mul_add(super::HALF_LOG_TWO_PI, 0.125 + 0.5) + 2.0_f64.ln();
    assert_relative_eq!(prior.negative_log_density(&coefficients), expected_value);
    assert_relative_eq!(fused_value, expected_value);

    assert_eq!(
        prior.try_negative_log_density(&[0.0]).unwrap_err(),
        ModelError::BetaLength {
            expected: 2,
            actual: 1,
        }
    );
    assert_eq!(
        prior
            .try_add_gradient(&coefficients, &mut [0.0])
            .unwrap_err(),
        ModelError::GradientLength {
            expected: 2,
            actual: 1,
        }
    );
}

#[test]
fn posterior_uses_summed_unpenalized_likelihood_and_matches_finite_difference() {
    let y = [0.0, 1.0, 2.0];
    let blocks = ParameterBlocks::new((
        ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 0),
        ParameterBlock::<Sigma, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 0),
    ));
    let model = Gamlss::try_new(NormalMuSigma::new(), blocks, &y).unwrap();
    let prior = GaussianCoefficientPrior::isotropic(model.nparams(), 0.0, 2.0).unwrap();
    let mut potential = PosteriorPotential::try_new(model, prior).unwrap();
    let coefficients = [0.2, -0.1];
    let mut gradient = [0.0; 2];
    let value = potential
        .value_gradient(&coefficients, &mut gradient)
        .unwrap();

    for index in 0..coefficients.len() {
        let mut plus = coefficients;
        let mut minus = coefficients;
        plus[index] += 1.0e-6;
        minus[index] -= 1.0e-6;
        let finite_difference =
            (potential.value(&plus).unwrap() - potential.value(&minus).unwrap()) / 2.0e-6;
        assert_relative_eq!(gradient[index], finite_difference, epsilon = 1.0e-6);
    }

    let mut pointwise = [0.0; 3];
    potential
        .pointwise_log_likelihood_into(&coefficients, &mut pointwise)
        .unwrap();
    let likelihood_nll = -pointwise.iter().sum::<f64>();
    assert_relative_eq!(
        value,
        likelihood_nll + potential.prior().negative_log_density(&coefficients)
    );
}

#[test]
fn workspace_posterior_reuses_buffers_and_exposes_weighted_pointwise_values() {
    let y = [0.0, 1.0, 2.0];
    let weights = [2.0, 0.5, 0.0];
    let blocks = ParameterBlocks::new((
        ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 0),
        ParameterBlock::<Sigma, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 0),
    ));
    let model = Gamlss::try_new_weighted(NormalMuSigma::new(), blocks, &y, &weights).unwrap();
    let prior = GaussianCoefficientPrior::isotropic(model.nparams(), 0.0, 2.0).unwrap();
    let potential = PosteriorPotential::try_new(model, prior).unwrap();
    let coefficients = [0.2, -0.1];
    let mut expected_gradient = [0.0; 2];
    let expected_value = potential
        .try_value_gradient_into(&coefficients, &mut expected_gradient)
        .unwrap();

    let mut raw = [f64::NAN; 3];
    let mut weighted = [f64::NAN; 3];
    potential
        .raw_pointwise_log_likelihood_into(&coefficients, &mut raw)
        .unwrap();
    potential
        .weighted_pointwise_log_likelihood_into(&coefficients, &mut weighted)
        .unwrap();
    for ((weighted, raw), weight) in weighted.iter().zip(raw).zip(weights) {
        assert_relative_eq!(*weighted, weight * raw);
    }

    let mut workspace_potential = potential.into_workspace_objective();
    for _ in 0..2 {
        let mut gradient = [f64::NAN; 2];
        assert_relative_eq!(
            workspace_potential
                .value_gradient(&coefficients, &mut gradient)
                .unwrap(),
            expected_value
        );
        for (actual, expected) in gradient.iter().zip(expected_gradient) {
            assert_relative_eq!(*actual, expected);
        }
        assert_relative_eq!(
            workspace_potential.value(&coefficients).unwrap(),
            expected_value
        );
    }

    let mut workspace_weighted = [f64::NAN; 3];
    workspace_potential
        .weighted_pointwise_log_likelihood_into(&coefficients, &mut workspace_weighted)
        .unwrap();
    for (actual, expected) in workspace_weighted.iter().zip(weighted) {
        assert_relative_eq!(*actual, expected);
    }
    assert_relative_eq!(
        -weighted.iter().sum::<f64>()
            + workspace_potential
                .potential()
                .prior()
                .negative_log_density(&coefficients),
        expected_value
    );
}
