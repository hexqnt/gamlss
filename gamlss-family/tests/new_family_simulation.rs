#![cfg(feature = "rand")]
#![allow(clippy::float_cmp)]

use gamlss_core::TrySimulate;
use gamlss_family::*;
use rand::SeedableRng;

#[test]
fn new_family_samples_respect_support_and_reject_invalid_parameters() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(17);

    let geometric = GeometricMean::new();
    let sample = geometric
        .try_sample(&mut rng, &GeometricTheta { mean: 2.0 })
        .unwrap();
    assert!(sample >= 0.0 && sample.fract() == 0.0);
    assert!(
        geometric
            .try_sample(&mut rng, &GeometricTheta { mean: 0.0 })
            .is_err()
    );

    let binomial = BinomialFixedTrialsProbability::try_new(10).unwrap();
    let sample = binomial
        .try_sample(&mut rng, &BinomialTheta { probability: 0.4 })
        .unwrap();
    assert!((0.0..=10.0).contains(&sample) && sample.fract() == 0.0);
    assert!(
        binomial
            .try_sample(&mut rng, &BinomialTheta { probability: 1.0 })
            .is_err()
    );

    let categorical = Categorical::<3>::new();
    let sample = categorical
        .try_sample(
            &mut rng,
            &CategoricalTheta {
                probabilities: [0.2, 0.3, 0.5],
            },
        )
        .unwrap();
    assert!(matches!(sample, 0.0 | 1.0 | 2.0));
    assert!(
        categorical
            .try_sample(
                &mut rng,
                &CategoricalTheta {
                    probabilities: [0.2, 0.3, 0.4],
                },
            )
            .is_err()
    );

    let rayleigh = RayleighScale::new();
    let sample = rayleigh
        .try_sample(&mut rng, &RayleighTheta { scale: 1.2 })
        .unwrap();
    assert!(sample > 0.0 && sample.is_finite());
    assert!(
        rayleigh
            .try_sample(&mut rng, &RayleighTheta { scale: 0.0 })
            .is_err()
    );

    let log_logistic = LogLogisticScaleShape::new();
    let sample = log_logistic
        .try_sample(
            &mut rng,
            &LogLogisticTheta {
                scale: 1.2,
                shape: 2.0,
            },
        )
        .unwrap();
    assert!(sample > 0.0 && sample.is_finite());
    assert!(
        log_logistic
            .try_sample(
                &mut rng,
                &LogLogisticTheta {
                    scale: 1.2,
                    shape: 0.0,
                },
            )
            .is_err()
    );

    let chi = ChiDegreesOfFreedom::new();
    let sample = chi
        .try_sample(
            &mut rng,
            &ChiTheta {
                degrees_of_freedom: 3.0,
            },
        )
        .unwrap();
    assert!(sample >= 0.0 && sample.is_finite());
    assert!(
        chi.try_sample(
            &mut rng,
            &ChiTheta {
                degrees_of_freedom: 0.0,
            },
        )
        .is_err()
    );

    let chi_squared = ChiSquaredDegreesOfFreedom::new();
    let sample = chi_squared
        .try_sample(
            &mut rng,
            &ChiSquaredTheta {
                degrees_of_freedom: 3.0,
            },
        )
        .unwrap();
    assert!(sample >= 0.0 && sample.is_finite());
    assert!(
        chi_squared
            .try_sample(
                &mut rng,
                &ChiSquaredTheta {
                    degrees_of_freedom: 0.0,
                },
            )
            .is_err()
    );

    let generalized_pareto = GeneralizedParetoScaleShape::new();
    let sample = generalized_pareto
        .try_sample(
            &mut rng,
            &GeneralizedParetoTheta {
                scale: 2.0,
                shape: -0.2,
            },
        )
        .unwrap();
    assert!((0.0..10.0).contains(&sample));
    assert!(
        generalized_pareto
            .try_sample(
                &mut rng,
                &GeneralizedParetoTheta {
                    scale: 0.0,
                    shape: 0.0,
                },
            )
            .is_err()
    );
}
