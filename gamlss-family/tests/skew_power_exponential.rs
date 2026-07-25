use approx::assert_relative_eq;
use gamlss_core::{
    DenseDesign, Family, Gamlss, HasCdf, HasQuantile, Mu, NoPenalty, ParameterBlock,
    ParameterBlocks, Power, Sigma, SkewRatio,
};
use gamlss_family::{
    IndependentVec, PowerExponentialMuSigmaNu, PowerExponentialTheta,
    SkewPowerExponentialMeanSdSkewPower, SkewPowerExponentialMeanSdTheta,
    SkewPowerExponentialMuSigmaSkewPower, SkewPowerExponentialTheta,
};
use gamlss_special::ln_gamma;

use common::{
    assert_cdf_monotone, assert_close, assert_continuous_inverse,
    assert_gradient_matches_finite_difference, assert_nll_eta_matches_theta, integrate_simpson,
};

#[path = "common/helpers.rs"]
mod common;

#[test]
fn unit_skew_ratio_recovers_existing_power_exponential_exactly() {
    let skew = SkewPowerExponentialMuSigmaSkewPower::new();
    let symmetric = PowerExponentialMuSigmaNu::new();
    let skew_theta = SkewPowerExponentialTheta {
        mu: -0.4,
        sigma: 1.7,
        skew_ratio: 1.0,
        power: 0.9,
    };
    let symmetric_theta = PowerExponentialTheta {
        mu: skew_theta.mu,
        sigma: skew_theta.sigma,
        nu: skew_theta.power,
    };

    for y in [-8.0, -1.0, -0.4, 0.0, 3.0, 9.0] {
        assert_relative_eq!(
            skew.nll(y, &skew_theta, &mut ()),
            symmetric.nll(y, &symmetric_theta, &mut ()),
            epsilon = 2.0e-15
        );
        assert_relative_eq!(
            skew.cdf(y, &skew_theta),
            symmetric.cdf(y, &symmetric_theta),
            epsilon = 2.0e-15
        );
    }
    for probability in [0.01, 0.2, 0.5, 0.8, 0.99] {
        assert_relative_eq!(
            skew.quantile(probability, &skew_theta),
            symmetric.quantile(probability, &symmetric_theta),
            epsilon = 2.0e-12
        );
    }
}

#[test]
fn density_matches_gamlss_sep3_after_documented_scale_conversion() {
    let family = SkewPowerExponentialMuSigmaSkewPower::new();
    let theta = SkewPowerExponentialTheta {
        mu: 0.3,
        sigma: 1.4,
        skew_ratio: 1.7,
        power: 1.3,
    };
    let standardized_scale =
        (0.5 * (ln_gamma(1.0 / theta.power) - ln_gamma(3.0 / theta.power))).exp();
    let gamlss_sigma = theta.sigma * standardized_scale / (1.0 / theta.power).exp2();

    for y in [-2.5, 0.3, 0.9, 4.0] {
        let z = (y - theta.mu) / gamlss_sigma;
        let radial = if y < theta.mu {
            (theta.skew_ratio * z.abs()).powf(theta.power)
        } else {
            (z.abs() / theta.skew_ratio).powf(theta.power)
        };
        let gamlss_sep3_log_density = (-0.5_f64).mul_add(radial, -gamlss_sigma.ln())
            + theta.skew_ratio.ln()
            - theta.skew_ratio.mul_add(theta.skew_ratio, 1.0).ln()
            - std::f64::consts::LN_2 / theta.power
            - ln_gamma(1.0 + 1.0 / theta.power);
        assert_relative_eq!(
            -family.nll(y, &theta, &mut ()),
            gamlss_sep3_log_density,
            epsilon = 2.0e-14
        );
    }
}

#[test]
fn cdf_has_two_piece_mass_and_reflection_identities() {
    let family = SkewPowerExponentialMuSigmaSkewPower::new();
    let theta = SkewPowerExponentialTheta {
        mu: 0.0,
        sigma: 1.3,
        skew_ratio: 2.2,
        power: 1.4,
    };
    let reflected = SkewPowerExponentialTheta {
        skew_ratio: 1.0 / theta.skew_ratio,
        ..theta
    };

    assert_relative_eq!(
        family.cdf(theta.mu, &theta),
        1.0 / theta.skew_ratio.mul_add(theta.skew_ratio, 1.0),
        epsilon = 2.0e-16
    );
    assert_cdf_monotone(&family, theta, &[-20.0, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0]);
    for y in [-8.0, -1.5, -0.1, 0.0, 0.7, 6.0] {
        assert_relative_eq!(
            family.cdf(y, &theta) + family.cdf(-y, &reflected),
            1.0,
            epsilon = 4.0e-15
        );
    }
    for probability in [1.0e-8, 0.01, 0.25, 0.8, 1.0 - 1.0e-8] {
        assert_continuous_inverse(&family, probability, theta, 2.0e-10);
    }
}

#[test]
fn mean_sd_parameterization_matches_its_public_mode_scale_conversion() {
    let mean_sd = SkewPowerExponentialMeanSdSkewPower::new();
    let mode_scale = SkewPowerExponentialMuSigmaSkewPower::new();
    let theta = SkewPowerExponentialMeanSdTheta {
        mean: 1.2,
        sigma: 1.7,
        skew_ratio: 2.0,
        power: 1.5,
    };
    let equivalent = theta.mode_scale().unwrap();
    let round_trip = equivalent.mean_sd().unwrap();
    assert_relative_eq!(round_trip.mean, theta.mean, epsilon = 4.0e-15);
    assert_relative_eq!(round_trip.sigma, theta.sigma, epsilon = 4.0e-15);
    assert_relative_eq!(round_trip.skew_ratio, theta.skew_ratio, epsilon = 0.0);
    assert_relative_eq!(round_trip.power, theta.power, epsilon = 0.0);
    assert!(theta.is_valid());

    for y in [-4.0, 0.0, 1.2, 3.0, 8.0] {
        assert_relative_eq!(
            mean_sd.nll(y, &theta, &mut ()),
            mode_scale.nll(y, &equivalent, &mut ()),
            epsilon = 2.0e-15
        );
        assert_relative_eq!(
            mean_sd.cdf(y, &theta),
            mode_scale.cdf(y, &equivalent),
            epsilon = 2.0e-15
        );
    }

    let lower = mean_sd.quantile(1.0e-7, &theta);
    let upper = mean_sd.quantile(1.0 - 1.0e-7, &theta);
    let density = |y| (-mean_sd.nll(y, &theta, &mut ())).exp();
    let mass = integrate_simpson(lower, upper, 8192, density);
    let first_moment = integrate_simpson(lower, upper, 8192, |y| y * density(y));
    let second_moment = integrate_simpson(lower, upper, 8192, |y| y * y * density(y));
    assert_close(mass, 1.0, 0.0, 2.0e-6);
    assert_close(first_moment / mass, theta.mean, 0.0, 2.0e-5);
    let normalized_mean = first_moment / mass;
    let variance = normalized_mean.mul_add(-normalized_mean, second_moment / mass);
    assert_close(variance.sqrt(), theta.sigma, 0.0, 8.0e-5);
}

#[test]
fn gradients_and_eta_theta_contracts_cover_both_parameterizations() {
    let mode_scale = SkewPowerExponentialMuSigmaSkewPower::new();
    let mean_sd = SkewPowerExponentialMeanSdSkewPower::new();
    let eta = [0.2, 0.3, 1.7_f64.ln(), 1.3_f64.ln()];

    for y in [-1.0, 1.2] {
        assert_gradient_matches_finite_difference::<_, 4>(&mode_scale, y, eta);
        assert_gradient_matches_finite_difference::<_, 4>(&mean_sd, y, eta);
        assert_nll_eta_matches_theta::<_, 4>(&mode_scale, y, eta);
        assert_nll_eta_matches_theta::<_, 4>(&mean_sd, y, eta);
    }
}

#[test]
fn typed_compilation_exposes_semantic_skew_ratio_and_power_blocks() {
    let observations = [-1.0, 0.2, 1.5];
    let blocks = ParameterBlocks::try_new((
        ParameterBlock::<Mu, _, _>::linear(
            DenseDesign::intercept(observations.len()),
            NoPenalty,
            0,
        ),
        ParameterBlock::<Sigma, _, _>::linear(
            DenseDesign::intercept(observations.len()),
            NoPenalty,
            0,
        ),
        ParameterBlock::<SkewRatio, _, _>::linear(
            DenseDesign::intercept(observations.len()),
            NoPenalty,
            0,
        ),
        ParameterBlock::<Power, _, _>::linear(
            DenseDesign::intercept(observations.len()),
            NoPenalty,
            0,
        ),
    ))
    .unwrap();
    let model = Gamlss::try_new(
        SkewPowerExponentialMuSigmaSkewPower::new(),
        blocks,
        &observations,
    )
    .unwrap();
    let parameters = model.initial_parameters().unwrap();
    assert_eq!(parameters.len(), 4);
    assert!(parameters.iter().all(|value| value.is_finite()));

    let _: gamlss_family::prelude::SkewPowerExponentialMuSigmaSkewPower =
        gamlss_family::prelude::SkewPowerExponentialMuSigmaSkewPower::new();
}

#[test]
fn independent_multivariate_product_uses_the_univariate_family() {
    let component = SkewPowerExponentialMuSigmaSkewPower::new();
    let family = IndependentVec::<_, 2>::new(component);
    let theta = [
        SkewPowerExponentialTheta {
            mu: -0.3,
            sigma: 1.2,
            skew_ratio: 1.8,
            power: 1.4,
        },
        SkewPowerExponentialTheta {
            mu: 0.7,
            sigma: 0.8,
            skew_ratio: 0.6,
            power: 2.2,
        },
    ];
    let observation = [-1.0, 1.5];
    let expected_nll = component.nll(observation[0], &theta[0], &mut ())
        + component.nll(observation[1], &theta[1], &mut ());

    assert_relative_eq!(
        family.nll(observation, &theta, &mut family.workspace()),
        expected_nll,
        epsilon = 2.0e-15
    );
    assert_relative_eq!(
        family.cdf(observation, &theta),
        component.cdf(observation[0], &theta[0]) * component.cdf(observation[1], &theta[1]),
        epsilon = 2.0e-15
    );
}

#[test]
fn invalid_domains_are_non_panicking_and_explicit() {
    let mode_scale = SkewPowerExponentialMuSigmaSkewPower::new();
    let valid = SkewPowerExponentialTheta {
        mu: 0.0,
        sigma: 1.0,
        skew_ratio: 1.5,
        power: 1.2,
    };
    for invalid in [
        SkewPowerExponentialTheta {
            sigma: 0.0,
            ..valid
        },
        SkewPowerExponentialTheta {
            skew_ratio: 0.0,
            ..valid
        },
        SkewPowerExponentialTheta {
            power: 0.0,
            ..valid
        },
        SkewPowerExponentialTheta {
            mu: f64::NAN,
            ..valid
        },
    ] {
        assert!(mode_scale.nll(0.2, &invalid, &mut ()).is_infinite());
        assert!(mode_scale.cdf(0.2, &invalid).is_nan());
        assert!(mode_scale.quantile(0.5, &invalid).is_nan());
        assert!(!invalid.is_valid());
    }
    assert!(mode_scale.nll(f64::NAN, &valid, &mut ()).is_infinite());
    assert!(mode_scale.cdf(f64::INFINITY, &valid).is_nan());

    let mean_sd = SkewPowerExponentialMeanSdSkewPower::new();
    let invalid_mean_sd = SkewPowerExponentialMeanSdTheta {
        mean: 0.0,
        sigma: -1.0,
        skew_ratio: 1.5,
        power: 1.2,
    };
    assert!(mean_sd.nll(0.2, &invalid_mean_sd, &mut ()).is_infinite());
    assert!(mean_sd.cdf(0.2, &invalid_mean_sd).is_nan());
    assert!(!invalid_mean_sd.is_valid());
    assert!(invalid_mean_sd.mode_scale().is_none());
}

#[cfg(feature = "rand")]
#[test]
fn both_parameterizations_sample_and_reject_invalid_parameters() {
    use gamlss_core::TrySimulate;
    use rand::SeedableRng;

    let mut rng = rand::rngs::StdRng::seed_from_u64(37);
    let mode_scale = SkewPowerExponentialMuSigmaSkewPower::new();
    let valid = SkewPowerExponentialTheta {
        mu: 0.3,
        sigma: 1.4,
        skew_ratio: 1.8,
        power: 1.3,
    };
    assert!(
        mode_scale
            .try_sample(&mut rng, &valid)
            .is_ok_and(f64::is_finite)
    );
    assert!(
        mode_scale
            .try_sample(
                &mut rng,
                &SkewPowerExponentialTheta {
                    skew_ratio: 0.0,
                    ..valid
                }
            )
            .is_err()
    );

    let mean_sd = SkewPowerExponentialMeanSdSkewPower::new();
    assert!(
        mean_sd
            .try_sample(
                &mut rng,
                &SkewPowerExponentialMeanSdTheta {
                    mean: 0.3,
                    sigma: 1.4,
                    skew_ratio: 1.8,
                    power: 1.3,
                }
            )
            .is_ok_and(f64::is_finite)
    );
}
