#![allow(clippy::suboptimal_flops)]
use gamlss_core::{Family, HasCdf, HasQuantile};
use gamlss_family::*;

use common::assert_close;

#[path = "common/helpers.rs"]
mod common;

#[test]
fn total_mean_parameterizations_match_component_equivalents() {
    let zip_component = ZipComponentMeanZeroProbability::new();
    let zip_total = ZipTotalMeanZeroProbability::new();
    let zip_component_theta = ZipComponentMeanZeroProbabilityTheta {
        component_mean: 2.0,
        zero_probability: 0.3,
    };
    let zip_total_theta = ZipTotalMeanZeroProbabilityTheta {
        total_mean: (1.0 - zip_component_theta.zero_probability)
            * zip_component_theta.component_mean,
        zero_probability: zip_component_theta.zero_probability,
    };
    assert_close(
        zip_total.nll(3.0, &zip_total_theta, &mut zip_total.workspace()),
        zip_component.nll(3.0, &zip_component_theta, &mut zip_component.workspace()),
        0.0,
        1.0e-12,
    );
    assert_close(
        zip_total.cdf(3.0, &zip_total_theta),
        zip_component.cdf(3.0, &zip_component_theta),
        0.0,
        1.0e-12,
    );

    let zaga_component = ZagaComponentMeanCvZeroProbability::new();
    let zaga_total = ZagaTotalMeanCvZeroProbability::new();
    let zaga_component_theta = ZagaComponentMeanCvZeroProbabilityTheta {
        component_mean: 2.0,
        cv: 0.6,
        zero_probability: 0.25,
    };
    let zaga_total_theta = ZagaTotalMeanCvZeroProbabilityTheta {
        total_mean: (1.0 - zaga_component_theta.zero_probability)
            * zaga_component_theta.component_mean,
        cv: zaga_component_theta.cv,
        zero_probability: zaga_component_theta.zero_probability,
    };
    assert_close(
        zaga_total.nll(1.4, &zaga_total_theta, &mut zaga_total.workspace()),
        zaga_component.nll(1.4, &zaga_component_theta, &mut zaga_component.workspace()),
        0.0,
        1.0e-12,
    );
    assert_close(
        zaga_total.cdf(1.4, &zaga_total_theta),
        zaga_component.cdf(1.4, &zaga_component_theta),
        0.0,
        1.0e-12,
    );

    let zinb_component = ZinbComponentMeanSizeZeroProbability::new();
    let zinb_total = ZinbTotalMeanSizeZeroProbability::new();
    let zinb_component_theta = ZinbComponentMeanSizeZeroProbabilityTheta {
        component_mean: 2.0,
        size: 1.5,
        zero_probability: 0.25,
    };
    let zinb_total_theta = ZinbTotalMeanSizeZeroProbabilityTheta {
        total_mean: (1.0 - zinb_component_theta.zero_probability)
            * zinb_component_theta.component_mean,
        size: zinb_component_theta.size,
        zero_probability: zinb_component_theta.zero_probability,
    };
    assert_close(
        zinb_total.nll(3.0, &zinb_total_theta, &mut zinb_total.workspace()),
        zinb_component.nll(3.0, &zinb_component_theta, &mut zinb_component.workspace()),
        0.0,
        1.0e-12,
    );
    assert_close(
        zinb_total.cdf(3.0, &zinb_total_theta),
        zinb_component.cdf(3.0, &zinb_component_theta),
        0.0,
        1.0e-12,
    );
}

#[test]
fn tweedie_mean_cv_matches_mean_dispersion_equivalent() {
    let mean_cv = TweedieMeanCvPower::new();
    let mean_dispersion = TweedieMeanDispersionPower::new();
    let theta = TweedieMeanCvPowerTheta {
        mean: 1.4,
        cv: 0.7,
        power: 1.5,
    };
    let canonical: TweedieTheta = theta.into();

    assert_close(
        mean_cv.nll(1.1, &theta, &mut mean_cv.workspace()),
        mean_dispersion.nll(1.1, &canonical, &mut mean_dispersion.workspace()),
        0.0,
        1.0e-12,
    );
    assert_close(
        mean_cv.cdf(1.1, &theta),
        mean_dispersion.cdf(1.1, &canonical),
        0.0,
        1.0e-12,
    );
    assert_close(
        mean_cv.quantile(0.7, &theta),
        mean_dispersion.quantile(0.7, &canonical),
        0.0,
        1.0e-10,
    );
}

#[test]
fn student_t_dynamic_matches_fixed_df_equivalent() {
    let dynamic = StudentTMuSigmaTau::new();
    let fixed = StudentTMuSigma::try_new(5.0).unwrap();
    let dynamic_theta = StudentTMuSigmaTauTheta {
        mu: 0.2,
        sigma: 1.3,
        tau: 5.0,
    };
    let fixed_theta = StudentTTheta {
        mu: dynamic_theta.mu,
        sigma: dynamic_theta.sigma,
    };

    assert_close(
        dynamic.nll(0.7, &dynamic_theta, &mut dynamic.workspace()),
        fixed.nll(0.7, &fixed_theta, &mut fixed.workspace()),
        0.0,
        1.0e-12,
    );
    assert_close(
        dynamic.cdf(0.7, &dynamic_theta),
        fixed.cdf(0.7, &fixed_theta),
        0.0,
        1.0e-12,
    );
    assert_close(
        dynamic.quantile(0.7, &dynamic_theta),
        fixed.quantile(0.7, &fixed_theta),
        0.0,
        1.0e-10,
    );
}

#[test]
fn skew_normal_mean_sd_matches_location_scale_equivalent() {
    let mean_sd = SkewNormalMeanSdNu::new();
    let location_scale = SkewNormalMuSigmaNu::new();
    let mean_sd_theta = SkewNormalMeanSdTheta {
        mean: 0.2,
        sigma: 1.3,
        nu: 0.7,
    };
    let delta = mean_sd_theta.nu / mean_sd_theta.nu.hypot(1.0);
    let standardized_mean = (2.0 / std::f64::consts::PI).sqrt() * delta;
    let standardized_variance = 1.0 - standardized_mean * standardized_mean;
    let scale = mean_sd_theta.sigma / standardized_variance.sqrt();
    let location_scale_theta = SkewNormalTheta {
        mu: mean_sd_theta.mean - scale * standardized_mean,
        sigma: scale,
        nu: mean_sd_theta.nu,
    };

    assert_close(
        mean_sd.nll(0.7, &mean_sd_theta, &mut mean_sd.workspace()),
        location_scale.nll(0.7, &location_scale_theta, &mut location_scale.workspace()),
        0.0,
        1.0e-12,
    );
    assert_close(
        mean_sd.cdf(0.7, &mean_sd_theta),
        location_scale.cdf(0.7, &location_scale_theta),
        0.0,
        1.0e-12,
    );
    assert_close(
        mean_sd.quantile(0.7, &mean_sd_theta),
        location_scale.quantile(0.7, &location_scale_theta),
        0.0,
        1.0e-10,
    );
}

#[test]
fn skew_student_t_mean_sd_matches_location_scale_equivalent() {
    let mean_sd = SkewStudentTMeanSdNuTau::new();
    let location_scale = SkewStudentTMuSigmaNuTau::new();
    let mean_sd_theta = SkewStudentTMeanSdTheta {
        mean: 0.2,
        sigma: 1.3,
        nu: 0.7,
        tau: 5.0,
    };
    let delta = mean_sd_theta.nu / mean_sd_theta.nu.hypot(1.0);
    let log_mean_factor = 0.5 * mean_sd_theta.tau.ln()
        + statrs::function::gamma::ln_gamma(0.5 * (mean_sd_theta.tau - 1.0))
        - 0.5 * std::f64::consts::PI.ln()
        - statrs::function::gamma::ln_gamma(0.5 * mean_sd_theta.tau);
    let standardized_mean = delta * log_mean_factor.exp();
    let standardized_variance =
        mean_sd_theta.tau / (mean_sd_theta.tau - 2.0) - standardized_mean * standardized_mean;
    let scale = mean_sd_theta.sigma / standardized_variance.sqrt();
    let location_scale_theta = SkewStudentTTheta {
        mu: mean_sd_theta.mean - scale * standardized_mean,
        sigma: scale,
        nu: mean_sd_theta.nu,
        tau: mean_sd_theta.tau,
    };

    assert_close(
        mean_sd.nll(0.7, &mean_sd_theta, &mut mean_sd.workspace()),
        location_scale.nll(0.7, &location_scale_theta, &mut location_scale.workspace()),
        0.0,
        1.0e-12,
    );
    assert_close(
        mean_sd.cdf(0.7, &mean_sd_theta),
        location_scale.cdf(0.7, &location_scale_theta),
        0.0,
        1.0e-12,
    );
    assert_close(
        mean_sd.quantile(0.7, &mean_sd_theta),
        location_scale.quantile(0.7, &location_scale_theta),
        0.0,
        1.0e-10,
    );
}

#[test]
fn extended_families_match_expected_symmetric_special_cases() {
    let normal = NormalMuSigma::new();
    let normal_theta = NormalTheta {
        mu: 0.2,
        sigma: 1.3,
    };

    let skew_normal = SkewNormalMuSigmaNu::new();
    let skew_normal_theta = SkewNormalTheta {
        mu: normal_theta.mu,
        sigma: normal_theta.sigma,
        nu: 0.0,
    };
    assert_close(
        skew_normal.nll(0.7, &skew_normal_theta, &mut skew_normal.workspace()),
        normal.nll(0.7, &normal_theta, &mut normal.workspace()),
        1.0e-9,
        1.0e-9,
    );
    assert_close(
        skew_normal.cdf(0.7, &skew_normal_theta),
        normal.cdf(0.7, &normal_theta),
        0.0,
        2.0e-7,
    );

    let skew_normal_mean_sd = SkewNormalMeanSdNu::new();
    let skew_normal_mean_sd_theta = SkewNormalMeanSdTheta {
        mean: normal_theta.mu,
        sigma: normal_theta.sigma,
        nu: 0.0,
    };
    assert_close(
        skew_normal_mean_sd.nll(
            0.7,
            &skew_normal_mean_sd_theta,
            &mut skew_normal_mean_sd.workspace(),
        ),
        normal.nll(0.7, &normal_theta, &mut normal.workspace()),
        1.0e-9,
        1.0e-9,
    );
    assert_close(
        skew_normal_mean_sd.cdf(0.7, &skew_normal_mean_sd_theta),
        normal.cdf(0.7, &normal_theta),
        0.0,
        2.0e-7,
    );

    let skew_t_mean_sd = SkewStudentTMeanSdNuTau::new();
    let student_t_stddev = StudentTMuSdTau::new();
    let skew_t_mean_sd_theta = SkewStudentTMeanSdTheta {
        mean: 0.2,
        sigma: 1.3,
        nu: 0.0,
        tau: 5.0,
    };
    let student_t_stddev_theta = StudentTMuSdTauTheta {
        mu: skew_t_mean_sd_theta.mean,
        sigma: skew_t_mean_sd_theta.sigma,
        tau: skew_t_mean_sd_theta.tau,
    };
    assert_close(
        skew_t_mean_sd.nll(0.7, &skew_t_mean_sd_theta, &mut skew_t_mean_sd.workspace()),
        student_t_stddev.nll(
            0.7,
            &student_t_stddev_theta,
            &mut student_t_stddev.workspace(),
        ),
        0.0,
        1.0e-12,
    );
    assert_close(
        skew_t_mean_sd.cdf(0.7, &skew_t_mean_sd_theta),
        student_t_stddev.cdf(0.7, &student_t_stddev_theta),
        0.0,
        2.0e-9,
    );

    let power_exponential = PowerExponentialMuSigmaNu::new();
    assert_close(
        power_exponential.nll(
            0.7,
            &PowerExponentialTheta {
                mu: normal_theta.mu,
                sigma: normal_theta.sigma,
                nu: 2.0,
            },
            &mut power_exponential.workspace(),
        ),
        normal.nll(0.7, &normal_theta, &mut normal.workspace()),
        1.0e-12,
        1.0e-12,
    );

    let shash = ShashMuSigmaNuTau::new();
    let shash_theta = ShashTheta {
        mu: normal_theta.mu,
        sigma: normal_theta.sigma,
        nu: 1.0,
        tau: 1.0,
    };
    assert_close(
        shash.nll(0.7, &shash_theta, &mut shash.workspace()),
        normal.nll(0.7, &normal_theta, &mut normal.workspace()),
        1.0e-12,
        1.0e-12,
    );
    assert_close(
        shash.cdf(0.7, &shash_theta),
        normal.cdf(0.7, &normal_theta),
        0.0,
        2.0e-7,
    );
}

#[test]
fn mean_sd_skew_parameterizations_handle_extreme_finite_skewness() {
    assert!(
        SkewNormalMeanSdNu::new()
            .nll(
                0.0,
                &SkewNormalMeanSdTheta {
                    mean: 0.0,
                    sigma: 1.0,
                    nu: 1.0e200,
                },
                &mut SkewNormalMeanSdNu::new().workspace(),
            )
            .is_finite()
    );
    assert!(
        SkewStudentTMeanSdNuTau::new()
            .nll(
                0.0,
                &SkewStudentTMeanSdTheta {
                    mean: 0.0,
                    sigma: 1.0,
                    nu: 1.0e200,
                    tau: 5.0,
                },
                &mut SkewStudentTMeanSdNuTau::new().workspace(),
            )
            .is_finite()
    );
}
