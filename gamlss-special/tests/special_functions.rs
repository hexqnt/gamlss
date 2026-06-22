use approx::assert_relative_eq;
use gamlss_special::{
    digamma, discrete_quantile, integrate_finite, invert_bounded_cdf, invert_positive_cdf,
    invert_real_cdf, ln_beta, ln_gamma, log_add_exp, log_ndtr, normal_mills_ratio, owens_t,
    regularized_beta, regularized_gamma_lower, student_t_cdf_standardized,
    student_t_log_pdf_standardized, student_t_nll_constant, unit_normal_cdf, unit_normal_quantile,
};
use statrs::distribution::{Continuous, ContinuousCDF, Normal as StatrsNormal, StudentsT};

fn assert_close(actual: f64, expected: f64, rel_tol: f64, abs_tol: f64) {
    let diff = (actual - expected).abs();
    let scale = actual.abs().max(expected.abs()).max(1.0);
    assert!(
        diff <= abs_tol.max(rel_tol * scale),
        "actual {actual:?} differs from expected {expected:?}; diff={diff:?}, rel_tol={rel_tol:?}, abs_tol={abs_tol:?}"
    );
}

fn assert_tail_close(actual: f64, expected: f64, rel_tol: f64) {
    let diff = (actual - expected).abs();
    let scale = expected.abs().max(f64::MIN_POSITIVE);
    assert!(
        diff <= rel_tol * scale,
        "actual {actual:?} differs from expected {expected:?}; diff={diff:?}, rel_tol={rel_tol:?}"
    );
}

#[test]
fn ln_gamma_matches_known_constants() {
    assert_relative_eq!(ln_gamma(1.0), 0.0, epsilon = 1.0e-12);
    assert_relative_eq!(
        ln_gamma(0.5),
        0.5 * std::f64::consts::PI.ln(),
        epsilon = 1.0e-12
    );
    assert_relative_eq!(ln_gamma(5.0), 24.0_f64.ln(), epsilon = 1.0e-12);
}

#[test]
fn ln_gamma_matches_statrs_reference_grid() {
    for value in [
        0.001_f64, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 25.0, 75.0, 150.0,
    ] {
        assert_close(
            ln_gamma(value),
            statrs::function::gamma::ln_gamma(value),
            2.0e-14,
            2.0e-13,
        );
    }
}

#[test]
fn ln_gamma_reflection_uses_absolute_gamma_and_rejects_poles() {
    assert_close(
        ln_gamma(-0.5),
        (2.0 * std::f64::consts::PI.sqrt()).ln(),
        0.0,
        2.0e-14,
    );
    assert_close(
        ln_gamma(-1.5),
        (4.0 * std::f64::consts::PI.sqrt() / 3.0).ln(),
        0.0,
        2.0e-14,
    );

    assert!(ln_gamma(0.0).is_nan());
    assert!(ln_gamma(-1.0).is_nan());
    assert!(ln_gamma(f64::NEG_INFINITY).is_nan());
    assert_eq!(ln_gamma(f64::INFINITY), f64::INFINITY);
}

#[test]
fn ln_beta_matches_known_constants() {
    assert_relative_eq!(ln_beta(1.0, 1.0), 0.0, epsilon = 1.0e-12);
    assert_relative_eq!(
        ln_beta(0.5, 0.5),
        std::f64::consts::PI.ln(),
        epsilon = 1.0e-12
    );
    assert!(ln_beta(0.0, 1.0).is_nan());
}

#[test]
fn ln_beta_matches_statrs_reference_grid() {
    for (a, b) in [
        (0.001_f64, 0.002_f64),
        (0.01, 0.1),
        (0.5, 0.5),
        (1.0, 1.0),
        (2.0, 7.0),
        (25.0, 30.0),
        (75.0, 80.0),
        (150.0, 2.5),
    ] {
        assert_close(
            ln_beta(a, b),
            statrs::function::beta::ln_beta(a, b),
            2.0e-14,
            2.0e-13,
        );
    }
}

#[test]
fn digamma_matches_known_constants_and_recurrence() {
    let euler_gamma = 0.577_215_664_901_532_9;

    assert_relative_eq!(digamma(1.0), -euler_gamma, epsilon = 1.0e-10);
    assert_relative_eq!(
        digamma(0.5),
        -euler_gamma - 2.0 * 2.0_f64.ln(),
        epsilon = 1.0e-10
    );
    assert_relative_eq!(digamma(4.25), digamma(3.25) + 1.0 / 3.25, epsilon = 1.0e-12);
}

#[test]
fn digamma_matches_statrs_reference_grid() {
    for value in [0.001_f64, 0.01, 0.1, 0.5, 1.0, 1.5, 8.0, 30.0, 120.0] {
        assert_close(
            digamma(value),
            statrs::function::gamma::digamma(value),
            2.0e-12,
            2.0e-11,
        );
    }
}

#[test]
fn unit_normal_cdf_matches_reference_points() {
    assert_relative_eq!(unit_normal_cdf(0.0), 0.5, epsilon = 1.0e-15);
    assert_relative_eq!(
        unit_normal_cdf(1.0),
        0.841_344_746_068_542_9,
        epsilon = 1.0e-15
    );
    assert_relative_eq!(
        unit_normal_cdf(-1.0),
        0.158_655_253_931_457_07,
        epsilon = 1.0e-15
    );
}

#[test]
fn unit_normal_cdf_and_quantile_match_statrs_reference_grid() {
    let reference = StatrsNormal::new(0.0, 1.0).unwrap();

    for z in [
        -12.0_f64, -10.0, -8.0, -6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0, 8.0, 10.0, 12.0,
    ] {
        assert_close(unit_normal_cdf(z), reference.cdf(z), 0.0, 2.0e-10);
    }

    for p in [
        1.0e-15,
        1.0e-12,
        1.0e-10,
        1.0e-8,
        1.0e-4,
        0.01,
        0.5,
        0.99,
        1.0 - 1.0e-4,
        1.0 - 1.0e-8,
        1.0 - 1.0e-10,
        1.0 - 1.0e-12,
        1.0 - 1.0e-15,
    ] {
        let quantile = unit_normal_quantile(p);
        assert_close(quantile, reference.inverse_cdf(p), 0.0, 2.0e-12);
        assert_close(unit_normal_cdf(quantile), p, 0.0, 2.0e-15);
    }
}

#[test]
fn unit_normal_cdf_matches_statrs_in_far_tails() {
    let reference = StatrsNormal::new(0.0, 1.0).unwrap();

    for z in [-8.0_f64, -10.0, -12.0] {
        assert_tail_close(unit_normal_cdf(z), reference.cdf(z), 1.0e-10);
    }

    for z in [8.0_f64, 10.0, 12.0] {
        assert_close(unit_normal_cdf(z), reference.cdf(z), 0.0, 2.0e-15);
    }
}

#[test]
fn log_ndtr_and_mills_ratio_stay_finite_in_left_tail() {
    for z in [-2.0, 0.0, 3.0] {
        assert_relative_eq!(log_ndtr(z).exp(), unit_normal_cdf(z), epsilon = 1.0e-14);
    }

    let left_tail = log_ndtr(-40.0);
    assert!(left_tail.is_finite());
    assert!(left_tail < -800.0);

    let mills = normal_mills_ratio(-40.0);
    assert!(mills.is_finite());
    assert!(mills > 40.0 && mills < 40.1, "mills was {mills}");

    assert_eq!(log_ndtr(f64::INFINITY), 0.0);
    assert_eq!(log_ndtr(f64::NEG_INFINITY), f64::NEG_INFINITY);
    assert!(log_ndtr(f64::NAN).is_nan());
}

#[test]
fn log_ndtr_matches_statrs_reference_where_cdf_is_representable() {
    let reference = StatrsNormal::new(0.0, 1.0).unwrap();
    for z in [-10.0_f64, -8.0, -5.0, -2.0, 0.0, 2.0, 5.0] {
        assert_close(log_ndtr(z), reference.cdf(z).ln(), 0.0, 1.0e-4);
    }
}

#[test]
fn standardized_student_t_helpers_match_statrs_reference() {
    for nu in [1.5_f64, 2.5, 5.0, 30.0] {
        let reference = StudentsT::new(0.0, 1.0, nu).unwrap();
        assert_close(
            (-student_t_nll_constant(nu)).exp(),
            reference.pdf(0.0),
            2.0e-13,
            2.0e-13,
        );

        for t in [-4.0_f64, -1.0, 0.0, 1.0, 4.0] {
            assert_close(
                student_t_log_pdf_standardized(t, nu).exp(),
                reference.pdf(t),
                2.0e-13,
                2.0e-13,
            );
            assert_close(
                student_t_cdf_standardized(t, nu),
                reference.cdf(t),
                2.0e-12,
                2.0e-12,
            );
        }
    }

    assert!(student_t_log_pdf_standardized(0.0, 0.0).is_infinite());
    assert!(student_t_cdf_standardized(0.0, 0.0).is_nan());
}

#[test]
fn unit_normal_quantile_matches_reference_points() {
    assert_relative_eq!(unit_normal_quantile(0.5), 0.0, epsilon = 1.0e-9);
    assert_relative_eq!(unit_normal_quantile(0.841_344_746), 1.0, epsilon = 1.0e-6);
    assert_relative_eq!(unit_normal_quantile(0.158_655_254), -1.0, epsilon = 1.0e-6);

    let upper = unit_normal_quantile(1.0 - 1.0e-12);
    let lower = unit_normal_quantile(1.0e-12);
    assert!(upper.is_finite());
    assert_relative_eq!(upper, -lower, epsilon = 5.0e-6);
}

#[test]
fn regularized_beta_matches_simple_cases() {
    assert_relative_eq!(regularized_beta(1.0, 1.0, 0.25), 0.25, epsilon = 1.0e-14);
    assert_relative_eq!(regularized_beta(2.0, 1.0, 0.5), 0.25, epsilon = 1.0e-14);
    assert_relative_eq!(regularized_beta(1.0, 2.0, 0.5), 0.75, epsilon = 1.0e-14);
}

#[test]
fn regularized_beta_matches_statrs_reference_grid() {
    for (a, b, x) in [
        (0.1_f64, 0.2_f64, 1.0e-8_f64),
        (0.1, 0.2, 0.01),
        (0.1, 5.0, 0.8),
        (0.5, 0.5, 0.99),
        (2.0, 7.0, 0.25),
        (25.0, 30.0, 0.45),
        (75.0, 80.0, 0.55),
    ] {
        assert_close(
            regularized_beta(a, b, x),
            statrs::function::beta::beta_reg(a, b, x),
            2.0e-11,
            2.0e-12,
        );
    }
}

#[test]
fn regularized_gamma_lower_matches_simple_cases() {
    assert_relative_eq!(regularized_gamma_lower(1.0, 2.0), 1.0 - (-2.0_f64).exp());
    assert_relative_eq!(
        regularized_gamma_lower(2.0, 2.0),
        1.0 - 3.0 * (-2.0_f64).exp(),
        epsilon = 1.0e-14
    );
    assert_eq!(regularized_gamma_lower(2.0, 0.0), 0.0);
}

#[test]
fn regularized_gamma_lower_matches_statrs_reference_grid() {
    for (a, x) in [
        (0.1_f64, 1.0e-8_f64),
        (0.1, 0.01),
        (0.5, 0.5),
        (1.0, 20.0),
        (2.5, 1.5),
        (10.0, 12.0),
        (50.0, 45.0),
        (100.0, 130.0),
    ] {
        assert_close(
            regularized_gamma_lower(a, x),
            statrs::function::gamma::gamma_lr(a, x),
            2.0e-11,
            2.0e-12,
        );
    }
}

#[test]
fn regularized_gamma_lower_is_bounded_near_continued_fraction_singularity() {
    let value = regularized_gamma_lower(2.5, 1.5);

    assert!(value.is_finite());
    assert!((0.0..=1.0).contains(&value));
}

#[test]
fn owens_t_satisfies_basic_symmetry_and_special_cases() {
    for h in [-4.0_f64, -1.0, 0.0, 1.0, 4.0] {
        assert_eq!(owens_t(h, 0.0), 0.0);
        assert_close(owens_t(h, 1.3), owens_t(-h, 1.3), 0.0, 1.0e-14);
        assert_close(owens_t(h, -1.3), -owens_t(h, 1.3), 0.0, 1.0e-14);
    }

    for a in [-2.0_f64, -0.5, 0.5, 2.0] {
        assert_close(
            owens_t(0.0, a),
            a.atan() / (2.0 * std::f64::consts::PI),
            0.0,
            1.0e-12,
        );
    }
}

#[test]
fn discrete_quantile_returns_generalized_inverse() {
    assert_eq!(discrete_quantile(0.1, 10, |k| (k + 1) as f64 / 10.0), 0.0);
    assert_eq!(discrete_quantile(0.2, 10, |k| (k + 1) as f64 / 10.0), 1.0);
    assert_eq!(discrete_quantile(1.0, 9, |k| (k + 1) as f64 / 10.0), 9.0);
    assert!(discrete_quantile(1.0, 8, |k| (k + 1) as f64 / 10.0).is_nan());
}

#[test]
fn cdf_inversion_helpers_find_quantiles() {
    assert_relative_eq!(
        invert_bounded_cdf(0.25, 0.0, 1.0, |x| x),
        0.25,
        epsilon = 1.0e-14
    );
    assert_relative_eq!(
        invert_positive_cdf(0.75, |x| 1.0 - (-x).exp()),
        -(0.25_f64).ln(),
        epsilon = 1.0e-12
    );
    assert_relative_eq!(
        invert_real_cdf(0.75, unit_normal_cdf),
        unit_normal_quantile(0.75),
        epsilon = 1.0e-6
    );
}

#[test]
fn cdf_inversion_helpers_reject_invalid_brackets_and_nan_cdf_values() {
    assert!(invert_bounded_cdf(0.5, 1.0, 0.0, |x| x).is_nan());
    assert!(invert_bounded_cdf(0.5, 0.0, 1.0, |_| f64::NAN).is_nan());
    assert!(invert_bounded_cdf(0.5, 0.0, 1.0, |x| 1.0 - x).is_nan());
    assert!(invert_bounded_cdf(0.5, 0.0, 1.0, |x| 0.75 + 0.25 * x).is_nan());

    assert!(invert_positive_cdf(0.5, |_| f64::NAN).is_nan());
    assert!(invert_real_cdf(0.5, |_| f64::NAN).is_nan());
}

#[test]
fn integrate_finite_matches_polynomial_and_handles_invalid_bounds() {
    assert_relative_eq!(
        integrate_finite(0.0, 1.0, |x| x * x),
        1.0 / 3.0,
        epsilon = 1.0e-10
    );
    assert_eq!(integrate_finite(2.0, 2.0, |_| 1.0), 0.0);
    assert!(integrate_finite(2.0, 1.0, |_| 1.0).is_nan());
}

#[test]
fn log_add_exp_combines_log_terms_without_underflow() {
    assert_relative_eq!(
        log_add_exp(-1000.0, -1001.0),
        -1000.0 + (-1.0_f64).exp().ln_1p(),
        epsilon = 1.0e-12
    );

    let combined = log_add_exp(f64::NEG_INFINITY, -3.0);
    assert_relative_eq!(combined, -3.0, epsilon = 1.0e-12);
}
