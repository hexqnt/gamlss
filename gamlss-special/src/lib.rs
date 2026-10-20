#![forbid(unsafe_code)]
//! Special functions and numerical helpers for GAMLSS crates.
//!
//! This crate intentionally exposes small scalar `f64` free functions. Keeping
//! the API as free functions lets distribution crates call them without runtime
//! dispatch while still sharing one implementation of gamma/beta, normal-tail
//! and CDF inversion helpers.
//!
//! Invalid domains return `NaN` for helper-style APIs and non-finite likelihood
//! scale values where that convention is already part of the surrounding
//! distribution code.

const HALF_LOG_2_PI: f64 = 0.918_938_533_204_672_7;
const INV_SQRT_2_PI: f64 = 0.398_942_280_401_432_7;

mod beta;
mod exponential_integral;
mod gamma;
mod inversion;
mod normal;
mod probability;
mod quadrature;
mod student_t;

pub use beta::{
    ln_beta, ln_multivariate_beta, log_regularized_beta, regularized_beta,
    regularized_beta_complement,
};
pub use exponential_integral::exponential_integral_e1;
pub use gamma::{
    digamma, digamma_delta, digamma_minus_ln, ln_gamma, ln_gamma_delta, ln_gamma_stirling_residual,
    regularized_gamma_lower, regularized_gamma_upper,
};
pub use inversion::{discrete_quantile, invert_bounded_cdf, invert_positive_cdf, invert_real_cdf};
pub use normal::{
    log_ndtr, normal_mills_ratio, owens_t, unit_normal_cdf, unit_normal_log_pdf,
    unit_normal_log_sf, unit_normal_quantile, unit_normal_sf,
};
pub use probability::{
    baseline_softmax, bernoulli_kl, categorical_kl, included_count, is_nonnegative_integer,
    log_add_exp,
};
pub use quadrature::integrate_finite;
pub use student_t::{
    StandardStudentTKernel, student_t_cdf_standardized, student_t_log_cdf_standardized,
    student_t_log_pdf_standardized, student_t_nll_constant,
};

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::{
        StandardStudentTKernel, exponential_integral_e1, log_regularized_beta, regularized_beta,
        student_t_cdf_standardized, student_t_log_cdf_standardized, student_t_log_pdf_standardized,
    };

    #[test]
    fn exponential_integral_e1_matches_reference_values() {
        assert_relative_eq!(
            exponential_integral_e1(0.1),
            1.822_923_958_419_390_6,
            epsilon = 1.0e-14
        );
        assert_relative_eq!(
            exponential_integral_e1(1.0),
            0.219_383_934_395_520_29,
            epsilon = 1.0e-14
        );
        assert_relative_eq!(
            exponential_integral_e1(10.0),
            0.000_004_156_968_929_685_325,
            epsilon = 1.0e-18
        );
    }

    #[test]
    fn exponential_integral_e1_handles_invalid_domains() {
        assert!(exponential_integral_e1(-1.0).is_nan());
        assert!(exponential_integral_e1(f64::NAN).is_nan());
        assert!(exponential_integral_e1(0.0).is_infinite());
        assert!(exponential_integral_e1(f64::INFINITY).abs() <= f64::EPSILON);
    }

    #[test]
    fn log_regularized_beta_matches_probability_in_regular_range() {
        for (a, b, x) in [(0.7, 0.5, 0.01), (2.5, 1.3, 0.4), (4.0, 2.0, 0.9)] {
            assert_relative_eq!(
                log_regularized_beta(a, b, x),
                regularized_beta(a, b, x).ln(),
                epsilon = 1.0e-13
            );
        }
    }

    #[test]
    fn student_t_log_cdf_matches_cdf_and_retains_far_tail() {
        for (t, nu) in [(-8.0, 3.0), (-0.7, 5.0), (0.0, 7.0), (2.5, 11.0)] {
            assert_relative_eq!(
                student_t_log_cdf_standardized(t, nu),
                student_t_cdf_standardized(t, nu).ln(),
                epsilon = 1.0e-13
            );
        }
        let far_tail = student_t_log_cdf_standardized(-1.0e100, 5.0);
        assert!(far_tail.is_finite());
        assert!(far_tail < -1_000.0);

        let cauchy_tail = student_t_log_cdf_standardized(-1.0e100, 1.0);
        let cauchy_asymptotic = 100.0_f64.mul_add(-10.0_f64.ln(), -std::f64::consts::PI.ln());
        assert_relative_eq!(cauchy_tail, cauchy_asymptotic, epsilon = 1.0e-12);
        assert!(student_t_log_cdf_standardized(0.0, 0.0).is_nan());
        assert!(student_t_log_cdf_standardized(f64::NAN, 5.0).is_nan());
        let negative_infinity = student_t_log_cdf_standardized(f64::NEG_INFINITY, 5.0);
        assert!(negative_infinity.is_infinite() && negative_infinity.is_sign_negative());
        assert!(student_t_log_cdf_standardized(f64::INFINITY, 5.0).abs() <= f64::EPSILON);
        assert!(student_t_log_pdf_standardized(-1.0e308, 5.0).is_finite());
        let small_df_far_tail = student_t_cdf_standardized(-1.0e200, 0.01);
        assert!(small_df_far_tail > 0.0);
        assert!(small_df_far_tail < 0.5);
    }

    #[test]
    #[allow(clippy::float_cmp)] // The function API constructs the identical prepared kernel.
    fn prepared_student_t_density_matches_function_api() {
        for nu in [0.01, 1.0, 5.0, 1.0e16] {
            let kernel = StandardStudentTKernel::try_new(nu).unwrap();
            assert_eq!(kernel.degrees_of_freedom(), nu);
            assert_eq!(kernel.nll_constant(), super::student_t_nll_constant(nu));
            for t in [-1.0e308, -2.0, 0.0, 3.5, f64::INFINITY] {
                assert_eq!(kernel.log_pdf(t), student_t_log_pdf_standardized(t, nu));
            }
        }
        assert!(StandardStudentTKernel::try_new(0.0).is_none());
        assert!(StandardStudentTKernel::try_new(f64::INFINITY).is_none());
        assert!(StandardStudentTKernel::try_new(f64::NAN).is_none());
    }
}
