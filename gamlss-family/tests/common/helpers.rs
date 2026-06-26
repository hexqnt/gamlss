#![allow(dead_code)]

use gamlss_core::{Family, HasCdf, HasDensity, HasQuantile, ParameterParts};
use gamlss_family::NegativeBinomialTheta;
use proptest::prelude::*;
use statrs::distribution::{Continuous, ContinuousCDF, Discrete, DiscreteCDF};

pub const CASES: u32 = 32;
pub const PROB_MIN: f64 = 1.0e-8;
pub const PROB_MAX: f64 = 1.0 - PROB_MIN;
pub const FD_REL_TOL: f64 = 2.0e-5;
pub const FD_ABS_TOL: f64 = 2.0e-5;
pub const NEW_FAMILY_FD_REL_TOL: f64 = 8.0e-4;
pub const NEW_FAMILY_FD_ABS_TOL: f64 = 8.0e-4;
pub const REAL_REFERENCE_POINTS: [f64; 5] = [-3.0, -1.0, 0.0, 1.0, 3.0];
pub const POSITIVE_REFERENCE_POINTS: [f64; 5] = [0.1, 0.5, 1.0, 2.0, 4.0];
pub const REFERENCE_PROBABILITIES: [f64; 5] = [0.01, 0.1, 0.5, 0.9, 0.99];
pub const TAIL_PROBABILITIES: [f64; 6] = [
    1.0e-12,
    1.0e-10,
    1.0e-8,
    1.0 - 1.0e-8,
    1.0 - 1.0e-10,
    1.0 - 1.0e-12,
];

pub struct ContinuousReferenceTolerances {
    pub cdf_abs: f64,
    pub density_rel: f64,
    pub density_abs: f64,
    pub quantile_abs: f64,
}

pub fn proptest_config() -> ProptestConfig {
    ProptestConfig {
        cases: CASES,
        ..ProptestConfig::default()
    }
}

pub fn assert_close(actual: f64, expected: f64, rel_tol: f64, abs_tol: f64) {
    assert_close_with_context(actual, expected, rel_tol, abs_tol, "");
}

pub fn assert_close_with_context(
    actual: f64,
    expected: f64,
    rel_tol: f64,
    abs_tol: f64,
    context: &str,
) {
    let diff = (actual - expected).abs();
    let scale = actual.abs().max(expected.abs()).max(1.0);
    let within_tolerance = diff <= abs_tol.max(rel_tol * scale);
    if context.is_empty() {
        assert!(
            within_tolerance,
            "actual {actual:?} differs from expected {expected:?}; diff={diff:?}, rel_tol={rel_tol:?}, abs_tol={abs_tol:?}"
        );
    } else {
        assert!(
            within_tolerance,
            "{context} actual {actual:?} differs from expected {expected:?}; diff={diff:?}, rel_tol={rel_tol:?}, abs_tol={abs_tol:?}"
        );
    }
}

pub fn assert_nll_eta_matches_theta<F, const K: usize>(family: &F, y: f64, eta: [f64; K])
where
    F: for<'obs> Family<Observation<'obs> = f64>,
    F::Eta: Copy + ParameterParts<K>,
{
    let eta = F::Eta::from_array(eta);
    let via_eta = family.nll_eta(y, eta);
    let via_theta = family.nll(y, family.theta(eta));

    if via_eta.is_finite() && via_theta.is_finite() {
        assert_close(via_eta, via_theta, 1.0e-12, 1.0e-12);
        return;
    }

    assert!(
        (via_eta.is_nan() && via_theta.is_nan())
            || (via_eta.is_infinite()
                && via_theta.is_infinite()
                && via_eta.is_sign_positive() == via_theta.is_sign_positive()),
        "nll_eta and nll(theta(eta)) disagree: {via_eta:?} vs {via_theta:?}"
    );
}

pub fn assert_gradient_matches_finite_difference<F, const K: usize>(
    family: &F,
    y: f64,
    eta: [f64; K],
) where
    F: for<'obs> Family<Observation<'obs> = f64>,
    F::Eta: Copy + ParameterParts<K>,
    F::NllGradientEta: ParameterParts<K>,
{
    assert_gradient_matches_finite_difference_with(
        family,
        y,
        eta,
        FD_REL_TOL,
        FD_ABS_TOL,
        |value| f64::EPSILON.sqrt() * value.abs().max(1.0),
    );
}

pub fn assert_new_family_gradient_matches_finite_difference<F, const K: usize>(
    family: &F,
    y: f64,
    eta: [f64; K],
) where
    F: for<'obs> Family<Observation<'obs> = f64>,
    F::Eta: Copy + ParameterParts<K>,
    F::NllGradientEta: ParameterParts<K>,
{
    assert_gradient_matches_finite_difference_with(
        family,
        y,
        eta,
        NEW_FAMILY_FD_REL_TOL,
        NEW_FAMILY_FD_ABS_TOL,
        |value| 1.0e-5 * value.abs().max(1.0),
    );
}

fn assert_gradient_matches_finite_difference_with<F, const K: usize, S>(
    family: &F,
    y: f64,
    eta: [f64; K],
    rel_tol: f64,
    abs_tol: f64,
    step: S,
) where
    F: for<'obs> Family<Observation<'obs> = f64>,
    F::Eta: Copy + ParameterParts<K>,
    F::NllGradientEta: ParameterParts<K>,
    S: Fn(f64) -> f64,
{
    let (_, gradient) = family.nll_and_gradient_eta(y, F::Eta::from_array(eta));

    for index in 0..K {
        let epsilon = step(eta[index]);
        let mut plus = eta;
        plus[index] += epsilon;
        let mut minus = eta;
        minus[index] -= epsilon;

        let finite_difference = (family.nll_eta(y, F::Eta::from_array(plus))
            - family.nll_eta(y, F::Eta::from_array(minus)))
            / (2.0 * epsilon);
        let actual = gradient.part(index);

        assert!(
            actual.is_finite(),
            "gradient component {index} is not finite: {actual:?}"
        );
        assert!(
            finite_difference.is_finite(),
            "finite-difference component {index} is not finite: {finite_difference:?}"
        );
        assert_close(actual, finite_difference, rel_tol, abs_tol);
    }
}

pub fn assert_continuous_inverse<F>(family: &F, p: f64, theta: F::Theta, tolerance: f64)
where
    F: for<'obs> Family<Observation<'obs> = f64> + HasCdf + HasQuantile,
    F::Theta: Copy,
{
    let y = family.quantile(p, theta);
    assert!(
        y.is_finite(),
        "{} quantile({p}) returned {y:?}",
        std::any::type_name::<F>()
    );
    let cdf = family.cdf(y, theta);
    assert_close_with_context(
        cdf,
        p,
        0.0,
        tolerance,
        &format!("{} y={y:?}", std::any::type_name::<F>()),
    );
}

pub fn assert_discrete_inverse<F>(family: &F, p: f64, theta: F::Theta)
where
    F: for<'obs> Family<Observation<'obs> = f64> + HasCdf + HasQuantile,
    F::Theta: Copy,
{
    let q = family.quantile(p, theta);
    assert!(q.is_finite(), "quantile({p}) returned {q:?}");
    assert_eq!(q.fract(), 0.0, "discrete quantile should be integral");

    let cdf_at_q = family.cdf(q, theta);
    assert!(
        cdf_at_q + 1.0e-14 >= p,
        "cdf(q) must be at least p; p={p:?}, q={q:?}, cdf={cdf_at_q:?}"
    );
    if q > 0.0 {
        let cdf_below_q = family.cdf(q - 1.0, theta);
        assert!(
            cdf_below_q < p + 1.0e-14,
            "cdf(q - 1) must be below p; p={p:?}, q={q:?}, cdf={cdf_below_q:?}"
        );
    }
}

pub fn assert_cdf_monotone<F>(family: &F, theta: F::Theta, ys: &[f64])
where
    F: for<'obs> Family<Observation<'obs> = f64> + HasCdf,
    F::Theta: Copy,
{
    let mut previous = f64::NEG_INFINITY;
    for &y in ys {
        let cdf = family.cdf(y, theta);
        assert!(cdf.is_finite(), "cdf({y}) returned {cdf:?}");
        assert!(
            cdf + 1.0e-14 >= previous,
            "cdf is not monotone at y={y:?}: {cdf:?} < {previous:?}"
        );
        previous = cdf;
    }
}

pub fn integrate_simpson<F>(lower: f64, upper: f64, intervals: usize, mut f: F) -> f64
where
    F: FnMut(f64) -> f64,
{
    assert_eq!(intervals % 2, 0);
    let step = (upper - lower) / intervals as f64;
    let mut sum = f(lower) + f(upper);
    for index in 1..intervals {
        let weight = if index % 2 == 0 { 2.0 } else { 4.0 };
        sum += weight * f(lower + index as f64 * step);
    }
    sum * step / 3.0
}

pub fn assert_density_integrates_over_quantile_bracket<F>(
    family: &F,
    theta: F::Theta,
    tail_probability: f64,
    tolerance: f64,
) where
    F: for<'obs> Family<Observation<'obs> = f64> + HasCdf + HasQuantile + HasDensity,
    F::Theta: Copy,
{
    let lower = family.quantile(tail_probability, theta);
    let upper = family.quantile(1.0 - tail_probability, theta);
    assert!(
        lower.is_finite() && upper.is_finite() && lower < upper,
        "invalid integration bracket [{lower:?}, {upper:?}]"
    );

    let integral = integrate_simpson(lower, upper, 1024, |y| family.density(y, theta));
    let expected = family.cdf(upper, theta) - family.cdf(lower, theta);
    assert_close(integral, expected, 0.0, tolerance);
    assert_close(integral, 1.0, 0.0, tolerance + 2.0 * tail_probability);
}

pub fn assert_discrete_mass_sums_to_one<F>(
    family: &F,
    theta: F::Theta,
    upper_p: f64,
    tolerance: f64,
) where
    F: for<'obs> Family<Observation<'obs> = f64> + HasQuantile + HasDensity,
    F::Theta: Copy,
{
    let upper = family.quantile(upper_p, theta);
    assert!(
        upper.is_finite() && upper >= 0.0,
        "invalid discrete upper quantile {upper:?}"
    );
    let upper = upper as u64;
    let mass = (0..=upper)
        .map(|count| family.density(count as f64, theta))
        .sum::<f64>();
    assert_close(mass, upper_p, 0.0, tolerance + (1.0 - upper_p));
    assert_close(mass, 1.0, 0.0, tolerance + (1.0 - upper_p));
}

pub fn nb_success_probability(theta: NegativeBinomialTheta) -> f64 {
    theta.shape / (theta.shape + theta.mu)
}

pub fn statrs_discrete_quantile<F>(p: f64, mut cdf: F) -> u64
where
    F: FnMut(u64) -> f64,
{
    let mut high = 1_u64;

    #[allow(clippy::while_float)]
    while cdf(high) < p {
        high = high.saturating_mul(2);
        assert!(high > 1, "discrete reference quantile search overflowed");
    }

    let mut low = 0_u64;
    while low < high {
        let mid = low + (high - low) / 2;
        if cdf(mid) < p {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    low
}

pub fn assert_continuous_statrs_reference<F, R>(
    family: &F,
    theta: F::Theta,
    reference: &R,
    ys: &[f64],
    tolerances: ContinuousReferenceTolerances,
) where
    F: for<'obs> Family<Observation<'obs> = f64> + HasCdf + HasDensity + HasQuantile,
    F::Theta: Copy,
    R: Continuous<f64, f64> + ContinuousCDF<f64, f64>,
{
    for &y in ys {
        assert_close(
            family.cdf(y, theta),
            reference.cdf(y),
            0.0,
            tolerances.cdf_abs,
        );
        assert_close(
            family.density(y, theta),
            reference.pdf(y),
            tolerances.density_rel,
            tolerances.density_abs,
        );
    }

    for p in REFERENCE_PROBABILITIES {
        assert_close(
            family.quantile(p, theta),
            reference.inverse_cdf(p),
            0.0,
            tolerances.quantile_abs,
        );
    }
}

pub fn assert_discrete_statrs_reference<F, R>(
    family: &F,
    theta: F::Theta,
    reference: &R,
    counts: impl Iterator<Item = u64>,
    tolerance: f64,
) where
    F: for<'obs> Family<Observation<'obs> = f64> + HasCdf + HasDensity,
    F::Theta: Copy,
    R: Discrete<u64, f64> + DiscreteCDF<u64, f64>,
{
    for count in counts {
        assert_close(
            family.cdf(count as f64, theta),
            reference.cdf(count),
            0.0,
            tolerance,
        );
        assert_close(
            family.density(count as f64, theta),
            reference.pmf(count),
            tolerance,
            tolerance,
        );
    }
}

pub fn assert_discrete_or_continuous_generalized_inverse<F>(
    family: &F,
    p: f64,
    theta: F::Theta,
    tolerance: f64,
) where
    F: for<'obs> Family<Observation<'obs> = f64> + HasCdf + HasQuantile,
    F::Theta: Copy,
{
    let q = family.quantile(p, theta);
    assert!(q.is_finite(), "quantile({p}) returned {q:?}");
    assert!(
        family.cdf(q, theta) + tolerance >= p,
        "cdf(q) must be at least p; p={p:?}, q={q:?}, cdf={:?}",
        family.cdf(q, theta)
    );
}
