//! Adaptive finite-interval quadrature.

/// Integrates a finite interval with adaptive Simpson quadrature.
///
/// Returns `NaN` for non-finite bounds, reversed bounds, or non-finite function
/// evaluations.
#[must_use]
pub fn integrate_finite<F>(lower: f64, upper: f64, mut function: F) -> f64
where
    F: FnMut(f64) -> f64,
{
    const EPSILON: f64 = 1.0e-10;
    const MAX_DEPTH: u32 = 24;

    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return f64::NAN;
    }

    #[allow(clippy::float_cmp)]
    if lower == upper {
        return 0.0;
    }

    let mid = lower.midpoint(upper);
    let f_lower = function(lower);
    let f_mid = function(mid);
    let f_upper = function(upper);
    if !f_lower.is_finite() || !f_mid.is_finite() || !f_upper.is_finite() {
        return f64::NAN;
    }

    let whole = simpson(lower, upper, f_lower, f_mid, f_upper);
    adaptive_simpson(
        &mut function,
        lower,
        upper,
        [f_lower, f_mid, f_upper],
        whole,
        EPSILON,
        MAX_DEPTH,
    )
}

fn adaptive_simpson<F>(
    function: &mut F,
    lower: f64,
    upper: f64,
    values: [f64; 3],
    whole: f64,
    epsilon: f64,
    depth: u32,
) -> f64
where
    F: FnMut(f64) -> f64,
{
    let [f_lower, f_mid, f_upper] = values;
    let mid = lower.midpoint(upper);
    let left_mid = lower.midpoint(mid);
    let right_mid = mid.midpoint(upper);
    let f_left_mid = function(left_mid);
    let f_right_mid = function(right_mid);
    if !f_left_mid.is_finite() || !f_right_mid.is_finite() {
        return f64::NAN;
    }

    let left = simpson(lower, mid, f_lower, f_left_mid, f_mid);
    let right = simpson(mid, upper, f_mid, f_right_mid, f_upper);
    let delta = left + right - whole;
    if depth == 0 || delta.abs() <= 15.0 * epsilon {
        return left + right + delta / 15.0;
    }

    let left_integral = adaptive_simpson(
        function,
        lower,
        mid,
        [f_lower, f_left_mid, f_mid],
        left,
        0.5 * epsilon,
        depth - 1,
    );
    let right_integral = adaptive_simpson(
        function,
        mid,
        upper,
        [f_mid, f_right_mid, f_upper],
        right,
        0.5 * epsilon,
        depth - 1,
    );

    left_integral + right_integral
}

#[inline]
fn simpson(lower: f64, upper: f64, f_lower: f64, f_mid: f64, f_upper: f64) -> f64 {
    (upper - lower) * (4.0f64.mul_add(f_mid, f_lower) + f_upper) / 6.0
}
