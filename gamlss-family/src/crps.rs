//! Internal numerical helpers for CRPS implementations.

use gamlss_special::integrate_finite;

#[derive(Debug, Clone, Copy)]
enum Tail {
    Left,
    Right,
}

/// Integrates the CRPS definition on the real line after mapping both tails to
/// the unit interval around the observation.
///
/// `mapping_scale` only controls the change of variables; it does not alter the
/// distribution. The left endpoint uses the left limit of the CDF so that the
/// helper also handles an atom exactly at the observation.
pub fn integrate_cdf_crps<F>(y: f64, mapping_scale: f64, cdf: F) -> f64
where
    F: Fn(f64) -> f64,
{
    if !y.is_finite() || mapping_scale <= 0.0 || !mapping_scale.is_finite() {
        return f64::NAN;
    }

    let left = integrate_finite(0.0, 1.0, |unit| {
        transformed_tail_integrand(unit, y, mapping_scale, Tail::Left, &cdf)
    });
    if !left.is_finite() {
        return f64::NAN;
    }
    let right = integrate_finite(0.0, 1.0, |unit| {
        transformed_tail_integrand(unit, y, mapping_scale, Tail::Right, &cdf)
    });
    let transformed_integral = left + right;
    if !transformed_integral.is_finite() {
        return f64::NAN;
    }

    mapping_scale * transformed_integral.max(0.0)
}

fn transformed_tail_integrand<F>(unit: f64, y: f64, mapping_scale: f64, tail: Tail, cdf: &F) -> f64
where
    F: Fn(f64) -> f64,
{
    let complement = 1.0 - unit;
    if complement <= 0.0 {
        return 0.0;
    }

    let distance = unit / complement;
    let offset = mapping_scale * distance;
    let x = match tail {
        Tail::Left => {
            if unit <= 0.0 {
                // Simpson quadrature evaluates the endpoint. Using the left
                // limit avoids assigning an atom at `y` to both integrals.
                y.next_down()
            } else {
                y - offset
            }
        }
        Tail::Right => y + offset,
    };
    if !x.is_finite() {
        return 0.0;
    }

    let probability = cdf(x);
    if !probability.is_finite() {
        return f64::NAN;
    }
    let tail_probability = match tail {
        Tail::Left => probability.clamp(0.0, 1.0),
        Tail::Right => (1.0 - probability).clamp(0.0, 1.0),
    };
    if tail_probability == 0.0 {
        return 0.0;
    }

    let ratio = tail_probability / complement;
    ratio * ratio
}

/// Exact CRPS sum for an integer-valued distribution with finite support
/// `0..=upper`, using log masses to avoid underflow at either endpoint.
///
/// The log masses are evaluated twice: the first pass normalizes them and the
/// second accumulates the score. This keeps the helper allocation-free.
pub fn finite_discrete_crps_from_log_pmf<F>(y: u64, upper: u64, log_pmf: F) -> f64
where
    F: Fn(u64) -> f64,
{
    if y > upper {
        return f64::NAN;
    }

    let mut log_normalizer = f64::NEG_INFINITY;
    for value in 0..=upper {
        log_normalizer = gamlss_special::log_add_exp(log_normalizer, log_pmf(value));
    }
    if !log_normalizer.is_finite() {
        return f64::NAN;
    }

    let mut cdf = 0.0;
    let mut score = 0.0;
    for value in 0..upper {
        cdf = (cdf + (log_pmf(value) - log_normalizer).exp()).min(1.0);
        let residual = if value < y { cdf } else { 1.0 - cdf };
        score = residual.mul_add(residual, score);
    }
    score.max(0.0)
}

/// CRPS of `zero_probability * delta_0 + (1 - zero_probability) * base`.
pub fn zero_inflated_crps(
    y: f64,
    zero_probability: f64,
    base_crps_at_observation: f64,
    base_crps_at_zero: f64,
) -> f64 {
    if !y.is_finite()
        || !(0.0..=1.0).contains(&zero_probability)
        || !base_crps_at_observation.is_finite()
        || !base_crps_at_zero.is_finite()
    {
        return f64::NAN;
    }
    let component_probability = 1.0 - zero_probability;
    let adjusted_component_score =
        (-zero_probability).mul_add(base_crps_at_zero, base_crps_at_observation);
    component_probability
        .mul_add(adjusted_component_score, zero_probability * y.abs())
        .max(0.0)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_special::unit_normal_cdf;

    use super::{finite_discrete_crps_from_log_pmf, integrate_cdf_crps, zero_inflated_crps};
    use crate::constants::{INV_SQRT_2_PI, INV_SQRT_PI};

    #[test]
    #[allow(clippy::suboptimal_flops)]
    fn transformed_cdf_integral_matches_normal_closed_form() {
        for y in [-3.0, 0.0, 0.7, 5.0] {
            let numeric = integrate_cdf_crps(y, 1.0, unit_normal_cdf);
            let cdf = unit_normal_cdf(y);
            let density = INV_SQRT_2_PI * (-0.5 * y * y).exp();
            let expected = y * (2.0 * cdf - 1.0) + 2.0 * density - INV_SQRT_PI;
            assert_relative_eq!(numeric, expected, epsilon = 2.0e-9);
        }
    }

    #[test]
    fn transformed_cdf_integral_is_independent_of_mapping_scale() {
        let y = 0.7;
        let expected = integrate_cdf_crps(y, 1.0, unit_normal_cdf);
        for mapping_scale in [0.2, 5.0] {
            assert_relative_eq!(
                integrate_cdf_crps(y, mapping_scale, unit_normal_cdf),
                expected,
                epsilon = 3.0e-9
            );
        }
    }

    #[test]
    #[allow(clippy::suboptimal_flops)]
    fn transformed_cdf_integral_uses_left_limit_at_atom() {
        let zero_probability = 0.3;
        let score = integrate_cdf_crps(0.0, 1.0, |x| {
            if x < 0.0 {
                0.0
            } else if x == 0.0 {
                zero_probability
            } else {
                zero_probability + (1.0 - zero_probability) * (1.0 - (-x).exp())
            }
        });
        assert_relative_eq!(
            score,
            0.5 * (1.0 - zero_probability).powi(2),
            epsilon = 2.0e-9
        );
    }

    #[test]
    fn finite_discrete_sum_matches_bernoulli_score() {
        let probability = 0.4_f64;
        let arbitrary_log_scale = 123.0;
        assert_relative_eq!(
            finite_discrete_crps_from_log_pmf(1, 1, |value| if value == 0 {
                (1.0 - probability).ln() + arbitrary_log_scale
            } else {
                probability.ln() + arbitrary_log_scale
            }),
            (1.0 - probability).powi(2),
            epsilon = 5.0e-15
        );
    }

    #[test]
    fn zero_inflated_identity_matches_cdf_integral() {
        let y = 0.8_f64;
        let zero_probability = 0.3;
        let exponential_crps =
            |observation: f64| 2.0_f64.mul_add((-observation).exp(), observation - 1.5);
        let identity = zero_inflated_crps(
            y,
            zero_probability,
            exponential_crps(y),
            exponential_crps(0.0),
        );
        let integral = integrate_cdf_crps(y, 1.0, |x| {
            if x < 0.0 {
                0.0
            } else {
                (1.0 - zero_probability).mul_add(-(-x).exp_m1(), zero_probability)
            }
        });

        assert_relative_eq!(identity, integral, epsilon = 2.0e-9);
    }
}
