use gamlss_core::ParameterParts;

const REL_EPSILON: f64 = 1.0e-6;
const CENTRAL_DIFFERENCE_RETRIES: usize = 4;
const STEP_SHRINK_FACTOR: f64 = 0.5;

/// Finite-difference fallback for eta gradients of numerically defined families.
///
/// Prefer analytic gradients for hot-path families. This helper is used for
/// distributions whose likelihoods involve CDFs, adaptive integration, or
/// infinite series in the current implementation. It tries central differences
/// with progressively smaller steps first, then falls back to one-sided
/// differences when nearby probes still cross a distribution boundary.
pub(crate) fn finite_difference_gradient_eta<F, Eta, const K: usize>(
    eta: Eta,
    mut nll_at: F,
) -> [f64; K]
where
    Eta: Copy + ParameterParts<K>,
    F: FnMut(Eta) -> f64,
{
    let mut base_nll = None;
    let mut gradient = [0.0; K];
    let eta_parts = parts_to_array::<Eta, K>(eta);
    for index in 0..K {
        gradient[index] =
            finite_difference_component(eta, eta_parts, index, &mut base_nll, &mut nll_at);
    }

    gradient
}

fn finite_difference_component<F, Eta, const K: usize>(
    eta: Eta,
    eta_parts: [f64; K],
    index: usize,
    base_nll: &mut Option<f64>,
    nll_at: &mut F,
) -> f64
where
    Eta: Copy + ParameterParts<K>,
    F: FnMut(Eta) -> f64,
{
    let mut step = REL_EPSILON * eta_parts[index].abs().max(1.0);
    let mut one_sided_candidate = None;

    for _ in 0..=CENTRAL_DIFFERENCE_RETRIES {
        let mut plus = eta_parts;
        let mut minus = eta_parts;
        plus[index] += step;
        minus[index] -= step;
        let plus_nll = nll_at(Eta::from_array(plus));
        let minus_nll = nll_at(Eta::from_array(minus));

        if plus_nll.is_finite() && minus_nll.is_finite() {
            return (plus_nll - minus_nll) / (2.0 * step);
        }

        if plus_nll.is_finite() || minus_nll.is_finite() {
            one_sided_candidate = Some((step, plus_nll, minus_nll));
        }

        step *= STEP_SHRINK_FACTOR;
    }

    match one_sided_candidate {
        Some((step, plus_nll, minus_nll)) => {
            let base_nll = cached_base_nll(base_nll, eta, nll_at);

            if plus_nll.is_finite() && base_nll.is_finite() {
                (plus_nll - base_nll) / step
            } else if minus_nll.is_finite() && base_nll.is_finite() {
                (base_nll - minus_nll) / step
            } else {
                f64::NAN
            }
        }
        None => f64::NAN,
    }
}

fn cached_base_nll<F, Eta>(base_nll: &mut Option<f64>, eta: Eta, nll_at: &mut F) -> f64
where
    Eta: Copy,
    F: FnMut(Eta) -> f64,
{
    match *base_nll {
        Some(value) => value,
        None => {
            let value = nll_at(eta);
            *base_nll = Some(value);
            value
        }
    }
}

fn parts_to_array<Eta, const K: usize>(eta: Eta) -> [f64; K]
where
    Eta: ParameterParts<K>,
{
    let mut values = [0.0; K];
    for (index, value) in values.iter_mut().enumerate() {
        *value = eta.part(index);
    }
    values
}

#[cfg(test)]
mod tests {
    use super::{CENTRAL_DIFFERENCE_RETRIES, finite_difference_gradient_eta};

    #[test]
    fn central_difference_path_does_not_evaluate_base_nll() {
        let mut calls = 0;
        let gradient =
            finite_difference_gradient_eta::<_, (f64, f64), 2>((1.0, 2.0), |(first, second)| {
                calls += 1;
                second.mul_add(second, first * first)
            });

        assert_eq!(calls, 4);
        assert!((gradient[0] - 2.0).abs() < 1.0e-9);
        assert!((gradient[1] - 4.0).abs() < 1.0e-9);
    }

    #[test]
    fn one_sided_fallback_evaluates_base_nll_lazily() {
        let mut calls = 0;
        let gradient = finite_difference_gradient_eta::<_, f64, 1>(1.0, |probe| {
            calls += 1;
            if probe < 1.0 {
                f64::INFINITY
            } else {
                probe * probe
            }
        });

        assert_eq!(calls, 2 * (CENTRAL_DIFFERENCE_RETRIES + 1) + 1);
        assert!((gradient[0] - 2.0).abs() < 1.0e-5);
    }

    #[test]
    fn central_difference_retries_with_smaller_step_before_fallback() {
        let mut calls = 0;
        let gradient = finite_difference_gradient_eta::<_, f64, 1>(1.0, |probe| {
            calls += 1;
            if (probe - 1.0).abs() <= 3.0e-7 {
                probe * probe
            } else {
                f64::INFINITY
            }
        });

        assert_eq!(calls, 6);
        assert!((gradient[0] - 2.0).abs() < 1.0e-9);
    }
}
