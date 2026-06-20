use gamlss_core::ParameterParts;

const REL_EPSILON: f64 = 1.0e-6;

/// Finite-difference fallback for eta gradients of numerically defined families.
///
/// Prefer analytic gradients for hot-path families. This helper is used for
/// distributions whose likelihoods involve CDFs, adaptive integration, or
/// infinite series in the current implementation. It tries a central
/// difference first, then falls back to one-sided differences when a nearby
/// probe crosses a distribution boundary.
pub(crate) fn finite_difference_gradient_eta<F, Eta, const K: usize>(
    eta: Eta,
    mut nll_at: F,
) -> [f64; K]
where
    Eta: Copy + ParameterParts<K>,
    F: FnMut(Eta) -> f64,
{
    let base_nll = nll_at(eta);
    let mut gradient = [0.0; K];
    let base = parts_to_array::<Eta, K>(eta);
    for index in 0..K {
        let step = REL_EPSILON * base[index].abs().max(1.0);
        let mut plus = base;
        let mut minus = base;
        plus[index] += step;
        minus[index] -= step;
        let plus_nll = nll_at(Eta::from_array(plus));
        let minus_nll = nll_at(Eta::from_array(minus));
        gradient[index] = if plus_nll.is_finite() && minus_nll.is_finite() {
            (plus_nll - minus_nll) / (2.0 * step)
        } else if plus_nll.is_finite() && base_nll.is_finite() {
            (plus_nll - base_nll) / step
        } else if minus_nll.is_finite() && base_nll.is_finite() {
            (base_nll - minus_nll) / step
        } else {
            f64::NAN
        };
    }

    gradient
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
