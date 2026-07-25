/// Opt-in contract for validating scalar observations before model construction.
///
/// The contract lives with a distribution family so higher-level builders do
/// not need to duplicate its observation-domain rules. Implementations should
/// be cheap, deterministic predicates; callers use static dispatch.
pub trait ScalarObservationDomain {
    /// Returns whether `observation` belongs to the family's scalar domain.
    fn observation_in_domain(&self, observation: f64) -> bool;
}

/// Returns `true` for finite values strictly greater than zero.
#[inline]
pub fn is_positive_finite(value: f64) -> bool {
    value > 0.0 && value.is_finite()
}

/// Returns `true` for finite probabilities in the closed interval `[0, 1]`.
#[inline]
pub fn is_probability(value: f64) -> bool {
    (0.0..=1.0).contains(&value)
}

/// Returns `true` for finite probabilities in the open interval `(0, 1)`.
#[inline]
pub fn is_strict_probability(value: f64) -> bool {
    value > 0.0 && value < 1.0
}

/// Returns `true` for a representable interior simplex with at least two components.
#[inline]
#[allow(clippy::cast_precision_loss)]
pub fn is_interior_simplex(values: &[f64]) -> bool {
    values.len() >= 2
        && values.iter().all(|value| *value > 0.0 && value.is_finite())
        && (values.iter().sum::<f64>() - 1.0).abs() <= 16.0 * f64::EPSILON * values.len() as f64
}

/// Returns `true` for a finite location and a positive finite scale.
#[inline]
pub fn is_finite_location_scale(location: f64, scale: f64) -> bool {
    location.is_finite() && is_positive_finite(scale)
}

#[cfg(test)]
mod tests {
    use super::ScalarObservationDomain;
    use crate::{
        BetaMeanPrecision, GammaMeanCv, InverseGaussianMeanCv, InverseGaussianMuShape,
        LogNormalMeanLogSd, NormalMuSigma, WeibullMeanShape,
    };

    fn assert_positive_domain(family: &impl ScalarObservationDomain) {
        assert!(family.observation_in_domain(1.0));
        assert!(!family.observation_in_domain(0.0));
        assert!(!family.observation_in_domain(-1.0));
        assert!(!family.observation_in_domain(f64::INFINITY));
        assert!(!family.observation_in_domain(f64::NAN));
    }

    #[test]
    fn formula_families_own_their_scalar_observation_domains() {
        let normal = NormalMuSigma::new();
        assert!(normal.observation_in_domain(-1.0));
        assert!(normal.observation_in_domain(0.0));
        assert!(!normal.observation_in_domain(f64::INFINITY));
        assert!(!normal.observation_in_domain(f64::NAN));

        let beta = BetaMeanPrecision::new();
        assert!(beta.observation_in_domain(0.5));
        assert!(!beta.observation_in_domain(0.0));
        assert!(!beta.observation_in_domain(1.0));
        assert!(!beta.observation_in_domain(f64::NAN));

        assert_positive_domain(&GammaMeanCv::new());
        assert_positive_domain(&LogNormalMeanLogSd::new());
        assert_positive_domain(&WeibullMeanShape::new());
        assert_positive_domain(&InverseGaussianMuShape::new());
        assert_positive_domain(&InverseGaussianMeanCv::new());
    }
}
