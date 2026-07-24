use gamlss_core::PositiveLink;

/// Evaluates a positive inverse link together with its natural logarithm.
///
/// Built-in links provide the logarithm directly when that avoids a
/// transcendental round trip. The fallback preserves support for downstream
/// custom [`PositiveLink`] implementations.
#[inline]
pub fn positive_inverse_and_log<L>(eta: f64) -> (f64, f64)
where
    L: PositiveLink<f64>,
{
    let (inverse, log_inverse) = L::inverse_and_log_inverse(eta);
    (inverse, log_inverse.unwrap_or_else(|| inverse.ln()))
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{Link, PositiveLink};

    use super::positive_inverse_and_log;

    struct CustomPositiveLink;

    impl Link<f64> for CustomPositiveLink {
        fn inverse(eta: f64) -> f64 {
            eta.exp() + 1.0
        }

        fn derivative_inverse(eta: f64) -> f64 {
            eta.exp()
        }
    }

    impl PositiveLink<f64> for CustomPositiveLink {
        fn derivative_log_inverse(eta: f64) -> f64 {
            let exp_eta = eta.exp();
            exp_eta / (exp_eta + 1.0)
        }
    }

    #[test]
    fn downstream_link_fallback_computes_the_missing_logarithm() {
        let (inverse, log_inverse) = positive_inverse_and_log::<CustomPositiveLink>(0.0);
        assert_relative_eq!(inverse, 2.0);
        assert_relative_eq!(log_inverse, std::f64::consts::LN_2);
    }
}
