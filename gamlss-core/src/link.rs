const INITIAL_POSITIVE_FLOOR: f64 = 1.0e-12;
const INITIAL_PROBABILITY_FLOOR: f64 = 1.0e-12;

/// Identity link: `theta = eta`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Identity;

impl Link<f64> for Identity {
    #[inline]
    fn inverse(eta: f64) -> f64 {
        eta
    }

    #[inline]
    fn derivative_inverse(_: f64) -> f64 {
        1.0
    }
}

impl InitialEtaFromTheta<f64> for Identity {
    #[inline]
    fn initial_eta_from_theta(theta: f64) -> f64 {
        theta
    }
}

/// Log link: `theta = exp(eta)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Log;

impl Link<f64> for Log {
    #[inline]
    fn inverse(eta: f64) -> f64 {
        eta.exp()
    }

    #[inline]
    fn derivative_inverse(eta: f64) -> f64 {
        eta.exp()
    }
}

impl PositiveLink<f64> for Log {}

impl InitialEtaFromTheta<f64> for Log {
    #[inline]
    fn initial_eta_from_theta(theta: f64) -> f64 {
        theta.max(INITIAL_POSITIVE_FLOOR).ln()
    }
}

/// Numerically stable positive link: `theta = ln(1 + exp(eta))`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Softplus;

impl Link<f64> for Softplus {
    #[inline]
    fn inverse(eta: f64) -> f64 {
        if eta > 30.0 {
            eta
        } else if eta < -30.0 {
            eta.exp()
        } else {
            eta.exp().ln_1p()
        }
    }

    #[inline]
    fn derivative_inverse(eta: f64) -> f64 {
        if eta >= 0.0 {
            1.0 / (1.0 + (-eta).exp())
        } else {
            let exp_eta = eta.exp();
            exp_eta / (1.0 + exp_eta)
        }
    }
}

impl PositiveLink<f64> for Softplus {}

impl InitialEtaFromTheta<f64> for Softplus {
    #[inline]
    fn initial_eta_from_theta(theta: f64) -> f64 {
        let theta = theta.max(INITIAL_POSITIVE_FLOOR);
        if theta > 30.0 {
            theta
        } else {
            theta.exp_m1().max(INITIAL_POSITIVE_FLOOR).ln()
        }
    }
}

/// Inverse logit link: `theta` lies in the interval `(0, 1)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Logit;

impl Link<f64> for Logit {
    #[inline]
    fn inverse(eta: f64) -> f64 {
        if eta >= 0.0 {
            let z = (-eta).exp();
            1.0 / (1.0 + z)
        } else {
            let z = eta.exp();
            z / (1.0 + z)
        }
    }

    #[inline]
    fn derivative_inverse(eta: f64) -> f64 {
        let p = Self::inverse(eta);
        p * (1.0 - p)
    }
}

impl UnitIntervalLink<f64> for Logit {}

impl InitialEtaFromTheta<f64> for Logit {
    #[inline]
    fn initial_eta_from_theta(theta: f64) -> f64 {
        let probability = theta.clamp(INITIAL_PROBABILITY_FLOOR, 1.0 - INITIAL_PROBABILITY_FLOOR);
        probability.ln() - (1.0 - probability).ln()
    }
}

/// Shifted log link: `theta = OFFSET + exp(eta)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LogPlus<const OFFSET: i64>;

impl<const OFFSET: i64> Link<f64> for LogPlus<OFFSET> {
    #[inline]
    #[allow(clippy::cast_precision_loss)]
    fn inverse(eta: f64) -> f64 {
        OFFSET as f64 + eta.exp()
    }

    #[inline]
    fn derivative_inverse(eta: f64) -> f64 {
        eta.exp()
    }
}

impl<const OFFSET: i64> InitialEtaFromTheta<f64> for LogPlus<OFFSET> {
    #[inline]
    #[allow(clippy::cast_precision_loss)]
    fn initial_eta_from_theta(theta: f64) -> f64 {
        (theta - OFFSET as f64).max(INITIAL_POSITIVE_FLOOR).ln()
    }
}

/// Clamped log link: `theta = exp(clamp(eta, MIN, MAX))`.
///
/// The derivative is zero outside the active interval, matching a hard clamp
/// on the predictor scale. Callers should choose `MIN <= MAX`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ClampedLog<const MIN: i64, const MAX: i64>;

impl<const MIN: i64, const MAX: i64> Link<f64> for ClampedLog<MIN, MAX> {
    #[inline]
    #[allow(clippy::cast_precision_loss)]
    fn inverse(eta: f64) -> f64 {
        let min = MIN as f64;
        let max = MAX as f64;
        debug_assert!(min <= max);

        if eta < min {
            min.exp()
        } else if eta > max {
            max.exp()
        } else {
            eta.exp()
        }
    }

    #[inline]
    #[allow(clippy::cast_precision_loss)]
    fn derivative_inverse(eta: f64) -> f64 {
        let min = MIN as f64;
        let max = MAX as f64;
        debug_assert!(min <= max);

        if (min..=max).contains(&eta) {
            eta.exp()
        } else {
            0.0
        }
    }
}

impl<const MIN: i64, const MAX: i64> PositiveLink<f64> for ClampedLog<MIN, MAX> {}

impl<const MIN: i64, const MAX: i64> InitialEtaFromTheta<f64> for ClampedLog<MIN, MAX> {
    #[inline]
    #[allow(clippy::cast_precision_loss)]
    fn initial_eta_from_theta(theta: f64) -> f64 {
        let min = MIN as f64;
        let max = MAX as f64;
        debug_assert!(min <= max);

        theta.max(INITIAL_POSITIVE_FLOOR).ln().clamp(min, max)
    }
}

/// Link function, represented via the inverse transform.
///
/// The likelihood hot path uses `inverse(eta)`, where `eta` is on the linear
/// predictor scale, together with the derivative of the inverse for the chain
/// rule.
pub trait Link<S> {
    /// Converts a value from the predictor scale to the parameter scale.
    fn inverse(eta: S) -> S;
    /// Derivative of `inverse` with respect to `eta`.
    fn derivative_inverse(eta: S) -> S;
}

/// Initializer-time conversion from natural parameter scale to predictor scale.
///
/// This is the opposite direction of [`Link::inverse`]. It is deliberately
/// used for constructing robust starts and may clamp boundary values to keep
/// optimizer starts finite.
///
/// Custom links used with built-in families must implement this trait to
/// participate in [`crate::ParameterizedFamily`] and model initialization.
pub trait InitialEtaFromTheta<S>: Link<S> {
    /// Converts a natural-scale parameter value into a finite link-scale start.
    fn initial_eta_from_theta(theta: S) -> S;
}

/// Marker for link functions that guarantee a result in `(0, +inf)`.
///
/// This contract is suitable for scale/rate/shape-like parameters without an
/// upper bound. For probabilities use [`UnitIntervalLink`].
pub trait PositiveLink<S>: Link<S> {}

/// Marker for link functions that guarantee a result in `(0, 1)`.
///
/// This contract is suitable for probability parameters, such as Bernoulli
/// success probability or beta mean.
pub trait UnitIntervalLink<S>: Link<S> {}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::{ClampedLog, InitialEtaFromTheta, Link, Log, LogPlus, Logit, Softplus};

    #[test]
    #[allow(clippy::float_cmp)]
    fn clamped_log_clamps_value_and_derivative() {
        type LinkUnderTest = ClampedLog<-2, 2>;

        assert_relative_eq!(LinkUnderTest::inverse(-3.0), (-2.0_f64).exp());
        assert_relative_eq!(LinkUnderTest::inverse(1.0), 1.0_f64.exp());
        assert_relative_eq!(LinkUnderTest::inverse(3.0), 2.0_f64.exp());

        assert_eq!(LinkUnderTest::derivative_inverse(-3.0), 0.0);
        assert_relative_eq!(LinkUnderTest::derivative_inverse(1.0), 1.0_f64.exp());
        assert_eq!(LinkUnderTest::derivative_inverse(3.0), 0.0);
    }

    #[test]
    #[allow(clippy::float_cmp, clippy::items_after_statements)]
    fn initial_eta_from_theta_guards_link_boundaries() {
        assert!(Log::initial_eta_from_theta(0.0).is_finite());
        assert!(Softplus::initial_eta_from_theta(1.0e-300).is_finite());
        assert_relative_eq!(
            Softplus::inverse(Softplus::initial_eta_from_theta(3.0)),
            3.0
        );

        let low_logit = Logit::initial_eta_from_theta(0.0);
        let high_logit = Logit::initial_eta_from_theta(1.0);
        assert!(low_logit.is_finite());
        assert!(high_logit.is_finite());
        assert!(low_logit < 0.0);
        assert!(high_logit > 0.0);

        type Shifted = LogPlus<2>;
        assert!(Shifted::initial_eta_from_theta(2.0).is_finite());

        type Clamped = ClampedLog<-2, 2>;
        assert_eq!(Clamped::initial_eta_from_theta(1.0e-20), -2.0);
        assert_eq!(Clamped::initial_eta_from_theta(1.0e20), 2.0);
    }
}
