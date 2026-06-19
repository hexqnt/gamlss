#![forbid(unsafe_code)]
//! Распределения, likelihood и NLL gradients для GAMLSS.

/// Bernoulli distribution.
pub mod bernoulli;
/// Beta distribution.
pub mod beta;
/// Exponential distribution.
pub mod exponential;
/// Gamma distribution.
pub mod gamma;
/// Maximum-type Gumbel distribution.
pub mod gumbel;
mod initial;
/// Inverse Gaussian distribution.
pub mod inverse_gaussian;
/// Распределение Лапласа.
pub mod laplace;
/// Log-normal distribution.
pub mod log_normal;
/// Logistic distribution.
pub mod logistic;
/// Lomax distribution.
pub mod lomax;
/// Negative binomial distribution.
pub mod negative_binomial;
/// Нормальное распределение.
pub mod normal;
/// Poisson distribution.
pub mod poisson;
mod special;
/// Распределение Стьюдента с фиксированным числом степеней свободы.
pub mod student_t;
/// Weibull distribution.
pub mod weibull;

pub use bernoulli::{Bernoulli, BernoulliEta, BernoulliTheta, DefaultBernoulli};
pub use beta::{Beta, BetaEta, BetaTheta, DefaultBeta};
pub use exponential::{DefaultExponential, Exponential, ExponentialEta, ExponentialTheta};
pub use gamma::{DefaultGamma, Gamma, GammaEta, GammaTheta};
pub use gumbel::{DefaultGumbel, Gumbel, GumbelEta, GumbelTheta};
pub use inverse_gaussian::{
    DefaultInverseGaussian, InverseGaussian, InverseGaussianEta, InverseGaussianTheta,
};
pub use laplace::{DefaultLaplace, Laplace, LaplaceEta, LaplaceTheta};
pub use log_normal::{DefaultLogNormal, LogNormal, LogNormalEta, LogNormalTheta};
pub use logistic::{DefaultLogistic, Logistic, LogisticEta, LogisticTheta};
pub use lomax::{DefaultLomax, Lomax, LomaxEta, LomaxTheta};
pub use negative_binomial::{
    DefaultNegativeBinomial, NegativeBinomial, NegativeBinomialEta, NegativeBinomialTheta,
};
pub use normal::{DefaultNormal, Normal, NormalEta, NormalGamlss, NormalTheta, normal_gamlss};
pub use poisson::{DefaultPoisson, Poisson, PoissonEta, PoissonTheta};
pub use student_t::{DefaultStudentT, StudentT, StudentTEta, StudentTTheta};
pub use weibull::{DefaultWeibull, Weibull, WeibullEta, WeibullTheta};

/// Наиболее часто используемые импорты из `gamlss-family`.
pub mod prelude {
    pub use crate::{
        Bernoulli, BernoulliEta, BernoulliTheta, Beta, BetaEta, BetaTheta, DefaultBernoulli,
        DefaultBeta, DefaultExponential, DefaultGamma, DefaultGumbel, DefaultInverseGaussian,
        DefaultLaplace, DefaultLogNormal, DefaultLogistic, DefaultLomax, DefaultNegativeBinomial,
        DefaultNormal, DefaultPoisson, DefaultStudentT, DefaultWeibull, Exponential,
        ExponentialEta, ExponentialTheta, Gamma, GammaEta, GammaTheta, Gumbel, GumbelEta,
        GumbelTheta, InverseGaussian, InverseGaussianEta, InverseGaussianTheta, Laplace,
        LaplaceEta, LaplaceTheta, LogNormal, LogNormalEta, LogNormalTheta, Logistic, LogisticEta,
        LogisticTheta, Lomax, LomaxEta, LomaxTheta, NegativeBinomial, NegativeBinomialEta,
        NegativeBinomialTheta, Normal, NormalEta, NormalGamlss, NormalTheta, Poisson, PoissonEta,
        PoissonTheta, StudentT, StudentTEta, StudentTTheta, Weibull, WeibullEta, WeibullTheta,
        normal_gamlss,
    };
}

#[cfg(test)]
pub(crate) mod test_support {
    use approx::assert_relative_eq;
    use gamlss_core::{Family, ParameterParts};

    const DEFAULT_EPSILON: f64 = 1.0e-6;
    const DEFAULT_TOLERANCE: f64 = 1.0e-6;

    pub(crate) fn assert_gradient_matches_finite_difference<F, const K: usize>(
        family: &F,
        y: f64,
        eta: [f64; K],
    ) where
        F: for<'obs> Family<Observation<'obs> = f64>,
        F::Eta: ParameterParts<K>,
        F::NllGradientEta: ParameterParts<K>,
    {
        assert_gradient_matches_finite_difference_with_tolerance::<F, K>(
            family,
            y,
            eta,
            DEFAULT_EPSILON,
            DEFAULT_TOLERANCE,
        );
    }

    pub(crate) fn assert_gradient_matches_finite_difference_with_tolerance<F, const K: usize>(
        family: &F,
        y: f64,
        eta: [f64; K],
        epsilon: f64,
        tolerance: f64,
    ) where
        F: for<'obs> Family<Observation<'obs> = f64>,
        F::Eta: ParameterParts<K>,
        F::NllGradientEta: ParameterParts<K>,
    {
        let (_, gradient) = family.nll_and_gradient_eta(y, F::Eta::from_array(eta));

        for index in 0..K {
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
                "gradient component {index} is not finite: {actual}"
            );
            assert!(
                finite_difference.is_finite(),
                "finite-difference gradient component {index} is not finite: {finite_difference}"
            );
            assert_relative_eq!(actual, finite_difference, epsilon = tolerance);
        }
    }

    pub(crate) fn statrs_discrete_quantile<F>(p: f64, mut cdf: F) -> u64
    where
        F: FnMut(u64) -> f64,
    {
        let mut high = 1_u64;
        while cdf(high) < p {
            high *= 2;
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
}

#[cfg(test)]
mod initializer_tests {
    use gamlss_core::{Family, ParameterParts, ParameterizedFamily};

    use crate::{
        DefaultBernoulli, DefaultBeta, DefaultExponential, DefaultGamma, DefaultGumbel,
        DefaultInverseGaussian, DefaultLaplace, DefaultLogNormal, DefaultLogistic, DefaultLomax,
        DefaultNegativeBinomial, DefaultNormal, DefaultPoisson, DefaultStudentT, DefaultWeibull,
    };

    fn assert_finite_initial_eta<F, const K: usize>(family: F, data: &[f64], probe: f64)
    where
        F: for<'obs> Family<Observation<'obs> = f64> + ParameterizedFamily<K>,
        F::Eta: Copy + ParameterParts<K>,
        F::NllGradientEta: ParameterParts<K>,
    {
        let obs: &[f64] = data;
        let eta = family.initial_eta_from_observations(&obs);
        for index in 0..K {
            assert!(
                eta.part(index).is_finite(),
                "initializer component {index} is not finite"
            );
        }
        assert!(
            family.nll_eta(probe, eta).is_finite(),
            "initialized eta should produce finite nll"
        );
    }

    #[test]
    fn built_in_initializers_return_finite_eta_and_likelihood() {
        assert_finite_initial_eta::<_, 1>(DefaultBernoulli::new(), &[0.0, 1.0, 1.0], 1.0);
        assert_finite_initial_eta::<_, 2>(DefaultBeta::new(), &[0.1, 0.5, 0.9], 0.5);
        assert_finite_initial_eta::<_, 1>(DefaultExponential::new(), &[0.2, 1.0, 2.0], 1.0);
        assert_finite_initial_eta::<_, 2>(DefaultGamma::new(), &[0.2, 1.0, 2.0], 1.0);
        assert_finite_initial_eta::<_, 2>(DefaultGumbel::new(), &[-1.0, 0.0, 3.0], 0.0);
        assert_finite_initial_eta::<_, 2>(DefaultInverseGaussian::new(), &[0.5, 1.0, 2.0], 1.0);
        assert_finite_initial_eta::<_, 2>(DefaultLaplace::new(), &[-1.0, 0.0, 3.0], 0.0);
        assert_finite_initial_eta::<_, 2>(DefaultLogNormal::new(), &[0.5, 1.0, 4.0], 1.0);
        assert_finite_initial_eta::<_, 2>(DefaultLogistic::new(), &[-1.0, 0.0, 3.0], 0.0);
        assert_finite_initial_eta::<_, 2>(DefaultLomax::new(), &[0.0, 1.0, 3.0, 8.0], 1.0);
        assert_finite_initial_eta::<_, 2>(
            DefaultNegativeBinomial::new(),
            &[0.0, 1.0, 4.0, 6.0],
            1.0,
        );
        assert_finite_initial_eta::<_, 2>(DefaultNormal::new(), &[-1.0, 0.0, 3.0], 0.0);
        assert_finite_initial_eta::<_, 1>(DefaultPoisson::new(), &[0.0, 2.0, 4.0], 2.0);
        assert_finite_initial_eta::<_, 2>(DefaultStudentT::default(), &[-1.0, 0.0, 3.0], 0.0);
        assert_finite_initial_eta::<_, 2>(DefaultWeibull::new(), &[0.5, 1.5, 3.0], 1.5);
    }

    #[test]
    fn initializers_handle_edge_samples_without_nonfinite_eta() {
        assert_finite_initial_eta::<_, 2>(DefaultBeta::new(), &[0.0, 1.0, f64::NAN], 0.5);
        assert_finite_initial_eta::<_, 2>(DefaultGamma::new(), &[2.0, 2.0, f64::INFINITY], 2.0);
        assert_finite_initial_eta::<_, 2>(DefaultNegativeBinomial::new(), &[0.0, 0.0, 0.0], 0.0);
        assert_finite_initial_eta::<_, 1>(DefaultPoisson::new(), &[0.0, 0.0, 0.0], 0.0);
    }
}
