#![forbid(unsafe_code)]
//! Распределения, likelihood и score для GAMLSS.

/// Распределение Лапласа.
pub mod laplace;
/// Нормальное распределение.
pub mod normal;
/// Распределение Стьюдента с фиксированным числом степеней свободы.
pub mod student_t;

pub use laplace::{DefaultLaplace, Laplace, LaplaceEta, LaplaceTheta};
pub use normal::{DefaultNormal, Normal, NormalEta, NormalGamlss, NormalTheta, normal_gamlss};
pub use student_t::{DefaultStudentT, StudentT, StudentTEta, StudentTTheta};

/// Наиболее часто используемые импорты из `gamlss-family`.
pub mod prelude {
    pub use crate::{
        DefaultLaplace, DefaultNormal, DefaultStudentT, Laplace, LaplaceEta, LaplaceTheta, Normal,
        NormalEta, NormalGamlss, NormalTheta, StudentT, StudentTEta, StudentTTheta, normal_gamlss,
    };
}

#[cfg(test)]
pub(crate) mod test_support {
    use approx::assert_relative_eq;
    use gamlss_core::{Family, ParameterParts};

    const DEFAULT_EPSILON: f64 = 1.0e-6;
    const DEFAULT_TOLERANCE: f64 = 1.0e-6;

    pub(crate) fn assert_score_matches_finite_difference<F, const K: usize>(
        family: &F,
        y: f64,
        eta: [f64; K],
    ) where
        F: Family,
        F::Eta: ParameterParts<K>,
        F::ScoreEta: ParameterParts<K>,
    {
        assert_score_matches_finite_difference_with_tolerance::<F, K>(
            family,
            y,
            eta,
            DEFAULT_EPSILON,
            DEFAULT_TOLERANCE,
        );
    }

    pub(crate) fn assert_score_matches_finite_difference_with_tolerance<F, const K: usize>(
        family: &F,
        y: f64,
        eta: [f64; K],
        epsilon: f64,
        tolerance: f64,
    ) where
        F: Family,
        F::Eta: ParameterParts<K>,
        F::ScoreEta: ParameterParts<K>,
    {
        let (_, score) = family.nll_and_score_eta(y, F::Eta::from_array(eta));

        for index in 0..K {
            let mut plus = eta;
            plus[index] += epsilon;
            let mut minus = eta;
            minus[index] -= epsilon;

            let finite_difference = (family.nll_eta(y, F::Eta::from_array(plus))
                - family.nll_eta(y, F::Eta::from_array(minus)))
                / (2.0 * epsilon);
            let actual = score.part(index);

            assert!(
                actual.is_finite(),
                "score component {index} is not finite: {actual}"
            );
            assert!(
                finite_difference.is_finite(),
                "finite-difference score component {index} is not finite: {finite_difference}"
            );
            assert_relative_eq!(actual, finite_difference, epsilon = tolerance);
        }
    }
}
