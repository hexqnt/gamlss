#![forbid(unsafe_code)]
//! Spline-базисы, spline design matrices и штрафы.

pub mod bspline;
pub mod cyclic;
pub mod error;
pub mod fourier;
pub mod ispline;
mod local;
pub mod monotone;
pub mod mspline;
pub mod natural;
pub mod open_uniform;
pub mod order;
pub mod penalty;
pub mod periodic;
pub mod row_basis;
pub mod tensor;

pub use bspline::{BSplineBasis, pspline_design};
pub use cyclic::{CyclicSplineDesign, CyclicSplineSpec};
pub use error::{FourierError, SplineError};
pub use fourier::FourierDesign;
pub use ispline::{ISplineBasis, ISplineDesign};
pub use monotone::{MonotoneDirection, MonotoneISplineDesign};
pub use mspline::{MSplineBasis, MSplineDesign};
pub use natural::{NaturalCubicSplineBasis, NaturalCubicSplineDesign};
pub use open_uniform::{OpenUniformSplineBasis, OpenUniformSplineDesign};
pub use order::SplineOrder;
pub use penalty::{
    CyclicDifferencePenalty, DifferencePenalty, EdgeMonotonicPenalty, PreparedDifferencePenalty,
    SlopeLimitPenalty,
};
pub use periodic::{PeriodicSplineDesign, PeriodicSplineSpec};
pub use row_basis::SplineRowBasis;
pub use tensor::TensorSplineDesign;

/// Наиболее часто используемые импорты из `gamlss-spline`.
pub mod prelude {
    pub use crate::{
        BSplineBasis, CyclicDifferencePenalty, CyclicSplineDesign, CyclicSplineSpec,
        DifferencePenalty, EdgeMonotonicPenalty, FourierDesign, FourierError, ISplineBasis,
        ISplineDesign, MSplineBasis, MSplineDesign, MonotoneDirection, MonotoneISplineDesign,
        NaturalCubicSplineBasis, NaturalCubicSplineDesign, OpenUniformSplineBasis,
        OpenUniformSplineDesign, PeriodicSplineDesign, PeriodicSplineSpec,
        PreparedDifferencePenalty, SlopeLimitPenalty, SplineError, SplineOrder, SplineRowBasis,
        TensorSplineDesign, pspline_design,
    };
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{Penalty, PredictorBlock};

    use super::{
        BSplineBasis, CyclicDifferencePenalty, CyclicSplineDesign, CyclicSplineSpec,
        DifferencePenalty, EdgeMonotonicPenalty, FourierDesign, FourierError, ISplineBasis,
        MSplineBasis, MonotoneDirection, MonotoneISplineDesign, NaturalCubicSplineBasis,
        OpenUniformSplineBasis, OpenUniformSplineDesign, PeriodicSplineDesign, PeriodicSplineSpec,
        PreparedDifferencePenalty, SlopeLimitPenalty, SplineError, SplineOrder, TensorSplineDesign,
    };

    #[test]
    fn open_uniform_bspline_partitions_unity_inside_range() {
        let x = [0.0, 0.25, 0.5, 0.75, 1.0];
        let basis = BSplineBasis::open_uniform_from_data(&x, 6, 3).unwrap();

        for value in x {
            let sum = basis.evaluate(value).iter().sum::<f64>();
            assert_relative_eq!(sum, 1.0, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn difference_penalty_gradient_matches_finite_difference() {
        let penalty = DifferencePenalty::new(0.7, 2);
        let beta = vec![0.2, -0.4, 0.9, 1.1, -0.3];
        let eps = 1.0e-6;
        let mut grad = vec![0.0; beta.len()];

        penalty.add_gradient(&beta, &mut grad);

        for index in 0..beta.len() {
            let mut plus = beta.clone();
            plus[index] += eps;
            let mut minus = beta.clone();
            minus[index] -= eps;
            let finite_difference = (penalty.value(&plus) - penalty.value(&minus)) / (2.0 * eps);

            assert_relative_eq!(grad[index], finite_difference, epsilon = 1.0e-6);
        }
    }

    #[test]
    fn prepared_difference_penalty_matches_unprepared() {
        let beta = [0.2, -0.4, 0.9, 1.1, -0.3];

        for order in 0..=2 {
            let unprepared = DifferencePenalty::new(0.7, order);
            let prepared = PreparedDifferencePenalty::new(0.7, order);
            let mut unprepared_grad = vec![0.0; beta.len()];
            let mut prepared_grad = vec![0.0; beta.len()];

            unprepared.add_gradient(&beta, &mut unprepared_grad);
            prepared.add_gradient(&beta, &mut prepared_grad);

            assert_relative_eq!(
                prepared.value(&beta),
                unprepared.value(&beta),
                epsilon = 1.0e-12
            );
            for (prepared, unprepared) in prepared_grad.iter().zip(&unprepared_grad) {
                assert_relative_eq!(prepared, unprepared, epsilon = 1.0e-12);
            }
        }
    }

    #[test]
    fn open_uniform_basis_visitor_matches_design_row_basis() {
        let x = [-0.25, 0.0, 0.2, 0.75, 1.25];
        let basis = OpenUniformSplineBasis::new(0.0, 1.0, 6, SplineOrder::Cubic).unwrap();
        let design = basis.design(&x).unwrap();

        for (row, value) in x.iter().copied().enumerate() {
            let mut from_design = Vec::new();
            let mut from_basis = Vec::new();

            super::SplineRowBasis::for_each_row_basis(&design, row, |index, weight| {
                from_design.push((index, weight));
            });
            basis
                .for_each_value_basis(value, |index, weight| {
                    from_basis.push((index, weight));
                })
                .unwrap();

            assert_eq!(from_basis.len(), from_design.len());
            for ((basis_index, basis_weight), (design_index, design_weight)) in
                from_basis.iter().zip(&from_design)
            {
                assert_eq!(basis_index, design_index);
                assert_relative_eq!(basis_weight, design_weight, epsilon = 1.0e-12);
            }
        }
    }

    #[test]
    fn open_uniform_basis_visitor_rejects_non_finite_input() {
        let basis = OpenUniformSplineBasis::new(0.0, 1.0, 6, SplineOrder::Cubic).unwrap();
        let err = basis.for_each_value_basis(f64::NAN, |_, _| {}).unwrap_err();

        assert_eq!(err, SplineError::NonFiniteValue);
    }

    #[test]
    fn cyclic_spline_design_wraps_and_partitions_unity() {
        let design =
            CyclicSplineDesign::new(&[0.0, 0.25, 1.0, -0.25], 8, SplineOrder::Cubic).unwrap();
        let beta = vec![1.0; design.nparams()];

        for row in 0..design.nrows() {
            assert_relative_eq!(design.eta_row(row, &beta), 1.0, epsilon = 1.0e-12);
        }

        let ramp = (0..8).map(|value| value as f64).collect::<Vec<_>>();
        assert_relative_eq!(design.eta_row(0, &ramp), design.eta_row(2, &ramp));
    }

    #[test]
    fn fourier_design_evaluates_harmonics_without_materialized_matrix() {
        let design = FourierDesign::new(&[0.0, 0.25, 0.5, 1.25], 1.0, 2, true).unwrap();
        let beta = [0.5, 1.0, 2.0, -0.25, 0.75];

        assert_eq!(design.nrows(), 4);
        assert_eq!(design.nparams(), 5);
        assert_eq!(design.order(), 2);
        assert_relative_eq!(design.period(), 1.0, epsilon = 1.0e-12);
        assert!(design.include_intercept());

        for row in 0..design.nrows() {
            let x = design.x()[row];
            let phase1 = std::f64::consts::TAU * x;
            let phase2 = 2.0 * std::f64::consts::TAU * x;
            let expected = beta[0]
                + beta[1] * phase1.sin()
                + beta[2] * phase1.cos()
                + beta[3] * phase2.sin()
                + beta[4] * phase2.cos();

            assert_relative_eq!(design.eta_row(row, &beta), expected, epsilon = 1.0e-12);
        }

        assert_relative_eq!(
            design.eta_row(1, &beta),
            design.eta_row(3, &beta),
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn fourier_design_without_intercept_uses_two_coefficients_per_harmonic() {
        let design = FourierDesign::new(&[1.0], 4.0, 1, false).unwrap();
        let beta = [2.0, 3.0];
        let phase = std::f64::consts::TAU * 1.0 / 4.0;

        assert_eq!(design.nparams(), 2);
        assert_relative_eq!(
            design.eta_row(0, &beta),
            beta[0] * phase.sin() + beta[1] * phase.cos(),
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn fourier_row_basis_matches_eta_unit_coefficients() {
        let design = FourierDesign::new(&[0.0, 0.2, 0.7], 1.0, 3, true).unwrap();

        for row in 0..design.nrows() {
            let mut basis = vec![0.0; design.nparams()];
            crate::SplineRowBasis::for_each_row_basis(&design, row, |index, weight| {
                basis[index] = weight;
            });

            for index in 0..design.nparams() {
                let mut beta = vec![0.0; design.nparams()];
                beta[index] = 1.0;
                assert_relative_eq!(basis[index], design.eta_row(row, &beta), epsilon = 1.0e-12);
            }
        }
    }

    #[test]
    fn fourier_design_rejects_invalid_inputs() {
        assert_eq!(
            FourierDesign::new(&[f64::NAN], 1.0, 1, true).unwrap_err(),
            FourierError::NonFiniteValue
        );
        assert_eq!(
            FourierDesign::new(&[0.0], 0.0, 1, true).unwrap_err(),
            FourierError::InvalidPeriod
        );
        assert_eq!(
            FourierDesign::new(&[0.0], 1.0, 0, true).unwrap_err(),
            FourierError::InvalidOrder
        );
        assert_eq!(
            FourierDesign::new(&[0.0], 1.0, usize::MAX, true).unwrap_err(),
            FourierError::CoefficientOverflow
        );
    }

    #[test]
    fn fourier_design_gradient_matches_finite_difference() {
        let design = FourierDesign::new(&[0.0, 0.2, 0.7, 1.4], 1.0, 3, true).unwrap();
        let beta = vec![0.3, -0.5, 1.2, 0.4, -0.8, 0.9, 0.1];
        let scores = vec![0.7, -1.1, 0.2, 1.4];
        let eps = 1.0e-6;
        let mut grad = vec![0.0; design.nparams()];

        design.add_gradient(&scores, &beta, &mut grad);

        for index in 0..beta.len() {
            let mut plus = beta.clone();
            plus[index] += eps;
            let mut minus = beta.clone();
            minus[index] -= eps;

            let objective = |candidate: &[f64]| {
                (0..design.nrows())
                    .map(|row| scores[row] * design.eta_row(row, candidate))
                    .sum::<f64>()
            };
            let finite_difference = (objective(&plus) - objective(&minus)) / (2.0 * eps);

            assert_relative_eq!(grad[index], finite_difference, epsilon = 1.0e-6);
        }
    }

    #[test]
    fn open_uniform_spline_design_handles_boundaries_and_extrapolation() {
        let design = OpenUniformSplineDesign::with_range(
            &[-0.5, 0.0, 0.5, 1.0, 1.5],
            0.0,
            1.0,
            6,
            SplineOrder::Cubic,
        )
        .unwrap();
        let beta = vec![1.0; design.nparams()];

        for row in 0..design.nrows() {
            assert_relative_eq!(design.eta_row(row, &beta), 1.0, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn open_uniform_spline_basis_reuses_training_range_for_new_data() {
        let train = [0.0, 0.5, 1.0];
        let basis = OpenUniformSplineBasis::from_data(&train, 6, SplineOrder::Cubic).unwrap();
        let train_design = basis.design(&train).unwrap();
        let new_design = basis.design(&[-0.25, 0.25, 1.25]).unwrap();

        assert_eq!(train_design.basis(), basis);
        assert_eq!(new_design.nparams(), train_design.nparams());
        assert_relative_eq!(basis.min(), 0.0, epsilon = 1.0e-12);
        assert_relative_eq!(basis.max(), 1.0, epsilon = 1.0e-12);

        let beta = vec![1.0; new_design.nparams()];
        for row in 0..new_design.nrows() {
            assert_relative_eq!(new_design.eta_row(row, &beta), 1.0, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn cyclic_spline_spec_reuses_basis_shape_for_new_data() {
        let spec = CyclicSplineSpec::new(6, SplineOrder::Cubic).unwrap();
        let train_design = spec.design(&[0.0, 0.5]).unwrap();
        let new_design = spec.design(&[0.25, 1.25]).unwrap();

        assert_eq!(train_design.spec(), spec);
        assert_eq!(new_design.nparams(), train_design.nparams());

        let beta = vec![1.0; new_design.nparams()];
        for row in 0..new_design.nrows() {
            assert_relative_eq!(new_design.eta_row(row, &beta), 1.0, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn cyclic_difference_penalty_gradient_matches_finite_difference() {
        let penalty = CyclicDifferencePenalty::new(0.7, 2);
        let beta = vec![0.2, -0.4, 0.9, 1.1, -0.3];
        assert_penalty_gradient_matches_finite_difference(&penalty, &beta);
    }

    #[test]
    fn edge_monotonic_penalty_gradient_matches_finite_difference() {
        let penalty = EdgeMonotonicPenalty::new(3.0);
        let beta = vec![0.2, 0.8, 0.4, -0.1];
        assert_penalty_gradient_matches_finite_difference(&penalty, &beta);
    }

    #[test]
    fn slope_limit_penalty_gradient_matches_finite_difference() {
        let penalty = SlopeLimitPenalty::new(5.0, 2.0, Some(0.4), Some(0.3));
        let beta = vec![0.6, 0.1, -0.1, 0.4];
        assert_penalty_gradient_matches_finite_difference(&penalty, &beta);
    }

    #[test]
    fn slope_limit_penalty_ignores_invalid_inputs() {
        let cases = [
            SlopeLimitPenalty::new(f64::NAN, 2.0, Some(0.4), Some(0.3)),
            SlopeLimitPenalty::new(5.0, f64::NAN, Some(0.4), Some(0.3)),
            SlopeLimitPenalty::new(5.0, 2.0, Some(f64::NAN), Some(-0.3)),
        ];
        let beta = vec![0.6, 0.1, -0.1, 0.4];

        for penalty in cases {
            let mut grad = vec![0.0; beta.len()];
            assert_eq!(penalty.value(&beta), 0.0);
            penalty.add_gradient(&beta, &mut grad);
            assert_eq!(grad, vec![0.0; beta.len()]);
        }
    }

    #[test]
    fn natural_cubic_validates_knots_and_has_linear_edges() {
        assert_eq!(
            NaturalCubicSplineBasis::new(vec![0.0]).unwrap_err(),
            SplineError::NotEnoughKnots { min: 2 }
        );
        assert_eq!(
            NaturalCubicSplineBasis::new(vec![0.0, f64::NAN]).unwrap_err(),
            SplineError::InvalidKnots
        );

        let basis = NaturalCubicSplineBasis::new(vec![0.0, 0.5, 1.0]).unwrap();
        let left = basis.evaluate_derivative(-0.25);
        let at_left = basis.evaluate_derivative(0.0);
        let right = basis.evaluate_derivative(1.25);
        let at_right = basis.evaluate_derivative(1.0);

        for index in 0..basis.n_basis() {
            assert_relative_eq!(left[index], at_left[index], epsilon = 1.0e-12);
            assert_relative_eq!(right[index], at_right[index], epsilon = 1.0e-12);
        }
    }

    #[test]
    fn natural_cubic_design_gradient_matches_finite_difference() {
        let design = NaturalCubicSplineBasis::new(vec![0.0, 0.4, 0.7, 1.0])
            .unwrap()
            .design(&[0.0, 0.2, 0.8, 1.1])
            .unwrap();
        assert_predictor_gradient_matches_finite_difference(
            &design,
            &[0.4, -0.2, 0.7, 1.1],
            &[0.3, -0.8, 0.5, 1.0],
        );
    }

    #[test]
    fn spline_derivatives_match_finite_difference() {
        let x = [0.1, 0.25, 0.6, 0.9];
        let beta = vec![0.1, -0.4, 0.7, 0.2, -0.1, 0.5];
        assert_eta_derivative_matches_coordinate_difference(
            |points| {
                OpenUniformSplineDesign::with_range(points, 0.0, 1.0, 6, SplineOrder::Cubic)
                    .unwrap()
            },
            |design, row, beta| design.eta_derivative_row(row, beta),
            &x,
            &beta,
        );

        let cyclic = CyclicSplineDesign::new(&x, 6, SplineOrder::Cubic).unwrap();
        assert_eta_derivative_matches_coordinate_difference(
            |points| CyclicSplineDesign::new(points, 6, SplineOrder::Cubic).unwrap(),
            |design, row, beta| design.eta_derivative_row(row, beta),
            &x,
            &beta,
        );

        let natural = NaturalCubicSplineBasis::new(vec![0.0, 0.4, 0.7, 1.0])
            .unwrap()
            .design(&x)
            .unwrap();
        let natural_beta = vec![0.1, -0.4, 0.7, 0.2];
        assert_eta_derivative_matches_coordinate_difference(
            |points| {
                NaturalCubicSplineBasis::new(vec![0.0, 0.4, 0.7, 1.0])
                    .unwrap()
                    .design(points)
                    .unwrap()
            },
            |design, row, beta| design.eta_derivative_row(row, beta),
            &x,
            &natural_beta,
        );

        assert!(cyclic.eta_derivative_row(0, &beta).is_finite());
        assert!(natural.eta_derivative_row(0, &natural_beta).is_finite());
    }

    #[test]
    fn periodic_spline_wraps_physical_coordinates_and_rejects_invalid_period() {
        assert_eq!(
            PeriodicSplineSpec::new(6, SplineOrder::Cubic, 0.0, 0.0).unwrap_err(),
            SplineError::InvalidPeriod
        );

        let design =
            PeriodicSplineDesign::new(&[0.0, 12.0, 3.0, 15.0], 6, SplineOrder::Cubic, 12.0, 0.0)
                .unwrap();
        let beta = vec![0.2, -0.1, 0.7, 0.4, -0.3, 0.9];

        assert_relative_eq!(
            design.eta_row(0, &beta),
            design.eta_row(1, &beta),
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            design.eta_row(2, &beta),
            design.eta_row(3, &beta),
            epsilon = 1.0e-12
        );
        assert!(design.eta_derivative_row(2, &beta).is_finite());
    }

    #[test]
    fn tensor_spline_matches_rowwise_kronecker_and_gradient() {
        let x = [0.0, 0.3, 0.8];
        let left =
            OpenUniformSplineDesign::with_range(&x, 0.0, 1.0, 4, SplineOrder::Cubic).unwrap();
        let right = CyclicSplineDesign::new(&x, 5, SplineOrder::Cubic).unwrap();
        let tensor = TensorSplineDesign::new(left.clone(), right.clone()).unwrap();
        let beta = (0..tensor.nparams())
            .map(|index| index as f64 / 10.0)
            .collect::<Vec<_>>();

        let row = 1;
        let mut expected = 0.0;
        for left_index in 0..left.nparams() {
            let mut left_unit = vec![0.0; left.nparams()];
            left_unit[left_index] = 1.0;
            let left_weight = left.eta_row(row, &left_unit);
            for right_index in 0..right.nparams() {
                let mut right_unit = vec![0.0; right.nparams()];
                right_unit[right_index] = 1.0;
                let right_weight = right.eta_row(row, &right_unit);
                expected +=
                    beta[left_index * right.nparams() + right_index] * left_weight * right_weight;
            }
        }

        assert_relative_eq!(tensor.eta_row(row, &beta), expected, epsilon = 1.0e-12);
        assert_predictor_gradient_matches_finite_difference(&tensor, &beta, &[0.5, -0.2, 0.9]);

        let short = CyclicSplineDesign::new(&[0.0], 5, SplineOrder::Cubic).unwrap();
        assert_eq!(
            TensorSplineDesign::new(left, short).unwrap_err(),
            SplineError::RowMismatch {
                expected: 3,
                actual: 1
            }
        );
    }

    #[test]
    fn m_and_i_splines_are_nonnegative_monotone_and_differentiable() {
        let x = [0.0, 0.2, 0.5, 0.8, 1.0];
        let m = MSplineBasis::open_uniform_from_data(&x, 6, 3).unwrap();
        let i = ISplineBasis::open_uniform_from_data(&x, 6, 3).unwrap();

        for value in [0.1, 0.35, 0.7] {
            for weight in m.evaluate(value) {
                assert!(weight >= -1.0e-12);
            }
        }

        for basis_index in 0..i.n_basis() {
            let a = i.evaluate(0.25)[basis_index];
            let b = i.evaluate(0.75)[basis_index];
            assert!(a <= b + 1.0e-12);
            assert_relative_eq!(i.evaluate(-0.1)[basis_index], 0.0, epsilon = 1.0e-12);
            assert_relative_eq!(i.evaluate(1.1)[basis_index], 1.0, epsilon = 1.0e-9);
        }

        let point = 0.43;
        let eps = 1.0e-6;
        for basis_index in 0..i.n_basis() {
            let finite = (i.evaluate(point + eps)[basis_index]
                - i.evaluate(point - eps)[basis_index])
                / (2.0 * eps);
            assert_relative_eq!(
                i.evaluate_derivative(point)[basis_index],
                finite,
                epsilon = 1.0e-5
            );
        }
    }

    #[test]
    fn monotone_i_spline_is_hard_monotone_and_gradient_matches_finite_difference() {
        let x = [0.0, 0.2, 0.5, 0.8, 1.0];
        let basis = ISplineBasis::open_uniform_from_data(&x, 5, 2).unwrap();
        let design =
            MonotoneISplineDesign::new(&x, basis.clone(), MonotoneDirection::Increasing).unwrap();
        let beta = vec![0.1, -1.0, 0.2, -0.4, 0.7, 1.0];

        for row in 1..design.nrows() {
            assert!(design.eta_row(row - 1, &beta) <= design.eta_row(row, &beta) + 1.0e-12);
            assert!(design.eta_derivative_row(row, &beta) >= -1.0e-10);
        }

        assert_predictor_gradient_matches_finite_difference(
            &design,
            &beta,
            &[0.2, -0.3, 0.5, 0.7, -0.1],
        );

        let decreasing =
            MonotoneISplineDesign::new(&x, basis, MonotoneDirection::Decreasing).unwrap();
        for row in 1..decreasing.nrows() {
            assert!(decreasing.eta_row(row - 1, &beta) >= decreasing.eta_row(row, &beta) - 1.0e-12);
        }
    }

    fn assert_penalty_gradient_matches_finite_difference<P>(penalty: &P, beta: &[f64])
    where
        P: Penalty,
    {
        let eps = 1.0e-6;
        let mut grad = vec![0.0; beta.len()];

        penalty.add_gradient(beta, &mut grad);

        for index in 0..beta.len() {
            let mut plus = beta.to_vec();
            plus[index] += eps;
            let mut minus = beta.to_vec();
            minus[index] -= eps;
            let finite_difference = (penalty.value(&plus) - penalty.value(&minus)) / (2.0 * eps);

            assert_relative_eq!(grad[index], finite_difference, epsilon = 1.0e-6);
        }
    }

    fn assert_predictor_gradient_matches_finite_difference<P>(
        predictor: &P,
        beta: &[f64],
        scores: &[f64],
    ) where
        P: PredictorBlock,
    {
        let eps = 1.0e-6;
        let mut grad = vec![0.0; beta.len()];

        predictor.add_gradient(scores, beta, &mut grad);

        for index in 0..beta.len() {
            let mut plus = beta.to_vec();
            plus[index] += eps;
            let mut minus = beta.to_vec();
            minus[index] -= eps;

            let objective = |candidate: &[f64]| {
                (0..predictor.nrows())
                    .map(|row| scores[row] * predictor.eta_row(row, candidate))
                    .sum::<f64>()
            };
            let finite_difference = (objective(&plus) - objective(&minus)) / (2.0 * eps);

            assert_relative_eq!(grad[index], finite_difference, epsilon = 1.0e-6);
        }
    }

    fn assert_eta_derivative_matches_coordinate_difference<P>(
        build: impl Fn(&[f64]) -> P,
        derivative: impl Fn(&P, usize, &[f64]) -> f64,
        x: &[f64],
        beta: &[f64],
    ) where
        P: PredictorBlock,
    {
        let eps = 1.0e-6;
        let design = build(x);
        for row in 0..x.len() {
            let mut plus_x = x.to_vec();
            plus_x[row] += eps;
            let plus = build(&plus_x);
            let mut minus_x = x.to_vec();
            minus_x[row] -= eps;
            let minus = build(&minus_x);
            let finite_difference =
                (plus.eta_row(row, beta) - minus.eta_row(row, beta)) / (2.0 * eps);

            assert_relative_eq!(
                derivative(&design, row, beta),
                finite_difference,
                epsilon = 1.0e-5
            );
        }
    }
}
