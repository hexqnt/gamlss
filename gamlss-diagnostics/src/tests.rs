use approx::assert_relative_eq;
use gamlss_core::{
    CholeskyScale, DenseDesign, Gamlss, Identity, LinearPredictorBlock, Log,
    LowerTriangularParameterBlock, ModelError, Mu, NoPenalty, ParameterBlock, ParameterBlocks,
    Sigma, VectorParameterBlock,
};
use gamlss_family::{MvNormalCholeskyDefault, Normal};

use super::{
    CdfDiagnosticsExt, CrpsDiagnosticsExt, PredictionDiagnosticsExt,
    PredictionRosenblattDiagnosticsExt, RosenblattDiagnosticsExt, normalize_pit_values,
};

type TestModel<'a> = Gamlss<
    Normal<Identity, Log>,
    ParameterBlocks<(
        ParameterBlock<Mu, LinearPredictorBlock<DenseDesign>, NoPenalty>,
        ParameterBlock<Sigma, LinearPredictorBlock<DenseDesign>, NoPenalty>,
    )>,
    &'a [f64],
>;

type WeightedTestModel<'a> = Gamlss<
    Normal<Identity, Log>,
    ParameterBlocks<(
        ParameterBlock<Mu, LinearPredictorBlock<DenseDesign>, NoPenalty>,
        ParameterBlock<Sigma, LinearPredictorBlock<DenseDesign>, NoPenalty>,
    )>,
    (&'a [f64], &'a [f64]),
>;

fn normal_intercept_model(y: &[f64]) -> TestModel<'_> {
    let blocks = ParameterBlocks::new((
        ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 0),
        ParameterBlock::<Sigma, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 0),
    ));

    Gamlss::try_new(Normal::<Identity, Log>::new(), blocks, y).expect("valid normal model")
}

fn weighted_normal_intercept_model<'a>(y: &'a [f64], weights: &'a [f64]) -> WeightedTestModel<'a> {
    let blocks = ParameterBlocks::new((
        ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 0),
        ParameterBlock::<Sigma, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 0),
    ));

    Gamlss::try_new_weighted(Normal::<Identity, Log>::new(), blocks, y, weights)
        .expect("valid weighted normal model")
}

#[test]
fn pit_values_use_training_observations_and_fitted_parameters() {
    let y = [0.0, 1.0, -1.0];
    let model = normal_intercept_model(&y);

    let pit = model.pit_values(&[0.0, 0.0]).expect("valid parameters");
    let mut pit_into = vec![f64::NAN; y.len()];
    model
        .pit_values_into(&[0.0, 0.0], &mut pit_into)
        .expect("valid parameters");

    assert_relative_eq!(pit[0], 0.5, epsilon = 1.0e-7);
    assert_relative_eq!(pit[1], 0.841_344_746, epsilon = 1.0e-7);
    assert_relative_eq!(pit[2], 0.158_655_254, epsilon = 1.0e-7);
    assert_eq!(pit_into, pit);
}

#[test]
fn quantile_residuals_transform_pit_values_to_standard_normal_scale() {
    let y = [-1.0, 0.0, 1.0];
    let model = normal_intercept_model(&y);

    let residuals = model
        .quantile_residuals(&[0.0, 0.0])
        .expect("valid parameters");

    assert_relative_eq!(residuals[0], -1.0, epsilon = 1.0e-6);
    assert_relative_eq!(residuals[1], 0.0, epsilon = 1.0e-6);
    assert_relative_eq!(residuals[2], 1.0, epsilon = 1.0e-6);
}

#[test]
fn quantile_residuals_map_invalid_and_boundary_pit_values() {
    let residuals = normalize_pit_values(vec![f64::NAN, -0.1, 0.0, 1.0, 1.1]);

    assert!(residuals[0].is_nan());
    assert!(residuals[1].is_nan());
    assert!(residuals[2].is_infinite());
    assert!(residuals[2].is_sign_negative());
    assert!(residuals[3].is_infinite());
    assert!(residuals[3].is_sign_positive());
    assert!(residuals[4].is_nan());
}

#[test]
fn pit_values_reject_wrong_parameter_length() {
    let y = [0.0];
    let model = normal_intercept_model(&y);

    assert_eq!(
        model.pit_values(&[]).unwrap_err(),
        ModelError::BetaLength {
            expected: 2,
            actual: 0
        }
    );
    assert_eq!(
        model.pit_values_into(&[0.0, 0.0], &mut []).unwrap_err(),
        ModelError::ResponseLength {
            expected: 1,
            actual: 0
        }
    );
}

#[test]
fn crps_values_use_training_observations_and_fitted_parameters() {
    let y = [0.0, 1.0];
    let model = normal_intercept_model(&y);

    let crps = model.crps_values(&[0.0, 0.0]).expect("valid parameters");
    let mut crps_into = vec![f64::NAN; y.len()];
    model
        .crps_values_into(&[0.0, 0.0], &mut crps_into)
        .expect("valid parameters");

    assert_relative_eq!(crps[0], 0.233_694_977_255_109_13, epsilon = 1.0e-12);
    assert_relative_eq!(crps[1], 0.602_441_357_627_616_5, epsilon = 1.0e-12);
    assert_eq!(crps_into, crps);
    assert_relative_eq!(
        model.mean_crps(&[0.0, 0.0]).expect("valid parameters"),
        crps[0].midpoint(crps[1]),
        epsilon = 1.0e-12
    );
}

#[test]
fn mean_crps_matches_materialized_values_in_row_order() {
    let y = [-100.0, -1.0, 0.0, 0.25, 5.0, 1.0e6];
    let model = normal_intercept_model(&y);
    for parameters in [[0.0, 0.0], [1.0, -0.5], [-3.0, 2.0]] {
        let values = model.crps_values(&parameters).unwrap();
        #[allow(clippy::cast_precision_loss)]
        let expected = values.iter().sum::<f64>() / values.len() as f64;
        assert_eq!(model.mean_crps(&parameters).unwrap(), expected);
    }
    assert_eq!(
        model.mean_crps(&[]).unwrap_err(),
        ModelError::BetaLength {
            expected: 2,
            actual: 0,
        }
    );
}

#[test]
fn unweighted_mean_crps_retains_zero_weighted_nan_observations() {
    let model = weighted_normal_intercept_model(&[f64::NAN, 1.0], &[0.0, 2.0]);
    assert!(model.mean_crps(&[0.0, 0.0]).unwrap().is_nan());
    assert!(model.weighted_mean_crps(&[0.0, 0.0]).unwrap().is_finite());
}

#[test]
fn mean_crps_preserves_observation_length_callback_order() {
    use std::cell::Cell;

    use gamlss_core::ObservationView;

    struct Observations {
        length_calls: Cell<usize>,
    }

    impl<'row> ObservationView<'row> for Observations {
        type Observation = f64;

        fn len(&self) -> usize {
            self.length_calls.set(self.length_calls.get() + 1);
            1
        }

        fn observation_at(&'row self, _row: usize) -> f64 {
            assert_eq!(self.length_calls.get(), 3);
            0.0
        }

        fn weight_at(&self, _row: usize) -> f64 {
            1.0
        }
    }

    let blocks = ParameterBlocks::new((
        ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(1), NoPenalty, 0),
        ParameterBlock::<Sigma, _, _>::linear(DenseDesign::intercept(1), NoPenalty, 0),
    ));
    let model = Gamlss::try_new_with_observations(
        Normal::<Identity, Log>::new(),
        blocks,
        Observations {
            length_calls: Cell::new(0),
        },
    )
    .unwrap();

    model.obs().length_calls.set(0);
    assert!(model.mean_crps(&[0.0, 0.0]).unwrap().is_finite());
    assert_eq!(model.obs().length_calls.get(), 3);

    model.obs().length_calls.set(0);
    assert!(matches!(
        model.mean_crps(&[]),
        Err(ModelError::BetaLength { .. })
    ));
    assert_eq!(model.obs().length_calls.get(), 2);
}

#[test]
fn diagnostics_with_blocks_match_training_helpers_for_training_blocks() {
    let y = [-1.0, 0.0, 1.0];
    let model = normal_intercept_model(&y);
    let parameters = [0.0, 0.0];
    let obs = &y[..];
    let prediction = model.prediction_view(model.blocks()).unwrap();
    let diagnostics = prediction.with_observations(&obs).unwrap();

    assert_eq!(
        model
            .pit_values_with_blocks(&parameters, model.blocks(), &obs)
            .unwrap(),
        model.pit_values(&parameters).unwrap()
    );
    assert_eq!(
        diagnostics.pit_values(&parameters).unwrap(),
        model.pit_values(&parameters).unwrap()
    );
    assert_eq!(
        model
            .quantile_residuals_with_blocks(&parameters, model.blocks(), &obs)
            .unwrap(),
        model.quantile_residuals(&parameters).unwrap()
    );
    assert_eq!(
        diagnostics.quantile_residuals(&parameters).unwrap(),
        model.quantile_residuals(&parameters).unwrap()
    );
    assert_eq!(
        model
            .crps_values_with_blocks(&parameters, model.blocks(), &obs)
            .unwrap(),
        model.crps_values(&parameters).unwrap()
    );
    assert_eq!(
        diagnostics.crps_values(&parameters).unwrap(),
        model.crps_values(&parameters).unwrap()
    );

    let mut pit = vec![f64::NAN; diagnostics.nrows()];
    diagnostics.pit_values_into(&parameters, &mut pit).unwrap();
    assert_eq!(pit, model.pit_values(&parameters).unwrap());

    let mut crps = vec![f64::NAN; diagnostics.nrows()];
    diagnostics
        .crps_values_into(&parameters, &mut crps)
        .unwrap();
    assert_eq!(crps, model.crps_values(&parameters).unwrap());
}

#[test]
fn diagnostics_with_blocks_reject_observation_row_mismatch() {
    let y = [0.0, 1.0];
    let model = normal_intercept_model(&y);
    let obs = &[0.0][..];

    assert_eq!(
        model
            .pit_values_with_blocks(&[0.0, 0.0], model.blocks(), &obs)
            .unwrap_err(),
        ModelError::ResponseLength {
            expected: 2,
            actual: 1
        }
    );
}

#[test]
fn weighted_mean_crps_uses_observation_weights() {
    let y = [0.0, 1.0];
    let weights = [0.0, 2.0];
    let model = weighted_normal_intercept_model(&y, &weights);
    let crps = model.crps_values(&[0.0, 0.0]).unwrap();

    assert_relative_eq!(
        model.weighted_mean_crps(&[0.0, 0.0]).unwrap(),
        crps[1],
        epsilon = 1.0e-12
    );
}

#[test]
fn weighted_mean_crps_skips_zero_weighted_observations_before_crps() {
    let y = [f64::NAN, 1.0];
    let weights = [0.0, 2.0];
    let model = weighted_normal_intercept_model(&y, &weights);

    assert_relative_eq!(
        model.weighted_mean_crps(&[0.0, 0.0]).unwrap(),
        0.602_441_357_627_616_5,
        epsilon = 1.0e-12
    );
}

#[test]
fn weighted_mean_crps_returns_nan_when_all_weights_are_zero() {
    let y = [0.0, 1.0];
    let weights = [0.0, 0.0];
    let model = weighted_normal_intercept_model(&y, &weights);

    assert!(model.weighted_mean_crps(&[0.0, 0.0]).unwrap().is_nan());
}

#[test]
fn crps_values_reject_wrong_parameter_length() {
    let y = [0.0];
    let model = normal_intercept_model(&y);

    assert_eq!(
        model.crps_values(&[]).unwrap_err(),
        ModelError::BetaLength {
            expected: 2,
            actual: 0
        }
    );
    assert_eq!(
        model.crps_values_into(&[0.0, 0.0], &mut []).unwrap_err(),
        ModelError::ResponseLength {
            expected: 1,
            actual: 0
        }
    );
}

#[test]
fn rosenblatt_diagnostics_use_row_major_coordinate_order() {
    let y = [[0.0, 0.0], [1.0, 1.5]];
    let n = y.len();
    let mu = VectorParameterBlock::<Mu, 2, _, _>::new(
        [
            LinearPredictorBlock::new(DenseDesign::intercept(n)),
            LinearPredictorBlock::new(DenseDesign::intercept(n)),
        ],
        NoPenalty,
        0,
    );
    let cholesky = LowerTriangularParameterBlock::<CholeskyScale, 2, _, _>::new(
        vec![
            LinearPredictorBlock::new(DenseDesign::intercept(n)),
            LinearPredictorBlock::new(DenseDesign::intercept(n)),
            LinearPredictorBlock::new(DenseDesign::intercept(n)),
        ],
        NoPenalty,
        0,
    );
    let blocks = ParameterBlocks::new((mu, cholesky));
    let model = Gamlss::try_new_with_observations(
        MvNormalCholeskyDefault::<2>::new(),
        blocks,
        y.as_slice(),
    )
    .unwrap();
    let parameters = [0.0, 0.0, 0.0, 0.5, 0.0];

    let values = model.rosenblatt_values(&parameters).unwrap();
    assert_eq!(values.len(), 4);
    assert_relative_eq!(values[0], 0.5, epsilon = 1.0e-12);
    assert_relative_eq!(values[1], 0.5, epsilon = 1.0e-12);
    assert_relative_eq!(values[2], 0.841_344_746, epsilon = 1.0e-9);
    assert_relative_eq!(values[3], 0.841_344_746, epsilon = 1.0e-9);

    let prediction = model.prediction_view(model.blocks()).unwrap();
    assert_eq!(
        prediction
            .rosenblatt_values(&parameters, &y.as_slice())
            .unwrap(),
        values
    );
    assert_eq!(
        model
            .rosenblatt_values_with_blocks(&parameters, model.blocks(), &y.as_slice())
            .unwrap(),
        values
    );
    assert_eq!(
        model
            .rosenblatt_values_into(&parameters, &mut [0.0; 3])
            .unwrap_err(),
        ModelError::ResponseLength {
            expected: 4,
            actual: 3,
        }
    );
}

#[test]
fn diagnostics_prelude_exposes_extension_traits() {
    use crate::prelude::*;

    let y = [0.0];
    let model = normal_intercept_model(&y);

    assert_relative_eq!(
        model.pit_values(&[0.0, 0.0]).unwrap()[0],
        0.5,
        epsilon = 1.0e-7
    );
    assert_relative_eq!(
        model.quantile_residuals(&[0.0, 0.0]).unwrap()[0],
        0.0,
        epsilon = 1.0e-6
    );
    assert_relative_eq!(
        model.mean_crps(&[0.0, 0.0]).unwrap(),
        0.233_694_977_255_109_13,
        epsilon = 1.0e-12
    );
}
