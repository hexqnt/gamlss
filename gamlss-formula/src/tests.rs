#![allow(clippy::float_cmp, clippy::cast_precision_loss)]
use std::collections::BTreeMap;

use approx::assert_relative_eq;
use gamlss_core::{DenseDesign, DesignMatrix, Objective, Penalty, PredictorBlock};
use gamlss_spline::{
    ISplineBasis, MonotoneDirection, MonotoneISplineDesign, SplineError, SplineOrder,
};

use super::*;

#[derive(Debug, Clone)]
struct TestData {
    nrows: usize,
    columns: BTreeMap<String, Vec<f64>>,
    bool_columns: BTreeMap<String, Vec<bool>>,
    cat_columns: BTreeMap<String, Vec<String>>,
    owned: bool,
}

impl TestData {
    fn borrowed(columns: &[(&str, &[f64])]) -> Self {
        let nrows = columns.first().map_or(0, |(_, values)| values.len());
        Self {
            nrows,
            columns: columns
                .iter()
                .map(|(name, values)| ((*name).to_owned(), values.to_vec()))
                .collect(),
            bool_columns: BTreeMap::new(),
            cat_columns: BTreeMap::new(),
            owned: false,
        }
    }

    fn owned(columns: &[(&str, &[f64])]) -> Self {
        let mut data = Self::borrowed(columns);
        data.owned = true;
        data
    }

    fn with_bool_column(mut self, name: &str, values: &[bool]) -> Self {
        self.bool_columns.insert(name.to_owned(), values.to_vec());
        self
    }

    fn with_cat_column(mut self, name: &str, values: &[&str]) -> Self {
        self.cat_columns.insert(
            name.to_owned(),
            values.iter().map(|value| (*value).to_owned()).collect(),
        );
        self
    }
}

impl DataView for TestData {
    fn nrows(&self) -> usize {
        self.nrows
    }

    fn f64_col(&self, col: &Col<f64>) -> Result<NumericCol<'_>, FormulaError> {
        let values = self
            .columns
            .get(col.name())
            .ok_or_else(|| FormulaError::UnknownColumn(col.name().to_owned()))?;
        if self.owned {
            Ok(NumericCol::Owned(values.clone()))
        } else {
            Ok(NumericCol::Borrowed(values))
        }
    }

    fn bool_col(&self, col: &Col<bool>) -> Result<BoolCol<'_>, FormulaError> {
        let values = self
            .bool_columns
            .get(col.name())
            .ok_or_else(|| FormulaError::UnknownColumn(col.name().to_owned()))?;
        Ok(BoolCol::Borrowed(values))
    }

    fn cat_col(&self, col: &Col<Category>) -> Result<CatCol<'_>, FormulaError> {
        let values = self
            .cat_columns
            .get(col.name())
            .ok_or_else(|| FormulaError::UnknownColumn(col.name().to_owned()))?;
        Ok(CatCol::Borrowed(values))
    }
}

#[test]
fn builds_normal_with_typed_columns_and_metadata() {
    let data = TestData::borrowed(&[("y", &[0.0, 1.0, 2.0]), ("x", &[1.0, 2.0, 3.0])]);
    let y = col::<f64>("y");
    let x = col::<f64>("x");

    let mut built = normal()
        .response(y.clone())
        .mu(intercept() + linear(x))
        .sigma(intercept())
        .build(&data)
        .unwrap();
    let beta = vec![0.0, 0.5, -0.2];

    assert_eq!(built.model().nparams(), 3);
    assert_eq!(built.layout().unique_slice("mu").unwrap().unwrap(), 0..2);
    assert_eq!(built.layout().unique_slice("sigma").unwrap().unwrap(), 2..3);
    assert_eq!(built.schema().response.col, y);
    assert_eq!(built.terms()[0].terms[1].range(), 1..2);
    assert!(built.model_mut().value(&beta).unwrap().is_finite());
}

#[test]
fn no_intercept_builds_parameter_without_implicit_intercept() {
    let data = TestData::borrowed(&[("y", &[0.0, 1.0, 2.0]), ("x", &[2.0, 3.0, 4.0])]);
    let built = normal()
        .response(col("y"))
        .mu(no_intercept() + linear(col("x")))
        .sigma(intercept())
        .build(&data)
        .unwrap();

    assert_eq!(built.coefficient_names(), vec!["mu.x", "sigma.(Intercept)"]);
    assert_eq!(built.layout().unique_slice("mu").unwrap().unwrap(), 0..1);
    assert_eq!(built.layout().unique_slice("sigma").unwrap().unwrap(), 1..2);

    let predicted = built.predict_theta(&[0.5, 0.0], &data).unwrap();
    assert_relative_eq!(predicted[0].mu, 1.0);
    assert_relative_eq!(predicted[2].mu, 2.0);
}

#[test]
fn supports_owned_response_columns() {
    let data = TestData::owned(&[("y", &[0.0, 1.0, 2.0]), ("x", &[1.0, 2.0, 3.0])]);
    let built = normal()
        .response(col("y"))
        .mu(linear(col("x")))
        .sigma(intercept())
        .build(&data)
        .unwrap();

    assert_eq!(built.model().obs().as_slice(), &[0.0, 1.0, 2.0]);
    assert!(matches!(built.model().obs(), NumericResponse::Owned(_)));
}

#[test]
fn validates_unknown_columns_and_lengths() {
    let data = TestData::borrowed(&[("y", &[0.0, 1.0]), ("x", &[1.0])]);

    let err = normal()
        .response(col("missing"))
        .mu(intercept())
        .sigma(intercept())
        .build(&data)
        .unwrap_err();
    assert_eq!(err, FormulaError::UnknownColumn("missing".to_owned()));

    let err = normal()
        .response(col("y"))
        .mu(linear(col("x")))
        .sigma(intercept())
        .build(&data)
        .unwrap_err();
    assert_eq!(
        err,
        FormulaError::ColumnLength {
            name: "x".to_owned(),
            expected: 2,
            actual: 1,
        }
    );
}

#[test]
fn numeric_inputs_response_domains_and_weights_are_validated() {
    let nonfinite = TestData::borrowed(&[("y", &[0.0, 1.0]), ("x", &[1.0, f64::NAN])]);
    let err = normal()
        .response(col("y"))
        .mu(linear(col("x")))
        .sigma(intercept())
        .build(&nonfinite)
        .unwrap_err();
    assert_eq!(
        err,
        FormulaError::NonFiniteValue {
            name: "x".to_owned(),
            row: 1,
        }
    );

    let invalid_gamma = TestData::borrowed(&[("y", &[0.0, 1.0])]);
    let err = gamma()
        .response(col("y"))
        .mean(intercept())
        .cv(intercept())
        .build(&invalid_gamma)
        .unwrap_err();
    assert_eq!(
        err,
        FormulaError::InvalidResponseDomain {
            name: "y".to_owned(),
            family: "gamma",
            row: 0,
        }
    );

    let invalid_weights = TestData::borrowed(&[("y", &[0.0, 1.0]), ("w", &[1.0, -0.1])]);
    let err = normal()
        .response(col("y"))
        .weights(col("w"))
        .mu(intercept())
        .sigma(intercept())
        .build(&invalid_weights)
        .unwrap_err();
    assert_eq!(
        err,
        FormulaError::InvalidWeight {
            name: "w".to_owned(),
            row: 1,
        }
    );
}

#[test]
fn zero_weights_exclude_rows_like_core_observation_weights() {
    let weighted = TestData::borrowed(&[("y", &[0.0, 10.0]), ("w", &[1.0, 0.0])]);
    let unweighted_first = TestData::borrowed(&[("y", &[0.0])]);
    let mut weighted = normal()
        .response(col("y"))
        .weights(col("w"))
        .mu(intercept())
        .sigma(intercept())
        .build(&weighted)
        .unwrap();
    let mut unweighted_first = normal()
        .response(col("y"))
        .mu(intercept())
        .sigma(intercept())
        .build(&unweighted_first)
        .unwrap();
    let theta = [0.0, 0.0];

    assert_relative_eq!(
        weighted.model_mut().value(&theta).unwrap(),
        unweighted_first.model_mut().value(&theta).unwrap(),
        epsilon = 1.0e-12
    );
}

#[test]
fn rejects_empty_data() {
    let data = TestData::borrowed(&[("y", &[]), ("x", &[])]);
    let err = normal()
        .response(col("y"))
        .mu(linear(col("x")))
        .sigma(intercept())
        .build(&data)
        .unwrap_err();

    assert_eq!(err, FormulaError::EmptyData);
}

#[test]
fn rejects_duplicate_parameter_terms_without_losing_first_spec() {
    let data = TestData::borrowed(&[("y", &[0.0, 1.0]), ("x", &[1.0, 2.0])]);
    let err = normal()
        .response(col("y"))
        .mu(linear(col("x")))
        .mu(intercept())
        .sigma(intercept())
        .build(&data)
        .unwrap_err();

    assert_eq!(err, FormulaError::DuplicateParameter("mu"));
}

#[test]
fn builds_supported_default_families() {
    let data = TestData::borrowed(&[
        ("y_pos", &[0.5, 1.0, 1.5]),
        ("y_unit", &[0.2, 0.5, 0.8]),
        ("x", &[1.0, 2.0, 3.0]),
    ]);
    let y_pos = col("y_pos");
    let y_unit = col("y_unit");
    let x = col("x");

    let gamma = gamma()
        .response(y_pos.clone())
        .mean(intercept() + linear(x.clone()))
        .cv(intercept())
        .build(&data)
        .unwrap();
    assert_eq!(gamma.layout().unique_slice("mean").unwrap().unwrap(), 0..2);
    assert_eq!(gamma.layout().unique_slice("cv").unwrap().unwrap(), 2..3);

    let log_normal = log_normal()
        .response(y_pos.clone())
        .mean(intercept())
        .log_sd(intercept() + linear(x.clone()))
        .build(&data)
        .unwrap();
    assert_eq!(
        log_normal.layout().unique_slice("mean").unwrap().unwrap(),
        0..1
    );
    assert_eq!(
        log_normal.layout().unique_slice("log_sd").unwrap().unwrap(),
        1..3
    );

    let weibull = weibull()
        .response(y_pos.clone())
        .mean(intercept() + linear(x.clone()))
        .shape(intercept())
        .build(&data)
        .unwrap();
    assert_eq!(
        weibull.layout().unique_slice("mean").unwrap().unwrap(),
        0..2
    );
    assert_eq!(
        weibull.layout().unique_slice("shape").unwrap().unwrap(),
        2..3
    );

    let inverse_gaussian = inverse_gaussian()
        .response(y_pos)
        .mu(intercept() + linear(x.clone()))
        .shape(intercept())
        .build(&data)
        .unwrap();
    assert_eq!(
        inverse_gaussian
            .layout()
            .unique_slice("mu")
            .unwrap()
            .unwrap(),
        0..2
    );
    assert_eq!(
        inverse_gaussian
            .layout()
            .unique_slice("shape")
            .unwrap()
            .unwrap(),
        2..3
    );

    let beta = beta()
        .response(y_unit)
        .mu(intercept())
        .precision(intercept() + linear(x))
        .build(&data)
        .unwrap();
    assert_eq!(beta.layout().unique_slice("mu").unwrap().unwrap(), 0..1);
    assert_eq!(
        beta.layout().unique_slice("precision").unwrap().unwrap(),
        1..3
    );
}

#[test]
fn pspline_stores_basis_and_reuses_it_for_prediction() {
    let train = TestData::borrowed(&[
        ("y", &[0.1, 0.2, 0.3, 0.4, 0.5]),
        ("x", &[0.0, 0.25, 0.5, 0.75, 1.0]),
    ]);
    let new_data = TestData::borrowed(&[("x", &[-0.25, 0.25, 1.25])]);
    let built = normal()
        .response(col("y"))
        .mu(intercept() + pspline(col("x")).k(6).lambda(0.7))
        .sigma(intercept())
        .build(&train)
        .unwrap();

    let FittedTerm::PSpline {
        basis,
        lambda,
        range,
        ..
    } = &built.terms()[0].terms[1]
    else {
        panic!("expected pspline metadata");
    };
    assert_eq!(basis.n_basis(), 6);
    assert_eq!(*lambda, 0.7);
    assert_eq!(range.clone(), 1..7);

    let blocks = built.prediction_blocks(&new_data).unwrap();
    assert_eq!(blocks.as_inner().0.len(), 7);
    let theta = vec![0.0; built.model().nparams()];
    let predicted = built.predict_theta(&theta, &new_data).unwrap();
    assert_eq!(predicted.len(), 3);
}

#[test]
fn mixed_terms_keep_layout_ranges_and_dense_order() {
    let data = TestData::borrowed(&[
        ("y", &[0.1, 0.2, 0.3, 0.4, 0.5]),
        ("x", &[1.0, 2.0, 3.0, 4.0, 5.0]),
        ("z", &[0.0, 0.25, 0.5, 0.75, 1.0]),
    ]);
    let built = normal()
        .response(col("y"))
        .mu(intercept() + linear(col("x")) + pspline(col("z")).k(6))
        .sigma(intercept())
        .build(&data)
        .unwrap();

    assert_eq!(built.layout().unique_slice("mu").unwrap().unwrap(), 0..8);
    assert_eq!(built.layout().unique_slice("sigma").unwrap().unwrap(), 8..9);
    assert_eq!(built.terms()[0].terms[0].range(), 0..1);
    assert_eq!(built.terms()[0].terms[1].range(), 1..2);
    assert_eq!(built.terms()[0].terms[2].range(), 2..8);

    let FittedTerm::Linear { coefficient, .. } = &built.terms()[0].terms[1] else {
        panic!("expected linear term");
    };
    assert_eq!(coefficient, "mu.x");

    let FittedTerm::PSpline { coefficients, .. } = &built.terms()[0].terms[2] else {
        panic!("expected pspline term");
    };
    assert_eq!(coefficients[0], "mu.z:pspline[0]");
    assert_eq!(coefficients[5], "mu.z:pspline[5]");

    for (row, values) in built
        .model()
        .blocks()
        .as_inner()
        .0
        .x()
        .dense()
        .values()
        .as_chunks::<8>()
        .0
        .iter()
        .enumerate()
    {
        assert_relative_eq!(values[0], 1.0);
        assert_relative_eq!(values[1], (row + 1) as f64);
    }
}

#[test]
fn offset_changes_predictions_without_adding_coefficients() {
    let data = TestData::borrowed(&[
        ("y", &[0.0, 1.0, 2.0]),
        ("x", &[1.0, 2.0, 3.0]),
        ("o", &[10.0, 20.0, 30.0]),
    ]);
    let built = normal()
        .response(col("y"))
        .mu(intercept() + linear(col("x")) + offset(col("o")))
        .sigma(intercept())
        .build(&data)
        .unwrap();
    let theta = [1.0, 2.0, 0.0];
    let predicted = built.predict_theta(&theta, &data).unwrap();

    assert_eq!(built.layout().unique_slice("mu").unwrap().unwrap(), 0..2);
    assert_relative_eq!(predicted[0].mu, 13.0);
    assert_relative_eq!(predicted[1].mu, 25.0);
}

#[test]
fn factor_indicator_and_interaction_build_expected_metadata() {
    let data = TestData::borrowed(&[
        ("y", &[0.0, 1.0, 2.0, 3.0]),
        ("x", &[1.0, 2.0, 3.0, 4.0]),
        ("z", &[0.5, 1.0, 1.5, 2.0]),
    ])
    .with_bool_column("flag", &[false, true, false, true])
    .with_cat_column("group", &["b", "a", "c", "b"]);
    let built = normal()
        .response(col("y"))
        .mu(intercept()
            + factor(col("group"))
            + indicator(col("flag"))
            + interaction(col("x"), col("z")))
        .sigma(intercept())
        .build(&data)
        .unwrap();

    assert_eq!(built.layout().unique_slice("mu").unwrap().unwrap(), 0..5);
    let names = built.coefficient_names();
    assert_eq!(
        names[..5],
        [
            "mu.(Intercept)",
            "mu.group[b]",
            "mu.group[c]",
            "mu.flag",
            "mu.x:z",
        ]
    );

    let FittedTerm::Factor {
        levels, baseline, ..
    } = &built.terms_for("mu").unwrap()[1]
    else {
        panic!("expected factor");
    };
    assert_eq!(levels, &["a", "b", "c"]);
    assert_eq!(baseline, "a");
}

#[test]
fn factor_prediction_rejects_unseen_levels() {
    let train = TestData::borrowed(&[("y", &[0.0, 1.0]), ("x", &[1.0, 2.0])])
        .with_cat_column("group", &["a", "b"]);
    let new_data = TestData::borrowed(&[("x", &[1.0])]).with_cat_column("group", &["missing"]);
    let built = normal()
        .response(col("y"))
        .mu(factor(col("group")))
        .sigma(intercept())
        .build(&train)
        .unwrap();

    let err = built.prediction_blocks(&new_data).unwrap_err();
    assert_eq!(
        err,
        FormulaError::UnknownCategoryLevel {
            name: "group".to_owned(),
            level: "missing".to_owned(),
            row: 0,
        }
    );
}

#[test]
fn unsupported_bool_and_category_columns_have_typed_errors() {
    #[derive(Debug)]
    struct NumericOnly {
        y: Vec<f64>,
    }
    impl DataView for NumericOnly {
        fn nrows(&self) -> usize {
            self.y.len()
        }

        fn f64_col(&self, col: &Col<f64>) -> Result<NumericCol<'_>, FormulaError> {
            match col.name() {
                "y" => Ok(NumericCol::Borrowed(&self.y)),
                name => Err(FormulaError::UnknownColumn(name.to_owned())),
            }
        }
    }

    let data = NumericOnly { y: vec![0.0] };
    let err = normal()
        .response(col("y"))
        .mu(indicator(col("flag")))
        .sigma(intercept())
        .build(&data)
        .unwrap_err();
    assert_eq!(
        err,
        FormulaError::UnsupportedColumnType {
            name: "flag".to_owned(),
            requested: "bool",
        }
    );
}

#[test]
fn advanced_numeric_terms_build_and_predict() {
    let data = TestData::borrowed(&[
        ("y", &[0.1, 0.2, 0.3, 0.4, 0.5]),
        ("x", &[0.0, 0.25, 0.5, 0.75, 1.0]),
        ("z", &[1.0, 1.25, 1.5, 1.75, 2.0]),
        ("season", &[0.0, 0.25, 0.5, 0.75, 1.0]),
    ]);
    let built = normal()
        .response(col("y"))
        .mu(cyclic_pspline(col("x")).k(6)
            + fourier(col("season")).period(1.0).order(2)
            + tensor_pspline(col("x"), col("z")).k(4, 4)
            + monotone(col("z")).k(5))
        .sigma(intercept())
        .build(&data)
        .unwrap();
    let theta = vec![0.0; built.model().nparams()];
    let predicted = built.predict_theta(&theta, &data).unwrap();

    assert_eq!(predicted.len(), 5);
    assert!(predicted.iter().all(|theta| theta.mu.is_finite()));
    assert!(
        built
            .terms_for("mu")
            .unwrap()
            .iter()
            .any(|term| matches!(term, FittedTerm::Monotone { .. }))
    );
}

#[test]
fn monotone_weighted_gradient_skips_zero_score_multiplier_rows() {
    let basis = ISplineBasis::open_uniform_from_data(&[0.0, 0.5, 1.0], 5, 3).unwrap();
    let nparams = basis.n_basis() + 1;
    let block = FormulaPredictorBlock::new(
        DenseDesign::from_row_major(3, 0, Vec::new()).unwrap(),
        None,
        vec![crate::predictor::MonotoneSegment {
            range: 0..nparams,
            design: MonotoneISplineDesign::new(
                &[0.0, 0.5, 1.0],
                basis,
                MonotoneDirection::Increasing,
            )
            .unwrap(),
        }],
        nparams,
    );
    let scores = [1.0, 0.0, 2.0];
    let multiplier = [1.0, f64::NAN, 1.0];
    let beta = vec![0.0; nparams];
    let mut expected = vec![0.0; nparams];
    let mut weighted = vec![0.0; nparams];
    let mut tiled_weighted = vec![0.0; nparams];

    block.add_gradient(&scores, &beta, &mut expected);
    block.add_weighted_gradient(&scores, &multiplier, &beta, &mut weighted);
    block.add_weighted_gradient_by_range(
        0..1,
        &scores[..1],
        multiplier.as_slice(),
        &beta,
        &mut tiled_weighted,
    );
    block.add_weighted_gradient_by_range(
        1..3,
        &scores[1..],
        multiplier.as_slice(),
        &beta,
        &mut tiled_weighted,
    );

    assert_eq!(weighted, expected);
    assert_eq!(tiled_weighted, expected);
    assert!(weighted.iter().all(|value| value.is_finite()));
}

#[test]
fn reusable_prediction_design_matches_convenience_prediction() {
    let train = TestData::borrowed(&[("y", &[0.0, 1.0, 2.0]), ("x", &[1.0, 2.0, 3.0])]);
    let new_data = TestData::borrowed(&[("x", &[4.0, 5.0])]);
    let built = normal()
        .response(col("y"))
        .mu(intercept() + linear(col("x")))
        .sigma(intercept())
        .build(&train)
        .unwrap();
    let theta = [1.0, 2.0, 0.0];
    let design = built.prediction_design(&new_data).unwrap();

    assert_eq!(
        built.predict_theta(&theta, &new_data).unwrap(),
        built.predict_theta_with_design(&theta, &design).unwrap()
    );
    assert!(built.terms_for("missing").is_none());
}

#[test]
fn pspline_builder_exposes_configured_options() {
    let x = col::<f64>("x");
    let term = pspline(x.clone())
        .k(8)
        .order(SplineOrder::Quadratic)
        .lambda(0.25)
        .penalty_order(3);

    assert_eq!(term.col(), &x);
    assert_eq!(term.n_basis(), 8);
    assert_eq!(term.spline_order(), SplineOrder::Quadratic);
    assert_eq!(term.penalty_lambda(), 0.25);
    assert_eq!(term.difference_order(), 3);
}

#[test]
fn tensor_builder_exposes_anisotropic_options_and_kind() {
    let x = col::<f64>("x");
    let z = col::<f64>("z");
    let full = te(x.clone(), z.clone())
        .k(7, 8)
        .order(SplineOrder::Quadratic, SplineOrder::Linear)
        .lambda(0.25, 3.0)
        .penalty_order(1, 3);
    let interaction = ti(x.clone(), z.clone());

    assert_eq!(full.left(), &x);
    assert_eq!(full.right(), &z);
    assert_eq!(full.basis_counts(), (7, 8));
    assert_eq!(
        full.spline_orders(),
        (SplineOrder::Quadratic, SplineOrder::Linear)
    );
    assert_eq!(full.penalty_lambdas(), (0.25, 3.0));
    assert_eq!(full.difference_orders(), (1, 3));
    assert_eq!(full.kind(), TensorSmoothKind::Full);
    assert_eq!(interaction.kind(), TensorSmoothKind::Interaction);
}

#[test]
fn tensor_pspline_compiles_anisotropic_penalty_and_preserves_it_for_prediction() {
    let data = TestData::borrowed(&[
        ("y", &[0.1, 0.2, 0.4, 0.7, 1.0]),
        ("x", &[0.0, 0.2, 0.5, 0.8, 1.0]),
        ("z", &[-1.0, -0.3, 0.1, 0.6, 1.0]),
    ]);
    let built = normal()
        .response(col("y"))
        .mu(tensor_pspline(col("x"), col("z"))
            .k(4, 5)
            .lambda(0.3, 2.0)
            .penalty_order(1, 2))
        .sigma(intercept())
        .build(&data)
        .unwrap();
    let FittedTerm::TensorPSpline {
        range,
        kind,
        left_lambda,
        right_lambda,
        left_penalty_order,
        right_penalty_order,
        ..
    } = &built.terms_for("mu").unwrap()[0]
    else {
        panic!("expected tensor P-spline metadata");
    };
    assert_eq!(range.len(), 20);
    assert_eq!(*kind, TensorSmoothKind::Full);
    assert_eq!((*left_lambda, *right_lambda), (0.3, 2.0));
    assert_eq!((*left_penalty_order, *right_penalty_order), (1, 2));

    let beta = (0..range.len())
        .map(|index| (index as f64 * 0.37).sin())
        .collect::<Vec<_>>();
    let training_penalty = built.model().blocks().as_inner().0.penalty();
    let prediction_blocks = built.prediction_blocks(&data).unwrap();
    let prediction_penalty = prediction_blocks.as_inner().0.penalty();
    assert!(training_penalty.value(&beta) > 0.0);
    assert_relative_eq!(
        training_penalty.value(&beta),
        prediction_penalty.value(&beta),
        epsilon = 1.0e-12
    );
    assert_penalty_gradient_matches_finite_difference(training_penalty, &beta);
}

#[test]
fn tensor_interaction_removes_marginal_constant_directions() {
    let x_values = [0.0, 0.2, 0.5, 0.8, 1.0];
    let z_values = [-1.0, -0.3, 0.1, 0.6, 1.0];
    let data = TestData::borrowed(&[
        ("y", &[0.1, 0.2, 0.4, 0.7, 1.0]),
        ("x", &x_values),
        ("z", &z_values),
    ]);
    let built = normal()
        .response(col("y"))
        .mu(tensor_pspline_interaction(col("x"), col("z"))
            .k(4, 5)
            .lambda(0.4, 1.7)
            .penalty_order(1, 2))
        .sigma(intercept())
        .build(&data)
        .unwrap();
    let FittedTerm::TensorPSpline {
        range,
        left_basis,
        right_basis,
        kind,
        ..
    } = &built.terms_for("mu").unwrap()[0]
    else {
        panic!("expected tensor interaction metadata");
    };
    assert_eq!(*kind, TensorSmoothKind::Interaction);
    assert_eq!(range.len(), (4 - 1) * (5 - 1));

    let dense = built.model().blocks().as_inner().0.x().dense();
    assert_eq!(dense.ncols(), range.len());
    let mut left = vec![0.0; left_basis.n_basis()];
    let mut right = vec![0.0; right_basis.n_basis()];
    for row in 0..data.nrows() {
        left.fill(0.0);
        right.fill(0.0);
        left_basis
            .for_each_value_basis(x_values[row], |index, value| left[index] = value)
            .unwrap();
        right_basis
            .for_each_value_basis(z_values[row], |index, value| right[index] = value)
            .unwrap();
        let actual = &dense.values()[row * range.len()..(row + 1) * range.len()];
        for left_index in 0..left.len() - 1 {
            for right_index in 0..right.len() - 1 {
                let index = left_index * (right.len() - 1) + right_index;
                let left_leading = (left_index + 1) as f64;
                let right_leading = (right_index + 1) as f64;
                let left_contrast = (left[..=left_index].iter().sum::<f64>()
                    - left_leading * left[left_index + 1])
                    / (left_leading * (left_leading + 1.0)).sqrt();
                let right_contrast = (right[..=right_index].iter().sum::<f64>()
                    - right_leading * right[right_index + 1])
                    / (right_leading * (right_leading + 1.0)).sqrt();
                let expected = left_contrast * right_contrast;
                assert_relative_eq!(actual[index], expected, epsilon = 1.0e-14);
            }
        }
    }

    let beta = (0..range.len())
        .map(|index| (index as f64 * 0.41).cos())
        .collect::<Vec<_>>();
    let penalty = built.model().blocks().as_inner().0.penalty();
    assert!(penalty.value(&beta) > 0.0);
    assert_penalty_gradient_matches_finite_difference(penalty, &beta);

    let prediction = built.prediction_blocks(&data).unwrap();
    assert_eq!(prediction.as_inner().0.x().dense().values(), dense.values());
}

fn assert_penalty_gradient_matches_finite_difference<P>(penalty: &P, beta: &[f64])
where
    P: Penalty,
{
    let mut gradient = vec![0.0; beta.len()];
    penalty.add_gradient(beta, &mut gradient);
    let step = 1.0e-6;
    for index in 0..beta.len() {
        let mut lower = beta.to_vec();
        let mut upper = beta.to_vec();
        lower[index] -= step;
        upper[index] += step;
        let finite_difference = (penalty.value(&upper) - penalty.value(&lower)) / (2.0 * step);
        assert_relative_eq!(finite_difference, gradient[index], epsilon = 2.0e-7);
    }
}

#[test]
fn pspline_rejects_invalid_training_data() {
    let data = TestData::borrowed(&[("y", &[0.0, 1.0]), ("x", &[1.0, 1.0])]);
    let err = normal()
        .response(col("y"))
        .mu(pspline(col("x")).k(6))
        .sigma(intercept())
        .build(&data)
        .unwrap_err();

    assert_eq!(err, FormulaError::Spline(SplineError::InvalidRange));
}

#[test]
fn formula_penalty_changes_objective_and_gradient() {
    let data = TestData::borrowed(&[
        ("y", &[0.1, 0.2, 0.3, 0.4, 0.5]),
        ("x", &[0.0, 0.25, 0.5, 0.75, 1.0]),
    ]);
    let mut built = normal()
        .response(col("y"))
        .mu(pspline(col("x")).k(6).lambda(2.0).penalty_order(1))
        .sigma(intercept())
        .build(&data)
        .unwrap();
    let mut theta = vec![0.0; built.model().nparams()];
    theta[0] = 1.0;
    theta[1] = -1.0;
    theta[2] = 0.5;
    let mut grad = vec![0.0; theta.len()];

    let objective = built.model_mut().value(&theta).unwrap();
    built.model_mut().gradient(&theta, &mut grad).unwrap();
    let no_penalty_theta = vec![0.0; theta.len()];
    let baseline = built.model_mut().value(&no_penalty_theta).unwrap();

    assert!(objective > baseline);
    assert!(grad[..6].iter().any(|value| value.abs() > 1.0e-8));
}

#[test]
fn cyclic_pspline_prediction_blocks_preserve_penalty_metadata() {
    let data = TestData::borrowed(&[
        ("y", &[0.1, 0.2, 0.3, 0.4, 0.5]),
        ("phase", &[0.0, 0.2, 0.4, 0.6, 0.8]),
    ]);
    let built = normal()
        .response(col("y"))
        .mu(cyclic_pspline(col("phase"))
            .k(6)
            .lambda(2.0)
            .penalty_order(1))
        .sigma(intercept())
        .build(&data)
        .unwrap();
    let blocks = built.prediction_blocks(&data).unwrap();
    let theta = [0.0, 1.0, -1.0, 0.5, 0.25, -0.25];
    let mut grad = [0.0; 6];

    let value = blocks.as_inner().0.penalty().value(&theta);
    blocks
        .as_inner()
        .0
        .penalty()
        .add_gradient(&theta, &mut grad);

    assert!(value > 0.0);
    assert!(grad.iter().any(|value| f64::abs(*value) > 1.0e-8));
}

#[test]
fn multiple_spline_segments_gradient_matches_finite_difference() {
    let data = TestData::borrowed(&[
        ("y", &[0.2, 0.3, 0.5, 0.8, 1.3]),
        ("x1", &[0.0, 0.25, 0.5, 0.75, 1.0]),
        ("x2", &[1.0, 1.25, 1.5, 1.75, 2.0]),
    ]);
    let mut built = normal()
        .response(col("y"))
        .mu(pspline(col("x1")).k(5).lambda(0.5) + pspline(col("x2")).k(5).lambda(0.8))
        .sigma(intercept())
        .build(&data)
        .unwrap();
    let theta = vec![
        0.1, -0.2, 0.3, -0.1, 0.2, 0.0, 0.25, -0.15, 0.35, -0.05, -0.3,
    ];
    let eps = 1.0e-6;
    let mut grad = vec![0.0; theta.len()];

    built.model_mut().gradient(&theta, &mut grad).unwrap();

    for index in 0..theta.len() {
        let mut plus = theta.clone();
        let mut minus = theta.clone();
        plus[index] += eps;
        minus[index] -= eps;

        let finite_difference = (built.model_mut().value(&plus).unwrap()
            - built.model_mut().value(&minus).unwrap())
            / (2.0 * eps);
        assert_relative_eq!(grad[index], finite_difference, epsilon = 1.0e-5);
    }
}

#[test]
fn prediction_reuses_training_spline_range_not_newdata_range() {
    let train = TestData::borrowed(&[("y", &[0.0, 0.1, 0.2, 0.3]), ("x", &[0.0, 0.25, 0.75, 1.0])]);
    let new_data = TestData::borrowed(&[("x", &[10.0, 11.0])]);
    let built = normal()
        .response(col("y"))
        .mu(pspline(col("x")).k(5))
        .sigma(intercept())
        .build(&train)
        .unwrap();

    let FittedTerm::PSpline { basis, .. } = &built.terms()[0].terms[0] else {
        panic!("expected pspline");
    };
    assert_relative_eq!(basis.min(), 0.0);
    assert_relative_eq!(basis.max(), 1.0);

    let blocks = built.prediction_blocks(&new_data).unwrap();
    assert_eq!(blocks.as_inner().0.len(), 5);
}
