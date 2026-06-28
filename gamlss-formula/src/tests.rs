#![allow(clippy::float_cmp, clippy::cast_precision_loss)]
use std::collections::BTreeMap;

use approx::assert_relative_eq;
use gamlss_core::{Objective, Penalty};
use gamlss_spline::{SplineError, SplineOrder};

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
    assert_eq!(built.layout().slice("mu").unwrap(), 0..2);
    assert_eq!(built.layout().slice("sigma").unwrap(), 2..3);
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
    assert_eq!(built.layout().slice("mu").unwrap(), 0..1);
    assert_eq!(built.layout().slice("sigma").unwrap(), 1..2);

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
    assert_eq!(gamma.layout().slice("mean").unwrap(), 0..2);
    assert_eq!(gamma.layout().slice("cv").unwrap(), 2..3);

    let log_normal = log_normal()
        .response(y_pos.clone())
        .mean(intercept())
        .log_sd(intercept() + linear(x.clone()))
        .build(&data)
        .unwrap();
    assert_eq!(log_normal.layout().slice("mean").unwrap(), 0..1);
    assert_eq!(log_normal.layout().slice("log_sd").unwrap(), 1..3);

    let weibull = weibull()
        .response(y_pos.clone())
        .mean(intercept() + linear(x.clone()))
        .shape(intercept())
        .build(&data)
        .unwrap();
    assert_eq!(weibull.layout().slice("mean").unwrap(), 0..2);
    assert_eq!(weibull.layout().slice("shape").unwrap(), 2..3);

    let inverse_gaussian = inverse_gaussian()
        .response(y_pos)
        .mu(intercept() + linear(x.clone()))
        .shape(intercept())
        .build(&data)
        .unwrap();
    assert_eq!(inverse_gaussian.layout().slice("mu").unwrap(), 0..2);
    assert_eq!(inverse_gaussian.layout().slice("shape").unwrap(), 2..3);

    let beta = beta()
        .response(y_unit)
        .mu(intercept())
        .precision(intercept() + linear(x))
        .build(&data)
        .unwrap();
    assert_eq!(beta.layout().slice("mu").unwrap(), 0..1);
    assert_eq!(beta.layout().slice("precision").unwrap(), 1..3);
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
    assert_eq!(blocks.0.len(), 7);
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

    assert_eq!(built.layout().slice("mu").unwrap(), 0..8);
    assert_eq!(built.layout().slice("sigma").unwrap(), 8..9);
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

    assert_eq!(built.layout().slice("mu").unwrap(), 0..2);
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

    assert_eq!(built.layout().slice("mu").unwrap(), 0..5);
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

    let value = blocks.0.penalty().value(&theta);
    blocks.0.penalty().add_gradient(&theta, &mut grad);

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
    assert_eq!(blocks.0.len(), 5);
}
