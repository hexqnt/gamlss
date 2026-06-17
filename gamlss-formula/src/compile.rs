use std::collections::BTreeSet;
use std::ops::Range;

use gamlss_core::{DenseDesign, ModelError};
use gamlss_spline::{
    CyclicSplineSpec, FourierDesign, ISplineBasis, OpenUniformSplineBasis, SplineRowBasis,
    TensorSplineDesign,
};

use crate::predictor::{FormulaPredictorBlock, MonotoneSegment};
use crate::{
    BoolCol, CatCol, Col, DataView, FittedTerm, FormulaError, FormulaPenalty, NumericCol,
    NumericResponse, ParameterTerms, TermExpr, TermSpec,
};

#[derive(Debug, Clone, Copy)]
pub(crate) enum ResponseDomain {
    Finite,
    Positive,
    Unit,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ParameterBuild {
    pub(crate) predictor: FormulaPredictorBlock,
    pub(crate) penalty: FormulaPenalty,
    pub(crate) terms: ParameterTerms,
}

#[derive(Debug)]
enum PreparedDenseTerm<'a> {
    Intercept {
        range: Range<usize>,
    },
    Numeric {
        values: NumericCol<'a>,
        range: Range<usize>,
    },
    OwnedColumns {
        values: Vec<f64>,
        range: Range<usize>,
    },
    PSpline {
        values: NumericCol<'a>,
        basis: OpenUniformSplineBasis,
        range: Range<usize>,
    },
    CyclicPSpline {
        values: NumericCol<'a>,
        spec: CyclicSplineSpec,
        range: Range<usize>,
    },
    Fourier {
        design: FourierDesign,
        range: Range<usize>,
    },
    TensorPSpline {
        left: NumericCol<'a>,
        right: NumericCol<'a>,
        left_basis: OpenUniformSplineBasis,
        right_basis: OpenUniformSplineBasis,
        range: Range<usize>,
    },
}

struct RowMajorDesignBuilder {
    nrows: usize,
    ncols: usize,
    values: Vec<f64>,
}

impl RowMajorDesignBuilder {
    fn new(nrows: usize, ncols: usize) -> Result<Self, FormulaError> {
        let len = nrows
            .checked_mul(ncols)
            .ok_or(ModelError::ArithmeticOverflow {
                context: "formula design row-major value count",
            })?;

        Ok(Self {
            nrows,
            ncols,
            values: vec![0.0; len],
        })
    }

    fn fill_intercept(&mut self, range: &Range<usize>) {
        debug_assert_eq!(range.len(), 1);

        let col = range.start;
        for row in 0..self.nrows {
            self.set(row, col, 1.0);
        }
    }

    fn fill_column(&mut self, range: &Range<usize>, values: &[f64]) {
        debug_assert_eq!(range.len(), 1);
        debug_assert_eq!(values.len(), self.nrows);

        let col = range.start;
        for (row, value) in values.iter().copied().enumerate() {
            self.set(row, col, value);
        }
    }

    fn fill_pspline(
        &mut self,
        range: &Range<usize>,
        basis: &OpenUniformSplineBasis,
        values: &[f64],
    ) -> Result<(), FormulaError> {
        debug_assert_eq!(range.len(), basis.n_basis());
        debug_assert_eq!(values.len(), self.nrows);

        for (row, value) in values.iter().copied().enumerate() {
            let row_offset = self.row_offset(row) + range.start;
            basis.for_each_value_basis(value, |local_col, weight| {
                self.values[row_offset + local_col] = weight;
            })?;
        }
        Ok(())
    }

    fn fill_row_basis<B>(&mut self, range: &Range<usize>, basis: &B)
    where
        B: SplineRowBasis,
    {
        debug_assert_eq!(range.len(), basis.nparams());
        debug_assert_eq!(basis.nrows(), self.nrows);

        for row in 0..self.nrows {
            let row_offset = self.row_offset(row) + range.start;
            basis.for_each_row_basis(row, |local_col, weight| {
                self.values[row_offset + local_col] = weight;
            });
        }
    }

    fn finish(self) -> Result<DenseDesign, FormulaError> {
        Ok(DenseDesign::from_row_major(
            self.nrows,
            self.ncols,
            self.values,
        )?)
    }

    fn set(&mut self, row: usize, col: usize, value: f64) {
        let index = self.row_offset(row) + col;
        self.values[index] = value;
    }

    fn row_offset(&self, row: usize) -> usize {
        row * self.ncols
    }
}

pub(crate) fn validate_col_len(
    name: &str,
    values: &[f64],
    expected: usize,
) -> Result<(), FormulaError> {
    validate_len(name, values.len(), expected)
}

fn validate_bool_col_len(name: &str, values: &[bool], expected: usize) -> Result<(), FormulaError> {
    validate_len(name, values.len(), expected)
}

fn validate_cat_col_len(
    name: &str,
    values: &[String],
    expected: usize,
) -> Result<(), FormulaError> {
    validate_len(name, values.len(), expected)
}

fn validate_len(name: &str, actual: usize, expected: usize) -> Result<(), FormulaError> {
    if actual == expected {
        Ok(())
    } else {
        Err(FormulaError::ColumnLength {
            name: name.to_owned(),
            expected,
            actual,
        })
    }
}

fn validate_finite(name: &str, values: &[f64]) -> Result<(), FormulaError> {
    for (row, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(FormulaError::NonFiniteValue {
                name: name.to_owned(),
                row,
            });
        }
    }
    Ok(())
}

fn validate_weights(name: &str, values: &[f64]) -> Result<(), FormulaError> {
    for (row, value) in values.iter().copied().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(FormulaError::InvalidWeight {
                name: name.to_owned(),
                row,
            });
        }
    }
    Ok(())
}

fn validate_response_domain(
    family: &'static str,
    domain: ResponseDomain,
    name: &str,
    values: &[f64],
) -> Result<(), FormulaError> {
    for (row, value) in values.iter().copied().enumerate() {
        let valid = match domain {
            ResponseDomain::Finite => value.is_finite(),
            ResponseDomain::Positive => value.is_finite() && value > 0.0,
            ResponseDomain::Unit => value.is_finite() && value > 0.0 && value < 1.0,
        };
        if !valid {
            return Err(FormulaError::InvalidResponseDomain {
                name: name.to_owned(),
                family,
                row,
            });
        }
    }
    Ok(())
}

pub(crate) fn required_response<'a, D>(
    family: &'static str,
    domain: ResponseDomain,
    data: &'a D,
    response: &Option<Col<f64>>,
    weights: &Option<Col<f64>>,
) -> Result<(Col<f64>, NumericResponse<'a>), FormulaError>
where
    D: DataView + ?Sized,
{
    let col = response.clone().ok_or(FormulaError::MissingResponse)?;
    let values = data.f64_col(&col)?;
    validate_col_len(col.name(), values.as_slice(), data.nrows())?;
    validate_response_domain(family, domain, col.name(), values.as_slice())?;

    let Some(weight_col) = weights else {
        return Ok((col, values.into_response()));
    };

    let weight_values = data.f64_col(weight_col)?;
    validate_col_len(weight_col.name(), weight_values.as_slice(), data.nrows())?;
    validate_weights(weight_col.name(), weight_values.as_slice())?;
    Ok((
        col,
        NumericResponse::Weighted {
            values,
            weights: weight_values,
        },
    ))
}

pub(crate) fn fit_terms<D>(
    parameter: &'static str,
    expr: Option<TermExpr>,
    data: &D,
) -> Result<ParameterBuild, FormulaError>
where
    D: DataView + ?Sized,
{
    let terms = expr.unwrap_or_default().into_terms();
    let nrows = data.nrows();
    let mut fitted = Vec::with_capacity(terms.len());
    let mut prepared = Vec::with_capacity(terms.len());
    let mut monotone = Vec::new();
    let mut penalty = FormulaPenalty::default();
    let mut offset_values: Option<Vec<f64>> = None;
    let mut offset = 0usize;

    for term in terms {
        match term {
            TermSpec::Intercept => {
                let range = checked_range(offset, 1)?;
                prepared.push(PreparedDenseTerm::Intercept {
                    range: range.clone(),
                });
                fitted.push(FittedTerm::Intercept {
                    range: range.clone(),
                    coefficient: format!("{parameter}.(Intercept)"),
                });
                offset = range.end;
            }
            TermSpec::Linear { col } => {
                let values = numeric_col(data, &col)?;
                let range = checked_range(offset, 1)?;
                prepared.push(PreparedDenseTerm::Numeric {
                    values,
                    range: range.clone(),
                });
                fitted.push(FittedTerm::Linear {
                    coefficient: format!("{parameter}.{}", col.name()),
                    col,
                    range: range.clone(),
                });
                offset = range.end;
            }
            TermSpec::Offset { col } => {
                let values = numeric_col(data, &col)?;
                merge_offset(&mut offset_values, values.as_slice());
                fitted.push(FittedTerm::Offset { col });
            }
            TermSpec::Indicator { col } => {
                let values = bool_col(data, &col)?;
                let range = checked_range(offset, 1)?;
                prepared.push(PreparedDenseTerm::OwnedColumns {
                    values: values
                        .as_slice()
                        .iter()
                        .map(|value| f64::from(*value))
                        .collect(),
                    range: range.clone(),
                });
                fitted.push(FittedTerm::Indicator {
                    coefficient: format!("{parameter}.{}", col.name()),
                    col,
                    range: range.clone(),
                });
                offset = range.end;
            }
            TermSpec::Factor { col } => {
                let values = cat_col(data, &col)?;
                let levels = values
                    .as_slice()
                    .iter()
                    .cloned()
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>();
                let baseline = levels[0].clone();
                let width = levels.len().saturating_sub(1);
                let range = checked_range(offset, width)?;
                let columns = factor_columns(col.name(), values.as_slice(), &levels)?;
                let coefficients = levels
                    .iter()
                    .skip(1)
                    .map(|level| format!("{parameter}.{}[{level}]", col.name()))
                    .collect();
                prepared.push(PreparedDenseTerm::OwnedColumns {
                    values: columns,
                    range: range.clone(),
                });
                fitted.push(FittedTerm::Factor {
                    col,
                    range: range.clone(),
                    levels,
                    baseline,
                    coefficients,
                });
                offset = range.end;
            }
            TermSpec::Interaction { left, right } => {
                let left_values = numeric_col(data, &left)?;
                let right_values = numeric_col(data, &right)?;
                let range = checked_range(offset, 1)?;
                let values = left_values
                    .as_slice()
                    .iter()
                    .zip(right_values.as_slice())
                    .map(|(left, right)| left * right)
                    .collect();
                prepared.push(PreparedDenseTerm::OwnedColumns {
                    values,
                    range: range.clone(),
                });
                fitted.push(FittedTerm::Interaction {
                    coefficient: format!("{parameter}.{}:{}", left.name(), right.name()),
                    left,
                    right,
                    range: range.clone(),
                });
                offset = range.end;
            }
            TermSpec::PSpline(term) => {
                let values = numeric_col(data, &term.col)?;
                let basis =
                    OpenUniformSplineBasis::from_data(values.as_slice(), term.k, term.order)?;
                let width = basis.n_basis();
                let range = checked_range(offset, width)?;
                let coefficients = indexed_names(parameter, term.col.name(), "pspline", width);
                penalty.add_spline(range.clone(), term.lambda, term.penalty_order);
                prepared.push(PreparedDenseTerm::PSpline {
                    values,
                    basis,
                    range: range.clone(),
                });
                fitted.push(FittedTerm::PSpline {
                    col: term.col,
                    range: range.clone(),
                    basis,
                    lambda: term.lambda,
                    penalty_order: term.penalty_order,
                    coefficients,
                });
                offset = range.end;
            }
            TermSpec::CyclicPSpline(term) => {
                let values = numeric_col(data, &term.col)?;
                let spec = CyclicSplineSpec::new(term.k, term.order)?;
                let width = spec.n_basis();
                let range = checked_range(offset, width)?;
                let coefficients =
                    indexed_names(parameter, term.col.name(), "cyclic_pspline", width);
                penalty.add_cyclic_spline(range.clone(), term.lambda, term.penalty_order);
                prepared.push(PreparedDenseTerm::CyclicPSpline {
                    values,
                    spec,
                    range: range.clone(),
                });
                fitted.push(FittedTerm::CyclicPSpline {
                    col: term.col,
                    range: range.clone(),
                    spec,
                    lambda: term.lambda,
                    penalty_order: term.penalty_order,
                    coefficients,
                });
                offset = range.end;
            }
            TermSpec::Fourier(term) => {
                let values = numeric_col(data, &term.col)?;
                let design = FourierDesign::new(
                    values.as_slice(),
                    term.period,
                    term.order,
                    term.include_intercept,
                )?;
                let width = SplineRowBasis::nparams(&design);
                let range = checked_range(offset, width)?;
                let coefficients = fourier_names(
                    parameter,
                    term.col.name(),
                    term.order,
                    term.include_intercept,
                );
                prepared.push(PreparedDenseTerm::Fourier {
                    design,
                    range: range.clone(),
                });
                fitted.push(FittedTerm::Fourier {
                    col: term.col,
                    range: range.clone(),
                    period: term.period,
                    order: term.order,
                    include_intercept: term.include_intercept,
                    coefficients,
                });
                offset = range.end;
            }
            TermSpec::TensorPSpline(term) => {
                let left = numeric_col(data, &term.left)?;
                let right = numeric_col(data, &term.right)?;
                let left_basis = OpenUniformSplineBasis::from_data(
                    left.as_slice(),
                    term.left_k,
                    term.left_order,
                )?;
                let right_basis = OpenUniformSplineBasis::from_data(
                    right.as_slice(),
                    term.right_k,
                    term.right_order,
                )?;
                let width = left_basis
                    .n_basis()
                    .checked_mul(right_basis.n_basis())
                    .ok_or(ModelError::ArithmeticOverflow {
                        context: "formula tensor coefficient count",
                    })?;
                let range = checked_range(offset, width)?;
                let left_name = term.left.name().to_owned();
                let right_name = term.right.name().to_owned();
                let coefficients = (0..left_basis.n_basis())
                    .flat_map(|left_index| {
                        let left_name = left_name.clone();
                        let right_name = right_name.clone();
                        (0..right_basis.n_basis()).map(move |right_index| {
                            format!(
                                "{parameter}.{}:{}:tensor[{left_index},{right_index}]",
                                left_name, right_name
                            )
                        })
                    })
                    .collect();
                prepared.push(PreparedDenseTerm::TensorPSpline {
                    left,
                    right,
                    left_basis,
                    right_basis,
                    range: range.clone(),
                });
                fitted.push(FittedTerm::TensorPSpline {
                    left: term.left,
                    right: term.right,
                    range: range.clone(),
                    left_basis,
                    right_basis,
                    coefficients,
                });
                offset = range.end;
            }
            TermSpec::Monotone(term) => {
                let values = numeric_col(data, &term.col)?;
                let basis =
                    ISplineBasis::open_uniform_from_data(values.as_slice(), term.k, term.degree)?;
                let width =
                    1usize
                        .checked_add(basis.n_basis())
                        .ok_or(ModelError::ArithmeticOverflow {
                            context: "formula monotone coefficient count",
                        })?;
                let range = checked_range(offset, width)?;
                let mut coefficients = Vec::with_capacity(width);
                coefficients.push(format!(
                    "{parameter}.{}:monotone[intercept]",
                    term.col.name()
                ));
                coefficients.extend(
                    (0..basis.n_basis())
                        .map(|index| format!("{parameter}.{}:monotone[{index}]", term.col.name())),
                );
                monotone.push(MonotoneSegment {
                    range: range.clone(),
                    values: values.as_slice().to_vec(),
                    basis: basis.clone(),
                    direction: term.direction,
                });
                fitted.push(FittedTerm::Monotone {
                    col: term.col,
                    range: range.clone(),
                    basis,
                    direction: term.direction,
                    coefficients,
                });
                offset = range.end;
            }
        }
    }

    Ok(ParameterBuild {
        predictor: predictor_from_prepared(nrows, offset, prepared, offset_values, monotone)?,
        penalty,
        terms: ParameterTerms {
            parameter,
            terms: fitted,
        },
    })
}

pub(crate) fn predictor_from_fitted_terms<D>(
    parameter: &'static str,
    terms: &[FittedTerm],
    data: &D,
) -> Result<FormulaPredictorBlock, FormulaError>
where
    D: DataView + ?Sized,
{
    let nrows = data.nrows();
    let mut prepared = Vec::with_capacity(terms.len());
    let mut monotone = Vec::new();
    let mut offset_values: Option<Vec<f64>> = None;
    let mut nparams = 0usize;

    for term in terms {
        match term {
            FittedTerm::Intercept { range, .. } => {
                nparams = nparams.max(range.end);
                prepared.push(PreparedDenseTerm::Intercept {
                    range: range.clone(),
                });
            }
            FittedTerm::Linear { col, range, .. } => {
                let values = numeric_col(data, col)?;
                nparams = nparams.max(range.end);
                prepared.push(PreparedDenseTerm::Numeric {
                    values,
                    range: range.clone(),
                });
            }
            FittedTerm::Offset { col } => {
                let values = numeric_col(data, col)?;
                merge_offset(&mut offset_values, values.as_slice());
            }
            FittedTerm::Indicator { col, range, .. } => {
                let values = bool_col(data, col)?;
                nparams = nparams.max(range.end);
                prepared.push(PreparedDenseTerm::OwnedColumns {
                    values: values
                        .as_slice()
                        .iter()
                        .map(|value| f64::from(*value))
                        .collect(),
                    range: range.clone(),
                });
            }
            FittedTerm::Factor {
                col, range, levels, ..
            } => {
                let values = cat_col(data, col)?;
                nparams = nparams.max(range.end);
                prepared.push(PreparedDenseTerm::OwnedColumns {
                    values: factor_columns(col.name(), values.as_slice(), levels)?,
                    range: range.clone(),
                });
            }
            FittedTerm::Interaction {
                left, right, range, ..
            } => {
                let left_values = numeric_col(data, left)?;
                let right_values = numeric_col(data, right)?;
                nparams = nparams.max(range.end);
                prepared.push(PreparedDenseTerm::OwnedColumns {
                    values: left_values
                        .as_slice()
                        .iter()
                        .zip(right_values.as_slice())
                        .map(|(left, right)| left * right)
                        .collect(),
                    range: range.clone(),
                });
            }
            FittedTerm::PSpline {
                col, basis, range, ..
            } => {
                let values = numeric_col(data, col)?;
                nparams = nparams.max(range.end);
                prepared.push(PreparedDenseTerm::PSpline {
                    values,
                    basis: *basis,
                    range: range.clone(),
                });
            }
            FittedTerm::CyclicPSpline {
                col, spec, range, ..
            } => {
                let values = numeric_col(data, col)?;
                nparams = nparams.max(range.end);
                prepared.push(PreparedDenseTerm::CyclicPSpline {
                    values,
                    spec: *spec,
                    range: range.clone(),
                });
            }
            FittedTerm::Fourier {
                col,
                range,
                period,
                order,
                include_intercept,
                ..
            } => {
                let values = numeric_col(data, col)?;
                let design =
                    FourierDesign::new(values.as_slice(), *period, *order, *include_intercept)?;
                nparams = nparams.max(range.end);
                prepared.push(PreparedDenseTerm::Fourier {
                    design,
                    range: range.clone(),
                });
            }
            FittedTerm::TensorPSpline {
                left,
                right,
                range,
                left_basis,
                right_basis,
                ..
            } => {
                let left = numeric_col(data, left)?;
                let right = numeric_col(data, right)?;
                nparams = nparams.max(range.end);
                prepared.push(PreparedDenseTerm::TensorPSpline {
                    left,
                    right,
                    left_basis: *left_basis,
                    right_basis: *right_basis,
                    range: range.clone(),
                });
            }
            FittedTerm::Monotone {
                col,
                range,
                basis,
                direction,
                ..
            } => {
                let values = numeric_col(data, col)?;
                nparams = nparams.max(range.end);
                monotone.push(MonotoneSegment {
                    range: range.clone(),
                    values: values.as_slice().to_vec(),
                    basis: basis.clone(),
                    direction: *direction,
                });
            }
        }
    }

    if prepared.is_empty() && monotone.is_empty() && offset_values.is_none() {
        return Err(FormulaError::DuplicateParameter(parameter));
    }

    predictor_from_prepared(nrows, nparams, prepared, offset_values, monotone)
}

fn numeric_col<'a, D>(data: &'a D, col: &Col<f64>) -> Result<NumericCol<'a>, FormulaError>
where
    D: DataView + ?Sized,
{
    let values = data.f64_col(col)?;
    validate_col_len(col.name(), values.as_slice(), data.nrows())?;
    validate_finite(col.name(), values.as_slice())?;
    Ok(values)
}

fn bool_col<'a, D>(data: &'a D, col: &Col<bool>) -> Result<BoolCol<'a>, FormulaError>
where
    D: DataView + ?Sized,
{
    let values = data.bool_col(col)?;
    validate_bool_col_len(col.name(), values.as_slice(), data.nrows())?;
    Ok(values)
}

fn cat_col<'a, D>(data: &'a D, col: &Col<crate::Category>) -> Result<CatCol<'a>, FormulaError>
where
    D: DataView + ?Sized,
{
    let values = data.cat_col(col)?;
    validate_cat_col_len(col.name(), values.as_slice(), data.nrows())?;
    Ok(values)
}

fn merge_offset(offset: &mut Option<Vec<f64>>, values: &[f64]) {
    if let Some(offset) = offset {
        for (out, value) in offset.iter_mut().zip(values) {
            *out += value;
        }
    } else {
        *offset = Some(values.to_vec());
    }
}

fn predictor_from_prepared(
    nrows: usize,
    nparams: usize,
    terms: Vec<PreparedDenseTerm<'_>>,
    offset: Option<Vec<f64>>,
    monotone: Vec<MonotoneSegment>,
) -> Result<FormulaPredictorBlock, FormulaError> {
    let dense = dense_from_prepared_terms(nrows, dense_ncols(&terms), &terms)?;
    Ok(FormulaPredictorBlock::new(dense, offset, monotone, nparams))
}

fn dense_ncols(terms: &[PreparedDenseTerm<'_>]) -> usize {
    terms
        .iter()
        .map(|term| match term {
            PreparedDenseTerm::Intercept { range }
            | PreparedDenseTerm::Numeric { range, .. }
            | PreparedDenseTerm::OwnedColumns { range, .. }
            | PreparedDenseTerm::PSpline { range, .. }
            | PreparedDenseTerm::CyclicPSpline { range, .. }
            | PreparedDenseTerm::Fourier { range, .. }
            | PreparedDenseTerm::TensorPSpline { range, .. } => range.end,
        })
        .max()
        .unwrap_or(0)
}

fn checked_range(start: usize, width: usize) -> Result<Range<usize>, FormulaError> {
    let end = start
        .checked_add(width)
        .ok_or(ModelError::ArithmeticOverflow {
            context: "formula term coefficient range",
        })?;
    Ok(start..end)
}

fn dense_from_prepared_terms(
    nrows: usize,
    ncols: usize,
    terms: &[PreparedDenseTerm<'_>],
) -> Result<DenseDesign, FormulaError> {
    let mut builder = RowMajorDesignBuilder::new(nrows, ncols)?;

    for term in terms {
        match term {
            PreparedDenseTerm::Intercept { range } => builder.fill_intercept(range),
            PreparedDenseTerm::Numeric { values, range } => {
                builder.fill_column(range, values.as_slice());
            }
            PreparedDenseTerm::OwnedColumns { values, range } => {
                if range.len() == 1 {
                    builder.fill_column(range, values);
                } else {
                    fill_flat_columns(&mut builder, range, values);
                }
            }
            PreparedDenseTerm::PSpline {
                values,
                basis,
                range,
            } => builder.fill_pspline(range, basis, values.as_slice())?,
            PreparedDenseTerm::CyclicPSpline {
                values,
                spec,
                range,
            } => {
                let design = spec.design(values.as_slice())?;
                builder.fill_row_basis(range, &design);
            }
            PreparedDenseTerm::Fourier { design, range } => builder.fill_row_basis(range, design),
            PreparedDenseTerm::TensorPSpline {
                left,
                right,
                left_basis,
                right_basis,
                range,
            } => {
                let left_design = left_basis.design(left.as_slice())?;
                let right_design = right_basis.design(right.as_slice())?;
                let design = TensorSplineDesign::new(left_design, right_design)?;
                builder.fill_row_basis(range, &design);
            }
        }
    }

    builder.finish()
}

fn fill_flat_columns(builder: &mut RowMajorDesignBuilder, range: &Range<usize>, values: &[f64]) {
    debug_assert_eq!(values.len(), builder.nrows * range.len());
    for row in 0..builder.nrows {
        for local_col in 0..range.len() {
            builder.set(
                row,
                range.start + local_col,
                values[row * range.len() + local_col],
            );
        }
    }
}

fn factor_columns(
    name: &str,
    values: &[String],
    levels: &[String],
) -> Result<Vec<f64>, FormulaError> {
    let width = levels.len().saturating_sub(1);
    let mut columns = vec![0.0; values.len() * width];
    for (row, value) in values.iter().enumerate() {
        let Some(level_index) = levels.iter().position(|level| level == value) else {
            return Err(FormulaError::UnknownCategoryLevel {
                name: name.to_owned(),
                level: value.clone(),
                row,
            });
        };
        if level_index > 0 {
            columns[row * width + level_index - 1] = 1.0;
        }
    }
    Ok(columns)
}

fn indexed_names(parameter: &str, column: &str, term: &str, width: usize) -> Vec<String> {
    (0..width)
        .map(|index| format!("{parameter}.{column}:{term}[{index}]"))
        .collect()
}

fn fourier_names(
    parameter: &str,
    column: &str,
    order: usize,
    include_intercept: bool,
) -> Vec<String> {
    let mut names = Vec::new();
    if include_intercept {
        names.push(format!("{parameter}.{column}:fourier[intercept]"));
    }
    for harmonic in 1..=order {
        names.push(format!("{parameter}.{column}:fourier[sin{harmonic}]"));
        names.push(format!("{parameter}.{column}:fourier[cos{harmonic}]"));
    }
    names
}
