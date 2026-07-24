use gamlss_core::ModelError;

#[inline]
pub fn validate_gram_lengths(
    nrows: usize,
    nparams: usize,
    row_weights: &[f64],
    out: &[f64],
) -> Result<(), ModelError> {
    validate_row_values_len(nrows, row_weights)?;
    let expected_values = nparams
        .checked_mul(nparams)
        .ok_or(ModelError::ArithmeticOverflow {
            context: "spline Gram value count",
        })?;
    if out.len() != expected_values {
        return Err(ModelError::DesignSize {
            expected_values,
            actual_values: out.len(),
        });
    }
    Ok(())
}

#[inline]
pub fn validate_transpose_lengths(
    nrows: usize,
    nparams: usize,
    row_scores: &[f64],
    out: &[f64],
) -> Result<(), ModelError> {
    validate_row_values_len(nrows, row_scores)?;
    if out.len() != nparams {
        return Err(ModelError::GradientLength {
            expected: nparams,
            actual: out.len(),
        });
    }
    Ok(())
}

#[inline]
const fn validate_row_values_len(nrows: usize, row_values: &[f64]) -> Result<(), ModelError> {
    if row_values.len() != nrows {
        return Err(ModelError::WeightLength {
            expected: nrows,
            actual: row_values.len(),
        });
    }
    Ok(())
}
