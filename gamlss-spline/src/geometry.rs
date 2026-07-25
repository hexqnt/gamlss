use gamlss_core::{ModelError, RowMultiplier};

use crate::SplineRowBasis;

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

#[inline]
pub fn add_row_basis_weighted_gram<B>(
    basis: &B,
    row_weights: &[f64],
    out: &mut [f64],
) -> Result<(), ModelError>
where
    B: SplineRowBasis + ?Sized,
{
    validate_gram_lengths(basis.nrows(), basis.nparams(), row_weights, out)?;
    for (row, weight) in row_weights.iter().copied().enumerate() {
        if weight != 0.0 {
            add_row_basis_scaled_outer(basis, row, weight, out);
        }
    }
    Ok(())
}

#[inline]
pub fn add_row_basis_weighted_gram_by<B, M>(
    basis: &B,
    row_weights: &[f64],
    multiplier: &M,
    out: &mut [f64],
) -> Result<(), ModelError>
where
    B: SplineRowBasis + ?Sized,
    M: RowMultiplier + ?Sized,
{
    validate_gram_lengths(basis.nrows(), basis.nparams(), row_weights, out)?;
    for (row, weight) in row_weights.iter().copied().enumerate() {
        if weight == 0.0 {
            continue;
        }
        let scaled_weight = weight * multiplier.multiplier_at(row);
        if scaled_weight != 0.0 {
            add_row_basis_scaled_outer(basis, row, scaled_weight, out);
        }
    }
    Ok(())
}

#[inline]
pub fn add_row_basis_t_mul_vec<B>(
    basis: &B,
    row_scores: &[f64],
    out: &mut [f64],
) -> Result<(), ModelError>
where
    B: SplineRowBasis + ?Sized,
{
    validate_transpose_lengths(basis.nrows(), basis.nparams(), row_scores, out)?;
    for (row, score) in row_scores.iter().copied().enumerate() {
        if score != 0.0 {
            basis.for_each_row_basis(row, |index, weight| {
                out[index] = score.mul_add(weight, out[index]);
            });
        }
    }
    Ok(())
}

#[inline]
pub fn add_row_basis_t_mul_vec_by<B, M>(
    basis: &B,
    row_scores: &[f64],
    multiplier: &M,
    out: &mut [f64],
) -> Result<(), ModelError>
where
    B: SplineRowBasis + ?Sized,
    M: RowMultiplier + ?Sized,
{
    validate_transpose_lengths(basis.nrows(), basis.nparams(), row_scores, out)?;
    for (row, score) in row_scores.iter().copied().enumerate() {
        if score == 0.0 {
            continue;
        }
        let scaled_score = score * multiplier.multiplier_at(row);
        if scaled_score != 0.0 {
            basis.for_each_row_basis(row, |index, weight| {
                out[index] = scaled_score.mul_add(weight, out[index]);
            });
        }
    }
    Ok(())
}

#[inline]
fn add_row_basis_scaled_outer<B>(basis: &B, row: usize, scale: f64, out: &mut [f64])
where
    B: SplineRowBasis + ?Sized,
{
    let nparams = basis.nparams();
    basis.for_each_row_basis(row, |left_index, left_weight| {
        let scaled_left = scale * left_weight;
        basis.for_each_row_basis(row, |right_index, right_weight| {
            let index = left_index * nparams + right_index;
            out[index] = scaled_left.mul_add(right_weight, out[index]);
        });
    });
}
