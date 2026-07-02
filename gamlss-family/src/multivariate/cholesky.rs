//! Cholesky indexing helpers shared by multivariate families.

/// Number of entries in a packed lower-triangular `dimension x dimension` matrix.
#[must_use]
pub const fn packed_len(dimension: usize) -> Option<usize> {
    match dimension.checked_add(1) {
        Some(next) => match dimension.checked_mul(next) {
            Some(product) => Some(product / 2),
            None => None,
        },
        None => None,
    }
}

/// Row-major packed lower-triangular index for `(row, col)`.
#[must_use]
pub const fn packed_index(row: usize, col: usize) -> Option<usize> {
    if col <= row {
        match row.checked_add(1) {
            Some(next) => match row.checked_mul(next) {
                Some(product) => match (product / 2).checked_add(col) {
                    Some(index) => Some(index),
                    None => None,
                },
                None => None,
            },
            None => None,
        }
    } else {
        None
    }
}
