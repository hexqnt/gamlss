#[inline]
pub fn pow_usize(mut base: f64, mut exponent: usize) -> f64 {
    let mut value = 1.0;
    while exponent > 0 {
        if exponent & 1 == 1 {
            value *= base;
        }
        exponent >>= 1;
        if exponent > 0 {
            base *= base;
        }
    }
    value
}

#[inline]
pub fn dot(left: &[f64], right: &[f64]) -> f64 {
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .copied()
        .zip(right.iter().copied())
        .fold(0.0, |value, (left, right)| left.mul_add(right, value))
}

#[inline]
pub fn squared_norm(values: &[f64]) -> f64 {
    dot(values, values)
}
