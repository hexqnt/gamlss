//! Shared helpers for power transforms with fitted lambda parameters.

use crate::transforms::TransformError;

const INV_PHI: f64 = 0.618_033_988_749_894_9;
const LAMBDA_SEARCH_LOWER: f64 = -5.0;
const LAMBDA_SEARCH_UPPER: f64 = 5.0;
const LAMBDA_SEARCH_RADIUS: f64 = 2.0;
const LAMBDA_SEARCH_ITERATIONS: usize = 80;
const LAMBDA_SEARCH_GRID: [f64; 7] = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct ProfileAccumulator {
    count: f64,
    mean: f64,
    sum_squares: f64,
}

impl ProfileAccumulator {
    pub(super) fn push(&mut self, value: f64) -> Option<()> {
        if !value.is_finite() {
            return None;
        }

        self.count += 1.0;
        let delta = value - self.mean;
        self.mean += delta / self.count;
        self.sum_squares = delta.mul_add(value - self.mean, self.sum_squares);
        Some(())
    }

    pub(super) fn log_likelihood(self) -> Option<f64> {
        if self.count < 2.0 {
            return None;
        }

        let variance = self.sum_squares / self.count;
        if variance.is_finite() && variance > 0.0 {
            Some(-0.5 * self.count * variance.ln())
        } else {
            None
        }
    }
}
pub(super) fn fit_lambda(
    y: &[f64],
    objective: impl Fn(&[f64], f64) -> Option<f64>,
) -> Result<f64, TransformError> {
    let mut best_lambda = 0.0;
    let mut best_score =
        objective(y, best_lambda).ok_or(TransformError::InvalidParameter { name: "lambda" })?;

    for candidate in LAMBDA_SEARCH_GRID {
        if let Some(score) = objective(y, candidate)
            && score > best_score
        {
            best_lambda = candidate;
            best_score = score;
        }
    }

    let mut left = (best_lambda - LAMBDA_SEARCH_RADIUS).max(LAMBDA_SEARCH_LOWER);
    let mut right = (best_lambda + LAMBDA_SEARCH_RADIUS).min(LAMBDA_SEARCH_UPPER);
    if (left - right).abs() <= f64::EPSILON {
        return Ok(best_lambda);
    }

    let mut c = INV_PHI.mul_add(-(right - left), right);
    let mut d = INV_PHI.mul_add(right - left, left);
    let mut c_score = objective(y, c).unwrap_or(f64::NEG_INFINITY);
    let mut d_score = objective(y, d).unwrap_or(f64::NEG_INFINITY);

    for _ in 0..LAMBDA_SEARCH_ITERATIONS {
        if c_score < d_score {
            left = c;
            c = d;
            c_score = d_score;
            d = INV_PHI.mul_add(right - left, left);
            d_score = objective(y, d).unwrap_or(f64::NEG_INFINITY);
        } else {
            right = d;
            d = c;
            d_score = c_score;
            c = INV_PHI.mul_add(-(right - left), right);
            c_score = objective(y, c).unwrap_or(f64::NEG_INFINITY);
        }
    }

    let lambda = f64::midpoint(left, right);
    if lambda.is_finite() {
        Ok(lambda)
    } else {
        Err(TransformError::InvalidParameter { name: "lambda" })
    }
}
