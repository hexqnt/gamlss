//! Seedable synthetic data generators.
//!
//! The normal helpers accept parameter functions directly, while the
//! [`continuous`] module provides a structured family-independent
//! data-generating process.

use std::{error::Error, fmt};

use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

/// Structured continuous synthetic datasets.
pub mod continuous;

/// Errors returned when a generator cannot produce a finite observation.
#[derive(Clone, Copy, Debug)]
pub enum GenerationError {
    /// A covariate is non-finite.
    NonFiniteCovariate { index: usize, value: f64 },
    /// A location function evaluated to a non-finite value.
    NonFiniteLocation { index: usize, value: f64 },
    /// A scale function evaluated to a non-finite or non-positive value.
    InvalidScale { index: usize, value: f64 },
    /// Sampling overflowed or otherwise produced a non-finite response.
    NonFiniteResponse { index: usize, value: f64 },
}

impl Error for GenerationError {}

impl fmt::Display for GenerationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteCovariate { index, value } => {
                write!(
                    formatter,
                    "covariate at index {index} is not finite: {value}"
                )
            }
            Self::NonFiniteLocation { index, value } => {
                write!(
                    formatter,
                    "location at index {index} is not finite: {value}"
                )
            }
            Self::InvalidScale { index, value } => {
                write!(
                    formatter,
                    "scale at index {index} must be finite and positive: {value}"
                )
            }
            Self::NonFiniteResponse { index, value } => {
                write!(
                    formatter,
                    "response at index {index} is not finite: {value}"
                )
            }
        }
    }
}

/// Samples one normal response per covariate.
///
/// For each `x`, this samples `Y ~ Normal(mu(x), sigma(x))`. Passing a seeded
/// RNG makes generation reproducible for a fixed version of the `rand`
/// ecosystem.
///
/// # Errors
///
/// Returns an error when a covariate or location is non-finite, a scale is not
/// finite and strictly positive, or sampling produces a non-finite response.
pub fn normal<R, Mu, Sigma>(
    x: &[f64],
    mu: Mu,
    sigma: Sigma,
    rng: &mut R,
) -> Result<Vec<f64>, GenerationError>
where
    R: Rng + ?Sized,
    Mu: Fn(f64) -> f64,
    Sigma: Fn(f64) -> f64,
{
    let mut y = Vec::with_capacity(x.len());
    for (index, &x_value) in x.iter().enumerate() {
        if !x_value.is_finite() {
            return Err(GenerationError::NonFiniteCovariate {
                index,
                value: x_value,
            });
        }
        let location = mu(x_value);
        if !location.is_finite() {
            return Err(GenerationError::NonFiniteLocation {
                index,
                value: location,
            });
        }
        let scale = sigma(x_value);
        if !scale.is_finite() || scale <= 0.0 {
            return Err(GenerationError::InvalidScale {
                index,
                value: scale,
            });
        }
        let standard_normal: f64 = StandardNormal.sample(rng);
        let response = scale.mul_add(standard_normal, location);
        if !response.is_finite() {
            return Err(GenerationError::NonFiniteResponse {
                index,
                value: response,
            });
        }
        y.push(response);
    }
    Ok(y)
}

/// Samples `Y ~ Normal(intercept + slope * x, sigma)` for every covariate.
///
/// # Errors
///
/// Returns the same validation errors as [`normal`].
pub fn normal_linear<R>(
    x: &[f64],
    intercept: f64,
    slope: f64,
    sigma: f64,
    rng: &mut R,
) -> Result<Vec<f64>, GenerationError>
where
    R: Rng + ?Sized,
{
    normal(
        x,
        |x_value| slope.mul_add(x_value, intercept),
        |_| sigma,
        rng,
    )
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::StdRng};

    use super::{GenerationError, normal, normal_linear};

    #[test]
    fn normal_generation_is_reproducible_for_a_seed() {
        let x = [-1.0, 0.0, 1.0];
        let mut left_rng = StdRng::seed_from_u64(42);
        let mut right_rng = StdRng::seed_from_u64(42);

        let left = normal(&x, |x| 0.5_f64.mul_add(x, 1.0), |_| 0.3, &mut left_rng).unwrap();
        let right = normal_linear(&x, 1.0, 0.5, 0.3, &mut right_rng).unwrap();

        assert_eq!(left, right);
        assert_eq!(left.len(), x.len());
    }

    #[test]
    fn normal_accepts_a_changing_scale() {
        let mut rng = StdRng::seed_from_u64(7);

        let y = normal(
            &[0.0, 1.0],
            |x| x,
            |x| 0.2_f64.mul_add(x, -1.0).exp(),
            &mut rng,
        )
        .unwrap();

        assert_eq!(y.len(), 2);
        assert!(y.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn invalid_scale_is_reported() {
        let mut rng = StdRng::seed_from_u64(7);

        assert!(matches!(
            normal(&[0.0], |_| 0.0, |_| 0.0, &mut rng),
            Err(GenerationError::InvalidScale {
                index: 0,
                value: 0.0,
            })
        ));
    }
}
