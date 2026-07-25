//! Structured continuous synthetic datasets.
//!
//! The data-generating process combines trend, seasonality,
//! heteroskedasticity, changing asymmetry, moderately heavy tails, and
//! cross-component dependence. Its structure is fixed; callers choose only the
//! number of observations, response dimension, and seed.

#![allow(
    clippy::cast_precision_loss,
    clippy::suboptimal_flops,
    reason = "the fixed DGP converts row/component indices and spells out fixed formulas"
)]

use std::{error::Error, fmt};

use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};

use crate::Dataset;

const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653;
const AR_CORRELATION: f64 = 0.4;
// sqrt(1 - AR_CORRELATION^2), kept fixed with the DGP definition.
const AR_INNOVATION_SCALE: f64 = 0.916_515_138_991_168;
const INV_SQRT_2_PI: f64 = 0.398_942_280_401_432_7;

/// Errors returned by the structured continuous generators.
#[derive(Clone, Copy, Debug)]
pub enum GenerationError {
    /// A dataset was requested with no observations.
    EmptyDataset,
    /// A multivariate dataset was requested with `D == 0`.
    ZeroResponseDimension,
    /// One coordinate of a generated response is non-finite.
    NonFiniteResponse {
        /// Observation index.
        index: usize,
        /// Response coordinate index; zero for a scalar response.
        component: usize,
        /// Non-finite value produced by the DGP.
        value: f64,
    },
}

impl Error for GenerationError {}

impl fmt::Display for GenerationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDataset => write!(formatter, "dataset must contain at least one row"),
            Self::ZeroResponseDimension => {
                write!(formatter, "response dimension must be positive")
            }
            Self::NonFiniteResponse {
                index,
                component,
                value,
            } => write!(
                formatter,
                "response at index {index}, component {component} is not finite: {value}"
            ),
        }
    }
}

/// Generates a structured scalar-response continuous dataset.
///
/// The covariate is an evenly spaced coordinate on `[0, 1]`. Responses contain
/// the same fixed structural effects as [`multivariate`]. For a given `n` and
/// `seed`, this produces the scalar projection of `multivariate::<1>`.
///
/// Reproducibility is guaranteed for the same crate versions and target.
///
/// # Errors
///
/// Returns [`GenerationError::EmptyDataset`] when `n == 0`, or
/// [`GenerationError::NonFiniteResponse`] if response arithmetic overflows.
pub fn univariate(n: usize, seed: u64) -> Result<Dataset<Vec<f64>, Vec<f64>>, GenerationError> {
    let data = generate::<1>(n, seed)?;
    let (x, y) = data.into_parts();
    Ok(Dataset::new(
        x,
        y.into_iter().map(|[value]| value).collect(),
    ))
}

/// Generates a structured `D`-dimensional continuous dataset.
///
/// Each component has a deterministic phase-shifted trend, two seasonal
/// harmonics, changing scale, and changing left/right spread. A shared smooth
/// scale mixture gives moderately heavy tails. Latent innovations follow a
/// fixed AR(1) dependence structure with adjacent correlation `0.4`.
///
/// Callers choose the number of observations, the const-generic response
/// dimension, and the random seed. The structural characteristics of the
/// generated data are fixed.
///
/// Reproducibility is guaranteed for the same crate versions and target.
///
/// # Errors
///
/// Returns [`GenerationError::EmptyDataset`] when `n == 0`,
/// [`GenerationError::ZeroResponseDimension`] when `D == 0`, or
/// [`GenerationError::NonFiniteResponse`] if response arithmetic overflows.
pub fn multivariate<const D: usize>(
    n: usize,
    seed: u64,
) -> Result<Dataset<Vec<f64>, Vec<[f64; D]>>, GenerationError> {
    generate::<D>(n, seed)
}

fn generate<const D: usize>(
    n: usize,
    seed: u64,
) -> Result<Dataset<Vec<f64>, Vec<[f64; D]>>, GenerationError> {
    if n == 0 {
        return Err(GenerationError::EmptyDataset);
    }
    if D == 0 {
        return Err(GenerationError::ZeroResponseDimension);
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);

    for index in 0..n {
        let x_value = unit_coordinate(index, n);
        let response = response_row::<D, _>(index, x_value, &mut rng)?;
        x.push(x_value);
        y.push(response);
    }

    Ok(Dataset::new(x, y))
}

fn unit_coordinate(index: usize, n: usize) -> f64 {
    if n == 1 {
        0.5
    } else {
        index as f64 / (n - 1) as f64
    }
}

fn response_row<const D: usize, R>(
    index: usize,
    x: f64,
    rng: &mut R,
) -> Result<[f64; D], GenerationError>
where
    R: rand::Rng + ?Sized,
{
    let standard_normal = StandardNormal;
    let tail_driver: f64 = standard_normal.sample(rng);
    let tail_scale = 0.85 + 0.15 * tail_driver * tail_driver;
    let mut previous_latent = 0.0;
    let mut response = [0.0; D];

    for (component, value) in response.iter_mut().enumerate() {
        let innovation: f64 = standard_normal.sample(rng);
        let latent = if component == 0 {
            innovation
        } else {
            AR_CORRELATION.mul_add(previous_latent, AR_INNOVATION_SCALE * innovation)
        };
        previous_latent = latent;

        let location = location(x, component);
        let scale = scale(x, component);
        let asymmetry = asymmetry(x, component);
        let left_scale = scale * (-asymmetry).exp();
        let right_scale = scale * asymmetry.exp();
        let heavy_tailed_latent = tail_scale * latent;
        let split_scale = if heavy_tailed_latent.is_sign_negative() {
            left_scale
        } else {
            right_scale
        };

        // E[tail_scale] = 1, so this correction keeps `location` equal to the
        // conditional mean despite the different left and right scales.
        let mean_correction = INV_SQRT_2_PI * (right_scale - left_scale);
        *value = split_scale.mul_add(heavy_tailed_latent, location - mean_correction);
        if !value.is_finite() {
            return Err(GenerationError::NonFiniteResponse {
                index,
                component,
                value: *value,
            });
        }
    }

    Ok(response)
}

fn component_phase(component: usize) -> f64 {
    component as f64 * GOLDEN_ANGLE
}

fn location(x: f64, component: usize) -> f64 {
    let phase = component_phase(component);
    let centered = 2.0 * x - 1.0;
    let trend = (0.65 + 0.15 * phase.cos()) * centered;
    let curved_trend = 0.22 * phase.sin() * (centered * centered - 1.0 / 3.0);
    let primary_season = 0.55 * (std::f64::consts::TAU * x + phase).sin();
    let secondary_season = 0.18 * (2.0 * std::f64::consts::TAU * x - 0.5 * phase).cos();
    0.3 * phase.sin() + trend + curved_trend + primary_season + secondary_season
}

fn scale(x: f64, component: usize) -> f64 {
    let phase = component_phase(component);
    let centered = 2.0 * x - 1.0;
    let log_scale = -0.5
        + 0.22 * phase.cos() * centered
        + 0.28 * (std::f64::consts::TAU * x + 0.7 * phase).sin()
        + 0.1 * (2.0 * std::f64::consts::TAU * x - phase).cos();
    log_scale.exp()
}

fn asymmetry(x: f64, component: usize) -> f64 {
    let phase = component_phase(component);
    let centered = 2.0 * x - 1.0;
    0.34 * (std::f64::consts::TAU * x + 0.5 * phase).sin() + 0.16 * centered * (phase + 0.3).cos()
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "the fixed seeded DGP and exact grid endpoints are deterministic invariants"
)]
mod tests {
    use super::{GenerationError, asymmetry, location, multivariate, scale, univariate};

    #[test]
    fn univariate_generation_is_reproducible_and_aligned() {
        let left = univariate(257, 42).unwrap();
        let right = univariate(257, 42).unwrap();

        assert_eq!(left, right);
        assert_eq!(left.x.len(), 257);
        assert_eq!(left.x.len(), left.y.len());
        assert_eq!(left.x[0], 0.0);
        assert_eq!(left.x[256], 1.0);
        assert!(left.y.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn multivariate_generation_has_requested_shape() {
        let data = multivariate::<8>(129, 7).unwrap();

        assert_eq!(data.x.len(), 129);
        assert_eq!(data.y.len(), 129);
        assert!(data.y.iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    fn scalar_and_one_dimensional_generators_match() {
        let scalar = univariate(64, 123).unwrap();
        let vector = multivariate::<1>(64, 123).unwrap();

        assert_eq!(scalar.x, vector.x);
        assert_eq!(
            scalar.y,
            vector
                .y
                .into_iter()
                .map(|[value]| value)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn seed_changes_responses_but_not_covariates() {
        let left = multivariate::<4>(32, 1).unwrap();
        let right = multivariate::<4>(32, 2).unwrap();

        assert_eq!(left.x, right.x);
        assert_ne!(left.y, right.y);
    }

    #[test]
    fn singleton_uses_domain_midpoint() {
        let data = univariate(1, 42).unwrap();
        assert_eq!(data.x, [0.5]);
    }

    #[test]
    fn empty_and_zero_dimensional_requests_are_rejected() {
        assert!(matches!(
            univariate(0, 42),
            Err(GenerationError::EmptyDataset)
        ));
        assert!(matches!(
            multivariate::<0>(1, 42),
            Err(GenerationError::ZeroResponseDimension)
        ));
    }

    #[test]
    fn fixed_structure_varies_over_covariate_and_component() {
        let x_values = [0.0, 0.25, 0.5, 0.75, 1.0];
        let locations = x_values.map(|x| location(x, 0));
        let scales = x_values.map(|x| scale(x, 0));
        let asymmetries = x_values.map(|x| asymmetry(x, 0));

        assert!(locations.windows(2).any(|pair| pair[0] != pair[1]));
        assert!(scales.windows(2).any(|pair| pair[0] != pair[1]));
        assert!(asymmetries.iter().any(|value| *value < 0.0));
        assert!(asymmetries.iter().any(|value| *value > 0.0));
        assert_ne!(location(0.25, 0), location(0.25, 1));
        assert_ne!(scale(0.25, 0), scale(0.25, 1));
    }
}
