//! Shared mathematical constants used by distribution kernels.

/// Natural logarithm of 2.
pub const LOG_2: f64 = std::f64::consts::LN_2;

/// Euler-Mascheroni constant.
pub const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;

/// Half of the natural logarithm of `2 * pi`.
pub const HALF_LOG_2_PI: f64 = 0.918_938_533_204_672_7;

/// Reciprocal square root of `2 * pi`.
pub const INV_SQRT_2_PI: f64 = 0.398_942_280_401_432_7;

/// Reciprocal square root of `pi`.
pub const INV_SQRT_PI: f64 = 0.564_189_583_547_756_3;

/// Square root of `2 / pi`.
pub const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
