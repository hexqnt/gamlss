#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::needless_range_loop,
    clippy::suboptimal_flops
)]

use std::marker::PhantomData;

use gamlss_core::{
    CholeskyScale, CompilableFamily, Family, FixedDimensionalFamily, HasObservationDimension,
    Identity, InitialEtaFromTheta, Link, Log, LogLocation, ModelError, ObservationView,
    PositiveLink,
    shape::{Product, ShapeValues, Simplex, StrictLower, strict_lower_triangular_packed_len},
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};
use gamlss_special::baseline_softmax;

use crate::{
    domain::is_interior_simplex,
    multivariate::{elliptical::LowerTriangularMatrix, normal::kernel},
};

/// Default-link additive-log-ratio logistic-normal family.
pub type LogisticNormalAlrCholeskyDefault<const K: usize> =
    LogisticNormalAlrCholesky<K, Identity, Log, Identity>;

/// Logistic-normal distribution on a `K`-component simplex using additive log ratios.
///
/// The final simplex component is the structural baseline. For `i < K - 1`,
/// `z_i = log(y_i / y_{K-1})` follows a multivariate normal distribution with
/// location `mu` and Cholesky factor `L`. The density with respect to the first
/// `K - 1` simplex coordinates includes the Jacobian `1 / product(y_i)`.
///
/// The fixed-size predictor carriers retain one structural location slot and a
/// shifted strict-lower Cholesky shape. This represents exactly `K - 1`
/// locations and `K * (K - 1) / 2` Cholesky entries without nightly generic
/// const expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogisticNormalAlrCholesky<
    const K: usize,
    LocationLink = Identity,
    DiagonalLink = Log,
    OffDiagonalLink = Identity,
> {
    marker: PhantomData<(LocationLink, DiagonalLink, OffDiagonalLink)>,
}

impl<const K: usize, LocationLink, DiagonalLink, OffDiagonalLink>
    LogisticNormalAlrCholesky<K, LocationLink, DiagonalLink, OffDiagonalLink>
where
    LocationLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    /// Creates a stateless family after validating the simplex dimension.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `K < 2`.
    pub const fn try_new() -> Result<Self, ModelError> {
        if K < 2 {
            return Err(ModelError::InvalidParameter {
                parameter: "logistic-normal component count",
                expected: "at least two",
            });
        }
        Ok(Self {
            marker: PhantomData,
        })
    }

    /// Creates a stateless family.
    ///
    /// # Panics
    ///
    /// Panics when `K < 2`.
    #[must_use]
    pub const fn new() -> Self {
        assert!(K >= 2, "logistic-normal requires at least two components");
        Self {
            marker: PhantomData,
        }
    }

    fn theta_from_eta(eta: &LogisticNormalAlrCholeskyEta<K>) -> LogisticNormalAlrCholeskyTheta<K> {
        let mut location = [0.0; K];
        for component in 0..K - 1 {
            location[component] = LocationLink::inverse(eta.log_ratio_location[component]);
        }
        let mut cholesky = FixedLogRatioCholesky::zeros();
        for row in 0..K - 1 {
            for col in 0..=row {
                let eta_value = eta.cholesky.lower(row, col);
                let value = if row == col {
                    DiagonalLink::inverse(eta_value)
                } else {
                    OffDiagonalLink::inverse(eta_value)
                };
                cholesky
                    .set_lower(row, col, value)
                    .expect("valid log-ratio Cholesky index");
            }
        }
        LogisticNormalAlrCholeskyTheta::from_parts_unchecked(location, cholesky)
    }

    fn nan_eta() -> LogisticNormalAlrCholeskyEta<K> {
        let mut cholesky = FixedLogRatioCholesky::zeros();
        for row in 0..K.saturating_sub(1) {
            for col in 0..=row {
                cholesky
                    .set_lower(row, col, f64::NAN)
                    .expect("valid log-ratio Cholesky index");
            }
        }
        LogisticNormalAlrCholeskyEta {
            log_ratio_location: [f64::NAN; K],
            cholesky,
        }
    }

    fn nll_theta(observation: [f64; K], theta: &LogisticNormalAlrCholeskyTheta<K>) -> f64 {
        let Some((log_ratios, log_jacobian)) = log_ratio_observation(observation) else {
            return f64::INFINITY;
        };
        if !valid_theta(theta) {
            return f64::INFINITY;
        }
        let dimension = K - 1;
        let mut z = [0.0; K];
        kernel::nll(
            dimension,
            &log_ratios[..dimension],
            &theta.log_ratio_location[..dimension],
            &theta.cholesky,
            &mut z[..dimension],
        ) + log_jacobian
    }

    fn nll_and_gradient_eta_values(
        observation: [f64; K],
        eta: &LogisticNormalAlrCholeskyEta<K>,
    ) -> (f64, LogisticNormalAlrCholeskyEta<K>) {
        let Some((log_ratios, log_jacobian)) = log_ratio_observation(observation) else {
            return (f64::INFINITY, Self::nan_eta());
        };
        let theta = Self::theta_from_eta(eta);
        if !valid_theta(&theta) {
            return (f64::INFINITY, Self::nan_eta());
        }
        let dimension = K - 1;
        let mut z = [0.0; K];
        let mut a = [0.0; K];
        let normal_nll = kernel::nll_and_score(
            dimension,
            &log_ratios[..dimension],
            &theta.log_ratio_location[..dimension],
            &theta.cholesky,
            &mut z[..dimension],
            &mut a[..dimension],
        );
        let nll = normal_nll + log_jacobian;
        if !nll.is_finite() {
            return (nll, Self::nan_eta());
        }

        let mut gradient =
            LogisticNormalAlrCholeskyEta::new([0.0; K], FixedLogRatioCholesky::zeros());
        for component in 0..dimension {
            gradient.log_ratio_location[component] =
                -a[component] * LocationLink::derivative_inverse(eta.log_ratio_location[component]);
        }
        for row in 0..dimension {
            for col in 0..=row {
                let eta_value = eta.cholesky.lower(row, col);
                let score = if row == col {
                    DiagonalLink::derivative_log_inverse(eta_value)
                        - a[row] * z[col] * DiagonalLink::derivative_inverse(eta_value)
                } else {
                    kernel::cholesky_score(row, col, z[col], &a, &theta.cholesky)
                        * OffDiagonalLink::derivative_inverse(eta_value)
                };
                gradient
                    .cholesky
                    .set_lower(row, col, score)
                    .expect("valid log-ratio Cholesky index");
            }
        }
        (nll, gradient)
    }
}

impl<const K: usize, LocationLink, DiagonalLink, OffDiagonalLink> Default
    for LogisticNormalAlrCholesky<K, LocationLink, DiagonalLink, OffDiagonalLink>
where
    LocationLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const K: usize, LocationLink, DiagonalLink, OffDiagonalLink> Family
    for LogisticNormalAlrCholesky<K, LocationLink, DiagonalLink, OffDiagonalLink>
where
    LocationLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    type Observation<'obs> = [f64; K];
    type Eta = LogisticNormalAlrCholeskyEta<K>;
    type Theta = LogisticNormalAlrCholeskyTheta<K>;
    type GradientEta = LogisticNormalAlrCholeskyEta<K>;
    type Workspace = ();

    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut ()) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    fn nll(
        &self,
        observation: Self::Observation<'_>,
        theta: &Self::Theta,
        _workspace: &mut (),
    ) -> f64 {
        Self::nll_theta(observation, theta)
    }

    fn nll_and_gradient_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        _workspace: &mut (),
    ) -> (f64, Self::GradientEta) {
        Self::nll_and_gradient_eta_values(observation, eta)
    }
}

impl<const K: usize, LocationLink, DiagonalLink, OffDiagonalLink> FixedDimensionalFamily<K>
    for LogisticNormalAlrCholesky<K, LocationLink, DiagonalLink, OffDiagonalLink>
where
    LocationLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
}

impl<const K: usize, LocationLink, DiagonalLink, OffDiagonalLink> HasObservationDimension
    for LogisticNormalAlrCholesky<K, LocationLink, DiagonalLink, OffDiagonalLink>
where
    LocationLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    fn observation_dimension(&self) -> usize {
        K
    }
}

#[cfg(feature = "rand")]
impl<Rng, const K: usize, LocationLink, DiagonalLink, OffDiagonalLink> TrySimulate<Rng>
    for LogisticNormalAlrCholesky<K, LocationLink, DiagonalLink, OffDiagonalLink>
where
    Rng: rand::Rng,
    LocationLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    type Sample = [f64; K];

    fn try_sample(
        &self,
        rng: &mut Rng,
        theta: &Self::Theta,
    ) -> Result<Self::Sample, SimulationError> {
        if !valid_theta(theta) {
            return Err(SimulationError::InvalidParameters(
                "logistic-normal ALR theta",
            ));
        }
        let dimension = K - 1;
        let normal = rand_distr::StandardNormal;
        let mut standard = [0.0; K];
        for value in standard.iter_mut().take(dimension) {
            *value = rand_distr::Distribution::sample(&normal, rng);
        }
        let mut logits = theta.log_ratio_location;
        for row in 0..dimension {
            for col in 0..=row {
                logits[row] += theta.cholesky.lower(row, col) * standard[col];
            }
        }
        logits[K - 1] = 0.0;
        let sample = baseline_softmax(logits);
        if is_interior_simplex(&sample) {
            Ok(sample)
        } else {
            Err(SimulationError::NumericalFailure(
                "logistic-normal softmax transform",
            ))
        }
    }
}

impl<const K: usize, LocationLink, DiagonalLink, OffDiagonalLink> CompilableFamily
    for LogisticNormalAlrCholesky<K, LocationLink, DiagonalLink, OffDiagonalLink>
where
    LocationLink: InitialEtaFromTheta<f64>,
    DiagonalLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    OffDiagonalLink: InitialEtaFromTheta<f64>,
{
    type Shape = Product<Simplex<LogLocation, K>, StrictLower<CholeskyScale, K>>;

    fn eta_from_shape(values: ShapeValues<Self::Shape>) -> Self::Eta {
        LogisticNormalAlrCholeskyEta::new(
            values.0,
            FixedLogRatioCholesky::from_shifted_strict_lower(values.1),
        )
    }

    fn gradient_to_shape(gradient: &Self::GradientEta) -> ShapeValues<Self::Shape> {
        let mut shifted = [[0.0; K]; K];
        for row in 0..K - 1 {
            for col in 0..=row {
                shifted[row + 1][col] = gradient.cholesky.lower(row, col);
            }
        }
        (gradient.log_ratio_location, shifted)
    }

    fn initial_shape<'obs, Obs>(&self, obs: &'obs Obs) -> ShapeValues<Self::Shape>
    where
        Obs: ObservationView<'obs, Observation = [f64; K]> + 'obs,
    {
        let dimension = K - 1;
        let mut weight_sum = 0.0;
        let mut location = [0.0; K];
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            let observation = obs.observation_at(row);
            let Some((log_ratios, _)) = log_ratio_observation(observation) else {
                continue;
            };
            if weight > 0.0 {
                weight_sum += weight;
                for component in 0..dimension {
                    location[component] += weight * log_ratios[component];
                }
            }
        }
        if weight_sum > 0.0 {
            location
                .iter_mut()
                .take(dimension)
                .for_each(|value| *value /= weight_sum);
        }

        let mut variance = [0.0; K];
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            let observation = obs.observation_at(row);
            let Some((log_ratios, _)) = log_ratio_observation(observation) else {
                continue;
            };
            if weight > 0.0 {
                for component in 0..dimension {
                    variance[component] +=
                        weight * (log_ratios[component] - location[component]).powi(2);
                }
            }
        }
        let mut shifted = [[OffDiagonalLink::initial_eta_from_theta(0.0); K]; K];
        for row in 0..dimension {
            let scale = if weight_sum > 0.0 {
                (variance[row] / weight_sum).sqrt().max(1.0e-6)
            } else {
                1.0
            };
            shifted[row + 1][row] = DiagonalLink::initial_eta_from_theta(scale);
        }
        location[K - 1] = 0.0;
        (location.map(LocationLink::initial_eta_from_theta), shifted)
    }

    fn validate_compiled(&self) -> Result<(), ModelError> {
        Self::try_new().map(|_| ())
    }
}

/// Fixed storage for the `(K - 1) x (K - 1)` ALR Cholesky factor.
///
/// The backing array remains `K x K` for stable Rust const generics; only the
/// first `K - 1` lower-triangular rows are meaningful.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedLogRatioCholesky<const K: usize> {
    values: [[f64; K]; K],
}

impl<const K: usize> FixedLogRatioCholesky<K> {
    /// Creates a zero-valued carrier.
    #[must_use]
    pub const fn zeros() -> Self {
        Self {
            values: [[0.0; K]; K],
        }
    }

    /// Creates a carrier from packed lower-triangular ALR entries.
    ///
    /// Packed order is `(0,0), (1,0), (1,1), ...` in the latent `K - 1`
    /// dimensional normal geometry.
    pub fn try_from_packed(values: &[f64]) -> Result<Self, ModelError> {
        let expected =
            strict_lower_triangular_packed_len(K).ok_or(ModelError::ArithmeticOverflow {
                context: "log-ratio Cholesky storage length",
            })?;
        if K < 2 || values.len() != expected {
            return Err(ModelError::InvalidParameter {
                parameter: "log-ratio cholesky",
                expected: "K * (K - 1) / 2 packed values with K >= 2",
            });
        }
        let mut out = Self::zeros();
        let mut values = values.iter().copied();
        for row in 0..K - 1 {
            for col in 0..=row {
                let Some(value) = values.next() else {
                    return Err(ModelError::InvalidParameter {
                        parameter: "log-ratio cholesky",
                        expected: "K * (K - 1) / 2 packed values with K >= 2",
                    });
                };
                out.values[row][col] = value;
            }
        }
        Ok(out)
    }

    fn from_shifted_strict_lower(values: [[f64; K]; K]) -> Self {
        let mut out = Self::zeros();
        for row in 0..K.saturating_sub(1) {
            for col in 0..=row {
                out.values[row][col] = values[row + 1][col];
            }
        }
        out
    }

    /// Returns a latent lower-triangular entry.
    #[must_use]
    pub const fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row < K.saturating_sub(1) && col <= row {
            Some(self.values[row][col])
        } else {
            None
        }
    }

    /// Sets a latent lower-triangular entry.
    pub const fn set_lower(
        &mut self,
        row: usize,
        col: usize,
        value: f64,
    ) -> Result<(), ModelError> {
        if row < K.saturating_sub(1) && col <= row {
            self.values[row][col] = value;
            Ok(())
        } else {
            Err(ModelError::InvalidParameter {
                parameter: "log-ratio Cholesky index",
                expected: "lower-triangular entry within K - 1 latent dimensions",
            })
        }
    }

    #[inline]
    const fn lower(&self, row: usize, col: usize) -> f64 {
        self.values[row][col]
    }

    /// Returns the `K x K` backing rows; the final row and upper triangle are structural zeros.
    #[must_use]
    pub const fn as_full_rows(&self) -> &[[f64; K]; K] {
        &self.values
    }
}

impl<const K: usize> LowerTriangularMatrix for FixedLogRatioCholesky<K> {
    fn dimension(&self) -> usize {
        K.saturating_sub(1)
    }

    fn lower(&self, row: usize, col: usize) -> f64 {
        self.lower(row, col)
    }
}

/// Link-scale ALR location and Cholesky predictors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogisticNormalAlrCholeskyEta<const K: usize> {
    log_ratio_location: [f64; K],
    cholesky: FixedLogRatioCholesky<K>,
}

impl<const K: usize> LogisticNormalAlrCholeskyEta<K> {
    /// Creates predictors and normalizes the final structural location to zero.
    #[must_use]
    pub const fn new(mut log_ratio_location: [f64; K], cholesky: FixedLogRatioCholesky<K>) -> Self {
        if K > 0 {
            log_ratio_location[K - 1] = 0.0;
        }
        Self {
            log_ratio_location,
            cholesky,
        }
    }

    /// ALR location predictors followed by the zero baseline slot.
    #[must_use]
    pub const fn log_ratio_location(&self) -> &[f64; K] {
        &self.log_ratio_location
    }

    /// ALR Cholesky predictors.
    #[must_use]
    pub const fn cholesky(&self) -> &FixedLogRatioCholesky<K> {
        &self.cholesky
    }

    /// Mutably borrows ALR location predictors.
    #[must_use]
    pub const fn log_ratio_location_mut(&mut self) -> &mut [f64; K] {
        &mut self.log_ratio_location
    }

    /// Mutably borrows ALR Cholesky predictors.
    #[must_use]
    pub const fn cholesky_mut(&mut self) -> &mut FixedLogRatioCholesky<K> {
        &mut self.cholesky
    }
}

/// Natural-scale ALR normal parameters of the logistic-normal distribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogisticNormalAlrCholeskyTheta<const K: usize> {
    log_ratio_location: [f64; K],
    cholesky: FixedLogRatioCholesky<K>,
}

impl<const K: usize> LogisticNormalAlrCholeskyTheta<K> {
    /// Creates checked parameters and normalizes the final baseline location to zero.
    pub fn try_new(
        mut log_ratio_location: [f64; K],
        cholesky: FixedLogRatioCholesky<K>,
    ) -> Result<Self, ModelError> {
        if K > 0 {
            log_ratio_location[K - 1] = 0.0;
        }
        let theta = Self {
            log_ratio_location,
            cholesky,
        };
        if valid_theta(&theta) {
            Ok(theta)
        } else {
            Err(ModelError::InvalidParameter {
                parameter: "logistic-normal ALR theta",
                expected: "K >= 2, finite ALR locations and a finite Cholesky factor with positive diagonal",
            })
        }
    }

    const fn from_parts_unchecked(
        log_ratio_location: [f64; K],
        cholesky: FixedLogRatioCholesky<K>,
    ) -> Self {
        Self {
            log_ratio_location,
            cholesky,
        }
    }

    /// ALR normal location followed by the zero baseline slot.
    #[must_use]
    pub const fn log_ratio_location(&self) -> &[f64; K] {
        &self.log_ratio_location
    }

    /// ALR normal Cholesky factor.
    #[must_use]
    pub const fn cholesky(&self) -> &FixedLogRatioCholesky<K> {
        &self.cholesky
    }

    /// Returns the softmax of the ALR location, often called the compositional center.
    #[must_use]
    pub fn compositional_center(&self) -> [f64; K] {
        baseline_softmax(self.log_ratio_location)
    }

    /// Returns one covariance entry of the latent ALR normal distribution.
    #[must_use]
    pub fn log_ratio_covariance(&self, row: usize, col: usize) -> Option<f64> {
        let dimension = K.saturating_sub(1);
        if row >= dimension || col >= dimension {
            return None;
        }
        Some(
            (0..=row.min(col))
                .map(|index| self.cholesky.lower(row, index) * self.cholesky.lower(col, index))
                .sum(),
        )
    }
}

fn log_ratio_observation<const K: usize>(observation: [f64; K]) -> Option<([f64; K], f64)> {
    if !is_interior_simplex(&observation) {
        return None;
    }
    let log_values = observation.map(f64::ln);
    let baseline = log_values[K - 1];
    let mut log_ratios = [0.0; K];
    for component in 0..K - 1 {
        log_ratios[component] = log_values[component] - baseline;
    }
    Some((log_ratios, log_values.iter().sum()))
}

fn valid_theta<const K: usize>(theta: &LogisticNormalAlrCholeskyTheta<K>) -> bool {
    let dimension = K.saturating_sub(1);
    K >= 2
        && theta.log_ratio_location[K - 1] == 0.0
        && kernel::valid_theta(
            dimension,
            &theta.log_ratio_location[..dimension],
            &theta.cholesky,
        )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use approx::assert_relative_eq;
    use gamlss_core::{
        CholeskyScale, DenseDesign, Family, Gamlss, LinearPredictorBlock, LogLocation, NoPenalty,
        ParameterBlocks, SimplexLogitParameterBlock, StrictLowerTriangularParameterBlock,
    };

    use super::{
        FixedLogRatioCholesky, LogisticNormalAlrCholeskyDefault, LogisticNormalAlrCholeskyEta,
        LogisticNormalAlrCholeskyTheta,
    };
    use crate::constants::HALF_LOG_2_PI;

    #[test]
    fn two_component_density_matches_logit_normal_change_of_variables() {
        let family = LogisticNormalAlrCholeskyDefault::<2>::new();
        let theta = LogisticNormalAlrCholeskyTheta::try_new(
            [0.2, 9.0],
            FixedLogRatioCholesky::try_from_packed(&[0.8]).unwrap(),
        )
        .unwrap();
        let y = [0.3_f64, 0.7];
        let z = (y[0] / y[1]).ln();
        let expected =
            HALF_LOG_2_PI + 0.8_f64.ln() + 0.5 * ((z - 0.2) / 0.8).powi(2) + y[0].ln() + y[1].ln();
        assert_relative_eq!(family.nll(y, &theta, &mut ()), expected, epsilon = 1.0e-14);
        assert_eq!(theta.log_ratio_location()[1], 0.0);
    }

    #[test]
    fn multicomponent_density_matches_alr_normal_plus_jacobian() {
        use crate::multivariate::{
            matrix::FixedLowerTriangular,
            normal::{MvNormalCholeskyDefault, MvNormalCholeskyTheta},
        };

        let family = LogisticNormalAlrCholeskyDefault::<3>::new();
        let theta = LogisticNormalAlrCholeskyTheta::try_new(
            [0.2, -0.3, 0.0],
            FixedLogRatioCholesky::try_from_packed(&[0.8, 0.25, 0.7]).unwrap(),
        )
        .unwrap();
        let normal = MvNormalCholeskyDefault::<2>::new();
        let normal_theta = MvNormalCholeskyTheta::try_new(
            [0.2, -0.3],
            FixedLowerTriangular::from_lower_rows([[0.8, 0.0], [0.25, 0.7]]),
        )
        .unwrap();
        let y = [0.2_f64, 0.3, 0.5];
        let log_ratios = [(y[0] / y[2]).ln(), (y[1] / y[2]).ln()];
        let expected = normal.nll(log_ratios, &normal_theta, &mut ())
            + y.iter().copied().map(f64::ln).sum::<f64>();
        assert_relative_eq!(family.nll(y, &theta, &mut ()), expected, epsilon = 1.0e-14);
    }

    #[test]
    fn alr_transform_preserves_normal_parameter_gradient() {
        use crate::multivariate::{
            matrix::FixedLowerTriangular,
            normal::{MvNormalCholeskyDefault, MvNormalCholeskyEta},
        };

        let family = LogisticNormalAlrCholeskyDefault::<4>::new();
        let normal = MvNormalCholeskyDefault::<3>::new();
        let eta = LogisticNormalAlrCholeskyEta::new(
            [0.2, -0.3, 0.4, 0.0],
            FixedLogRatioCholesky::try_from_packed(&[-0.1, 0.25, 0.2, -0.15, 0.3, -0.25]).unwrap(),
        );
        let normal_eta = MvNormalCholeskyEta::new(
            [0.2, -0.3, 0.4],
            FixedLowerTriangular::from_lower_rows([
                [-0.1, 0.0, 0.0],
                [0.25, 0.2, 0.0],
                [-0.15, 0.3, -0.25],
            ]),
        );
        let observation = [0.2_f64, 0.3, 0.1, 0.4];
        let log_ratios = [
            (observation[0] / observation[3]).ln(),
            (observation[1] / observation[3]).ln(),
            (observation[2] / observation[3]).ln(),
        ];
        let (nll, gradient) = family.nll_and_gradient_eta(observation, &eta, &mut ());
        let (normal_nll, normal_gradient) =
            normal.nll_and_gradient_eta(log_ratios, &normal_eta, &mut ());

        assert_relative_eq!(
            nll,
            normal_nll + observation.iter().copied().map(f64::ln).sum::<f64>(),
            epsilon = 1.0e-14
        );
        for component in 0..3 {
            assert_relative_eq!(
                gradient.log_ratio_location()[component],
                normal_gradient.mu()[component],
                epsilon = 1.0e-14
            );
        }
        for row in 0..3 {
            for col in 0..=row {
                assert_relative_eq!(
                    gradient.cholesky().get(row, col).unwrap(),
                    normal_gradient.cholesky().get(row, col).unwrap(),
                    epsilon = 1.0e-14
                );
            }
        }
        assert_eq!(gradient.log_ratio_location()[3], 0.0);
    }

    #[test]
    fn analytic_gradient_matches_finite_difference() {
        let family = LogisticNormalAlrCholeskyDefault::<3>::new();
        let eta = LogisticNormalAlrCholeskyEta::new(
            [0.2, -0.3, 8.0],
            FixedLogRatioCholesky::try_from_packed(&[-0.1, 0.25, 0.2]).unwrap(),
        );
        let y = [0.2, 0.3, 0.5];
        let (_, gradient) = family.nll_and_gradient_eta(y, &eta, &mut ());
        for component in 0..2 {
            let mut lower = eta;
            let mut upper = eta;
            lower.log_ratio_location_mut()[component] -= 1.0e-6;
            upper.log_ratio_location_mut()[component] += 1.0e-6;
            let numeric =
                (family.nll_eta(y, &upper, &mut ()) - family.nll_eta(y, &lower, &mut ())) / 2.0e-6;
            assert_relative_eq!(
                gradient.log_ratio_location()[component],
                numeric,
                epsilon = 1.0e-6
            );
        }
        for row in 0..2 {
            for col in 0..=row {
                let mut lower = eta;
                let mut upper = eta;
                let value = eta.cholesky().get(row, col).unwrap();
                lower
                    .cholesky_mut()
                    .set_lower(row, col, value - 1.0e-6)
                    .unwrap();
                upper
                    .cholesky_mut()
                    .set_lower(row, col, value + 1.0e-6)
                    .unwrap();
                let numeric = (family.nll_eta(y, &upper, &mut ())
                    - family.nll_eta(y, &lower, &mut ()))
                    / 2.0e-6;
                assert_relative_eq!(
                    gradient.cholesky().get(row, col).unwrap(),
                    numeric,
                    epsilon = 1.0e-6
                );
            }
        }
        assert_eq!(gradient.log_ratio_location()[2], 0.0);
    }

    #[test]
    fn carrier_and_domains_enforce_structural_geometry() {
        let cholesky = FixedLogRatioCholesky::<3>::try_from_packed(&[0.8, 0.2, 0.7]).unwrap();
        assert_eq!(cholesky.get(0, 0), Some(0.8));
        assert_eq!(cholesky.get(1, 1), Some(0.7));
        assert_eq!(cholesky.get(2, 0), None);
        assert!(FixedLogRatioCholesky::<3>::try_from_packed(&[0.8]).is_err());
        assert!(LogisticNormalAlrCholeskyDefault::<1>::try_new().is_err());

        let family = LogisticNormalAlrCholeskyDefault::<3>::new();
        let theta = LogisticNormalAlrCholeskyTheta::try_new([0.2, -0.3, 0.0], cholesky).unwrap();
        assert!(family.nll([0.2, 0.3, 0.6], &theta, &mut ()).is_infinite());
        assert!(family.nll([0.0, 0.5, 0.5], &theta, &mut ()).is_infinite());
        assert_relative_eq!(theta.compositional_center().iter().sum::<f64>(), 1.0);
    }

    #[test]
    fn compiled_shape_has_only_free_alr_coordinates() {
        let y = [[0.2, 0.3, 0.5], [0.4, 0.2, 0.4], [0.1, 0.7, 0.2]];
        let n = y.len();
        let location = SimplexLogitParameterBlock::<LogLocation, 3, _, _>::new(
            vec![
                LinearPredictorBlock::new(DenseDesign::intercept(n)),
                LinearPredictorBlock::new(DenseDesign::intercept(n)),
            ],
            NoPenalty,
            99,
        );
        let cholesky = StrictLowerTriangularParameterBlock::<CholeskyScale, 3, _, _>::new(
            vec![
                LinearPredictorBlock::new(DenseDesign::intercept(n)),
                LinearPredictorBlock::new(DenseDesign::intercept(n)),
                LinearPredictorBlock::new(DenseDesign::intercept(n)),
            ],
            NoPenalty,
            99,
        );
        let model = Gamlss::try_new_with_observations(
            LogisticNormalAlrCholeskyDefault::<3>::new(),
            ParameterBlocks::new((location, cholesky)),
            y.as_slice(),
        )
        .unwrap();
        let beta = [0.2, -0.1, -0.2, 0.15, 0.1];
        let mut gradient = [0.0; 5];
        model.try_value_gradient_into(&beta, &mut gradient).unwrap();
        assert_eq!(model.nparams(), 5);
        for index in 0..5 {
            let mut lower = beta;
            let mut upper = beta;
            lower[index] -= 1.0e-6;
            upper[index] += 1.0e-6;
            let numeric =
                (model.try_value(&upper).unwrap() - model.try_value(&lower).unwrap()) / 2.0e-6;
            assert_relative_eq!(gradient[index], numeric, epsilon = 1.0e-6);
        }
    }

    #[cfg(feature = "rand")]
    #[test]
    fn sampling_returns_an_interior_simplex() {
        use gamlss_core::TrySimulate;
        use rand::SeedableRng;

        let family = LogisticNormalAlrCholeskyDefault::<3>::new();
        let theta = LogisticNormalAlrCholeskyTheta::try_new(
            [0.2, -0.3, 0.0],
            FixedLogRatioCholesky::try_from_packed(&[0.8, 0.2, 0.7]).unwrap(),
        )
        .unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let sample = family.try_sample(&mut rng, &theta).unwrap();
        assert!(sample.iter().all(|value| value.is_finite() && *value > 0.0));
        assert_relative_eq!(sample.iter().sum::<f64>(), 1.0);
    }
}
