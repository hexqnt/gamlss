#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::suboptimal_flops
)]

use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    CompilableFamily, Family, FixedDimensionalFamily, HasConditionalCdf, HasMarginalCdf,
    HasObservationDimension, HasRosenblattTransform, Identity, InitialEtaFromTheta, Link, Log,
    ModelError, Mu, ObservationView, PartialCorrelation, PositiveLink, Sigma,
    shape::{Product, ShapeValues, StrictLower, Vector, strict_lower_triangular_packed_len},
};
use gamlss_special::unit_normal_cdf;

use crate::multivariate::matrix::FixedLowerTriangular;

use super::kernel;

/// Default-link `D R D` multivariate normal parameterization.
pub type MvNormalMeanStdPartialCorrDefault<const D: usize> =
    MvNormalMeanStdPartialCorr<D, Identity, Log>;

/// Fixed-dimensional partial-correlation predictors backed by a square array.
///
/// Construction and iteration use strict-lower row-major order `(1,0), (2,0),
/// (2,1), (3,0), ...`. Values are kept on the predictor scale by
/// [`MvNormalMeanStdPartialCorrEta`]; the family maps them with `tanh` before
/// building the correlation Cholesky factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedPartialCorrelations<const D: usize> {
    values: [[f64; D]; D],
}

impl<const D: usize> FixedPartialCorrelations<D> {
    /// Creates a carrier from strict-lower values after validating their length.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `values.len()` is not
    /// `D * (D - 1) / 2`.
    pub fn try_new(values: Vec<f64>) -> Result<Self, ModelError> {
        let expected = Self::checked_len().ok_or(ModelError::ArithmeticOverflow {
            context: "strict-lower partial-correlation storage length",
        })?;
        if values.len() != expected {
            return Err(ModelError::InvalidParameter {
                parameter: "partial_corr",
                expected: "D * (D - 1) / 2 strict-lower values",
            });
        }
        let mut out = [[0.0; D]; D];
        let mut values = values.into_iter();
        for (row, row_values) in out.iter_mut().enumerate().skip(1) {
            for value in row_values.iter_mut().take(row) {
                let Some(packed_value) = values.next() else {
                    return Err(ModelError::InvalidParameter {
                        parameter: "partial_corr",
                        expected: "D * (D - 1) / 2 strict-lower values",
                    });
                };
                *value = packed_value;
            }
        }
        Ok(Self { values: out })
    }

    /// Creates a zero-valued carrier.
    #[must_use]
    pub const fn zeros() -> Self {
        Self {
            values: [[0.0; D]; D],
        }
    }

    /// Checked number of strict-lower values for dimension `D`.
    #[must_use]
    pub const fn checked_len() -> Option<usize> {
        strict_lower_triangular_packed_len(D)
    }

    /// Returns true when there are no strict-lower entries.
    #[must_use]
    pub const fn is_empty() -> bool {
        D < 2
    }

    /// Visits values in row-major strict-lower order.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = f64> + '_ {
        self.values
            .iter()
            .enumerate()
            .skip(1)
            .flat_map(|(row, values)| values[..row].iter().copied())
    }

    /// Returns a strict-lower entry, or `None` for invalid/diagonal/upper indices.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        (row < D && col < row).then(|| self.values[row][col])
    }

    /// Returns a mutable strict-lower entry, or `None` for invalid/diagonal/upper indices.
    #[must_use]
    pub const fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut f64> {
        if row < D && col < row {
            Some(&mut self.values[row][col])
        } else {
            None
        }
    }

    #[inline]
    const fn lower(&self, row: usize, col: usize) -> f64 {
        self.values[row][col]
    }
}

/// Generic multivariate normal parameterized as `Sigma = D R D`.
///
/// Means and marginal standard deviations are represented explicitly. The
/// correlation matrix is built from strict-lower partial-correlation predictors
/// mapped through `tanh`, then converted to a correlation Cholesky factor. For
/// each row, the partial correlations condition on the preceding response
/// coordinates, so their interpretation depends on response-coordinate order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MvNormalMeanStdPartialCorr<const D: usize, MuLink = Identity, SigmaLink = Log> {
    marker: PhantomData<(MuLink, SigmaLink)>,
}

impl<const D: usize, MuLink, SigmaLink> MvNormalMeanStdPartialCorr<D, MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    /// Creates a stateless family value after checking the compile-time dimension.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `D == 0`.
    #[inline]
    pub const fn try_new() -> Result<Self, ModelError> {
        if D == 0 {
            return Err(ModelError::InvalidParameter {
                parameter: "dimension",
                expected: "positive",
            });
        }
        Ok(Self {
            marker: PhantomData,
        })
    }

    /// Creates a stateless family value.
    ///
    /// # Panics
    ///
    /// Panics when the compile-time dimension is zero.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        assert!(D > 0, "multivariate normal dimension must be positive");
        Self {
            marker: PhantomData,
        }
    }

    fn theta_from_eta(
        eta: &MvNormalMeanStdPartialCorrEta<D>,
    ) -> MvNormalMeanStdPartialCorrTheta<D> {
        let mut mu = [0.0; D];
        let mut sigma = [0.0; D];
        for component in 0..D {
            mu[component] = MuLink::inverse(eta.mu[component]);
            sigma[component] = SigmaLink::inverse(eta.sigma[component]);
        }

        let partial_corr = partial_corr_from_eta(&eta.partial_corr);
        let correlation_cholesky = correlation_cholesky_from_partial(&partial_corr);
        let scale_cholesky = scale_cholesky_from_correlation(&sigma, &correlation_cholesky);

        MvNormalMeanStdPartialCorrTheta::from_canonical_parts(
            mu,
            sigma,
            partial_corr,
            correlation_cholesky,
            scale_cholesky,
        )
    }

    fn nan_eta() -> MvNormalMeanStdPartialCorrEta<D> {
        nan_eta()
    }

    fn nll_theta(observation: [f64; D], theta: &MvNormalMeanStdPartialCorrTheta<D>) -> f64 {
        let mut z = [0.0; D];
        kernel::nll(D, &observation, &theta.mu, &theta.scale_cholesky, &mut z)
    }

    fn nll_and_gradient_eta_values(
        observation: [f64; D],
        eta: &MvNormalMeanStdPartialCorrEta<D>,
    ) -> (f64, MvNormalMeanStdPartialCorrEta<D>) {
        let theta = Self::theta_from_eta(eta);
        Self::generic_nll_and_gradient_eta_values(observation, eta, &theta)
    }

    fn generic_nll_and_gradient_eta_values(
        observation: [f64; D],
        eta: &MvNormalMeanStdPartialCorrEta<D>,
        theta: &MvNormalMeanStdPartialCorrTheta<D>,
    ) -> (f64, MvNormalMeanStdPartialCorrEta<D>) {
        let mut z = [0.0; D];
        let mut a = [0.0; D];
        let nll = kernel::nll_and_score(
            D,
            &observation,
            &theta.mu,
            &theta.scale_cholesky,
            &mut z,
            &mut a,
        );
        if !nll.is_finite() {
            return (nll, Self::nan_eta());
        }

        let gradient = gradient_from_cholesky_score::<D, MuLink, SigmaLink>(eta, theta, &z, &a);
        (nll, gradient)
    }
}

impl<const D: usize, MuLink, SigmaLink> Default for MvNormalMeanStdPartialCorr<D, MuLink, SigmaLink>
where
    MuLink: InitialEtaFromTheta<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize, MuLink, SigmaLink> Family for MvNormalMeanStdPartialCorr<D, MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    type Eta = MvNormalMeanStdPartialCorrEta<D>;
    type Theta = MvNormalMeanStdPartialCorrTheta<D>;
    type GradientEta = MvNormalMeanStdPartialCorrEta<D>;
    type Observation<'obs> = [f64; D];
    type Workspace = ();

    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline]
    fn nll(
        &self,
        observation: Self::Observation<'_>,
        theta: &Self::Theta,
        _workspace: &mut Self::Workspace,
    ) -> f64 {
        Self::nll_theta(observation, theta)
    }

    #[inline]
    fn nll_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> f64 {
        Self::nll_theta(observation, &Self::theta_from_eta(eta))
    }

    #[inline]
    fn nll_and_gradient_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        Self::nll_and_gradient_eta_values(observation, eta)
    }
}

impl<const D: usize, MuLink, SigmaLink> FixedDimensionalFamily<D>
    for MvNormalMeanStdPartialCorr<D, MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
}

impl<const D: usize, MuLink, SigmaLink> HasObservationDimension
    for MvNormalMeanStdPartialCorr<D, MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn observation_dimension(&self) -> usize {
        D
    }
}

impl<const D: usize, MuLink, SigmaLink> HasMarginalCdf
    for MvNormalMeanStdPartialCorr<D, MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn marginal_cdf(&self, component: usize, y: f64, theta: &Self::Theta) -> f64 {
        if !y.is_finite() || component >= D || !valid_theta(theta) {
            return f64::NAN;
        }
        unit_normal_cdf((y - theta.mu[component]) / theta.sigma[component])
    }
}

impl<const D: usize, MuLink, SigmaLink> HasConditionalCdf
    for MvNormalMeanStdPartialCorr<D, MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn conditional_cdf(
        &self,
        component: usize,
        y: f64,
        preceding: &[f64],
        theta: &Self::Theta,
    ) -> f64 {
        let mut standardized = [0.0; D];
        kernel::conditional_cdf(
            D,
            component,
            y,
            preceding,
            &theta.mu,
            &theta.scale_cholesky,
            &mut standardized,
        )
    }
}

impl<const D: usize, MuLink, SigmaLink> HasRosenblattTransform
    for MvNormalMeanStdPartialCorr<D, MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn rosenblatt_into(
        &self,
        observation: Self::Observation<'_>,
        theta: &Self::Theta,
        out: &mut [f64],
    ) -> Result<(), ModelError> {
        if out.len() != D {
            return Err(ModelError::ResponseLength {
                expected: D,
                actual: out.len(),
            });
        }
        kernel::rosenblatt_into(D, &observation, &theta.mu, &theta.scale_cholesky, out);
        Ok(())
    }
}

#[cfg(feature = "rand")]
impl<Rng, const D: usize, MuLink, SigmaLink> CanSimulate<Rng>
    for MvNormalMeanStdPartialCorr<D, MuLink, SigmaLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    type Sample = [f64; D];

    fn sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Self::Sample {
        if !valid_theta(theta) {
            return [f64::NAN; D];
        }
        let standard = rand_distr::StandardNormal;
        let mut z = [0.0; D];
        for value in &mut z {
            *value = rand_distr::Distribution::sample(&standard, rng);
        }
        let mut out = theta.mu;
        for row in 0..D {
            for col in 0..=row {
                out[row] += theta.scale_cholesky.lower(row, col) * z[col];
            }
        }
        out
    }
}

impl<const D: usize, MuLink, SigmaLink> CompilableFamily
    for MvNormalMeanStdPartialCorr<D, MuLink, SigmaLink>
where
    MuLink: InitialEtaFromTheta<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Shape =
        Product<Product<Vector<Mu, D>, Vector<Sigma, D>>, StrictLower<PartialCorrelation, D>>;

    fn eta_from_shape(values: ShapeValues<Self::Shape>) -> MvNormalMeanStdPartialCorrEta<D> {
        let mut partial_corr = FixedPartialCorrelations::zeros();
        for row in 1..D {
            for col in 0..row {
                *partial_corr
                    .get_mut(row, col)
                    .expect("valid strict-lower index") = values.1[row][col];
            }
        }
        MvNormalMeanStdPartialCorrEta::new(values.0.0, values.0.1, partial_corr)
    }

    fn gradient_to_shape(gradient: &MvNormalMeanStdPartialCorrEta<D>) -> ShapeValues<Self::Shape> {
        let partial_corr = std::array::from_fn(|row| {
            std::array::from_fn(|col| gradient.partial_corr.get(row, col).unwrap_or(0.0))
        });
        ((gradient.mu, gradient.sigma), partial_corr)
    }

    fn initial_shape<'obs, Obs>(&self, obs: &'obs Obs) -> ShapeValues<Self::Shape>
    where
        Obs: ObservationView<'obs, Observation = [f64; D]> + 'obs,
    {
        let mut weight_sum = 0.0;
        let mut means = [0.0; D];
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            let value = obs.observation_at(row);
            if weight > 0.0 && value.iter().all(|entry| entry.is_finite()) {
                weight_sum += weight;
                for component in 0..D {
                    means[component] += weight * value[component];
                }
            }
        }
        if weight_sum > 0.0 {
            for mean in &mut means {
                *mean /= weight_sum;
            }
        }

        let mut covariance = [[0.0; D]; D];
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            let value = obs.observation_at(row);
            if weight > 0.0 && value.iter().all(|entry| entry.is_finite()) {
                for i in 0..D {
                    for j in 0..=i {
                        covariance[i][j] += weight * (value[i] - means[i]) * (value[j] - means[j]);
                    }
                }
            }
        }
        if weight_sum > 0.0 {
            for i in 0..D {
                for j in 0..=i {
                    covariance[i][j] /= weight_sum;
                    covariance[j][i] = covariance[i][j];
                }
            }
        }
        let sigma = std::array::from_fn(|index| covariance[index][index].sqrt().max(1.0e-6));
        let mut correlation = [[0.0; D]; D];
        for i in 0..D {
            correlation[i][i] = 1.0;
            for j in 0..i {
                let empirical = covariance[i][j] / (sigma[i] * sigma[j]);
                correlation[i][j] = 0.8 * empirical.clamp(-0.99, 0.99);
                correlation[j][i] = correlation[i][j];
            }
        }
        let mut cholesky = [[0.0; D]; D];
        for row in 0..D {
            for col in 0..=row {
                let correction = (0..col)
                    .map(|k| cholesky[row][k] * cholesky[col][k])
                    .sum::<f64>();
                cholesky[row][col] = if row == col {
                    (correlation[row][row] - correction).max(1.0e-8).sqrt()
                } else {
                    (correlation[row][col] - correction) / cholesky[col][col]
                };
            }
        }
        let mut partial_eta = [[0.0; D]; D];
        for row in 1..D {
            let mut prefix = 1.0;
            for col in 0..row {
                let partial = (cholesky[row][col] / prefix).clamp(-0.95, 0.95);
                partial_eta[row][col] = partial.atanh();
                prefix *= (1.0 - partial * partial).sqrt();
            }
        }

        (
            (
                means.map(MuLink::initial_eta_from_theta),
                sigma.map(SigmaLink::initial_eta_from_theta),
            ),
            partial_eta,
        )
    }
}

/// Link-scale predictors for `D R D` multivariate normal.
#[derive(Debug, Clone, PartialEq)]
pub struct MvNormalMeanStdPartialCorrEta<const D: usize> {
    /// Mean predictors.
    pub mu: [f64; D],
    /// Marginal standard-deviation predictors.
    pub sigma: [f64; D],
    /// Strict-lower partial-correlation predictors.
    pub partial_corr: FixedPartialCorrelations<D>,
}

impl<const D: usize> MvNormalMeanStdPartialCorrEta<D> {
    /// Creates link-scale predictors.
    #[must_use]
    #[inline]
    pub const fn new(
        mu: [f64; D],
        sigma: [f64; D],
        partial_corr: FixedPartialCorrelations<D>,
    ) -> Self {
        Self {
            mu,
            sigma,
            partial_corr,
        }
    }
}

/// Natural-scale parameters for `D R D` multivariate normal.
#[derive(Debug, Clone, PartialEq)]
pub struct MvNormalMeanStdPartialCorrTheta<const D: usize> {
    /// Mean vector.
    mu: [f64; D],
    /// Positive marginal standard deviations.
    sigma: [f64; D],
    /// Strict-lower partial correlations on `(-1, 1)`.
    partial_corr: FixedPartialCorrelations<D>,
    /// Correlation Cholesky factor.
    correlation_cholesky: FixedLowerTriangular<D>,
    /// Scale Cholesky factor for `Sigma = D R D`.
    scale_cholesky: FixedLowerTriangular<D>,
}

impl<const D: usize> MvNormalMeanStdPartialCorrTheta<D> {
    /// Creates a valid natural-scale state from its canonical parameters.
    ///
    /// Derived correlation and scale Cholesky factors are built internally, so
    /// likelihood, marginal and simulation methods always observe the same
    /// covariance geometry.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `D == 0`, a canonical
    /// value is non-finite, a standard deviation is not positive, or a partial
    /// correlation is outside `(-1, 1)`.
    pub fn try_new(
        mu: [f64; D],
        sigma: [f64; D],
        partial_corr: FixedPartialCorrelations<D>,
    ) -> Result<Self, ModelError> {
        if D == 0 {
            return Err(ModelError::InvalidParameter {
                parameter: "dimension",
                expected: "at least one response component",
            });
        }
        if !mu.iter().all(|value| value.is_finite()) {
            return Err(ModelError::InvalidParameter {
                parameter: "mu",
                expected: "finite",
            });
        }
        if !sigma.iter().all(|value| value.is_finite() && *value > 0.0) {
            return Err(ModelError::InvalidParameter {
                parameter: "sigma",
                expected: "finite and > 0",
            });
        }
        if !partial_corr
            .iter()
            .all(|value| value.is_finite() && value.abs() < 1.0)
        {
            return Err(ModelError::InvalidParameter {
                parameter: "partial_corr",
                expected: "finite and strictly between -1 and 1",
            });
        }

        let correlation_cholesky = correlation_cholesky_from_partial(&partial_corr);
        let scale_cholesky = scale_cholesky_from_correlation(&sigma, &correlation_cholesky);
        let theta = Self::from_canonical_parts(
            mu,
            sigma,
            partial_corr,
            correlation_cholesky,
            scale_cholesky,
        );
        if !kernel::valid_theta(D, &theta.mu, &theta.scale_cholesky) {
            return Err(ModelError::InvalidParameter {
                parameter: "partial_corr",
                expected: "a representable positive-definite correlation geometry",
            });
        }
        Ok(theta)
    }

    const fn from_canonical_parts(
        mu: [f64; D],
        sigma: [f64; D],
        partial_corr: FixedPartialCorrelations<D>,
        correlation_cholesky: FixedLowerTriangular<D>,
        scale_cholesky: FixedLowerTriangular<D>,
    ) -> Self {
        Self {
            mu,
            sigma,
            partial_corr,
            correlation_cholesky,
            scale_cholesky,
        }
    }

    /// Mean vector.
    #[must_use]
    pub const fn mu(&self) -> &[f64; D] {
        &self.mu
    }

    /// Marginal standard deviations.
    #[must_use]
    pub const fn sigma(&self) -> &[f64; D] {
        &self.sigma
    }

    /// Canonical strict-lower partial correlations.
    #[must_use]
    pub const fn partial_corr(&self) -> &FixedPartialCorrelations<D> {
        &self.partial_corr
    }

    /// Derived correlation Cholesky factor.
    #[must_use]
    pub const fn correlation_cholesky(&self) -> &FixedLowerTriangular<D> {
        &self.correlation_cholesky
    }

    /// Derived scale Cholesky factor for `Sigma = D R D`.
    #[must_use]
    pub const fn scale_cholesky(&self) -> &FixedLowerTriangular<D> {
        &self.scale_cholesky
    }

    /// Returns one covariance entry from `D R D`.
    #[must_use]
    pub fn covariance(&self, row: usize, col: usize) -> Option<f64> {
        covariance_from_cholesky(&self.scale_cholesky, row, col)
    }
}

fn partial_corr_from_eta<const D: usize>(
    eta: &FixedPartialCorrelations<D>,
) -> FixedPartialCorrelations<D> {
    let mut out = FixedPartialCorrelations::zeros();
    for row in 1..D {
        for col in 0..row {
            *out.get_mut(row, col).expect("valid strict-lower index") =
                stable_partial_corr(eta.lower(row, col)).0;
        }
    }
    out
}

fn stable_partial_corr(eta: f64) -> (f64, f64, f64) {
    if !eta.is_finite() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let interior = f64::from_bits(1.0_f64.to_bits() - 1);
    let limit = interior.atanh();
    let transformed = eta.clamp(-limit, limit);
    let partial_corr = transformed.tanh();
    let one_minus_p2 = (1.0 - partial_corr) * (1.0 + partial_corr);
    let sech = one_minus_p2.sqrt();
    let derivative = if eta.abs() <= limit {
        one_minus_p2
    } else {
        0.0
    };
    (partial_corr, sech, derivative)
}

fn correlation_cholesky_from_partial<const D: usize>(
    partial_corr: &FixedPartialCorrelations<D>,
) -> FixedLowerTriangular<D> {
    let mut out = FixedLowerTriangular::zeros();
    for row in 0..D {
        let mut prefix = 1.0;
        for col in 0..row {
            let p = partial_corr.lower(row, col);
            out.set_lower(row, col, p * prefix)
                .expect("valid lower index");
            prefix *= ((1.0 - p) * (1.0 + p)).sqrt();
        }
        out.set_lower(row, row, prefix).expect("valid lower index");
    }
    out
}

fn scale_cholesky_from_correlation<const D: usize>(
    sigma: &[f64; D],
    correlation_cholesky: &FixedLowerTriangular<D>,
) -> FixedLowerTriangular<D> {
    let mut out = FixedLowerTriangular::zeros();
    for row in 0..D {
        for col in 0..=row {
            out.set_lower(row, col, sigma[row] * correlation_cholesky.lower(row, col))
                .expect("valid lower index");
        }
    }
    out
}

fn covariance_from_cholesky<const D: usize>(
    cholesky: &FixedLowerTriangular<D>,
    row: usize,
    col: usize,
) -> Option<f64> {
    if row >= D || col >= D {
        return None;
    }
    let limit = row.min(col);
    Some(
        (0..=limit)
            .map(|index| cholesky.lower(row, index) * cholesky.lower(col, index))
            .sum(),
    )
}

fn valid_theta<const D: usize>(theta: &MvNormalMeanStdPartialCorrTheta<D>) -> bool {
    D > 0
        && theta.mu.iter().all(|value| value.is_finite())
        && theta
            .sigma
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        && theta
            .partial_corr
            .iter()
            .all(|value| value.is_finite() && value.abs() < 1.0)
        && kernel::valid_theta(D, &theta.mu, &theta.scale_cholesky)
}

fn nan_eta<const D: usize>() -> MvNormalMeanStdPartialCorrEta<D> {
    MvNormalMeanStdPartialCorrEta {
        mu: [f64::NAN; D],
        sigma: [f64::NAN; D],
        partial_corr: FixedPartialCorrelations {
            values: [[f64::NAN; D]; D],
        },
    }
}

fn zero_eta<const D: usize>() -> MvNormalMeanStdPartialCorrEta<D> {
    MvNormalMeanStdPartialCorrEta {
        mu: [0.0; D],
        sigma: [0.0; D],
        partial_corr: FixedPartialCorrelations::zeros(),
    }
}

fn gradient_from_cholesky_score<const D: usize, MuLink, SigmaLink>(
    eta: &MvNormalMeanStdPartialCorrEta<D>,
    theta: &MvNormalMeanStdPartialCorrTheta<D>,
    z: &[f64; D],
    a: &[f64; D],
) -> MvNormalMeanStdPartialCorrEta<D>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    let mut gradient = zero_eta();

    let mut correlation_score = [[0.0; D]; D];
    for row in 0..D {
        for col in 0..=row {
            correlation_score[row][col] = -theta.sigma[row] * a[row] * z[col];
            if row == col {
                correlation_score[row][col] += 1.0 / theta.correlation_cholesky.lower(row, row);
            }
        }
        gradient.mu[row] = -a[row] * MuLink::derivative_inverse(eta.mu[row]);
        let residual_score = (0..=row)
            .map(|col| -a[row] * z[col] * theta.correlation_cholesky.lower(row, col))
            .sum::<f64>();
        gradient.sigma[row] = SigmaLink::derivative_log_inverse(eta.sigma[row])
            + residual_score * SigmaLink::derivative_inverse(eta.sigma[row]);
    }

    for row in 1..D {
        let mut prefixes = [1.0; D];
        let mut prefix = 1.0;
        for k in 0..row {
            prefixes[k] = prefix;
            prefix *= stable_partial_corr(eta.partial_corr.lower(row, k)).1;
        }
        let mut later_adjoint =
            correlation_score[row][row] * theta.correlation_cholesky.lower(row, row);
        for k in (0..row).rev() {
            let (partial_corr, _, derivative) = stable_partial_corr(eta.partial_corr.lower(row, k));
            let direct = correlation_score[row][k] * prefixes[k] * derivative;
            let log_sech_derivative = if derivative == 0.0 {
                0.0
            } else {
                -partial_corr
            };
            *gradient
                .partial_corr
                .get_mut(row, k)
                .expect("valid strict-lower index") = direct + log_sech_derivative * later_adjoint;
            later_adjoint += correlation_score[row][k] * theta.correlation_cholesky.lower(row, k);
        }
    }

    gradient
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{
        DenseDesign, Family, Gamlss, HasConditionalCdf, HasMarginalCdf, HasRosenblattTransform,
        LinearPredictorBlock, ModelError, Mu, NoPenalty, ParameterBlocks, PartialCorrelation,
        Sigma, StrictLowerTriangularParameterBlock, VectorParameterBlock,
    };

    use super::{
        FixedPartialCorrelations, MvNormalMeanStdPartialCorrDefault, MvNormalMeanStdPartialCorrEta,
    };
    use crate::multivariate::matrix::FixedLowerTriangular;
    use crate::multivariate::normal::{MvNormalCholeskyDefault, MvNormalCholeskyTheta};
    use crate::{NormalMuSigma, NormalTheta};

    fn assert_gradient_matches_finite_difference<const D: usize>(
        y: [f64; D],
        eta: &MvNormalMeanStdPartialCorrEta<D>,
    ) {
        let family = MvNormalMeanStdPartialCorrDefault::<D>::new();
        let (_, gradient) = family.nll_and_gradient_eta(y, eta, &mut family.workspace());

        for component in 0..D {
            let mut plus = eta.clone();
            plus.mu[component] += 1.0e-6;
            let mut minus = eta.clone();
            minus.mu[component] -= 1.0e-6;
            let fd = (family.nll_eta(y, &plus, &mut family.workspace())
                - family.nll_eta(y, &minus, &mut family.workspace()))
                / 2.0e-6;
            assert_relative_eq!(gradient.mu[component], fd, epsilon = 1.0e-6);

            let mut plus = eta.clone();
            plus.sigma[component] += 1.0e-6;
            let mut minus = eta.clone();
            minus.sigma[component] -= 1.0e-6;
            let fd = (family.nll_eta(y, &plus, &mut family.workspace())
                - family.nll_eta(y, &minus, &mut family.workspace()))
                / 2.0e-6;
            assert_relative_eq!(gradient.sigma[component], fd, epsilon = 1.0e-6);
        }

        for row in 1..D {
            for col in 0..row {
                let current = eta.partial_corr.get(row, col).unwrap();
                let mut plus = eta.clone();
                *plus.partial_corr.get_mut(row, col).unwrap() = current + 1.0e-6;
                let mut minus = eta.clone();
                *minus.partial_corr.get_mut(row, col).unwrap() = current - 1.0e-6;
                let fd = (family.nll_eta(y, &plus, &mut family.workspace())
                    - family.nll_eta(y, &minus, &mut family.workspace()))
                    / 2.0e-6;
                assert_relative_eq!(
                    gradient.partial_corr.get(row, col).unwrap(),
                    fd,
                    epsilon = 1.0e-6
                );
            }
        }
    }

    fn assert_capabilities_match_cholesky<const D: usize>() {
        let family = MvNormalMeanStdPartialCorrDefault::<D>::new();
        let mut partial_corr = FixedPartialCorrelations::zeros();
        for row in 1..D {
            for col in 0..row {
                *partial_corr.get_mut(row, col).unwrap() = 0.07 * (row + col + 1) as f64;
            }
        }
        let eta = MvNormalMeanStdPartialCorrEta::new(
            std::array::from_fn(|component| 0.2 * component as f64 - 0.3),
            std::array::from_fn(|component| (0.8 + 0.2 * component as f64).ln()),
            partial_corr,
        );
        let theta = family.theta(&eta, &mut family.workspace());
        let cholesky_family = MvNormalCholeskyDefault::<D>::new();
        let cholesky_theta =
            MvNormalCholeskyTheta::try_new(*theta.mu(), *theta.scale_cholesky()).unwrap();
        let observation = std::array::from_fn(|component| 0.4 - 0.15 * component as f64);

        for component in 0..D {
            assert_relative_eq!(
                family.marginal_cdf(component, observation[component], &theta),
                cholesky_family.marginal_cdf(component, observation[component], &cholesky_theta,),
                epsilon = 1.0e-12
            );
            assert_relative_eq!(
                family.conditional_cdf(component, observation[component], &observation, &theta,),
                cholesky_family.conditional_cdf(
                    component,
                    observation[component],
                    &observation,
                    &cholesky_theta,
                ),
                epsilon = 1.0e-12
            );
        }

        let mut actual = [0.0; D];
        let mut expected = [0.0; D];
        family
            .rosenblatt_into(observation, &theta, &mut actual)
            .unwrap();
        cholesky_family
            .rosenblatt_into(observation, &cholesky_theta, &mut expected)
            .unwrap();
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert_relative_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn fixed_partial_correlations_validate_length_and_indices() {
        assert_eq!(FixedPartialCorrelations::<4>::checked_len(), Some(6));
        assert!(FixedPartialCorrelations::<4>::try_new(vec![0.0; 5]).is_err());
        let mut correlations = FixedPartialCorrelations::<3>::try_new(vec![0.1, 0.2, 0.3]).unwrap();
        assert_eq!(correlations.get(1, 0), Some(0.1));
        assert_eq!(correlations.get(2, 1), Some(0.3));
        assert_eq!(correlations.get(0, 0), None);
        *correlations.get_mut(2, 0).unwrap() = -0.2;
        assert_eq!(correlations.get(2, 0), Some(-0.2));
    }

    #[test]
    fn checked_constructor_rejects_zero_dimension() {
        assert_eq!(
            MvNormalMeanStdPartialCorrDefault::<0>::try_new(),
            Err(ModelError::InvalidParameter {
                parameter: "dimension",
                expected: "positive",
            })
        );
        assert!(MvNormalMeanStdPartialCorrDefault::<1>::try_new().is_ok());
    }

    #[test]
    fn conditional_and_rosenblatt_capabilities_match_cholesky_for_d1_through_d4() {
        assert_capabilities_match_cholesky::<1>();
        assert_capabilities_match_cholesky::<2>();
        assert_capabilities_match_cholesky::<3>();
        assert_capabilities_match_cholesky::<4>();
    }

    #[test]
    fn natural_theta_is_built_from_one_canonical_geometry() {
        let partial_corr = FixedPartialCorrelations::try_new(vec![0.25]).unwrap();
        let theta = super::MvNormalMeanStdPartialCorrTheta::<2>::try_new(
            [0.4, -0.3],
            [0.8, 1.2],
            partial_corr,
        )
        .unwrap();

        assert_relative_eq!(theta.mu()[0], 0.4);
        assert_relative_eq!(theta.mu()[1], -0.3);
        assert_relative_eq!(theta.sigma()[0], 0.8);
        assert_relative_eq!(theta.sigma()[1], 1.2);
        assert_eq!(theta.partial_corr().get(1, 0), Some(0.25));
        assert_relative_eq!(
            theta.correlation_cholesky().get(1, 0).unwrap(),
            0.25,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            theta.scale_cholesky().get(1, 0).unwrap(),
            0.3,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(theta.covariance(1, 0).unwrap(), 0.24, epsilon = 1.0e-12);

        assert!(
            super::MvNormalMeanStdPartialCorrTheta::<0>::try_new(
                [],
                [],
                FixedPartialCorrelations::zeros(),
            )
            .is_err()
        );
        assert!(
            super::MvNormalMeanStdPartialCorrTheta::<2>::try_new(
                [0.0, 0.0],
                [1.0, 0.0],
                FixedPartialCorrelations::try_new(vec![0.0]).unwrap(),
            )
            .is_err()
        );
        assert!(
            super::MvNormalMeanStdPartialCorrTheta::<2>::try_new(
                [0.0, 0.0],
                [1.0, 1.0],
                FixedPartialCorrelations::try_new(vec![1.0]).unwrap(),
            )
            .is_err()
        );
    }

    #[test]
    fn d2_fast_path_matches_cholesky_form() {
        let drd = MvNormalMeanStdPartialCorrDefault::<2>::new();
        let rho_eta = 0.25_f64.atanh();
        let eta = MvNormalMeanStdPartialCorrEta::new(
            [0.4, -0.3],
            [0.8_f64.ln(), 1.2_f64.ln()],
            FixedPartialCorrelations::try_new(vec![rho_eta]).unwrap(),
        );
        let theta = drd.theta(&eta, &mut drd.workspace());
        let cholesky = MvNormalCholeskyDefault::<2>::new();
        let cholesky_theta = MvNormalCholeskyTheta::try_new(
            [0.4, -0.3],
            FixedLowerTriangular::from_lower_rows([
                [0.8, 0.0],
                [1.2 * 0.25, 1.2 * (1.0 - 0.25_f64.powi(2)).sqrt()],
            ]),
        )
        .unwrap();
        assert_relative_eq!(
            drd.nll([1.1, -0.7], &theta, &mut drd.workspace()),
            cholesky.nll([1.1, -0.7], &cholesky_theta, &mut cholesky.workspace()),
            epsilon = 1.0e-12
        );
        assert_eq!(theta.covariance(1, 0), Some(0.8 * 1.2 * 0.25));
    }

    #[test]
    fn d1_fast_path_matches_scalar_normal() {
        let drd = MvNormalMeanStdPartialCorrDefault::<1>::new();
        let eta = MvNormalMeanStdPartialCorrEta::new(
            [0.4],
            [0.8_f64.ln()],
            FixedPartialCorrelations::zeros(),
        );
        let theta = drd.theta(&eta, &mut drd.workspace());
        let normal = NormalMuSigma::new();
        let normal_theta = NormalTheta {
            mu: theta.mu[0],
            sigma: theta.sigma[0],
        };
        assert_relative_eq!(
            drd.nll([1.1], &theta, &mut drd.workspace()),
            normal.nll(1.1, &normal_theta, &mut normal.workspace()),
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn generic_drd_gradient_matches_finite_difference() {
        assert_gradient_matches_finite_difference::<1>(
            [1.1],
            &MvNormalMeanStdPartialCorrEta::new(
                [0.4],
                [0.8_f64.ln()],
                FixedPartialCorrelations::zeros(),
            ),
        );
        assert_gradient_matches_finite_difference::<2>(
            [1.1, -0.7],
            &MvNormalMeanStdPartialCorrEta::new(
                [0.4, -0.3],
                [0.8_f64.ln(), 1.2_f64.ln()],
                FixedPartialCorrelations::try_new(vec![0.25_f64.atanh()]).unwrap(),
            ),
        );
        assert_gradient_matches_finite_difference::<3>(
            [1.1, -0.7, 0.2],
            &MvNormalMeanStdPartialCorrEta::new(
                [0.4, -0.3, 0.1],
                [0.8_f64.ln(), 1.2_f64.ln(), 0.6_f64.ln()],
                FixedPartialCorrelations::try_new(vec![0.2, -0.1, 0.3]).unwrap(),
            ),
        );
    }

    #[test]
    fn extreme_finite_partial_correlation_eta_stays_valid_and_finite() {
        let family = MvNormalMeanStdPartialCorrDefault::<2>::new();
        for eta_value in [-100.0, -20.0, 20.0, 100.0] {
            let eta = MvNormalMeanStdPartialCorrEta::new(
                [0.0, 0.0],
                [0.0, 0.0],
                FixedPartialCorrelations::try_new(vec![eta_value]).unwrap(),
            );
            let theta = family.theta(&eta, &mut family.workspace());
            let (nll, gradient) =
                family.nll_and_gradient_eta([0.0, 0.0], &eta, &mut family.workspace());

            assert!(theta.partial_corr().get(1, 0).unwrap().abs() < 1.0);
            assert!(theta.correlation_cholesky().get(1, 1).unwrap() > 0.0);
            assert!(nll.is_finite());
            assert!(gradient.partial_corr.get(1, 0).unwrap().is_finite());
        }
    }

    #[test]
    fn marginal_cdf_uses_explicit_marginal_sigma() {
        let family = MvNormalMeanStdPartialCorrDefault::<3>::new();
        let eta = MvNormalMeanStdPartialCorrEta::new(
            [0.0, 1.0, -1.0],
            [2.0_f64.ln(), 3.0_f64.ln(), 4.0_f64.ln()],
            FixedPartialCorrelations::zeros(),
        );
        let theta = family.theta(&eta, &mut family.workspace());
        assert_relative_eq!(family.marginal_cdf(0, 0.0, &theta), 0.5, epsilon = 1.0e-12);
        assert_relative_eq!(theta.covariance(2, 2).unwrap(), 16.0, epsilon = 1.0e-12);
    }

    #[test]
    fn compiled_blocks_are_fit_ready() {
        let y = [[0.2, -0.3], [1.0, 0.4], [-0.5, 0.8]];
        let n = y.len();
        let mu = VectorParameterBlock::<Mu, 2, _, _>::new(
            [
                LinearPredictorBlock::new(DenseDesign::intercept(n)),
                LinearPredictorBlock::new(DenseDesign::intercept(n)),
            ],
            NoPenalty,
            99,
        );
        let sigma = VectorParameterBlock::<Sigma, 2, _, _>::new(
            [
                LinearPredictorBlock::new(DenseDesign::intercept(n)),
                LinearPredictorBlock::new(DenseDesign::intercept(n)),
            ],
            NoPenalty,
            99,
        );
        let rho = StrictLowerTriangularParameterBlock::<PartialCorrelation, 2, _, _>::new(
            vec![LinearPredictorBlock::new(DenseDesign::intercept(n))],
            NoPenalty,
            99,
        );
        let blocks = ParameterBlocks::new((mu, sigma, rho));
        let model = Gamlss::try_new_with_observations(
            MvNormalMeanStdPartialCorrDefault::<2>::new(),
            blocks,
            y.as_slice(),
        )
        .unwrap();
        let beta = vec![0.1, -0.2, 0.0, 0.3, 0.15];
        let eta = model.predict_eta_row(&beta, 0).unwrap();

        assert_eq!(model.nparams(), 5);
        assert_relative_eq!(eta.mu[0], 0.1);
        assert_relative_eq!(eta.sigma[1], 0.3);
        assert_relative_eq!(eta.partial_corr.get(1, 0).unwrap(), 0.15);

        let mut gradient = vec![0.0; beta.len()];
        model.try_value_gradient_into(&beta, &mut gradient).unwrap();
        for index in 0..beta.len() {
            let mut plus = beta.clone();
            plus[index] += 1.0e-6;
            let mut minus = beta.clone();
            minus[index] -= 1.0e-6;
            let fd = (model.try_value(&plus).unwrap() - model.try_value(&minus).unwrap()) / 2.0e-6;
            assert_relative_eq!(gradient[index], fd, epsilon = 1.0e-6);
        }
    }
}
