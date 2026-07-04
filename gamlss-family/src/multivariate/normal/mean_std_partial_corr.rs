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
    Family, FixedDimensionalFamily, HasMarginalCdf, Identity, Link, LocationScalePartialCorr,
    LocationScalePartialCorrSpec, Log, ModelError, Mu, PartialCorrelation, PositiveLink, Sigma,
};
use gamlss_special::unit_normal_cdf;

use crate::constants::HALF_LOG_2_PI;
use crate::univariate::normal::{NormalTheta, normal_nll_gradient_theta, normal_nll_theta};

use super::{FixedLowerTriangular, kernel};

/// Default-link `D R D` multivariate normal parameterization.
pub type MvNormalMeanStdPartialCorrDefault<const D: usize> =
    MvNormalMeanStdPartialCorr<D, Identity, Log>;

/// Packed strict-lower storage for dimension-generic partial-correlation predictors.
///
/// Packed order is `(1,0), (2,0), (2,1), (3,0), ...`. Values are kept on the
/// predictor scale by [`MvNormalMeanStdPartialCorrEta`]; the family maps them
/// with `tanh` before building the correlation Cholesky factor.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedPartialCorr<const D: usize> {
    values: Vec<f64>,
}

impl<const D: usize> PackedPartialCorr<D> {
    /// Creates packed storage after validating the expected length.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `values.len()` is not
    /// `D * (D - 1) / 2`.
    pub fn try_new(values: Vec<f64>) -> Result<Self, ModelError> {
        if values.len() != Self::len() {
            return Err(ModelError::InvalidParameter {
                parameter: "partial_corr",
                expected: "D * (D - 1) / 2 strict-lower values",
            });
        }
        Ok(Self { values })
    }

    /// Creates zero-valued packed storage.
    #[must_use]
    pub fn zeros() -> Self {
        Self {
            values: vec![0.0; Self::len()],
        }
    }

    /// Expected packed length for dimension `D`.
    #[must_use]
    pub const fn len() -> usize {
        D.saturating_mul(D.saturating_sub(1)) / 2
    }

    /// Returns true when there are no strict-lower entries.
    #[must_use]
    pub const fn is_empty() -> bool {
        Self::len() == 0
    }

    /// Packed values.
    #[must_use]
    #[inline]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Mutable packed values.
    #[must_use]
    #[inline]
    pub fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }

    /// Returns a strict-lower entry, or `None` for invalid/diagonal/upper indices.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        Self::packed_index(row, col).and_then(|index| self.values.get(index).copied())
    }

    /// Returns a mutable strict-lower entry, or `None` for invalid/diagonal/upper indices.
    #[must_use]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut f64> {
        Self::packed_index(row, col).and_then(|index| self.values.get_mut(index))
    }

    const fn packed_index(row: usize, col: usize) -> Option<usize> {
        if row < D && col < row {
            Some(row * (row - 1) / 2 + col)
        } else {
            None
        }
    }

    #[inline]
    fn lower(&self, row: usize, col: usize) -> f64 {
        self.values[Self::packed_index(row, col).expect("valid strict-lower index")]
    }
}

/// Generic multivariate normal parameterized as `Sigma = D R D`.
///
/// Means and marginal standard deviations are represented explicitly. The
/// correlation matrix is built from strict-lower partial-correlation predictors
/// mapped through `tanh`, then converted to a correlation Cholesky factor. For
/// `D == 2`, a private bivariate fast path evaluates the density and gradient
/// directly without materializing Cholesky workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MvNormalMeanStdPartialCorr<const D: usize, MuLink = Identity, SigmaLink = Log> {
    marker: PhantomData<(MuLink, SigmaLink)>,
}

impl<const D: usize, MuLink, SigmaLink> MvNormalMeanStdPartialCorr<D, MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    /// Creates a stateless family value.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
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

        MvNormalMeanStdPartialCorrTheta {
            mu,
            sigma,
            partial_corr,
            correlation_cholesky,
            scale_cholesky,
        }
    }

    fn nan_eta() -> MvNormalMeanStdPartialCorrEta<D> {
        nan_eta()
    }

    fn nll_theta(observation: [f64; D], theta: &MvNormalMeanStdPartialCorrTheta<D>) -> f64 {
        match D {
            1 => univariate_nll(observation, theta),
            2 => bivariate_nll(observation, theta),
            _ => {
                let mut z = [0.0; D];
                kernel::nll(D, &observation, &theta.mu, &theta.scale_cholesky, &mut z)
            }
        }
    }

    fn nll_and_gradient_eta_values(
        observation: [f64; D],
        eta: &MvNormalMeanStdPartialCorrEta<D>,
    ) -> (f64, MvNormalMeanStdPartialCorrEta<D>) {
        let theta = Self::theta_from_eta(eta);
        match D {
            1 => univariate_nll_and_gradient_eta::<D, MuLink, SigmaLink>(observation, eta, &theta),
            2 => bivariate_nll_and_gradient_eta::<D, MuLink, SigmaLink>(observation, eta, &theta),
            _ => Self::generic_nll_and_gradient_eta_values(observation, eta, &theta),
        }
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
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
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
    type ParamSpec = LocationScalePartialCorr<Mu, Sigma, PartialCorrelation, D>;

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

impl<const D: usize, MuLink, SigmaLink>
    LocationScalePartialCorrSpec<MvNormalMeanStdPartialCorr<D, MuLink, SigmaLink>, D>
    for LocationScalePartialCorr<Mu, Sigma, PartialCorrelation, D>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    type LocationParameter = Mu;
    type ScaleParameter = Sigma;
    type PartialCorrelationParameter = PartialCorrelation;

    fn eta_from_location_scale_partial_corr(
        location: [f64; D],
        scale: [f64; D],
        partial_corr: [[f64; D]; D],
    ) -> MvNormalMeanStdPartialCorrEta<D> {
        let values = (0..D)
            .flat_map(|row| (0..row).map(move |col| partial_corr[row][col]))
            .collect();
        MvNormalMeanStdPartialCorrEta::new(
            location,
            scale,
            PackedPartialCorr::try_new(values).expect("packed strict-lower length matches D"),
        )
    }

    fn location_gradient_part(
        gradient: &MvNormalMeanStdPartialCorrEta<D>,
        component: usize,
    ) -> f64 {
        gradient.mu[component]
    }

    fn scale_gradient_part(gradient: &MvNormalMeanStdPartialCorrEta<D>, component: usize) -> f64 {
        gradient.sigma[component]
    }

    fn partial_corr_gradient_part(
        gradient: &MvNormalMeanStdPartialCorrEta<D>,
        row: usize,
        col: usize,
    ) -> f64 {
        gradient
            .partial_corr
            .get(row, col)
            .expect("valid strict-lower partial-correlation index")
    }
}

impl<const D: usize, MuLink, SigmaLink> FixedDimensionalFamily<D>
    for MvNormalMeanStdPartialCorr<D, MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
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

/// Link-scale predictors for `D R D` multivariate normal.
#[derive(Debug, Clone, PartialEq)]
pub struct MvNormalMeanStdPartialCorrEta<const D: usize> {
    /// Mean predictors.
    pub mu: [f64; D],
    /// Marginal standard-deviation predictors.
    pub sigma: [f64; D],
    /// Strict-lower partial-correlation predictors.
    pub partial_corr: PackedPartialCorr<D>,
}

impl<const D: usize> MvNormalMeanStdPartialCorrEta<D> {
    /// Creates link-scale predictors.
    #[must_use]
    #[inline]
    pub const fn new(mu: [f64; D], sigma: [f64; D], partial_corr: PackedPartialCorr<D>) -> Self {
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
    pub mu: [f64; D],
    /// Positive marginal standard deviations.
    pub sigma: [f64; D],
    /// Strict-lower partial correlations on `(-1, 1)`.
    pub partial_corr: PackedPartialCorr<D>,
    /// Correlation Cholesky factor.
    pub correlation_cholesky: FixedLowerTriangular<D>,
    /// Scale Cholesky factor for `Sigma = D R D`.
    pub scale_cholesky: FixedLowerTriangular<D>,
}

impl<const D: usize> MvNormalMeanStdPartialCorrTheta<D> {
    /// Returns one covariance entry from `D R D`.
    #[must_use]
    pub fn covariance(&self, row: usize, col: usize) -> Option<f64> {
        covariance_from_cholesky(&self.scale_cholesky, row, col)
    }
}

fn partial_corr_from_eta<const D: usize>(eta: &PackedPartialCorr<D>) -> PackedPartialCorr<D> {
    PackedPartialCorr {
        values: eta.values.iter().map(|value| value.tanh()).collect(),
    }
}

fn correlation_cholesky_from_partial<const D: usize>(
    partial_corr: &PackedPartialCorr<D>,
) -> FixedLowerTriangular<D> {
    let mut out = FixedLowerTriangular::zeros();
    for row in 0..D {
        let mut prefix = 1.0;
        for col in 0..row {
            let p = partial_corr.lower(row, col);
            out.set_lower(row, col, p * prefix)
                .expect("valid lower index");
            prefix *= (1.0 - p * p).sqrt();
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
            .values
            .iter()
            .all(|value| value.is_finite() && value.abs() < 1.0)
        && kernel::valid_theta(D, &theta.mu, &theta.scale_cholesky)
}

fn nan_eta<const D: usize>() -> MvNormalMeanStdPartialCorrEta<D> {
    MvNormalMeanStdPartialCorrEta {
        mu: [f64::NAN; D],
        sigma: [f64::NAN; D],
        partial_corr: PackedPartialCorr {
            values: vec![f64::NAN; PackedPartialCorr::<D>::len()],
        },
    }
}

fn zero_eta<const D: usize>() -> MvNormalMeanStdPartialCorrEta<D> {
    MvNormalMeanStdPartialCorrEta {
        mu: [0.0; D],
        sigma: [0.0; D],
        partial_corr: PackedPartialCorr::zeros(),
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

    let mut cholesky_score = [[0.0; D]; D];
    for row in 0..D {
        for col in 0..=row {
            cholesky_score[row][col] =
                kernel::cholesky_score(row, col, z[col], a, &theta.scale_cholesky);
        }
        gradient.mu[row] = -a[row] * MuLink::derivative_inverse(eta.mu[row]);
        gradient.sigma[row] = (0..=row)
            .map(|col| cholesky_score[row][col] * theta.correlation_cholesky.lower(row, col))
            .sum::<f64>()
            * SigmaLink::derivative_inverse(eta.sigma[row]);
    }

    for row in 1..D {
        for k in 0..row {
            let p = theta.partial_corr.lower(row, k);
            let one_minus_p2 = 1.0 - p * p;
            let prefix_k = row_prefix(&theta.partial_corr, row, k);
            let mut d_nll_d_p = cholesky_score[row][k] * theta.sigma[row] * prefix_k;
            for col in (k + 1)..=row {
                d_nll_d_p += cholesky_score[row][col]
                    * theta.sigma[row]
                    * theta.correlation_cholesky.lower(row, col)
                    * (-p / one_minus_p2);
            }
            *gradient
                .partial_corr
                .get_mut(row, k)
                .expect("valid strict-lower index") = d_nll_d_p * one_minus_p2;
        }
    }

    gradient
}

fn row_prefix<const D: usize>(
    partial_corr: &PackedPartialCorr<D>,
    row: usize,
    before_col: usize,
) -> f64 {
    (0..before_col)
        .map(|col| {
            let p = partial_corr.lower(row, col);
            (1.0 - p * p).sqrt()
        })
        .product()
}

fn univariate_nll<const D: usize>(
    observation: [f64; D],
    theta: &MvNormalMeanStdPartialCorrTheta<D>,
) -> f64 {
    if D != 1 {
        return f64::INFINITY;
    }
    normal_nll_theta(
        observation[0],
        NormalTheta {
            mu: theta.mu[0],
            sigma: theta.sigma[0],
        },
    )
}

fn univariate_nll_and_gradient_eta<const D: usize, MuLink, SigmaLink>(
    observation: [f64; D],
    eta: &MvNormalMeanStdPartialCorrEta<D>,
    theta: &MvNormalMeanStdPartialCorrTheta<D>,
) -> (f64, MvNormalMeanStdPartialCorrEta<D>)
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    if D != 1 {
        return (f64::INFINITY, nan_eta());
    }

    let (nll, gradient_theta) = normal_nll_gradient_theta(
        observation[0],
        NormalTheta {
            mu: theta.mu[0],
            sigma: theta.sigma[0],
        },
    );
    if !nll.is_finite() {
        return (nll, nan_eta());
    }

    let mut gradient = zero_eta();
    gradient.mu[0] = gradient_theta.mu * MuLink::derivative_inverse(eta.mu[0]);
    gradient.sigma[0] = gradient_theta.sigma * SigmaLink::derivative_inverse(eta.sigma[0]);
    (nll, gradient)
}

fn bivariate_nll<const D: usize>(
    observation: [f64; D],
    theta: &MvNormalMeanStdPartialCorrTheta<D>,
) -> f64 {
    if D != 2 || !valid_theta(theta) || !observation.iter().all(|value| value.is_finite()) {
        return f64::INFINITY;
    }
    let x0 = (observation[0] - theta.mu[0]) / theta.sigma[0];
    let x1 = (observation[1] - theta.mu[1]) / theta.sigma[1];
    let rho = theta.partial_corr.lower(1, 0);
    let one_minus_rho2 = 1.0 - rho * rho;
    let q = x0 * x0 - 2.0 * rho * x0 * x1 + x1 * x1;
    2.0 * HALF_LOG_2_PI
        + theta.sigma[0].ln()
        + theta.sigma[1].ln()
        + 0.5 * one_minus_rho2.ln()
        + 0.5 * q / one_minus_rho2
}

fn bivariate_nll_and_gradient_eta<const D: usize, MuLink, SigmaLink>(
    observation: [f64; D],
    eta: &MvNormalMeanStdPartialCorrEta<D>,
    theta: &MvNormalMeanStdPartialCorrTheta<D>,
) -> (f64, MvNormalMeanStdPartialCorrEta<D>)
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    let nll = bivariate_nll(observation, theta);
    if !nll.is_finite() {
        return (nll, nan_eta());
    }

    let x0 = (observation[0] - theta.mu[0]) / theta.sigma[0];
    let x1 = (observation[1] - theta.mu[1]) / theta.sigma[1];
    let rho = theta.partial_corr.lower(1, 0);
    let one_minus_rho2 = 1.0 - rho * rho;
    let q = x0 * x0 - 2.0 * rho * x0 * x1 + x1 * x1;

    let mut gradient = zero_eta();
    gradient.mu[0] =
        (rho * x1 - x0) / (theta.sigma[0] * one_minus_rho2) * MuLink::derivative_inverse(eta.mu[0]);
    gradient.mu[1] =
        (rho * x0 - x1) / (theta.sigma[1] * one_minus_rho2) * MuLink::derivative_inverse(eta.mu[1]);
    gradient.sigma[0] = (1.0 / theta.sigma[0]
        - x0 * (x0 - rho * x1) / (theta.sigma[0] * one_minus_rho2))
        * SigmaLink::derivative_inverse(eta.sigma[0]);
    gradient.sigma[1] = (1.0 / theta.sigma[1]
        - x1 * (x1 - rho * x0) / (theta.sigma[1] * one_minus_rho2))
        * SigmaLink::derivative_inverse(eta.sigma[1]);
    let d_nll_d_rho =
        -rho / one_minus_rho2 - x0 * x1 / one_minus_rho2 + rho * q / one_minus_rho2.powi(2);
    *gradient
        .partial_corr
        .get_mut(1, 0)
        .expect("valid strict-lower index") = d_nll_d_rho * one_minus_rho2;
    (nll, gradient)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{
        DenseDesign, Family, Gamlss, HasMarginalCdf, LinearPredictorBlock, Mu, NoPenalty,
        ParameterBlocks, PartialCorrelation, Sigma, StrictLowerTriangularParameterBlock,
        VectorParameterBlock,
    };

    use super::{
        MvNormalMeanStdPartialCorrDefault, MvNormalMeanStdPartialCorrEta, PackedPartialCorr,
    };
    use crate::multivariate::normal::{
        FixedLowerTriangular, MvNormalCholeskyDefault, MvNormalCholeskyTheta,
    };
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

    #[test]
    fn packed_partial_corr_validates_length_and_indices() {
        assert_eq!(PackedPartialCorr::<4>::len(), 6);
        assert!(PackedPartialCorr::<4>::try_new(vec![0.0; 5]).is_err());
        let mut packed = PackedPartialCorr::<3>::try_new(vec![0.1, 0.2, 0.3]).unwrap();
        assert_eq!(packed.get(1, 0), Some(0.1));
        assert_eq!(packed.get(2, 1), Some(0.3));
        assert_eq!(packed.get(0, 0), None);
        *packed.get_mut(2, 0).unwrap() = -0.2;
        assert_eq!(packed.get(2, 0), Some(-0.2));
    }

    #[test]
    fn d2_fast_path_matches_cholesky_form() {
        let drd = MvNormalMeanStdPartialCorrDefault::<2>::new();
        let rho_eta = 0.25_f64.atanh();
        let eta = MvNormalMeanStdPartialCorrEta::new(
            [0.4, -0.3],
            [0.8_f64.ln(), 1.2_f64.ln()],
            PackedPartialCorr::try_new(vec![rho_eta]).unwrap(),
        );
        let theta = drd.theta(&eta, &mut drd.workspace());
        let cholesky = MvNormalCholeskyDefault::<2>::new();
        let cholesky_theta = MvNormalCholeskyTheta::new(
            [0.4, -0.3],
            FixedLowerTriangular::from_lower_rows([
                [0.8, 0.0],
                [1.2 * 0.25, 1.2 * (1.0 - 0.25_f64.powi(2)).sqrt()],
            ]),
        );
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
        let eta =
            MvNormalMeanStdPartialCorrEta::new([0.4], [0.8_f64.ln()], PackedPartialCorr::zeros());
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
            &MvNormalMeanStdPartialCorrEta::new([0.4], [0.8_f64.ln()], PackedPartialCorr::zeros()),
        );
        assert_gradient_matches_finite_difference::<2>(
            [1.1, -0.7],
            &MvNormalMeanStdPartialCorrEta::new(
                [0.4, -0.3],
                [0.8_f64.ln(), 1.2_f64.ln()],
                PackedPartialCorr::try_new(vec![0.25_f64.atanh()]).unwrap(),
            ),
        );
        assert_gradient_matches_finite_difference::<3>(
            [1.1, -0.7, 0.2],
            &MvNormalMeanStdPartialCorrEta::new(
                [0.4, -0.3, 0.1],
                [0.8_f64.ln(), 1.2_f64.ln(), 0.6_f64.ln()],
                PackedPartialCorr::try_new(vec![0.2, -0.1, 0.3]).unwrap(),
            ),
        );
    }

    #[test]
    fn marginal_cdf_uses_explicit_marginal_sigma() {
        let family = MvNormalMeanStdPartialCorrDefault::<3>::new();
        let eta = MvNormalMeanStdPartialCorrEta::new(
            [0.0, 1.0, -1.0],
            [2.0_f64.ln(), 3.0_f64.ln(), 4.0_f64.ln()],
            PackedPartialCorr::zeros(),
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
