#![allow(clippy::needless_range_loop, clippy::suboptimal_flops)]

//! Location/kernel-SD/partial-correlation multivariate skew-normal parameterization.

use std::marker::PhantomData;

use gamlss_core::{
    CompilableFamily, Family, FixedDimensionalFamily, HasObservationDimension, Identity,
    InitialEtaFromTheta, KernelSigma, Link, Log, ModelError, Mu, Nu, ObservationView,
    PartialCorrelation, PositiveLink,
    shape::{Product, ShapeValues, StrictLower, Vector},
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

use crate::multivariate::{
    elliptical,
    matrix::FixedLowerTriangular,
    normal::{
        FixedPartialCorrelations, MvNormalMeanStdPartialCorr, kernel,
        mean_std_partial_corr::{
            correlation_cholesky_from_partial, covariance_from_cholesky, partial_corr_from_eta,
            partial_corr_gradient_from_cholesky_score, scale_cholesky_from_correlation,
        },
    },
};

#[cfg(feature = "rand")]
use super::try_sample_location_scale;
use super::{skew_gradient_terms, skew_nll};

/// Default-link skew-normal with location, kernel SDs and partial correlations.
pub type MvSkewNormalLocationKernelStdPartialCorrDefault<const D: usize> =
    MvSkewNormalLocationKernelStdPartialCorr<D, Identity, Log, Identity>;

/// Multivariate skew-normal with interpretable Gaussian-kernel scale predictors.
///
/// For `z = L^-1 (y - mu)`, the density is
/// `2 phi_D(z) Phi(alpha' z) / det(L)`, where `L = diag(kernel_sigma) C`
/// and `C C'` is a correlation matrix assembled from ordered partial
/// correlations. Thus `mu` is a location vector and `kernel_sigma` describes
/// the Gaussian kernel, not the skew-normal response mean and marginal SDs
/// when `alpha != 0`. The actual moments are available from
/// [`MvSkewNormalLocationKernelStdPartialCorrTheta::mean`] and
/// [`MvSkewNormalLocationKernelStdPartialCorrTheta::covariance`].
///
/// This form keeps all predictors unconstrained and is useful when marginal
/// kernel scales and an ordered conditional-dependence structure are easier to
/// model than raw Cholesky entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MvSkewNormalLocationKernelStdPartialCorr<
    const D: usize,
    MuLink = Identity,
    SigmaLink = Log,
    ShapeLink = Identity,
> {
    marker: PhantomData<(MuLink, SigmaLink, ShapeLink)>,
}

impl<const D: usize, MuLink, SigmaLink, ShapeLink>
    MvSkewNormalLocationKernelStdPartialCorr<D, MuLink, SigmaLink, ShapeLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    ShapeLink: Link<f64>,
{
    /// Creates a stateless family after checking the compile-time dimension.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `D == 0`.
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

    /// Creates a stateless family.
    ///
    /// # Panics
    ///
    /// Panics when `D == 0`.
    #[must_use]
    pub const fn new() -> Self {
        assert!(D > 0, "multivariate skew-normal dimension must be positive");
        Self {
            marker: PhantomData,
        }
    }

    fn theta_from_eta(
        eta: &MvSkewNormalLocationKernelStdPartialCorrEta<D>,
    ) -> MvSkewNormalLocationKernelStdPartialCorrTheta<D> {
        let mu = eta.mu.map(MuLink::inverse);
        let kernel_sigma = eta.kernel_sigma.map(SigmaLink::inverse);
        let partial_corr = partial_corr_from_eta(&eta.partial_corr);
        let correlation_cholesky = correlation_cholesky_from_partial(&partial_corr);
        let kernel_cholesky = scale_cholesky_from_correlation(&kernel_sigma, &correlation_cholesky);
        let shape = eta.shape.map(ShapeLink::inverse);
        MvSkewNormalLocationKernelStdPartialCorrTheta::from_parts_unchecked(
            mu,
            kernel_sigma,
            partial_corr,
            correlation_cholesky,
            kernel_cholesky,
            shape,
        )
    }

    const fn nan_eta() -> MvSkewNormalLocationKernelStdPartialCorrEta<D> {
        MvSkewNormalLocationKernelStdPartialCorrEta {
            mu: [f64::NAN; D],
            kernel_sigma: [f64::NAN; D],
            partial_corr: FixedPartialCorrelations::filled_strict_lower(f64::NAN),
            shape: [f64::NAN; D],
        }
    }

    fn nll_theta(
        observation: [f64; D],
        theta: &MvSkewNormalLocationKernelStdPartialCorrTheta<D>,
    ) -> f64 {
        if !valid_theta(theta) {
            return f64::INFINITY;
        }
        let mut z = [0.0; D];
        let gaussian_nll = kernel::nll(D, &observation, &theta.mu, &theta.kernel_cholesky, &mut z);
        skew_nll(gaussian_nll, &z, &theta.shape)
    }

    fn nll_and_gradient_eta_values(
        observation: [f64; D],
        eta: &MvSkewNormalLocationKernelStdPartialCorrEta<D>,
    ) -> (f64, MvSkewNormalLocationKernelStdPartialCorrEta<D>) {
        let theta = Self::theta_from_eta(eta);
        if !valid_theta(&theta) {
            return (f64::INFINITY, Self::nan_eta());
        }

        let mut z = [0.0; D];
        let gaussian_nll = kernel::nll(D, &observation, &theta.mu, &theta.kernel_cholesky, &mut z);
        let Some(terms) = skew_gradient_terms(gaussian_nll, &z, &theta.shape) else {
            return (f64::INFINITY, Self::nan_eta());
        };
        let mut location_score = [0.0; D];
        if !elliptical::transpose_solve(
            D,
            &theta.kernel_cholesky,
            &terms.standardized,
            &mut location_score,
        ) {
            return (f64::INFINITY, Self::nan_eta());
        }

        let mut cholesky_score = [[0.0; D]; D];
        let kernel_sigma = std::array::from_fn(|row| {
            let residual = (0..=row)
                .map(|col| {
                    -location_score[row] * z[col] * theta.correlation_cholesky.lower(row, col)
                })
                .sum::<f64>();
            SigmaLink::derivative_log_inverse(eta.kernel_sigma[row])
                + residual * SigmaLink::derivative_inverse(eta.kernel_sigma[row])
        });
        for row in 0..D {
            for col in 0..=row {
                cholesky_score[row][col] = -theta.kernel_sigma[row] * location_score[row] * z[col]
                    + if row == col {
                        theta.correlation_cholesky.lower(row, row).recip()
                    } else {
                        0.0
                    };
            }
        }

        let gradient = MvSkewNormalLocationKernelStdPartialCorrEta {
            mu: std::array::from_fn(|component| {
                -location_score[component] * MuLink::derivative_inverse(eta.mu[component])
            }),
            kernel_sigma,
            partial_corr: partial_corr_gradient_from_cholesky_score(
                &eta.partial_corr,
                &theta.correlation_cholesky,
                &cholesky_score,
            ),
            shape: std::array::from_fn(|component| {
                terms.shape[component] * ShapeLink::derivative_inverse(eta.shape[component])
            }),
        };
        (terms.nll, gradient)
    }
}

impl<const D: usize, MuLink, SigmaLink, ShapeLink> Default
    for MvSkewNormalLocationKernelStdPartialCorr<D, MuLink, SigmaLink, ShapeLink>
where
    MuLink: InitialEtaFromTheta<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    ShapeLink: InitialEtaFromTheta<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize, MuLink, SigmaLink, ShapeLink> Family
    for MvSkewNormalLocationKernelStdPartialCorr<D, MuLink, SigmaLink, ShapeLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    ShapeLink: Link<f64>,
{
    type Eta = MvSkewNormalLocationKernelStdPartialCorrEta<D>;
    type Theta = MvSkewNormalLocationKernelStdPartialCorrTheta<D>;
    type GradientEta = MvSkewNormalLocationKernelStdPartialCorrEta<D>;
    type Observation<'obs> = [f64; D];
    type Workspace = ();

    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut ()) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    fn nll(&self, observation: [f64; D], theta: &Self::Theta, _workspace: &mut ()) -> f64 {
        Self::nll_theta(observation, theta)
    }

    fn nll_eta(&self, observation: [f64; D], eta: &Self::Eta, _workspace: &mut ()) -> f64 {
        Self::nll_theta(observation, &Self::theta_from_eta(eta))
    }

    fn nll_and_gradient_eta(
        &self,
        observation: [f64; D],
        eta: &Self::Eta,
        _workspace: &mut (),
    ) -> (f64, Self::GradientEta) {
        Self::nll_and_gradient_eta_values(observation, eta)
    }
}

impl<const D: usize, MuLink, SigmaLink, ShapeLink> FixedDimensionalFamily<D>
    for MvSkewNormalLocationKernelStdPartialCorr<D, MuLink, SigmaLink, ShapeLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    ShapeLink: Link<f64>,
{
}

impl<const D: usize, MuLink, SigmaLink, ShapeLink> HasObservationDimension
    for MvSkewNormalLocationKernelStdPartialCorr<D, MuLink, SigmaLink, ShapeLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    ShapeLink: Link<f64>,
{
    fn observation_dimension(&self) -> usize {
        D
    }
}

#[cfg(feature = "rand")]
impl<Rng, const D: usize, MuLink, SigmaLink, ShapeLink> TrySimulate<Rng>
    for MvSkewNormalLocationKernelStdPartialCorr<D, MuLink, SigmaLink, ShapeLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    ShapeLink: Link<f64>,
{
    type Sample = [f64; D];

    fn try_sample(
        &self,
        rng: &mut Rng,
        theta: &Self::Theta,
    ) -> Result<Self::Sample, SimulationError> {
        if !valid_theta(theta) {
            return Err(SimulationError::InvalidParameters(
                "skew-normal location/kernel-SD/partial-correlation theta",
            ));
        }
        try_sample_location_scale(rng, theta.mu, &theta.kernel_cholesky, &theta.shape)
    }
}

impl<const D: usize, MuLink, SigmaLink, ShapeLink> CompilableFamily
    for MvSkewNormalLocationKernelStdPartialCorr<D, MuLink, SigmaLink, ShapeLink>
where
    MuLink: InitialEtaFromTheta<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    ShapeLink: InitialEtaFromTheta<f64>,
{
    type Shape = Product<
        Product<Product<Vector<Mu, D>, Vector<KernelSigma, D>>, StrictLower<PartialCorrelation, D>>,
        Vector<Nu, D>,
    >;

    fn eta_from_shape(values: ShapeValues<Self::Shape>) -> Self::Eta {
        MvSkewNormalLocationKernelStdPartialCorrEta::new(
            values.0.0.0,
            values.0.0.1,
            FixedPartialCorrelations::from_lower_rows(values.0.1),
            values.1,
        )
    }

    fn gradient_to_shape(gradient: &Self::GradientEta) -> ShapeValues<Self::Shape> {
        (
            (
                (gradient.mu, gradient.kernel_sigma),
                gradient.partial_corr.lower_rows(),
            ),
            gradient.shape,
        )
    }

    fn initial_shape<'obs, Obs>(&self, obs: &'obs Obs) -> ShapeValues<Self::Shape>
    where
        Obs: ObservationView<'obs, Observation = [f64; D]> + 'obs,
    {
        let normal = MvNormalMeanStdPartialCorr::<D, MuLink, SigmaLink>::new();
        (
            normal.initial_shape(obs),
            [ShapeLink::initial_eta_from_theta(0.0); D],
        )
    }
}

/// Link-scale predictors for the location/kernel-SD parameterization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MvSkewNormalLocationKernelStdPartialCorrEta<const D: usize> {
    /// Location predictors.
    pub mu: [f64; D],
    /// Gaussian-kernel marginal SD predictors.
    pub kernel_sigma: [f64; D],
    /// Strict-lower ordered partial-correlation predictors.
    pub partial_corr: FixedPartialCorrelations<D>,
    /// Whitened-coordinate shape predictors.
    pub shape: [f64; D],
}

impl<const D: usize> MvSkewNormalLocationKernelStdPartialCorrEta<D> {
    /// Creates link-scale predictors.
    #[must_use]
    pub const fn new(
        mu: [f64; D],
        kernel_sigma: [f64; D],
        partial_corr: FixedPartialCorrelations<D>,
        shape: [f64; D],
    ) -> Self {
        Self {
            mu,
            kernel_sigma,
            partial_corr,
            shape,
        }
    }
}

/// Natural-scale parameters and derived moments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MvSkewNormalLocationKernelStdPartialCorrTheta<const D: usize> {
    mu: [f64; D],
    kernel_sigma: [f64; D],
    partial_corr: FixedPartialCorrelations<D>,
    correlation_cholesky: FixedLowerTriangular<D>,
    kernel_cholesky: FixedLowerTriangular<D>,
    shape: [f64; D],
}

impl<const D: usize> MvSkewNormalLocationKernelStdPartialCorrTheta<D> {
    /// Creates checked natural-scale parameters and their derived factors.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] for zero dimension, non-finite
    /// entries, non-positive kernel SDs, or partial correlations outside
    /// `(-1, 1)`.
    pub fn try_new(
        mu: [f64; D],
        kernel_sigma: [f64; D],
        partial_corr: FixedPartialCorrelations<D>,
        shape: [f64; D],
    ) -> Result<Self, ModelError> {
        let correlation_cholesky = correlation_cholesky_from_partial(&partial_corr);
        let kernel_cholesky = scale_cholesky_from_correlation(&kernel_sigma, &correlation_cholesky);
        let theta = Self::from_parts_unchecked(
            mu,
            kernel_sigma,
            partial_corr,
            correlation_cholesky,
            kernel_cholesky,
            shape,
        );
        if valid_theta(&theta) {
            Ok(theta)
        } else {
            Err(ModelError::InvalidParameter {
                parameter: "skew-normal location/kernel-SD/partial-correlation theta",
                expected: "positive dimension, finite values, positive kernel SDs, and partial correlations strictly between -1 and 1",
            })
        }
    }

    const fn from_parts_unchecked(
        mu: [f64; D],
        kernel_sigma: [f64; D],
        partial_corr: FixedPartialCorrelations<D>,
        correlation_cholesky: FixedLowerTriangular<D>,
        kernel_cholesky: FixedLowerTriangular<D>,
        shape: [f64; D],
    ) -> Self {
        Self {
            mu,
            kernel_sigma,
            partial_corr,
            correlation_cholesky,
            kernel_cholesky,
            shape,
        }
    }

    /// Location vector `mu`.
    #[must_use]
    pub const fn mu(&self) -> &[f64; D] {
        &self.mu
    }

    /// Gaussian-kernel marginal standard deviations.
    #[must_use]
    pub const fn kernel_sigma(&self) -> &[f64; D] {
        &self.kernel_sigma
    }

    /// Canonical ordered partial correlations.
    #[must_use]
    pub const fn partial_corr(&self) -> &FixedPartialCorrelations<D> {
        &self.partial_corr
    }

    /// Correlation Cholesky factor.
    #[must_use]
    pub const fn correlation_cholesky(&self) -> &FixedLowerTriangular<D> {
        &self.correlation_cholesky
    }

    /// Cholesky factor of the Gaussian-kernel scale.
    #[must_use]
    pub const fn kernel_cholesky(&self) -> &FixedLowerTriangular<D> {
        &self.kernel_cholesky
    }

    /// Whitened-coordinate shape vector.
    #[must_use]
    pub const fn shape(&self) -> &[f64; D] {
        &self.shape
    }

    /// Actual response mean, accounting for skewness.
    #[must_use]
    pub fn mean(&self) -> [f64; D] {
        let shift = self.mean_shift();
        std::array::from_fn(|component| self.mu[component] + shift[component])
    }

    /// Returns one Gaussian-kernel covariance entry.
    #[must_use]
    pub fn kernel_covariance(&self, row: usize, col: usize) -> Option<f64> {
        covariance_from_cholesky(&self.kernel_cholesky, row, col)
    }

    /// Returns one actual response covariance entry, accounting for skewness.
    #[must_use]
    pub fn covariance(&self, row: usize, col: usize) -> Option<f64> {
        let kernel = self.kernel_covariance(row, col)?;
        let shift = self.mean_shift();
        Some(kernel - shift[row] * shift[col])
    }

    fn mean_shift(&self) -> [f64; D] {
        let normalization = self
            .shape
            .iter()
            .fold(1.0_f64, |norm, shape| norm.hypot(*shape));
        let delta = self.shape.map(|shape| shape / normalization);
        let factor = (2.0 / std::f64::consts::PI).sqrt();
        std::array::from_fn(|row| {
            factor
                * (0..=row)
                    .map(|col| self.kernel_cholesky.lower(row, col) * delta[col])
                    .sum::<f64>()
        })
    }
}

fn valid_theta<const D: usize>(theta: &MvSkewNormalLocationKernelStdPartialCorrTheta<D>) -> bool {
    D > 0
        && theta
            .kernel_sigma
            .iter()
            .all(|sigma| sigma.is_finite() && *sigma > 0.0)
        && theta
            .partial_corr
            .iter()
            .all(|partial| partial.is_finite() && partial.abs() < 1.0)
        && theta.shape.iter().all(|shape| shape.is_finite())
        && elliptical::valid_location_scale(D, &theta.mu, &theta.kernel_cholesky)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{
        DenseDesign, Family, Gamlss, KernelSigma, LinearPredictorBlock, ModelError, Mu, NoPenalty,
        Nu, ParameterBlocks, PartialCorrelation, StrictLowerTriangularParameterBlock,
        VectorParameterBlock,
    };

    use super::{
        MvSkewNormalLocationKernelStdPartialCorrDefault,
        MvSkewNormalLocationKernelStdPartialCorrEta, MvSkewNormalLocationKernelStdPartialCorrTheta,
    };
    use crate::multivariate::{
        normal::FixedPartialCorrelations,
        skew_normal::{MvSkewNormalCholeskyDefault, MvSkewNormalCholeskyTheta},
    };

    fn partial_corr() -> FixedPartialCorrelations<2> {
        FixedPartialCorrelations::try_new(vec![0.35]).unwrap()
    }

    #[test]
    fn checked_constructors_reject_invalid_domains() {
        assert_eq!(
            MvSkewNormalLocationKernelStdPartialCorrDefault::<0>::try_new(),
            Err(ModelError::InvalidParameter {
                parameter: "dimension",
                expected: "positive",
            })
        );
        assert!(
            MvSkewNormalLocationKernelStdPartialCorrTheta::<0>::try_new(
                [],
                [],
                FixedPartialCorrelations::zeros(),
                [],
            )
            .is_err()
        );
        assert!(
            MvSkewNormalLocationKernelStdPartialCorrTheta::try_new(
                [0.0; 2],
                [1.0, 0.0],
                FixedPartialCorrelations::zeros(),
                [0.0; 2],
            )
            .is_err()
        );
        assert!(
            MvSkewNormalLocationKernelStdPartialCorrTheta::try_new(
                [0.0; 2],
                [1.0; 2],
                FixedPartialCorrelations::try_new(vec![1.0]).unwrap(),
                [0.0; 2],
            )
            .is_err()
        );
        assert!(
            MvSkewNormalLocationKernelStdPartialCorrTheta::try_new(
                [0.0; 2],
                [1.0; 2],
                FixedPartialCorrelations::zeros(),
                [f64::NAN, 0.0],
            )
            .is_err()
        );
    }

    #[test]
    fn likelihood_matches_cholesky_parameterization() {
        let theta = MvSkewNormalLocationKernelStdPartialCorrTheta::try_new(
            [0.2, -0.3],
            [1.1, 0.7],
            partial_corr(),
            [0.8, -0.4],
        )
        .unwrap();
        let cholesky_theta = MvSkewNormalCholeskyTheta::try_new(
            *theta.mu(),
            *theta.kernel_cholesky(),
            *theta.shape(),
        )
        .unwrap();
        let observation = [1.0, -0.8];
        assert_relative_eq!(
            MvSkewNormalLocationKernelStdPartialCorrDefault::<2>::new().nll(
                observation,
                &theta,
                &mut ()
            ),
            MvSkewNormalCholeskyDefault::<2>::new().nll(observation, &cholesky_theta, &mut ()),
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn actual_moments_equal_kernel_moments_at_zero_shape() {
        let theta = MvSkewNormalLocationKernelStdPartialCorrTheta::try_new(
            [0.2, -0.3],
            [1.1, 0.7],
            partial_corr(),
            [0.0, 0.0],
        )
        .unwrap();
        assert_relative_eq!(theta.mean()[0], 0.2, epsilon = 1.0e-14);
        assert_relative_eq!(theta.mean()[1], -0.3, epsilon = 1.0e-14);
        for row in 0..2 {
            for col in 0..2 {
                assert_relative_eq!(
                    theta.covariance(row, col).unwrap(),
                    theta.kernel_covariance(row, col).unwrap(),
                    epsilon = 1.0e-14
                );
            }
        }
    }

    #[test]
    fn analytic_gradient_matches_finite_difference() {
        let family = MvSkewNormalLocationKernelStdPartialCorrDefault::<2>::new();
        let eta = MvSkewNormalLocationKernelStdPartialCorrEta::new(
            [0.1, -0.2],
            [0.15, -0.25],
            FixedPartialCorrelations::try_new(vec![0.3]).unwrap(),
            [0.7, -0.4],
        );
        let observation = [0.7, -0.8];
        let (_, gradient) = family.nll_and_gradient_eta(observation, &eta, &mut ());
        let epsilon = 1.0e-6;

        for component in 0..2 {
            let mut plus = eta;
            plus.mu[component] += epsilon;
            let mut minus = eta;
            minus.mu[component] -= epsilon;
            let finite_difference = (family.nll_eta(observation, &plus, &mut ())
                - family.nll_eta(observation, &minus, &mut ()))
                / (2.0 * epsilon);
            assert_relative_eq!(gradient.mu[component], finite_difference, epsilon = 1.0e-6);

            let mut plus = eta;
            plus.kernel_sigma[component] += epsilon;
            let mut minus = eta;
            minus.kernel_sigma[component] -= epsilon;
            let finite_difference = (family.nll_eta(observation, &plus, &mut ())
                - family.nll_eta(observation, &minus, &mut ()))
                / (2.0 * epsilon);
            assert_relative_eq!(
                gradient.kernel_sigma[component],
                finite_difference,
                epsilon = 1.0e-6
            );

            let mut plus = eta;
            plus.shape[component] += epsilon;
            let mut minus = eta;
            minus.shape[component] -= epsilon;
            let finite_difference = (family.nll_eta(observation, &plus, &mut ())
                - family.nll_eta(observation, &minus, &mut ()))
                / (2.0 * epsilon);
            assert_relative_eq!(
                gradient.shape[component],
                finite_difference,
                epsilon = 1.0e-6
            );
        }

        let mut plus = eta;
        *plus.partial_corr.get_mut(1, 0).unwrap() += epsilon;
        let mut minus = eta;
        *minus.partial_corr.get_mut(1, 0).unwrap() -= epsilon;
        let finite_difference = (family.nll_eta(observation, &plus, &mut ())
            - family.nll_eta(observation, &minus, &mut ()))
            / (2.0 * epsilon);
        assert_relative_eq!(
            gradient.partial_corr.get(1, 0).unwrap(),
            finite_difference,
            epsilon = 1.0e-6
        );
    }

    #[test]
    fn static_shape_is_fit_ready() {
        let response = [[0.2, -0.3], [0.8, 0.4], [-0.5, 0.7]];
        let rows = response.len();
        let intercept = || LinearPredictorBlock::new(DenseDesign::intercept(rows));
        let mu = VectorParameterBlock::<Mu, 2, _, _>::new([intercept(), intercept()], NoPenalty, 0);
        let sigma = VectorParameterBlock::<KernelSigma, 2, _, _>::new(
            [intercept(), intercept()],
            NoPenalty,
            0,
        );
        let partial = StrictLowerTriangularParameterBlock::<PartialCorrelation, 2, _, _>::new(
            vec![intercept()],
            NoPenalty,
            0,
        );
        let shape =
            VectorParameterBlock::<Nu, 2, _, _>::new([intercept(), intercept()], NoPenalty, 0);
        let model = Gamlss::try_new_with_observations(
            MvSkewNormalLocationKernelStdPartialCorrDefault::<2>::new(),
            ParameterBlocks::new(((mu, sigma, partial), shape)),
            response.as_slice(),
        )
        .unwrap();
        let beta: [f64; 7] = [0.1, -0.2, 0.0, -0.1, 0.2, 0.5, -0.3];
        let mut gradient: [f64; 7] = [0.0; 7];
        let value = model.try_value_gradient_into(&beta, &mut gradient).unwrap();
        assert!(value.is_finite());
        assert!(gradient.iter().all(|score| score.is_finite()));
        assert_eq!(
            model
                .parameter_layout()
                .unique_slice("kernel_sigma")
                .unwrap(),
            Some(2..4)
        );
        assert_eq!(
            model.parameter_layout().unique_slice("sigma").unwrap(),
            None
        );
    }

    #[cfg(feature = "rand")]
    #[test]
    fn sampling_returns_finite_vectors() {
        use gamlss_core::TrySimulate;
        use rand::SeedableRng;

        let family = MvSkewNormalLocationKernelStdPartialCorrDefault::<2>::new();
        let theta = MvSkewNormalLocationKernelStdPartialCorrTheta::try_new(
            [0.2, -0.1],
            [1.0, 0.8],
            partial_corr(),
            [1.2, -0.7],
        )
        .unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(43);
        assert!(
            family
                .try_sample(&mut rng, &theta)
                .is_ok_and(|sample| sample.iter().all(|value| value.is_finite()))
        );
    }
}
