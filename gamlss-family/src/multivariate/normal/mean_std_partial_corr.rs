#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::suboptimal_flops
)]

use std::marker::PhantomData;

use gamlss_core::{
    CompilableFamily, Family, FixedDimensionalFamily, HasConditionalCdf, HasMarginalCdf,
    HasObservationDimension, HasRosenblattTransform, Identity, InitialEtaFromTheta, Link, Log,
    ModelError, Mu, ObservationView, PartialCorrelation, PositiveLink, Sigma,
    shape::{Product, ShapeValues, StrictLower, Vector},
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};
use gamlss_special::unit_normal_cdf;

use crate::multivariate::{
    correlation::{
        FixedPartialCorrelations, correlation_cholesky_from_partial, covariance_from_cholesky,
        partial_corr_from_eta, partial_corr_gradient_from_cholesky_score,
        scale_cholesky_from_correlation,
    },
    initial,
    matrix::FixedLowerTriangular,
};

use super::kernel;

/// Default-link `D R D` multivariate normal parameterization.
pub type MvNormalMeanStdPartialCorrDefault<const D: usize> =
    MvNormalMeanStdPartialCorr<D, Identity, Log>;

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
impl<Rng, const D: usize, MuLink, SigmaLink> TrySimulate<Rng>
    for MvNormalMeanStdPartialCorr<D, MuLink, SigmaLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    type Sample = [f64; D];

    fn try_sample(
        &self,
        rng: &mut Rng,
        theta: &Self::Theta,
    ) -> Result<Self::Sample, SimulationError> {
        if !valid_theta(theta) {
            return Err(SimulationError::InvalidParameters(
                "MVN mean/SD/partial-correlation theta",
            ));
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
        if out.iter().all(|value| value.is_finite()) {
            Ok(out)
        } else {
            Err(SimulationError::NumericalFailure(
                "MVN mean/SD/partial-correlation transform",
            ))
        }
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
        MvNormalMeanStdPartialCorrEta::new(
            values.0.0,
            values.0.1,
            FixedPartialCorrelations::from_lower_rows(values.1),
        )
    }

    fn gradient_to_shape(gradient: &MvNormalMeanStdPartialCorrEta<D>) -> ShapeValues<Self::Shape> {
        (
            (gradient.mu, gradient.sigma),
            gradient.partial_corr.lower_rows(),
        )
    }

    fn initial_shape<'obs, Obs>(&self, obs: &'obs Obs) -> ShapeValues<Self::Shape>
    where
        Obs: ObservationView<'obs, Observation = [f64; D]> + 'obs,
    {
        let (mu, sigma, partial_corr) =
            initial::location_scale_partial_correlation::<D, MuLink, SigmaLink, Obs>(obs);
        ((mu, sigma), partial_corr)
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
        partial_corr: FixedPartialCorrelations::filled_strict_lower(f64::NAN),
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

    gradient.partial_corr = partial_corr_gradient_from_cholesky_score(
        &eta.partial_corr,
        &theta.correlation_cholesky,
        &correlation_score,
    );

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

    fn assert_parameterization_contract_matches_cholesky<const D: usize>() {
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

        assert_relative_eq!(
            family.nll(observation, &theta, &mut family.workspace()),
            cholesky_family.nll(
                observation,
                &cholesky_theta,
                &mut cholesky_family.workspace(),
            ),
            epsilon = 1.0e-12
        );
        for row in 0..D {
            for col in 0..D {
                assert_relative_eq!(
                    theta.covariance(row, col).unwrap(),
                    cholesky_theta.covariance(row, col).unwrap(),
                    epsilon = 1.0e-12
                );
            }
        }

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

        let mut invalid_observation = observation;
        invalid_observation[0] = f64::NAN;
        let (invalid_nll, invalid_gradient) =
            family.nll_and_gradient_eta(invalid_observation, &eta, &mut family.workspace());
        assert!(invalid_nll.is_infinite());
        assert!(invalid_gradient.mu.iter().all(|value| value.is_nan()));
        assert!(invalid_gradient.sigma.iter().all(|value| value.is_nan()));
        assert!(invalid_gradient.partial_corr.iter().all(f64::is_nan));
        assert!(
            cholesky_family
                .nll(
                    invalid_observation,
                    &cholesky_theta,
                    &mut cholesky_family.workspace(),
                )
                .is_infinite()
        );
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
    fn nll_geometry_invalid_domains_and_capabilities_match_cholesky_for_d1_through_d4() {
        assert_parameterization_contract_matches_cholesky::<1>();
        assert_parameterization_contract_matches_cholesky::<2>();
        assert_parameterization_contract_matches_cholesky::<3>();
        assert_parameterization_contract_matches_cholesky::<4>();
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
        assert_gradient_matches_finite_difference::<4>(
            [1.1, -0.7, 0.2, 0.8],
            &MvNormalMeanStdPartialCorrEta::new(
                [0.4, -0.3, 0.1, 0.25],
                [0.8_f64.ln(), 1.2_f64.ln(), 0.6_f64.ln(), 1.1_f64.ln()],
                FixedPartialCorrelations::try_new(vec![0.2, -0.1, 0.3, 0.15, -0.25, 0.05]).unwrap(),
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
