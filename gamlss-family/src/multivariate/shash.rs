use std::marker::PhantomData;

use gamlss_core::{
    CompilableFamily, Family, FixedDimensionalFamily, HasConditionalCdf, HasMarginalCdf,
    HasObservationDimension, HasRosenblattTransform, Identity, InitialEtaFromTheta, Link, Log,
    ModelError, Mu, Nu, ObservationView, PartialCorrelation, PositiveLink, Sigma, Tau,
    shape::{Product, ShapeValues, StrictLower, Vector},
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};
use gamlss_special::unit_normal_cdf;

use crate::{
    multivariate::{
        correlation::{
            FixedPartialCorrelations, correlation_cholesky_from_partial, covariance_from_cholesky,
            partial_corr_from_eta, partial_corr_gradient_from_cholesky_score,
        },
        initial,
        matrix::FixedLowerTriangular,
        normal::kernel as normal_kernel,
    },
    shash_kernel,
};

/// Default-link multivariate SHASH with per-component `mu/sigma/nu/tau` and latent partial correlations.
pub type MvShashMuSigmaNuTauPartialCorrDefault<const D: usize> =
    MvShashMuSigmaNuTauPartialCorr<D, Identity, Log, Log, Log>;

/// Full-name alias for [`MvShashMuSigmaNuTauPartialCorr`].
pub type MvSinhArcsinhMuSigmaNuTauPartialCorr<
    const D: usize,
    MuLink = Identity,
    SigmaLink = Log,
    NuLink = Log,
    TauLink = Log,
> = MvShashMuSigmaNuTauPartialCorr<D, MuLink, SigmaLink, NuLink, TauLink>;

/// Full-name alias for [`MvShashMuSigmaNuTauPartialCorrDefault`].
pub type MvSinhArcsinhMuSigmaNuTauPartialCorrDefault<const D: usize> =
    MvShashMuSigmaNuTauPartialCorrDefault<D>;

/// Full-name alias for [`MvShashMuSigmaNuTauPartialCorrEta`].
pub type MvSinhArcsinhMuSigmaNuTauPartialCorrEta<const D: usize> =
    MvShashMuSigmaNuTauPartialCorrEta<D>;

/// Full-name alias for [`MvShashMuSigmaNuTauPartialCorrTheta`].
pub type MvSinhArcsinhMuSigmaNuTauPartialCorrTheta<const D: usize> =
    MvShashMuSigmaNuTauPartialCorrTheta<D>;

/// [Jones--Pewsey] multivariate sinh-arcsinh-normal family.
///
/// Each response component is standardized as `x_i = (y_i - mu_i) / sigma_i` and transformed to `z_i = sinh(tau_i * asinh(x_i) - ln(nu_i))`. In the original Jones--Pewsey notation this corresponds to `epsilon_i = -ln(nu_i)` and `delta_i = tau_i`. The latent vector `z` is multivariate standard normal with correlation matrix `R`, represented by ordered partial correlations.
///
/// The partial correlations describe the latent Gaussian dependence and depend on response-coordinate order. Except in the normal case `nu_i = tau_i = 1`, the entries of `R` are not Pearson correlations of the observed response. The componentwise monotone transforms nevertheless preserve the Gaussian copula and the signs of pairwise correlations.
///
/// A separate unconstrained latent Cholesky scale is deliberately absent. The Jones--Pewsey construction starts from a standardized correlated normal vector; allowing non-unit latent marginal variances would no longer preserve the stated SHASH marginals and, in the normal special case, would also confound latent scale with `sigma`. This parameterization therefore uses one response-scale parameter per component and a correlation-only latent Cholesky factor.
///
/// [Jones--Pewsey]: https://doi.org/10.1093/biomet/asp053
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MvShashMuSigmaNuTauPartialCorr<
    const D: usize,
    MuLink = Identity,
    SigmaLink = Log,
    NuLink = Log,
    TauLink = Log,
> {
    marker: PhantomData<(MuLink, SigmaLink, NuLink, TauLink)>,
}

impl<const D: usize, MuLink, SigmaLink, NuLink, TauLink>
    MvShashMuSigmaNuTauPartialCorr<D, MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    /// Creates a stateless family after checking the compile-time dimension.
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

    /// Creates a stateless family.
    ///
    /// # Panics
    ///
    /// Panics when `D == 0`.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        assert!(D > 0, "multivariate SHASH dimension must be positive");
        Self {
            marker: PhantomData,
        }
    }

    fn theta_from_eta(
        eta: &MvShashMuSigmaNuTauPartialCorrEta<D>,
    ) -> MvShashMuSigmaNuTauPartialCorrTheta<D> {
        let mu = std::array::from_fn(|index| MuLink::inverse(eta.mu[index]));
        let sigma = std::array::from_fn(|index| SigmaLink::inverse(eta.sigma[index]));
        let nu = std::array::from_fn(|index| NuLink::inverse(eta.nu[index]));
        let tau = std::array::from_fn(|index| TauLink::inverse(eta.tau[index]));
        let partial_corr = partial_corr_from_eta(&eta.partial_corr);
        let correlation_cholesky = correlation_cholesky_from_partial(&partial_corr);
        MvShashMuSigmaNuTauPartialCorrTheta::from_canonical_parts(
            mu,
            sigma,
            nu,
            tau,
            partial_corr,
            correlation_cholesky,
        )
    }

    fn transform_observation(
        observation: &[f64; D],
        theta: &MvShashMuSigmaNuTauPartialCorrTheta<D>,
    ) -> Option<TransformedObservation<D>> {
        if !valid_theta(theta) || observation.iter().any(|value| !value.is_finite()) {
            return None;
        }

        let mut standardized = [0.0; D];
        let mut asinh_standardized = [0.0; D];
        let mut latent = [0.0; D];
        let mut negative_log_jacobian = 0.0;
        for component in 0..D {
            standardized[component] =
                (observation[component] - theta.mu[component]) / theta.sigma[component];
            let transformed = shash_kernel::transform_standardized(
                standardized[component],
                theta.nu[component],
                theta.tau[component],
            );
            asinh_standardized[component] = transformed.asinh_x;
            latent[component] = transformed.latent;
            if !standardized[component].is_finite()
                || !asinh_standardized[component].is_finite()
                || !transformed.h.is_finite()
                || !latent[component].is_finite()
            {
                return None;
            }
            negative_log_jacobian += theta.sigma[component].ln() - theta.tau[component].ln()
                + standardized[component].hypot(1.0).ln()
                - shash_kernel::log_cosh(transformed.h);
        }
        negative_log_jacobian
            .is_finite()
            .then_some(TransformedObservation {
                standardized,
                asinh_standardized,
                latent,
                negative_log_jacobian,
            })
    }

    fn nll_theta(observation: [f64; D], theta: &MvShashMuSigmaNuTauPartialCorrTheta<D>) -> f64 {
        let Some(transformed) = Self::transform_observation(&observation, theta) else {
            return f64::INFINITY;
        };

        let mut whitened = [0.0; D];
        let zero = [0.0; D];
        let nll = normal_kernel::nll(
            D,
            &transformed.latent,
            &zero,
            &theta.correlation_cholesky,
            &mut whitened,
        ) + transformed.negative_log_jacobian;
        if nll.is_finite() { nll } else { f64::INFINITY }
    }

    fn nll_and_gradient_eta_values(
        observation: [f64; D],
        eta: &MvShashMuSigmaNuTauPartialCorrEta<D>,
    ) -> (f64, MvShashMuSigmaNuTauPartialCorrEta<D>) {
        let theta = Self::theta_from_eta(eta);
        let Some(transformed) = Self::transform_observation(&observation, &theta) else {
            return (f64::INFINITY, nan_eta());
        };

        let zero = [0.0; D];
        let mut whitened = [0.0; D];
        let mut latent_score = [0.0; D];
        let normal_nll = normal_kernel::nll_and_score(
            D,
            &transformed.latent,
            &zero,
            &theta.correlation_cholesky,
            &mut whitened,
            &mut latent_score,
        );
        let nll = normal_nll + transformed.negative_log_jacobian;
        if !nll.is_finite() {
            return (f64::INFINITY, nan_eta());
        }

        let mut gradient = zero_eta();
        for (component, latent_derivative) in latent_score.iter().copied().enumerate() {
            let cosh_h = transformed.latent[component].hypot(1.0);
            let d_h = latent_derivative.mul_add(cosh_h, -transformed.latent[component] / cosh_h);
            let inverse_hypot = 1.0 / transformed.standardized[component].hypot(1.0);
            let d_x = (d_h * theta.tau[component]).mul_add(
                inverse_hypot,
                (transformed.standardized[component] * inverse_hypot) * inverse_hypot,
            );
            gradient.mu[component] =
                -d_x / theta.sigma[component] * MuLink::derivative_inverse(eta.mu[component]);
            gradient.sigma[component] = transformed.standardized[component].mul_add(-d_x, 1.0)
                * SigmaLink::derivative_log_inverse(eta.sigma[component]);
            gradient.nu[component] = -d_h * NuLink::derivative_log_inverse(eta.nu[component]);
            gradient.tau[component] = (d_h * theta.tau[component])
                .mul_add(transformed.asinh_standardized[component], -1.0)
                * TauLink::derivative_log_inverse(eta.tau[component]);
        }

        let mut correlation_score = [[0.0; D]; D];
        for (row, score_row) in correlation_score.iter_mut().enumerate() {
            for (col, score) in score_row.iter_mut().enumerate().take(row + 1) {
                *score = normal_kernel::cholesky_score(
                    row,
                    col,
                    whitened[col],
                    &latent_score,
                    &theta.correlation_cholesky,
                );
            }
        }
        gradient.partial_corr = partial_corr_gradient_from_cholesky_score(
            &eta.partial_corr,
            &theta.correlation_cholesky,
            &correlation_score,
        );

        if valid_gradient(&gradient) {
            (nll, gradient)
        } else {
            (nll, nan_eta())
        }
    }
}

impl<const D: usize, MuLink, SigmaLink, NuLink, TauLink> Default
    for MvShashMuSigmaNuTauPartialCorr<D, MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize, MuLink, SigmaLink, NuLink, TauLink> Family
    for MvShashMuSigmaNuTauPartialCorr<D, MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    type Eta = MvShashMuSigmaNuTauPartialCorrEta<D>;
    type Theta = MvShashMuSigmaNuTauPartialCorrTheta<D>;
    type GradientEta = MvShashMuSigmaNuTauPartialCorrEta<D>;
    type Observation<'obs> = [f64; D];
    type Workspace = ();

    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    fn nll(
        &self,
        observation: Self::Observation<'_>,
        theta: &Self::Theta,
        _workspace: &mut Self::Workspace,
    ) -> f64 {
        Self::nll_theta(observation, theta)
    }

    fn nll_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> f64 {
        Self::nll_theta(observation, &Self::theta_from_eta(eta))
    }

    fn nll_and_gradient_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        Self::nll_and_gradient_eta_values(observation, eta)
    }
}

impl<const D: usize, MuLink, SigmaLink, NuLink, TauLink> FixedDimensionalFamily<D>
    for MvShashMuSigmaNuTauPartialCorr<D, MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
}

impl<const D: usize, MuLink, SigmaLink, NuLink, TauLink> HasObservationDimension
    for MvShashMuSigmaNuTauPartialCorr<D, MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    fn observation_dimension(&self) -> usize {
        D
    }
}

impl<const D: usize, MuLink, SigmaLink, NuLink, TauLink> HasMarginalCdf
    for MvShashMuSigmaNuTauPartialCorr<D, MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    fn marginal_cdf(&self, component: usize, y: f64, theta: &Self::Theta) -> f64 {
        if component >= D || !y.is_finite() || !valid_theta(theta) {
            return f64::NAN;
        }
        let x = (y - theta.mu[component]) / theta.sigma[component];
        let transformed =
            shash_kernel::transform_standardized(x, theta.nu[component], theta.tau[component]);
        unit_normal_cdf(transformed.latent)
    }
}

impl<const D: usize, MuLink, SigmaLink, NuLink, TauLink> HasConditionalCdf
    for MvShashMuSigmaNuTauPartialCorr<D, MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    fn conditional_cdf(
        &self,
        component: usize,
        y: f64,
        preceding: &[f64],
        theta: &Self::Theta,
    ) -> f64 {
        if component >= D || preceding.len() < component || !y.is_finite() || !valid_theta(theta) {
            return f64::NAN;
        }

        let current_x = (y - theta.mu[component]) / theta.sigma[component];
        let current = shash_kernel::transform_standardized(
            current_x,
            theta.nu[component],
            theta.tau[component],
        )
        .latent;
        if current == f64::NEG_INFINITY {
            return 0.0;
        }
        if current == f64::INFINITY {
            return 1.0;
        }
        if !current.is_finite() {
            return f64::NAN;
        }

        let mut latent_preceding = [0.0; D];
        for index in 0..component {
            if !preceding[index].is_finite() {
                return f64::NAN;
            }
            let x = (preceding[index] - theta.mu[index]) / theta.sigma[index];
            latent_preceding[index] =
                shash_kernel::transform_standardized(x, theta.nu[index], theta.tau[index]).latent;
            if !latent_preceding[index].is_finite() {
                return f64::NAN;
            }
        }

        let zero = [0.0; D];
        let mut standardized = [0.0; D];
        normal_kernel::conditional_cdf(
            D,
            component,
            current,
            &latent_preceding,
            &zero,
            &theta.correlation_cholesky,
            &mut standardized,
        )
    }
}

impl<const D: usize, MuLink, SigmaLink, NuLink, TauLink> HasRosenblattTransform
    for MvShashMuSigmaNuTauPartialCorr<D, MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
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
        if !valid_theta(theta) || observation.iter().any(|value| !value.is_finite()) {
            out.fill(f64::NAN);
            return Ok(());
        }

        let mut latent = [0.0; D];
        for component in 0..D {
            let x = (observation[component] - theta.mu[component]) / theta.sigma[component];
            latent[component] =
                shash_kernel::transform_standardized(x, theta.nu[component], theta.tau[component])
                    .latent;
        }
        let zero = [0.0; D];
        normal_kernel::rosenblatt_into(D, &latent, &zero, &theta.correlation_cholesky, out);
        Ok(())
    }
}

#[cfg(feature = "rand")]
impl<Rng, const D: usize, MuLink, SigmaLink, NuLink, TauLink> TrySimulate<Rng>
    for MvShashMuSigmaNuTauPartialCorr<D, MuLink, SigmaLink, NuLink, TauLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    NuLink: PositiveLink<f64>,
    TauLink: PositiveLink<f64>,
{
    type Sample = [f64; D];

    fn try_sample(
        &self,
        rng: &mut Rng,
        theta: &Self::Theta,
    ) -> Result<Self::Sample, SimulationError> {
        if !valid_theta(theta) {
            return Err(SimulationError::InvalidParameters(
                "multivariate SHASH theta",
            ));
        }

        let standard = rand_distr::StandardNormal;
        let independent: [f64; D] =
            std::array::from_fn(|_| rand_distr::Distribution::sample(&standard, rng));
        let mut sample = [0.0; D];
        for (row, sample_row) in sample.iter_mut().enumerate() {
            let latent = (0..=row).fold(0.0, |value, col| {
                theta
                    .correlation_cholesky
                    .lower(row, col)
                    .mul_add(independent[col], value)
            });
            let standardized =
                shash_kernel::inverse_standardized(latent, theta.nu[row], theta.tau[row]);
            *sample_row = theta.sigma[row].mul_add(standardized, theta.mu[row]);
        }
        if sample.iter().all(|value| value.is_finite()) {
            Ok(sample)
        } else {
            Err(SimulationError::NumericalFailure(
                "multivariate SHASH inverse transform",
            ))
        }
    }
}

impl<const D: usize, MuLink, SigmaLink, NuLink, TauLink> CompilableFamily
    for MvShashMuSigmaNuTauPartialCorr<D, MuLink, SigmaLink, NuLink, TauLink>
where
    MuLink: InitialEtaFromTheta<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    NuLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    TauLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Shape = Product<
        Product<Vector<Mu, D>, Vector<Sigma, D>>,
        Product<Product<Vector<Nu, D>, Vector<Tau, D>>, StrictLower<PartialCorrelation, D>>,
    >;

    fn eta_from_shape(values: ShapeValues<Self::Shape>) -> Self::Eta {
        MvShashMuSigmaNuTauPartialCorrEta::new(
            values.0.0,
            values.0.1,
            values.1.0.0,
            values.1.0.1,
            FixedPartialCorrelations::from_lower_rows(values.1.1),
        )
    }

    fn gradient_to_shape(gradient: &Self::GradientEta) -> ShapeValues<Self::Shape> {
        (
            (gradient.mu, gradient.sigma),
            (
                (gradient.nu, gradient.tau),
                gradient.partial_corr.lower_rows(),
            ),
        )
    }

    fn initial_shape<'obs, Obs>(&self, obs: &'obs Obs) -> ShapeValues<Self::Shape>
    where
        Obs: ObservationView<'obs, Observation = [f64; D]> + 'obs,
    {
        let (mu, sigma, partial_corr) =
            initial::location_scale_partial_correlation::<D, MuLink, SigmaLink, Obs>(obs);
        (
            (mu, sigma),
            (
                (
                    [NuLink::initial_eta_from_theta(1.0); D],
                    [TauLink::initial_eta_from_theta(1.0); D],
                ),
                partial_corr,
            ),
        )
    }

    fn validate_compiled(&self) -> Result<(), ModelError> {
        Self::try_new().map(|_| ())
    }
}

struct TransformedObservation<const D: usize> {
    standardized: [f64; D],
    asinh_standardized: [f64; D],
    latent: [f64; D],
    negative_log_jacobian: f64,
}

/// Link-scale predictors for multivariate SHASH.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MvShashMuSigmaNuTauPartialCorrEta<const D: usize> {
    /// Per-component location predictors.
    pub mu: [f64; D],
    /// Per-component scale predictors; the links map them to positive natural parameters.
    pub sigma: [f64; D],
    /// Per-component skew-ratio predictors; with the default log link these are signed skewness predictors.
    pub nu: [f64; D],
    /// Per-component tail-parameter predictors; the links map them to positive natural parameters.
    pub tau: [f64; D],
    /// Strict-lower latent partial-correlation predictors.
    pub partial_corr: FixedPartialCorrelations<D>,
}

impl<const D: usize> MvShashMuSigmaNuTauPartialCorrEta<D> {
    /// Creates link-scale predictors.
    #[must_use]
    pub const fn new(
        mu: [f64; D],
        sigma: [f64; D],
        nu: [f64; D],
        tau: [f64; D],
        partial_corr: FixedPartialCorrelations<D>,
    ) -> Self {
        Self {
            mu,
            sigma,
            nu,
            tau,
            partial_corr,
        }
    }
}

/// Natural-scale parameters for multivariate SHASH.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MvShashMuSigmaNuTauPartialCorrTheta<const D: usize> {
    mu: [f64; D],
    sigma: [f64; D],
    nu: [f64; D],
    tau: [f64; D],
    partial_corr: FixedPartialCorrelations<D>,
    correlation_cholesky: FixedLowerTriangular<D>,
}

impl<const D: usize> MvShashMuSigmaNuTauPartialCorrTheta<D> {
    /// Creates checked natural-scale parameters and derives the latent correlation Cholesky factor.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] for zero dimension, non-finite parameters, non-positive `sigma/nu/tau`, invalid partial correlations, or unrepresentable correlation geometry.
    pub fn try_new(
        mu: [f64; D],
        sigma: [f64; D],
        nu: [f64; D],
        tau: [f64; D],
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
        if !nu.iter().all(|value| value.is_finite() && *value > 0.0) {
            return Err(ModelError::InvalidParameter {
                parameter: "nu",
                expected: "finite and > 0",
            });
        }
        if !tau.iter().all(|value| value.is_finite() && *value > 0.0) {
            return Err(ModelError::InvalidParameter {
                parameter: "tau",
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
        let theta =
            Self::from_canonical_parts(mu, sigma, nu, tau, partial_corr, correlation_cholesky);
        if !valid_theta(&theta) {
            return Err(ModelError::InvalidParameter {
                parameter: "partial_corr",
                expected: "a representable positive-definite latent correlation geometry",
            });
        }
        Ok(theta)
    }

    const fn from_canonical_parts(
        mu: [f64; D],
        sigma: [f64; D],
        nu: [f64; D],
        tau: [f64; D],
        partial_corr: FixedPartialCorrelations<D>,
        correlation_cholesky: FixedLowerTriangular<D>,
    ) -> Self {
        Self {
            mu,
            sigma,
            nu,
            tau,
            partial_corr,
            correlation_cholesky,
        }
    }

    /// Per-component transformation locations.
    #[must_use]
    pub const fn mu(&self) -> &[f64; D] {
        &self.mu
    }

    /// Per-component positive transformation scales.
    #[must_use]
    pub const fn sigma(&self) -> &[f64; D] {
        &self.sigma
    }

    /// Per-component positive skew ratios.
    #[must_use]
    pub const fn nu(&self) -> &[f64; D] {
        &self.nu
    }

    /// Per-component positive tail parameters.
    #[must_use]
    pub const fn tau(&self) -> &[f64; D] {
        &self.tau
    }

    /// Canonical strict-lower partial correlations of the latent Gaussian vector.
    #[must_use]
    pub const fn partial_corr(&self) -> &FixedPartialCorrelations<D> {
        &self.partial_corr
    }

    /// Cholesky factor of the latent Gaussian correlation matrix.
    #[must_use]
    pub const fn correlation_cholesky(&self) -> &FixedLowerTriangular<D> {
        &self.correlation_cholesky
    }

    /// Returns one latent Gaussian correlation entry.
    ///
    /// This is generally not the Pearson correlation of the observed SHASH response.
    #[must_use]
    pub fn latent_correlation(&self, row: usize, col: usize) -> Option<f64> {
        covariance_from_cholesky(&self.correlation_cholesky, row, col)
    }
}

fn valid_theta<const D: usize>(theta: &MvShashMuSigmaNuTauPartialCorrTheta<D>) -> bool {
    D > 0
        && theta.mu.iter().all(|value| value.is_finite())
        && theta
            .sigma
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        && theta
            .nu
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        && theta
            .tau
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        && theta
            .partial_corr
            .iter()
            .all(|value| value.is_finite() && value.abs() < 1.0)
        && normal_kernel::valid_theta(D, &[0.0; D], &theta.correlation_cholesky)
}

fn zero_eta<const D: usize>() -> MvShashMuSigmaNuTauPartialCorrEta<D> {
    MvShashMuSigmaNuTauPartialCorrEta {
        mu: [0.0; D],
        sigma: [0.0; D],
        nu: [0.0; D],
        tau: [0.0; D],
        partial_corr: FixedPartialCorrelations::zeros(),
    }
}

fn nan_eta<const D: usize>() -> MvShashMuSigmaNuTauPartialCorrEta<D> {
    MvShashMuSigmaNuTauPartialCorrEta {
        mu: [f64::NAN; D],
        sigma: [f64::NAN; D],
        nu: [f64::NAN; D],
        tau: [f64::NAN; D],
        partial_corr: FixedPartialCorrelations::filled_strict_lower(f64::NAN),
    }
}

fn valid_gradient<const D: usize>(gradient: &MvShashMuSigmaNuTauPartialCorrEta<D>) -> bool {
    gradient.mu.iter().all(|value| value.is_finite())
        && gradient.sigma.iter().all(|value| value.is_finite())
        && gradient.nu.iter().all(|value| value.is_finite())
        && gradient.tau.iter().all(|value| value.is_finite())
        && gradient.partial_corr.iter().all(f64::is_finite)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    #[cfg(feature = "rand")]
    use gamlss_core::TrySimulate;
    use gamlss_core::{
        DenseDesign, Family, Gamlss, HasCdf, HasConditionalCdf, HasMarginalCdf,
        HasRosenblattTransform, LinearPredictorBlock, Mu, NoPenalty, Nu, ParameterBlocks,
        PartialCorrelation, Sigma, StrictLowerTriangularParameterBlock, Tau, VectorParameterBlock,
    };

    use super::{
        FixedPartialCorrelations, MvShashMuSigmaNuTauPartialCorrDefault,
        MvShashMuSigmaNuTauPartialCorrEta, MvShashMuSigmaNuTauPartialCorrTheta, valid_gradient,
    };
    use crate::{
        ShashMuSigmaNuTau, ShashTheta,
        multivariate::normal::{
            MvNormalMeanStdPartialCorrDefault, MvNormalMeanStdPartialCorrTheta,
        },
    };

    fn representative_eta() -> MvShashMuSigmaNuTauPartialCorrEta<2> {
        let mut partial_corr = FixedPartialCorrelations::zeros();
        *partial_corr.get_mut(1, 0).unwrap() = 0.35;
        MvShashMuSigmaNuTauPartialCorrEta::new(
            [0.2, -0.3],
            [1.1_f64.ln(), 0.8_f64.ln()],
            [1.3_f64.ln(), 0.7_f64.ln()],
            [0.9_f64.ln(), 1.2_f64.ln()],
            partial_corr,
        )
    }

    fn assert_gradient_matches_finite_difference<const D: usize>(
        observation: [f64; D],
        eta: MvShashMuSigmaNuTauPartialCorrEta<D>,
    ) {
        let family = MvShashMuSigmaNuTauPartialCorrDefault::<D>::new();
        let (_, gradient) = family.nll_and_gradient_eta(observation, &eta, &mut family.workspace());
        let epsilon = 1.0e-6;

        for component in 0..D {
            for parameter in 0..4 {
                let mut plus = eta;
                let mut minus = eta;
                let plus_value = match parameter {
                    0 => &mut plus.mu[component],
                    1 => &mut plus.sigma[component],
                    2 => &mut plus.nu[component],
                    _ => &mut plus.tau[component],
                };
                *plus_value += epsilon;
                let minus_value = match parameter {
                    0 => &mut minus.mu[component],
                    1 => &mut minus.sigma[component],
                    2 => &mut minus.nu[component],
                    _ => &mut minus.tau[component],
                };
                *minus_value -= epsilon;
                let finite_difference =
                    (family.nll_eta(observation, &plus, &mut family.workspace())
                        - family.nll_eta(observation, &minus, &mut family.workspace()))
                        / (2.0 * epsilon);
                let actual = match parameter {
                    0 => gradient.mu[component],
                    1 => gradient.sigma[component],
                    2 => gradient.nu[component],
                    _ => gradient.tau[component],
                };
                assert_relative_eq!(actual, finite_difference, epsilon = 5.0e-6);
            }
        }

        for row in 1..D {
            for col in 0..row {
                let mut plus = eta;
                let mut minus = eta;
                *plus.partial_corr.get_mut(row, col).unwrap() += epsilon;
                *minus.partial_corr.get_mut(row, col).unwrap() -= epsilon;
                let finite_difference =
                    (family.nll_eta(observation, &plus, &mut family.workspace())
                        - family.nll_eta(observation, &minus, &mut family.workspace()))
                        / (2.0 * epsilon);
                assert_relative_eq!(
                    gradient.partial_corr.get(row, col).unwrap(),
                    finite_difference,
                    epsilon = 5.0e-6
                );
            }
        }
    }

    #[test]
    fn one_dimension_matches_scalar_shash() {
        let family = MvShashMuSigmaNuTauPartialCorrDefault::<1>::new();
        let theta = MvShashMuSigmaNuTauPartialCorrTheta::try_new(
            [0.2],
            [1.3],
            [1.5],
            [0.8],
            FixedPartialCorrelations::zeros(),
        )
        .unwrap();
        let scalar = ShashMuSigmaNuTau::new();
        let scalar_theta = ShashTheta {
            mu: 0.2,
            sigma: 1.3,
            nu: 1.5,
            tau: 0.8,
        };

        for y in [-2.0, 0.1, 1.7] {
            assert_relative_eq!(
                family.nll([y], &theta, &mut family.workspace()),
                scalar.nll(y, &scalar_theta, &mut scalar.workspace()),
                epsilon = 1.0e-12
            );
            assert_relative_eq!(
                family.marginal_cdf(0, y, &theta),
                scalar.cdf(y, &scalar_theta),
                epsilon = 1.0e-12
            );
        }
    }

    #[test]
    fn bivariate_density_and_diagnostics_match_independent_reference_values() {
        let family = MvShashMuSigmaNuTauPartialCorrDefault::<2>::new();
        let theta = MvShashMuSigmaNuTauPartialCorrTheta::try_new(
            [0.2, -0.3],
            [1.1, 0.8],
            [1.3, 0.7],
            [0.9, 1.2],
            FixedPartialCorrelations::try_new(vec![0.35]).unwrap(),
        )
        .unwrap();
        let observation = [0.6, -1.1];

        assert_relative_eq!(
            family.nll(observation, &theta, &mut family.workspace()),
            2.095_356_456_435_074_3,
            epsilon = 2.0e-15
        );
        assert_relative_eq!(
            family.marginal_cdf(0, observation[0], &theta),
            0.523_176_144_604_723,
            epsilon = 2.0e-15
        );
        assert_relative_eq!(
            family.marginal_cdf(1, observation[1], &theta),
            0.223_685_339_745_502_42,
            epsilon = 2.0e-15
        );
        assert_relative_eq!(
            family.conditional_cdf(1, observation[1], &observation[..1], &theta),
            0.202_471_244_479_011_05,
            epsilon = 2.0e-15
        );

        let mut rosenblatt = [0.0; 2];
        family
            .rosenblatt_into(observation, &theta, &mut rosenblatt)
            .unwrap();
        assert_relative_eq!(rosenblatt[0], 0.523_176_144_604_723, epsilon = 2.0e-15);
        assert_relative_eq!(rosenblatt[1], 0.202_471_244_479_011_05, epsilon = 2.0e-15);
        assert_relative_eq!(theta.latent_correlation(1, 0).unwrap(), 0.35);
    }

    #[test]
    fn normal_shape_case_matches_multivariate_normal() {
        let family = MvShashMuSigmaNuTauPartialCorrDefault::<2>::new();
        let mut partial_corr = FixedPartialCorrelations::zeros();
        *partial_corr.get_mut(1, 0).unwrap() = -0.4;
        let theta = MvShashMuSigmaNuTauPartialCorrTheta::try_new(
            [0.2, -0.1],
            [1.3, 0.7],
            [1.0; 2],
            [1.0; 2],
            partial_corr,
        )
        .unwrap();
        let normal = MvNormalMeanStdPartialCorrDefault::<2>::new();
        let normal_theta =
            MvNormalMeanStdPartialCorrTheta::try_new([0.2, -0.1], [1.3, 0.7], partial_corr)
                .unwrap();
        let observation = [0.5, -0.8];

        assert_relative_eq!(
            family.nll(observation, &theta, &mut family.workspace()),
            normal.nll(observation, &normal_theta, &mut normal.workspace()),
            epsilon = 1.0e-12
        );
        for component in 0..2 {
            assert_relative_eq!(
                family.marginal_cdf(component, observation[component], &theta),
                normal.marginal_cdf(component, observation[component], &normal_theta),
                epsilon = 1.0e-12
            );
            assert_relative_eq!(
                family.conditional_cdf(component, observation[component], &observation, &theta),
                normal.conditional_cdf(
                    component,
                    observation[component],
                    &observation,
                    &normal_theta,
                ),
                epsilon = 1.0e-12
            );
        }

        let mut actual = [0.0; 2];
        let mut expected = [0.0; 2];
        family
            .rosenblatt_into(observation, &theta, &mut actual)
            .unwrap();
        normal
            .rosenblatt_into(observation, &normal_theta, &mut expected)
            .unwrap();
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert_relative_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn zero_latent_correlation_factorizes_into_scalar_shash_marginals() {
        let family = MvShashMuSigmaNuTauPartialCorrDefault::<2>::new();
        let theta = MvShashMuSigmaNuTauPartialCorrTheta::try_new(
            [0.2, -0.3],
            [1.1, 0.8],
            [1.3, 0.7],
            [0.9, 1.2],
            FixedPartialCorrelations::zeros(),
        )
        .unwrap();
        let scalar = ShashMuSigmaNuTau::new();
        let observation = [0.6, -1.1];
        let expected = (0..2)
            .map(|component| {
                scalar.nll(
                    observation[component],
                    &ShashTheta {
                        mu: theta.mu()[component],
                        sigma: theta.sigma()[component],
                        nu: theta.nu()[component],
                        tau: theta.tau()[component],
                    },
                    &mut scalar.workspace(),
                )
            })
            .sum::<f64>();

        assert_relative_eq!(
            family.nll(observation, &theta, &mut family.workspace()),
            expected,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn constructors_reject_invalid_dimension_and_natural_parameters() {
        assert!(MvShashMuSigmaNuTauPartialCorrDefault::<0>::try_new().is_err());
        assert!(
            MvShashMuSigmaNuTauPartialCorrTheta::<0>::try_new(
                [],
                [],
                [],
                [],
                FixedPartialCorrelations::zeros(),
            )
            .is_err()
        );
        for (sigma, nu, tau) in [
            ([0.0, 1.0], [1.0; 2], [1.0; 2]),
            ([1.0; 2], [f64::NAN, 1.0], [1.0; 2]),
            ([1.0; 2], [1.0; 2], [-0.1, 1.0]),
        ] {
            assert!(
                MvShashMuSigmaNuTauPartialCorrTheta::try_new(
                    [0.0; 2],
                    sigma,
                    nu,
                    tau,
                    FixedPartialCorrelations::zeros(),
                )
                .is_err()
            );
        }
        assert!(
            MvShashMuSigmaNuTauPartialCorrTheta::try_new(
                [0.0; 2],
                [1.0; 2],
                [1.0; 2],
                [1.0; 2],
                FixedPartialCorrelations::try_new(vec![1.0]).unwrap(),
            )
            .is_err()
        );
    }

    #[test]
    fn analytic_gradient_matches_finite_difference() {
        assert_gradient_matches_finite_difference([0.6, -1.1], representative_eta());
        assert_gradient_matches_finite_difference(
            [1.1, -0.7, 0.2],
            MvShashMuSigmaNuTauPartialCorrEta::new(
                [0.4, -0.3, 0.1],
                [0.8_f64.ln(), 1.2_f64.ln(), 0.6_f64.ln()],
                [1.3_f64.ln(), 0.7_f64.ln(), 1.1_f64.ln()],
                [0.9_f64.ln(), 1.2_f64.ln(), 0.75_f64.ln()],
                FixedPartialCorrelations::try_new(vec![0.2, -0.1, 0.3]).unwrap(),
            ),
        );
    }

    #[test]
    fn log_link_gradient_remains_finite_for_subnormal_scale_and_tailweight() {
        let family = MvShashMuSigmaNuTauPartialCorrDefault::<2>::new();
        let eta = MvShashMuSigmaNuTauPartialCorrEta::new(
            [0.0; 2],
            [-744.0, 0.0],
            [0.0; 2],
            [0.0, -744.0],
            FixedPartialCorrelations::zeros(),
        );
        let (nll, gradient) = family.nll_and_gradient_eta([0.0; 2], &eta, &mut family.workspace());

        assert!(nll.is_finite());
        assert!(valid_gradient(&gradient));
        assert_relative_eq!(gradient.sigma[0], 1.0, epsilon = f64::EPSILON);
        assert_relative_eq!(gradient.tau[1], -1.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn invalid_observations_and_diagnostic_requests_follow_family_contracts() {
        let family = MvShashMuSigmaNuTauPartialCorrDefault::<2>::new();
        let eta = representative_eta();
        let theta = family.theta(&eta, &mut family.workspace());
        let (nll, gradient) =
            family.nll_and_gradient_eta([f64::NAN, 0.0], &eta, &mut family.workspace());

        assert!(nll.is_infinite() && nll.is_sign_positive());
        assert!(gradient.mu.iter().all(|value| value.is_nan()));
        assert!(family.marginal_cdf(2, 0.0, &theta).is_nan());
        assert!(family.marginal_cdf(0, f64::NAN, &theta).is_nan());
        assert!(family.conditional_cdf(1, 0.0, &[], &theta).is_nan());
        assert!(family.conditional_cdf(2, 0.0, &[0.0; 2], &theta).is_nan());
        assert!(
            family
                .rosenblatt_into([0.0; 2], &theta, &mut [0.0; 1])
                .is_err()
        );
        let mut invalid = [0.0; 2];
        family
            .rosenblatt_into([f64::NAN, 0.0], &theta, &mut invalid)
            .unwrap();
        assert!(invalid.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn extreme_finite_partial_correlation_eta_stays_valid_and_finite() {
        let family = MvShashMuSigmaNuTauPartialCorrDefault::<2>::new();
        for eta_value in [-100.0, -20.0, 20.0, 100.0] {
            let eta = MvShashMuSigmaNuTauPartialCorrEta::new(
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                FixedPartialCorrelations::try_new(vec![eta_value]).unwrap(),
            );
            let theta = family.theta(&eta, &mut family.workspace());
            let (nll, gradient) =
                family.nll_and_gradient_eta([0.1, -0.2], &eta, &mut family.workspace());

            assert!(theta.partial_corr().get(1, 0).unwrap().abs() < 1.0);
            assert!(theta.correlation_cholesky().get(1, 1).unwrap() > 0.0);
            assert!(nll.is_finite());
            assert!(gradient.partial_corr.get(1, 0).unwrap().is_finite());
        }
    }

    #[test]
    fn compiled_blocks_are_fit_ready() {
        let observations = [[0.2, -0.3], [1.0, 0.4], [-0.5, 0.8]];
        let n = observations.len();
        let vector_block = || {
            [
                LinearPredictorBlock::new(DenseDesign::intercept(n)),
                LinearPredictorBlock::new(DenseDesign::intercept(n)),
            ]
        };
        let mu = VectorParameterBlock::<Mu, 2, _, _>::new(vector_block(), NoPenalty, 99);
        let sigma = VectorParameterBlock::<Sigma, 2, _, _>::new(vector_block(), NoPenalty, 99);
        let nu = VectorParameterBlock::<Nu, 2, _, _>::new(vector_block(), NoPenalty, 99);
        let tau = VectorParameterBlock::<Tau, 2, _, _>::new(vector_block(), NoPenalty, 99);
        let partial_corr = StrictLowerTriangularParameterBlock::<PartialCorrelation, 2, _, _>::new(
            vec![LinearPredictorBlock::new(DenseDesign::intercept(n))],
            NoPenalty,
            99,
        );
        let model = Gamlss::try_new_with_observations(
            MvShashMuSigmaNuTauPartialCorrDefault::<2>::new(),
            ParameterBlocks::new(((mu, sigma), ((nu, tau), partial_corr))),
            observations.as_slice(),
        )
        .unwrap();
        let beta = vec![0.1, -0.2, 0.0, 0.3, 0.15, -0.1, -0.2, 0.25, 0.1];
        let eta = model.predict_eta_row(&beta, 0).unwrap();
        let initial = model.initial_parameters().unwrap();

        assert_eq!(model.nparams(), 9);
        assert!(initial.iter().all(|value| value.is_finite()));
        assert!(model.try_value(&initial).unwrap().is_finite());
        assert_relative_eq!(eta.mu[0], 0.1);
        assert_relative_eq!(eta.sigma[1], 0.3);
        assert_relative_eq!(eta.nu[0], 0.15);
        assert_relative_eq!(eta.tau[1], 0.25);
        assert_relative_eq!(eta.partial_corr.get(1, 0).unwrap(), 0.1);

        let mut gradient = vec![0.0; beta.len()];
        model.try_value_gradient_into(&beta, &mut gradient).unwrap();
        for index in 0..beta.len() {
            let mut plus = beta.clone();
            plus[index] += 1.0e-6;
            let mut minus = beta.clone();
            minus[index] -= 1.0e-6;
            let finite_difference =
                (model.try_value(&plus).unwrap() - model.try_value(&minus).unwrap()) / 2.0e-6;
            assert_relative_eq!(gradient[index], finite_difference, epsilon = 2.0e-6);
        }
    }

    #[cfg(feature = "rand")]
    #[test]
    fn sampling_returns_finite_vectors_and_rejects_invalid_theta() {
        use rand::SeedableRng;

        let family = MvShashMuSigmaNuTauPartialCorrDefault::<2>::new();
        let eta = representative_eta();
        let theta = family.theta(&eta, &mut family.workspace());
        let mut rng = rand::rngs::StdRng::seed_from_u64(41);
        assert!(
            family
                .try_sample(&mut rng, &theta)
                .is_ok_and(|sample| sample.iter().all(|value| value.is_finite()))
        );

        let partial_corr = FixedPartialCorrelations::zeros();
        let invalid = MvShashMuSigmaNuTauPartialCorrTheta::from_canonical_parts(
            [0.0; 2],
            [1.0, 0.0],
            [1.0; 2],
            [1.0; 2],
            partial_corr,
            super::correlation_cholesky_from_partial(&partial_corr),
        );
        assert!(family.try_sample(&mut rng, &invalid).is_err());
    }
}
