#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::suboptimal_flops
)]

use std::marker::PhantomData;

use gamlss_core::{
    AboveTwoLink, CompilableFamily, Family, FixedDimensionalFamily, HasMarginalCdf,
    HasObservationDimension, Identity, InitialEtaFromTheta, Link, Log, LogPlus, ModelError, Mu,
    ObservationView, PartialCorrelation, PositiveLink, Sigma, Tau,
    shape::{Product, Scalar, ShapeValues, StrictLower, Vector},
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};
use gamlss_special::student_t_cdf_standardized;

use crate::multivariate::{
    correlation::{
        FixedPartialCorrelations, correlation_cholesky_from_partial, covariance_from_cholesky,
        partial_corr_from_eta, partial_corr_gradient_from_cholesky_score,
        scale_cholesky_from_correlation,
    },
    elliptical, initial,
    matrix::FixedLowerTriangular,
};

#[cfg(feature = "rand")]
use super::try_sample_location_scale;
use super::{direct_tau_score, nll_location_scale, robust_weight};

/// Default-link multivariate Student-t with explicit marginal standard deviations and partial correlations.
pub type MvStudentTMeanStdPartialCorrDefault<const D: usize> =
    MvStudentTMeanStdPartialCorr<D, Identity, Log, LogPlus<2>>;

/// Multivariate Student-t parameterized by mean, marginal SD, partial correlation and degrees of freedom.
///
/// For `tau > 2`, the covariance is `D R D`, where `D` contains the modeled
/// marginal standard deviations and `R` is built from ordered partial
/// correlations. The internal Student-t scatter factor is
/// `sqrt((tau - 2) / tau) * D * chol(R)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MvStudentTMeanStdPartialCorr<
    const D: usize,
    MuLink = Identity,
    SigmaLink = Log,
    TauLink = LogPlus<2>,
> {
    marker: PhantomData<(MuLink, SigmaLink, TauLink)>,
}

impl<const D: usize, MuLink, SigmaLink, TauLink>
    MvStudentTMeanStdPartialCorr<D, MuLink, SigmaLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    TauLink: AboveTwoLink<f64>,
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
        assert!(D > 0, "multivariate Student-t dimension must be positive");
        Self {
            marker: PhantomData,
        }
    }

    fn theta_from_eta(
        eta: &MvStudentTMeanStdPartialCorrEta<D>,
    ) -> MvStudentTMeanStdPartialCorrTheta<D> {
        let mu = eta.mu.map(MuLink::inverse);
        let sigma = eta.sigma.map(SigmaLink::inverse);
        let tau = TauLink::inverse(eta.tau);
        let partial_corr = partial_corr_from_eta(&eta.partial_corr);
        let correlation_cholesky = correlation_cholesky_from_partial(&partial_corr);
        let covariance_cholesky = scale_cholesky_from_correlation(&sigma, &correlation_cholesky);
        let scale_factor = scatter_per_standard_deviation(tau);
        let scale_cholesky = scaled_cholesky(&covariance_cholesky, scale_factor);
        MvStudentTMeanStdPartialCorrTheta::from_parts_unchecked(
            mu,
            sigma,
            partial_corr,
            correlation_cholesky,
            scale_cholesky,
            tau,
        )
    }

    fn nan_eta() -> MvStudentTMeanStdPartialCorrEta<D> {
        let mut partial_corr = FixedPartialCorrelations::zeros();
        for row in 1..D {
            for col in 0..row {
                *partial_corr
                    .get_mut(row, col)
                    .expect("valid strict-lower index") = f64::NAN;
            }
        }
        MvStudentTMeanStdPartialCorrEta {
            mu: [f64::NAN; D],
            sigma: [f64::NAN; D],
            partial_corr,
            tau: f64::NAN,
        }
    }

    fn nll_theta(observation: [f64; D], theta: &MvStudentTMeanStdPartialCorrTheta<D>) -> f64 {
        let mut z = [0.0; D];
        nll_location_scale(
            observation,
            &theta.mu,
            &theta.scale_cholesky,
            theta.tau,
            &mut z,
        )
    }

    fn nll_and_gradient_eta_values(
        observation: [f64; D],
        eta: &MvStudentTMeanStdPartialCorrEta<D>,
    ) -> (f64, MvStudentTMeanStdPartialCorrEta<D>) {
        let theta = Self::theta_from_eta(eta);
        if !valid_theta(&theta) {
            return (f64::INFINITY, Self::nan_eta());
        }
        let mut z = [0.0; D];
        let nll = nll_location_scale(
            observation,
            &theta.mu,
            &theta.scale_cholesky,
            theta.tau,
            &mut z,
        );
        if !nll.is_finite() {
            return (nll, Self::nan_eta());
        }
        let mut a = [0.0; D];
        if !elliptical::transpose_solve(D, &theta.scale_cholesky, &z, &mut a) {
            return (f64::INFINITY, Self::nan_eta());
        }

        let quadratic = z.iter().map(|value| value * value).sum::<f64>();
        let d = D as f64;
        let weight = robust_weight(d, theta.tau, quadratic);
        let scatter_factor = scatter_per_standard_deviation(theta.tau);
        let mut gradient = zero_eta();
        let mut correlation_score = [[0.0; D]; D];

        for row in 0..D {
            gradient.mu[row] = -weight * a[row] * MuLink::derivative_inverse(eta.mu[row]);
            let residual_score = (0..=row)
                .map(|col| {
                    -weight
                        * a[row]
                        * z[col]
                        * theta.correlation_cholesky.lower(row, col)
                        * scatter_factor
                })
                .sum::<f64>();
            gradient.sigma[row] = SigmaLink::derivative_log_inverse(eta.sigma[row])
                + residual_score * SigmaLink::derivative_inverse(eta.sigma[row]);

            for col in 0..=row {
                correlation_score[row][col] =
                    -weight * scatter_factor * theta.sigma[row] * a[row] * z[col];
                if row == col {
                    correlation_score[row][col] += 1.0 / theta.correlation_cholesky.lower(row, row);
                }
            }
        }

        gradient.partial_corr = partial_corr_gradient_from_cholesky_score(
            &eta.partial_corr,
            &theta.correlation_cholesky,
            &correlation_score,
        );

        let scatter_tau_score = (d - weight * quadratic) / theta.tau / (theta.tau - 2.0);
        let tau_score = direct_tau_score(d, theta.tau, quadratic) + scatter_tau_score;
        gradient.tau = tau_score * TauLink::derivative_inverse(eta.tau);
        (nll, gradient)
    }
}

impl<const D: usize, MuLink, SigmaLink, TauLink> Default
    for MvStudentTMeanStdPartialCorr<D, MuLink, SigmaLink, TauLink>
where
    MuLink: InitialEtaFromTheta<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    TauLink: AboveTwoLink<f64> + InitialEtaFromTheta<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize, MuLink, SigmaLink, TauLink> Family
    for MvStudentTMeanStdPartialCorr<D, MuLink, SigmaLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    TauLink: AboveTwoLink<f64>,
{
    type Observation<'obs> = [f64; D];
    type Eta = MvStudentTMeanStdPartialCorrEta<D>;
    type Theta = MvStudentTMeanStdPartialCorrTheta<D>;
    type GradientEta = MvStudentTMeanStdPartialCorrEta<D>;
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

impl<const D: usize, MuLink, SigmaLink, TauLink> FixedDimensionalFamily<D>
    for MvStudentTMeanStdPartialCorr<D, MuLink, SigmaLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    TauLink: AboveTwoLink<f64>,
{
}

impl<const D: usize, MuLink, SigmaLink, TauLink> HasObservationDimension
    for MvStudentTMeanStdPartialCorr<D, MuLink, SigmaLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    TauLink: AboveTwoLink<f64>,
{
    fn observation_dimension(&self) -> usize {
        D
    }
}

impl<const D: usize, MuLink, SigmaLink, TauLink> HasMarginalCdf
    for MvStudentTMeanStdPartialCorr<D, MuLink, SigmaLink, TauLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    TauLink: AboveTwoLink<f64>,
{
    fn marginal_cdf(&self, component: usize, y: f64, theta: &Self::Theta) -> f64 {
        if component >= D || !y.is_finite() || !valid_theta(theta) {
            return f64::NAN;
        }
        let scale = theta.sigma[component] * scatter_per_standard_deviation(theta.tau);
        student_t_cdf_standardized((y - theta.mu[component]) / scale, theta.tau)
    }
}

#[cfg(feature = "rand")]
impl<Rng, const D: usize, MuLink, SigmaLink, TauLink> TrySimulate<Rng>
    for MvStudentTMeanStdPartialCorr<D, MuLink, SigmaLink, TauLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
    TauLink: AboveTwoLink<f64>,
{
    type Sample = [f64; D];

    fn try_sample(
        &self,
        rng: &mut Rng,
        theta: &Self::Theta,
    ) -> Result<Self::Sample, SimulationError> {
        if !valid_theta(theta) {
            return Err(SimulationError::InvalidParameters(
                "MV Student-t mean/SD/partial-correlation theta",
            ));
        }
        try_sample_location_scale(
            rng,
            theta.mu,
            &theta.scale_cholesky,
            theta.tau,
            "Student-t degrees of freedom",
            "MV Student-t mean/SD/partial-correlation scale mixture transform",
        )
    }
}

impl<const D: usize, MuLink, SigmaLink, TauLink> CompilableFamily
    for MvStudentTMeanStdPartialCorr<D, MuLink, SigmaLink, TauLink>
where
    MuLink: InitialEtaFromTheta<f64>,
    SigmaLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    TauLink: AboveTwoLink<f64> + InitialEtaFromTheta<f64>,
{
    type Shape = Product<
        Product<Product<Vector<Mu, D>, Vector<Sigma, D>>, StrictLower<PartialCorrelation, D>>,
        Scalar<Tau>,
    >;

    fn eta_from_shape(values: ShapeValues<Self::Shape>) -> Self::Eta {
        let mut partial_corr = FixedPartialCorrelations::zeros();
        for row in 1..D {
            for col in 0..row {
                *partial_corr
                    .get_mut(row, col)
                    .expect("valid strict-lower index") = values.0.1[row][col];
            }
        }
        MvStudentTMeanStdPartialCorrEta::new(values.0.0.0, values.0.0.1, partial_corr, values.1)
    }

    fn gradient_to_shape(gradient: &Self::GradientEta) -> ShapeValues<Self::Shape> {
        let partial_corr = std::array::from_fn(|row| {
            std::array::from_fn(|col| gradient.partial_corr.get(row, col).unwrap_or(0.0))
        });
        (((gradient.mu, gradient.sigma), partial_corr), gradient.tau)
    }

    fn initial_shape<'obs, Obs>(&self, obs: &'obs Obs) -> ShapeValues<Self::Shape>
    where
        Obs: ObservationView<'obs, Observation = [f64; D]> + 'obs,
    {
        let (mu, sigma, partial_corr) =
            initial::location_scale_partial_correlation::<D, MuLink, SigmaLink, Obs>(obs);
        (
            ((mu, sigma), partial_corr),
            TauLink::initial_eta_from_theta(10.0),
        )
    }

    fn validate_compiled(&self) -> Result<(), ModelError> {
        Self::try_new().map(|_| ())
    }
}

/// Link-scale predictors for the mean/SD/partial-correlation Student-t parameterization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MvStudentTMeanStdPartialCorrEta<const D: usize> {
    /// Mean predictors.
    pub mu: [f64; D],
    /// Marginal standard-deviation predictors.
    pub sigma: [f64; D],
    /// Strict-lower partial-correlation predictors.
    pub partial_corr: FixedPartialCorrelations<D>,
    /// Degrees-of-freedom predictor.
    pub tau: f64,
}

impl<const D: usize> MvStudentTMeanStdPartialCorrEta<D> {
    /// Creates link-scale predictors.
    #[must_use]
    pub const fn new(
        mu: [f64; D],
        sigma: [f64; D],
        partial_corr: FixedPartialCorrelations<D>,
        tau: f64,
    ) -> Self {
        Self {
            mu,
            sigma,
            partial_corr,
            tau,
        }
    }
}

/// Natural-scale parameters with explicit covariance standard deviations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MvStudentTMeanStdPartialCorrTheta<const D: usize> {
    mu: [f64; D],
    sigma: [f64; D],
    partial_corr: FixedPartialCorrelations<D>,
    correlation_cholesky: FixedLowerTriangular<D>,
    scale_cholesky: FixedLowerTriangular<D>,
    tau: f64,
}

impl<const D: usize> MvStudentTMeanStdPartialCorrTheta<D> {
    /// Creates checked natural-scale parameters and their derived scatter factor.
    pub fn try_new(
        mu: [f64; D],
        sigma: [f64; D],
        partial_corr: FixedPartialCorrelations<D>,
        tau: f64,
    ) -> Result<Self, ModelError> {
        let correlation_cholesky = correlation_cholesky_from_partial(&partial_corr);
        let covariance_cholesky = scale_cholesky_from_correlation(&sigma, &correlation_cholesky);
        let scale_cholesky =
            scaled_cholesky(&covariance_cholesky, scatter_per_standard_deviation(tau));
        let theta = Self::from_parts_unchecked(
            mu,
            sigma,
            partial_corr,
            correlation_cholesky,
            scale_cholesky,
            tau,
        );
        if valid_theta(&theta) {
            Ok(theta)
        } else {
            Err(ModelError::InvalidParameter {
                parameter: "MV Student-t mean/SD/partial-correlation theta",
                expected: "positive dimension, finite means, positive SDs, partial correlations in (-1, 1), and tau > 2",
            })
        }
    }

    const fn from_parts_unchecked(
        mu: [f64; D],
        sigma: [f64; D],
        partial_corr: FixedPartialCorrelations<D>,
        correlation_cholesky: FixedLowerTriangular<D>,
        scale_cholesky: FixedLowerTriangular<D>,
        tau: f64,
    ) -> Self {
        Self {
            mu,
            sigma,
            partial_corr,
            correlation_cholesky,
            scale_cholesky,
            tau,
        }
    }

    /// Mean vector.
    #[must_use]
    pub const fn mu(&self) -> &[f64; D] {
        &self.mu
    }

    /// Marginal standard deviations of the covariance matrix.
    #[must_use]
    pub const fn sigma(&self) -> &[f64; D] {
        &self.sigma
    }

    /// Ordered strict-lower partial correlations.
    #[must_use]
    pub const fn partial_corr(&self) -> &FixedPartialCorrelations<D> {
        &self.partial_corr
    }

    /// Correlation Cholesky factor.
    #[must_use]
    pub const fn correlation_cholesky(&self) -> &FixedLowerTriangular<D> {
        &self.correlation_cholesky
    }

    /// Internal Student-t scatter Cholesky factor.
    #[must_use]
    pub const fn scale_cholesky(&self) -> &FixedLowerTriangular<D> {
        &self.scale_cholesky
    }

    /// Shared degrees of freedom, strictly above two.
    #[must_use]
    pub const fn tau(&self) -> f64 {
        self.tau
    }

    /// Returns one covariance entry from `D R D`.
    #[must_use]
    pub fn covariance(&self, row: usize, col: usize) -> Option<f64> {
        covariance_from_cholesky(&self.correlation_cholesky, row, col)
            .map(|correlation| self.sigma[row] * self.sigma[col] * correlation)
    }
}

fn scatter_per_standard_deviation(tau: f64) -> f64 {
    ((tau - 2.0) / tau).sqrt()
}

fn scaled_cholesky<const D: usize>(
    cholesky: &FixedLowerTriangular<D>,
    factor: f64,
) -> FixedLowerTriangular<D> {
    let mut out = FixedLowerTriangular::zeros();
    for row in 0..D {
        for col in 0..=row {
            out.set_lower(row, col, factor * cholesky.lower(row, col))
                .expect("valid lower index");
        }
    }
    out
}

fn valid_theta<const D: usize>(theta: &MvStudentTMeanStdPartialCorrTheta<D>) -> bool {
    D > 0
        && theta.tau > 2.0
        && theta.tau.is_finite()
        && theta.mu.iter().all(|value| value.is_finite())
        && theta
            .sigma
            .iter()
            .all(|value| *value > 0.0 && value.is_finite())
        && theta
            .partial_corr
            .iter()
            .all(|value| value.is_finite() && value.abs() < 1.0)
        && elliptical::valid_location_scale(D, &theta.mu, &theta.scale_cholesky)
}

fn zero_eta<const D: usize>() -> MvStudentTMeanStdPartialCorrEta<D> {
    MvStudentTMeanStdPartialCorrEta {
        mu: [0.0; D],
        sigma: [0.0; D],
        partial_corr: FixedPartialCorrelations::zeros(),
        tau: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{Family, HasMarginalCdf};

    use super::{
        MvStudentTMeanStdPartialCorrDefault, MvStudentTMeanStdPartialCorrEta,
        MvStudentTMeanStdPartialCorrTheta,
    };
    use crate::multivariate::student_t::MvStudentTCholeskyDefault;
    use crate::multivariate::{FixedPartialCorrelations, student_t::MvStudentTCholeskyTheta};

    fn example_eta() -> MvStudentTMeanStdPartialCorrEta<3> {
        MvStudentTMeanStdPartialCorrEta::new(
            [0.2, -0.3, 0.5],
            [0.1, -0.2, 0.3],
            FixedPartialCorrelations::try_new(vec![0.2, -0.1, 0.3]).unwrap(),
            8.0_f64.ln(),
        )
    }

    #[test]
    fn eta_gradient_matches_finite_difference() {
        let family = MvStudentTMeanStdPartialCorrDefault::<3>::new();
        let eta = example_eta();
        let y = [0.8, -0.6, 1.1];
        let (_, gradient) = family.nll_and_gradient_eta(y, &eta, &mut ());

        for component in 0..3 {
            let mut lower = eta;
            let mut upper = eta;
            lower.mu[component] -= 1.0e-6;
            upper.mu[component] += 1.0e-6;
            let numeric =
                (family.nll_eta(y, &upper, &mut ()) - family.nll_eta(y, &lower, &mut ())) / 2.0e-6;
            assert_relative_eq!(gradient.mu[component], numeric, epsilon = 2.0e-6);

            let mut lower = eta;
            let mut upper = eta;
            lower.sigma[component] -= 1.0e-6;
            upper.sigma[component] += 1.0e-6;
            let numeric =
                (family.nll_eta(y, &upper, &mut ()) - family.nll_eta(y, &lower, &mut ())) / 2.0e-6;
            assert_relative_eq!(gradient.sigma[component], numeric, epsilon = 2.0e-6);
        }
        for row in 1..3 {
            for col in 0..row {
                let mut lower = eta;
                let mut upper = eta;
                *lower.partial_corr.get_mut(row, col).unwrap() -= 1.0e-6;
                *upper.partial_corr.get_mut(row, col).unwrap() += 1.0e-6;
                let numeric = (family.nll_eta(y, &upper, &mut ())
                    - family.nll_eta(y, &lower, &mut ()))
                    / 2.0e-6;
                assert_relative_eq!(
                    gradient.partial_corr.get(row, col).unwrap(),
                    numeric,
                    epsilon = 2.0e-6
                );
            }
        }
        let mut lower = eta;
        let mut upper = eta;
        lower.tau -= 1.0e-6;
        upper.tau += 1.0e-6;
        let numeric =
            (family.nll_eta(y, &upper, &mut ()) - family.nll_eta(y, &lower, &mut ())) / 2.0e-6;
        assert_relative_eq!(gradient.tau, numeric, epsilon = 2.0e-6);
    }

    #[test]
    fn covariance_parameterization_matches_cholesky_family() {
        let family = MvStudentTMeanStdPartialCorrDefault::<2>::new();
        let partial = FixedPartialCorrelations::try_new(vec![0.3]).unwrap();
        let theta =
            MvStudentTMeanStdPartialCorrTheta::try_new([0.2, -0.1], [1.2, 0.8], partial, 7.0)
                .unwrap();
        let cholesky_theta =
            MvStudentTCholeskyTheta::try_new(*theta.mu(), *theta.scale_cholesky(), theta.tau())
                .unwrap();
        let cholesky = MvStudentTCholeskyDefault::<2>::new();
        let y = [0.7, -0.4];
        assert_relative_eq!(
            family.nll(y, &theta, &mut ()),
            cholesky.nll(y, &cholesky_theta, &mut ()),
            epsilon = 1.0e-13
        );
        assert_relative_eq!(theta.covariance(0, 0).unwrap(), 1.2_f64.powi(2));
        assert_relative_eq!(theta.covariance(1, 1).unwrap(), 0.8_f64.powi(2));
        assert_relative_eq!(theta.covariance(1, 0).unwrap(), 1.2 * 0.8 * 0.3);
    }

    #[test]
    fn one_dimensional_marginal_uses_actual_standard_deviation() {
        let family = MvStudentTMeanStdPartialCorrDefault::<1>::new();
        let theta = MvStudentTMeanStdPartialCorrTheta::try_new(
            [0.4],
            [1.7],
            FixedPartialCorrelations::zeros(),
            8.0,
        )
        .unwrap();
        assert_relative_eq!(family.marginal_cdf(0, 0.4, &theta), 0.5, epsilon = 1.0e-14);
        assert_relative_eq!(
            theta.covariance(0, 0).unwrap(),
            1.7_f64.powi(2),
            epsilon = 1.0e-14
        );
    }

    #[test]
    fn finite_tau_predictors_remain_well_defined_at_the_lower_link_saturation() {
        let family = MvStudentTMeanStdPartialCorrDefault::<2>::new();
        let eta = MvStudentTMeanStdPartialCorrEta::new(
            [0.0; 2],
            [0.0; 2],
            FixedPartialCorrelations::try_new(vec![0.0]).unwrap(),
            -1_000.0,
        );
        let theta = family.theta(&eta, &mut ());
        assert!(theta.tau() > 2.0);

        let (nll, gradient) = family.nll_and_gradient_eta([0.0; 2], &eta, &mut ());
        assert!(nll.is_finite());
        assert!(gradient.mu.iter().all(|value| value.is_finite()));
        assert!(gradient.sigma.iter().all(|value| value.is_finite()));
        assert!(gradient.partial_corr.iter().all(f64::is_finite));
        assert!(gradient.tau.is_finite());
    }

    #[test]
    fn invalid_domains_are_rejected() {
        assert!(MvStudentTMeanStdPartialCorrDefault::<0>::try_new().is_err());
        assert!(
            MvStudentTMeanStdPartialCorrTheta::<2>::try_new(
                [0.0; 2],
                [1.0; 2],
                FixedPartialCorrelations::try_new(vec![0.0]).unwrap(),
                2.0,
            )
            .is_err()
        );
    }
}
