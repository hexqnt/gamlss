#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::suboptimal_flops
)]

//! Cholesky-scatter multivariate power-exponential parameterization.

use std::marker::PhantomData;

use crate::multivariate::{elliptical, initial, matrix::FixedLowerTriangular};
use gamlss_core::{
    CompilableFamily, Family, FixedDimensionalFamily, HasObservationDimension, Identity,
    InitialEtaFromTheta, Link, LocationCholesky, Log, ModelError, ObservationView, PositiveLink,
    Power,
    shape::{Product, Scalar, ShapeValues},
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

use super::{direct_power_score, radial_nll_constant};

/// Default-link multivariate power-exponential with Cholesky scale.
pub type MvPowerExponentialCholeskyDefault<const D: usize> =
    MvPowerExponentialCholesky<D, Identity, Log, Identity, Log>;

/// Multivariate power-exponential parameterized by location, Cholesky scale and power.
///
/// For `z = L^-1 (y - mu)` and `q = z' z`, the radial kernel is
/// `exp(-q^(power / 2) / 2)`. `power = 2` recovers the multivariate normal;
/// smaller values give sharper centers and heavier tails, while larger values
/// give flatter centers and lighter tails. Except at `power = 2`, `L L'` is a
/// scatter matrix rather than the response covariance.
///
/// This family is intended for symmetric joint forecast errors, returns and
/// physical measurements whose tail weight differs from both normal and
/// Student-t behavior. A zero residual uses the same zero-score convention as
/// the scalar power-exponential family, including at powers whose density is
/// not differentiable at the center.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MvPowerExponentialCholesky<
    const D: usize,
    MuLink = Identity,
    DiagonalLink = Log,
    OffDiagonalLink = Identity,
    PowerLink = Log,
> {
    marker: PhantomData<(MuLink, DiagonalLink, OffDiagonalLink, PowerLink)>,
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink, PowerLink>
    MvPowerExponentialCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, PowerLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
    PowerLink: PositiveLink<f64>,
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
        assert!(
            D > 0,
            "multivariate power-exponential dimension must be positive"
        );
        Self {
            marker: PhantomData,
        }
    }

    fn theta_from_eta(
        eta: &MvPowerExponentialCholeskyEta<D>,
    ) -> MvPowerExponentialCholeskyTheta<D> {
        let mu = eta.mu.map(MuLink::inverse);
        let mut cholesky = FixedLowerTriangular::zeros();
        for row in 0..D {
            for col in 0..=row {
                let eta_value = eta.cholesky.lower(row, col);
                let value = if row == col {
                    DiagonalLink::inverse(eta_value)
                } else {
                    OffDiagonalLink::inverse(eta_value)
                };
                cholesky
                    .set_lower(row, col, value)
                    .expect("valid lower-triangular index");
            }
        }
        MvPowerExponentialCholeskyTheta::from_parts_unchecked(
            mu,
            cholesky,
            PowerLink::inverse(eta.power),
        )
    }

    fn nan_eta() -> MvPowerExponentialCholeskyEta<D> {
        MvPowerExponentialCholeskyEta {
            mu: [f64::NAN; D],
            cholesky: FixedLowerTriangular::from_lower_rows([[f64::NAN; D]; D]),
            power: f64::NAN,
        }
    }

    fn nll_theta(observation: [f64; D], theta: &MvPowerExponentialCholeskyTheta<D>) -> f64 {
        if !valid_theta(theta) {
            return f64::INFINITY;
        }
        let mut z = [0.0; D];
        let Some((quadratic, log_det_scale)) =
            elliptical::standardize(D, &observation, &theta.mu, &theta.cholesky, &mut z)
        else {
            return f64::INFINITY;
        };
        let radial_power = quadratic.powf(0.5 * theta.power);
        let nll = log_det_scale + radial_nll_constant(D as f64, theta.power) + 0.5 * radial_power;
        if nll.is_finite() { nll } else { f64::INFINITY }
    }

    fn nll_and_gradient_eta_values(
        observation: [f64; D],
        eta: &MvPowerExponentialCholeskyEta<D>,
    ) -> (f64, MvPowerExponentialCholeskyEta<D>) {
        let theta = Self::theta_from_eta(eta);
        if !valid_theta(&theta) {
            return (f64::INFINITY, Self::nan_eta());
        }
        let mut z = [0.0; D];
        let Some((quadratic, log_det_scale)) =
            elliptical::standardize(D, &observation, &theta.mu, &theta.cholesky, &mut z)
        else {
            return (f64::INFINITY, Self::nan_eta());
        };
        let radial_power = quadratic.powf(0.5 * theta.power);
        let nll = log_det_scale + radial_nll_constant(D as f64, theta.power) + 0.5 * radial_power;
        if !nll.is_finite() {
            return (f64::INFINITY, Self::nan_eta());
        }

        let radial_weight = if quadratic == 0.0 {
            0.0
        } else {
            0.5 * theta.power * quadratic.powf(0.5 * theta.power - 1.0)
        };
        let standardized_gradient = z.map(|standardized| radial_weight * standardized);
        let mut location_score = [0.0; D];
        if !elliptical::transpose_solve(
            D,
            &theta.cholesky,
            &standardized_gradient,
            &mut location_score,
        ) {
            return (f64::INFINITY, Self::nan_eta());
        }

        let mut gradient = MvPowerExponentialCholeskyEta {
            mu: std::array::from_fn(|component| {
                -location_score[component] * MuLink::derivative_inverse(eta.mu[component])
            }),
            cholesky: FixedLowerTriangular::zeros(),
            power: direct_power_score(D as f64, theta.power, quadratic, radial_power)
                * PowerLink::derivative_inverse(eta.power),
        };
        for row in 0..D {
            for col in 0..=row {
                let eta_value = eta.cholesky.lower(row, col);
                let score = if row == col {
                    DiagonalLink::derivative_log_inverse(eta_value)
                        - location_score[row] * z[col] * DiagonalLink::derivative_inverse(eta_value)
                } else {
                    -location_score[row] * z[col] * OffDiagonalLink::derivative_inverse(eta_value)
                };
                gradient
                    .cholesky
                    .set_lower(row, col, score)
                    .expect("valid lower-triangular index");
            }
        }
        (nll, gradient)
    }
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink, PowerLink> Default
    for MvPowerExponentialCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, PowerLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
    PowerLink: PositiveLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink, PowerLink> Family
    for MvPowerExponentialCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, PowerLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
    PowerLink: PositiveLink<f64>,
{
    type Eta = MvPowerExponentialCholeskyEta<D>;
    type Theta = MvPowerExponentialCholeskyTheta<D>;
    type GradientEta = MvPowerExponentialCholeskyEta<D>;
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

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink, PowerLink> FixedDimensionalFamily<D>
    for MvPowerExponentialCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, PowerLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
    PowerLink: PositiveLink<f64>,
{
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink, PowerLink> HasObservationDimension
    for MvPowerExponentialCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, PowerLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
    PowerLink: PositiveLink<f64>,
{
    fn observation_dimension(&self) -> usize {
        D
    }
}

#[cfg(feature = "rand")]
impl<Rng, const D: usize, MuLink, DiagonalLink, OffDiagonalLink, PowerLink> TrySimulate<Rng>
    for MvPowerExponentialCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, PowerLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
    PowerLink: PositiveLink<f64>,
{
    type Sample = [f64; D];

    fn try_sample(
        &self,
        rng: &mut Rng,
        theta: &Self::Theta,
    ) -> Result<Self::Sample, SimulationError> {
        if !valid_theta(theta) {
            return Err(SimulationError::InvalidParameters(
                "multivariate power-exponential theta",
            ));
        }
        let dimension = D as f64;
        let gamma = rand_distr::Gamma::new(dimension / theta.power, 1.0)
            .map_err(|_| SimulationError::BackendRejected("power-exponential radial shape"))?;
        let radial_gamma = rand_distr::Distribution::sample(&gamma, rng);
        let radius = (2.0 * radial_gamma).powf(1.0 / theta.power);
        let normal = rand_distr::StandardNormal;
        let direction: [f64; D] =
            std::array::from_fn(|_| rand_distr::Distribution::sample(&normal, rng));
        let direction_norm = direction
            .iter()
            .fold(0.0_f64, |norm, value| norm.hypot(*value));
        if !radius.is_finite() || direction_norm == 0.0 || !direction_norm.is_finite() {
            return Err(SimulationError::NumericalFailure(
                "power-exponential radial transform",
            ));
        }
        let standardized = direction.map(|value| radius * value / direction_norm);
        let mut out = theta.mu;
        for row in 0..D {
            for col in 0..=row {
                out[row] += theta.cholesky.lower(row, col) * standardized[col];
            }
        }
        if out.iter().all(|value| value.is_finite()) {
            Ok(out)
        } else {
            Err(SimulationError::NumericalFailure(
                "multivariate power-exponential transform",
            ))
        }
    }
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink, PowerLink> CompilableFamily
    for MvPowerExponentialCholesky<D, MuLink, DiagonalLink, OffDiagonalLink, PowerLink>
where
    MuLink: InitialEtaFromTheta<f64>,
    DiagonalLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    OffDiagonalLink: InitialEtaFromTheta<f64>,
    PowerLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
{
    type Shape = Product<LocationCholesky<D>, Scalar<Power>>;

    fn eta_from_shape(values: ShapeValues<Self::Shape>) -> Self::Eta {
        MvPowerExponentialCholeskyEta::new(
            values.0.0,
            FixedLowerTriangular::from_lower_rows(values.0.1),
            values.1,
        )
    }

    fn gradient_to_shape(gradient: &Self::GradientEta) -> ShapeValues<Self::Shape> {
        let lower = std::array::from_fn(|row| {
            std::array::from_fn(|col| gradient.cholesky.get(row, col).unwrap_or(0.0))
        });
        ((gradient.mu, lower), gradient.power)
    }

    fn initial_shape<'obs, Obs>(&self, obs: &'obs Obs) -> ShapeValues<Self::Shape>
    where
        Obs: ObservationView<'obs, Observation = [f64; D]> + 'obs,
    {
        (
            initial::location_cholesky::<D, MuLink, DiagonalLink, OffDiagonalLink, Obs>(obs),
            PowerLink::initial_eta_from_theta(2.0),
        )
    }
}

/// Link-scale predictors for multivariate power-exponential.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MvPowerExponentialCholeskyEta<const D: usize> {
    mu: [f64; D],
    cholesky: FixedLowerTriangular<D>,
    power: f64,
}

impl<const D: usize> MvPowerExponentialCholeskyEta<D> {
    /// Creates link-scale predictors.
    #[must_use]
    pub const fn new(mu: [f64; D], cholesky: FixedLowerTriangular<D>, power: f64) -> Self {
        Self {
            mu,
            cholesky,
            power,
        }
    }

    /// Location predictors.
    #[must_use]
    pub const fn mu(&self) -> &[f64; D] {
        &self.mu
    }

    /// Mutable location predictors.
    #[must_use]
    pub const fn mu_mut(&mut self) -> &mut [f64; D] {
        &mut self.mu
    }

    /// Lower-triangular Cholesky predictors.
    #[must_use]
    pub const fn cholesky(&self) -> &FixedLowerTriangular<D> {
        &self.cholesky
    }

    /// Mutable lower-triangular Cholesky predictors.
    #[must_use]
    pub const fn cholesky_mut(&mut self) -> &mut FixedLowerTriangular<D> {
        &mut self.cholesky
    }

    /// Positive radial-power predictor.
    #[must_use]
    pub const fn power(&self) -> f64 {
        self.power
    }

    /// Mutably borrows the radial-power predictor.
    #[must_use]
    pub const fn power_mut(&mut self) -> &mut f64 {
        &mut self.power
    }
}

/// Natural-scale multivariate power-exponential parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MvPowerExponentialCholeskyTheta<const D: usize> {
    mu: [f64; D],
    cholesky: FixedLowerTriangular<D>,
    power: f64,
}

impl<const D: usize> MvPowerExponentialCholeskyTheta<D> {
    /// Creates checked natural-scale parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] unless `D > 0`, location and
    /// Cholesky entries are finite, the Cholesky diagonal is strictly positive,
    /// and `power` is finite and positive.
    pub fn try_new(
        mu: [f64; D],
        cholesky: FixedLowerTriangular<D>,
        power: f64,
    ) -> Result<Self, ModelError> {
        let theta = Self {
            mu,
            cholesky,
            power,
        };
        if valid_theta(&theta) {
            Ok(theta)
        } else {
            Err(ModelError::InvalidParameter {
                parameter: "multivariate power-exponential theta",
                expected: "positive dimension, finite values, positive Cholesky diagonal, and positive power",
            })
        }
    }

    const fn from_parts_unchecked(
        mu: [f64; D],
        cholesky: FixedLowerTriangular<D>,
        power: f64,
    ) -> Self {
        Self {
            mu,
            cholesky,
            power,
        }
    }

    /// Location vector.
    #[must_use]
    pub const fn mu(&self) -> &[f64; D] {
        &self.mu
    }

    /// Cholesky factor of the scatter matrix.
    #[must_use]
    pub const fn cholesky(&self) -> &FixedLowerTriangular<D> {
        &self.cholesky
    }

    /// Positive radial power; `2` is the multivariate normal case.
    #[must_use]
    pub const fn power(&self) -> f64 {
        self.power
    }

    /// Returns one scatter entry from `L L'`.
    #[must_use]
    pub fn scatter(&self, row: usize, col: usize) -> Option<f64> {
        if row >= D || col >= D {
            return None;
        }
        Some(
            (0..=row.min(col))
                .map(|index| self.cholesky.lower(row, index) * self.cholesky.lower(col, index))
                .sum(),
        )
    }
}

fn valid_theta<const D: usize>(theta: &MvPowerExponentialCholeskyTheta<D>) -> bool {
    theta.power > 0.0
        && theta.power.is_finite()
        && elliptical::valid_location_scale(D, &theta.mu, &theta.cholesky)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{
        CholeskyScale, DenseDesign, Family, Gamlss, LinearPredictorBlock,
        LowerTriangularParameterBlock, ModelError, Mu, NoPenalty, ParameterBlock, ParameterBlocks,
        Power, VectorParameterBlock,
    };

    use super::{
        MvPowerExponentialCholeskyDefault, MvPowerExponentialCholeskyEta,
        MvPowerExponentialCholeskyTheta,
    };
    use crate::multivariate::{
        matrix::FixedLowerTriangular,
        normal::{MvNormalCholeskyDefault, MvNormalCholeskyEta},
    };

    #[test]
    fn checked_constructor_rejects_zero_dimension() {
        assert_eq!(
            MvPowerExponentialCholeskyDefault::<0>::try_new(),
            Err(ModelError::InvalidParameter {
                parameter: "dimension",
                expected: "positive",
            })
        );
    }

    #[test]
    fn power_two_matches_multivariate_normal() {
        let power_exponential = MvPowerExponentialCholeskyDefault::<2>::new();
        let normal = MvNormalCholeskyDefault::<2>::new();
        let eta = MvPowerExponentialCholeskyEta::new(
            [0.2, -0.4],
            FixedLowerTriangular::from_lower_rows([[0.1, 0.0], [0.3, -0.2]]),
            2.0_f64.ln(),
        );
        let normal_eta = MvNormalCholeskyEta::new(*eta.mu(), *eta.cholesky());
        let observation = [0.8, -0.1];
        let (power_nll, power_gradient) =
            power_exponential.nll_and_gradient_eta(observation, &eta, &mut ());
        let (normal_nll, normal_gradient) =
            normal.nll_and_gradient_eta(observation, &normal_eta, &mut ());
        assert_relative_eq!(power_nll, normal_nll, epsilon = 1.0e-12);
        for component in 0..2 {
            assert_relative_eq!(
                power_gradient.mu()[component],
                normal_gradient.mu()[component],
                epsilon = 1.0e-12
            );
        }
        for row in 0..2 {
            for col in 0..=row {
                assert_relative_eq!(
                    power_gradient.cholesky().get(row, col).unwrap(),
                    normal_gradient.cholesky().get(row, col).unwrap(),
                    epsilon = 1.0e-12
                );
            }
        }
    }

    #[test]
    fn analytic_gradient_matches_finite_difference() {
        let family = MvPowerExponentialCholeskyDefault::<2>::new();
        let eta = MvPowerExponentialCholeskyEta::new(
            [0.1, -0.2],
            FixedLowerTriangular::from_lower_rows([[0.0, 0.0], [0.2, -0.1]]),
            1.4_f64.ln(),
        );
        let observation = [0.7, -0.8];
        let (_, gradient) = family.nll_and_gradient_eta(observation, &eta, &mut ());
        let epsilon = 1.0e-6;

        for component in 0..2 {
            let mut plus = eta;
            plus.mu_mut()[component] += epsilon;
            let mut minus = eta;
            minus.mu_mut()[component] -= epsilon;
            let finite_difference = (family.nll_eta(observation, &plus, &mut ())
                - family.nll_eta(observation, &minus, &mut ()))
                / (2.0 * epsilon);
            assert_relative_eq!(
                gradient.mu()[component],
                finite_difference,
                epsilon = 1.0e-6
            );
        }
        for row in 0..2 {
            for col in 0..=row {
                let mut plus = eta;
                let current = plus.cholesky().get(row, col).unwrap();
                plus.cholesky_mut()
                    .set_lower(row, col, current + epsilon)
                    .unwrap();
                let mut minus = eta;
                minus
                    .cholesky_mut()
                    .set_lower(row, col, current - epsilon)
                    .unwrap();
                let finite_difference = (family.nll_eta(observation, &plus, &mut ())
                    - family.nll_eta(observation, &minus, &mut ()))
                    / (2.0 * epsilon);
                assert_relative_eq!(
                    gradient.cholesky().get(row, col).unwrap(),
                    finite_difference,
                    epsilon = 1.0e-6
                );
            }
        }
        let mut plus = eta;
        *plus.power_mut() += epsilon;
        let mut minus = eta;
        *minus.power_mut() -= epsilon;
        let finite_difference = (family.nll_eta(observation, &plus, &mut ())
            - family.nll_eta(observation, &minus, &mut ()))
            / (2.0 * epsilon);
        assert_relative_eq!(gradient.power(), finite_difference, epsilon = 1.0e-6);
    }

    #[test]
    fn invalid_domains_are_rejected() {
        assert!(
            MvPowerExponentialCholeskyTheta::<0>::try_new([], FixedLowerTriangular::zeros(), 2.0,)
                .is_err()
        );
        assert!(
            MvPowerExponentialCholeskyTheta::try_new(
                [0.0, 0.0],
                FixedLowerTriangular::from_lower_rows([[1.0, 0.0], [0.0, 1.0]]),
                0.0,
            )
            .is_err()
        );
        assert!(
            MvPowerExponentialCholeskyTheta::try_new(
                [0.0, 0.0],
                FixedLowerTriangular::from_lower_rows([[1.0, 0.0], [0.0, 0.0]]),
                2.0,
            )
            .is_err()
        );
    }

    #[test]
    fn static_shape_is_fit_ready() {
        let response = [[0.2, -0.3], [0.8, 0.4], [-0.5, 0.7]];
        let rows = response.len();
        let intercept = || LinearPredictorBlock::new(DenseDesign::intercept(rows));
        let mu = VectorParameterBlock::<Mu, 2, _, _>::new([intercept(), intercept()], NoPenalty, 0);
        let cholesky = LowerTriangularParameterBlock::<CholeskyScale, 2, _, _>::new(
            vec![intercept(), intercept(), intercept()],
            NoPenalty,
            0,
        );
        let power =
            ParameterBlock::<Power, _, _>::linear(DenseDesign::intercept(rows), NoPenalty, 0);
        let model = Gamlss::try_new_with_observations(
            MvPowerExponentialCholeskyDefault::<2>::new(),
            ParameterBlocks::new((mu, cholesky, power)),
            response.as_slice(),
        )
        .unwrap();
        let beta = [0.1, -0.2, 0.0, 0.1, -0.1, 2.0_f64.ln()];
        let mut gradient = [0.0; 6];
        let value = model.try_value_gradient_into(&beta, &mut gradient).unwrap();
        assert!(value.is_finite());
        assert!(gradient.iter().all(|score| score.is_finite()));
    }

    #[cfg(feature = "rand")]
    #[test]
    fn sampling_returns_finite_vectors() {
        use gamlss_core::TrySimulate;
        use rand::SeedableRng;

        let family = MvPowerExponentialCholeskyDefault::<2>::new();
        let theta = MvPowerExponentialCholeskyTheta::try_new(
            [0.2, -0.1],
            FixedLowerTriangular::from_lower_rows([[1.0, 0.0], [0.3, 0.8]]),
            1.4,
        )
        .unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(37);
        assert!(
            family
                .try_sample(&mut rng, &theta)
                .is_ok_and(|sample| sample.iter().all(|value| value.is_finite()))
        );
    }
}
