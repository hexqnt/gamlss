use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::{CanSimulate, SimulationError, TrySimulate};
use gamlss_core::{
    CompilableFamily, Family, FixedDimensionalFamily, HasConditionalCdf, HasMarginalCdf,
    HasObservationDimension, HasRosenblattTransform, Identity, InitialEtaFromTheta, Link,
    LocationCholesky, Log, ModelError, ObservationView, PositiveLink, shape::ShapeValues,
};
use gamlss_special::unit_normal_cdf;

use crate::multivariate::matrix::FixedLowerTriangular;

use super::kernel;

impl<const D: usize> kernel::LowerTriangularMatrix for FixedLowerTriangular<D> {
    #[inline]
    fn dimension(&self) -> usize {
        D
    }

    #[inline]
    fn lower(&self, row: usize, col: usize) -> f64 {
        self.lower(row, col)
    }
}

/// Fixed-dimensional multivariate normal parameterized by mean and Cholesky scale.
///
/// The natural-scale `cholesky` matrix is lower triangular. Diagonal entries
/// are transformed with `DiagonalLink`; off-diagonal lower entries are
/// transformed with `OffDiagonalLink`; upper entries are ignored and normalized
/// to zero by [`Family::theta`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MvNormalCholesky<
    const D: usize,
    MuLink = Identity,
    DiagonalLink = Log,
    OffDiagonalLink = Identity,
> {
    marker: PhantomData<(MuLink, DiagonalLink, OffDiagonalLink)>,
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink>
    MvNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
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

    fn theta_from_eta(eta: MvNormalCholeskyEta<D>) -> MvNormalCholeskyTheta<D> {
        let mut mu = [0.0; D];
        let mut cholesky = FixedLowerTriangular::zeros();

        for (mu, eta_mu) in mu.iter_mut().zip(eta.mu) {
            *mu = MuLink::inverse(eta_mu);
        }

        for row in 0..D {
            for col in 0..=row {
                let value = if row == col {
                    DiagonalLink::inverse(eta.cholesky.lower(row, col))
                } else {
                    OffDiagonalLink::inverse(eta.cholesky.lower(row, col))
                };
                cholesky
                    .set_lower(row, col, value)
                    .expect("row and col are valid lower-triangular indices");
            }
        }

        MvNormalCholeskyTheta::from_parts_unchecked(mu, cholesky)
    }

    fn nan_eta() -> MvNormalCholeskyEta<D> {
        MvNormalCholeskyEta {
            mu: [f64::NAN; D],
            cholesky: FixedLowerTriangular::from_lower_rows([[f64::NAN; D]; D]),
        }
    }

    fn nll_theta(observation: [f64; D], theta: MvNormalCholeskyTheta<D>) -> f64 {
        let mut z = [0.0; D];
        kernel::nll(D, &observation, &theta.mu, &theta.cholesky, &mut z)
    }

    fn nll_and_gradient_eta_values(
        observation: [f64; D],
        eta: MvNormalCholeskyEta<D>,
    ) -> (f64, MvNormalCholeskyEta<D>) {
        let theta = Self::theta_from_eta(eta);
        let mut z = [0.0; D];
        let mut a = [0.0; D];
        let nll =
            kernel::nll_and_score(D, &observation, &theta.mu, &theta.cholesky, &mut z, &mut a);
        if !nll.is_finite() {
            return (nll, Self::nan_eta());
        }

        let mut gradient = MvNormalCholeskyEta {
            mu: [0.0; D],
            cholesky: FixedLowerTriangular::zeros(),
        };

        for ((gradient_mu, a), eta_mu) in gradient.mu.iter_mut().zip(a).zip(eta.mu) {
            *gradient_mu = -a * MuLink::derivative_inverse(eta_mu);
        }

        for row in 0..D {
            for (col, z_col) in z.iter().copied().take(row + 1).enumerate() {
                let eta_value = eta.cholesky.lower(row, col);
                let d_nll_d_l = if row == col {
                    DiagonalLink::derivative_log_inverse(eta_value)
                        - a[row] * z_col * DiagonalLink::derivative_inverse(eta_value)
                } else {
                    kernel::cholesky_score(row, col, z_col, &a, &theta.cholesky)
                        * OffDiagonalLink::derivative_inverse(eta_value)
                };
                gradient
                    .cholesky
                    .set_lower(row, col, d_nll_d_l)
                    .expect("row and col are valid lower-triangular indices");
            }
        }

        (nll, gradient)
    }

    fn marginal_scale(component: usize, theta: &MvNormalCholeskyTheta<D>) -> f64 {
        kernel::marginal_scale(D, component, &theta.mu, &theta.cholesky)
    }
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink> Default
    for MvNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink> Family
    for MvNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    type Eta = MvNormalCholeskyEta<D>;
    type Theta = MvNormalCholeskyTheta<D>;
    type GradientEta = MvNormalCholeskyEta<D>;
    type Observation<'obs> = [f64; D];
    type Workspace = ();

    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    #[inline]
    fn nll(
        &self,
        observation: Self::Observation<'_>,
        theta: &Self::Theta,
        _workspace: &mut Self::Workspace,
    ) -> f64 {
        Self::nll_theta(observation, *theta)
    }

    #[inline]
    fn nll_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> f64 {
        Self::nll_theta(observation, Self::theta_from_eta(*eta))
    }

    #[inline]
    fn nll_and_gradient_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        Self::nll_and_gradient_eta_values(observation, *eta)
    }
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink> FixedDimensionalFamily<D>
    for MvNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink> HasMarginalCdf
    for MvNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    fn marginal_cdf(&self, component: usize, y: f64, theta: &Self::Theta) -> f64 {
        if !y.is_finite() || component >= D {
            return f64::NAN;
        }

        let scale = Self::marginal_scale(component, theta);
        if !scale.is_finite() || scale <= 0.0 {
            return f64::NAN;
        }

        unit_normal_cdf((y - theta.mu[component]) / scale)
    }
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink> HasObservationDimension
    for MvNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    fn observation_dimension(&self) -> usize {
        D
    }
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink> HasConditionalCdf
    for MvNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    fn conditional_cdf(
        &self,
        component: usize,
        y: f64,
        preceding: &[f64],
        theta: &Self::Theta,
    ) -> f64 {
        if component >= D
            || preceding.len() < component
            || !y.is_finite()
            || !kernel::valid_theta(D, theta.mu(), theta.cholesky())
        {
            return f64::NAN;
        }
        let mut standardized = [0.0; D];
        for row in 0..component {
            let mut residual = preceding[row] - theta.mu()[row];
            for col in 0..row {
                residual -= theta.cholesky().lower(row, col) * standardized[col];
            }
            standardized[row] = residual / theta.cholesky().lower(row, row);
        }
        let conditional_mean = (0..component).fold(theta.mu()[component], |mean, col| {
            theta
                .cholesky()
                .lower(component, col)
                .mul_add(standardized[col], mean)
        });
        unit_normal_cdf((y - conditional_mean) / theta.cholesky().lower(component, component))
    }
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink> HasRosenblattTransform
    for MvNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
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
        for component in 0..D {
            out[component] =
                self.conditional_cdf(component, observation[component], &observation, theta);
        }
        Ok(())
    }
}

#[cfg(feature = "rand")]
impl<Rng, const D: usize, MuLink, DiagonalLink, OffDiagonalLink> CanSimulate<Rng>
    for MvNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    type Sample = [f64; D];

    fn sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Self::Sample {
        if !kernel::valid_theta(D, &theta.mu, &theta.cholesky) {
            return [f64::NAN; D];
        }

        let standard = rand_distr::StandardNormal;
        let mut z = [0.0; D];
        for value in &mut z {
            *value = rand_distr::Distribution::sample(&standard, rng);
        }

        let mut out = theta.mu;
        for row in 0..D {
            for (col, z_col) in z.iter().copied().take(row + 1).enumerate() {
                out[row] += theta.cholesky.lower(row, col) * z_col;
            }
        }
        out
    }
}

#[cfg(feature = "rand")]
impl<Rng, const D: usize, MuLink, DiagonalLink, OffDiagonalLink> TrySimulate<Rng>
    for MvNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    type Sample = [f64; D];

    fn try_sample(
        &self,
        rng: &mut Rng,
        theta: &Self::Theta,
    ) -> Result<Self::Sample, SimulationError> {
        if !kernel::valid_theta(D, &theta.mu, &theta.cholesky) {
            return Err(SimulationError::InvalidParameters("MVN Cholesky theta"));
        }
        let standard = rand_distr::StandardNormal;
        let z: [f64; D] = std::array::from_fn(|_| rand_distr::Distribution::sample(&standard, rng));
        let mut out = theta.mu;
        for row in 0..D {
            for (col, z_col) in z.iter().copied().take(row + 1).enumerate() {
                out[row] += theta.cholesky.lower(row, col) * z_col;
            }
        }
        Ok(out)
    }
}

impl<const D: usize, MuLink, DiagonalLink, OffDiagonalLink> CompilableFamily
    for MvNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink>
where
    MuLink: InitialEtaFromTheta<f64>,
    DiagonalLink: InitialEtaFromTheta<f64> + PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    type Shape = LocationCholesky<D>;

    fn eta_from_shape(values: ShapeValues<Self::Shape>) -> MvNormalCholeskyEta<D> {
        MvNormalCholeskyEta::new(values.0, FixedLowerTriangular::from_lower_rows(values.1))
    }

    fn gradient_to_shape(gradient: &MvNormalCholeskyEta<D>) -> ShapeValues<Self::Shape> {
        let lower = std::array::from_fn(|row| {
            std::array::from_fn(|col| gradient.cholesky.get(row, col).unwrap_or(0.0))
        });
        (gradient.mu, lower)
    }

    fn initial_shape<'obs, Obs>(&self, obs: &'obs Obs) -> ShapeValues<Self::Shape>
    where
        Obs: ObservationView<'obs, Observation = [f64; D]> + 'obs,
    {
        let mut weight_sum = [0.0; D];
        let mut means = [0.0; D];
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            if weight == 0.0 {
                continue;
            }
            let observation = obs.observation_at(row);
            for component in 0..D {
                let value = observation[component];
                if value.is_finite() {
                    weight_sum[component] += weight;
                    means[component] += weight * value;
                }
            }
        }
        for component in 0..D {
            if weight_sum[component] > 0.0 {
                means[component] /= weight_sum[component];
            }
        }

        let mut variances = [0.0; D];
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            if weight == 0.0 {
                continue;
            }
            let observation = obs.observation_at(row);
            for component in 0..D {
                let value = observation[component];
                if value.is_finite() && weight_sum[component] > 0.0 {
                    let residual = value - means[component];
                    variances[component] += weight * residual * residual;
                }
            }
        }

        let mut vector_eta = [0.0; D];
        let mut lower_eta = [[0.0; D]; D];
        for component in 0..D {
            vector_eta[component] = MuLink::initial_eta_from_theta(means[component]);
            let scale = if weight_sum[component] > 0.0 {
                (variances[component] / weight_sum[component])
                    .sqrt()
                    .max(1.0e-6)
            } else {
                1.0
            };
            lower_eta[component][component] = DiagonalLink::initial_eta_from_theta(scale);
        }

        (vector_eta, lower_eta)
    }
}

/// Multivariate normal predictors on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MvNormalCholeskyEta<const D: usize> {
    /// Mean predictors.
    mu: [f64; D],
    /// Lower-triangular Cholesky predictors. Upper-triangular entries are ignored.
    cholesky: FixedLowerTriangular<D>,
}

impl<const D: usize> MvNormalCholeskyEta<D> {
    /// Creates fixed-dimensional link-scale predictors.
    #[must_use]
    #[inline]
    pub const fn new(mu: [f64; D], cholesky: FixedLowerTriangular<D>) -> Self {
        Self { mu, cholesky }
    }

    /// Mean predictors.
    #[must_use]
    #[inline]
    pub const fn mu(&self) -> &[f64; D] {
        &self.mu
    }

    /// Lower-triangular Cholesky predictors.
    #[must_use]
    #[inline]
    pub const fn cholesky(&self) -> &FixedLowerTriangular<D> {
        &self.cholesky
    }

    /// Mutably borrows mean predictors.
    #[must_use]
    #[inline]
    pub const fn mu_mut(&mut self) -> &mut [f64; D] {
        &mut self.mu
    }

    /// Mutably borrows lower-triangular Cholesky predictors.
    #[must_use]
    #[inline]
    pub const fn cholesky_mut(&mut self) -> &mut FixedLowerTriangular<D> {
        &mut self.cholesky
    }
}

/// Multivariate normal parameters on the natural scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MvNormalCholeskyTheta<const D: usize> {
    /// Mean vector.
    mu: [f64; D],
    /// Lower-triangular Cholesky scale factor. Upper-triangular entries are ignored.
    cholesky: FixedLowerTriangular<D>,
}

impl<const D: usize> MvNormalCholeskyTheta<D> {
    /// Creates checked fixed-dimensional natural-scale parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] unless the dimension is
    /// positive, all entries are finite, and every Cholesky diagonal entry is
    /// strictly positive.
    #[inline]
    pub fn try_new(mu: [f64; D], cholesky: FixedLowerTriangular<D>) -> Result<Self, ModelError> {
        if !kernel::valid_theta(D, &mu, &cholesky) {
            return Err(ModelError::InvalidParameter {
                parameter: "multivariate normal theta",
                expected: "positive dimension, finite values, and positive Cholesky diagonal",
            });
        }
        Ok(Self { mu, cholesky })
    }

    #[inline]
    const fn from_parts_unchecked(mu: [f64; D], cholesky: FixedLowerTriangular<D>) -> Self {
        Self { mu, cholesky }
    }

    /// Mean vector.
    #[must_use]
    #[inline]
    pub const fn mu(&self) -> &[f64; D] {
        &self.mu
    }

    /// Lower-triangular Cholesky scale factor.
    #[must_use]
    #[inline]
    pub const fn cholesky(&self) -> &FixedLowerTriangular<D> {
        &self.cholesky
    }

    /// Returns one covariance entry from `L L'`.
    #[must_use]
    pub fn covariance(&self, row: usize, col: usize) -> Option<f64> {
        if row >= D || col >= D {
            return None;
        }
        let limit = row.min(col);
        Some(
            (0..=limit)
                .map(|index| self.cholesky.lower(row, index) * self.cholesky.lower(col, index))
                .sum(),
        )
    }

    /// Returns the marginal scale of one component.
    #[must_use]
    pub fn marginal_scale(&self, component: usize) -> Option<f64> {
        self.covariance(component, component).map(f64::sqrt)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{
        CholeskyScale, DenseDesign, Family, FixedDimensionalFamily, Gamlss, HasMarginalCdf,
        LinearPredictorBlock, LowerTriangularParameterBlock, Mu, NoPenalty, ObjectiveScale,
        ParameterBlocks, RidgePenalty, VectorParameterBlock,
    };

    use super::{FixedLowerTriangular, MvNormalCholeskyEta, MvNormalCholeskyTheta};
    use crate::constants::HALF_LOG_2_PI;
    use crate::multivariate::normal::MvNormalCholeskyDefault;
    use crate::{NormalMuSigma, NormalTheta};

    fn assert_fixed_dimensional_family<F: FixedDimensionalFamily<D>, const D: usize>() {}

    fn finite_difference_gradient<const D: usize>(
        y: [f64; D],
        eta: MvNormalCholeskyEta<D>,
        epsilon: f64,
        tolerance: f64,
    ) {
        let family = MvNormalCholeskyDefault::<D>::new();
        let (_, gradient) = family.nll_and_gradient_eta(y, &eta, &mut family.workspace());

        for index in 0..D {
            let mut plus = eta;
            plus.mu_mut()[index] += epsilon;
            let mut minus = eta;
            minus.mu_mut()[index] -= epsilon;
            let finite_difference = (family.nll_eta(y, &plus, &mut family.workspace())
                - family.nll_eta(y, &minus, &mut family.workspace()))
                / (2.0 * epsilon);
            assert_relative_eq!(gradient.mu()[index], finite_difference, epsilon = tolerance);
        }

        for row in 0..D {
            for col in 0..=row {
                let mut plus = eta;
                let current = plus.cholesky().get(row, col).unwrap();
                plus.cholesky_mut()
                    .set_lower(row, col, current + epsilon)
                    .unwrap();
                let mut minus = eta;
                let current = minus.cholesky().get(row, col).unwrap();
                minus
                    .cholesky_mut()
                    .set_lower(row, col, current - epsilon)
                    .unwrap();
                let finite_difference = (family.nll_eta(y, &plus, &mut family.workspace())
                    - family.nll_eta(y, &minus, &mut family.workspace()))
                    / (2.0 * epsilon);
                assert_relative_eq!(
                    gradient.cholesky().get(row, col).unwrap(),
                    finite_difference,
                    epsilon = tolerance
                );
            }
        }
    }

    fn intercept(n: usize) -> LinearPredictorBlock<DenseDesign> {
        LinearPredictorBlock::new(DenseDesign::intercept(n))
    }

    #[test]
    fn marker_trait_is_implemented() {
        assert_fixed_dimensional_family::<MvNormalCholeskyDefault<3>, 3>();
    }

    #[test]
    fn lower_triangular_get_is_panic_free_for_invalid_indices() {
        let matrix = FixedLowerTriangular::<2>::zeros();

        assert_eq!(matrix.get(2, 0), None);
        assert_eq!(matrix.get(0, 1), None);
        assert_eq!(matrix.get(usize::MAX, 0), None);
        assert_eq!(FixedLowerTriangular::<0>::zeros().get(0, 0), None);
    }

    #[test]
    fn gradient_matches_finite_difference_for_representative_dimensions() {
        finite_difference_gradient::<1>(
            [1.7],
            MvNormalCholeskyEta::new([0.4], FixedLowerTriangular::from_lower_rows([[-0.2]])),
            1.0e-6,
            1.0e-6,
        );
        finite_difference_gradient::<2>(
            [1.7, -0.8],
            MvNormalCholeskyEta::new(
                [0.4, -0.3],
                FixedLowerTriangular::from_lower_rows([[-0.2, 0.0], [0.25, 0.1]]),
            ),
            1.0e-6,
            1.0e-6,
        );
        finite_difference_gradient::<3>(
            [1.7, -0.8, 0.2],
            MvNormalCholeskyEta::new(
                [0.4, -0.3, 0.1],
                FixedLowerTriangular::from_lower_rows([
                    [-0.2, 0.0, 0.0],
                    [0.25, 0.1, 0.0],
                    [-0.1, 0.2, 0.3],
                ]),
            ),
            1.0e-6,
            1.0e-6,
        );
    }

    #[test]
    fn log_diagonal_gradient_stays_finite_for_subnormal_scale() {
        let family = MvNormalCholeskyDefault::<1>::new();
        let eta =
            MvNormalCholeskyEta::new([0.0], FixedLowerTriangular::from_lower_rows([[-710.0]]));
        let (nll, gradient) = family.nll_and_gradient_eta([0.0], &eta, &mut family.workspace());

        assert!(nll.is_finite());
        assert_relative_eq!(
            gradient.cholesky().get(0, 0).unwrap(),
            1.0,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn one_dimensional_case_matches_scalar_normal() {
        let mv = MvNormalCholeskyDefault::<1>::new();
        let normal = NormalMuSigma::new();
        let theta =
            MvNormalCholeskyTheta::try_new([0.4], FixedLowerTriangular::from_lower_rows([[0.8]]))
                .unwrap();
        let scalar_theta = NormalTheta {
            mu: 0.4,
            sigma: 0.8,
        };

        assert_relative_eq!(
            mv.nll([1.7], &theta, &mut mv.workspace()),
            normal.nll(1.7, &scalar_theta, &mut normal.workspace()),
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn diagonal_cholesky_matches_independent_scalar_normals() {
        let family = MvNormalCholeskyDefault::<3>::new();
        let theta = MvNormalCholeskyTheta::try_new(
            [0.4, -0.3, 0.1],
            FixedLowerTriangular::from_lower_rows([
                [0.8, 0.0, 0.0],
                [0.0, 1.2, 0.0],
                [0.0, 0.0, 0.5],
            ]),
        )
        .unwrap();
        let y = [1.7, -0.8, 0.2];
        let normal = NormalMuSigma::new();
        let expected = normal.nll(
            y[0],
            &NormalTheta {
                mu: theta.mu()[0],
                sigma: theta.cholesky().get(0, 0).unwrap(),
            },
            &mut normal.workspace(),
        ) + normal.nll(
            y[1],
            &NormalTheta {
                mu: theta.mu()[1],
                sigma: theta.cholesky().get(1, 1).unwrap(),
            },
            &mut normal.workspace(),
        ) + normal.nll(
            y[2],
            &NormalTheta {
                mu: theta.mu()[2],
                sigma: theta.cholesky().get(2, 2).unwrap(),
            },
            &mut normal.workspace(),
        );

        assert_relative_eq!(
            family.nll(y, &theta, &mut family.workspace()),
            expected,
            epsilon = 1.0e-12
        );
    }

    #[test]
    #[allow(clippy::manual_midpoint)]
    fn non_diagonal_nll_matches_hand_computed_value() {
        let family = MvNormalCholeskyDefault::<2>::new();
        let theta = MvNormalCholeskyTheta::try_new(
            [0.0, 0.0],
            FixedLowerTriangular::from_lower_rows([[2.0, 0.0], [1.0, 3.0]]),
        )
        .unwrap();
        let y = [4.0, 7.0];
        let z0 = 2.0;
        let z1 = (7.0 - z0) / 3.0;
        let expected =
            2.0 * HALF_LOG_2_PI + 2.0_f64.ln() + 3.0_f64.ln() + 0.5 * (z0 * z0 + z1 * z1);

        assert_relative_eq!(
            family.nll(y, &theta, &mut family.workspace()),
            expected,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn invalid_domains_return_infinite_nll_and_nan_gradient() {
        let family = MvNormalCholeskyDefault::<2>::new();
        let valid_eta = MvNormalCholeskyEta::new(
            [0.0, 0.0],
            FixedLowerTriangular::from_lower_rows([[0.0, 0.0], [0.2, 0.0]]),
        );
        assert!(
            MvNormalCholeskyTheta::try_new(
                [0.0, 0.0],
                FixedLowerTriangular::from_lower_rows([[1.0, 0.0], [0.2, 0.0]]),
            )
            .is_err()
        );
        assert!(
            MvNormalCholeskyTheta::try_new(
                [f64::NAN, 0.0],
                FixedLowerTriangular::from_lower_rows([[1.0, 0.0], [0.2, 1.0]]),
            )
            .is_err()
        );
        let invalid_theta = MvNormalCholeskyTheta::from_parts_unchecked(
            [0.0, 0.0],
            FixedLowerTriangular::from_lower_rows([[1.0, 0.0], [0.2, 0.0]]),
        );

        assert!(
            family
                .nll(
                    [f64::NAN, 0.0],
                    &family.theta(&valid_eta, &mut family.workspace()),
                    &mut family.workspace()
                )
                .is_infinite()
        );
        assert!(
            family
                .nll([0.0, 0.0], &invalid_theta, &mut family.workspace())
                .is_infinite()
        );

        let (nll, gradient) =
            family.nll_and_gradient_eta([f64::NAN, 0.0], &valid_eta, &mut family.workspace());
        assert!(nll.is_infinite());
        assert!(gradient.mu().iter().all(|value| value.is_nan()));
        assert!((0..2).all(|row| {
            (0..=row).all(|col| gradient.cholesky().get(row, col).unwrap().is_nan())
        }));
    }

    #[test]
    #[should_panic(expected = "multivariate normal dimension must be positive")]
    fn zero_dimension_is_invalid() {
        let _ = MvNormalCholeskyDefault::<0>::new();
    }

    #[test]
    fn zero_dimension_theta_is_rejected() {
        assert!(MvNormalCholeskyTheta::<0>::try_new([], FixedLowerTriangular::zeros()).is_err());
    }

    #[test]
    fn marginal_cdf_uses_component_variance() {
        let family = MvNormalCholeskyDefault::<2>::new();
        let theta = MvNormalCholeskyTheta::try_new(
            [0.0, 1.0],
            FixedLowerTriangular::from_lower_rows([[2.0, 0.0], [3.0, 4.0]]),
        )
        .unwrap();

        assert_eq!(theta.covariance(0, 0), Some(4.0));
        assert_eq!(theta.covariance(1, 0), Some(6.0));
        assert_eq!(theta.marginal_scale(1), Some(5.0));
        assert_eq!(theta.covariance(2, 0), None);
        assert_relative_eq!(family.marginal_cdf(0, 0.0, &theta), 0.5, epsilon = 1.0e-12);
        assert_relative_eq!(
            family.marginal_cdf(1, 6.0, &theta),
            0.841_344_746,
            epsilon = 1.0e-9
        );
        assert!(family.marginal_cdf(2, 0.0, &theta).is_nan());
    }

    #[test]
    fn structured_blocks_compile_multivariate_gamlss_objective() {
        let y = [[1.0, 0.5], [0.2, -0.3], [1.3, 0.7]];
        let n = y.len();
        let mu =
            VectorParameterBlock::<Mu, 2, _, _>::new([intercept(n), intercept(n)], NoPenalty, 99);
        let cholesky = LowerTriangularParameterBlock::<CholeskyScale, 2, _, _>::new(
            vec![intercept(n), intercept(n), intercept(n)],
            NoPenalty,
            99,
        );
        let blocks = ParameterBlocks::new((mu, cholesky));
        let model = Gamlss::try_new_with_observations(
            MvNormalCholeskyDefault::<2>::new(),
            blocks,
            y.as_slice(),
        )
        .unwrap();

        assert_eq!(model.nparams(), 5);
        assert_eq!(model.parameter_layout().slice("mu"), Some(0..2));
        assert_eq!(model.parameter_layout().slice("cholesky"), Some(2..5));

        let beta = vec![0.1, -0.2, 0.0, 0.3, 0.1];
        let eta = model.predict_eta_row(&beta, 0).unwrap();
        assert_relative_eq!(eta.mu()[0], 0.1, epsilon = 1.0e-12);
        assert_relative_eq!(eta.mu()[1], -0.2, epsilon = 1.0e-12);
        assert_relative_eq!(eta.cholesky().get(0, 0).unwrap(), 0.0, epsilon = 1.0e-12);
        assert_relative_eq!(eta.cholesky().get(1, 0).unwrap(), 0.3, epsilon = 1.0e-12);
        assert_relative_eq!(eta.cholesky().get(1, 1).unwrap(), 0.1, epsilon = 1.0e-12);

        let mut gradient = vec![0.0; model.nparams()];
        let value = model.try_value_gradient_into(&beta, &mut gradient).unwrap();
        assert!(value.is_finite());
        assert!(gradient.iter().all(|value| value.is_finite()));

        let epsilon = 1.0e-6;
        for index in 0..beta.len() {
            let mut plus = beta.clone();
            plus[index] += epsilon;
            let mut minus = beta.clone();
            minus[index] -= epsilon;
            let finite_difference = (model.try_value(&plus).unwrap()
                - model.try_value(&minus).unwrap())
                / (2.0 * epsilon);
            assert_relative_eq!(gradient[index], finite_difference, epsilon = 1.0e-6);
        }
    }

    #[test]
    fn structured_blocks_use_data_aware_initial_parameters() {
        let y = [[1.0, 2.0], [3.0, 6.0]];
        let n = y.len();
        let mu =
            VectorParameterBlock::<Mu, 2, _, _>::new([intercept(n), intercept(n)], NoPenalty, 99);
        let cholesky = LowerTriangularParameterBlock::<CholeskyScale, 2, _, _>::new(
            vec![intercept(n), intercept(n), intercept(n)],
            NoPenalty,
            99,
        );
        let blocks = ParameterBlocks::new((mu, cholesky));
        let model = Gamlss::try_new_with_observations(
            MvNormalCholeskyDefault::<2>::new(),
            blocks,
            y.as_slice(),
        )
        .unwrap();

        let beta = model.initial_parameters().unwrap();

        assert_relative_eq!(beta[0], 2.0, epsilon = 1.0e-12);
        assert_relative_eq!(beta[1], 4.0, epsilon = 1.0e-12);
        assert_relative_eq!(beta[2], 0.0, epsilon = 1.0e-12);
        assert_relative_eq!(beta[3], 0.0, epsilon = 1.0e-12);
        assert_relative_eq!(beta[4], 2.0_f64.ln(), epsilon = 1.0e-12);
    }

    #[test]
    fn mean_objective_scales_likelihood_but_not_structured_penalties() {
        type PenalizedMvBlocks = ParameterBlocks<(
            VectorParameterBlock<Mu, 2, LinearPredictorBlock<DenseDesign>, RidgePenalty>,
            LowerTriangularParameterBlock<
                CholeskyScale,
                2,
                LinearPredictorBlock<DenseDesign>,
                NoPenalty,
            >,
        )>;

        fn model_with_scale(
            scale: ObjectiveScale,
        ) -> Gamlss<MvNormalCholeskyDefault<2>, PenalizedMvBlocks, &'static [[f64; 2]]> {
            static Y: [[f64; 2]; 3] = [[1.0, 0.5], [0.2, -0.3], [1.3, 0.7]];
            let n = Y.len();
            let mu = VectorParameterBlock::<Mu, 2, _, _>::new(
                [intercept(n), intercept(n)],
                RidgePenalty::new_unchecked(0.5),
                99,
            );
            let cholesky = LowerTriangularParameterBlock::<CholeskyScale, 2, _, _>::new(
                vec![intercept(n), intercept(n), intercept(n)],
                NoPenalty,
                99,
            );
            Gamlss::try_new_with_observations(
                MvNormalCholeskyDefault::<2>::new(),
                ParameterBlocks::new((mu, cholesky)),
                Y.as_slice(),
            )
            .unwrap()
            .with_objective_scale(scale)
        }

        let beta = vec![0.1, -0.2, 0.0, 0.3, 0.1];
        let sum_model = model_with_scale(ObjectiveScale::Sum);
        let mean_model = model_with_scale(ObjectiveScale::Mean);
        let penalty = f64::midpoint(beta[0] * beta[0], beta[1] * beta[1]);
        let expected_mean = (sum_model.try_value(&beta).unwrap() - penalty) / 3.0 + penalty;

        assert_relative_eq!(mean_model.try_value(&beta).unwrap(), expected_mean);
    }

    #[cfg(feature = "rand")]
    #[test]
    fn sampling_returns_finite_values_or_nan_for_invalid_theta() {
        use gamlss_core::CanSimulate;

        let family = MvNormalCholeskyDefault::<2>::new();
        let mut rng = rand::rng();
        let valid = family.sample(
            &mut rng,
            &MvNormalCholeskyTheta::try_new(
                [0.0, 1.0],
                FixedLowerTriangular::from_lower_rows([[2.0, 0.0], [3.0, 4.0]]),
            )
            .unwrap(),
        );
        assert!(valid.iter().all(|value| value.is_finite()));

        let invalid = family.sample(
            &mut rng,
            &MvNormalCholeskyTheta::from_parts_unchecked(
                [0.0, 1.0],
                FixedLowerTriangular::from_lower_rows([[2.0, 0.0], [3.0, 0.0]]),
            ),
        );
        assert!(invalid.iter().all(|value| value.is_nan()));
    }
}
