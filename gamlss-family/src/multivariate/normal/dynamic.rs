use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{Family, HasMarginalCdf, Identity, Link, Log, ModelError, PositiveLink};
use gamlss_special::unit_normal_cdf;

use super::kernel;

/// Runtime-dimensional packed lower-triangular matrix storage.
///
/// Values are stored in row-major lower-triangular order:
/// `(0,0), (1,0), (1,1), (2,0), ...`.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedLowerTriangular {
    dimension: usize,
    values: Vec<f64>,
}

impl PackedLowerTriangular {
    /// Creates packed lower-triangular storage after validating its length.
    pub fn try_new(dimension: usize, values: Vec<f64>) -> Result<Self, ModelError> {
        if dimension == 0 {
            return Err(ModelError::InvalidParameter {
                parameter: "dimension",
                expected: "positive",
            });
        }
        let expected = kernel::cholesky_len(dimension).ok_or(ModelError::ArithmeticOverflow {
            context: "lower-triangular storage length",
        })?;
        if values.len() != expected {
            return Err(ModelError::InvalidParameter {
                parameter: "cholesky",
                expected: "D * (D + 1) / 2 lower-triangular values",
            });
        }
        Ok(Self { dimension, values })
    }

    fn filled(dimension: usize, value: f64) -> Self {
        Self {
            dimension,
            values: vec![value; kernel::cholesky_len(dimension).unwrap_or(0)],
        }
    }

    /// Observation/parameter dimension.
    #[must_use]
    #[inline]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Packed lower-triangular values.
    #[must_use]
    #[inline]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Mutable packed lower-triangular values.
    #[must_use]
    #[inline]
    pub fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }

    /// Returns a lower-triangular entry, or `None` for invalid/upper entries.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row < self.dimension && col <= row {
            kernel::packed_index(row, col).and_then(|index| self.values.get(index).copied())
        } else {
            None
        }
    }

    /// Returns a mutable lower-triangular entry, or `None` for invalid/upper entries.
    #[must_use]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut f64> {
        if row < self.dimension && col <= row {
            kernel::packed_index(row, col).and_then(|index| self.values.get_mut(index))
        } else {
            None
        }
    }

    #[inline]
    fn lower(&self, row: usize, col: usize) -> f64 {
        self.values[kernel::packed_index(row, col).expect("valid lower-triangular index")]
    }
}

impl kernel::LowerTriangularMatrix for PackedLowerTriangular {
    #[inline]
    fn dimension(&self) -> usize {
        self.dimension
    }

    #[inline]
    fn lower(&self, row: usize, col: usize) -> f64 {
        self.lower(row, col)
    }
}

/// Runtime-dimensional multivariate normal parameterized by mean and Cholesky scale.
///
/// The natural-scale `cholesky` matrix uses packed lower-triangular storage.
/// Diagonal entries are transformed with `DiagonalLink`; off-diagonal lower
/// entries are transformed with `OffDiagonalLink`; upper entries are ignored
/// and normalized to zero by [`Family::theta`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynMvNormalCholesky<MuLink = Identity, DiagonalLink = Log, OffDiagonalLink = Identity> {
    dimension: usize,
    marker: PhantomData<(MuLink, DiagonalLink, OffDiagonalLink)>,
}

impl<MuLink, DiagonalLink, OffDiagonalLink>
    DynMvNormalCholesky<MuLink, DiagonalLink, OffDiagonalLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    /// Creates a stateless family value for a runtime dimension.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `dimension == 0`.
    #[inline]
    pub const fn new(dimension: usize) -> Result<Self, ModelError> {
        if dimension == 0 {
            return Err(ModelError::InvalidParameter {
                parameter: "dimension",
                expected: "positive",
            });
        }
        Ok(Self {
            dimension,
            marker: PhantomData,
        })
    }

    /// Returns the configured observation dimension.
    #[must_use]
    #[inline]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the expected packed lower-triangular Cholesky storage length.
    #[must_use]
    #[inline]
    pub const fn cholesky_len_for_dimension(dimension: usize) -> Option<usize> {
        kernel::cholesky_len(dimension)
    }

    const fn eta_has_expected_len(&self, eta: &DynMvNormalCholeskyEta) -> bool {
        eta.mu.len() == self.dimension && eta.cholesky.dimension() == self.dimension
    }

    fn nan_eta(&self) -> DynMvNormalCholeskyEta {
        DynMvNormalCholeskyEta {
            mu: vec![f64::NAN; self.dimension],
            cholesky: PackedLowerTriangular::filled(self.dimension, f64::NAN),
        }
    }

    fn nan_theta(&self) -> DynMvNormalCholeskyTheta {
        DynMvNormalCholeskyTheta {
            mu: vec![f64::NAN; self.dimension],
            cholesky: PackedLowerTriangular::filled(self.dimension, f64::NAN),
        }
    }

    fn theta_from_eta(&self, eta: &DynMvNormalCholeskyEta) -> DynMvNormalCholeskyTheta {
        if !self.eta_has_expected_len(eta) {
            return self.nan_theta();
        }

        let mut mu = vec![0.0; self.dimension];
        let mut cholesky = PackedLowerTriangular::filled(self.dimension, 0.0);

        for index in 0..self.dimension {
            mu[index] = MuLink::inverse(eta.mu[index]);
        }

        for row in 0..self.dimension {
            for col in 0..=row {
                let value = if row == col {
                    DiagonalLink::inverse(eta.cholesky.lower(row, col))
                } else {
                    OffDiagonalLink::inverse(eta.cholesky.lower(row, col))
                };
                *cholesky
                    .get_mut(row, col)
                    .expect("row and col are valid lower-triangular indices") = value;
            }
        }

        DynMvNormalCholeskyTheta { mu, cholesky }
    }

    fn nll_theta(&self, observation: &[f64], theta: &DynMvNormalCholeskyTheta) -> f64 {
        let mut z = vec![0.0; self.dimension];
        kernel::nll(
            self.dimension,
            observation,
            &theta.mu,
            &theta.cholesky,
            &mut z,
        )
    }

    fn nll_and_gradient_eta_values(
        &self,
        observation: &[f64],
        eta: &DynMvNormalCholeskyEta,
    ) -> (f64, DynMvNormalCholeskyEta) {
        if !self.eta_has_expected_len(eta) {
            return (f64::INFINITY, self.nan_eta());
        }

        let theta = self.theta_from_eta(eta);
        let mut z = vec![0.0; self.dimension];
        let mut a = vec![0.0; self.dimension];
        let nll = kernel::nll_and_score(
            self.dimension,
            observation,
            &theta.mu,
            &theta.cholesky,
            &mut z,
            &mut a,
        );
        if !nll.is_finite() {
            return (nll, self.nan_eta());
        }

        let mut gradient = DynMvNormalCholeskyEta {
            mu: vec![0.0; self.dimension],
            cholesky: PackedLowerTriangular::filled(self.dimension, 0.0),
        };

        for index in 0..self.dimension {
            gradient.mu[index] = -a[index] * MuLink::derivative_inverse(eta.mu[index]);
        }

        for row in 0..self.dimension {
            for (col, z_col) in z.iter().copied().take(row + 1).enumerate() {
                let mut d_nll_d_l = kernel::cholesky_score(row, col, z_col, &a, &theta.cholesky);
                if row == col {
                    d_nll_d_l *= DiagonalLink::derivative_inverse(eta.cholesky.lower(row, col));
                } else {
                    d_nll_d_l *= OffDiagonalLink::derivative_inverse(eta.cholesky.lower(row, col));
                }
                *gradient
                    .cholesky
                    .get_mut(row, col)
                    .expect("row and col are valid lower-triangular indices") = d_nll_d_l;
            }
        }

        (nll, gradient)
    }

    fn marginal_scale(&self, component: usize, theta: &DynMvNormalCholeskyTheta) -> f64 {
        kernel::marginal_scale(self.dimension, component, &theta.mu, &theta.cholesky)
    }
}

impl<MuLink, DiagonalLink, OffDiagonalLink> Default
    for DynMvNormalCholesky<MuLink, DiagonalLink, OffDiagonalLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    fn default() -> Self {
        Self {
            dimension: 1,
            marker: PhantomData,
        }
    }
}

impl<MuLink, DiagonalLink, OffDiagonalLink> Family
    for DynMvNormalCholesky<MuLink, DiagonalLink, OffDiagonalLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    type Eta = DynMvNormalCholeskyEta;
    type Theta = DynMvNormalCholeskyTheta;
    type NllGradientEta = DynMvNormalCholeskyEta;
    type Observation<'obs> = &'obs [f64];

    #[inline]
    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        self.theta_from_eta(&eta)
    }

    #[inline]
    fn nll(&self, observation: Self::Observation<'_>, theta: Self::Theta) -> f64 {
        self.nll_theta(observation, &theta)
    }

    #[inline]
    fn nll_eta(&self, observation: Self::Observation<'_>, eta: Self::Eta) -> f64 {
        self.nll_theta(observation, &self.theta_from_eta(&eta))
    }

    #[inline]
    fn nll_and_gradient_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: Self::Eta,
    ) -> (f64, Self::NllGradientEta) {
        self.nll_and_gradient_eta_values(observation, &eta)
    }
}

impl<MuLink, DiagonalLink, OffDiagonalLink> HasMarginalCdf
    for DynMvNormalCholesky<MuLink, DiagonalLink, OffDiagonalLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    fn marginal_cdf(&self, component: usize, y: f64, theta: Self::Theta) -> f64 {
        if !y.is_finite() || component >= self.dimension {
            return f64::NAN;
        }

        let scale = self.marginal_scale(component, &theta);
        if !scale.is_finite() || scale <= 0.0 {
            return f64::NAN;
        }

        unit_normal_cdf((y - theta.mu[component]) / scale)
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink, DiagonalLink, OffDiagonalLink> CanSimulate<Rng>
    for DynMvNormalCholesky<MuLink, DiagonalLink, OffDiagonalLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    type Sample = Vec<f64>;

    fn sample(&self, rng: &mut Rng, theta: Self::Theta) -> Self::Sample {
        if !kernel::valid_theta(self.dimension, &theta.mu, &theta.cholesky) {
            return vec![f64::NAN; self.dimension];
        }

        let standard = rand_distr::StandardNormal;
        let mut z = vec![0.0; self.dimension];
        for value in &mut z {
            *value = rand_distr::Distribution::sample(&standard, rng);
        }

        let mut out = theta.mu;
        for row in 0..self.dimension {
            for (col, z_col) in z.iter().copied().take(row + 1).enumerate() {
                out[row] += theta.cholesky.lower(row, col) * z_col;
            }
        }
        out
    }
}

/// Runtime-dimensional multivariate normal predictors on the link scale.
#[derive(Debug, Clone, PartialEq)]
pub struct DynMvNormalCholeskyEta {
    /// Mean predictors.
    mu: Vec<f64>,
    /// Packed lower-triangular Cholesky predictor storage.
    cholesky: PackedLowerTriangular,
}

impl DynMvNormalCholeskyEta {
    /// Creates a runtime-dimensional eta container from owned parts.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `mu.len()` does not match
    /// the Cholesky dimension.
    pub fn new(mu: Vec<f64>, cholesky: PackedLowerTriangular) -> Result<Self, ModelError> {
        if mu.len() != cholesky.dimension() {
            return Err(ModelError::InvalidParameter {
                parameter: "mu",
                expected: "same length as Cholesky dimension",
            });
        }
        Ok(Self { mu, cholesky })
    }

    /// Mean predictors.
    #[must_use]
    #[inline]
    pub fn mu(&self) -> &[f64] {
        &self.mu
    }

    /// Mutable mean predictors.
    #[must_use]
    #[inline]
    pub fn mu_mut(&mut self) -> &mut [f64] {
        &mut self.mu
    }

    /// Packed lower-triangular Cholesky predictors.
    #[must_use]
    #[inline]
    pub const fn cholesky(&self) -> &PackedLowerTriangular {
        &self.cholesky
    }

    /// Mutable packed lower-triangular Cholesky predictors.
    #[must_use]
    #[inline]
    pub const fn cholesky_mut(&mut self) -> &mut PackedLowerTriangular {
        &mut self.cholesky
    }

    /// Returns a Cholesky predictor entry.
    #[must_use]
    #[inline]
    pub fn cholesky_entry(&self, row: usize, col: usize) -> Option<f64> {
        self.cholesky.get(row, col)
    }

    /// Returns a mutable Cholesky predictor entry.
    #[must_use]
    #[inline]
    pub fn cholesky_entry_mut(&mut self, row: usize, col: usize) -> Option<&mut f64> {
        self.cholesky.get_mut(row, col)
    }
}

/// Runtime-dimensional multivariate normal parameters on the natural scale.
#[derive(Debug, Clone, PartialEq)]
pub struct DynMvNormalCholeskyTheta {
    /// Mean vector.
    mu: Vec<f64>,
    /// Packed lower-triangular Cholesky scale-factor storage.
    cholesky: PackedLowerTriangular,
}

impl DynMvNormalCholeskyTheta {
    /// Creates a runtime-dimensional theta container from owned parts.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `mu.len()` does not match
    /// the Cholesky dimension.
    pub fn new(mu: Vec<f64>, cholesky: PackedLowerTriangular) -> Result<Self, ModelError> {
        if mu.len() != cholesky.dimension() {
            return Err(ModelError::InvalidParameter {
                parameter: "mu",
                expected: "same length as Cholesky dimension",
            });
        }
        Ok(Self { mu, cholesky })
    }

    /// Mean vector.
    #[must_use]
    #[inline]
    pub fn mu(&self) -> &[f64] {
        &self.mu
    }

    /// Packed lower-triangular Cholesky scale factor.
    #[must_use]
    #[inline]
    pub const fn cholesky(&self) -> &PackedLowerTriangular {
        &self.cholesky
    }

    /// Returns a Cholesky scale-factor entry.
    #[must_use]
    #[inline]
    pub fn cholesky_entry(&self, row: usize, col: usize) -> Option<f64> {
        self.cholesky.get(row, col)
    }

    /// Returns a mutable Cholesky scale-factor entry.
    #[must_use]
    #[inline]
    pub fn cholesky_entry_mut(&mut self, row: usize, col: usize) -> Option<&mut f64> {
        self.cholesky.get_mut(row, col)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{Family, HasMarginalCdf};

    use super::{DynMvNormalCholeskyEta, DynMvNormalCholeskyTheta, PackedLowerTriangular};
    use crate::multivariate::normal::DynMvNormalCholeskyDefault;

    fn finite_difference_dynamic_gradient(
        y: &[f64],
        eta: &DynMvNormalCholeskyEta,
        epsilon: f64,
        tolerance: f64,
    ) {
        let family = DynMvNormalCholeskyDefault::new(y.len()).unwrap();
        let (_, gradient) = family.nll_and_gradient_eta(y, eta.clone());

        for index in 0..eta.mu().len() {
            let mut plus = eta.clone();
            plus.mu_mut()[index] += epsilon;
            let mut minus = eta.clone();
            minus.mu_mut()[index] -= epsilon;
            let finite_difference =
                (family.nll_eta(y, plus) - family.nll_eta(y, minus)) / (2.0 * epsilon);
            assert_relative_eq!(gradient.mu()[index], finite_difference, epsilon = tolerance);
        }

        for row in 0..y.len() {
            for col in 0..=row {
                let mut plus = eta.clone();
                *plus.cholesky_entry_mut(row, col).unwrap() += epsilon;
                let mut minus = eta.clone();
                *minus.cholesky_entry_mut(row, col).unwrap() -= epsilon;
                let finite_difference =
                    (family.nll_eta(y, plus) - family.nll_eta(y, minus)) / (2.0 * epsilon);
                assert_relative_eq!(
                    gradient.cholesky_entry(row, col).unwrap(),
                    finite_difference,
                    epsilon = tolerance
                );
            }
        }
    }

    #[test]
    fn dynamic_gradient_matches_finite_difference() {
        finite_difference_dynamic_gradient(
            &[1.7, -0.8, 0.2],
            &DynMvNormalCholeskyEta::new(
                vec![0.4, -0.3, 0.1],
                PackedLowerTriangular::try_new(3, vec![-0.2, 0.25, 0.1, -0.1, 0.2, 0.3]).unwrap(),
            )
            .unwrap(),
            1.0e-6,
            1.0e-6,
        );
    }

    #[test]
    fn invalid_lengths_return_constructor_errors() {
        assert!(DynMvNormalCholeskyDefault::new(0).is_err());
        assert!(PackedLowerTriangular::try_new(2, vec![0.0; 4]).is_err());
        assert!(
            DynMvNormalCholeskyEta::new(
                vec![0.0],
                PackedLowerTriangular::try_new(2, vec![0.0; 3]).unwrap(),
            )
            .is_err()
        );
    }

    #[test]
    fn marginal_cdf_uses_component_variance() {
        let family = DynMvNormalCholeskyDefault::new(2).unwrap();
        let theta = DynMvNormalCholeskyTheta::new(
            vec![0.0, 1.0],
            PackedLowerTriangular::try_new(2, vec![2.0, 3.0, 4.0]).unwrap(),
        )
        .unwrap();

        assert_eq!(
            DynMvNormalCholeskyDefault::cholesky_len_for_dimension(2),
            Some(3)
        );
        assert_eq!(theta.cholesky_entry(1, 0), Some(3.0));
        assert_eq!(theta.cholesky_entry(2, 0), None);
        assert_relative_eq!(
            family.marginal_cdf(0, 0.0, theta.clone()),
            0.5,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(
            family.marginal_cdf(1, 6.0, theta),
            0.841_344_746,
            epsilon = 1.0e-9
        );
    }

    #[cfg(feature = "rand")]
    #[test]
    fn sampling_returns_finite_values_for_valid_theta() {
        use gamlss_core::CanSimulate;

        let family = DynMvNormalCholeskyDefault::new(2).unwrap();
        let mut rng = rand::rng();
        let valid = family.sample(
            &mut rng,
            DynMvNormalCholeskyTheta::new(
                vec![0.0, 1.0],
                PackedLowerTriangular::try_new(2, vec![2.0, 3.0, 4.0]).unwrap(),
            )
            .unwrap(),
        );
        assert_eq!(valid.len(), 2);
        assert!(valid.iter().all(|value| value.is_finite()));
    }
}
