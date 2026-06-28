use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{Family, HasMarginalCdf, Identity, Link, Log, PositiveLink};
use gamlss_special::unit_normal_cdf;

use super::kernel;

/// Runtime-dimensional multivariate normal parameterized by mean and Cholesky scale.
///
/// The natural-scale `cholesky` matrix uses full row-major `D x D` storage.
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
    #[must_use]
    #[inline]
    pub const fn new(dimension: usize) -> Self {
        Self {
            dimension,
            marker: PhantomData,
        }
    }

    /// Returns the configured observation dimension.
    #[must_use]
    #[inline]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the expected full row-major Cholesky storage length for this dimension.
    #[must_use]
    #[inline]
    pub const fn cholesky_len_for_dimension(dimension: usize) -> Option<usize> {
        kernel::cholesky_len(dimension)
    }

    const fn cholesky_len(&self) -> Option<usize> {
        kernel::cholesky_len(self.dimension)
    }

    fn eta_has_expected_len(&self, eta: &DynMvNormalCholeskyEta) -> bool {
        self.cholesky_len()
            .is_some_and(|len| eta.mu.len() == self.dimension && eta.cholesky.len() == len)
    }

    fn nan_eta(&self) -> DynMvNormalCholeskyEta {
        DynMvNormalCholeskyEta {
            mu: vec![f64::NAN; self.dimension],
            cholesky: vec![f64::NAN; self.cholesky_len().unwrap_or(0)],
        }
    }

    fn nan_theta(&self) -> DynMvNormalCholeskyTheta {
        DynMvNormalCholeskyTheta {
            mu: vec![f64::NAN; self.dimension],
            cholesky: vec![f64::NAN; self.cholesky_len().unwrap_or(0)],
        }
    }

    fn theta_from_eta(&self, eta: &DynMvNormalCholeskyEta) -> DynMvNormalCholeskyTheta {
        if !self.eta_has_expected_len(eta) {
            return self.nan_theta();
        }

        let mut mu = vec![0.0; self.dimension];
        let mut cholesky = vec![0.0; eta.cholesky.len()];

        for index in 0..self.dimension {
            mu[index] = MuLink::inverse(eta.mu[index]);
        }

        for row in 0..self.dimension {
            for col in 0..=row {
                let index = kernel::matrix_index(self.dimension, row, col);
                cholesky[index] = if row == col {
                    DiagonalLink::inverse(eta.cholesky[index])
                } else {
                    OffDiagonalLink::inverse(eta.cholesky[index])
                };
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
            cholesky: vec![0.0; eta.cholesky.len()],
        };

        for index in 0..self.dimension {
            gradient.mu[index] = -a[index] * MuLink::derivative_inverse(eta.mu[index]);
        }

        for row in 0..self.dimension {
            for (col, z_col) in z.iter().copied().take(row + 1).enumerate() {
                let index = kernel::matrix_index(self.dimension, row, col);
                let mut d_nll_d_l =
                    kernel::cholesky_score(self.dimension, row, col, z_col, &a, &theta.cholesky);
                if row == col {
                    d_nll_d_l *= DiagonalLink::derivative_inverse(eta.cholesky[index]);
                } else {
                    d_nll_d_l *= OffDiagonalLink::derivative_inverse(eta.cholesky[index]);
                }
                gradient.cholesky[index] = d_nll_d_l;
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
        Self::new(1)
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
                out[row] += theta.cholesky[kernel::matrix_index(self.dimension, row, col)] * z_col;
            }
        }
        out
    }
}

/// Runtime-dimensional multivariate normal predictors on the link scale.
#[derive(Debug, Clone, PartialEq)]
pub struct DynMvNormalCholeskyEta {
    /// Mean predictors.
    pub mu: Vec<f64>,
    /// Full row-major `D x D` Cholesky predictor storage. Upper-triangular entries are ignored.
    pub cholesky: Vec<f64>,
}

impl DynMvNormalCholeskyEta {
    /// Creates a runtime-dimensional eta container from owned parts.
    #[must_use]
    #[inline]
    pub const fn new(mu: Vec<f64>, cholesky: Vec<f64>) -> Self {
        Self { mu, cholesky }
    }

    /// Returns a Cholesky predictor entry for full row-major `dimension x dimension` storage.
    #[must_use]
    #[inline]
    pub fn cholesky_entry(&self, dimension: usize, row: usize, col: usize) -> Option<f64> {
        kernel::checked_matrix_index(dimension, row, col)
            .and_then(|index| self.cholesky.get(index).copied())
    }

    /// Returns a mutable Cholesky predictor entry for full row-major `dimension x dimension` storage.
    #[must_use]
    #[inline]
    pub fn cholesky_entry_mut(
        &mut self,
        dimension: usize,
        row: usize,
        col: usize,
    ) -> Option<&mut f64> {
        kernel::checked_matrix_index(dimension, row, col)
            .and_then(|index| self.cholesky.get_mut(index))
    }
}

/// Runtime-dimensional multivariate normal parameters on the natural scale.
#[derive(Debug, Clone, PartialEq)]
pub struct DynMvNormalCholeskyTheta {
    /// Mean vector.
    pub mu: Vec<f64>,
    /// Full row-major `D x D` Cholesky scale-factor storage. Upper-triangular entries are ignored.
    pub cholesky: Vec<f64>,
}

impl DynMvNormalCholeskyTheta {
    /// Creates a runtime-dimensional theta container from owned parts.
    #[must_use]
    #[inline]
    pub const fn new(mu: Vec<f64>, cholesky: Vec<f64>) -> Self {
        Self { mu, cholesky }
    }

    /// Returns a Cholesky scale-factor entry for full row-major `dimension x dimension` storage.
    #[must_use]
    #[inline]
    pub fn cholesky_entry(&self, dimension: usize, row: usize, col: usize) -> Option<f64> {
        kernel::checked_matrix_index(dimension, row, col)
            .and_then(|index| self.cholesky.get(index).copied())
    }

    /// Returns a mutable Cholesky scale-factor entry for full row-major `dimension x dimension` storage.
    #[must_use]
    #[inline]
    pub fn cholesky_entry_mut(
        &mut self,
        dimension: usize,
        row: usize,
        col: usize,
    ) -> Option<&mut f64> {
        kernel::checked_matrix_index(dimension, row, col)
            .and_then(|index| self.cholesky.get_mut(index))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{Family, HasMarginalCdf};

    use super::{DynMvNormalCholeskyEta, DynMvNormalCholeskyTheta};
    use crate::multivariate::normal::DynMvNormalCholeskyDefault;

    fn finite_difference_dynamic_gradient(
        y: &[f64],
        eta: &DynMvNormalCholeskyEta,
        epsilon: f64,
        tolerance: f64,
    ) {
        let family = DynMvNormalCholeskyDefault::new(y.len());
        let (_, gradient) = family.nll_and_gradient_eta(y, eta.clone());

        for index in 0..eta.mu.len() {
            let mut plus = eta.clone();
            plus.mu[index] += epsilon;
            let mut minus = eta.clone();
            minus.mu[index] -= epsilon;
            let finite_difference =
                (family.nll_eta(y, plus) - family.nll_eta(y, minus)) / (2.0 * epsilon);
            assert_relative_eq!(gradient.mu[index], finite_difference, epsilon = tolerance);
        }

        for row in 0..y.len() {
            for col in 0..=row {
                let mut plus = eta.clone();
                *plus.cholesky_entry_mut(y.len(), row, col).unwrap() += epsilon;
                let mut minus = eta.clone();
                *minus.cholesky_entry_mut(y.len(), row, col).unwrap() -= epsilon;
                let finite_difference =
                    (family.nll_eta(y, plus) - family.nll_eta(y, minus)) / (2.0 * epsilon);
                assert_relative_eq!(
                    gradient.cholesky_entry(y.len(), row, col).unwrap(),
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
                vec![-0.2, 0.0, 0.0, 0.25, 0.1, 0.0, -0.1, 0.2, 0.3],
            ),
            1.0e-6,
            1.0e-6,
        );
    }

    #[test]
    fn invalid_lengths_return_infinite_nll_and_nan_gradient() {
        let family = DynMvNormalCholeskyDefault::new(2);
        let bad_eta = DynMvNormalCholeskyEta::new(vec![0.0], vec![0.0; 4]);
        let theta = family.theta(bad_eta.clone());

        assert!(family.nll(&[0.0, 0.0], theta).is_infinite());

        let (nll, gradient) = family.nll_and_gradient_eta(&[0.0, 0.0], bad_eta);
        assert!(nll.is_infinite());
        assert_eq!(gradient.mu.len(), 2);
        assert_eq!(gradient.cholesky.len(), 4);
        assert!(gradient.mu.iter().all(|value| value.is_nan()));
        assert!(gradient.cholesky.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn marginal_cdf_uses_component_variance() {
        let family = DynMvNormalCholeskyDefault::new(2);
        let theta = DynMvNormalCholeskyTheta::new(vec![0.0, 1.0], vec![2.0, 0.0, 3.0, 4.0]);

        assert_eq!(
            DynMvNormalCholeskyDefault::cholesky_len_for_dimension(2),
            Some(4)
        );
        assert_eq!(theta.cholesky_entry(2, 1, 0), Some(3.0));
        assert_eq!(theta.cholesky_entry(2, 2, 0), None);
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

        let family = DynMvNormalCholeskyDefault::new(2);
        let mut rng = rand::rng();
        let valid = family.sample(
            &mut rng,
            DynMvNormalCholeskyTheta::new(vec![0.0, 1.0], vec![2.0, 0.0, 3.0, 4.0]),
        );
        assert_eq!(valid.len(), 2);
        assert!(valid.iter().all(|value| value.is_finite()));
    }
}
