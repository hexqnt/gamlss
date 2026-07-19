use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::{CanSimulate, SimulationError, TrySimulate};
use gamlss_core::{
    CholeskyScale, DynamicLayoutKey, DynamicallyCompilableFamily, Family, HasConditionalCdf,
    HasMarginalCdf, HasObservationDimension, HasRosenblattTransform, Identity, Link, Log,
    ModelError, Mu, ParameterAxis, ParameterName, ParameterPath, PositiveLink,
};
use gamlss_special::unit_normal_cdf;

use crate::multivariate::{
    cholesky::{packed_index, packed_len},
    matrix::PackedLowerTriangular,
};

use super::kernel;

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
    /// Returns [`ModelError::InvalidParameter`] when `dimension == 0` and
    /// [`ModelError::ArithmeticOverflow`] when its packed triangle length is
    /// not representable.
    #[inline]
    pub const fn new(dimension: usize) -> Result<Self, ModelError> {
        if dimension == 0 {
            return Err(ModelError::InvalidParameter {
                parameter: "dimension",
                expected: "positive",
            });
        }
        if packed_len(dimension).is_none() {
            return Err(ModelError::ArithmeticOverflow {
                context: "dynamic lower-triangular storage length",
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
        packed_len(dimension)
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
        DynMvNormalCholeskyTheta::from_parts_unchecked(
            vec![f64::NAN; self.dimension],
            PackedLowerTriangular::filled(self.dimension, f64::NAN),
        )
    }

    fn theta_from_eta(&self, eta: &DynMvNormalCholeskyEta) -> DynMvNormalCholeskyTheta {
        if !self.eta_has_expected_len(eta) {
            return self.nan_theta();
        }

        let mut mu = vec![0.0; self.dimension];
        let mut cholesky = PackedLowerTriangular::filled(self.dimension, 0.0);

        for (mu, eta_mu) in mu.iter_mut().zip(eta.mu.iter().copied()) {
            *mu = MuLink::inverse(eta_mu);
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

        DynMvNormalCholeskyTheta::from_parts_unchecked(mu, cholesky)
    }

    fn nll_theta(
        &self,
        observation: &[f64],
        theta: &DynMvNormalCholeskyTheta,
        workspace: &mut DynMvNormalCholeskyWorkspace,
    ) -> f64 {
        workspace.resize(self.dimension);
        kernel::nll(
            self.dimension,
            observation,
            &theta.mu,
            &theta.cholesky,
            &mut workspace.z,
        )
    }

    fn fill_workspace_theta(
        &self,
        values: &[f64],
        workspace: &mut DynMvNormalCholeskyWorkspace,
    ) -> bool {
        if values.len() != self.dimension + packed_len(self.dimension).unwrap_or(0) {
            return false;
        }
        workspace.resize(self.dimension);
        for index in 0..self.dimension {
            workspace.natural_mu[index] = MuLink::inverse(values[index]);
        }
        let mut packed = self.dimension;
        for row in 0..self.dimension {
            for col in 0..=row {
                workspace.natural_cholesky.values_mut()[packed - self.dimension] = if row == col {
                    DiagonalLink::inverse(values[packed])
                } else {
                    OffDiagonalLink::inverse(values[packed])
                };
                packed += 1;
            }
        }
        kernel::valid_theta(
            self.dimension,
            &workspace.natural_mu,
            &workspace.natural_cholesky,
        )
    }

    fn nll_and_gradient_eta_values(
        &self,
        observation: &[f64],
        eta: &DynMvNormalCholeskyEta,
        workspace: &mut DynMvNormalCholeskyWorkspace,
    ) -> (f64, DynMvNormalCholeskyEta) {
        if !self.eta_has_expected_len(eta) {
            return (f64::INFINITY, self.nan_eta());
        }

        let theta = self.theta_from_eta(eta);
        workspace.resize(self.dimension);
        let nll = kernel::nll_and_score(
            self.dimension,
            observation,
            &theta.mu,
            &theta.cholesky,
            &mut workspace.z,
            &mut workspace.a,
        );
        if !nll.is_finite() {
            return (nll, self.nan_eta());
        }

        let mut gradient = DynMvNormalCholeskyEta {
            mu: vec![0.0; self.dimension],
            cholesky: PackedLowerTriangular::filled(self.dimension, 0.0),
        };

        for ((gradient_mu, a), eta_mu) in gradient
            .mu
            .iter_mut()
            .zip(workspace.a.iter().copied())
            .zip(eta.mu.iter().copied())
        {
            *gradient_mu = -a * MuLink::derivative_inverse(eta_mu);
        }

        for row in 0..self.dimension {
            for (col, z_col) in workspace.z.iter().copied().take(row + 1).enumerate() {
                let eta_value = eta.cholesky.lower(row, col);
                let d_nll_d_l = if row == col {
                    DiagonalLink::derivative_log_inverse(eta_value)
                        - workspace.a[row] * z_col * DiagonalLink::derivative_inverse(eta_value)
                } else {
                    kernel::cholesky_score(row, col, z_col, &workspace.a, &theta.cholesky)
                        * OffDiagonalLink::derivative_inverse(eta_value)
                };
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
    type GradientEta = DynMvNormalCholeskyEta;
    type Observation<'obs> = &'obs [f64];
    type Workspace = DynMvNormalCholeskyWorkspace;

    #[inline]
    fn workspace(&self) -> Self::Workspace {
        DynMvNormalCholeskyWorkspace::new(self.dimension)
    }

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        self.theta_from_eta(eta)
    }

    #[inline]
    fn nll(
        &self,
        observation: Self::Observation<'_>,
        theta: &Self::Theta,
        workspace: &mut Self::Workspace,
    ) -> f64 {
        self.nll_theta(observation, theta, workspace)
    }

    #[inline]
    fn nll_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        workspace: &mut Self::Workspace,
    ) -> f64 {
        self.nll_theta(observation, &self.theta_from_eta(eta), workspace)
    }

    #[inline]
    fn nll_and_gradient_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        self.nll_and_gradient_eta_values(observation, eta, workspace)
    }
}

impl<MuLink, DiagonalLink, OffDiagonalLink> DynamicallyCompilableFamily
    for DynMvNormalCholesky<MuLink, DiagonalLink, OffDiagonalLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    fn dynamic_parameter_count(&self) -> usize {
        self.dimension + packed_len(self.dimension).unwrap_or(0)
    }

    fn dynamic_layout_key(&self) -> DynamicLayoutKey {
        DynamicLayoutKey::from_dimension(self.dimension)
    }

    fn eta_from_flat(&self, values: &[f64]) -> Self::Eta {
        if values.len() != self.dynamic_parameter_count() {
            return self.nan_eta();
        }
        DynMvNormalCholeskyEta {
            mu: values[..self.dimension].to_vec(),
            cholesky: PackedLowerTriangular::from_validated_parts(
                self.dimension,
                values[self.dimension..].to_vec(),
            ),
        }
    }

    fn gradient_to_flat(&self, gradient: &Self::GradientEta, out: &mut [f64]) {
        if out.len() != self.dynamic_parameter_count() || !self.eta_has_expected_len(gradient) {
            out.fill(f64::NAN);
            return;
        }
        out[..self.dimension].copy_from_slice(&gradient.mu);
        out[self.dimension..].copy_from_slice(gradient.cholesky.values());
    }

    fn nll_eta_flat(
        &self,
        observation: Self::Observation<'_>,
        values: &[f64],
        workspace: &mut Self::Workspace,
    ) -> f64 {
        if !self.fill_workspace_theta(values, workspace) {
            return f64::INFINITY;
        }
        kernel::nll(
            self.dimension,
            observation,
            &workspace.natural_mu,
            &workspace.natural_cholesky,
            &mut workspace.z,
        )
    }

    fn nll_and_gradient_eta_flat(
        &self,
        observation: Self::Observation<'_>,
        values: &[f64],
        gradient: &mut [f64],
        workspace: &mut Self::Workspace,
    ) -> f64 {
        if gradient.len() != self.dynamic_parameter_count()
            || !self.fill_workspace_theta(values, workspace)
        {
            gradient.fill(f64::NAN);
            return f64::INFINITY;
        }
        let nll = kernel::nll_and_score(
            self.dimension,
            observation,
            &workspace.natural_mu,
            &workspace.natural_cholesky,
            &mut workspace.z,
            &mut workspace.a,
        );
        if !nll.is_finite() {
            gradient.fill(f64::NAN);
            return nll;
        }

        for index in 0..self.dimension {
            gradient[index] = -workspace.a[index] * MuLink::derivative_inverse(values[index]);
        }
        let mut packed = self.dimension;
        for row in 0..self.dimension {
            for col in 0..=row {
                let eta = values[packed];
                gradient[packed] = if row == col {
                    DiagonalLink::derivative_log_inverse(eta)
                        - workspace.a[row]
                            * workspace.z[col]
                            * DiagonalLink::derivative_inverse(eta)
                } else {
                    kernel::cholesky_score(
                        row,
                        col,
                        workspace.z[col],
                        &workspace.a,
                        &workspace.natural_cholesky,
                    ) * OffDiagonalLink::derivative_inverse(eta)
                };
                packed += 1;
            }
        }
        nll
    }

    fn dynamic_parameter_coordinate(&self, index: usize) -> (&'static str, ParameterPath) {
        if index < self.dimension {
            return (
                Mu::NAME,
                ParameterPath::from_axis(ParameterAxis::Vector { component: index }),
            );
        }
        let packed = index - self.dimension;
        for row in 0..self.dimension {
            let row_start = packed_index(row, 0)
                .expect("validated dimension has representable packed Cholesky indices");
            if packed < row_start + row + 1 {
                return (
                    CholeskyScale::NAME,
                    ParameterPath::from_axis(ParameterAxis::Lower {
                        row,
                        col: packed - row_start,
                    }),
                );
            }
        }
        (CholeskyScale::NAME, ParameterPath::whole())
    }

    fn validate_dynamic_compiled(&self) -> Result<(), ModelError> {
        Self::new(self.dimension).map(|_| ())
    }
}

impl<MuLink, DiagonalLink, OffDiagonalLink> HasMarginalCdf
    for DynMvNormalCholesky<MuLink, DiagonalLink, OffDiagonalLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    fn marginal_cdf(&self, component: usize, y: f64, theta: &Self::Theta) -> f64 {
        if !y.is_finite() || component >= self.dimension {
            return f64::NAN;
        }

        let scale = self.marginal_scale(component, theta);
        if !scale.is_finite() || scale <= 0.0 {
            return f64::NAN;
        }

        unit_normal_cdf((y - theta.mu[component]) / scale)
    }
}

impl<MuLink, DiagonalLink, OffDiagonalLink> HasObservationDimension
    for DynMvNormalCholesky<MuLink, DiagonalLink, OffDiagonalLink>
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    fn observation_dimension(&self) -> usize {
        self.dimension
    }
}

impl<MuLink, DiagonalLink, OffDiagonalLink> HasConditionalCdf
    for DynMvNormalCholesky<MuLink, DiagonalLink, OffDiagonalLink>
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
        let mut standardized = vec![0.0; component];
        kernel::conditional_cdf(
            self.dimension,
            component,
            y,
            preceding,
            theta.mu(),
            theta.cholesky(),
            &mut standardized,
        )
    }
}

impl<MuLink, DiagonalLink, OffDiagonalLink> HasRosenblattTransform
    for DynMvNormalCholesky<MuLink, DiagonalLink, OffDiagonalLink>
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
        if out.len() != self.dimension {
            return Err(ModelError::ResponseLength {
                expected: self.dimension,
                actual: out.len(),
            });
        }
        kernel::rosenblatt_into(
            self.dimension,
            observation,
            theta.mu(),
            theta.cholesky(),
            out,
        );
        Ok(())
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

    fn sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Self::Sample {
        if !kernel::valid_theta(self.dimension, &theta.mu, &theta.cholesky) {
            return vec![f64::NAN; self.dimension];
        }

        let standard = rand_distr::StandardNormal;
        let mut z = vec![0.0; self.dimension];
        for value in &mut z {
            *value = rand_distr::Distribution::sample(&standard, rng);
        }

        let mut out = theta.mu.clone();
        for row in 0..self.dimension {
            for (col, z_col) in z.iter().copied().take(row + 1).enumerate() {
                out[row] += theta.cholesky.lower(row, col) * z_col;
            }
        }
        out
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink, DiagonalLink, OffDiagonalLink> TrySimulate<Rng>
    for DynMvNormalCholesky<MuLink, DiagonalLink, OffDiagonalLink>
where
    Rng: rand::Rng,
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
{
    type Sample = Vec<f64>;

    fn try_sample(
        &self,
        rng: &mut Rng,
        theta: &Self::Theta,
    ) -> Result<Self::Sample, SimulationError> {
        if !kernel::valid_theta(self.dimension, &theta.mu, &theta.cholesky) {
            return Err(SimulationError::InvalidParameters("dynamic MVN theta"));
        }
        let standard = rand_distr::StandardNormal;
        let z: Vec<f64> = (0..self.dimension)
            .map(|_| rand_distr::Distribution::sample(&standard, rng))
            .collect();
        let mut out = theta.mu.clone();
        for row in 0..self.dimension {
            for (col, z_col) in z.iter().copied().take(row + 1).enumerate() {
                out[row] += theta.cholesky.lower(row, col) * z_col;
            }
        }
        Ok(out)
    }
}

/// Reusable buffers for runtime-dimensional MVN likelihood evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct DynMvNormalCholeskyWorkspace {
    z: Vec<f64>,
    a: Vec<f64>,
    natural_mu: Vec<f64>,
    natural_cholesky: PackedLowerTriangular,
}

impl DynMvNormalCholeskyWorkspace {
    fn new(dimension: usize) -> Self {
        Self {
            z: vec![0.0; dimension],
            a: vec![0.0; dimension],
            natural_mu: vec![0.0; dimension],
            natural_cholesky: PackedLowerTriangular::filled(dimension, 0.0),
        }
    }

    fn resize(&mut self, dimension: usize) {
        self.z.resize(dimension, 0.0);
        self.a.resize(dimension, 0.0);
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
    /// Creates checked runtime-dimensional natural-scale parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] unless the dimension is
    /// positive, dimensions match, all entries are finite, and every Cholesky
    /// diagonal entry is strictly positive.
    pub fn try_new(mu: Vec<f64>, cholesky: PackedLowerTriangular) -> Result<Self, ModelError> {
        if !kernel::valid_theta(cholesky.dimension(), &mu, &cholesky) {
            return Err(ModelError::InvalidParameter {
                parameter: "dynamic multivariate normal theta",
                expected: "matching positive dimensions, finite values, and positive Cholesky diagonal",
            });
        }
        Ok(Self { mu, cholesky })
    }

    #[inline]
    const fn from_parts_unchecked(mu: Vec<f64>, cholesky: PackedLowerTriangular) -> Self {
        Self { mu, cholesky }
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

    /// Returns one covariance entry from `L L'`.
    #[must_use]
    pub fn covariance(&self, row: usize, col: usize) -> Option<f64> {
        if row >= self.mu.len() || col >= self.mu.len() {
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
        DenseDesign, DynamicParameterBlocks, Family, Gamlss, HasMarginalCdf, LinearPredictorBlock,
        ModelError, Mu, NoPenalty, ObservationView, ParameterAxis,
    };

    use super::{DynMvNormalCholeskyEta, DynMvNormalCholeskyTheta, PackedLowerTriangular};
    use crate::multivariate::normal::DynMvNormalCholeskyDefault;

    #[derive(Debug, Clone)]
    struct BorrowedRows(Vec<Vec<f64>>);

    impl<'row> ObservationView<'row> for BorrowedRows {
        type Observation = &'row [f64];

        fn len(&self) -> usize {
            self.0.len()
        }

        fn observation_at(&'row self, row: usize) -> Self::Observation {
            &self.0[row]
        }

        fn weight_at(&self, _row: usize) -> f64 {
            1.0
        }
    }

    fn finite_difference_dynamic_gradient(
        y: &[f64],
        eta: &DynMvNormalCholeskyEta,
        epsilon: f64,
        tolerance: f64,
    ) {
        let family = DynMvNormalCholeskyDefault::new(y.len()).unwrap();
        let (_, gradient) = family.nll_and_gradient_eta(y, eta, &mut family.workspace());

        for index in 0..eta.mu().len() {
            let mut plus = eta.clone();
            plus.mu_mut()[index] += epsilon;
            let mut minus = eta.clone();
            minus.mu_mut()[index] -= epsilon;
            let finite_difference = (family.nll_eta(y, &plus, &mut family.workspace())
                - family.nll_eta(y, &minus, &mut family.workspace()))
                / (2.0 * epsilon);
            assert_relative_eq!(gradient.mu()[index], finite_difference, epsilon = tolerance);
        }

        for row in 0..y.len() {
            for col in 0..=row {
                let mut plus = eta.clone();
                *plus.cholesky_entry_mut(row, col).unwrap() += epsilon;
                let mut minus = eta.clone();
                *minus.cholesky_entry_mut(row, col).unwrap() -= epsilon;
                let finite_difference = (family.nll_eta(y, &plus, &mut family.workspace())
                    - family.nll_eta(y, &minus, &mut family.workspace()))
                    / (2.0 * epsilon);
                assert_relative_eq!(
                    gradient.cholesky_entry(row, col).unwrap(),
                    finite_difference,
                    epsilon = tolerance
                );
            }
        }
    }

    #[test]
    fn dynamic_gradient_matches_finite_difference_for_d1_through_d4() {
        for dimension in 1..=4 {
            let observation = (0..dimension)
                .map(|component| 1.1 - 0.37 * component as f64)
                .collect::<Vec<_>>();
            let mu = (0..dimension)
                .map(|component| 0.4 - 0.16 * component as f64)
                .collect();
            let mut cholesky = Vec::with_capacity(dimension * (dimension + 1) / 2);
            for row in 0..dimension {
                for col in 0..=row {
                    cholesky.push(if row == col {
                        -0.2 + 0.09 * row as f64
                    } else {
                        0.05 * (row + col + 1) as f64
                    });
                }
            }
            let eta = DynMvNormalCholeskyEta::new(
                mu,
                PackedLowerTriangular::try_new(dimension, cholesky).unwrap(),
            )
            .unwrap();
            finite_difference_dynamic_gradient(&observation, &eta, 1.0e-6, 2.0e-6);
        }
    }

    #[test]
    fn runtime_dimension_is_fit_ready_with_in_place_flat_gradient() {
        let observations = BorrowedRows(vec![vec![0.2, -0.4], vec![1.1, 0.7], vec![-0.5, 0.3]]);
        let family = DynMvNormalCholeskyDefault::new(2).unwrap();
        let predictors = (0..5)
            .map(|_| {
                (
                    LinearPredictorBlock::new(DenseDesign::intercept(observations.len())),
                    NoPenalty,
                )
            })
            .collect();
        let blocks = DynamicParameterBlocks::try_new(&family, predictors).unwrap();
        let model = Gamlss::try_new_with_observations(family, blocks, observations).unwrap();
        let beta = [0.1, -0.2, 0.05, 0.15, -0.1];
        let mut gradient = [0.0; 5];
        let mut workspace = model.gradient_workspace();

        model
            .try_likelihood_value_gradient_into_workspace(&beta, &mut gradient, &mut workspace)
            .unwrap();
        model
            .try_likelihood_value_gradient_into_workspace(&beta, &mut gradient, &mut workspace)
            .unwrap();
        for index in 0..beta.len() {
            let mut plus = beta;
            let mut minus = beta;
            plus[index] += 1.0e-6;
            minus[index] -= 1.0e-6;
            let finite_difference = (model.try_likelihood_value(&plus).unwrap()
                - model.try_likelihood_value(&minus).unwrap())
                / 2.0e-6;
            assert_relative_eq!(gradient[index], finite_difference, epsilon = 2.0e-5);
        }

        let descriptors = model.parameter_descriptors();
        assert_eq!(descriptors.len(), 5);
        assert_eq!(
            descriptors[0].path.axes(),
            &[ParameterAxis::Vector { component: 0 }]
        );
        assert_eq!(
            descriptors[4].path.axes(),
            &[ParameterAxis::Lower { row: 1, col: 1 }]
        );

        let mut visited_descriptors = Vec::new();
        model.visit_parameter_descriptors(|index, descriptor| {
            visited_descriptors.push((index, descriptor));
        });
        assert_eq!(
            visited_descriptors,
            descriptors.into_iter().enumerate().collect::<Vec<_>>()
        );

        let mu_descriptors = model.parameter_descriptors_of::<Mu>();
        assert_eq!(mu_descriptors.len(), 2);
        assert_eq!(
            mu_descriptors[1].1.path.axes(),
            &[ParameterAxis::Vector { component: 1 }]
        );
        let unpacked = model.unpack_parameters(&beta).unwrap();
        assert_eq!(
            unpacked.block_at(1).unwrap().descriptor,
            mu_descriptors[1].1
        );
        assert_eq!(
            unpacked.block_at(4).unwrap().path().axes(),
            &[ParameterAxis::Lower { row: 1, col: 1 }]
        );
    }

    #[test]
    fn invalid_lengths_return_constructor_errors() {
        assert!(DynMvNormalCholeskyDefault::new(0).is_err());
        assert!(matches!(
            DynMvNormalCholeskyDefault::new(usize::MAX),
            Err(ModelError::ArithmeticOverflow { .. })
        ));
        assert!(PackedLowerTriangular::try_new(2, vec![0.0; 4]).is_err());
        assert!(
            DynMvNormalCholeskyEta::new(
                vec![0.0],
                PackedLowerTriangular::try_new(2, vec![0.0; 3]).unwrap(),
            )
            .is_err()
        );
        assert!(
            DynMvNormalCholeskyTheta::try_new(
                vec![0.0],
                PackedLowerTriangular::try_new(2, vec![1.0, 0.0, 1.0]).unwrap(),
            )
            .is_err()
        );
        assert!(
            DynMvNormalCholeskyTheta::try_new(
                vec![0.0, 0.0],
                PackedLowerTriangular::try_new(2, vec![1.0, f64::NAN, 1.0]).unwrap(),
            )
            .is_err()
        );
        assert!(
            DynMvNormalCholeskyTheta::try_new(
                vec![0.0, 0.0],
                PackedLowerTriangular::try_new(2, vec![1.0, 0.0, 0.0]).unwrap(),
            )
            .is_err()
        );
    }

    #[test]
    fn marginal_cdf_uses_component_variance() {
        let family = DynMvNormalCholeskyDefault::new(2).unwrap();
        let theta = DynMvNormalCholeskyTheta::try_new(
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
    }

    #[cfg(feature = "rand")]
    #[test]
    fn sampling_returns_finite_values_for_valid_theta() {
        use gamlss_core::CanSimulate;

        let family = DynMvNormalCholeskyDefault::new(2).unwrap();
        let mut rng = rand::rng();
        let valid = family.sample(
            &mut rng,
            &DynMvNormalCholeskyTheta::try_new(
                vec![0.0, 1.0],
                PackedLowerTriangular::try_new(2, vec![2.0, 3.0, 4.0]).unwrap(),
            )
            .unwrap(),
        );
        assert_eq!(valid.len(), 2);
        assert!(valid.iter().all(|value| value.is_finite()));
    }
}
