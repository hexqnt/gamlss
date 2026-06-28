use std::marker::PhantomData;
use std::ops::Range;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    CholeskyScale, Family, FixedDimensionalFamily, GamlssBlocks, GradientWorkspace, HasMarginalCdf,
    Identity, Link, Log, LowerTriangularParameterBlock, ModelError, Mu, ObservationView,
    ParameterLayout, ParameterName, ParameterSlice, Penalty, PositiveLink, PredictorBlock,
    VectorParameterBlock,
};
use gamlss_special::unit_normal_cdf;

use super::kernel;

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
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    fn theta_from_eta(eta: MvNormalCholeskyEta<D>) -> MvNormalCholeskyTheta<D> {
        let mut mu = [0.0; D];
        let mut cholesky = [[0.0; D]; D];

        for index in 0..D {
            mu[index] = MuLink::inverse(eta.mu[index]);
        }

        for row in 0..D {
            for col in 0..=row {
                cholesky[row][col] = if row == col {
                    DiagonalLink::inverse(eta.cholesky[row][col])
                } else {
                    OffDiagonalLink::inverse(eta.cholesky[row][col])
                };
            }
        }

        MvNormalCholeskyTheta { mu, cholesky }
    }

    const fn nan_eta() -> MvNormalCholeskyEta<D> {
        MvNormalCholeskyEta {
            mu: [f64::NAN; D],
            cholesky: [[f64::NAN; D]; D],
        }
    }

    fn nll_theta(observation: [f64; D], theta: MvNormalCholeskyTheta<D>) -> f64 {
        let mut z = [0.0; D];
        kernel::nll(
            D,
            &observation,
            &theta.mu,
            theta.cholesky.as_flattened(),
            &mut z,
        )
    }

    fn nll_and_gradient_eta_values(
        observation: [f64; D],
        eta: MvNormalCholeskyEta<D>,
    ) -> (f64, MvNormalCholeskyEta<D>) {
        let theta = Self::theta_from_eta(eta);
        let mut z = [0.0; D];
        let mut a = [0.0; D];
        let nll = kernel::nll_and_score(
            D,
            &observation,
            &theta.mu,
            theta.cholesky.as_flattened(),
            &mut z,
            &mut a,
        );
        if !nll.is_finite() {
            return (nll, Self::nan_eta());
        }

        let mut gradient = MvNormalCholeskyEta {
            mu: [0.0; D],
            cholesky: [[0.0; D]; D],
        };

        for index in 0..D {
            gradient.mu[index] = -a[index] * MuLink::derivative_inverse(eta.mu[index]);
        }

        for row in 0..D {
            for (col, z_col) in z.iter().copied().take(row + 1).enumerate() {
                let mut d_nll_d_l =
                    kernel::cholesky_score(D, row, col, z_col, &a, theta.cholesky.as_flattened());
                if row == col {
                    d_nll_d_l *= DiagonalLink::derivative_inverse(eta.cholesky[row][col]);
                } else {
                    d_nll_d_l *= OffDiagonalLink::derivative_inverse(eta.cholesky[row][col]);
                }
                gradient.cholesky[row][col] = d_nll_d_l;
            }
        }

        (nll, gradient)
    }

    fn marginal_scale(component: usize, theta: &MvNormalCholeskyTheta<D>) -> f64 {
        kernel::marginal_scale(D, component, &theta.mu, theta.cholesky.as_flattened())
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
    type NllGradientEta = MvNormalCholeskyEta<D>;
    type Observation<'obs> = [f64; D];

    #[inline]
    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        Self::theta_from_eta(eta)
    }

    #[inline]
    fn nll(&self, observation: Self::Observation<'_>, theta: Self::Theta) -> f64 {
        Self::nll_theta(observation, theta)
    }

    #[inline]
    fn nll_eta(&self, observation: Self::Observation<'_>, eta: Self::Eta) -> f64 {
        Self::nll_theta(observation, Self::theta_from_eta(eta))
    }

    #[inline]
    fn nll_and_gradient_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: Self::Eta,
    ) -> (f64, Self::NllGradientEta) {
        Self::nll_and_gradient_eta_values(observation, eta)
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
    fn marginal_cdf(&self, component: usize, y: f64, theta: Self::Theta) -> f64 {
        if !y.is_finite() || component >= D {
            return f64::NAN;
        }

        let scale = Self::marginal_scale(component, &theta);
        if !scale.is_finite() || scale <= 0.0 {
            return f64::NAN;
        }

        unit_normal_cdf((y - theta.mu[component]) / scale)
    }
}

impl<
    const D: usize,
    MuLink,
    DiagonalLink,
    OffDiagonalLink,
    MuPredictor,
    CholeskyPredictor,
    MuPenalty,
    CholeskyPenalty,
> GamlssBlocks<MvNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink>>
    for (
        VectorParameterBlock<Mu, D, MuPredictor, MuPenalty>,
        LowerTriangularParameterBlock<CholeskyScale, D, CholeskyPredictor, CholeskyPenalty>,
    )
where
    MuLink: Link<f64>,
    DiagonalLink: PositiveLink<f64>,
    OffDiagonalLink: Link<f64>,
    MuPredictor: PredictorBlock,
    CholeskyPredictor: PredictorBlock,
    MuPenalty: Penalty,
    CholeskyPenalty: Penalty,
{
    fn nrows(&self) -> usize {
        self.0.component(0).map_or(0, PredictorBlock::nrows)
    }

    fn len(&self) -> usize {
        <Self as GamlssBlocks<MvNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink>>>::try_len(
            self,
        )
        .expect("validated multivariate normal block layout must fit in usize")
    }

    fn try_len(&self) -> Result<usize, ModelError> {
        let mu = self.0.try_range()?;
        let cholesky = self.1.try_range()?;
        Ok(mu.end.max(cholesky.end))
    }

    fn validate(&self, nobs: usize) -> Result<(), ModelError> {
        if D == 0 {
            return Err(ModelError::InvalidParameter {
                parameter: "dimension",
                expected: "positive",
            });
        }

        validate_vector_block::<Mu, D, _, _>(&self.0, nobs)?;
        validate_lower_triangular_block::<CholeskyScale, D, _, _>(&self.1, nobs)?;
        self.0.penalty().validate_dim(self.0.len())?;
        self.1.penalty().validate_dim(self.1.len())?;

        let mu = self.0.try_range()?;
        let cholesky = self.1.try_range()?;
        if ranges_overlap(mu, cholesky) {
            return Err(ModelError::BlockOverlap {
                first: Mu::NAME,
                second: CholeskyScale::NAME,
            });
        }

        Ok(())
    }

    fn train_nll<'obs, Obs>(
        &self,
        family: &MvNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink>,
        obs: &'obs Obs,
        beta: &[f64],
    ) -> f64
    where
        Obs: ObservationView<'obs, Observation = [f64; D]> + 'obs,
    {
        let mut loss = 0.0;
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            if weight == 0.0 {
                continue;
            }
            let observation = obs.observation_at(row);
            loss += weight * family.nll_eta(observation, mv_cholesky_eta_row(self, beta, row));
        }
        loss
    }

    fn eta_row(
        &self,
        beta: &[f64],
        row: usize,
    ) -> <MvNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink> as Family>::Eta {
        mv_cholesky_eta_row(self, beta, row)
    }

    fn penalty_value(&self, beta: &[f64]) -> f64 {
        self.0.penalty().value(&beta[self.0.range()])
            + self.1.penalty().value(&beta[self.1.range()])
    }

    fn initial_parameters<'obs, Obs>(
        &self,
        family: &MvNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink>,
        obs: &'obs Obs,
    ) -> Vec<f64>
    where
        Obs: ObservationView<'obs, Observation = [f64; D]> + 'obs,
    {
        <Self as GamlssBlocks<MvNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink>>>::try_initial_parameters(
            self, family, obs,
        )
        .expect("validated multivariate normal block layout must fit in usize")
    }

    fn try_initial_parameters<'obs, Obs>(
        &self,
        _family: &MvNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink>,
        _obs: &'obs Obs,
    ) -> Result<Vec<f64>, ModelError>
    where
        Obs: ObservationView<'obs, Observation = [f64; D]> + 'obs,
    {
        let mut beta = vec![0.0; self.0.try_range()?.end.max(self.1.try_range()?.end)];
        for component in 0..D {
            let predictor = self
                .0
                .component(component)
                .expect("validated vector block has D components");
            let range = self
                .0
                .component_range(component)
                .expect("validated vector component has a coefficient range");
            predictor.set_constant_start(0.0, &mut beta[range]);
        }
        for matrix_row in 0..D {
            for matrix_col in 0..=matrix_row {
                let predictor = self
                    .1
                    .entry(matrix_row, matrix_col)
                    .expect("validated lower-triangular block has this entry");
                let range = self
                    .1
                    .entry_range(matrix_row, matrix_col)
                    .expect("validated lower-triangular entry has a coefficient range");
                predictor.set_constant_start(0.0, &mut beta[range]);
            }
        }
        Ok(beta)
    }

    fn add_penalty_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        self.0
            .penalty()
            .add_gradient(&beta[self.0.range()], &mut grad[self.0.range()]);
        self.1
            .penalty()
            .add_gradient(&beta[self.1.range()], &mut grad[self.1.range()]);
    }

    fn gradient_workspace(&self, nobs: usize) -> GradientWorkspace {
        let mut workspace = GradientWorkspace::new();
        let scalar_count = D + lower_triangular_len(D);
        workspace.prepare(scalar_count);
        for index in 0..scalar_count {
            workspace.prepare_row_gradient(index, nobs);
        }
        for component in 0..D {
            let len = self
                .0
                .component_range(component)
                .expect("validated vector component has a coefficient range")
                .len();
            let _ = workspace.local_gradient_mut(component, len);
        }
        for matrix_row in 0..D {
            for matrix_col in 0..=matrix_row {
                let workspace_index = cholesky_workspace_index::<D>(matrix_row, matrix_col);
                let len = self
                    .1
                    .entry_range(matrix_row, matrix_col)
                    .expect("validated lower-triangular entry has a coefficient range")
                    .len();
                let _ = workspace.local_gradient_mut(workspace_index, len);
            }
        }
        workspace
    }

    fn value_gradient_into_workspace<'obs, Obs>(
        &self,
        family: &MvNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink>,
        obs: &'obs Obs,
        beta: &[f64],
        grad: &mut [f64],
        workspace: &mut GradientWorkspace,
    ) -> f64
    where
        Obs: ObservationView<'obs, Observation = [f64; D]> + 'obs,
    {
        let scalar_count = D + lower_triangular_len(D);
        workspace.prepare(scalar_count);
        for index in 0..scalar_count {
            workspace.prepare_row_gradient(index, obs.len());
        }

        let mut loss = 0.0;
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            if weight == 0.0 {
                zero_row_gradients(workspace, row, scalar_count);
                continue;
            }

            let observation = obs.observation_at(row);
            let (nll, gradient) =
                family.nll_and_gradient_eta(observation, mv_cholesky_eta_row(self, beta, row));
            loss += weight * nll;

            for component in 0..D {
                workspace.set_row_gradient(component, row, weight * gradient.mu[component]);
            }
            for matrix_row in 0..D {
                for matrix_col in 0..=matrix_row {
                    workspace.set_row_gradient(
                        cholesky_workspace_index::<D>(matrix_row, matrix_col),
                        row,
                        weight * gradient.cholesky[matrix_row][matrix_col],
                    );
                }
            }
        }

        for component in 0..D {
            let predictor = self
                .0
                .component(component)
                .expect("validated vector block has D components");
            let range = self
                .0
                .component_range(component)
                .expect("validated vector component has a coefficient range");
            let (row_gradient, local_gradient) =
                workspace.row_gradient_and_local_gradient_mut(component, range.len());
            predictor.add_gradient(row_gradient, &beta[range.clone()], local_gradient);
            add_into(&mut grad[range], local_gradient);
        }

        for matrix_row in 0..D {
            for matrix_col in 0..=matrix_row {
                let predictor = self
                    .1
                    .entry(matrix_row, matrix_col)
                    .expect("validated lower-triangular block has this entry");
                let range = self
                    .1
                    .entry_range(matrix_row, matrix_col)
                    .expect("validated lower-triangular entry has a coefficient range");
                let (row_gradient, local_gradient) = workspace.row_gradient_and_local_gradient_mut(
                    cholesky_workspace_index::<D>(matrix_row, matrix_col),
                    range.len(),
                );
                predictor.add_gradient(row_gradient, &beta[range.clone()], local_gradient);
                add_into(&mut grad[range], local_gradient);
            }
        }

        loss += self.0.penalty().value(&beta[self.0.range()]);
        self.0
            .penalty()
            .add_gradient(&beta[self.0.range()], &mut grad[self.0.range()]);
        loss += self.1.penalty().value(&beta[self.1.range()]);
        self.1
            .penalty()
            .add_gradient(&beta[self.1.range()], &mut grad[self.1.range()]);

        loss
    }

    fn block_ranges(&self) -> Vec<Range<usize>> {
        vec![self.0.range(), self.1.range()]
    }

    fn parameter_layout(&self) -> ParameterLayout {
        ParameterLayout::new(vec![
            ParameterSlice {
                name: Mu::NAME,
                range: self.0.range(),
            },
            ParameterSlice {
                name: CholeskyScale::NAME,
                range: self.1.range(),
            },
        ])
    }

    fn parameter_slice_count(&self) -> usize {
        2
    }

    fn parameter_slice_matches(
        &self,
        index: usize,
        name: &'static str,
        range: Range<usize>,
    ) -> bool {
        match index {
            0 => name == Mu::NAME && range == self.0.range(),
            1 => name == CholeskyScale::NAME && range == self.1.range(),
            _ => false,
        }
    }

    fn visit_parameter_slices<V>(&self, mut visit: V)
    where
        V: FnMut(usize, &'static str, Range<usize>),
    {
        visit(0, Mu::NAME, self.0.range());
        visit(1, CholeskyScale::NAME, self.1.range());
    }

    fn has_same_parameter_layout<Other>(&self, other: &Other) -> bool
    where
        Other: GamlssBlocks<MvNormalCholesky<D, MuLink, DiagonalLink, OffDiagonalLink>>,
    {
        other.parameter_slice_count() == 2
            && other.parameter_slice_matches(0, Mu::NAME, self.0.range())
            && other.parameter_slice_matches(1, CholeskyScale::NAME, self.1.range())
    }
}

fn validate_vector_block<P, const D: usize, X, Penalty>(
    block: &VectorParameterBlock<P, D, X, Penalty>,
    nobs: usize,
) -> Result<(), ModelError>
where
    P: ParameterName,
    X: PredictorBlock,
{
    block.try_range()?;
    for component in 0..D {
        let predictor = block
            .component(component)
            .expect("vector block has D components");
        predictor.validate()?;
        validate_predictor_rows(P::NAME, predictor.nrows(), nobs)?;
    }
    Ok(())
}

fn mv_cholesky_eta_row<const D: usize, MuPredictor, CholeskyPredictor, MuPenalty, CholeskyPenalty>(
    blocks: &(
        VectorParameterBlock<Mu, D, MuPredictor, MuPenalty>,
        LowerTriangularParameterBlock<CholeskyScale, D, CholeskyPredictor, CholeskyPenalty>,
    ),
    beta: &[f64],
    row: usize,
) -> MvNormalCholeskyEta<D>
where
    MuPredictor: PredictorBlock,
    CholeskyPredictor: PredictorBlock,
{
    let mut eta = MvNormalCholeskyEta {
        mu: [0.0; D],
        cholesky: [[0.0; D]; D],
    };

    for component in 0..D {
        let predictor = blocks
            .0
            .component(component)
            .expect("validated vector block has D components");
        let range = blocks
            .0
            .component_range(component)
            .expect("validated vector component has a coefficient range");
        eta.mu[component] = predictor.eta_row(row, &beta[range]);
    }

    for matrix_row in 0..D {
        for matrix_col in 0..=matrix_row {
            let predictor = blocks
                .1
                .entry(matrix_row, matrix_col)
                .expect("validated lower-triangular block has this entry");
            let range = blocks
                .1
                .entry_range(matrix_row, matrix_col)
                .expect("validated lower-triangular entry has a coefficient range");
            eta.cholesky[matrix_row][matrix_col] = predictor.eta_row(row, &beta[range]);
        }
    }

    eta
}

fn validate_lower_triangular_block<P, const D: usize, X, Penalty>(
    block: &LowerTriangularParameterBlock<P, D, X, Penalty>,
    nobs: usize,
) -> Result<(), ModelError>
where
    P: ParameterName,
    X: PredictorBlock,
{
    block.try_range()?;
    let expected = lower_triangular_len(D);
    if block.entries().len() != expected {
        return Err(ModelError::InvalidParameter {
            parameter: P::NAME,
            expected: "D * (D + 1) / 2 predictor blocks",
        });
    }
    for predictor in block.entries() {
        predictor.validate()?;
        validate_predictor_rows(P::NAME, predictor.nrows(), nobs)?;
    }
    Ok(())
}

const fn validate_predictor_rows(
    parameter: &'static str,
    actual_rows: usize,
    expected_rows: usize,
) -> Result<(), ModelError> {
    if actual_rows == expected_rows {
        Ok(())
    } else {
        Err(ModelError::DesignRowMismatch {
            parameter,
            expected_rows,
            actual_rows,
        })
    }
}

const fn ranges_overlap(first: Range<usize>, second: Range<usize>) -> bool {
    first.start < second.end && second.start < first.end
}

const fn lower_triangular_len(dimension: usize) -> usize {
    dimension * (dimension + 1) / 2
}

const fn cholesky_workspace_index<const D: usize>(row: usize, col: usize) -> usize {
    D + row * (row + 1) / 2 + col
}

fn zero_row_gradients(workspace: &mut GradientWorkspace, row: usize, count: usize) {
    for index in 0..count {
        workspace.set_row_gradient(index, row, 0.0);
    }
}

fn add_into(out: &mut [f64], values: &[f64]) {
    for (out_value, value) in out.iter_mut().zip(values) {
        *out_value += value;
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

    fn sample(&self, rng: &mut Rng, theta: Self::Theta) -> Self::Sample {
        if !kernel::valid_theta(D, &theta.mu, theta.cholesky.as_flattened()) {
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
                out[row] += theta.cholesky[row][col] * z_col;
            }
        }
        out
    }
}

/// Multivariate normal predictors on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MvNormalCholeskyEta<const D: usize> {
    /// Mean predictors.
    pub mu: [f64; D],
    /// Lower-triangular Cholesky predictors. Upper-triangular entries are ignored.
    pub cholesky: [[f64; D]; D],
}

/// Multivariate normal parameters on the natural scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MvNormalCholeskyTheta<const D: usize> {
    /// Mean vector.
    pub mu: [f64; D],
    /// Lower-triangular Cholesky scale factor. Upper-triangular entries are ignored.
    pub cholesky: [[f64; D]; D],
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{
        CholeskyScale, DenseDesign, Family, FixedDimensionalFamily, Gamlss, HasMarginalCdf,
        LinearPredictorBlock, LowerTriangularParameterBlock, Mu, NoPenalty, ParameterBlocks,
        VectorParameterBlock,
    };

    use super::{MvNormalCholeskyEta, MvNormalCholeskyTheta};
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
        let (_, gradient) = family.nll_and_gradient_eta(y, eta);

        for index in 0..D {
            let mut plus = eta;
            plus.mu[index] += epsilon;
            let mut minus = eta;
            minus.mu[index] -= epsilon;
            let finite_difference =
                (family.nll_eta(y, plus) - family.nll_eta(y, minus)) / (2.0 * epsilon);
            assert_relative_eq!(gradient.mu[index], finite_difference, epsilon = tolerance);
        }

        for row in 0..D {
            for col in 0..=row {
                let mut plus = eta;
                plus.cholesky[row][col] += epsilon;
                let mut minus = eta;
                minus.cholesky[row][col] -= epsilon;
                let finite_difference =
                    (family.nll_eta(y, plus) - family.nll_eta(y, minus)) / (2.0 * epsilon);
                assert_relative_eq!(
                    gradient.cholesky[row][col],
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
    fn gradient_matches_finite_difference_for_representative_dimensions() {
        finite_difference_gradient::<1>(
            [1.7],
            MvNormalCholeskyEta {
                mu: [0.4],
                cholesky: [[-0.2]],
            },
            1.0e-6,
            1.0e-6,
        );
        finite_difference_gradient::<2>(
            [1.7, -0.8],
            MvNormalCholeskyEta {
                mu: [0.4, -0.3],
                cholesky: [[-0.2, 0.0], [0.25, 0.1]],
            },
            1.0e-6,
            1.0e-6,
        );
        finite_difference_gradient::<3>(
            [1.7, -0.8, 0.2],
            MvNormalCholeskyEta {
                mu: [0.4, -0.3, 0.1],
                cholesky: [[-0.2, 0.0, 0.0], [0.25, 0.1, 0.0], [-0.1, 0.2, 0.3]],
            },
            1.0e-6,
            1.0e-6,
        );
    }

    #[test]
    fn one_dimensional_case_matches_scalar_normal() {
        let mv = MvNormalCholeskyDefault::<1>::new();
        let normal = NormalMuSigma::new();
        let theta = MvNormalCholeskyTheta {
            mu: [0.4],
            cholesky: [[0.8]],
        };
        let scalar_theta = NormalTheta {
            mu: 0.4,
            sigma: 0.8,
        };

        assert_relative_eq!(
            mv.nll([1.7], theta),
            normal.nll(1.7, scalar_theta),
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn diagonal_cholesky_matches_independent_scalar_normals() {
        let family = MvNormalCholeskyDefault::<3>::new();
        let theta = MvNormalCholeskyTheta {
            mu: [0.4, -0.3, 0.1],
            cholesky: [[0.8, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 0.5]],
        };
        let y = [1.7, -0.8, 0.2];
        let normal = NormalMuSigma::new();
        let expected = normal.nll(
            y[0],
            NormalTheta {
                mu: theta.mu[0],
                sigma: theta.cholesky[0][0],
            },
        ) + normal.nll(
            y[1],
            NormalTheta {
                mu: theta.mu[1],
                sigma: theta.cholesky[1][1],
            },
        ) + normal.nll(
            y[2],
            NormalTheta {
                mu: theta.mu[2],
                sigma: theta.cholesky[2][2],
            },
        );

        assert_relative_eq!(family.nll(y, theta), expected, epsilon = 1.0e-12);
    }

    #[test]
    #[allow(clippy::manual_midpoint)]
    fn non_diagonal_nll_matches_hand_computed_value() {
        let family = MvNormalCholeskyDefault::<2>::new();
        let theta = MvNormalCholeskyTheta {
            mu: [0.0, 0.0],
            cholesky: [[2.0, 0.0], [1.0, 3.0]],
        };
        let y = [4.0, 7.0];
        let z0 = 2.0;
        let z1 = (7.0 - z0) / 3.0;
        let expected =
            2.0 * HALF_LOG_2_PI + 2.0_f64.ln() + 3.0_f64.ln() + 0.5 * (z0 * z0 + z1 * z1);

        assert_relative_eq!(family.nll(y, theta), expected, epsilon = 1.0e-12);
    }

    #[test]
    fn invalid_domains_return_infinite_nll_and_nan_gradient() {
        let family = MvNormalCholeskyDefault::<2>::new();
        let valid_eta = MvNormalCholeskyEta {
            mu: [0.0, 0.0],
            cholesky: [[0.0, 0.0], [0.2, 0.0]],
        };
        let invalid_theta = MvNormalCholeskyTheta {
            mu: [0.0, 0.0],
            cholesky: [[1.0, 0.0], [0.2, 0.0]],
        };

        assert!(
            family
                .nll([f64::NAN, 0.0], family.theta(valid_eta))
                .is_infinite()
        );
        assert!(family.nll([0.0, 0.0], invalid_theta).is_infinite());

        let (nll, gradient) = family.nll_and_gradient_eta([f64::NAN, 0.0], valid_eta);
        assert!(nll.is_infinite());
        assert!(gradient.mu.iter().all(|value| value.is_nan()));
        assert!(
            gradient
                .cholesky
                .iter()
                .flatten()
                .all(|value| value.is_nan())
        );
    }

    #[test]
    fn zero_dimension_is_invalid() {
        let family = MvNormalCholeskyDefault::<0>::new();
        assert!(
            family
                .nll(
                    [],
                    MvNormalCholeskyTheta {
                        mu: [],
                        cholesky: [],
                    },
                )
                .is_infinite()
        );
    }

    #[test]
    fn marginal_cdf_uses_component_variance() {
        let family = MvNormalCholeskyDefault::<2>::new();
        let theta = MvNormalCholeskyTheta {
            mu: [0.0, 1.0],
            cholesky: [[2.0, 0.0], [3.0, 4.0]],
        };

        assert_relative_eq!(family.marginal_cdf(0, 0.0, theta), 0.5, epsilon = 1.0e-12);
        assert_relative_eq!(
            family.marginal_cdf(1, 6.0, theta),
            0.841_344_746,
            epsilon = 1.0e-9
        );
        assert!(family.marginal_cdf(2, 0.0, theta).is_nan());
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
        assert_relative_eq!(eta.mu[0], 0.1, epsilon = 1.0e-12);
        assert_relative_eq!(eta.mu[1], -0.2, epsilon = 1.0e-12);
        assert_relative_eq!(eta.cholesky[0][0], 0.0, epsilon = 1.0e-12);
        assert_relative_eq!(eta.cholesky[1][0], 0.3, epsilon = 1.0e-12);
        assert_relative_eq!(eta.cholesky[1][1], 0.1, epsilon = 1.0e-12);

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

    #[cfg(feature = "rand")]
    #[test]
    fn sampling_returns_finite_values_or_nan_for_invalid_theta() {
        use gamlss_core::CanSimulate;

        let family = MvNormalCholeskyDefault::<2>::new();
        let mut rng = rand::rng();
        let valid = family.sample(
            &mut rng,
            MvNormalCholeskyTheta {
                mu: [0.0, 1.0],
                cholesky: [[2.0, 0.0], [3.0, 4.0]],
            },
        );
        assert!(valid.iter().all(|value| value.is_finite()));

        let invalid = family.sample(
            &mut rng,
            MvNormalCholeskyTheta {
                mu: [0.0, 1.0],
                cholesky: [[2.0, 0.0], [3.0, 0.0]],
            },
        );
        assert!(invalid.iter().all(|value| value.is_nan()));
    }
}
