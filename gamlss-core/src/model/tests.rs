use std::ops::Range;

use approx::assert_relative_eq;

use crate::{
    Broadcast, CholeskyScale, CompilableFamily, DenseDesign, DenseRows, DynamicLayoutKey,
    DynamicParameterBlocks, DynamicallyCompilableFamily, Family, Gamlss, GamlssBlocks,
    GlobalPenalty, HingeQuadraticPenalty, InitialEtaFromObservations, LinearFormBuilder,
    LinearPredictorBlock, Lower, LowerTriangularParameterBlock, ModelError, Mu, NoPenalty, Nu,
    Objective, ObjectiveScale, ObservationView, OffsetBlock, ParameterAxis, ParameterBlock,
    ParameterBlocks, ParameterDescriptor, ParameterLayout, ParameterName, ParameterPath,
    ParameterSlice, PredictorBlock, Product, RidgePenalty, Scalar, ScoreTilePolicy, Sigma,
    SumBlock, Tau, Vector, VectorParameterBlock,
};

#[derive(Debug, Clone, Copy)]
struct FixedSigmaNormal;

crate::impl_scalar_compilable_family!(
    impl for FixedSigmaNormal;
    parameters = (Mu,);
    arity = 1;
);

impl Family for FixedSigmaNormal {
    type Eta = f64;
    type Theta = f64;
    type GradientEta = f64;
    type Observation<'obs> = f64;
    type Workspace = ();
    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        *eta
    }

    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        let residual = y - theta;
        0.5 * residual * residual
    }

    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        let theta = self.theta(eta, _workspace);
        (self.nll(y, &theta, _workspace), eta - y)
    }
}

impl InitialEtaFromObservations<1> for FixedSigmaNormal {}

#[derive(Debug, Clone, Copy)]
struct SharedMeanPair;

impl Family for SharedMeanPair {
    type Eta = [f64; 2];
    type Theta = [f64; 2];
    type GradientEta = [f64; 2];
    type Observation<'obs> = [f64; 2];
    type Workspace = ();

    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        *eta
    }

    fn nll(
        &self,
        observation: Self::Observation<'_>,
        theta: &Self::Theta,
        _workspace: &mut Self::Workspace,
    ) -> f64 {
        observation
            .iter()
            .zip(theta)
            .map(|(y, mean)| 0.5 * (mean - y).powi(2))
            .sum()
    }

    fn nll_and_gradient_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        (
            self.nll(observation, eta, workspace),
            std::array::from_fn(|index| eta[index] - observation[index]),
        )
    }
}

impl CompilableFamily for SharedMeanPair {
    type Shape = Broadcast<Scalar<Mu>, 2>;

    fn eta_from_shape(values: <Self::Shape as crate::ParameterShape>::Values) -> Self::Eta {
        values
    }

    fn gradient_to_shape(
        gradient: &Self::GradientEta,
    ) -> <Self::Shape as crate::ParameterShape>::Values {
        *gradient
    }
}

#[derive(Debug, Clone, Copy)]
struct TwoParameterMock;

crate::impl_scalar_compilable_family!(
    impl for TwoParameterMock;
    parameters = (Mu, Sigma);
    arity = 2;
);

impl Family for TwoParameterMock {
    type Eta = (f64, f64);
    type Theta = (f64, f64);
    type GradientEta = (f64, f64);
    type Observation<'obs> = f64;
    type Workspace = ();
    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        *eta
    }

    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        let first = theta.0 - y;
        let second = theta.1 - 1.0;
        f64::midpoint(first * first, second * second)
    }

    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        let gradient = (eta.0 - y, eta.1 - 1.0);
        (self.nll(y, eta, _workspace), gradient)
    }
}

impl InitialEtaFromObservations<2> for TwoParameterMock {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RuntimeLayoutMock {
    key: usize,
    reverse: bool,
    whole_paths: bool,
    initial_count: usize,
}

impl Family for RuntimeLayoutMock {
    type Eta = [f64; 2];
    type Theta = [f64; 2];
    type GradientEta = [f64; 2];
    type Observation<'obs> = f64;
    type Workspace = ();

    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        *eta
    }

    fn nll(
        &self,
        observation: Self::Observation<'_>,
        theta: &Self::Theta,
        _workspace: &mut Self::Workspace,
    ) -> f64 {
        theta
            .iter()
            .map(|value| 0.5 * (value - observation).powi(2))
            .sum()
    }

    fn nll_and_gradient_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        (
            self.nll(observation, eta, workspace),
            [eta[0] - observation, eta[1] - observation],
        )
    }
}

impl DynamicallyCompilableFamily for RuntimeLayoutMock {
    fn dynamic_parameter_count(&self) -> usize {
        2
    }

    fn dynamic_layout_key(&self) -> DynamicLayoutKey {
        DynamicLayoutKey::new(vec![self.key, usize::from(self.whole_paths)])
    }

    fn eta_from_flat(&self, values: &[f64]) -> Self::Eta {
        assert_ne!(
            self.key,
            usize::MAX,
            "workspace value paths must not materialize dynamic eta"
        );
        let [first, second] = values else {
            return [f64::NAN; 2];
        };
        if self.reverse {
            [*second, *first]
        } else {
            [*first, *second]
        }
    }

    fn nll_eta_flat(
        &self,
        observation: Self::Observation<'_>,
        values: &[f64],
        _workspace: &mut Self::Workspace,
    ) -> f64 {
        let [first, second] = values else {
            return f64::INFINITY;
        };
        let (first, second) = if self.reverse {
            (*second, *first)
        } else {
            (*first, *second)
        };
        0.5_f64.mul_add(
            (first - observation).powi(2),
            0.5 * (second - observation).powi(2),
        )
    }

    fn nll_and_gradient_eta_flat(
        &self,
        observation: Self::Observation<'_>,
        values: &[f64],
        gradient: &mut [f64],
        workspace: &mut Self::Workspace,
    ) -> f64 {
        let [first, second] = values else {
            gradient.fill(f64::NAN);
            return f64::INFINITY;
        };
        let [first_gradient, second_gradient] = gradient else {
            gradient.fill(f64::NAN);
            return f64::INFINITY;
        };
        *first_gradient = *first - observation;
        *second_gradient = *second - observation;
        self.nll_eta_flat(observation, values, workspace)
    }

    fn gradient_to_flat(&self, gradient: &Self::GradientEta, out: &mut [f64]) {
        let [first, second] = out else {
            out.fill(f64::NAN);
            return;
        };
        if self.reverse {
            *first = gradient[1];
            *second = gradient[0];
        } else {
            *first = gradient[0];
            *second = gradient[1];
        }
    }

    fn dynamic_parameter_coordinate(&self, index: usize) -> (&'static str, ParameterPath) {
        if self.whole_paths {
            return (Mu::NAME, ParameterPath::whole());
        }
        let component = match (self.reverse, index) {
            (true, 0) => 1,
            (true, 1) => 0,
            _ => index,
        };
        (
            Mu::NAME,
            ParameterPath::from_axis(ParameterAxis::Vector { component }),
        )
    }

    fn initial_flat<'obs, Obs>(&self, _obs: &'obs Obs) -> Vec<f64>
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        vec![0.0; self.initial_count]
    }
}

fn runtime_layout_blocks(
    family: &RuntimeLayoutMock,
    nrows: usize,
) -> DynamicParameterBlocks<LinearPredictorBlock<DenseDesign>, NoPenalty> {
    DynamicParameterBlocks::try_new(
        family,
        (0..2)
            .map(|_| {
                (
                    LinearPredictorBlock::new(DenseDesign::intercept(nrows)),
                    NoPenalty,
                )
            })
            .collect(),
    )
    .unwrap()
}

#[derive(Debug, Clone, Copy)]
struct StructuredMock;

impl StructuredMock {
    #[allow(clippy::suboptimal_flops)]
    fn linear_prediction(eta: &([f64; 2], [[f64; 2]; 2])) -> f64 {
        eta.0[0] + 2.0 * eta.0[1] + 3.0 * eta.1[0][0] + 4.0 * eta.1[1][0] + 5.0 * eta.1[1][1]
    }
}

impl Family for StructuredMock {
    type Eta = ([f64; 2], [[f64; 2]; 2]);
    type Theta = Self::Eta;
    type GradientEta = Self::Eta;
    type Observation<'obs> = [f64; 2];
    type Workspace = ();
    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        *eta
    }

    fn nll(
        &self,
        observation: Self::Observation<'_>,
        theta: &Self::Theta,
        _workspace: &mut Self::Workspace,
    ) -> f64 {
        let residual = Self::linear_prediction(theta) - observation[0];
        0.5 * residual * residual
    }

    fn nll_and_gradient_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        let residual = Self::linear_prediction(eta) - observation[0];
        (
            self.nll(observation, eta, _workspace),
            (
                [residual, 2.0 * residual],
                [[3.0 * residual, 0.0], [4.0 * residual, 5.0 * residual]],
            ),
        )
    }
}

impl CompilableFamily for StructuredMock {
    type Shape = Product<Vector<Mu, 2>, Lower<CholeskyScale, 2>>;

    fn eta_from_shape(values: ([f64; 2], [[f64; 2]; 2])) -> Self::Eta {
        values
    }

    fn gradient_to_shape(gradient: &Self::GradientEta) -> ([f64; 2], [[f64; 2]; 2]) {
        *gradient
    }

    fn initial_shape<'obs, Obs>(&self, _obs: &'obs Obs) -> ([f64; 2], [[f64; 2]; 2])
    where
        Obs: ObservationView<'obs, Observation = [f64; 2]> + 'obs,
    {
        ([1.0, 2.0], [[3.0, 0.0], [4.0, 5.0]])
    }
}

#[derive(Debug, Clone, Copy)]
struct InitializingLocation;

crate::impl_scalar_compilable_family!(
    impl for InitializingLocation;
    parameters = (Mu,);
    arity = 1;
);

impl Family for InitializingLocation {
    type Eta = f64;
    type Theta = f64;
    type GradientEta = f64;
    type Observation<'obs> = f64;
    type Workspace = ();
    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        *eta
    }

    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        0.5 * (theta - y) * (theta - y)
    }

    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        (self.nll(y, eta, _workspace), *eta - y)
    }
}

impl InitialEtaFromObservations<1> for InitializingLocation {
    fn initial_eta_from_observations<'obs, Obs>(&self, _: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        2.0
    }
}

#[derive(Debug, Clone, Copy)]
struct NonFiniteInitializingLocation;

crate::impl_scalar_compilable_family!(
    impl for NonFiniteInitializingLocation;
    parameters = (Mu,);
    arity = 1;
);

impl Family for NonFiniteInitializingLocation {
    type Eta = f64;
    type Theta = f64;
    type GradientEta = f64;
    type Observation<'obs> = f64;
    type Workspace = ();
    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        *eta
    }

    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        0.5 * (theta - y) * (theta - y)
    }

    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        (self.nll(y, eta, _workspace), *eta - y)
    }
}

impl InitialEtaFromObservations<1> for NonFiniteInitializingLocation {
    fn initial_eta_from_observations<'obs, Obs>(&self, _: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        f64::NAN
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ShiftedObservations<'a> {
    y: &'a [f64],
    shift: f64,
    weight: f64,
}

impl<'row> ObservationView<'row> for ShiftedObservations<'_> {
    type Observation = f64;

    fn len(&self) -> usize {
        self.y.len()
    }

    fn observation_at(&'row self, row: usize) -> Self::Observation {
        self.y[row] + self.shift
    }

    fn weight_at(&self, _row: usize) -> f64 {
        self.weight
    }
}

#[test]
fn custom_one_parameter_family_uses_generic_family_contract() {
    let y = vec![1.0, 2.0];
    let x = DenseDesign::intercept(y.len());
    let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
    let mut model =
        Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
    let beta = vec![1.5];
    let mut grad = vec![0.0];

    assert_relative_eq!(model.value(&beta).unwrap(), 0.25);

    model.gradient(&beta, &mut grad).unwrap();

    assert_relative_eq!(grad[0], 0.0);
}

#[derive(Debug, Clone, Copy)]
struct BulkOnlyPredictor {
    nrows: usize,
}

impl PredictorBlock for BulkOnlyPredictor {
    fn nrows(&self) -> usize {
        self.nrows
    }

    fn nparams(&self) -> usize {
        1
    }

    fn eta_row(&self, _: usize, _: &[f64]) -> f64 {
        panic!("unmasked workspace objective must use bulk forward")
    }

    fn eta_range(&self, rows: Range<usize>, beta: &[f64], out: &mut [f64]) {
        assert_eq!(out.len(), rows.len());
        out.fill(beta[0]);
    }

    fn add_gradient_range(
        &self,
        rows: Range<usize>,
        scores: &[f64],
        _: &[f64],
        gradient: &mut [f64],
    ) {
        assert_eq!(scores.len(), rows.len());
        gradient[0] += scores.iter().sum::<f64>();
    }
}

#[test]
fn workspace_objective_uses_bulk_predictor_forward() {
    let y = [1.0, 2.0, 3.0];
    let mu = ParameterBlock::<Mu, _, _>::new(BulkOnlyPredictor { nrows: y.len() }, NoPenalty, 0);
    let model =
        Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
    let beta = [2.0];
    let mut gradient = [0.0];
    let mut workspace = model.gradient_workspace();

    let value = model
        .try_value_into_workspace(&beta, &mut workspace)
        .unwrap();
    let fused_value = model
        .try_value_gradient_into_workspace(&beta, &mut gradient, &mut workspace)
        .unwrap();

    assert_relative_eq!(value, 1.0);
    assert_relative_eq!(fused_value, value);
    assert_relative_eq!(gradient[0], 0.0);
}

#[test]
fn broadcast_shape_owns_one_predictor_and_sums_consumer_scores() {
    let y = [[1.0, 3.0], [2.0, 4.0]];
    let block = ParameterBlock::<Mu, _, _>::linear(
        DenseDesign::intercept(y.len()),
        RidgePenalty::new_unchecked(0.5),
        0,
    );
    let model = Gamlss::try_new_with_observations(
        SharedMeanPair,
        ParameterBlocks::from_assigned(block),
        y.as_slice(),
    )
    .unwrap();
    let beta = [0.5];
    let mut likelihood_gradient = [0.0];
    let mut objective_gradient = [0.0];

    model
        .try_likelihood_value_gradient_into(&beta, &mut likelihood_gradient)
        .unwrap();
    model
        .try_value_gradient_into(&beta, &mut objective_gradient)
        .unwrap();

    assert_relative_eq!(likelihood_gradient[0], -8.0);
    assert_relative_eq!(objective_gradient[0], -7.5);
    assert_eq!(model.parameter_descriptors().len(), 1);
    assert_eq!(model.parameter_descriptors()[0].range, 0..1);
}

#[test]
fn structured_vector_lower_triangular_blocks_use_generic_contract() {
    let y = [[3.0, 0.0], [5.0, 0.0]];
    let n = y.len();
    let vector = VectorParameterBlock::<Mu, 2, _, _>::new(
        [
            LinearPredictorBlock::new(DenseDesign::intercept(n)),
            LinearPredictorBlock::new(DenseDesign::intercept(n)),
        ],
        RidgePenalty::new_unchecked(0.25),
        99,
    );
    let lower = LowerTriangularParameterBlock::<CholeskyScale, 2, _, _>::new(
        vec![
            LinearPredictorBlock::new(DenseDesign::intercept(n)),
            LinearPredictorBlock::new(DenseDesign::intercept(n)),
            LinearPredictorBlock::new(DenseDesign::intercept(n)),
        ],
        NoPenalty,
        99,
    );
    let blocks = ParameterBlocks::new((vector, lower));
    let model = Gamlss::try_new_with_observations(StructuredMock, blocks, y.as_slice()).unwrap();

    assert_eq!(model.nparams(), 5);
    assert_eq!(
        model.parameter_layout().unique_slice("mu").unwrap(),
        Some(0..2)
    );
    assert_eq!(
        model.parameter_layout().unique_slice("cholesky").unwrap(),
        Some(2..5)
    );
    assert_eq!(
        model.parameter_descriptors(),
        vec![
            ParameterDescriptor::vector_component("mu", 0, 0..1),
            ParameterDescriptor::vector_component("mu", 1, 1..2),
            ParameterDescriptor::lower_triangular_entry("cholesky", 0, 0, 2..3),
            ParameterDescriptor::lower_triangular_entry("cholesky", 1, 0, 3..4),
            ParameterDescriptor::lower_triangular_entry("cholesky", 1, 1, 4..5),
        ]
    );

    let mut parts = Vec::new();
    model.visit_parameter_descriptors(|index, descriptor| {
        parts.push((index, descriptor.role, descriptor.path, descriptor.range));
    });
    assert_eq!(
        parts,
        vec![
            (
                0,
                "mu",
                ParameterPath::new(vec![ParameterAxis::Vector { component: 0 }]),
                0..1
            ),
            (
                1,
                "mu",
                ParameterPath::new(vec![ParameterAxis::Vector { component: 1 }]),
                1..2
            ),
            (
                2,
                "cholesky",
                ParameterPath::new(vec![ParameterAxis::Lower { row: 0, col: 0 }]),
                2..3,
            ),
            (
                3,
                "cholesky",
                ParameterPath::new(vec![ParameterAxis::Lower { row: 1, col: 0 }]),
                3..4,
            ),
            (
                4,
                "cholesky",
                ParameterPath::new(vec![ParameterAxis::Lower { row: 1, col: 1 }]),
                4..5,
            ),
        ]
    );
    assert_eq!(
        model.initial_parameters().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0]
    );

    let beta = vec![0.2, -0.1, 0.3, 0.4, -0.2];
    let eta = model.predict_eta_row(&beta, 0).unwrap();
    assert_relative_eq!(eta.0[0], 0.2);
    assert_relative_eq!(eta.0[1], -0.1);
    assert_relative_eq!(eta.1[0][0], 0.3);
    assert_relative_eq!(eta.1[0][1], 0.0);
    assert_relative_eq!(eta.1[1][0], 0.4);
    assert_relative_eq!(eta.1[1][1], -0.2);

    let mut gradient = vec![0.0; model.nparams()];
    let value = model.try_value_gradient_into(&beta, &mut gradient).unwrap();
    assert!(value.is_finite());

    let epsilon = 1.0e-6;
    for index in 0..beta.len() {
        let mut plus = beta.clone();
        plus[index] += epsilon;
        let mut minus = beta.clone();
        minus[index] -= epsilon;
        let finite_difference =
            (model.try_value(&plus).unwrap() - model.try_value(&minus).unwrap()) / (2.0 * epsilon);
        assert_relative_eq!(gradient[index], finite_difference, epsilon = 1.0e-6);
    }
}

#[test]
fn initial_parameters_default_to_zero_for_custom_family() {
    let y = vec![1.0, 2.0];
    let x = DenseDesign::intercept(y.len());
    let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
    let model =
        Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();

    assert_eq!(model.initial_parameters().unwrap(), vec![0.0]);
}

#[test]
fn initial_parameters_write_intercept_like_constant() {
    let y = vec![1.0, 2.0, 3.0];
    let x = DenseDesign::intercept(y.len());
    let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
    let mut model = Gamlss::try_new(
        InitializingLocation,
        ParameterBlocks::from_assigned((mu,)),
        &y,
    )
    .unwrap();
    let beta = model.initial_parameters().unwrap();

    assert_eq!(beta.len(), model.nparams());
    assert_eq!(beta, vec![2.0]);
    assert!(model.value(&beta).unwrap().is_finite());
}

#[test]
fn initial_parameters_leave_no_intercept_design_zero() {
    let y = vec![1.0, 2.0, 3.0];
    let x = DenseDesign::from_rows(&[[0.0], [1.0], [2.0]]);
    let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
    let model = Gamlss::try_new(
        InitializingLocation,
        ParameterBlocks::from_assigned((mu,)),
        &y,
    )
    .unwrap();

    assert_eq!(model.initial_parameters().unwrap(), vec![0.0]);
}

#[test]
fn initial_parameters_write_first_compatible_sum_term() {
    let y = vec![1.0, 2.0, 3.0];
    let first = LinearPredictorBlock::new(DenseDesign::from_rows(&[[0.0], [1.0], [2.0]]));
    let second = LinearPredictorBlock::new(DenseDesign::intercept(y.len()));
    let predictor = SumBlock::new((first, second));
    let mu = ParameterBlock::<Mu, _, _>::new(predictor, NoPenalty, 0);
    let model = Gamlss::try_new(
        InitializingLocation,
        ParameterBlocks::from_assigned((mu,)),
        &y,
    )
    .unwrap();

    assert_eq!(model.initial_parameters().unwrap(), vec![0.0, 2.0]);
}

#[test]
fn initial_parameters_account_for_sum_block_constant_baselines() {
    let y = vec![1.0, 2.0, 3.0];
    let offset = OffsetBlock::new(y.len(), 10.0);
    let intercept = LinearPredictorBlock::new(DenseDesign::intercept(y.len()));
    let predictor = SumBlock::new((offset, intercept));
    let mu = ParameterBlock::<Mu, _, _>::new(predictor, NoPenalty, 0);
    let model = Gamlss::try_new(
        InitializingLocation,
        ParameterBlocks::from_assigned((mu,)),
        &y,
    )
    .unwrap();
    let beta = model.initial_parameters().unwrap();

    assert_eq!(beta, vec![-8.0]);
    assert_eq!(model.predict_eta(&beta).unwrap(), vec![2.0, 2.0, 2.0]);
}

#[test]
fn initial_parameters_ignore_nonfinite_family_starts() {
    let y = vec![1.0, 2.0, 3.0];
    let x = DenseDesign::intercept(y.len());
    let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
    let model = Gamlss::try_new(
        NonFiniteInitializingLocation,
        ParameterBlocks::from_assigned((mu,)),
        &y,
    )
    .unwrap();

    assert_eq!(model.initial_parameters().unwrap(), vec![0.0]);
}

#[test]
fn model_accepts_user_defined_observation_view() {
    let y = vec![1.0, 2.0];
    let obs = ShiftedObservations {
        y: &y,
        shift: 1.0,
        weight: 0.5,
    };
    let x = DenseDesign::intercept(obs.len());
    let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
    let mut model = Gamlss::try_new_with_observations(
        FixedSigmaNormal,
        ParameterBlocks::from_assigned((mu,)),
        obs,
    )
    .unwrap();
    let beta = vec![2.5];
    let mut grad = vec![0.0];

    assert_relative_eq!(model.value(&beta).unwrap(), 0.125);

    model.gradient(&beta, &mut grad).unwrap();

    assert_relative_eq!(grad[0], 0.0);
}

#[test]
fn custom_observation_view_rejects_invalid_weight() {
    let y = vec![1.0, 2.0];
    let obs = ShiftedObservations {
        y: &y,
        shift: 0.0,
        weight: f64::NAN,
    };
    let x = DenseDesign::intercept(obs.len());
    let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);

    assert_eq!(
        Gamlss::try_new_with_observations(
            FixedSigmaNormal,
            ParameterBlocks::from_assigned((mu,)),
            obs
        )
        .unwrap_err(),
        ModelError::InvalidWeight { index: 0 }
    );
}

#[test]
#[allow(clippy::float_cmp)]
fn model_borrows_response_without_copying() {
    let y = vec![1.0, 2.0];
    let x = DenseDesign::intercept(y.len());
    let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
    let model =
        Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();

    assert_eq!(model.obs.as_ptr(), y.as_ptr());
    assert_eq!(model.obs, y.as_slice());
    assert_eq!(model.obs.weight_at(0), 1.0);
}

#[test]
fn unweighted_model_matches_unit_weights() {
    let y = vec![1.0, 2.0];
    let unit_weights = vec![1.0, 1.0];
    let x = DenseDesign::intercept(y.len());
    let mu = ParameterBlock::<Mu, _, _>::linear(x.clone(), NoPenalty, 0);
    let weighted_mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
    let model =
        Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
    let weighted = Gamlss::try_new_weighted(
        FixedSigmaNormal,
        ParameterBlocks::from_assigned((weighted_mu,)),
        &y,
        &unit_weights,
    )
    .unwrap();
    let beta = vec![1.5];
    let mut grad = vec![0.0];
    let mut weighted_grad = vec![0.0];

    assert_relative_eq!(
        model.try_value(&beta).unwrap(),
        weighted.try_value(&beta).unwrap()
    );

    model.try_gradient_into(&beta, &mut grad).unwrap();
    weighted
        .try_gradient_into(&beta, &mut weighted_grad)
        .unwrap();

    assert_relative_eq!(grad[0], weighted_grad[0]);
}

#[test]
fn model_exposes_weight_sum_and_effective_nobs() {
    let y = vec![1.0, 2.0, 3.0];
    let weights = vec![0.5, 0.0, 2.0];
    let x = DenseDesign::intercept(y.len());
    let mu = ParameterBlock::<Mu, _, _>::linear(x.clone(), NoPenalty, 0);
    let weighted_mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
    let model =
        Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
    let weighted = Gamlss::try_new_weighted(
        FixedSigmaNormal,
        ParameterBlocks::from_assigned((weighted_mu,)),
        &y,
        &weights,
    )
    .unwrap();

    assert_relative_eq!(model.weight_sum(), 3.0);
    assert_relative_eq!(model.effective_nobs(), 3.0);
    assert_relative_eq!(weighted.weight_sum(), 2.5);
    assert_relative_eq!(weighted.effective_nobs(), 2.5);

    let mean_weighted = weighted.with_objective_scale(ObjectiveScale::Mean);
    let row0_nll = 0.5 * (1.0_f64 - 2.0).powi(2);
    let row1_nll = 0.5 * (2.0_f64 - 2.0).powi(2);
    let row2_nll = 0.5 * (3.0_f64 - 2.0).powi(2);
    let weighted_nll_sum = weights
        .iter()
        .copied()
        .zip([row0_nll, row1_nll, row2_nll])
        .map(|(weight, nll)| weight * nll)
        .sum::<f64>();

    assert_relative_eq!(
        mean_weighted.try_value(&[2.0]).unwrap(),
        weighted_nll_sum / mean_weighted.weight_sum()
    );
}

#[test]
fn zero_weight_excludes_observation_from_value_and_gradient() {
    let y = vec![1.0, 10.0];
    let weights = vec![1.0, 0.0];
    let x = DenseDesign::intercept(y.len());
    let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
    let model = Gamlss::try_new_weighted(
        FixedSigmaNormal,
        ParameterBlocks::from_assigned((mu,)),
        &y,
        &weights,
    )
    .unwrap();
    let beta = vec![1.0];
    let mut grad = vec![f64::NAN];

    assert_relative_eq!(model.try_value(&beta).unwrap(), 0.0);

    model.try_gradient_into(&beta, &mut grad).unwrap();

    assert_relative_eq!(grad[0], 0.0);
}

#[test]
fn zero_weight_excludes_invalid_observation_from_value_and_gradient() {
    let y = vec![1.0, f64::NAN];
    let weights = vec![1.0, 0.0];
    let x = DenseDesign::intercept(y.len());
    let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
    let model = Gamlss::try_new_weighted(
        FixedSigmaNormal,
        ParameterBlocks::from_assigned((mu,)),
        &y,
        &weights,
    )
    .unwrap();
    let beta = vec![1.0];
    let mut grad = vec![f64::NAN];

    assert_relative_eq!(model.try_value(&beta).unwrap(), 0.0);

    model.try_gradient_into(&beta, &mut grad).unwrap();

    assert_relative_eq!(grad[0], 0.0);
}

#[test]
fn zero_weight_excludes_invalid_observation_and_design_row_from_value_and_gradient() {
    let y = vec![1.0, f64::NAN, 2.0];
    let weights = vec![1.0, 0.0, 1.0];
    let x =
        DenseDesign::from_row_major(3, 2, vec![1.0, 0.0, f64::NAN, f64::NAN, 1.0, 1.0]).unwrap();
    let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
    let model = Gamlss::try_new_weighted(
        FixedSigmaNormal,
        ParameterBlocks::from_assigned((mu,)),
        &y,
        &weights,
    )
    .unwrap();
    let beta = vec![1.0, 1.0];
    let mut grad = vec![f64::NAN, f64::NAN];

    assert_relative_eq!(model.try_value(&beta).unwrap(), 0.0);

    model.try_gradient_into(&beta, &mut grad).unwrap();

    assert!(grad.iter().all(|value| value.is_finite()));
    assert_relative_eq!(grad[0], 0.0);
    assert_relative_eq!(grad[1], 0.0);

    for tile_rows in [1, 2] {
        let policy = ScoreTilePolicy::try_max_rows(tile_rows).unwrap();
        let mut workspace = model.gradient_workspace_with_policy(policy);
        let workspace_value = model
            .try_value_into_workspace(&beta, &mut workspace)
            .unwrap();
        let workspace_likelihood = model
            .try_likelihood_value_into_workspace(&beta, &mut workspace)
            .unwrap();
        assert_relative_eq!(workspace_value, 0.0);
        assert_relative_eq!(workspace_likelihood, 0.0);

        grad.fill(f64::NAN);
        let value = model
            .try_value_gradient_into_workspace(&beta, &mut grad, &mut workspace)
            .unwrap();

        assert_relative_eq!(value, 0.0);
        assert!(grad.iter().all(|value| value.is_finite()));
        assert_relative_eq!(grad[0], 0.0);
        assert_relative_eq!(grad[1], 0.0);
    }
}

#[test]
fn weighted_model_rejects_invalid_weights() {
    let y = vec![1.0, 2.0];
    let short_weights = vec![1.0];
    let infinite_weights = vec![1.0, f64::INFINITY];
    let negative_weights = vec![1.0, -0.1];
    let mu = ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(2), NoPenalty, 0);

    assert_eq!(
        Gamlss::try_new_weighted(
            FixedSigmaNormal,
            ParameterBlocks::from_assigned((mu.clone(),)),
            &y,
            &short_weights,
        )
        .unwrap_err(),
        ModelError::WeightLength {
            expected: 2,
            actual: 1,
        }
    );
    assert_eq!(
        Gamlss::try_new_weighted(
            FixedSigmaNormal,
            ParameterBlocks::from_assigned((mu.clone(),)),
            &y,
            &infinite_weights,
        )
        .unwrap_err(),
        ModelError::InvalidWeight { index: 1 }
    );
    assert_eq!(
        Gamlss::try_new_weighted(
            FixedSigmaNormal,
            ParameterBlocks::from_assigned((mu,)),
            &y,
            &negative_weights
        )
        .unwrap_err(),
        ModelError::InvalidWeight { index: 1 }
    );
}

#[test]
fn scalar_response_is_permissive_but_strict_constructor_rejects_non_finite() {
    let y = vec![1.0, f64::NAN];
    let mu = ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(2), NoPenalty, 0);

    Gamlss::try_new(
        FixedSigmaNormal,
        ParameterBlocks::from_assigned((mu.clone(),)),
        &y,
    )
    .unwrap();

    assert_eq!(
        Gamlss::try_new_strict(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y)
            .unwrap_err(),
        ModelError::InvalidObservation { index: 1 }
    );
}

#[test]
fn finite_scalar_observation_adapter_rejects_non_finite() {
    let y = vec![1.0, f64::NEG_INFINITY];

    assert_eq!(
        crate::FiniteScalarObservations::new(&y).unwrap_err(),
        ModelError::InvalidObservation { index: 1 }
    );
}

#[test]
fn prediction_api_returns_eta_and_theta_for_one_parameter_model() {
    let y = vec![1.0, 2.0];
    let x = DenseDesign::from_rows(&[[1.0, 0.0], [1.0, 2.0]]);
    let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
    let model =
        Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
    let beta = vec![0.5, 0.25];

    assert_relative_eq!(model.predict_eta_row(&beta, 1).unwrap(), 1.0);
    assert_relative_eq!(model.predict_theta_row(&beta, 1).unwrap(), 1.0);
    assert_eq!(model.predict_eta(&beta).unwrap(), vec![0.5, 1.0]);
    assert_eq!(model.predict_theta(&beta).unwrap(), vec![0.5, 1.0]);

    let mut theta = vec![f64::NAN; model.nobs()];
    model.predict_theta_into(&beta, &mut theta).unwrap();
    assert_eq!(theta, model.predict_theta(&beta).unwrap());

    let mut streamed = Vec::new();
    model
        .for_each_theta(&beta, |row, theta| streamed.push((row, theta)))
        .unwrap();
    assert_eq!(streamed, vec![(0, 0.5), (1, 1.0)]);
    assert_eq!(
        model.predict_theta_into(&beta, &mut [0.0]).unwrap_err(),
        ModelError::ResponseLength {
            expected: 2,
            actual: 1,
        }
    );
}

#[test]
fn prediction_api_rejects_invalid_parameter_length_and_row() {
    let y = vec![1.0];
    let x = DenseDesign::intercept(y.len());
    let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
    let model =
        Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();

    assert_eq!(
        model.predict_eta_row(&[], 0).unwrap_err(),
        ModelError::BetaLength {
            expected: 1,
            actual: 0,
        }
    );
    assert_eq!(
        model.predict_eta_row(&[0.0], 1).unwrap_err(),
        ModelError::RowOutOfBounds { row: 1, nrows: 1 }
    );
}

#[test]
fn dynamic_layout_identity_is_checked_for_models_and_prediction() {
    let canonical = RuntimeLayoutMock {
        key: 7,
        reverse: false,
        whole_paths: false,
        initial_count: 2,
    };
    let different_key = RuntimeLayoutMock {
        key: 8,
        reverse: false,
        whole_paths: false,
        initial_count: 2,
    };
    let different_coordinates = RuntimeLayoutMock {
        key: 7,
        reverse: true,
        whole_paths: false,
        initial_count: 2,
    };
    let canonical_eta = canonical.eta_from_flat(&[1.0, 2.0]);
    assert_relative_eq!(canonical_eta[0], 1.0);
    assert_relative_eq!(canonical_eta[1], 2.0);
    let reversed_eta = different_coordinates.eta_from_flat(&[1.0, 2.0]);
    assert_relative_eq!(reversed_eta[0], 2.0);
    assert_relative_eq!(reversed_eta[1], 1.0);
    let mut flat_gradient = [0.0; 2];
    different_coordinates.gradient_to_flat(&[3.0, 4.0], &mut flat_gradient);
    assert_relative_eq!(flat_gradient[0], 4.0);
    assert_relative_eq!(flat_gradient[1], 3.0);

    let y = [0.0];
    assert_eq!(
        Gamlss::try_new(
            different_key,
            runtime_layout_blocks(&canonical, y.len()),
            &y,
        )
        .unwrap_err(),
        ModelError::DynamicLayoutMismatch {
            expected: DynamicLayoutKey::new(vec![8, 0]),
            got: DynamicLayoutKey::new(vec![7, 0]),
        }
    );
    assert!(matches!(
        Gamlss::try_new(
            different_coordinates,
            runtime_layout_blocks(&canonical, y.len()),
            &y,
        )
        .unwrap_err(),
        ModelError::DynamicCoordinateMismatch { index: 0, .. }
    ));

    let model = Gamlss::try_new(canonical, runtime_layout_blocks(&canonical, y.len()), &y).unwrap();
    let prediction_blocks = runtime_layout_blocks(&different_key, 2);
    match model.prediction_view(&prediction_blocks).unwrap_err() {
        ModelError::PredictionLayoutIdentityMismatch {
            expected_key,
            got_key,
            expected_descriptors,
            got_descriptors,
        } => {
            assert_eq!(expected_key, Some(DynamicLayoutKey::new(vec![7, 0])));
            assert_eq!(got_key, Some(DynamicLayoutKey::new(vec![8, 0])));
            assert_eq!(expected_descriptors, got_descriptors);
        }
        error => panic!("unexpected prediction validation error: {error:?}"),
    }

    let prediction_blocks = runtime_layout_blocks(&different_coordinates, 2);
    assert!(matches!(
        model.prediction_view(&prediction_blocks).unwrap_err(),
        ModelError::PredictionLayoutIdentityMismatch {
            expected_key: Some(_),
            got_key: Some(_),
            ..
        }
    ));
}

#[test]
fn dynamic_initializer_coordinate_count_is_checked() {
    let family = RuntimeLayoutMock {
        key: 10,
        reverse: false,
        whole_paths: false,
        initial_count: 1,
    };
    let y = [0.0];
    let model = Gamlss::try_new(family, runtime_layout_blocks(&family, y.len()), &y).unwrap();

    assert_eq!(
        model.initial_parameters().unwrap_err(),
        ModelError::DynamicInitialValueCount {
            expected: 2,
            actual: 1,
        }
    );
}

#[test]
fn descriptor_selectors_reject_ambiguous_roles_and_paths() {
    let family = RuntimeLayoutMock {
        key: 9,
        reverse: false,
        whole_paths: true,
        initial_count: 2,
    };
    let y = [0.0];
    let mut model = Gamlss::try_new(family, runtime_layout_blocks(&family, y.len()), &y).unwrap();
    let beta = [0.25, -0.5];
    let layout = model.parameter_layout();

    assert_eq!(layout.ranges_of::<Mu>(), vec![0..1, 1..2]);
    assert_eq!(
        layout.unique_slice_of::<Mu>().unwrap_err(),
        ModelError::AmbiguousParameter {
            name: "mu".to_owned(),
            matches: 2,
        }
    );
    assert_eq!(
        model
            .unique_parameter_descriptor_at_path::<Mu>(&ParameterPath::whole())
            .unwrap_err(),
        ModelError::AmbiguousParameter {
            name: "mu".to_owned(),
            matches: 2,
        }
    );
    assert_eq!(
        model.block_objective_for::<Mu>(beta.to_vec()).unwrap_err(),
        ModelError::AmbiguousParameter {
            name: "mu".to_owned(),
            matches: 2,
        }
    );

    let descriptors = model.parameter_descriptors_of::<Mu>();
    assert_eq!(descriptors.len(), 2);
    assert_eq!(descriptors[0].0, 0);
    assert_eq!(descriptors[1].0, 1);
    assert_eq!(descriptors[0].1.path, ParameterPath::whole());
    assert_eq!(descriptors[1].1.path, ParameterPath::whole());
    assert_eq!(
        model.parameter_descriptor_index(&descriptors[1].1).unwrap(),
        Some(1)
    );

    let unpacked = model.unpack_parameters(&beta).unwrap();
    assert_eq!(unpacked.blocks_of::<Mu>().count(), 2);
    assert_eq!(unpacked.block_at(1).unwrap().descriptor, descriptors[1].1);
    assert_eq!(unpacked.block_at(1).unwrap().coefficients, vec![-0.5]);
    assert_eq!(
        unpacked.unique_coefficients_of::<Mu>().unwrap_err(),
        ModelError::AmbiguousParameter {
            name: "mu".to_owned(),
            matches: 2,
        }
    );

    {
        let mut second = model.block_objective_at(1, beta.to_vec()).unwrap();
        assert_eq!(second.descriptor(), &descriptors[1].1);
        let mut gradient = [0.0];
        let value = second.value_gradient(&[-0.5], &mut gradient).unwrap();
        assert_relative_eq!(value, 0.15625);
        assert_relative_eq!(gradient[0], -0.5);
    }
    {
        let mut second = model
            .block_objective_for_descriptor(&descriptors[1].1, beta.to_vec())
            .unwrap();
        assert_eq!(second.dim(), 1);
        assert_relative_eq!(second.value(&[-0.5]).unwrap(), 0.15625);
    }
    assert_eq!(
        model.block_objective_at(2, beta.to_vec()).unwrap_err(),
        ModelError::ParameterDescriptorIndexOutOfBounds { index: 2, count: 2 }
    );
    let unknown = ParameterDescriptor::whole("mu", 2..3);
    assert_eq!(
        model
            .block_objective_for_descriptor(&unknown, beta.to_vec())
            .unwrap_err(),
        ModelError::UnknownParameterDescriptor {
            descriptor: unknown,
        }
    );

    let selected = descriptors[1].1.clone();
    let mut workspace_objective = model.into_workspace_objective();
    assert_eq!(
        workspace_objective
            .block_objective_for::<Mu>(beta.to_vec())
            .unwrap_err(),
        ModelError::AmbiguousParameter {
            name: "mu".to_owned(),
            matches: 2,
        }
    );
    {
        let descriptor_objective = workspace_objective
            .block_objective_for_descriptor(&selected, beta.to_vec())
            .unwrap();
        assert_eq!(descriptor_objective.descriptor(), &selected);
    }
    {
        let indexed_objective = workspace_objective
            .block_objective_at(1, beta.to_vec())
            .unwrap();
        assert_eq!(indexed_objective.descriptor(), &selected);
    }
}

#[test]
fn prediction_api_uses_compatible_blocks_for_new_rows() {
    let y = vec![1.0, 2.0];
    let train_x = DenseDesign::from_rows(&[[1.0, 0.0], [1.0, 1.0]]);
    let mu = ParameterBlock::<Mu, _, _>::linear(train_x, NoPenalty, 0);
    let model =
        Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
    let prediction_x = DenseDesign::from_rows(&[[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]);
    let prediction_mu = ParameterBlock::<Mu, _, _>::linear(prediction_x, NoPenalty, 0);
    let prediction_blocks = ParameterBlocks::from_assigned((prediction_mu,));
    let beta = vec![0.5, 0.25];
    let prediction = model.prediction_view(&prediction_blocks).unwrap();

    assert_relative_eq!(
        model
            .predict_eta_row_with_blocks(&beta, &prediction_blocks, 1)
            .unwrap(),
        1.25
    );
    assert_eq!(prediction.nrows(), 3);
    assert_eq!(prediction.nparams(), 2);
    assert_relative_eq!(prediction.predict_eta_row(&beta, 1).unwrap(), 1.25);
    assert_relative_eq!(prediction.predict_theta_row(&beta, 1).unwrap(), 1.25);
    assert_eq!(
        model
            .predict_eta_with_blocks(&beta, &prediction_blocks)
            .unwrap(),
        vec![1.0, 1.25, 1.5]
    );
    assert_eq!(prediction.predict_eta(&beta).unwrap(), vec![1.0, 1.25, 1.5]);
    assert_eq!(
        model
            .predict_theta_with_blocks(&beta, &prediction_blocks)
            .unwrap(),
        vec![1.0, 1.25, 1.5]
    );
    assert_eq!(
        prediction.predict_theta(&beta).unwrap(),
        vec![1.0, 1.25, 1.5]
    );

    let mut theta = vec![f64::NAN; 3];
    model
        .predict_theta_with_blocks_into(&beta, &prediction_blocks, &mut theta)
        .unwrap();
    assert_eq!(theta, vec![1.0, 1.25, 1.5]);
    theta.fill(f64::NAN);
    prediction.predict_theta_into(&beta, &mut theta).unwrap();
    assert_eq!(theta, vec![1.0, 1.25, 1.5]);

    let mut streamed = Vec::new();
    model
        .for_each_theta_with_blocks(&beta, &prediction_blocks, |row, theta| {
            streamed.push((row, theta));
        })
        .unwrap();
    assert_eq!(streamed, vec![(0, 1.0), (1, 1.25), (2, 1.5)]);
    streamed.clear();
    prediction
        .for_each_theta(&beta, |row, theta| streamed.push((row, theta)))
        .unwrap();
    assert_eq!(streamed, vec![(0, 1.0), (1, 1.25), (2, 1.5)]);
}

#[test]
fn prediction_api_rejects_incompatible_prediction_blocks() {
    let y = vec![1.0, 2.0];
    let train_x = DenseDesign::from_rows(&[[1.0, 0.0], [1.0, 1.0]]);
    let mu = ParameterBlock::<Mu, _, _>::linear(train_x, NoPenalty, 0);
    let model =
        Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
    let prediction_x = DenseDesign::from_rows(&[[1.0, 2.0, 3.0]]);
    let prediction_mu = ParameterBlock::<Mu, _, _>::linear(prediction_x, NoPenalty, 0);
    let prediction_blocks = ParameterBlocks::from_assigned((prediction_mu,));

    assert_eq!(
        model
            .predict_eta_with_blocks(&[0.5, 0.25], &prediction_blocks)
            .unwrap_err(),
        ModelError::PredictionLayoutMismatch {
            expected: ParameterLayout::new(vec![ParameterSlice {
                name: "mu",
                range: 0..2,
            }]),
            got: ParameterLayout::new(vec![ParameterSlice {
                name: "mu",
                range: 0..3,
            }]),
        }
    );
    assert_eq!(
        model.prediction_view(&prediction_blocks).unwrap_err(),
        ModelError::PredictionLayoutMismatch {
            expected: ParameterLayout::new(vec![ParameterSlice {
                name: "mu",
                range: 0..2,
            }]),
            got: ParameterLayout::new(vec![ParameterSlice {
                name: "mu",
                range: 0..3,
            }]),
        }
    );
}

#[test]
fn prediction_api_rejects_same_length_different_parameter_layout() {
    let y = vec![1.0, 2.0];
    let train_mu_x = DenseDesign::from_rows(&[[1.0, 0.0], [1.0, 1.0]]);
    let train_sigma_x = DenseDesign::intercept(y.len());
    let train_mu = ParameterBlock::<Mu, _, _>::linear(train_mu_x, NoPenalty, 0);
    let train_sigma = ParameterBlock::<Sigma, _, _>::linear(train_sigma_x, NoPenalty, 2);
    let model = Gamlss::try_new(
        TwoParameterMock,
        ParameterBlocks::from_assigned((train_mu, train_sigma)),
        &y,
    )
    .unwrap();

    let prediction_mu_x = DenseDesign::intercept(1);
    let prediction_sigma_x = DenseDesign::from_rows(&[[1.0, 2.0]]);
    let prediction_mu = ParameterBlock::<Mu, _, _>::linear(prediction_mu_x, NoPenalty, 0);
    let prediction_sigma = ParameterBlock::<Sigma, _, _>::linear(prediction_sigma_x, NoPenalty, 1);
    let prediction_blocks = ParameterBlocks::from_assigned((prediction_mu, prediction_sigma));

    assert_eq!(
        model
            .predict_eta_with_blocks(&[0.5, 0.25, 0.75], &prediction_blocks)
            .unwrap_err(),
        ModelError::PredictionLayoutMismatch {
            expected: ParameterLayout::new(vec![
                ParameterSlice {
                    name: "mu",
                    range: 0..2,
                },
                ParameterSlice {
                    name: "sigma",
                    range: 2..3,
                },
            ]),
            got: ParameterLayout::new(vec![
                ParameterSlice {
                    name: "mu",
                    range: 0..1,
                },
                ParameterSlice {
                    name: "sigma",
                    range: 1..3,
                },
            ]),
        }
    );
}

#[test]
fn rejects_overflowing_parameter_block_ranges() {
    let y = vec![1.0];
    let x = DenseDesign::intercept(y.len());
    let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, usize::MAX);

    assert_eq!(
        Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap_err(),
        ModelError::BlockRangeOverflow {
            parameter: "mu",
            offset: usize::MAX,
            len: 1,
        }
    );
}

#[test]
fn blocks_try_len_reports_overflowing_total_length() {
    let y = [1.0];
    let x = DenseDesign::intercept(y.len());
    let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, usize::MAX);
    let blocks = ParameterBlocks::from_assigned((mu,));

    assert_eq!(
        GamlssBlocks::<FixedSigmaNormal>::try_len(&blocks).unwrap_err(),
        ModelError::BlockRangeOverflow {
            parameter: "mu",
            offset: usize::MAX,
            len: 1,
        }
    );
}

#[test]
fn blocks_validate_rejects_invalid_local_penalty() {
    let y = vec![1.0];
    let x = DenseDesign::intercept(y.len());
    let mu = ParameterBlock::<Mu, _, _>::linear(x, RidgePenalty::new_unchecked(f64::NAN), 0);

    assert_eq!(
        Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap_err(),
        ModelError::InvalidParameter {
            parameter: "ridge penalty lambda",
            expected: "finite and >= 0",
        }
    );
}

#[test]
fn workspace_objective_matches_model_gradient_on_repeated_calls() {
    let y = vec![1.0, 2.0];
    let x = DenseDesign::from_rows(&[[1.0, 0.0], [1.0, 1.0]]);
    let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
    let model =
        Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
    let mut workspace_objective = model.clone().into_workspace_objective();

    for beta in [vec![1.0, 0.25], vec![1.5, -0.1]] {
        let mut expected_grad = vec![0.0; beta.len()];
        let mut workspace_grad = vec![0.0; beta.len()];

        model.try_gradient_into(&beta, &mut expected_grad).unwrap();
        let workspace_value = workspace_objective
            .value_gradient(&beta, &mut workspace_grad)
            .unwrap();

        assert_relative_eq!(workspace_value, model.try_value(&beta).unwrap());
        for (actual, expected) in workspace_grad.iter().zip(&expected_grad) {
            assert_relative_eq!(actual, expected);
        }
    }
}

#[test]
fn score_tile_policy_flows_into_workspace_and_objective() {
    let y = vec![1.0; 7];
    let x = DenseDesign::intercept(y.len());
    let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
    let model =
        Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
    let policy = ScoreTilePolicy::try_max_bytes(2 * size_of::<f64>()).unwrap();

    let default_workspace = model.gradient_workspace();
    assert_eq!(
        default_workspace.gradient().score_tile_policy(),
        ScoreTilePolicy::automatic()
    );
    assert_eq!(default_workspace.gradient().score_tile_rows(), y.len());

    let custom_workspace = model.gradient_workspace_with_policy(policy);
    assert_eq!(custom_workspace.gradient().score_tile_policy(), policy);
    assert_eq!(custom_workspace.gradient().score_tile_rows(), 2);
    assert_eq!(
        custom_workspace.gradient().score_tile_bytes(),
        2 * size_of::<f64>()
    );

    let objective = model.into_workspace_objective_with_policy(policy);
    assert_eq!(objective.workspace().gradient().score_tile_policy(), policy);
    assert_eq!(objective.workspace().gradient().score_tile_rows(), 2);
}

#[test]
fn static_executor_is_additive_across_score_tile_sizes() {
    let y = vec![1.0, -0.5, 2.0, 0.25, 3.0, -1.0, 1.5];
    let weights = vec![0.5, 0.0, 2.0, 1.5, 0.25, 3.0, 0.75];
    let x = DenseDesign::from_rows(&[
        [1.0, -1.0],
        [1.0, 0.0],
        [1.0, 0.5],
        [1.0, 1.0],
        [1.0, 1.5],
        [1.0, 2.0],
        [1.0, 3.0],
    ]);
    let mu = ParameterBlock::<Mu, _, _>::linear(x, RidgePenalty::new_unchecked(0.2), 0);
    let model = Gamlss::try_new_weighted(
        FixedSigmaNormal,
        ParameterBlocks::from_assigned((mu,)),
        &y,
        &weights,
    )
    .unwrap();
    let beta = [0.4, -0.3];
    let mut expected_gradient = [0.0; 2];
    let expected_value = model
        .try_value_gradient_into(&beta, &mut expected_gradient)
        .unwrap();

    for tile_rows in [1, 2, 3, 5, 20] {
        let policy = ScoreTilePolicy::try_max_rows(tile_rows).unwrap();
        let mut workspace = model.gradient_workspace_with_policy(policy);
        for _ in 0..2 {
            let mut gradient = [f64::NAN; 2];
            let value = model
                .try_value_gradient_into_workspace(&beta, &mut gradient, &mut workspace)
                .unwrap();

            assert_relative_eq!(value, expected_value, epsilon = 1.0e-12);
            for (actual, expected) in gradient.iter().zip(expected_gradient) {
                assert_relative_eq!(*actual, expected, epsilon = 1.0e-12);
            }
            assert_eq!(
                workspace.gradient().score_tile_rows(),
                tile_rows.min(y.len())
            );
            assert_eq!(
                workspace.gradient().score_tile_value_count(),
                tile_rows.min(y.len())
            );
        }
    }
}

#[test]
fn dynamic_executor_is_additive_across_score_tile_sizes() {
    let family = RuntimeLayoutMock {
        key: 41,
        reverse: false,
        whole_paths: false,
        initial_count: 2,
    };
    let y = [1.0, -0.5, 2.0, 0.25, 3.0, -1.0, 1.5];
    let weights = [0.5, 0.0, 2.0, 1.5, 0.25, 3.0, 0.75];
    let model = Gamlss::try_new_weighted(
        family,
        runtime_layout_blocks(&family, y.len()),
        &y,
        &weights,
    )
    .unwrap();
    let beta = [0.4, -0.3];
    let mut expected_gradient = [0.0; 2];
    let expected_value = model
        .try_value_gradient_into(&beta, &mut expected_gradient)
        .unwrap();

    for tile_rows in [1, 2, 3, 5, 20] {
        let policy = ScoreTilePolicy::try_max_rows(tile_rows).unwrap();
        let mut workspace = model.gradient_workspace_with_policy(policy);
        let mut gradient = [f64::NAN; 2];
        let value = model
            .try_value_gradient_into_workspace(&beta, &mut gradient, &mut workspace)
            .unwrap();

        assert_relative_eq!(value, expected_value, epsilon = 1.0e-12);
        for (actual, expected) in gradient.iter().zip(expected_gradient) {
            assert_relative_eq!(*actual, expected, epsilon = 1.0e-12);
        }
        assert_eq!(
            workspace.gradient().score_tile_rows(),
            tile_rows.min(y.len())
        );
        assert_eq!(
            workspace.gradient().score_tile_value_count(),
            2 * tile_rows.min(y.len())
        );
    }
}

#[test]
fn dynamic_workspace_value_and_pointwise_paths_stay_flat() {
    let family = RuntimeLayoutMock {
        key: usize::MAX,
        reverse: false,
        whole_paths: false,
        initial_count: 2,
    };
    let y = [0.0, 1.0];
    let weights = [2.0, 0.5];
    let model = Gamlss::try_new_weighted(
        family,
        runtime_layout_blocks(&family, y.len()),
        &y,
        &weights,
    )
    .unwrap();
    let beta = [0.5, -0.25];
    let expected_raw = [0.15625, 0.90625];
    let expected_value = 2.0_f64.mul_add(expected_raw[0], 0.5 * expected_raw[1]);
    let mut workspace = model.gradient_workspace();

    assert_relative_eq!(
        model
            .try_value_into_workspace(&beta, &mut workspace)
            .unwrap(),
        expected_value
    );
    assert_relative_eq!(
        model
            .try_likelihood_value_into_workspace(&beta, &mut workspace)
            .unwrap(),
        expected_value
    );

    let mut raw = [f64::NAN; 2];
    model.try_pointwise_nll_into(&beta, &mut raw).unwrap();
    for (actual, expected) in raw.iter().zip(expected_raw) {
        assert_relative_eq!(*actual, expected);
    }

    let mut weighted = [f64::NAN; 2];
    model
        .try_weighted_pointwise_log_likelihood_into_workspace(&beta, &mut weighted, &mut workspace)
        .unwrap();
    for (actual, expected) in weighted
        .iter()
        .zip([-2.0 * expected_raw[0], -0.5 * expected_raw[1]])
    {
        assert_relative_eq!(*actual, expected);
    }

    let mut objective = model.into_workspace_objective();
    assert_relative_eq!(objective.value(&beta).unwrap(), expected_value);
    assert_relative_eq!(objective.value(&beta).unwrap(), expected_value);
}

#[test]
fn value_gradient_matches_separate_value_and_gradient() {
    let y = vec![1.0, 2.0];
    let weights = vec![0.5, 2.0];
    let x = DenseDesign::from_rows(&[[1.0, 0.0], [1.0, 1.0]]);
    let mu = ParameterBlock::<Mu, _, _>::linear(x, RidgePenalty::new_unchecked(0.25), 0);
    let model = Gamlss::try_new_weighted(
        FixedSigmaNormal,
        ParameterBlocks::from_assigned((mu,)),
        &y,
        &weights,
    )
    .unwrap();
    let beta = vec![0.75, 0.5];
    let mut separate_grad = vec![0.0; beta.len()];
    let mut fused_grad = vec![f64::NAN; beta.len()];

    let separate_value = model.try_value(&beta).unwrap();
    model.try_gradient_into(&beta, &mut separate_grad).unwrap();
    let fused_value = model
        .try_value_gradient_into(&beta, &mut fused_grad)
        .unwrap();

    assert_relative_eq!(fused_value, separate_value);
    for (actual, expected) in fused_grad.iter().zip(&separate_grad) {
        assert_relative_eq!(actual, expected);
    }
}

#[test]
fn likelihood_and_pointwise_apis_exclude_penalties_and_preserve_weights() {
    let y = vec![1.0, 2.0];
    let weights = vec![2.0, 0.5];
    let x = DenseDesign::from_rows(&[[1.0, 0.0], [1.0, 1.0]]);
    let mu = ParameterBlock::<Mu, _, _>::linear(x, RidgePenalty::new_unchecked(0.25), 0);
    let model = Gamlss::try_new_weighted(
        FixedSigmaNormal,
        ParameterBlocks::from_assigned((mu,)),
        &y,
        &weights,
    )
    .unwrap()
    .with_objective_scale(ObjectiveScale::Mean);
    let beta = vec![0.5, 0.25];
    let mut raw = vec![0.0; y.len()];
    let mut weighted = vec![0.0; y.len()];
    let mut log_likelihood = vec![0.0; y.len()];
    let mut likelihood_gradient = vec![0.0; beta.len()];

    model.try_pointwise_nll_into(&beta, &mut raw).unwrap();
    model
        .try_weighted_pointwise_nll_into(&beta, &mut weighted)
        .unwrap();
    model
        .try_pointwise_log_likelihood_into(&beta, &mut log_likelihood)
        .unwrap();
    let likelihood = model
        .try_likelihood_value_gradient_into(&beta, &mut likelihood_gradient)
        .unwrap();

    assert_relative_eq!(raw[0], 0.125);
    assert_relative_eq!(raw[1], 0.78125);
    assert_relative_eq!(weighted[0], weights[0] * raw[0]);
    assert_relative_eq!(weighted[1], weights[1] * raw[1]);
    assert_relative_eq!(log_likelihood[0], -raw[0]);
    assert_relative_eq!(log_likelihood[1], -raw[1]);
    assert_relative_eq!(likelihood, weighted.iter().sum::<f64>());
    assert_relative_eq!(model.try_likelihood_value(&beta).unwrap(), likelihood);
    assert_relative_eq!(likelihood_gradient[0], -1.625);
    assert_relative_eq!(likelihood_gradient[1], -0.625);

    // Sum likelihood is intentionally independent of optimizer mean scaling.
    assert_relative_eq!(likelihood, 0.640_625);
    assert_relative_eq!(
        model.try_value(&beta).unwrap(),
        likelihood / weights.iter().sum::<f64>() + 0.078_125
    );
}

#[test]
fn mean_objective_scales_likelihood_but_not_penalties() {
    let y = vec![0.0, 3.0];
    let x = DenseDesign::intercept(y.len());
    let mu = ParameterBlock::<Mu, _, _>::linear(x, RidgePenalty::new_unchecked(0.5), 0);
    let model = Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y)
        .unwrap()
        .with_objective_scale(ObjectiveScale::Mean)
        .with_global_penalties(GlobalSquarePenalty { lambda: 2.0 });
    let beta = vec![1.0];
    let mut grad = vec![0.0];

    let value = model.clone().value(&beta).unwrap();
    let fused_value = model.clone().value_gradient(&beta, &mut grad).unwrap();

    assert_relative_eq!(value, 3.75);
    assert_relative_eq!(fused_value, value);
    assert_relative_eq!(grad[0], 4.5);
}

#[test]
fn value_gradient_rejects_invalid_lengths() {
    let y = vec![1.0];
    let x = DenseDesign::intercept(y.len());
    let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
    let model =
        Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
    let mut grad = vec![0.0];

    assert_eq!(
        model.try_value_gradient_into(&[], &mut grad).unwrap_err(),
        ModelError::BetaLength {
            expected: 1,
            actual: 0,
        }
    );

    assert_eq!(
        model.try_value_gradient_into(&[0.0], &mut []).unwrap_err(),
        ModelError::GradientLength {
            expected: 1,
            actual: 0,
        }
    );
}

#[derive(Debug, Clone, Copy)]
struct SoftplusIntercept {
    nrows: usize,
}

impl PredictorBlock for SoftplusIntercept {
    fn nrows(&self) -> usize {
        self.nrows
    }

    fn nparams(&self) -> usize {
        1
    }

    fn eta_row(&self, _: usize, beta: &[f64]) -> f64 {
        softplus(beta[0])
    }

    #[allow(clippy::suboptimal_flops)]
    fn add_gradient_range(&self, _: Range<usize>, scores: &[f64], beta: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(grad.len(), 1);
        grad[0] += scores.iter().sum::<f64>() * sigmoid(beta[0]);
    }
}

#[test]
fn sum_block_supports_user_defined_nonlinear_predictors() {
    let y = vec![1.0, 2.0];
    let linear = crate::LinearPredictorBlock::new(DenseDesign::intercept(y.len()));
    let nonlinear = SoftplusIntercept { nrows: y.len() };
    let predictor = SumBlock::new((linear, nonlinear));
    let mu = ParameterBlock::<Mu, _, _>::new(predictor, NoPenalty, 0);
    let mut model =
        Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
    let beta = vec![0.4, -0.2];
    let eps = 1.0e-6;
    let mut grad = vec![0.0; beta.len()];

    model.gradient(&beta, &mut grad).unwrap();

    for index in 0..beta.len() {
        let mut plus = beta.clone();
        plus[index] += eps;
        let mut minus = beta.clone();
        minus[index] -= eps;
        let finite_difference =
            (model.value(&plus).unwrap() - model.value(&minus).unwrap()) / (2.0 * eps);

        assert_relative_eq!(grad[index], finite_difference, epsilon = 1.0e-6);
    }
}

#[derive(Debug, Clone, Copy)]
struct StatefulLocation {
    target_shift: f64,
}

crate::impl_scalar_compilable_family!(
    impl for StatefulLocation;
    parameters = (Mu,);
    arity = 1;
);

impl Family for StatefulLocation {
    type Eta = f64;
    type Theta = f64;
    type GradientEta = f64;
    type Observation<'obs> = f64;
    type Workspace = ();
    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        *eta + self.target_shift
    }

    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        let residual = y - theta;
        0.5 * residual * residual
    }

    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        let theta = self.theta(eta, _workspace);
        (self.nll(y, &theta, _workspace), theta - y)
    }
}

impl InitialEtaFromObservations<1> for StatefulLocation {}

#[test]
fn family_instance_state_participates_in_objective() {
    let y = vec![2.0];
    let x = DenseDesign::intercept(y.len());
    let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
    let mut model = Gamlss::try_new(
        StatefulLocation { target_shift: 1.0 },
        ParameterBlocks::from_assigned((mu,)),
        &y,
    )
    .unwrap();
    let beta = vec![0.5];
    let mut grad = vec![0.0];

    assert_relative_eq!(model.value(&beta).unwrap(), 0.125);

    model.gradient(&beta, &mut grad).unwrap();

    assert_relative_eq!(grad[0], -0.5);
}

#[derive(Debug, Clone, Copy)]
struct BivariateLocation;

crate::impl_scalar_compilable_family!(
    impl for BivariateLocation;
    parameters = (Mu,);
    arity = 1;
);

impl Family for BivariateLocation {
    type Eta = f64;
    type Theta = f64;
    type GradientEta = f64;
    type Observation<'obs> = [f64; 2];
    type Workspace = ();
    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        *eta
    }

    fn nll(
        &self,
        observation: Self::Observation<'_>,
        theta: &Self::Theta,
        _workspace: &mut Self::Workspace,
    ) -> f64 {
        let first = theta - observation[0];
        let second = theta - observation[1];
        f64::midpoint(first * first, second * second)
    }

    fn nll_and_gradient_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        let gradient = (*eta - observation[0]) + (*eta - observation[1]);
        (self.nll(observation, eta, _workspace), gradient)
    }
}

impl InitialEtaFromObservations<1> for BivariateLocation {}

#[test]
fn model_accepts_multivariate_observation_rows() {
    let y = vec![[1.0, 3.0], [2.0, 4.0]];
    let x = DenseDesign::intercept(y.len());
    let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
    let mut model = Gamlss::try_new_with_observations(
        BivariateLocation,
        ParameterBlocks::from_assigned((mu,)),
        y.as_slice(),
    )
    .unwrap();
    let beta = vec![2.0];
    let mut grad = vec![0.0];

    assert_relative_eq!(model.value(&beta).unwrap(), 3.0);

    model.gradient(&beta, &mut grad).unwrap();

    assert_relative_eq!(grad[0], -2.0);
}

#[derive(Debug, Clone, Copy)]
struct BorrowedRowMean;

crate::impl_scalar_compilable_family!(
    impl for BorrowedRowMean;
    parameters = (Mu,);
    arity = 1;
);

impl Family for BorrowedRowMean {
    type Eta = f64;
    type Theta = f64;
    type GradientEta = f64;
    type Observation<'obs> = &'obs [f64];
    type Workspace = ();
    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        *eta
    }

    #[allow(clippy::cast_precision_loss)]
    fn nll(
        &self,
        observation: Self::Observation<'_>,
        theta: &Self::Theta,
        _workspace: &mut Self::Workspace,
    ) -> f64 {
        let mean = observation.iter().sum::<f64>() / observation.len() as f64;
        let residual = theta - mean;
        0.5 * residual * residual
    }

    #[allow(clippy::cast_precision_loss)]
    fn nll_and_gradient_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        let mean = observation.iter().sum::<f64>() / observation.len() as f64;
        (self.nll(observation, eta, _workspace), *eta - mean)
    }
}

impl InitialEtaFromObservations<1> for BorrowedRowMean {}

#[test]
fn model_accepts_borrowed_dynamic_observation_rows() {
    let values = [1.0, 3.0, 2.0, 4.0];
    let obs = DenseRows::try_new(&values, 2).unwrap();
    let x = DenseDesign::intercept(obs.len());
    let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
    let mut model = Gamlss::try_new_with_observations(
        BorrowedRowMean,
        ParameterBlocks::from_assigned((mu,)),
        obs,
    )
    .unwrap();
    let beta = vec![2.0];
    let mut grad = vec![0.0];

    assert_relative_eq!(model.value(&beta).unwrap(), 0.5);

    model.gradient(&beta, &mut grad).unwrap();

    assert_relative_eq!(grad[0], -1.0);
}

#[test]
fn dense_rows_validate_geometry_and_weights_once() {
    let values = [1.0, 3.0, 2.0, 4.0];
    let weights = [2.0, 0.5];
    let rows = DenseRows::try_new_weighted(&values, 2, &weights).unwrap();

    assert_eq!(rows.values(), values);
    assert_eq!(rows.width(), 2);
    assert_eq!(rows.nrows(), 2);
    assert_eq!(rows.weights(), Some(weights.as_slice()));
    assert_eq!(rows.observation_at(1), &[2.0, 4.0]);
    assert_relative_eq!(rows.weight_at(1), 0.5);

    assert_eq!(
        DenseRows::try_new(&values, 0).unwrap_err(),
        ModelError::InvalidParameter {
            parameter: "dense observation row width",
            expected: "positive",
        }
    );
    assert_eq!(
        DenseRows::try_new(&values[..3], 2).unwrap_err(),
        ModelError::DenseObservationSize {
            actual_values: 3,
            row_width: 2,
        }
    );
    assert_eq!(
        DenseRows::try_new_weighted(&values, 2, &[1.0]).unwrap_err(),
        ModelError::WeightLength {
            expected: 2,
            actual: 1,
        }
    );
    assert_eq!(
        DenseRows::try_new_weighted(&values, 2, &[1.0, f64::NAN]).unwrap_err(),
        ModelError::InvalidWeight { index: 1 }
    );
}

#[derive(Debug, Clone, Copy)]
struct ThreeParameterMock;

crate::impl_scalar_compilable_family!(
    impl for ThreeParameterMock;
    parameters = (Mu, Sigma, Nu);
    arity = 3;
);

impl Family for ThreeParameterMock {
    type Eta = (f64, f64, f64);
    type Theta = (f64, f64, f64);
    type GradientEta = (f64, f64, f64);
    type Observation<'obs> = f64;
    type Workspace = ();
    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        *eta
    }

    #[allow(clippy::suboptimal_flops)]
    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        let first = theta.0 - y;
        let second = theta.1 - 1.0;
        let third = theta.2 + 1.0;
        0.5 * (first * first + second * second + third * third)
    }

    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        let gradient = (eta.0 - y, eta.1 - 1.0, eta.2 + 1.0);
        (self.nll(y, eta, _workspace), gradient)
    }
}

impl InitialEtaFromObservations<3> for ThreeParameterMock {}

#[test]
fn custom_three_parameter_family_uses_generic_blocks() {
    let y = vec![2.0];
    let first = ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 0);
    let second =
        ParameterBlock::<Sigma, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 1);
    let third = ParameterBlock::<Nu, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 2);
    let mut model = Gamlss::try_new(
        ThreeParameterMock,
        ParameterBlocks::from_assigned((first, second, third)),
        &y,
    )
    .unwrap();
    let beta = vec![1.5, 0.5, -0.5];
    let mut grad = vec![0.0; 3];

    assert_relative_eq!(model.value(&beta).unwrap(), 0.375);

    model.gradient(&beta, &mut grad).unwrap();

    assert_relative_eq!(grad[0], -0.5);
    assert_relative_eq!(grad[1], -0.5);
    assert_relative_eq!(grad[2], 0.5);
}

#[derive(Debug, Clone, Copy)]
struct FourParameterMock;

crate::impl_scalar_compilable_family!(
    impl for FourParameterMock;
    parameters = (Mu, Sigma, Nu, Tau);
    arity = 4;
);

impl Family for FourParameterMock {
    type Eta = (f64, f64, f64, f64);
    type Theta = (f64, f64, f64, f64);
    type GradientEta = (f64, f64, f64, f64);
    type Observation<'obs> = f64;
    type Workspace = ();
    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        (eta.0 + 1.0, eta.1 + 2.0, eta.2 + 3.0, eta.3 + 4.0)
    }

    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        theta.0 + theta.1 + theta.2 + theta.3 + y
    }

    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        (
            {
                let theta = self.theta(eta, _workspace);
                self.nll(y, &theta, _workspace)
            },
            (1.0, 1.0, 1.0, 1.0),
        )
    }
}

impl InitialEtaFromObservations<4> for FourParameterMock {}

#[derive(Debug, Clone, Copy)]
struct Fifth;

impl ParameterName for Fifth {
    const NAME: &'static str = "fifth";
}

#[derive(Debug, Clone, Copy)]
struct FiveParameterMock;

crate::impl_scalar_compilable_family!(
    impl for FiveParameterMock;
    parameters = (Mu, Sigma, Nu, Tau, Fifth);
    arity = 5;
);

impl Family for FiveParameterMock {
    type Eta = (f64, f64, f64, f64, f64);
    type Theta = (f64, f64, f64, f64, f64);
    type GradientEta = (f64, f64, f64, f64, f64);
    type Observation<'obs> = f64;
    type Workspace = ();
    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        *eta
    }

    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        let targets = [y, 1.0, 2.0, 3.0, 4.0];
        let values: [f64; 5] = (*theta).into();
        0.5 * values
            .iter()
            .zip(targets)
            .map(|(value, target)| {
                let residual = value - target;
                residual * residual
            })
            .sum::<f64>()
    }

    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        (
            self.nll(y, eta, _workspace),
            (
                eta.0 - y,
                eta.1 - 1.0,
                eta.2 - 2.0,
                eta.3 - 3.0,
                eta.4 - 4.0,
            ),
        )
    }
}

impl InitialEtaFromObservations<5> for FiveParameterMock {}

#[test]
fn prediction_api_returns_eta_and_theta_for_four_parameter_model() {
    let y = vec![2.0];
    let first = ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 0);
    let second =
        ParameterBlock::<Sigma, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 1);
    let third = ParameterBlock::<Nu, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 2);
    let fourth = ParameterBlock::<Tau, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 3);
    let model = Gamlss::try_new(
        FourParameterMock,
        ParameterBlocks::from_assigned((first, second, third, fourth)),
        &y,
    )
    .unwrap();
    let beta = vec![0.5, 1.5, 2.5, 3.5];

    assert_eq!(
        model.predict_eta_row(&beta, 0).unwrap(),
        (0.5, 1.5, 2.5, 3.5)
    );
    assert_eq!(
        model.predict_theta_row(&beta, 0).unwrap(),
        (1.5, 3.5, 5.5, 7.5)
    );
}

#[test]
fn custom_five_parameter_family_uses_generic_blocks() {
    let y = vec![2.0];
    let blocks = ParameterBlocks::new((
        intercept_block::<Mu>(y.len()),
        intercept_block::<Sigma>(y.len()),
        intercept_block::<Nu>(y.len()),
        intercept_block::<Tau>(y.len()),
        intercept_block::<Fifth>(y.len()),
    ));
    let mut model = Gamlss::try_new(FiveParameterMock, blocks, &y).unwrap();
    let beta = vec![1.5, 0.5, 1.5, 2.5, 3.5];
    let mut grad = vec![0.0; 5];

    assert_relative_eq!(model.value(&beta).unwrap(), 0.625);

    model.gradient(&beta, &mut grad).unwrap();

    assert_eq!(
        model.parameter_layout().unique_slice("fifth").unwrap(),
        Some(4..5)
    );
    assert_relative_eq!(grad[0], -0.5);
    assert_relative_eq!(grad[1], -0.5);
    assert_relative_eq!(grad[2], -0.5);
    assert_relative_eq!(grad[3], -0.5);
    assert_relative_eq!(grad[4], -0.5);
}

fn intercept_block<P>(
    nrows: usize,
) -> ParameterBlock<P, LinearPredictorBlock<DenseDesign>, NoPenalty> {
    ParameterBlock::linear(DenseDesign::intercept(nrows), NoPenalty, 99)
}

#[test]
fn parameter_layout_and_unpack_use_distribution_parameter_names() {
    let y = vec![2.0];
    let first = ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 0);
    let second =
        ParameterBlock::<Sigma, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 1);
    let third = ParameterBlock::<Nu, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 2);
    let model = Gamlss::try_new(
        ThreeParameterMock,
        ParameterBlocks::from_assigned((first, second, third)),
        &y,
    )
    .unwrap();
    let parameters = vec![1.5, 0.5, -0.5];
    let layout = model.parameter_layout();
    let unpacked = model.unpack_parameters(&parameters).unwrap();

    assert_eq!(layout.len(), 3);
    assert!(!layout.is_empty());
    assert_eq!(layout.ncoefficients(), parameters.len());
    assert_eq!(layout.unique_slice("mu").unwrap(), Some(0..1));
    assert_eq!(layout.unique_slice_of::<Mu>().unwrap(), Some(0..1));
    assert_eq!(layout.unique_slice("sigma").unwrap(), Some(1..2));
    assert_eq!(layout.unique_slice_of::<Sigma>().unwrap(), Some(1..2));
    assert_eq!(layout.unique_slice("nu").unwrap(), Some(2..3));
    assert_eq!(layout.unique_slice_of::<Nu>().unwrap(), Some(2..3));
    assert_eq!(unpacked.unique_coefficients("mu").unwrap().unwrap(), &[1.5]);
    assert_eq!(
        unpacked.unique_coefficients_of::<Mu>().unwrap().unwrap(),
        &[1.5]
    );
    assert_eq!(
        unpacked.unique_block_of::<Mu>().unwrap().unwrap().name(),
        "mu"
    );
    assert_eq!(unpacked.block_at(0).unwrap().descriptor_index, 0);
    assert_eq!(
        unpacked.block_at(0).unwrap().descriptor,
        ParameterDescriptor::whole("mu", 0..1)
    );
    assert_eq!(
        unpacked
            .block_for_descriptor(&ParameterDescriptor::whole("sigma", 1..2))
            .unwrap()
            .coefficients,
        vec![0.5]
    );
    assert_eq!(
        unpacked.unique_coefficients("sigma").unwrap().unwrap(),
        &[0.5]
    );
    assert_eq!(
        unpacked.unique_coefficients("nu").unwrap().unwrap(),
        &[-0.5]
    );
}

#[test]
fn visitor_apis_match_allocating_layout_helpers() {
    let y = vec![2.0];
    let first = ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 0);
    let second =
        ParameterBlock::<Sigma, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 1);
    let third = ParameterBlock::<Nu, _, _>::linear(DenseDesign::intercept(y.len()), NoPenalty, 2);
    let model = Gamlss::try_new(
        ThreeParameterMock,
        ParameterBlocks::from_assigned((first, second, third)),
        &y,
    )
    .unwrap();

    let mut visited_ranges = Vec::new();
    model.visit_block_ranges(|index, range| visited_ranges.push((index, range)));
    assert_eq!(
        visited_ranges,
        model
            .block_ranges()
            .into_iter()
            .enumerate()
            .collect::<Vec<_>>()
    );

    let mut visited_slices = Vec::new();
    model.visit_parameter_slices(|index, name, range| {
        visited_slices.push((index, name, range));
    });
    let layout = model.parameter_layout();
    assert_eq!(
        visited_slices,
        layout
            .slices()
            .iter()
            .enumerate()
            .map(|(index, slice)| (index, slice.name, slice.range.clone()))
            .collect::<Vec<_>>()
    );

    let mut layout_visited = Vec::new();
    layout.visit_slices(|index, name, range| layout_visited.push((index, name, range)));
    assert_eq!(layout_visited, visited_slices);

    let mut visited_descriptors = Vec::new();
    model.visit_parameter_descriptors(|index, descriptor| {
        visited_descriptors.push((index, descriptor));
    });
    assert_eq!(
        visited_descriptors,
        model
            .parameter_descriptors()
            .into_iter()
            .enumerate()
            .collect::<Vec<_>>()
    );
}

#[test]
fn training_diagnostics_report_train_nll_penalty_and_gradient_norm() {
    let y = vec![1.0, 2.0];
    let x = DenseDesign::intercept(y.len());
    let mu = ParameterBlock::<Mu, _, _>::linear(x, RidgePenalty::new_unchecked(0.5), 0);
    let model =
        Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
    let parameters = vec![1.5];
    let mut grad = vec![f64::NAN; parameters.len()];
    let diagnostics = model.training_diagnostics(&parameters).unwrap();
    let diagnostics_into = model
        .training_diagnostics_into(&parameters, &mut grad)
        .unwrap();

    assert_relative_eq!(diagnostics.train_nll, 0.25);
    assert_relative_eq!(diagnostics.penalty, 1.125);
    assert_relative_eq!(diagnostics.objective, 1.375);
    assert_relative_eq!(diagnostics.gradient_norm, 1.5);
    assert_eq!(diagnostics.nonfinite_gradient_count, 0);
    assert_eq!(diagnostics_into, diagnostics);
    assert_relative_eq!(grad[0], 1.5);

    let mut workspace = model.gradient_workspace();
    grad.fill(f64::NAN);
    assert_eq!(
        model
            .training_diagnostics_into_workspace(&parameters, &mut grad, &mut workspace)
            .unwrap(),
        diagnostics
    );
    assert_relative_eq!(grad[0], 1.5);

    let mut workspace_model = model.into_workspace_objective();
    grad.fill(f64::NAN);
    assert_eq!(
        workspace_model
            .training_diagnostics_into(&parameters, &mut grad)
            .unwrap(),
        diagnostics
    );
    assert_relative_eq!(grad[0], 1.5);
}

#[derive(Debug, Clone, Copy)]
struct DifferenceGlobalPenalty {
    lambda: f64,
}

impl GlobalPenalty for DifferenceGlobalPenalty {
    fn value(&self, beta: &[f64]) -> f64 {
        let diff = beta[0] - beta[1];
        self.lambda * diff * diff
    }

    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        let diff = beta[0] - beta[1];
        let slope = 2.0 * self.lambda * diff;
        grad[0] += slope;
        grad[1] -= slope;
    }
}

#[derive(Debug, Clone, Copy)]
struct GlobalSquarePenalty {
    lambda: f64,
}

impl GlobalPenalty for GlobalSquarePenalty {
    fn value(&self, beta: &[f64]) -> f64 {
        self.lambda * beta[0] * beta[0]
    }

    #[allow(clippy::suboptimal_flops)]
    fn add_gradient(&self, beta: &[f64], grad: &mut [f64]) {
        grad[0] += 2.0 * self.lambda * beta[0];
    }
}

#[test]
fn global_penalty_adds_value_and_gradient_to_full_objective() {
    let y = vec![0.0, 0.0];
    let x = DenseDesign::from_rows(&[[1.0, 0.0], [0.0, 1.0]]);
    let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
    let mut model = Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y)
        .unwrap()
        .with_global_penalties(DifferenceGlobalPenalty { lambda: 1.0 });
    let beta = vec![1.0, -1.0];
    let mut grad = vec![0.0; beta.len()];

    assert_relative_eq!(model.value(&beta).unwrap(), 5.0);

    assert_relative_eq!(model.value_gradient(&beta, &mut grad).unwrap(), 5.0);

    assert_relative_eq!(grad[0], 5.0);
    assert_relative_eq!(grad[1], -5.0);
}

#[test]
fn try_with_global_penalties_validates_full_parameter_dimension() {
    let y = vec![0.0, 0.0];
    let x = DenseDesign::from_rows(&[[1.0, 0.0], [0.0, 1.0]]);
    let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
    let model =
        Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
    let penalty =
        HingeQuadraticPenalty::try_new(LinearFormBuilder::new().term(2, 1.0).build(), 1.0).unwrap();

    assert_eq!(
        model.try_with_global_penalties(penalty).unwrap_err(),
        ModelError::PenaltyIndexOutOfBounds { index: 2, dim: 2 }
    );
}

#[test]
fn workspace_try_with_global_penalties_validates_full_parameter_dimension() {
    let y = vec![0.0, 0.0];
    let x = DenseDesign::from_rows(&[[1.0, 0.0], [0.0, 1.0]]);
    let mu = ParameterBlock::<Mu, _, _>::linear(x, NoPenalty, 0);
    let model = Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y)
        .unwrap()
        .into_workspace_objective();
    let penalty =
        HingeQuadraticPenalty::try_new(LinearFormBuilder::new().term(2, 1.0).build(), 1.0).unwrap();

    assert_eq!(
        model.try_with_global_penalties(penalty).unwrap_err(),
        ModelError::PenaltyIndexOutOfBounds { index: 2, dim: 2 }
    );
}

#[test]
fn block_objective_for_projects_mu_coefficients() {
    // Simple 2-param mock: identity link for both, NLL = 0.5 * sum of squares.
    #[derive(Debug, Clone, Copy)]
    struct TwoParamMock;

    crate::impl_scalar_compilable_family!(
        impl for TwoParamMock;
        parameters = (Mu, Sigma);
        arity = 2;
    );

    impl Family for TwoParamMock {
        type Eta = (f64, f64);
        type Theta = (f64, f64);
        type GradientEta = (f64, f64);
        type Observation<'obs> = f64;
        type Workspace = ();
        #[inline]
        fn workspace(&self) -> Self::Workspace {}

        fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
            *eta
        }

        fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
            let first = theta.0 - y;
            let second = theta.1 - 1.0;
            f64::midpoint(first * first, second * second)
        }

        fn nll_and_gradient_eta(
            &self,
            y: f64,
            eta: &Self::Eta,
            _workspace: &mut Self::Workspace,
        ) -> (f64, Self::GradientEta) {
            let gradient = (eta.0 - y, eta.1 - 1.0);
            (self.nll(y, eta, _workspace), gradient)
        }
    }

    impl InitialEtaFromObservations<2> for TwoParamMock {}

    let y = vec![1.0, 2.0, 3.0];
    let x_mu = DenseDesign::from_rows(&[[1.0, 0.5], [1.0, 1.5], [1.0, 2.5]]);
    let x_sigma = DenseDesign::intercept(y.len());
    let mu = ParameterBlock::<Mu, _, _>::linear(x_mu, NoPenalty, 0);
    let sigma = ParameterBlock::<Sigma, _, _>::linear(x_sigma, NoPenalty, 0);
    let (mu, sigma) = ParameterBlocks::new((mu, sigma)).into_inner();
    let mut model = Gamlss::try_new(
        TwoParamMock,
        ParameterBlocks::from_assigned((mu, sigma)),
        &y,
    )
    .unwrap();

    let beta = vec![0.5, 0.2, 0.3];
    let nparams = model.nparams();

    // Scope the block objective to release the mutable borrow on model.
    let (mu_dim, mu_value, mu_grad) = {
        let mut mu_block = model.block_objective_for::<Mu>(beta.clone()).unwrap();
        let dim = mu_block.dim();
        let mut block_grad = vec![0.0; dim];
        let value = mu_block
            .value_gradient(&beta[..2], &mut block_grad)
            .unwrap();
        (dim, value, block_grad)
    };

    assert_eq!(mu_dim, 2);
    assert_relative_eq!(mu_value, model.value(&beta).unwrap());

    let mut full_grad = vec![0.0; nparams];
    model.gradient(&beta, &mut full_grad).unwrap();

    assert_relative_eq!(mu_grad[0], full_grad[0]);
    assert_relative_eq!(mu_grad[1], full_grad[1]);
}

#[test]
fn block_objective_for_rejects_wrong_full_beta_length() {
    let y = vec![1.0, 2.0];
    let mu = ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(2), NoPenalty, 0);
    let mut model =
        Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();

    assert_eq!(
        model.block_objective_for::<Mu>(Vec::new()).unwrap_err(),
        ModelError::BetaLength {
            expected: 1,
            actual: 0,
        }
    );
}

#[test]
fn workspace_block_objective_for_rejects_wrong_full_beta_length() {
    let y = vec![1.0, 2.0];
    let mu = ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(2), NoPenalty, 0);
    let model =
        Gamlss::try_new(FixedSigmaNormal, ParameterBlocks::from_assigned((mu,)), &y).unwrap();
    let mut objective = model.into_workspace_objective();

    assert_eq!(
        objective.block_objective_for::<Mu>(Vec::new()).unwrap_err(),
        ModelError::BetaLength {
            expected: 1,
            actual: 0,
        }
    );
}

fn softplus(value: f64) -> f64 {
    if value > 30.0 {
        value
    } else if value < -30.0 {
        value.exp()
    } else {
        value.exp().ln_1p()
    }
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

#[test]
fn shared_prediction_paths_preserve_validation_and_workspace_callback_order() {
    use std::{cell::RefCell, rc::Rc};

    type Calls = Rc<RefCell<Vec<&'static str>>>;
    struct TracingFamily(Calls);
    struct TracingObservations<'a>(&'a [f64], Calls);

    impl Family for TracingFamily {
        type Eta = f64;
        type Theta = f64;
        type GradientEta = f64;
        type Observation<'obs> = f64;
        type Workspace = ();

        fn workspace(&self) {
            self.0.borrow_mut().push("workspace");
        }

        fn theta(&self, eta: &f64, _workspace: &mut ()) -> f64 {
            self.0.borrow_mut().push("theta");
            *eta
        }

        fn nll(&self, y: f64, theta: &f64, _workspace: &mut ()) -> f64 {
            0.5 * (theta - y).powi(2)
        }

        fn nll_and_gradient_eta(&self, y: f64, eta: &f64, workspace: &mut ()) -> (f64, f64) {
            (self.nll(y, eta, workspace), eta - y)
        }
    }

    impl InitialEtaFromObservations<1> for TracingFamily {}
    crate::impl_scalar_compilable_family!(impl for TracingFamily; parameters = (Mu,); arity = 1;);

    impl<'row> ObservationView<'row> for TracingObservations<'_> {
        type Observation = f64;

        fn len(&self) -> usize {
            self.1.borrow_mut().push("len");
            self.0.len()
        }

        fn observation_at(&'row self, row: usize) -> f64 {
            self.0[row]
        }

        fn weight_at(&self, _: usize) -> f64 {
            1.0
        }
    }

    let calls = Calls::default();
    let y = [0.0, 1.0];
    let blocks = ParameterBlocks::new((ParameterBlock::<Mu, _, _>::linear(
        DenseDesign::intercept(y.len()),
        NoPenalty,
        0,
    ),));
    let model = Gamlss::try_new_with_observations(
        TracingFamily(calls.clone()),
        blocks,
        TracingObservations(&y, calls.clone()),
    )
    .unwrap();
    calls.borrow_mut().clear();

    assert!(matches!(
        model.predict_theta(&[]),
        Err(ModelError::BetaLength { .. })
    ));
    assert!(calls.borrow().is_empty());

    model.predict_theta(&[0.5]).unwrap();
    assert_eq!(*calls.borrow(), ["workspace", "len", "theta", "theta"]);
    calls.borrow_mut().clear();

    model
        .for_each_theta(&[0.5], |_, _| calls.borrow_mut().push("visit"))
        .unwrap();
    assert_eq!(
        *calls.borrow(),
        ["workspace", "len", "theta", "visit", "theta", "visit"]
    );
    calls.borrow_mut().clear();

    assert!(matches!(
        model.predict_theta_into(&[0.5], &mut []),
        Err(ModelError::ResponseLength { .. })
    ));
    assert_eq!(*calls.borrow(), ["len"]);
    calls.borrow_mut().clear();

    assert!(matches!(
        model.predict_theta_row(&[0.5], 2),
        Err(ModelError::RowOutOfBounds { .. })
    ));
    assert_eq!(*calls.borrow(), ["len"]);
}
