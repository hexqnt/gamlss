#![allow(
    clippy::float_cmp,
    clippy::suboptimal_flops,
    clippy::cast_precision_loss
)]
use gamlss_core::{
    AboveTwoLink, CholeskyScale, ClampedLog, ComponentMean, Cv, DegreesOfFreedom, DenseDesign,
    DenseInformation, DenseRows, Dispersion, DynamicLayoutKey, Family, FiniteScalarObservations,
    FixedDimensionalFamily, FloorSoftplusScalar, Gamlss, HasCdf, HasConditionalCdf, HasDensity,
    HasDeviance, HasDiagonalFisherInfo, HasExpectedInformation, HasInitialEta, HasLogDensity,
    HasMarginalCdf, HasObservationDimension, HasRosenblattTransform, IdiosyncraticRate,
    InitialEtaFromObservations, KernelSigma, LinearForm, LinearFormBuilder, LocationCholesky, Log,
    LogLocation, LogPlus, LogSd, Logit, LowerTriangularParameterBlock, Mean, Median, Mu,
    NegativeSoftplusScalar, NoPenalty, Nu, Objective, ObjectiveScale, ObservationView,
    OneProbability, ParameterAxis, ParameterBlock, ParameterBlocks, ParameterDescriptor,
    ParameterLayout, ParameterName, ParameterParts, ParameterPath, ParameterSlice, PositiveLink,
    Power, PredictorBlock, Probability, ScoreTilePolicy, ShapeValues, Sigma, Size, Softplus,
    SoftplusScalar, TotalMean, TrainingDiagnostics, UnitIntervalLink, VectorParameterBlock,
    ZeroProbability,
};

#[derive(Debug, Clone, Copy)]
struct DependentConstraintFamily;

impl DependentConstraintFamily {
    fn target(observation: &[f64]) -> f64 {
        observation.iter().sum::<f64>() / observation.len() as f64
    }
}

#[test]
fn score_tile_policy_remains_a_root_and_prelude_reexport() {
    let root_policy = ScoreTilePolicy::try_max_bytes(8 * 1024 * 1024).unwrap();
    let prelude_policy = gamlss_core::prelude::ScoreTilePolicy::try_max_rows(128).unwrap();

    assert_eq!(root_policy.max_bytes(), Some(8 * 1024 * 1024));
    assert_eq!(prelude_policy.max_rows(), Some(128));
    assert_eq!(ScoreTilePolicy::DEFAULT_BYTE_BUDGET, 4 * 1024 * 1024);
}

gamlss_core::impl_scalar_compilable_family!(
    impl for DependentConstraintFamily;
    parameters = (Mu, Nu);
    arity = 2;
);

impl Family for DependentConstraintFamily {
    type Eta = (f64, f64);
    type Theta = (f64, f64);
    type GradientEta = (f64, f64);
    type Observation<'obs> = &'obs [f64];
    type Workspace = ();
    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        let constrained = eta.1.tanh();
        (eta.0 + constrained, constrained)
    }

    fn nll(
        &self,
        observation: Self::Observation<'_>,
        theta: &Self::Theta,
        _workspace: &mut Self::Workspace,
    ) -> f64 {
        let residual = theta.0 - Self::target(observation);
        0.5 * residual * residual
    }

    fn nll_and_gradient_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        let theta = self.theta(eta, workspace);
        let residual = theta.0 - Self::target(observation);
        let d_constrained = 1.0 - theta.1 * theta.1;
        (
            self.nll(observation, &theta, workspace),
            (residual, residual * d_constrained),
        )
    }
}

impl InitialEtaFromObservations<2> for DependentConstraintFamily {}

impl HasDiagonalFisherInfo for DependentConstraintFamily {
    fn nll_gradient_and_diagonal_fisher_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta, Self::GradientEta) {
        let (nll, gradient) = self.nll_and_gradient_eta(observation, eta, workspace);
        let theta = self.theta(eta, workspace);
        let d_constrained = 1.0 - theta.1 * theta.1;
        (nll, gradient, (1.0, d_constrained.powi(2)))
    }
}

impl HasExpectedInformation<2> for DependentConstraintFamily {
    fn nll_gradient_and_expected_information_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta, DenseInformation<2>) {
        let (nll, gradient) = self.nll_and_gradient_eta(observation, eta, workspace);
        let theta = self.theta(eta, workspace);
        let d_constrained = 1.0 - theta.1 * theta.1;
        (
            nll,
            gradient,
            DenseInformation::new([[1.0, d_constrained], [d_constrained, d_constrained.powi(2)]]),
        )
    }
}

impl HasDeviance for DependentConstraintFamily {
    fn deviance(&self, observation: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        2.0 * self.nll(observation, theta, &mut ())
    }
}

impl HasCdf for DependentConstraintFamily {
    fn cdf(&self, observation: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        if Self::target(observation) <= theta.0 {
            1.0
        } else {
            0.0
        }
    }
}

impl HasObservationDimension for DependentConstraintFamily {
    fn observation_dimension(&self) -> usize {
        2
    }
}

impl HasConditionalCdf for DependentConstraintFamily {
    fn conditional_cdf(
        &self,
        component: usize,
        y: f64,
        _preceding: &[f64],
        theta: &Self::Theta,
    ) -> f64 {
        if component < 2 && y <= theta.0 {
            1.0
        } else if component < 2 {
            0.0
        } else {
            f64::NAN
        }
    }
}

impl HasRosenblattTransform for DependentConstraintFamily {
    fn rosenblatt_into(
        &self,
        observation: Self::Observation<'_>,
        theta: &Self::Theta,
        out: &mut [f64],
    ) -> Result<(), gamlss_core::ModelError> {
        if out.len() != 2 {
            return Err(gamlss_core::ModelError::ResponseLength {
                expected: 2,
                actual: out.len(),
            });
        }
        for component in 0..2 {
            out[component] =
                self.conditional_cdf(component, observation[component], observation, theta);
        }
        Ok(())
    }
}

impl FixedDimensionalFamily<2> for DependentConstraintFamily {}

impl HasInitialEta for DependentConstraintFamily {
    fn initial_eta(&self, observation: Self::Observation<'_>) -> Self::Eta {
        (Self::target(observation), 0.0)
    }
}

#[derive(Debug, Clone, Copy)]
struct ScalarCdfFamily;

impl Family for ScalarCdfFamily {
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

    fn nll(
        &self,
        observation: Self::Observation<'_>,
        theta: &Self::Theta,
        _workspace: &mut Self::Workspace,
    ) -> f64 {
        (observation - theta).abs()
    }

    fn nll_and_gradient_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        let nll = self.nll(observation, eta, workspace);
        let gradient = if observation < *eta { 1.0 } else { -1.0 };
        (nll, gradient)
    }
}

impl HasCdf for ScalarCdfFamily {
    fn cdf(&self, observation: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        if observation <= *theta { 1.0 } else { 0.0 }
    }
}

#[test]
fn convenience_predictor_helpers_remain_root_reexports() {
    let softplus = SoftplusScalar::new(1);
    let negative = NegativeSoftplusScalar::new(1);
    let floored = FloorSoftplusScalar::new(1, 0.5);

    assert_eq!(softplus.nparams(), 1);
    assert_eq!(negative.nparams(), 1);
    assert_eq!(floored.nparams(), 1);
}

#[test]
fn shape_aliases_remain_root_reexports() {
    let values: ShapeValues<LocationCholesky<2>> = ([1.0, 2.0], [[3.0, 0.0], [4.0, 5.0]]);

    assert_eq!(values.0, [1.0, 2.0]);
    assert_eq!(values.1[1], [4.0, 5.0]);
}

#[test]
fn training_diagnostics_remains_a_root_reexport() {
    let diagnostics = TrainingDiagnostics {
        objective: 1.0,
        train_nll: 0.75,
        penalty: 0.25,
        gradient_norm: 0.0,
        nonfinite_gradient_count: 0,
    };

    assert_eq!(diagnostics.objective, 1.0);
}

#[test]
fn finite_scalar_observations_remains_a_root_reexport() {
    let values = [1.0, 2.0];
    let obs = FiniteScalarObservations::new(&values).expect("finite values should validate");

    assert_eq!(obs.values(), values);
    assert_eq!(obs.len(), 2);
    assert_eq!(obs.observation_at(1), 2.0);
}

#[test]
fn objective_scale_and_linear_form_builder_remain_root_reexports() {
    let form = LinearForm::builder()
        .weighted_range(2..4, [0.5, -1.0])
        .constant(0.25)
        .build();
    let explicit = LinearFormBuilder::new()
        .term(2, 0.5)
        .term(3, -1.0)
        .constant(0.25)
        .build();

    assert_eq!(ObjectiveScale::default(), ObjectiveScale::Sum);
    assert_eq!(form, explicit);
    assert_eq!(form.value(&[0.0, 0.0, 2.0, 0.5]), 0.75);
}

#[test]
fn parameter_layout_helpers_remain_root_reexports() {
    let dynamic_key = DynamicLayoutKey::new(vec![3, 2]);
    assert_eq!(dynamic_key.parts(), &[3, 2]);
    assert_eq!(dynamic_key.into_parts(), vec![3, 2]);

    let layout = ParameterLayout::new(vec![
        ParameterSlice {
            name: "mu",
            range: 0..2,
        },
        ParameterSlice {
            name: "sigma",
            range: 2..3,
        },
    ]);

    assert_eq!(layout.len(), 2);
    assert!(!layout.is_empty());
    assert_eq!(layout.ncoefficients(), 3);
    assert_eq!(layout.unique_slice_of::<Mu>().unwrap(), Some(0..2));
    assert_eq!(layout.unique_slice_of::<Sigma>().unwrap(), Some(2..3));
    assert_eq!(layout.ranges_of::<Mu>(), vec![0..2]);

    let repeated = ParameterLayout::new(vec![
        ParameterSlice {
            name: "mu",
            range: 0..1,
        },
        ParameterSlice {
            name: "mu",
            range: 1..2,
        },
    ]);
    assert_eq!(repeated.ranges_of::<Mu>(), vec![0..1, 1..2]);
    assert!(matches!(
        repeated.unique_slice_of::<Mu>(),
        Err(gamlss_core::ModelError::AmbiguousParameter { matches: 2, .. })
    ));

    assert_eq!(
        layout.block_descriptors(),
        vec![
            ParameterDescriptor::whole("mu", 0..2),
            ParameterDescriptor::whole("sigma", 2..3),
        ]
    );

    let mut visited = Vec::new();
    layout.visit_block_descriptors(|index, descriptor| visited.push((index, descriptor)));
    assert_eq!(
        visited,
        vec![
            (0, ParameterDescriptor::whole("mu", 0..2)),
            (1, ParameterDescriptor::whole("sigma", 2..3)),
        ]
    );

    assert_eq!(
        ParameterDescriptor::vector_component("mu", 1, 4..6).path,
        ParameterPath::new(vec![ParameterAxis::Vector { component: 1 }])
    );
    assert_eq!(
        ParameterDescriptor::lower_triangular_entry("cholesky", 2, 1, 6..7).path,
        ParameterPath::new(vec![ParameterAxis::Lower { row: 2, col: 1 }])
    );
    assert_eq!(
        ParameterDescriptor::matrix_entry("loading", 3, 2, 7..9).path,
        ParameterPath::new(vec![ParameterAxis::Matrix { row: 3, col: 2 }])
    );
    assert_eq!(
        ParameterDescriptor::strict_lower_triangular_entry("partial_corr", 2, 0, 9..10).path,
        ParameterPath::new(vec![ParameterAxis::StrictLower { row: 2, col: 0 }])
    );
    assert_eq!(
        ParameterDescriptor::simplex_logit("mixture_weight", 1, 10..11).path,
        ParameterPath::new(vec![ParameterAxis::SimplexLogit { class: 1 }])
    );
    assert_eq!(
        ParameterDescriptor::component("mu", 2, 11..12).path,
        ParameterPath::new(vec![ParameterAxis::Component { index: 2 }])
    );
}

#[test]
fn semantic_parameter_markers_remain_root_reexports() {
    assert_eq!(Mean::NAME, "mean");
    assert_eq!(Median::NAME, "median");
    assert_eq!(ComponentMean::NAME, "component_mean");
    assert_eq!(TotalMean::NAME, "total_mean");
    assert_eq!(Cv::NAME, "cv");
    assert_eq!(LogSd::NAME, "log_sd");
    assert_eq!(LogLocation::NAME, "log_location");
    assert_eq!(Dispersion::NAME, "dispersion");
    assert_eq!(DegreesOfFreedom::NAME, "degrees_of_freedom");
    assert_eq!(Size::NAME, "size");
    assert_eq!(Probability::NAME, "probability");
    assert_eq!(ZeroProbability::NAME, "zero_probability");
    assert_eq!(OneProbability::NAME, "one_probability");
    assert_eq!(Power::NAME, "power");
    assert_eq!(CholeskyScale::NAME, "cholesky");
    assert_eq!(KernelSigma::NAME, "kernel_sigma");
    assert_eq!(IdiosyncraticRate::NAME, "idiosyncratic_rate");
}

#[test]
fn structured_parameter_blocks_remain_root_reexports() {
    let vector = VectorParameterBlock::<Mu, 2, _, _>::new(
        [
            gamlss_core::LinearPredictorBlock::new(DenseDesign::intercept(3)),
            gamlss_core::LinearPredictorBlock::new(DenseDesign::intercept(3)),
        ],
        NoPenalty,
        99,
    );
    let lower = LowerTriangularParameterBlock::<CholeskyScale, 2, _, _>::new(
        vec![
            gamlss_core::LinearPredictorBlock::new(DenseDesign::intercept(3)),
            gamlss_core::LinearPredictorBlock::new(DenseDesign::intercept(3)),
            gamlss_core::LinearPredictorBlock::new(DenseDesign::intercept(3)),
        ],
        NoPenalty,
        99,
    );

    let (vector, lower) = ParameterBlocks::new((vector, lower)).into_inner();

    assert_eq!(vector.range(), 0..2);
    assert_eq!(vector.component_range(1), Some(1..2));
    assert_eq!(lower.range(), 2..5);
    assert_eq!(lower.entry_range(1, 0), Some(3..4));
}

#[test]
fn link_domain_marker_traits_remain_root_reexports() {
    fn assert_positive_link<L: PositiveLink<f64>>() {}
    fn assert_above_two_link<L: AboveTwoLink<f64>>() {}
    fn assert_unit_interval_link<L: UnitIntervalLink<f64>>() {}

    assert_positive_link::<Log>();
    assert_positive_link::<Softplus>();
    assert_positive_link::<ClampedLog<-12, 12>>();
    assert_positive_link::<LogPlus<2>>();
    assert_above_two_link::<LogPlus<2>>();
    assert_unit_interval_link::<Logit>();
}

#[test]
fn public_api_supports_default_log_density_helper() {
    let theta = DependentConstraintFamily.theta(&(2.0, 0.0), &mut ());

    assert_eq!(
        DependentConstraintFamily.log_density(&[1.0, 3.0], &theta),
        -DependentConstraintFamily.nll(&[1.0, 3.0], &theta, &mut ())
    );
    assert_eq!(DependentConstraintFamily.density(&[1.0, 3.0], &theta), 1.0);
}

#[test]
fn cdf_contract_accepts_family_observations_and_scalar_marginals() {
    assert_eq!(DependentConstraintFamily.cdf(&[1.0, 3.0], &(2.0, 0.0)), 1.0);
    assert_eq!(DependentConstraintFamily.cdf(&[3.0, 5.0], &(2.0, 0.0)), 0.0);

    assert_eq!(ScalarCdfFamily.marginal_cdf(0, 1.0, &2.0), 1.0);
    assert!(ScalarCdfFamily.marginal_cdf(1, 1.0, &2.0).is_nan());
}

#[test]
fn multivariate_cdf_capability_traits_remain_root_reexports() {
    fn assert_conditional_cdf<F: HasConditionalCdf>() {}
    fn assert_rosenblatt_transform<F: HasRosenblattTransform>() {}
    fn assert_fixed_dimensional<F: FixedDimensionalFamily<2>>() {}

    assert_conditional_cdf::<DependentConstraintFamily>();
    assert_rosenblatt_transform::<DependentConstraintFamily>();
    assert_fixed_dimensional::<DependentConstraintFamily>();
}

#[test]
fn public_api_supports_borrowed_observations_nll_gradient_and_dense_information() {
    let values = [1.0, 3.0, 2.0, 4.0];
    let obs = DenseRows::try_new(&values, 2).unwrap();
    let mu = ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(obs.nrows()), NoPenalty, 0);
    let nu = ParameterBlock::<Nu, _, _>::linear(DenseDesign::intercept(obs.nrows()), NoPenalty, 1);
    let mut model = Gamlss::try_new_with_observations(
        DependentConstraintFamily,
        ParameterBlocks::from_assigned((mu, nu)),
        obs,
    )
    .expect("borrowed observation view should compile into a model");
    let beta = vec![2.0, 0.0];
    let mut grad = vec![0.0; 2];

    let value = model
        .value(&beta)
        .expect("objective value should be finite");
    model
        .gradient(&beta, &mut grad)
        .expect("objective gradient should be finite");

    assert_eq!(value, 0.5);
    assert_eq!(grad, vec![-1.0, -1.0]);

    let diagnostics = model
        .training_diagnostics(&beta)
        .expect("training diagnostics should use the public model API");
    assert_eq!(diagnostics.objective, 0.5);
    assert_eq!(diagnostics.train_nll, 0.5);
    assert_eq!(diagnostics.penalty, 0.0);
    assert_eq!(diagnostics.nonfinite_gradient_count, 0);

    let eta = (2.0, 0.0);
    let (nll, gradient, diagonal_fisher) =
        DependentConstraintFamily.nll_gradient_and_diagonal_fisher_eta(&[1.0, 3.0], &eta, &mut ());

    assert_eq!(nll, 0.0);
    assert_eq!(gradient.part(0), 0.0);
    assert_eq!(gradient.part(1), 0.0);
    assert_eq!(diagonal_fisher, (1.0, 1.0));

    let (nll, gradient, information) = DependentConstraintFamily
        .nll_gradient_and_expected_information_eta(&[1.0, 3.0], &eta, &mut ());

    assert_eq!(nll, 0.0);
    assert_eq!(gradient.part(0), 0.0);
    assert_eq!(gradient.part(1), 0.0);
    assert_eq!(information.get(0, 1), 1.0);
    assert_eq!(information.as_array(), &[[1.0, 1.0], [1.0, 1.0]]);
    assert_eq!(
        DenseInformation::<2>::diagonal([2.0, 3.0]).as_array(),
        &[[2.0, 0.0], [0.0, 3.0]]
    );
}

#[test]
fn public_api_supports_deviance_and_initial_eta_extension_traits() {
    let initial_eta = DependentConstraintFamily.initial_eta(&[1.0, 3.0]);

    assert_eq!(initial_eta, (2.0, 0.0));
    assert_eq!(
        {
            let theta = DependentConstraintFamily.theta(&initial_eta, &mut ());
            DependentConstraintFamily.deviance(&[1.0, 3.0], &theta)
        },
        0.0
    );
}
