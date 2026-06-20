use gamlss_core::{
    ClampedLog, ComponentMean, Cv, DenseDesign, DenseInformation, Dispersion, Family,
    FloorSoftplusScalar, Gamlss, HasDensity, HasDeviance, HasDiagonalFisherInfo,
    HasExpectedInformation, HasInitialEta, HasLogDensity, Identity, LinearForm, LinearFormBuilder,
    Log, LogLocation, LogSd, Logit, Mean, Median, Mu, NegativeSoftplusScalar, NoPenalty, Nu,
    Objective, ObjectiveScale, ObservationView, OneProbability, ParameterBlock, ParameterLayout,
    ParameterName, ParameterParts, ParameterSlice, ParameterizedFamily, PositiveLink, Power,
    PredictorBlock, Probability, Sigma, Size, Softplus, SoftplusScalar, TotalMean,
    TrainingDiagnostics, UnitIntervalLink, ZeroProbability,
};

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
    assert_eq!(layout.slice_of::<Mu>(), Some(0..2));
    assert_eq!(layout.slice_of::<Sigma>(), Some(2..3));
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
    assert_eq!(Size::NAME, "size");
    assert_eq!(Probability::NAME, "probability");
    assert_eq!(ZeroProbability::NAME, "zero_probability");
    assert_eq!(OneProbability::NAME, "one_probability");
    assert_eq!(Power::NAME, "power");
}

#[test]
fn link_domain_marker_traits_remain_root_reexports() {
    fn assert_positive_link<L: PositiveLink<f64>>() {}
    fn assert_unit_interval_link<L: UnitIntervalLink<f64>>() {}

    assert_positive_link::<Log>();
    assert_positive_link::<Softplus>();
    assert_positive_link::<ClampedLog<-12, 12>>();
    assert_unit_interval_link::<Logit>();
}

#[derive(Debug, Clone, PartialEq)]
struct BorrowedRows {
    rows: Vec<Vec<f64>>,
}

impl<'row> ObservationView<'row> for BorrowedRows {
    type Observation = &'row [f64];

    fn len(&self) -> usize {
        self.rows.len()
    }

    fn observation_at(&'row self, row: usize) -> Self::Observation {
        self.rows[row].as_slice()
    }

    fn weight_at(&self, _row: usize) -> f64 {
        1.0
    }
}

#[derive(Debug, Clone, Copy)]
struct DependentConstraintFamily;

impl DependentConstraintFamily {
    fn target(observation: &[f64]) -> f64 {
        observation.iter().sum::<f64>() / observation.len() as f64
    }
}

impl Family for DependentConstraintFamily {
    type Eta = (f64, f64);
    type Theta = (f64, f64);
    type NllGradientEta = (f64, f64);
    type Observation<'obs> = &'obs [f64];

    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        let constrained = eta.1.tanh();
        (eta.0 + constrained, constrained)
    }

    fn nll(&self, observation: Self::Observation<'_>, theta: Self::Theta) -> f64 {
        let residual = theta.0 - Self::target(observation);
        0.5 * residual * residual
    }

    fn nll_and_gradient_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: Self::Eta,
    ) -> (f64, Self::NllGradientEta) {
        let theta = self.theta(eta);
        let residual = theta.0 - Self::target(observation);
        let d_constrained = 1.0 - theta.1 * theta.1;
        (
            self.nll(observation, theta),
            (residual, residual * d_constrained),
        )
    }
}

impl ParameterizedFamily<2> for DependentConstraintFamily {
    type Params = (Mu, Nu);
    type Links = (Identity, Identity);
}

impl HasDiagonalFisherInfo for DependentConstraintFamily {
    fn nll_gradient_and_diagonal_fisher_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: Self::Eta,
    ) -> (f64, Self::NllGradientEta, Self::NllGradientEta) {
        let (nll, gradient) = self.nll_and_gradient_eta(observation, eta);
        let theta = self.theta(eta);
        let d_constrained = 1.0 - theta.1 * theta.1;
        (nll, gradient, (1.0, d_constrained.powi(2)))
    }
}

impl HasExpectedInformation<2> for DependentConstraintFamily {
    fn nll_gradient_and_expected_information_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: Self::Eta,
    ) -> (f64, Self::NllGradientEta, DenseInformation<2>) {
        let (nll, gradient) = self.nll_and_gradient_eta(observation, eta);
        let theta = self.theta(eta);
        let d_constrained = 1.0 - theta.1 * theta.1;
        (
            nll,
            gradient,
            DenseInformation::new([[1.0, d_constrained], [d_constrained, d_constrained.powi(2)]]),
        )
    }
}

impl HasDeviance for DependentConstraintFamily {
    fn deviance<'obs>(&self, observation: Self::Observation<'obs>, theta: Self::Theta) -> f64 {
        2.0 * self.nll(observation, theta)
    }
}

impl HasInitialEta for DependentConstraintFamily {
    fn initial_eta<'obs>(&self, observation: Self::Observation<'obs>) -> Self::Eta {
        (Self::target(observation), 0.0)
    }
}

#[test]
fn public_api_supports_default_log_density_helper() {
    let theta = DependentConstraintFamily.theta((2.0, 0.0));

    assert_eq!(
        DependentConstraintFamily.log_density(&[1.0, 3.0], theta),
        -DependentConstraintFamily.nll(&[1.0, 3.0], theta)
    );
    assert_eq!(DependentConstraintFamily.density(&[1.0, 3.0], theta), 1.0);
}

#[test]
fn public_api_supports_borrowed_observations_nll_gradient_and_dense_information() {
    let obs = BorrowedRows {
        rows: vec![vec![1.0, 3.0], vec![2.0, 4.0]],
    };
    let mu = ParameterBlock::<Mu, Identity, _, _>::linear(
        DenseDesign::intercept(obs.len()),
        NoPenalty,
        0,
    );
    let nu = ParameterBlock::<Nu, Identity, _, _>::linear(
        DenseDesign::intercept(obs.len()),
        NoPenalty,
        1,
    );
    let mut model = Gamlss::try_new_with_observations(DependentConstraintFamily, (mu, nu), obs)
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

    let (nll, gradient, diagonal_fisher) =
        DependentConstraintFamily.nll_gradient_and_diagonal_fisher_eta(&[1.0, 3.0], (2.0, 0.0));

    assert_eq!(nll, 0.0);
    assert_eq!(gradient.part(0), 0.0);
    assert_eq!(gradient.part(1), 0.0);
    assert_eq!(diagonal_fisher, (1.0, 1.0));

    let (nll, gradient, information) = DependentConstraintFamily
        .nll_gradient_and_expected_information_eta(&[1.0, 3.0], (2.0, 0.0));

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
        DependentConstraintFamily
            .deviance(&[1.0, 3.0], DependentConstraintFamily.theta(initial_eta)),
        0.0
    );
}
