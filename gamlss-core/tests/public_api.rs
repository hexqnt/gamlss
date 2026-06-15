use gamlss_core::{
    DenseDesign, DenseInformation, Family, FloorSoftplusScalar, Gamlss, HasDiagonalFisherInfo,
    HasExpectedInformation, Identity, Mu, NegativeSoftplusScalar, NoPenalty, Nu, Objective,
    ObservationView, ParameterBlock, ParameterParts, ParameterizedFamily, PredictorBlock,
    SoftplusScalar,
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
