use gamlss_core::{
    CompilableFamily, Family, FixedDimensionalFamily, HasCdf, HasMarginalCdf,
    HasObservationDimension, ModelError, ObservationView,
    shape::{Repeated, ShapeValues},
};
#[cfg(feature = "rand")]
use gamlss_core::{SimulationError, TrySimulate};

/// Independent fixed-size product of one scalar family.
///
/// This is the baseline multivariate construction:
///
/// `p(y_0, ..., y_{D-1}) = product_i p_i(y_i)`.
///
/// It deliberately models independence rather than pretending that every scalar
/// distribution has one canonical dependent multivariate analogue. Dependence
/// structures such as copulas, shared factors, or elliptical covariance should
/// be represented by separate families.
///
/// `D == 0` is treated as an invalid model shape by likelihood/CDF evaluation:
/// likelihood methods return `INFINITY`, and CDF evaluation returns `NaN`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndependentVec<F, const D: usize> {
    component: F,
}

impl<F, const D: usize> IndependentVec<F, D> {
    /// Creates an independent product family after checking the compile-time dimension.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `D == 0`.
    #[inline]
    pub fn try_new(component: F) -> Result<Self, ModelError> {
        if D == 0 {
            return Err(ModelError::InvalidParameter {
                parameter: "dimension",
                expected: "positive",
            });
        }
        Ok(Self { component })
    }

    /// Creates an independent product family from a scalar component family.
    ///
    /// # Panics
    ///
    /// Panics when the compile-time dimension is zero.
    #[must_use]
    #[inline]
    pub const fn new(component: F) -> Self {
        assert!(D > 0, "independent product dimension must be positive");
        Self { component }
    }

    /// Returns the scalar component family.
    #[must_use]
    #[inline]
    pub const fn component(&self) -> &F {
        &self.component
    }

    /// Consumes the product family and returns the scalar component family.
    #[must_use]
    #[inline]
    pub fn into_component(self) -> F {
        self.component
    }
}

impl<F, const D: usize> Default for IndependentVec<F, D>
where
    F: Default,
{
    fn default() -> Self {
        Self::new(F::default())
    }
}

impl<F, const D: usize> Family for IndependentVec<F, D>
where
    F: for<'obs> Family<Observation<'obs> = f64>,
{
    type Eta = [F::Eta; D];
    type Theta = [F::Theta; D];
    type GradientEta = [F::GradientEta; D];
    type Observation<'obs> = [f64; D];
    type Workspace = [F::Workspace; D];

    fn workspace(&self) -> Self::Workspace {
        std::array::from_fn(|_| self.component.workspace())
    }

    fn theta(&self, eta: &Self::Eta, workspace: &mut Self::Workspace) -> Self::Theta {
        std::array::from_fn(|component| {
            self.component
                .theta(&eta[component], &mut workspace[component])
        })
    }

    fn nll(
        &self,
        observation: Self::Observation<'_>,
        theta: &Self::Theta,
        workspace: &mut Self::Workspace,
    ) -> f64 {
        if D == 0 {
            return f64::INFINITY;
        }

        observation
            .iter()
            .copied()
            .enumerate()
            .map(|(component, y)| {
                self.component
                    .nll(y, &theta[component], &mut workspace[component])
            })
            .sum()
    }

    fn nll_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        workspace: &mut Self::Workspace,
    ) -> f64 {
        if D == 0 {
            return f64::INFINITY;
        }

        observation
            .iter()
            .copied()
            .enumerate()
            .map(|(component, y)| {
                self.component
                    .nll_eta(y, &eta[component], &mut workspace[component])
            })
            .sum()
    }

    fn nll_and_gradient_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        if D == 0 {
            return (
                f64::INFINITY,
                std::array::from_fn(|component| {
                    let (_, gradient) = self.component.nll_and_gradient_eta(
                        f64::NAN,
                        &eta[component],
                        &mut workspace[component],
                    );
                    gradient
                }),
            );
        }

        let mut loss = 0.0;
        let gradient = std::array::from_fn(|component| {
            let (nll, gradient) = self.component.nll_and_gradient_eta(
                observation[component],
                &eta[component],
                &mut workspace[component],
            );
            loss += nll;
            gradient
        });

        (loss, gradient)
    }
}

impl<F, const D: usize> FixedDimensionalFamily<D> for IndependentVec<F, D> where
    F: for<'obs> Family<Observation<'obs> = f64>
{
}

impl<F, const D: usize> HasObservationDimension for IndependentVec<F, D>
where
    F: for<'obs> Family<Observation<'obs> = f64>,
{
    fn observation_dimension(&self) -> usize {
        D
    }
}

impl<F, const D: usize> HasCdf for IndependentVec<F, D>
where
    F: for<'obs> Family<Observation<'obs> = f64> + HasCdf,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        if D == 0 {
            return f64::NAN;
        }

        y.iter()
            .copied()
            .enumerate()
            .map(|(component, value)| self.component.cdf(value, &theta[component]))
            .product()
    }
}

impl<F, const D: usize> HasMarginalCdf for IndependentVec<F, D>
where
    F: for<'obs> Family<Observation<'obs> = f64> + HasCdf,
{
    fn marginal_cdf(&self, component: usize, y: f64, theta: &Self::Theta) -> f64 {
        if component < D {
            self.component.cdf(y, &theta[component])
        } else {
            f64::NAN
        }
    }
}

#[cfg(feature = "rand")]
impl<Rng, F, const D: usize> TrySimulate<Rng> for IndependentVec<F, D>
where
    F: for<'obs> Family<Observation<'obs> = f64> + TrySimulate<Rng>,
{
    type Sample = [F::Sample; D];

    fn try_sample(
        &self,
        rng: &mut Rng,
        theta: &Self::Theta,
    ) -> Result<Self::Sample, SimulationError> {
        let mut samples = Vec::with_capacity(D);
        for component_theta in theta {
            samples.push(self.component.try_sample(rng, component_theta)?);
        }
        samples
            .try_into()
            .map_err(|_| SimulationError::NumericalFailure("independent-product sample dimension"))
    }

    fn try_sample_into(
        &self,
        rng: &mut Rng,
        theta: &Self::Theta,
        out: &mut Self::Sample,
    ) -> Result<(), SimulationError> {
        for (component, component_theta) in theta.iter().enumerate() {
            self.component
                .try_sample_into(rng, component_theta, &mut out[component])?;
        }
        Ok(())
    }
}

impl<F, const D: usize> CompilableFamily for IndependentVec<F, D>
where
    F: for<'obs> CompilableFamily<Observation<'obs> = f64>,
{
    type Shape = Repeated<F::Shape, D>;

    fn eta_from_shape(values: ShapeValues<Self::Shape>) -> <Self as Family>::Eta {
        values.map(F::eta_from_shape)
    }

    fn gradient_to_shape(gradient: &<Self as Family>::GradientEta) -> ShapeValues<Self::Shape> {
        std::array::from_fn(|component| F::gradient_to_shape(&gradient[component]))
    }

    fn initial_shape<'obs, Obs>(&self, obs: &'obs Obs) -> ShapeValues<Self::Shape>
    where
        Obs: ObservationView<'obs, Observation = [f64; D]> + 'obs,
    {
        std::array::from_fn(|component| {
            let mut values = Vec::with_capacity(obs.len());
            let mut weights = Vec::with_capacity(obs.len());
            for row in 0..obs.len() {
                values.push(obs.observation_at(row)[component]);
                weights.push(obs.weight_at(row));
            }
            self.component
                .initial_shape(&(values.as_slice(), weights.as_slice()))
        })
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{
        DenseDesign, Family, Gamlss, HasCdf, HasMarginalCdf, InitialEtaFromObservations,
        ModelError, Mu, NoPenalty, ParameterBlock, ParameterBlocks, Sigma,
    };

    use super::IndependentVec;
    use crate::{NormalEta, NormalMuSigma, NormalTheta};

    #[test]
    fn default_and_into_component_round_trip_component_family() {
        let family = IndependentVec::<NormalMuSigma, 2>::default();
        assert_eq!(family.component(), &NormalMuSigma::new());
        assert_eq!(family.into_component(), NormalMuSigma::new());
    }

    #[test]
    fn checked_constructor_rejects_zero_dimension() {
        assert_eq!(
            IndependentVec::<NormalMuSigma, 0>::try_new(NormalMuSigma::new()),
            Err(ModelError::InvalidParameter {
                parameter: "dimension",
                expected: "positive",
            })
        );
        assert!(IndependentVec::<NormalMuSigma, 1>::try_new(NormalMuSigma::new()).is_ok());
    }

    #[test]
    fn nll_is_sum_of_component_nlls() {
        let scalar = NormalMuSigma::new();
        let family = IndependentVec::<_, 2>::new(scalar);
        let theta = [
            NormalTheta {
                mu: 0.0,
                sigma: 1.0,
            },
            NormalTheta {
                mu: 1.0,
                sigma: 2.0,
            },
        ];
        let y = [0.5, 1.5];
        let mut workspace = family.workspace();

        let expected = scalar.nll(y[0], &theta[0], &mut ()) + scalar.nll(y[1], &theta[1], &mut ());

        assert_relative_eq!(family.nll(y, &theta, &mut workspace), expected);
    }

    #[test]
    fn gradient_matches_component_gradients() {
        let scalar = NormalMuSigma::new();
        let family = IndependentVec::<_, 2>::new(scalar);
        let eta = [
            NormalEta {
                mu: 0.0,
                sigma: 0.0,
            },
            NormalEta {
                mu: 1.0,
                sigma: 2.0_f64.ln(),
            },
        ];
        let y = [0.5, 1.5];
        let mut workspace = family.workspace();

        let (nll, gradient) = family.nll_and_gradient_eta(y, &eta, &mut workspace);
        let (expected_0, gradient_0) = scalar.nll_and_gradient_eta(y[0], &eta[0], &mut ());
        let (expected_1, gradient_1) = scalar.nll_and_gradient_eta(y[1], &eta[1], &mut ());

        assert_relative_eq!(nll, expected_0 + expected_1);
        assert_relative_eq!(gradient[0].mu, gradient_0.mu);
        assert_relative_eq!(gradient[0].sigma, gradient_0.sigma);
        assert_relative_eq!(gradient[1].mu, gradient_1.mu);
        assert_relative_eq!(gradient[1].sigma, gradient_1.sigma);
    }

    #[test]
    fn cdf_is_joint_product_and_marginal_cdf_selects_component() {
        let scalar = NormalMuSigma::new();
        let family = IndependentVec::<_, 2>::new(scalar);
        let theta = [
            NormalTheta {
                mu: 0.0,
                sigma: 1.0,
            },
            NormalTheta {
                mu: 1.0,
                sigma: 2.0,
            },
        ];

        let expected = scalar.cdf(0.5, &theta[0]) * scalar.cdf(1.5, &theta[1]);
        assert_relative_eq!(family.cdf([0.5, 1.5], &theta), expected);
        assert_relative_eq!(
            family.marginal_cdf(1, 1.5, &theta),
            scalar.cdf(1.5, &theta[1])
        );
        assert!(family.marginal_cdf(2, 1.5, &theta).is_nan());
    }

    #[test]
    fn repeated_scalar_component_specs_are_fit_ready() {
        let y = [[0.2, -0.3], [1.0, 0.4], [-0.5, 0.8]];
        let n = y.len();
        let blocks = ParameterBlocks::new(std::array::from_fn::<_, 2, _>(|_| {
            (
                ParameterBlock::<Mu, _, _>::linear(DenseDesign::intercept(n), NoPenalty, 99),
                ParameterBlock::<Sigma, _, _>::linear(DenseDesign::intercept(n), NoPenalty, 99),
            )
        }));
        let model = Gamlss::try_new_with_observations(
            IndependentVec::<NormalMuSigma, 2>::default(),
            blocks,
            y.as_slice(),
        )
        .unwrap();
        let beta = vec![0.1, 0.0, -0.2, 0.3];
        let eta = model.predict_eta_row(&beta, 0).unwrap();

        assert_eq!(model.nparams(), 4);
        assert_relative_eq!(eta[0].mu, 0.1);
        assert_relative_eq!(eta[1].mu, -0.2);
        assert_relative_eq!(eta[1].sigma, 0.3);

        let scalar = NormalMuSigma::new();
        let component_0 = [0.2, 1.0, -0.5];
        let component_1 = [-0.3, 0.4, 0.8];
        let expected_0 = scalar.initial_eta_from_observations(&component_0.as_slice());
        let expected_1 = scalar.initial_eta_from_observations(&component_1.as_slice());
        assert_eq!(
            model.initial_parameters().unwrap(),
            vec![
                expected_0.mu,
                expected_0.sigma,
                expected_1.mu,
                expected_1.sigma
            ]
        );

        let mut gradient = vec![0.0; beta.len()];
        model.try_value_gradient_into(&beta, &mut gradient).unwrap();
        for index in 0..beta.len() {
            let mut plus = beta.clone();
            plus[index] += 1.0e-6;
            let mut minus = beta.clone();
            minus[index] -= 1.0e-6;
            let fd = (model.try_value(&plus).unwrap() - model.try_value(&minus).unwrap()) / 2.0e-6;
            assert_relative_eq!(gradient[index], fd, epsilon = 1.0e-6);
        }
    }
}
