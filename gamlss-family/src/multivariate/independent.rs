#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{Family, FixedDimensionalFamily, HasCdf, HasMarginalCdf, Repeated};

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
    /// Creates an independent product family from a scalar component family.
    #[must_use]
    #[inline]
    pub const fn new(component: F) -> Self {
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
    type ParamSpec = Repeated<F::ParamSpec, D>;

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
impl<Rng, F, const D: usize> CanSimulate<Rng> for IndependentVec<F, D>
where
    F: for<'obs> Family<Observation<'obs> = f64> + CanSimulate<Rng>,
{
    type Sample = [F::Sample; D];

    fn sample(&self, rng: &mut Rng, theta: &Self::Theta) -> Self::Sample {
        std::array::from_fn(|component| self.component.sample(rng, &theta[component]))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::{Family, HasCdf, HasMarginalCdf};

    use super::IndependentVec;
    use crate::{NormalEta, NormalMuSigma, NormalTheta};

    #[test]
    fn default_and_into_component_round_trip_component_family() {
        let family = IndependentVec::<NormalMuSigma, 2>::default();
        assert_eq!(family.component(), &NormalMuSigma::new());
        assert_eq!(family.into_component(), NormalMuSigma::new());
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
}
