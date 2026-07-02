use gamlss_core::{Family, MixtureSpec, MixtureWeight, ModelError, Repeated, SimplexWeights};

/// Homogeneous fixed-size mixture of one component family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mixture<F, const C: usize> {
    component: F,
}

impl<F, const C: usize> Mixture<F, C> {
    /// Creates a homogeneous mixture family.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidParameter`] when `C < 2`.
    #[inline]
    pub fn try_new(component: F) -> Result<Self, ModelError> {
        if C < 2 {
            return Err(ModelError::InvalidParameter {
                parameter: "components",
                expected: "at least two mixture components",
            });
        }
        Ok(Self { component })
    }

    /// Returns the shared component family.
    #[must_use]
    #[inline]
    pub const fn component(&self) -> &F {
        &self.component
    }
}

impl<F, const C: usize> Family for Mixture<F, C>
where
    F: Family,
    F::GradientEta: Clone,
    for<'obs> F::Observation<'obs>: Clone,
{
    type Eta = MixtureEta<F::Eta, C>;
    type Theta = MixtureTheta<F::Theta, C>;
    type GradientEta = MixtureGradient<F::GradientEta, C>;
    type Observation<'obs> = F::Observation<'obs>;
    type Workspace = MixtureWorkspace<F::Workspace, C>;
    type ParamSpec = MixtureSpec<SimplexWeights<MixtureWeight, C>, Repeated<F::ParamSpec, C>, C>;

    fn workspace(&self) -> Self::Workspace {
        MixtureWorkspace {
            components: std::array::from_fn(|_| self.component.workspace()),
        }
    }

    fn theta(&self, eta: &Self::Eta, workspace: &mut Self::Workspace) -> Self::Theta {
        MixtureTheta {
            weights: softmax_baseline(eta.logits),
            components: std::array::from_fn(|index| {
                self.component
                    .theta(&eta.components[index], &mut workspace.components[index])
            }),
        }
    }

    fn nll(
        &self,
        observation: Self::Observation<'_>,
        theta: &Self::Theta,
        workspace: &mut Self::Workspace,
    ) -> f64 {
        if C < 2 || !valid_weights(&theta.weights) {
            return f64::INFINITY;
        }

        let terms: [f64; C] = std::array::from_fn(|index| {
            let nll = self.component.nll(
                observation.clone(),
                &theta.components[index],
                &mut workspace.components[index],
            );
            theta.weights[index].ln() - nll
        });
        -log_sum_exp(&terms)
    }

    fn nll_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        workspace: &mut Self::Workspace,
    ) -> f64 {
        if C < 2 {
            return f64::INFINITY;
        }

        let weights = softmax_baseline(eta.logits);
        let terms: [f64; C] = std::array::from_fn(|index| {
            let nll = self.component.nll_eta(
                observation.clone(),
                &eta.components[index],
                &mut workspace.components[index],
            );
            weights[index].ln() - nll
        });
        -log_sum_exp(&terms)
    }

    fn nll_and_gradient_eta(
        &self,
        observation: Self::Observation<'_>,
        eta: &Self::Eta,
        workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        if C < 2 {
            return (
                f64::INFINITY,
                MixtureGradient {
                    logits: [f64::NAN; C],
                    components: std::array::from_fn(|index| {
                        let (_, gradient) = self.component.nll_and_gradient_eta(
                            observation.clone(),
                            &eta.components[index],
                            &mut workspace.components[index],
                        );
                        ResponsibleGradient {
                            responsibility: f64::NAN,
                            gradient,
                        }
                    }),
                },
            );
        }

        let weights = softmax_baseline(eta.logits);
        let mut component_nll = [0.0; C];
        let component_gradients: [F::GradientEta; C] = std::array::from_fn(|index| {
            let (nll, gradient) = self.component.nll_and_gradient_eta(
                observation.clone(),
                &eta.components[index],
                &mut workspace.components[index],
            );
            component_nll[index] = nll;
            gradient
        });

        let terms: [f64; C] =
            std::array::from_fn(|index| weights[index].ln() - component_nll[index]);
        let log_mix = log_sum_exp(&terms);
        let nll = -log_mix;
        let responsibilities = terms.map(|term| (term - log_mix).exp());

        let mut logits = [0.0; C];
        for ((logit, weight), responsibility) in logits
            .iter_mut()
            .zip(weights.iter().copied())
            .zip(responsibilities.iter().copied())
            .take(C.saturating_sub(1))
        {
            *logit = weight - responsibility;
        }

        (
            nll,
            MixtureGradient {
                logits,
                components: std::array::from_fn(|index| ResponsibleGradient {
                    responsibility: responsibilities[index],
                    gradient: component_gradients[index].clone(),
                }),
            },
        )
    }
}

/// Link-scale predictors for a homogeneous fixed-size mixture.
#[derive(Debug, Clone, PartialEq)]
pub struct MixtureEta<Eta, const C: usize> {
    /// Baseline-softmax logits. The last entry is normalized to zero.
    pub logits: [f64; C],
    /// Component link-scale predictors.
    pub components: [Eta; C],
}

impl<Eta, const C: usize> MixtureEta<Eta, C> {
    /// Creates mixture predictors and normalizes the baseline logit to zero.
    #[must_use]
    pub const fn new(mut logits: [f64; C], components: [Eta; C]) -> Self {
        if C > 0 {
            logits[C - 1] = 0.0;
        }
        Self { logits, components }
    }
}

/// Natural-scale parameters for a homogeneous fixed-size mixture.
#[derive(Debug, Clone, PartialEq)]
pub struct MixtureTheta<Theta, const C: usize> {
    /// Normalized mixture weights.
    pub weights: [f64; C],
    /// Component natural-scale parameters.
    pub components: [Theta; C],
}

/// Component gradient paired with its posterior responsibility.
#[derive(Debug, Clone, PartialEq)]
pub struct ResponsibleGradient<Gradient> {
    /// Posterior component responsibility for the current observation.
    pub responsibility: f64,
    /// Unscaled component gradient on that component's eta scale.
    pub gradient: Gradient,
}

/// Link-scale NLL gradient for a homogeneous fixed-size mixture.
#[derive(Debug, Clone, PartialEq)]
pub struct MixtureGradient<Gradient, const C: usize> {
    /// Gradients for baseline-softmax logits. The last baseline entry is zero.
    pub logits: [f64; C],
    /// Component gradients paired with responsibility multipliers.
    pub components: [ResponsibleGradient<Gradient>; C],
}

/// Reusable per-component family workspaces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixtureWorkspace<W, const C: usize> {
    components: [W; C],
}

fn softmax_baseline<const C: usize>(mut logits: [f64; C]) -> [f64; C] {
    if C == 0 {
        return logits;
    }
    logits[C - 1] = 0.0;
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    let mut weights = [0.0; C];
    for (weight, logit) in weights.iter_mut().zip(logits) {
        *weight = (logit - max).exp();
        sum += *weight;
    }
    for weight in &mut weights {
        *weight /= sum;
    }
    weights
}

fn valid_weights(weights: &[f64]) -> bool {
    weights
        .iter()
        .all(|weight| weight.is_finite() && *weight > 0.0)
        && weights.iter().sum::<f64>().is_finite()
}

fn log_sum_exp(values: &[f64]) -> f64 {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return max;
    }
    max + values
        .iter()
        .map(|value| (*value - max).exp())
        .sum::<f64>()
        .ln()
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use gamlss_core::Family;

    use super::{Mixture, MixtureEta};
    use crate::{NormalEta, NormalMuSigma};

    #[test]
    fn rejects_less_than_two_components() {
        assert!(Mixture::<_, 1>::try_new(NormalMuSigma::new()).is_err());
    }

    #[test]
    fn nll_matches_manual_log_sum_exp() {
        let family = Mixture::<_, 2>::try_new(NormalMuSigma::new()).unwrap();
        let eta = mixture_eta();
        let y = 0.4;
        let mut workspace = family.workspace();
        let nll = family.nll_eta(y, &eta, &mut workspace);

        let mut normal_workspace = NormalMuSigma::new().workspace();
        let normal = NormalMuSigma::new();
        let weights = [
            eta.logits[0].exp() / (eta.logits[0].exp() + 1.0),
            1.0 / (eta.logits[0].exp() + 1.0),
        ];
        let terms = [
            weights[0].ln() - normal.nll_eta(y, &eta.components[0], &mut normal_workspace),
            weights[1].ln() - normal.nll_eta(y, &eta.components[1], &mut normal_workspace),
        ];
        let manual = -(terms[0].max(terms[1])
            + ((terms[0] - terms[0].max(terms[1])).exp()
                + (terms[1] - terms[0].max(terms[1])).exp())
            .ln());

        assert_relative_eq!(nll, manual, epsilon = 1.0e-12);
    }

    #[test]
    fn gradients_match_finite_differences() {
        let family = Mixture::<_, 2>::try_new(NormalMuSigma::new()).unwrap();
        let eta = mixture_eta();
        let y = 0.4;
        let mut workspace = family.workspace();
        let (_, gradient) = family.nll_and_gradient_eta(y, &eta, &mut workspace);

        let fd_logit = finite_difference(
            &family,
            &eta,
            y,
            |probe, value| {
                probe.logits[0] = value;
            },
            eta.logits[0],
        );
        assert_relative_eq!(gradient.logits[0], fd_logit, epsilon = 1.0e-6);
        assert_relative_eq!(gradient.logits[1], 0.0, epsilon = 1.0e-12);

        let fd_mu0 = finite_difference(
            &family,
            &eta,
            y,
            |probe, value| {
                probe.components[0].mu = value;
            },
            eta.components[0].mu,
        );
        assert_relative_eq!(
            gradient.components[0].responsibility * gradient.components[0].gradient.mu,
            fd_mu0,
            epsilon = 1.0e-6
        );
    }

    fn mixture_eta() -> MixtureEta<NormalEta, 2> {
        MixtureEta::new(
            [0.7, 12.0],
            [
                NormalEta {
                    mu: -0.3,
                    sigma: -0.2,
                },
                NormalEta {
                    mu: 1.1,
                    sigma: 0.4,
                },
            ],
        )
    }

    fn finite_difference(
        family: &Mixture<NormalMuSigma, 2>,
        eta: &MixtureEta<NormalEta, 2>,
        y: f64,
        mut set: impl FnMut(&mut MixtureEta<NormalEta, 2>, f64),
        center: f64,
    ) -> f64 {
        let step = 1.0e-6;
        let mut plus = eta.clone();
        let mut minus = eta.clone();
        set(&mut plus, center + step);
        set(&mut minus, center - step);
        let mut workspace = family.workspace();
        let plus_nll = family.nll_eta(y, &plus, &mut workspace);
        let minus_nll = family.nll_eta(y, &minus, &mut workspace);
        (plus_nll - minus_nll) / (2.0 * step)
    }
}
