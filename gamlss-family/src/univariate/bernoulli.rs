use std::marker::PhantomData;

#[cfg(feature = "rand")]
use gamlss_core::CanSimulate;
use gamlss_core::{
    Family, HasCdf, HasCrps, HasQuantile, InitialEtaFromObservations, InitialEtaFromTheta, Logit,
    Mu, ObservationView, ParameterParts, UnitIntervalLink,
};

use crate::initial::probability_floor;

/// Bernoulli distribution with logit link for success probability.
pub type BernoulliProbability = Bernoulli<Logit>;

/// Bernoulli family parameterized by success probability $p\in(0,1)$.
///
/// For $y\in\\{0,1\\}$, the probability mass is
///
/// $$
/// \Pr(Y=y\mid p)=p^y(1-p)^{1-y}.
/// $$
///
/// Its moments are $\mathbb{E}(Y)=p$ and $\operatorname{Var}(Y)=p(1-p)$. The default [`BernoulliProbability`] alias uses $p=\operatorname{logit}^{-1}(\eta_p)$.
///
/// In the Rust carriers, [`BernoulliTheta::mu`] stores $p$ and [`BernoulliEta::mu`] stores its predictor $\eta_p$.
///
/// The probability link must guarantee values in `(0, 1)`.
///
/// ```compile_fail
/// use gamlss_core::Identity;
/// use gamlss_family::Bernoulli;
///
/// let _ = Bernoulli::<Identity>::new();
/// ```
///
/// ### Parameterization examples
#[cfg_attr(
    doc,
    doc = include_str!("../../doc-assets/distributions/bernoulli.svg")
)]
#[allow(clippy::doc_markdown)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bernoulli<MuLink = Logit> {
    marker: PhantomData<MuLink>,
}

impl<MuLink> Bernoulli<MuLink>
where
    MuLink: UnitIntervalLink<f64>,
{
    /// Creates a stateless Bernoulli family.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    #[inline]
    fn theta_from_eta(eta: BernoulliEta) -> BernoulliTheta {
        BernoulliTheta {
            mu: MuLink::inverse(eta.mu),
        }
    }

    #[inline]
    #[allow(clippy::float_cmp)]
    fn valid_binary(y: f64) -> bool {
        y == 0.0 || y == 1.0
    }

    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn nll_theta(y: f64, theta: BernoulliTheta) -> f64 {
        if !Self::valid_binary(y) || theta.mu <= 0.0 || theta.mu >= 1.0 || !theta.mu.is_finite() {
            return f64::INFINITY;
        }

        -y * theta.mu.ln() - (1.0 - y) * (1.0 - theta.mu).ln()
    }

    #[inline]
    fn nll_and_gradient_eta_values(y: f64, eta: BernoulliEta) -> (f64, BernoulliEta) {
        let theta = Self::theta_from_eta(eta);
        let nll = Self::nll_theta(y, theta);
        if !nll.is_finite() {
            return (nll, BernoulliEta { mu: f64::NAN });
        }

        let d_mu = (1.0 - y) / (1.0 - theta.mu) - y / theta.mu;
        let gradient_eta = BernoulliEta {
            mu: d_mu * MuLink::derivative_inverse(eta.mu),
        };

        (nll, gradient_eta)
    }
}

impl<MuLink> Default for Bernoulli<MuLink>
where
    MuLink: UnitIntervalLink<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

gamlss_core::impl_scalar_compilable_family!(
    impl<MuLink> for Bernoulli<MuLink>;
    parameters = (Mu,);
    arity = 1;
);

impl<MuLink> Family for Bernoulli<MuLink>
where
    MuLink: UnitIntervalLink<f64>,
{
    type Eta = BernoulliEta;
    type Theta = BernoulliTheta;
    type GradientEta = BernoulliEta;
    type Observation<'obs> = f64;
    type Workspace = ();

    #[inline]
    fn workspace(&self) -> Self::Workspace {}

    fn theta(&self, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> Self::Theta {
        Self::theta_from_eta(*eta)
    }

    #[inline]
    fn nll(&self, y: f64, theta: &Self::Theta, _workspace: &mut Self::Workspace) -> f64 {
        Self::nll_theta(y, *theta)
    }

    #[inline]
    fn nll_eta(&self, y: f64, eta: &Self::Eta, _workspace: &mut Self::Workspace) -> f64 {
        Self::nll_theta(y, Self::theta_from_eta(*eta))
    }

    #[inline]
    fn nll_and_gradient_eta(
        &self,
        y: f64,
        eta: &Self::Eta,
        _workspace: &mut Self::Workspace,
    ) -> (f64, Self::GradientEta) {
        Self::nll_and_gradient_eta_values(y, *eta)
    }
}

impl<MuLink> InitialEtaFromObservations<1> for Bernoulli<MuLink>
where
    MuLink: InitialEtaFromTheta<f64> + UnitIntervalLink<f64>,
{
    fn initial_eta_from_observations<'obs, Obs>(&self, obs: &'obs Obs) -> Self::Eta
    where
        Obs: ObservationView<'obs, Observation = Self::Observation<'obs>> + 'obs,
    {
        let mut success_weight = 0.0;
        let mut total_weight = 0.0;

        #[allow(clippy::suboptimal_flops)]
        for row in 0..obs.len() {
            let weight = obs.weight_at(row);
            let y = obs.observation_at(row);
            if weight <= 0.0 || !weight.is_finite() || !Self::valid_binary(y) {
                continue;
            }

            success_weight += weight * y;
            total_weight += weight;
        }

        if total_weight <= 0.0 {
            return BernoulliEta::from_array([0.0]);
        }

        let mu = probability_floor((success_weight + 0.5) / (total_weight + 1.0));
        BernoulliEta {
            mu: MuLink::initial_eta_from_theta(mu),
        }
    }
}

impl<MuLink> HasCdf for Bernoulli<MuLink>
where
    MuLink: UnitIntervalLink<f64>,
{
    fn cdf(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        if !y.is_finite() || theta.mu <= 0.0 || theta.mu >= 1.0 || !theta.mu.is_finite() {
            return f64::NAN;
        }

        if y < 0.0 {
            0.0
        } else if y < 1.0 {
            1.0 - theta.mu
        } else {
            1.0
        }
    }
}

impl<MuLink> HasQuantile for Bernoulli<MuLink>
where
    MuLink: UnitIntervalLink<f64>,
{
    fn quantile(&self, p: f64, theta: &Self::Theta) -> f64 {
        if !(0.0..=1.0).contains(&p) || theta.mu <= 0.0 || theta.mu >= 1.0 || !theta.mu.is_finite()
        {
            return f64::NAN;
        }

        if p <= 1.0 - theta.mu { 0.0 } else { 1.0 }
    }
}

impl<MuLink> HasCrps for Bernoulli<MuLink>
where
    MuLink: UnitIntervalLink<f64>,
{
    fn crps(&self, y: Self::Observation<'_>, theta: &Self::Theta) -> f64 {
        if !Self::valid_binary(y) || theta.mu <= 0.0 || theta.mu >= 1.0 || !theta.mu.is_finite() {
            return f64::NAN;
        }

        let residual = theta.mu - y;
        residual * residual
    }
}

#[cfg(feature = "rand")]
impl<Rng, MuLink> CanSimulate<Rng> for Bernoulli<MuLink>
where
    Rng: rand::Rng,
    MuLink: UnitIntervalLink<f64>,
{
    type Sample = f64;

    fn sample(&self, rng: &mut Rng, theta: &Self::Theta) -> f64 {
        if theta.mu <= 0.0 || theta.mu >= 1.0 || !theta.mu.is_finite() {
            return f64::NAN;
        }

        f64::from(rand_distr::Distribution::sample(
            &rand_distr::Bernoulli::new(theta.mu)
                .expect("validated bernoulli probability must construct"),
            rng,
        ))
    }
}

/// Predictor for the Bernoulli family on the link scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BernoulliEta {
    /// Success-probability predictor.
    pub mu: f64,
}

impl ParameterParts<1> for BernoulliEta {
    #[inline]
    fn from_array(values: [f64; 1]) -> Self {
        Self { mu: values[0] }
    }

    #[inline]
    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mu,
            _ => unreachable!("bernoulli eta only has index 0"),
        }
    }
}

/// Natural-scale Bernoulli parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BernoulliTheta {
    /// Success probability in `(0, 1)`.
    pub mu: f64,
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    #[cfg(feature = "rand")]
    use gamlss_core::CanSimulate;
    use gamlss_core::{Family, HasCdf, HasCrps, HasQuantile};

    use super::{BernoulliProbability, BernoulliTheta};
    use crate::test_support::assert_gradient_matches_finite_difference;

    #[test]
    fn bernoulli_gradient_matches_finite_difference() {
        let family = BernoulliProbability::new();
        assert_gradient_matches_finite_difference::<_, 1>(&family, 1.0, [0.4]);
        assert_gradient_matches_finite_difference::<_, 1>(&family, 0.0, [0.4]);
    }

    #[test]
    fn bernoulli_rejects_invalid_domain_and_has_finite_nll_inside_domain() {
        let family = BernoulliProbability::new();
        let theta = BernoulliTheta { mu: 0.4 };

        assert!(family.nll(1.0, &theta, &mut family.workspace()).is_finite());
        assert!(family.nll(0.0, &theta, &mut family.workspace()).is_finite());
        assert!(
            family
                .nll(0.5, &theta, &mut family.workspace())
                .is_infinite()
        );
        assert!(
            family
                .nll(1.0, &BernoulliTheta { mu: 1.0 }, &mut family.workspace())
                .is_infinite()
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn bernoulli_cdf_matches_reference_points() {
        let family = BernoulliProbability::new();
        let theta = BernoulliTheta { mu: 0.4 };

        assert_eq!(family.cdf(-1.0, &theta), 0.0);
        assert_eq!(family.cdf(0.0, &theta), 0.6);
        assert_eq!(family.cdf(0.5, &theta), 0.6);
        assert_eq!(family.cdf(1.0, &theta), 1.0);
        assert!(family.cdf(f64::NAN, &theta).is_nan());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn bernoulli_quantile_matches_generalized_inverse_cdf() {
        let family = BernoulliProbability::new();
        let theta = BernoulliTheta { mu: 0.4 };

        assert_eq!(family.quantile(0.0, &theta), 0.0);
        assert_eq!(family.quantile(0.6, &theta), 0.0);
        assert_eq!(family.quantile(0.600_000_000_001, &theta), 1.0);
        assert_eq!(family.quantile(1.0, &theta), 1.0);
        assert!(family.quantile(f64::NAN, &theta).is_nan());
        assert!(family.quantile(0.5, &BernoulliTheta { mu: 1.0 }).is_nan());
    }

    #[test]
    fn bernoulli_crps_matches_squared_binary_error() {
        let family = BernoulliProbability::new();
        let theta = BernoulliTheta { mu: 0.4 };

        assert_relative_eq!(family.crps(1.0, &theta), 0.36, epsilon = 1.0e-12);
        assert_relative_eq!(family.crps(0.0, &theta), 0.16, epsilon = 1.0e-12);
    }

    #[test]
    fn bernoulli_crps_returns_nan_for_invalid_domains() {
        let family = BernoulliProbability::new();
        let theta = BernoulliTheta { mu: 0.4 };

        assert!(family.crps(0.5, &theta).is_nan());
        assert!(family.crps(1.0, &BernoulliTheta { mu: 1.0 }).is_nan());
    }

    #[test]
    fn bernoulli_crps_is_nonnegative_for_valid_domains() {
        let family = BernoulliProbability::new();
        let theta = BernoulliTheta { mu: 0.4 };

        assert!(family.crps(1.0, &theta) >= 0.0);
        assert!(family.crps(0.0, &theta) >= 0.0);
    }

    #[cfg(feature = "rand")]
    #[test]
    #[allow(clippy::float_cmp)]
    fn bernoulli_sampling_returns_binary_values_and_nan_for_invalid_theta() {
        use rand::SeedableRng;

        let family = BernoulliProbability::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let sample = family.sample(&mut rng, &BernoulliTheta { mu: 0.4 });
        assert!(sample == 0.0 || sample == 1.0);
        assert!(
            family
                .sample(&mut rng, &BernoulliTheta { mu: 1.0 })
                .is_nan()
        );
    }
}
