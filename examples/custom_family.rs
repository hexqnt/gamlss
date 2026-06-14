//! Example: define a custom distribution family and use it in a GAMLSS model.
//!
//! `UserNormal` intentionally duplicates the normal law to keep the formulas
//! familiar. A real custom family follows the same shape:
//! - choose typed parameter markers and link functions through
//!   `ParameterizedFamily`;
//! - convert link-scale predictors `Eta` to natural parameters `Theta`;
//! - return scalar negative log-likelihood and its score on the link scale.

#![allow(clippy::suboptimal_flops)]

use std::marker::PhantomData;

use gamlss::core::{
    DenseDesign, Family, Gamlss, HasCdf, Identity, Link, Log, Mu, NoPenalty, Objective,
    ParameterBlock, ParameterBlocks, ParameterParts, ParameterizedFamily, PositiveLink, Sigma,
};

const HALF_LOG_2_PI: f64 = 0.918_938_533_204_672_7;

#[derive(Debug, Clone, Copy, PartialEq)]
struct UserNormal<MuLink = Identity, SigmaLink = Log> {
    marker: PhantomData<(MuLink, SigmaLink)>,
}

impl<MuLink, SigmaLink> UserNormal<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct UserNormalEta {
    mu: f64,
    sigma: f64,
}

impl ParameterParts<2> for UserNormalEta {
    fn from_array(values: [f64; 2]) -> Self {
        Self {
            mu: values[0],
            sigma: values[1],
        }
    }

    fn part(&self, index: usize) -> f64 {
        match index {
            0 => self.mu,
            1 => self.sigma,
            _ => unreachable!("user normal eta only has indices 0 and 1"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct UserNormalTheta {
    mu: f64,
    sigma: f64,
}

impl<MuLink, SigmaLink> Family for UserNormal<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    type Eta = UserNormalEta;
    type Theta = UserNormalTheta;
    type ScoreEta = UserNormalEta;

    fn theta(&self, eta: Self::Eta) -> Self::Theta {
        UserNormalTheta {
            mu: MuLink::inverse(eta.mu),
            sigma: SigmaLink::inverse(eta.sigma),
        }
    }

    fn nll(&self, y: f64, theta: Self::Theta) -> f64 {
        if theta.sigma <= 0.0 || !theta.sigma.is_finite() {
            return f64::INFINITY;
        }

        let residual = y - theta.mu;
        let z = residual / theta.sigma;
        HALF_LOG_2_PI + theta.sigma.ln() + 0.5 * z * z
    }

    fn nll_and_score_eta(&self, y: f64, eta: Self::Eta) -> (f64, Self::ScoreEta) {
        let theta = self.theta(eta);
        let nll = self.nll(y, theta);

        let residual = y - theta.mu;
        let sigma2 = theta.sigma * theta.sigma;
        let d_nll_d_mu = (theta.mu - y) / sigma2;
        let d_nll_d_sigma = (1.0 / theta.sigma) - (residual * residual / (sigma2 * theta.sigma));

        let score_eta = UserNormalEta {
            mu: d_nll_d_mu * MuLink::derivative_inverse(eta.mu),
            sigma: d_nll_d_sigma * SigmaLink::derivative_inverse(eta.sigma),
        };

        (nll, score_eta)
    }
}

impl<MuLink, SigmaLink> ParameterizedFamily<2> for UserNormal<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    type Params = (Mu, Sigma);
    type Links = (MuLink, SigmaLink);
}

impl<MuLink, SigmaLink> HasCdf for UserNormal<MuLink, SigmaLink>
where
    MuLink: Link<f64>,
    SigmaLink: PositiveLink<f64>,
{
    fn cdf(&self, y: f64, theta: Self::Theta) -> f64 {
        if theta.sigma <= 0.0 || !theta.sigma.is_finite() {
            return f64::NAN;
        }

        let z = (y - theta.mu) / (theta.sigma * std::f64::consts::SQRT_2);
        f64::midpoint(1.0, erf_approx(z))
    }
}

fn erf_approx(x: f64) -> f64 {
    // Abramowitz-Stegun 7.1.26 approximation. Good enough for a helper example;
    // likelihood and score above do not depend on this approximation.
    let sign = if x.is_sign_negative() { -1.0 } else { 1.0 };
    let abs_x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * abs_x);
    let polynomial =
        (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t;

    sign * (1.0 - polynomial * (-abs_x * abs_x).exp())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let y = vec![1.0, 1.4, 1.8, 2.2, 2.6];
    let n = y.len();

    let blocks = ParameterBlocks::new((
        ParameterBlock::<Mu, Identity, _, _>::linear(DenseDesign::intercept(n), NoPenalty, 0),
        ParameterBlock::<Sigma, Log, _, _>::linear(DenseDesign::intercept(n), NoPenalty, 0),
    ));

    let family = UserNormal::<Identity, Log>::new();
    let mut model = Gamlss::try_new(family, blocks, y)?;
    let mut theta = model.initial_theta()?;
    let mut grad = vec![0.0; model.dim()];

    for _ in 0..2_000 {
        model.gradient(&theta, &mut grad)?;
        for (theta_value, grad_value) in theta.iter_mut().zip(&grad) {
            *theta_value -= 0.02 * grad_value;
        }
    }

    let diagnostics = model.diagnostics(&theta)?;
    let coefficients = model.unpack_theta(&theta)?;
    let mu_hat = coefficients.coefficients("mu").expect("mu block")[0];
    let sigma_hat = coefficients.coefficients("sigma").expect("sigma block")[0].exp();
    let median_cdf = family.cdf(
        mu_hat,
        UserNormalTheta {
            mu: mu_hat,
            sigma: sigma_hat,
        },
    );

    println!(
        "custom_family: objective={:.6}, grad_norm={:.6}, mu={mu_hat:.4}, sigma={sigma_hat:.4}, cdf(mu)={median_cdf:.4}",
        diagnostics.objective, diagnostics.gradient_norm,
    );

    Ok(())
}
