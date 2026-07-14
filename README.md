# gamlss

[![CI](https://github.com/hexqnt/gamlss/actions/workflows/ci.yml/badge.svg)](https://github.com/hexqnt/gamlss/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/gamlss.svg)](https://crates.io/crates/gamlss)[![docs.rs](https://img.shields.io/docsrs/gamlss)](https://docs.rs/gamlss)

Type-driven Rust crates for GAMLSS-style modeling.

> **Status:** Actively developed. Public API and numerical behavior may still change before a stable 1.0 release.

## Crate layout

The primary entry point for users is the `gamlss` crate:

`gamlss` is the batteries-included facade. The main and most stable path today goes through the low-level typed API exposed by `gamlss-core`, `gamlss-family`, `gamlss-spline`, `gamlss-special`, and `gamlss-transform`. By default, the facade also re-exports `gamlss-formula`; that layer is currently an experimental optional convenience crate, not the library's core API.

The workspace also publishes separate crates for finer control over APIs and dependencies:

- `gamlss-core` — type-driven core abstractions for links, parameter blocks, objectives, and compiled models.
- `gamlss-bayes` — normalized coefficient priors and summed-likelihood posterior potentials, without a sampler backend.
- `gamlss-family` — distributions, likelihoods, and score helpers.
- `gamlss-diagnostics` — post-fit PIT/CDF diagnostics, normalized quantile residuals, and CRPS summaries for supported families.
- `gamlss-special` — special functions and shared numerical helpers for likelihood, CDF, and quantile code.
- `gamlss-spline` — spline/Fourier predictors, penalties, and spline metadata.
- `gamlss-transform` — target preprocessing transforms.
- `gamlss-formula` — experimental optional formula/builder layer that compiles runtime specifications into typed models. It covers curated high-level workflows and is not expected to mirror every family, link, or parameterization available in the low-level crates.

For most use cases, depending on `gamlss` is enough. Bayesian support is opt-in through the `bayes` feature; individual workspace crates remain available when tighter dependency and API control is preferable. If you want a stricter low-level surface without the experimental formula layer, use `default-features = false` or depend on the individual crates directly.

## Cargo features

- `formula` is enabled by default and re-exports the experimental `gamlss-formula` namespace from the facade crate.
- `bayes` opt-in re-exports `gamlss-bayes` as `gamlss::bayes` and adds its common types to `gamlss::prelude`.
- `rand` enables the sampling API in `gamlss-family` through the facade crate: `rand = ["gamlss-family/rand"]`.
- `multivariate` enables fixed- and runtime-dimensional multivariate families.

## Running tests

Run the full workspace test suite:

```bash
cargo test --workspace --all-features
```

Numerical tests for `gamlss-family` can also be run separately:

```bash
cargo test -p gamlss-family --tests --all-features
```

## GAMLSS overview

GAMLSS can be read as distributional regression: the model describes not only the conditional mean of the response, but the full conditional distribution. This is useful when dispersion, skewness, tail behavior, or even the valid response domain changes with the features. For example, one part of a model may describe the distribution center, another a heteroskedastic scale, and a third the tail shape.

In general, GAMLSS defines the response distribution through a set of parameters from a chosen family:

$$
Y_i \mid x_i \sim D(\theta_{i1}, \ldots, \theta_{iK}),
$$

where `D(...)` is the selected parametric distribution. Each parameter is modeled with its own link function and predictor:

$$
g_k(\theta_{ik}) = \eta_{ik}
  = X_{k,i}\beta_k + \sum_j f_{k,j}(x_i),
\qquad k = 1,\ldots,K.
$$

In other words, different parameters of the same distribution can use different feature sets, spline terms, penalties, and domain constraints. A link function maps the unconstrained linear predictor `eta` into the parameter's valid domain: scale parameters usually need to be positive, while probability or mean parameters for a beta family must lie inside `(0, 1)`.

The classic `gamlss` convention often names the first four parameters `mu`, `sigma`, `nu`, and `tau`:

$$
(\theta_{i1}, \theta_{i2}, \theta_{i3}, \theta_{i4})
  = (\mu_i, \sigma_i, \nu_i, \tau_i),
$$

$$
Y_i \mid x_i \sim D(\mu_i, \sigma_i, \nu_i, \tau_i).
$$

Here `mu`, `sigma`, `nu`, and `tau` usually correspond to location, scale, skewness, and shape. This is a naming convention, not a required API shape: not every family uses all four parameters, and the typed core supports custom parameter markers for application-specific parameter counts and meanings. As a result, the library can express familiar location-scale models such as normal, log-normal, Laplace, and Student's t, as well as differently parameterized families such as gamma, Weibull, inverse Gaussian, and beta.

## Modeling features

These tables summarize what the library can model today, grouped by modeling regime. Planned regimes are listed separately so they do not look like current scalar GAMLSS capabilities.

### Scalar GAMLSS

| Area                                               | Status | What it means for modeling                                                                                                                                                                                                                  |
| -------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Distributional regression core                     | ✅     | Multiple distribution parameters can each have their own predictor, link, penalty, and covariates.                                                                                                                                          |
| Continuous response families                       | ✅     | Normal, log-normal, gamma, Weibull, inverse Gaussian, exponential, beta, logistic, Laplace, Student-t, skew-normal, skew Student-t, generalized gamma, generalized error / power exponential, Gumbel, GEV, Johnson SU, sinh-arcsinh, Lomax. |
| Discrete and zero-inflated families                | ✅     | Bernoulli, Poisson, negative binomial, zero-inflated Poisson, zero-inflated negative binomial, zero-adjusted gamma, beta inflated at zero and one, Tweedie.                                                                                 |
| Location, scale, shape, skewness and tail modeling | ✅     | Families can expose 1-4 typed parameters such as mean/location, scale/CV/precision/dispersion, skewness, degrees of freedom, shape, power, or zero probability.                                                                             |
| Smooth and structured predictors                   | ✅     | Linear terms, offsets, interactions, P-splines, cyclic/periodic smooths, Fourier terms, monotone I-splines, tensor-product splines, and smoothness/shape penalties.                                                                         |
| Likelihood, score and objective evaluation         | ✅     | Negative log-likelihood and analytical gradients for fitting; optimizer remains external.                                                                                                                                                   |
| Distribution functions and simulation              | 🧩     | CDF, quantile, CRPS, density/log-density and sampling are exposed through capability traits and vary by family.                                                                                                                             |
| Post-fit distribution diagnostics                  | ✅     | PIT/CDF values, normalized quantile residuals, CRPS values and summaries for supported families.                                                                                                                                            |
| Target/response transforms                         | ✅     | Log, log1p-shift, Box-Cox, Yeo-Johnson, standardization, robust standardization, min-max/max-abs scaling, quantile transforms, asinh scaling, and composable persisted transform state.                                                     |
| Formula/builder workflows                          | 🧪     | Curated dynamic builders for normal, beta, gamma, inverse Gaussian, log-normal and Weibull compile into typed models; low-level typed API covers more families.                                                                             |

### Multivariate GAMLSS

| Area                         | Status | What it means for modeling                                                                                             |
| ---------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------- |
| Joint response distributions | ✅     | Fixed and dynamic multivariate Normal, multivariate Student-t, Dirichlet, and independent products use structured observations and parameters. |
| Dependence modeling          | ✅/🧩  | Cholesky and marginal-scale/partial-correlation covariance models are fit-ready; copulas and factor covariance remain future slices. |
| Multivariate diagnostics     | ✅/🧩  | Explicit marginal PIT, observation dimension, conditional CDF, and Rosenblatt capabilities exist; coverage varies by family. |

### Mixture models

| Area                          | Status | What it means for modeling                                                                                    |
| ----------------------------- | ------ | ------------------------------------------------------------------------------------------------------------- |
| Finite mixture families       | ✅     | Homogeneous fixed-size mixtures compose any `CompilableFamily`, including multivariate Normal components. |
| Component-specific predictors | ✅     | `Repeated<ComponentShape, C>` gives each component its own typed predictors and penalties; `Broadcast` expresses an explicit shared owner. |
| Mixing weights                | ✅     | Baseline-softmax weight predictors support covariate-dependent gating and analytical responsibility gradients. |

### Bayesian GAMLSS

| Area                              | Status | What it means for modeling                                                                                           |
| --------------------------------- | ------ | -------------------------------------------------------------------------------------------------------------------- |
| Priors over distributional models | ✅/🧩  | `gamlss-bayes` provides a proper normalized diagonal Gaussian prior on predictor coefficients; transformed and smoothing priors remain explicit future work. |
| Posterior inference              | 🧩     | `PosteriorPotential` exposes summed unpenalized likelihood plus analytical prior gradient to external HMC/VI backends; no sampler is bundled. |
| Posterior predictive diagnostics  | 🧩     | Pointwise log-likelihood and fallible simulation foundations are available; chain-level summaries remain backend work. |

Fitting loops and optimizer integrations are intentionally outside the core API today; provide or adapt an optimizer against the objective and gradient traits.
