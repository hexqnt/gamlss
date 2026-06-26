# gamlss

[![CI](https://github.com/hexqnt/gamlss/actions/workflows/ci.yml/badge.svg)](https://github.com/hexqnt/gamlss/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/gamlss.svg)](https://crates.io/crates/gamlss)[![docs.rs](https://img.shields.io/docsrs/gamlss)](https://docs.rs/gamlss)

Type-driven Rust crates for GAMLSS-style modeling.

> **Status:** Actively developed. Public API and numerical behavior may still
> change before a stable 1.0 release.

## Crate layout

The primary entry point for users is the `gamlss` crate:

`gamlss` is the batteries-included facade. The main and most stable path today
goes through the low-level typed API exposed by `gamlss-core`, `gamlss-family`,
`gamlss-spline`, `gamlss-special`, and `gamlss-transform`. By default, the
facade also re-exports `gamlss-formula`; that layer is currently an
experimental optional convenience crate, not the library's core API.

The workspace also publishes separate crates for finer control over APIs and
dependencies:

- `gamlss-core` — type-driven core abstractions for links, parameter blocks,
  objectives, and compiled models.
- `gamlss-family` — distributions, likelihoods, and score helpers.
- `gamlss-diagnostics` — post-fit PIT/CDF diagnostics, normalized quantile
  residuals, and CRPS summaries for supported families.
- `gamlss-special` — special functions and shared numerical helpers for
  likelihood, CDF, and quantile code.
- `gamlss-spline` — spline/Fourier predictors, penalties, and spline metadata.
- `gamlss-transform` — target preprocessing transforms.
- `gamlss-formula` — experimental optional formula/builder layer that compiles
  runtime specifications into typed models. It covers curated high-level
  workflows and is not expected to mirror every family, link, or
  parameterization available in the low-level crates.

For most use cases, depending on `gamlss` is enough; the other crates are pulled
in transitively. If you want a stricter low-level surface without the
experimental formula layer, use `default-features = false` or depend on the
individual crates directly.

## Cargo features

- `formula` is enabled by default and re-exports the experimental
  `gamlss-formula` namespace from the facade crate.
- `rand` enables the sampling API in `gamlss-family` through the facade crate:
  `rand = ["gamlss-family/rand"]`.

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

GAMLSS can be read as distributional regression: the model describes not only
the conditional mean of the response, but the full conditional distribution.
This is useful when dispersion, skewness, tail behavior, or even the valid
response domain changes with the features. For example, one part of a model may
describe the distribution center, another a heteroskedastic scale, and a third
the tail shape.

In general, GAMLSS defines the response distribution through a set of parameters
from a chosen family:

$$
Y_i \mid x_i \sim D(\theta_{i1}, \ldots, \theta_{iK}),
$$

where `D(...)` is the selected parametric distribution. Each parameter is modeled
with its own link function and predictor:

$$
g_k(\theta_{ik}) = \eta_{ik}
  = X_{k,i}\beta_k + \sum_j f_{k,j}(x_i),
\qquad k = 1,\ldots,K.
$$

In other words, different parameters of the same distribution can use different
feature sets, spline terms, penalties, and domain constraints. A link function
maps the unconstrained linear predictor `eta` into the parameter's valid domain:
scale parameters usually need to be positive, while probability or mean
parameters for a beta family must lie inside `(0, 1)`.

The classic `gamlss` convention often names the first four parameters `mu`,
`sigma`, `nu`, and `tau`:

$$
(\theta_{i1}, \theta_{i2}, \theta_{i3}, \theta_{i4})
  = (\mu_i, \sigma_i, \nu_i, \tau_i),
$$

$$
Y_i \mid x_i \sim D(\mu_i, \sigma_i, \nu_i, \tau_i).
$$

Here `mu`, `sigma`, `nu`, and `tau` usually correspond to location, scale,
skewness, and shape. This is a naming convention, not a required API shape: not
every family uses all four parameters, and the typed core supports custom
parameter markers for application-specific parameter counts and meanings. As a
result, the library can express familiar location-scale models such as normal,
log-normal, Laplace, and Student's t, as well as differently parameterized
families such as gamma, Weibull, inverse Gaussian, and beta.
