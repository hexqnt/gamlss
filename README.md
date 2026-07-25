# gamlss

[![CI](https://github.com/hexqnt/gamlss/actions/workflows/ci.yml/badge.svg)](https://github.com/hexqnt/gamlss/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/gamlss.svg)](https://crates.io/crates/gamlss)
[![docs.rs](https://img.shields.io/docsrs/gamlss)](https://docs.rs/gamlss)

Type-driven Rust crates for GAMLSS-style distributional regression.

> **Status:** Actively developed. Public APIs and numerical behavior may change before 1.0.

## Overview

GAMLSS models the full conditional response distribution rather than only its mean:

$$
Y_i \mid x_i \sim D(\theta_{i1}, \ldots, \theta_{iK}),
\qquad g_k(\theta_{ik}) = X_{k,i}\beta_k + \sum_j f_{k,j}(x_i).
$$

Each distribution parameter can have its own link, covariates, smooth terms, and penalties. The typed API expresses parameter domains and model structure at compile time while keeping model evaluation backend- and optimizer-agnostic.

The library supports scalar distributional models, optional multivariate families, finite mixtures, smooth predictors, target transforms, post-fit diagnostics, and a lightweight Bayesian posterior-potential layer. Distribution functions, quantiles, CRPS, and sampling are exposed through capability traits and vary by family. See the [family capability matrix](docs/family-capabilities.md) for exact coverage.

Fitting loops and optimizer integrations intentionally remain outside the core API.

## Crates

Most users should depend on the `gamlss` facade. The workspace also publishes focused crates:

- `gamlss-core` and `gamlss-family` provide typed model abstractions, distributions, likelihoods, and scores.
- `gamlss-spline` and `gamlss-special` provide predictors, penalties, special functions, and numerical helpers.
- `gamlss-transform` and `gamlss-diagnostics` cover response preprocessing and post-fit diagnostics.
- `gamlss-formula` is an experimental builder layer for curated workflows; `gamlss-bayes` provides priors and posterior potentials without a sampler.

See [project structure](docs/project-structure.md) for API layers and crate boundaries.

## Features

- `formula` (default) re-exports the experimental builder API.
- `bayes` enables `gamlss::bayes` and its common prelude types.
- `rand` enables sampling for supported families.
- `multivariate` enables multivariate families.

Use `default-features = false` for the facade without the formula layer, or depend on individual crates for tighter dependency control.

## Development

Run the workspace checks with:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features
cargo test --workspace --all-features
```

Run the main benchmarks with:

```bash
cargo bench -p gamlss-family --bench objective --all-features
cargo bench -p gamlss-spline --bench spline_hot_paths
cargo bench -p gamlss-special --bench numeric_kernels
```

Each crate's `benches/README.md` documents focused Criterion filters and coverage. Compare benchmark results only on the same hardware and toolchain.
