# gamlss-family

Distribution families, likelihoods, and score helpers for GAMLSS-style modeling
in Rust.

> **Status:** Actively developed. Public API, internals, numerical behavior, and
> crate structure may still change before a stable 1.0 release.

This crate contains distribution-specific building blocks used by typed GAMLSS
models: family types, likelihoods, scores, and related helpers.

The public module tree separates scalar-response and vector-response families:

* `univariate` contains scalar distribution families and their
  parameterizations.
* `multivariate` contains vector-valued distribution families and supporting
  parameter storage helpers. It is enabled with the `multivariate` feature.

Use it when composing lower-level models or adding new family integrations.
