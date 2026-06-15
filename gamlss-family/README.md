# gamlss-family

Distribution families, likelihoods, and score helpers for GAMLSS-style modeling
in Rust.

> **Status:** This crate is an early MVP. It is not intended for production use
> yet. The public API, internals, numerical behavior, and crate structure may
> change substantially while the library is being developed.

This crate contains distribution-specific building blocks used by typed GAMLSS
models: family types, likelihoods, scores, and related helpers.

Use it when composing lower-level models or adding new family integrations.
