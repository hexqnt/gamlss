# gamlss-family

Distribution families, likelihoods, and score helpers for GAMLSS-style modeling in Rust.

> **Status:** Actively developed. Public API, internals, numerical behavior, and crate structure may still change before a stable 1.0 release.

This crate contains the distribution-specific layer used by typed GAMLSS models. It provides family definitions, likelihood evaluation, analytical score helpers, and distribution utilities that are shared by higher-level workspace crates.

Use it when composing lower-level models, working directly with distribution families, or adding new family integrations.
