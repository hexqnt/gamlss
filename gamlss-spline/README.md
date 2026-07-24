# gamlss-spline

Spline bases and smoothness penalties for typed GAMLSS models in Rust.

> **Status:** Actively developed. Public API, internals, numerical behavior, and crate structure may still change before a stable 1.0 release.

## Prepared and on-demand designs

Prepared designs cache row geometry for repeated evaluation. On-demand designs retain only coordinates and basis metadata, reducing memory at the cost of computation. For one-shot streams, use `SplineBasis1d::for_each_basis` directly.
