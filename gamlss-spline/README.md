# gamlss-spline

Spline bases and smoothness penalties for typed GAMLSS models in Rust.

> **Status:** Actively developed. Public API, internals, numerical behavior, and crate structure may still change before a stable 1.0 release.

## Design representations

Reusable `*Basis` and `*Spec` values hold fitted metadata and can build prepared or on-demand designs. Prepared designs cache row geometry for repeated evaluation; on-demand designs retain coordinates and recompute rows to save memory. Both implement the same spline and predictor traits. For one-shot streams, use `SplineBasis1d::for_each_basis` directly.

Most bases share common prepared or on-demand engines; I-spline and natural-cubic designs keep specialized geometry. `TensorSplineDesign` composes two existing designs row by row.
