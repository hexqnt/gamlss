# gamlss-spline

Spline bases and smoothness penalties for typed GAMLSS models in Rust.

> **Status:** Actively developed. Public API, internals, numerical behavior, and crate structure may still change before a stable 1.0 release.

## Basis and design representations

Reusable `*Basis` and `*Spec` values hold fitted metadata and can build prepared or on-demand designs. Prepared designs cache row geometry for repeated evaluation; on-demand designs retain coordinates and recompute rows to save memory. Both implement the same spline and predictor traits. For one-shot streams, use `SplineBasis1d::for_each_basis` directly. Fixed-dimensional multivariate bases use the const-generic `SplineBasisNd<D>` contract.

Derived bases use composition rather than copying their source representation: an M-spline owns its B-spline metadata, an I-spline owns its M-spline, and a C-spline owns its I-spline plus prepared integration prefixes. Dense constrained and Duchon designs delegate their linear algebra to `gamlss_core::DenseDesign`; specialized local-row implementations remain only where compact geometry changes storage or asymptotic work.

`DifferentiableSplineBasis1d` provides allocation-free derivative rows for B-, M-, I-, C-, natural-cubic, truncated-power, open-uniform, cyclic, periodic, and Fourier bases. `DifferentiableSplineRowBasis` carries the same capability into indexed designs, including partial derivatives of tensor products.

`OpenKnotVector` records the exact fitted clamped knot sequence together with `Uniform`, `Quantile`, `WeightedQuantile`, or `Explicit` placement metadata. B-, M-, I-, C-, and truncated-power constructors can reuse this state, so prediction never derives knots from new observations.

## Constraints and interactions

`CenteredSplineDesign` applies an empirical weighted sum-to-zero constraint. `HelmertContrastDesign` prepares an orthonormal constant-free coefficient transform, and `TensorInteractionDesign` tensors two such margins so the result cannot reproduce either marginal main effect. `TensorSplineDesign` remains the unconstrained row-wise Kronecker product and exposes coordinate-wise predictor derivatives.

`MonotoneISplineDesign` and `ConvexCSplineDesign` are structural nonlinear parameterizations: softplus-mapped coefficients enforce monotonicity or the requested `Convex`/`Concave` curvature for every unconstrained optimizer parameter vector. `CSplineBasis` caches prefix integrals of I-splines, so this specialization enforces curvature and avoids repeated integration over completed knot intervals; it does not claim a new unconstrained spline space.

## Smoothness penalties

`PenaltyKernel` separates reusable unscaled geometry from `ScaledPenalty<K>` and its smoothing parameter. Difference, dense, diagonal, symmetric-band, and low-rank internal kernels share the same value, gradient, and matrix implementation path; derivatives with respect to `log(lambda)` are exact. `WeightedDifferencePenalty` varies finite-difference strength by location, while `AdaptiveDifferencePenalty` combines several independently scaled spatial components without introducing another smoothing basis.

The crate also supports exact function-space roughness penalties. `BSplineDerivativePenalty` integrates an arbitrary positive derivative order over the conventional domain of an arbitrary-knot B-spline; `NaturalCubicRoughnessPenalty` integrates squared curvature using the cardinal basis' exact second-derivative operator; `FourierRoughnessPenalty` uses the exact diagonal harmonic spectrum over one period. `NullSpacePenalty` adds low-rank shrinkage only to unpenalized polynomial or intercept directions.

`TensorProductPenalty` combines two constant-curvature marginal penalties as `H_left ⊗ I + I ⊗ H_right`, retaining independent weights and orders without allocating during value, gradient, or matrix evaluation. `HelmertContrastPenalty` applies the same orthonormal parameterization as the matching design transform.

## Duchon regression splines

`DuchonSplineBasis<D>` adds a genuinely different isotropic multivariate space based on Euclidean radial semi-kernels and a deterministic low-rank spectral reduction. `DuchonSmoothness` stores the frequency exponent exactly as integer `2s`; all admissible integer and half-integer members can share the implementation. Thin-plate regression splines are exactly the `s = 0` member, so there is deliberately no duplicate thin-plate basis type. The prepared spectral parameterization diagonalizes the penalty, making penalty value and gradient operations linear in the requested rank.

This is not a private case of `TensorSplineDesign`: tensor products are tied to selected coordinate axes and marginal knot spaces, while the Duchon kernel depends only on Euclidean distance and is invariant to rotations. Construction uses a backend-free full symmetric eigendecomposition; for large datasets callers should pass an intentional representative subset of centers rather than all observations.

## Basis audit decisions

- A separate restricted-cubic basis is not included because `NaturalCubicSplineBasis` already spans that function space. A future convenience API should be an alias, constructor, or demonstrably faster facade.
- Monotone, convex, concave, centered, and pure-interaction forms remain design/constraint types over existing spaces; their separate types are justified by structural guarantees or prepared faster evaluation, not by claiming new unconstrained bases.
- Adaptive smoothing remains in the penalty layer because it changes local roughness weights rather than the B-spline columns.
- A separate thin-plate type would duplicate `DuchonSplineBasis<D>` at `s = 0` and should only be added if a benchmarked closed-form setup path is materially faster.
- `OpenUniformSplineBasis` remains a specialization of a general B-spline because its implicit knots and bounded local rows avoid storing or searching an arbitrary knot vector. Likewise, prepared and on-demand designs intentionally trade cached row work for memory rather than representing different spline spaces.
- `DifferencePenalty` and `PreparedDifferencePenalty` retain dimension-late compatibility APIs. `DifferencePenaltyKernel` fixes the dimension at construction so it can expose rank, nullity, bandwidth, and reusable scale-free geometry; the fixed and weighted kernels share one internal difference operator rather than parallel stencil implementations.
