# Objective benchmarks

Run the full suite or pass a Criterion filter after `--`:

```bash
cargo bench -p gamlss-family --bench objective --all-features
cargo bench -p gamlss-family --bench objective --all-features -- "mvn_cholesky_dynamic/d32"
```

Setup and allocations happen outside timed iterations. Criterion reports elapsed time and throughput in observations per second.

Coverage includes ordinary and workspace-backed objectives, fused gradients, pointwise log-likelihood, eta/theta prediction, and Rosenblatt transforms for fixed/dynamic MVN. Scaling cases cover `n = 100_000`, dense `p = 64`, dimensions through `D = 32`, and a runtime-dimensional `D = 50` smoke case.

Results are local comparison baselines, not portable guarantees or CI thresholds. Compare runs with the same CPU, toolchain, profile, features, and filter.
