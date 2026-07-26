# Spline hot-path benchmarks

Run the complete suite with:

```bash
cargo bench -p gamlss-spline --bench spline_hot_paths
```

Use a Criterion filter for a focused comparison, for example:

```bash
cargo bench -p gamlss-spline --bench spline_hot_paths -- "open_uniform_cubic/n100000/k64"
```

The suite compares compact prepared, on-demand, and equivalent dense spline predictors. Open-uniform cubic cases measure row prediction, predictor VJP and weighted Gram operations for `n = 1_000/100_000` and `k = 16/64`, plus local versus full-buffer basis evaluation on moderate cases. Focused `n = 100_000, k = 16` cases cover open-uniform quadratic and cyclic cubic row prediction and VJP hot paths.

Representative `n = 10_000` cases cover the other materially different execution strategies without multiplying every size/order combination: compact local M-splines at `k = 64`, prefix-compressed I-splines at `k = 16`, and prepared dense natural cubic geometry at `k = 32`. Each compares prepared, on-demand, and dense `eta`, VJP, and weighted Gram passes. The open-uniform groups also compare the specialized basis kernel with the general-knot B-spline evaluator.

Prepared, unprepared, and separated-kernel second-difference penalties are measured for `k = 16/64/256`. Inputs and reusable output buffers are created outside timed iterations; output clearing remains inside because all compared additive operations require it in normal use.

The `duchon_thin_plate/d2/n1000/centers49/k20` group separates out-of-sample radial basis evaluation from prepared prediction and VJP passes, and records the linear-time diagonal penalty value and gradient paths. Spectral setup is deliberately outside timed iterations because it is fitted metadata construction rather than an optimizer hot path.

Treat results as local comparison data rather than portable performance guarantees or CI thresholds.
