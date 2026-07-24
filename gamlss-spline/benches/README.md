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

Prepared and unprepared second-difference penalties are measured separately for `k = 16/64/256`. Inputs and reusable output buffers are created outside timed iterations; output clearing remains inside because both compared additive operations require it in normal use.

Treat results as local comparison data rather than portable performance guarantees or CI thresholds.
