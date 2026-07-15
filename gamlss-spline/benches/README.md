# Spline hot-path benchmarks

Run the complete suite with:

```bash
cargo bench -p gamlss-spline --bench spline_hot_paths
```

Use a Criterion filter for a focused comparison, for example:

```bash
cargo bench -p gamlss-spline --bench spline_hot_paths -- "open_uniform_cubic/n100000/k64"
```

The suite compares the allocation-free local open-uniform cubic spline predictor with an equivalent dense design materialized before timing. It measures row prediction, predictor VJP and weighted Gram operations for `n = 1_000/100_000` and `k = 16/64`, plus local versus full-buffer basis evaluation on moderate cases.

Prepared and unprepared second-difference penalties are measured separately for `k = 16/64/256`. Inputs and reusable output buffers are created outside timed iterations; output clearing remains inside because both compared additive operations require it in normal use.

Treat results as local comparison data rather than portable performance guarantees or CI thresholds.
