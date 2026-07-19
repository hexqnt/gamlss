# Objective benchmarks

Run the full suite or filter a Criterion benchmark after `--`:

```bash
cargo bench -p gamlss-family --bench objective --all-features
cargo bench -p gamlss-family --bench objective --all-features -- "mvn_cholesky_dynamic/d32"
```

Setup and allocations are excluded from timed iterations. Treat results as local baselines and compare runs under the same environment.

## Allocation and memory profiling

Build the dynamic-MVN profiler with release debug information:

```bash
CARGO_PROFILE_RELEASE_DEBUG=1 cargo build --release -p gamlss-family --example objective_memory --features multivariate
```

Arguments are `[dimension] [nobs] [ncols] [warm|cold|retained|run] [iterations]` (defaults: `8 1000 8 warm 10`). DHAT modes write `dhat-objective-*.json`:

```bash
target/release/examples/objective_memory 32 10000 32 warm 20
target/release/examples/objective_memory 32 10000 32 cold 5
target/release/examples/objective_memory 32 10000 32 retained 20
```

Use `run` with external RSS or cache tools:

```bash
/usr/bin/time -v target/release/examples/objective_memory 32 10000 32 run 20
perf stat -r 5 -e cycles,instructions,cache-references,cache-misses target/release/examples/objective_memory 32 10000 32 run 20
```
