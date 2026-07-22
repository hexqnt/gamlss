# gamlss-datasets

Built-in and synthetic datasets for GAMLSS examples and experiments.

CSV datasets are declared in `datasets.rs` and compiled into static slices. Loading performs no allocation or I/O.

```rust
datasets! {
    measurements {
        path: "data/measurements.csv",
        x: Date,
        y: [f64; 2],
    }
}
```

Supported `x` types: `f64`, `u8`, `u16`, `u32`, `u64`, `Date`, and `Time`. The `y` type is `f64` or `[f64; D]`. Multivariate responses are stored row-major as `&[[f64; D]]`. CSV starts with the `x` column followed by the `y` columns. Dates use `YYYY-MM-DD`; times use `HH:MM:SS[.fraction]`.


```rust
use gamlss_datasets::simulate::continuous::{multivariate, univariate};

let _scalar = univariate(10_000, 42)?;
let _multi = multivariate::<8>(1_000, 42)?;
# Ok::<(), gamlss_datasets::simulate::continuous::GenerationError>(())
```
