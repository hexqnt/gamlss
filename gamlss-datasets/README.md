# gamlss-datasets

Built-in and synthetic datasets for workspace examples. The crate is internal (`publish = false`).

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

Enable the `rand` feature for synthetic generators:

```rust
use gamlss_datasets::simulate::normal_linear;
use rand::{SeedableRng, rngs::StdRng};

let mut rng = StdRng::seed_from_u64(42);
let x = [0.0, 1.0, 2.0];
let y = normal_linear(&x, 1.0, 0.5, 0.3, &mut rng)?;
# Ok::<(), gamlss_datasets::simulate::GenerationError>(())
```
