# gamlss-datasets

Small built-in datasets and synthetic generators for the workspace examples and quick local experiments with `gamlss`.

The crate is internal to this repository (`publish = false`). Built-in data is authored in `data/*.csv` and compiled by `build.rs` into borrowed static slices, so loading performs no allocation or I/O. CSV values must be finite `f64` values with simple identifier column names; malformed data fails the build.

```rust
let data = gamlss_datasets::linear_normal();
assert_eq!(data.x.len(), data.y.len());
```

`linear_normal` and `heteroscedastic_normal` are synthetic and may be freely used in examples and tests.

Enable the `rand` feature for deterministic synthetic data generation:

```rust
use gamlss_datasets::simulate::normal;
use rand::{SeedableRng, rngs::StdRng};

let mut rng = StdRng::seed_from_u64(42);
let x = [0.0, 1.0, 2.0];
let y = normal(&x, |x| 1.0 + 0.5 * x, |_| 0.3, &mut rng)?;
# Ok::<(), gamlss_datasets::simulate::GenerationError>(())
```
