# gamlss-datasets

Non-trivial built-in datasets and synthetic generators for the workspace examples and quick local experiments with `gamlss`.

The crate is internal to this repository (`publish = false`). Built-in data is authored in `data/*.csv` and compiled by `build.rs` into borrowed static slices, so loading performs no allocation or I/O. CSV values must be finite `f64` values with simple identifier column names; malformed data fails the build.

```rust
let data = gamlss_datasets::a1();
assert_eq!(data.x.len(), data.y.len());
```

Trivial synthetic datasets are generated from an explicit data-generating process instead of being stored as CSV fixtures.

Enable the `rand` feature for deterministic synthetic data generation:

```rust
use gamlss_datasets::simulate::normal_linear;
use rand::{SeedableRng, rngs::StdRng};

let mut rng = StdRng::seed_from_u64(42);
let x = [0.0, 1.0, 2.0];
let y = normal_linear(&x, 1.0, 0.5, 0.3, &mut rng)?;
# Ok::<(), gamlss_datasets::simulate::GenerationError>(())
```
