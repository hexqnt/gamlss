# gamlss-datasets

Built-in and synthetic datasets for GAMLSS examples and experiments.

| Dataset                           |  Rows | `x`                  | `y`                             |
| --------------------------------- | ----: | -------------------- | ------------------------------- |
| `a1`                              | 2,000 | continuous predictor | scalar response                 |
| `faithful`                        |   272 | observation ID       | eruption duration, waiting time |
| `cbr_inflation_and_interest_rate` |   154 | month                | key rate, annual inflation      |

The CBR dataset contains monthly percentages from September 2013 through June 2026, with each month encoded as its first calendar day. Its source is the Bank of Russia's [Inflation and Bank of Russia key rate](https://www.cbr.ru/hd_base/infl/) table; the key rate is reported for the last day of each month.

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
