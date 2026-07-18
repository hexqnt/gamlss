# gamlss-diagnostics

Post-fit diagnostics utilities for Rust-native GAMLSS models.

This crate provides post-fit diagnostics without expanding the base
`gamlss_core::Family` trait. The current API includes:

- PIT/CDF values and normalized quantile residuals;
- row-major ordered Rosenblatt values for multivariate families;
- per-row, mean, and observation-weighted CRPS summaries;
- reusable diagnostics views for compatible prediction blocks and observations.

```rust,ignore
let pit = model.pit_values(parameters)?;
let residuals = model.quantile_residuals(parameters)?;
let mean_crps = model.mean_crps(parameters)?;
let rosenblatt = multivariate_model.rosenblatt_values(parameters)?;
```

Randomized residuals, worm plot data, centile curve data, fitted parameter
extraction, and additional distribution-level prediction utilities remain
planned as extension APIs.
