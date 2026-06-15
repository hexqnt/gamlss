# gamlss-diagnostics

Post-fit diagnostics utilities for Rust-native GAMLSS models.

This crate provides post-fit diagnostics without expanding the base
`gamlss_core::Family` trait. The current API starts with
`CdfDiagnosticsExt::pit_values` for training-row PIT/CDF values and
`CdfDiagnosticsExt::quantile_residuals` for normalized quantile residuals.

```rust
use gamlss::prelude::*;

# fn run<M>(model: &M, theta: &[f64]) -> Result<(), gamlss::core::ModelError>
# where
#     M: CdfDiagnosticsExt,
# {
let pit = model.pit_values(theta)?;
let residuals = model.quantile_residuals(theta)?;
# let _ = (pit, residuals);
# Ok(())
# }
```

Residuals, worm plot data, centile curve data, fitted parameter extraction and
distribution-level prediction utilities will be added here as extension APIs.
