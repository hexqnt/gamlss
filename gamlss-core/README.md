# gamlss-core

Low-level type-driven core abstractions for GAMLSS-style modeling in Rust.

> **Status:** This crate is an early MVP. It is not intended for production use
> yet. The public API, internals, numerical behavior, and crate structure may
> change substantially while the library is being developed.

Most users should depend on the top-level `gamlss` crate instead:

```toml
[dependencies]
gamlss = "0.1"
```

This crate contains the lightweight core building blocks: link functions,
parameter blocks, predictor contracts, objectives, compiled model evaluation,
and typed family/model interfaces.

Use `gamlss-core` directly only when building low-level extensions or
integration crates. It intentionally does not depend on optimizers, dataframe
libraries, or heavy matrix backends.
