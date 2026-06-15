# gamlss-core

Low-level type-driven core abstractions for GAMLSS-style modeling in Rust.

> **Status:** This crate is an early MVP. It is not intended for production use
> yet. The public API, internals, numerical behavior, and crate structure may
> change substantially while the library is being developed.

This crate contains the lightweight core: links, parameter blocks, predictor
contracts, objectives, compiled model evaluation, observation views, prediction
helpers, and typed family/model interfaces.

Use it when building low-level extensions, custom families, custom predictor
blocks, or optimizer integrations. It intentionally stays independent of
optimizers, dataframe libraries, and heavy matrix backends.
