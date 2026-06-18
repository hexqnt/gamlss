# gamlss-transform

Target transform layer for GAMLSS modeling.

> **Status:** This crate is an early MVP. It is not intended for production use
> yet. The public API, internals, numerical behavior, and crate structure may
> change substantially while the library is being developed.

This crate contains target preprocessing transforms with persisted state and
domain-aware inverse transforms.

Use it when composing lower-level preprocessing pipelines around typed GAMLSS
models.
