# gamlss-family

Distribution families, likelihoods, and score helpers for GAMLSS-style modeling
in Rust.

> **Status:** This crate is an early MVP. It is not intended for production use
> yet. The public API, internals, numerical behavior, and crate structure may
> change substantially while the library is being developed.

Most users should depend on the top-level `gamlss` crate instead:

```toml
[dependencies]
gamlss = "0.1"
```

This crate provides distribution-specific pieces used by typed GAMLSS models,
including likelihood and score implementations for supported families.

Use `gamlss-family` directly only when composing lower-level models or adding
new family integrations. In ordinary applications it is pulled in transitively
by `gamlss`.
