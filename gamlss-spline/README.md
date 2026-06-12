# gamlss-spline

Spline bases, penalties, and spline metadata for GAMLSS-style modeling in Rust.

> **Status:** This crate is an early MVP. It is not intended for production use
> yet. The public API, internals, numerical behavior, and crate structure may
> change substantially while the library is being developed.

Most users should depend on the top-level `gamlss` crate instead:

```toml
[dependencies]
gamlss = "0.1"
```

This crate contains spline design components and smoothness penalties used by
typed GAMLSS parameter blocks.

Use `gamlss-spline` directly only when building lower-level model components or
custom spline integrations. In ordinary applications it is pulled in
transitively by `gamlss`.
