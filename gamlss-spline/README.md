# gamlss-spline

Spline bases, penalties, and spline metadata for GAMLSS-style modeling in Rust.

> **Status:** This crate is an early MVP. It is not intended for production use
> yet. The public API, internals, numerical behavior, and crate structure may
> change substantially while the library is being developed.

This crate contains spline design components and smoothness penalties used by
typed GAMLSS parameter blocks.

Use it when building smooth or periodic predictor terms and attaching spline
penalties to typed parameter blocks.
