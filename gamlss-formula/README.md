# gamlss-formula

Optional formula and builder layer for compiling dynamic model specifications
into typed GAMLSS models.

> **Status:** This crate is an early MVP. It is not intended for production use
> yet. The public API, internals, numerical behavior, and crate structure may
> change substantially while the library is being developed.

Most users should depend on the top-level `gamlss` crate instead:

```toml
[dependencies]
gamlss = "0.1"
```

This crate is the dynamic boundary layer of the workspace. It is intended to
parse or construct model specifications and compile them into the typed
lower-level model representation.

Use `gamlss-formula` directly only when integrating custom formula parsing,
builder APIs, or higher-level model construction. In ordinary applications it is
available through the default `formula` feature of `gamlss`.
