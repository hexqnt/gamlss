# gamlss-formula

Optional formula and builder layer for compiling dynamic model specifications
into typed GAMLSS models.

> **Status:** This crate is an early MVP. It is not intended for production use
> yet. The public API, internals, numerical behavior, and crate structure may
> change substantially while the library is being developed.

This crate is the dynamic boundary layer of the workspace. It builds typed
models from runtime model specifications and simple column-oriented data.

Use it when integrating formula parsing, builder-style model construction, or
higher-level modeling interfaces on top of the typed core.
