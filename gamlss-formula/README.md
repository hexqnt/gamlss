# gamlss-formula

Experimental optional formula/builder layer for compiling user specifications
into typed `gamlss-core` models.

`gamlss-formula` is not the primary API of the project. The main direction is
the lower-level typed stack: `gamlss-core`, `gamlss-family`, `gamlss-spline`,
and `gamlss-transform`.

This crate currently provides:

- typed `DataView`/`Col<T>` inputs;
- curated term builders such as linear terms, factors, interactions and
  splines;
- selected default family specs: normal, gamma, log-normal, Weibull, inverse
  Gaussian and beta.

It intentionally does not provide string formula parsing, fit loops, optimizer
integration, diagnostics or dataframe adapters, and it is not expected to mirror
every low-level distribution, link or parameterization.
