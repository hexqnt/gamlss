# gamlss-special

Special functions and small numerical helpers shared by GAMLSS workspace crates.

The crate intentionally has no production dependencies. It provides scalar
`f64` functions used by distribution likelihoods, gradients, CDFs and quantile
helpers without introducing runtime dispatch.
