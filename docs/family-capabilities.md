# Family capabilities

Legend: ✓ implemented, ✗ not implemented.

`PDF` means the public density/mass helper exposed through `HasDensity`.
This is available for every `Family` through the blanket `HasLogDensity` /
`HasDensity` implementations in `gamlss-core`.
`Sampling` requires the `rand` feature. In the facade crate this is exposed as
`gamlss/rand`, which forwards to `gamlss-family/rand`.

| Family                | NLL | Gradient | CDF | PDF | Quantile | CRPS | Sampling |
| --------------------- | --- | -------- | --- | --- | -------- | ---- | -------- |
| `Beinf`               | ✓   | ✓        | ✓   | ✓   | ✓        | ✗    | ✗        |
| `Bernoulli`           | ✓   | ✓        | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Beta`                | ✓   | ✓        | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Exponential`         | ✓   | ✓        | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Gamma`               | ✓   | ✓        | ✓   | ✓   | ✓        | ✓    | ✓        |
| `GeneralizedGamma`    | ✓   | ✓        | ✓   | ✓   | ✓        | ✗    | ✗        |
| `Gev`                 | ✓   | ✓        | ✓   | ✓   | ✓        | ✗    | ✗        |
| `Gumbel`              | ✓   | ✓        | ✓   | ✓   | ✓        | ✗    | ✓        |
| `InverseGaussian`     | ✓   | ✓        | ✓   | ✓   | ✓        | ✓    | ✓        |
| `JohnsonSu`           | ✓   | ✓        | ✓   | ✓   | ✓        | ✗    | ✗        |
| `Laplace`             | ✓   | ✓        | ✓   | ✓   | ✓        | ✓    | ✓        |
| `LogNormal`           | ✓   | ✓        | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Logistic`            | ✓   | ✓        | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Lomax`               | ✓   | ✓        | ✓   | ✓   | ✓        | ✗    | ✓        |
| `NegativeBinomial`    | ✓   | ✓        | ✓   | ✓   | ✓        | ✗    | ✓        |
| `Normal`              | ✓   | ✓        | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Poisson`             | ✓   | ✓        | ✓   | ✓   | ✓        | ✓    | ✓        |
| `PowerExponential`    | ✓   | ✓        | ✓   | ✓   | ✓        | ✗    | ✗        |
| `Shash`               | ✓   | ✓        | ✓   | ✓   | ✓        | ✗    | ✗        |
| `SkewNormal`          | ✓   | ✓        | ✓   | ✓   | ✓        | ✗    | ✗        |
| `SkewStudentT`        | ✓   | ✓        | ✓   | ✓   | ✓        | ✗    | ✗        |
| `StudentT`            | ✓   | ✓        | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Tweedie`             | ✓   | ✓        | ✓   | ✓   | ✓        | ✗    | ✗        |
| `Weibull`             | ✓   | ✓        | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Zaga`                | ✓   | ✓        | ✓   | ✓   | ✓        | ✗    | ✗        |
| `Zinb`                | ✓   | ✓        | ✓   | ✓   | ✓        | ✗    | ✗        |
| `Zip`                 | ✓   | ✓        | ✓   | ✓   | ✓        | ✗    | ✗        |
