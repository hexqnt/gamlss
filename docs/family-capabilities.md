# Family capabilities

Legend: ✓ implemented, ✗ not implemented.

`PDF` means the public density/mass helper exposed through `HasDensity`.
`Sampling` requires the `rand` feature.

| Family             | NLL | Gradient | CDF | PDF | Quantile | CRPS | Sampling |
| ------------------ | --- | -------- | --- | --- | -------- | ---- | -------- |
| `Bernoulli`        | ✓   | ✓        | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Beta`             | ✓   | ✓        | ✓   | ✓   | ✓        | ✗    | ✓        |
| `Exponential`      | ✓   | ✓        | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Gamma`            | ✓   | ✓        | ✓   | ✓   | ✓        | ✗    | ✓        |
| `Gumbel`           | ✓   | ✓        | ✓   | ✓   | ✓        | ✗    | ✓        |
| `InverseGaussian`  | ✓   | ✓        | ✓   | ✓   | ✓        | ✗    | ✓        |
| `Laplace`          | ✓   | ✓        | ✓   | ✓   | ✓        | ✓    | ✓        |
| `LogNormal`        | ✓   | ✓        | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Logistic`         | ✓   | ✓        | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Lomax`            | ✓   | ✓        | ✓   | ✓   | ✓        | ✗    | ✓        |
| `NegativeBinomial` | ✓   | ✓        | ✓   | ✓   | ✓        | ✗    | ✓        |
| `Normal`           | ✓   | ✓        | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Poisson`          | ✓   | ✓        | ✓   | ✓   | ✓        | ✓    | ✓        |
| `StudentT`         | ✓   | ✓        | ✓   | ✓   | ✓        | ✗    | ✓        |
| `Weibull`          | ✓   | ✓        | ✓   | ✓   | ✓        | ✓    | ✓        |
