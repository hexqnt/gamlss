# Family capabilities

Legend: ✓ implemented, ✗ not implemented.

`PDF` means the public density/mass helper exposed through `HasDensity`.
This is available for every `Family` through the blanket `HasLogDensity` /
`HasDensity` implementations in `gamlss-core`.
`Sampling` requires the `rand` feature and uses the fallible `TrySimulate` contract. Sampling failures are returned as `SimulationError`; invalid sentinel values such as `NaN` are not part of the simulation API. `try_sample_into` generates into one caller-owned sample, `try_fill` fills a caller-owned slice for one parameter value, and `try_fill_varying` generates one sample for each element of a parameter slice. In the facade crate this is exposed as `gamlss/rand`, which forwards to `gamlss-family/rand`.

`Gradient kind` describes the implementation used by
`Family::nll_and_gradient_eta`. `finite-diff` families are intentionally marked
as slow path because each observation requires roughly `2K + 1` likelihood
evaluations for `K` parameters.

| Family             | NLL | Gradient | Gradient kind | CDF | PDF | Quantile | CRPS | Sampling |
| ------------------ | --- | -------- | ------------- | --- | --- | -------- | ---- | -------- |
| `Beinf`            | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✗    | ✓        |
| `Bernoulli`        | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Beta`             | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Exponential`      | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Gamma`            | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `GeneralizedGamma` | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✗    | ✓        |
| `Gev`              | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✗    | ✓        |
| `Gumbel`           | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `InverseGaussian`  | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `JohnsonSu`        | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✗    | ✓        |
| `Laplace`          | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `LogNormal`        | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Logistic`         | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Lomax`            | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✗    | ✓        |
| `NegativeBinomial` | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✗    | ✓        |
| `Normal`           | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Poisson`          | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `PowerExponential` | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✗    | ✓        |
| `Shash`            | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✗    | ✓        |
| `SkewNormal`       | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✗    | ✓        |
| `SkewStudentT`     | ✓   | ✓        | finite-diff   | ✓   | ✓   | ✓        | ✗    | ✓        |
| `StudentT`         | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Tweedie`          | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✗    | ✓        |
| `Weibull`          | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Zaga`             | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✗    | ✓        |
| `Zinb`             | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✗    | ✓        |
| `Zip`              | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✗    | ✓        |

## Multivariate families

Multivariate families are enabled with the `multivariate` feature. They are
named by construction: product, elliptical, or simplex. Dimension-specific
public aliases are intentionally deferred until the const-generics
specialization policy is settled.

| Family                                        | Fit-ready               | Gradient          | Marginal CDF        | Conditional/Rosenblatt | Sampling                |
| --------------------------------------------- | ----------------------- | ----------------- | ------------------- | ---------------------- | ----------------------- |
| `IndependentVec`                              | ✓ static                | component         | ✓                   | ✗                      | ✓ composition           |
| `MvNormalCholesky`                            | ✓ static                | analytic          | ✓                   | ✓                      | ✓                       |
| `MvNormalMeanStdPartialCorr`                  | ✓ static                | analytic          | ✓                   | ✗                      | ✓                       |
| `DynMvNormalCholesky`                         | ✓ runtime               | analytic in-place | ✓                   | ✓                      | ✓ allocation reuse      |
| `MvStudentTCholesky`                          | ✓ static                | analytic          | ✓                   | ✗                      | ✓                       |
| `MvStudentTMeanStdPartialCorr`                | ✓ static                | analytic          | ✓                   | ✗                      | ✓                       |
| `MvSkewNormalCholesky`                        | ✓ static                | analytic          | ✗                   | ✗                      | ✓                       |
| `MvSkewNormalLocationKernelStdPartialCorr`    | ✓ static                | analytic          | ✗                   | ✗                      | ✓                       |
| `MvSkewStudentTFixedTauCholesky`              | ✓ static                | analytic          | ✗                   | ✗                      | ✓                       |
| `MvPowerExponentialCholesky`                  | ✓ static                | analytic          | ✗                   | ✗                      | ✓                       |
| `MvPowerExponentialMeanStdPartialCorr`        | ✓ static                | analytic          | ✗                   | ✗                      | ✓                       |
| `MvPoissonCommonShock`                        | ✓ static                | analytic          | ✓                   | ✗                      | ✓                       |
| `DirichletMeanPrecision`                      | ✓ static                | analytic          | ✗                   | ✗                      | ✓                       |
| `Mixture<F, C>`                               | ✓ when `F` is fit-ready | responsibilities  | component-dependent | component-dependent    | ✓ when `F: TrySimulate` |

Application-oriented selection notes and the deferred-family queue are documented in [the multivariate family roadmap](multivariate-family-roadmap.md).

## Analytic gradient replacement queue

Finite-difference families are supported but slower training paths. They use numerical eta-gradient approximations and should be replaced with analytic gradients as usage and benchmark results justify it. The remaining univariate slow path is `SkewStudentT`: its location-scale parameterization is analytic for location, scale, and skewness, but the degrees-of-freedom component still uses finite differences because it requires differentiating the Student-t CDF with respect to degrees of freedom. The mean/SD parameterization currently uses a full finite-difference eta-gradient.
