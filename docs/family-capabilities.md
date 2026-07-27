# Family capabilities

Legend: ✓ implemented, ✗ not implemented. Aliases inherit their underlying family's capabilities.

`PDF` is provided through `HasDensity`. `Sampling` uses fallible `TrySimulate` and requires the `rand` feature. `Gradient kind` refers to `Family::nll_and_gradient_eta`; finite differences require about `2K + 1` likelihood evaluations per observation.

## Univariate families

| Family                  | NLL | Gradient | Gradient kind | CDF | PDF | Quantile | CRPS | Sampling |
| ----------------------- | --- | -------- | ------------- | --- | --- | -------- | ---- | -------- |
| `Beinf`                 | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Bernoulli`             | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Beta`                  | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `BetaBinomial`          | ✓   | ✓        | analytic      | ✓   | ✓   | ✗        | ✓    | ✗        |
| `BinomialFixedTrials`   | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `BinomialVaryingTrials` | ✓   | ✓        | analytic      | ✓   | ✓   | ✗        | ✓    | ✗        |
| `Categorical`           | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Chi`                   | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `ChiSquared`            | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Exponential`           | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Gamma`                 | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `GeneralizedGamma`      | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `GeneralizedPareto`     | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Geometric`             | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Gev`                   | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Gumbel`                | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `InverseGaussian`       | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `JohnsonSu`             | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Laplace`               | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `LogLogistic`           | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `LogNormal`             | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Logistic`              | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Lomax`                 | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `NegativeBinomial`      | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Normal`                | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Poisson`               | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `PowerExponential`      | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Rayleigh`              | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Shash`                 | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `SkewNormal`            | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `SkewPowerExponential`  | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `SkewStudentT`          | ✓   | ✓        | finite-diff¹  | ✓   | ✓   | ✓        | ✓    | ✓        |
| `StudentT`              | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Tweedie`               | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Weibull`               | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Zaga`                  | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `ZeroAdjustedStudentT`  | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Zinb`                  | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |
| `Zip`                   | ✓   | ✓        | analytic      | ✓   | ✓   | ✓        | ✓    | ✓        |

¹ `SkewStudentT` uses a hybrid analytic/finite-difference gradient in its location-scale parameterization and a full finite-difference gradient in its mean/SD parameterization.

`BetaBinomial` and `BinomialVaryingTrials` need an observation-specific trial count, which their scalar quantile and sampling APIs cannot accept.

## Multivariate families

Requires the `multivariate` feature. CDF columns refer to the corresponding `gamlss-core` traits, not a joint multivariate CDF; sampling also requires `rand`.

| Family                                     | Fit-ready | Gradient          | Marginal CDF | Conditional CDF | Rosenblatt | Sampling           |
| ------------------------------------------ | --------- | ----------------- | ------------ | --------------- | ---------- | ------------------ |
| `IndependentVec`                           | ✓ static  | component         | ✓            | ✗               | ✗          | ✓ composition      |
| `MvNormalCholesky`                         | ✓ static  | analytic          | ✓            | ✓               | ✓          | ✓                  |
| `MvNormalMeanStdPartialCorr`               | ✓ static  | analytic          | ✓            | ✓               | ✓          | ✓                  |
| `DynMvNormalCholesky`                      | ✓ runtime | analytic in-place | ✓            | ✓               | ✓          | ✓ allocation reuse |
| `MvStudentTCholesky`                       | ✓ static  | analytic          | ✓            | ✗               | ✗          | ✓                  |
| `MvStudentTMeanStdPartialCorr`             | ✓ static  | analytic          | ✓            | ✗               | ✗          | ✓                  |
| `MvLogNormalCholesky`                      | ✓ static  | analytic          | ✓            | ✓               | ✓          | ✓                  |
| `MvSkewNormalCholesky`                     | ✓ static  | analytic          | ✗            | ✗               | ✗          | ✓                  |
| `MvSkewNormalLocationKernelStdPartialCorr` | ✓ static  | analytic          | ✗            | ✗               | ✗          | ✓                  |
| `MvSkewStudentTFixedTauCholesky`           | ✓ static  | analytic          | ✗            | ✗               | ✗          | ✓                  |
| `MvPowerExponentialCholesky`               | ✓ static  | analytic          | ✗            | ✗               | ✗          | ✓                  |
| `MvPowerExponentialMeanStdPartialCorr`     | ✓ static  | analytic          | ✗            | ✗               | ✗          | ✓                  |
| `MvShashMuSigmaNuTauPartialCorr`           | ✓ static  | analytic          | ✓            | ✓               | ✓          | ✓                  |
| `MvPoissonCommonShock`                     | ✓ static  | analytic          | ✓            | ✗               | ✗          | ✓                  |
| `MultinomialFixedTrials`                   | ✓ static  | analytic          | ✓            | ✓               | ✓          | ✓                  |
| `MultinomialVaryingTrials`                 | ✓ static  | analytic          | ✗            | ✗               | ✗          | ✗                  |
| `DirichletMultinomialFixedTrials`          | ✓ static  | analytic          | ✓            | ✓               | ✓          | ✓                  |
| `DirichletMultinomialVaryingTrials`        | ✓ static  | analytic          | ✗            | ✗               | ✗          | ✗                  |
| `DirichletMeanPrecision`                   | ✓ static  | analytic          | ✗            | ✗               | ✗          | ✓                  |
| `LogisticNormalAlrCholesky`                | ✓ static  | analytic          | ✗            | ✗               | ✗          | ✓                  |
