# gamlss-transform

Target transform layer for GAMLSS modeling.

> **Status:** Actively developed. Public API, internals, numerical behavior, and crate structure may still change before a stable 1.0 release.

This crate contains target preprocessing transforms with persisted state and domain-aware inverse transforms.

Use it when composing lower-level preprocessing pipelines around typed GAMLSS models.

## Built-in transforms

| Transform               | Target domain                          | Fitted state                                     | Inverse behavior                                                 |
| ----------------------- | -------------------------------------- | ------------------------------------------------ | ---------------------------------------------------------------- |
| `Standardize`           | finite real values                     | mean and RMS scale                               | linear inverse                                                   |
| `RobustStandardize`     | finite real values                     | median and IQR                                   | linear inverse                                                   |
| `MinMaxScale`           | finite real values with non-zero range | minimum and range                                | linear inverse; no clipping                                      |
| `MaxAbsScale`           | finite real values                     | max absolute value, or one for all-zero targets  | linear inverse                                                   |
| `AsinhScale`            | finite real values                     | robust absolute scale                            | signed `sinh` inverse                                            |
| `Log`                   | strictly positive values               | stateless                                        | exponential inverse                                              |
| `Log1pShift`            | finite values above fitted lower bound | shift from training minimum                      | `expm1` inverse minus shift                                      |
| `IdentityPositive`      | strictly positive values               | stateless                                        | identity inverse                                                 |
| `BoxCox`                | strictly positive values               | fitted lambda                                    | checked inverse rejects values outside the lambda domain         |
| `BoxCoxFixed<N, D>`     | strictly positive values               | fixed `N / D` lambda                             | checked inverse rejects values outside the lambda domain         |
| `YeoJohnson`            | finite real values                     | fitted lambda                                    | checked inverse rejects values outside the lambda domain         |
| `YeoJohnsonFixed<N, D>` | finite real values                     | fixed `N / D` lambda                             | checked inverse rejects values outside the lambda domain         |
| `QuantileUniform`       | finite real values                     | sorted unique values and empirical probabilities | inverse clamps to the fitted target range                        |
| `QuantileNormal`        | finite real values                     | sorted unique values and empirical probabilities | normal-scale inverse clamps through the fitted probability range |

Fitted `BoxCox` and `YeoJohnson` estimate lambda with a deterministic profile-likelihood search on `[-5, 5]` and do not depend on an optimizer crate.

`QuantileUniform` and `QuantileNormal` clamp new values outside the fitted range to the fitted boundary probabilities. Their inverse transforms clamp probabilities back to the fitted target range.
