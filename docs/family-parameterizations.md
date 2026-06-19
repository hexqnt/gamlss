# Параметризации распределений

Этот документ описывает параметризации на естественной шкале, которые
поддерживает `gamlss-family`.

`Eta`-значения — это предикторы на link-шкале. `Theta`-значения — параметры на
естественной шкале, которые используются в likelihood, CDF, quantile, CRPS и
sampling helpers. Link-функции задаются type-параметрами; в таблице ниже
перечислены default links из `Default*` aliases.

| Распределение      | Параметризация на естественной шкале                                | Смысл параметров                                                                                                                                                                              | Default links                     | Support наблюдений                                         |
| ------------------ | ------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ---------------------------------------------------------- |
| `Bernoulli`        | `mu`                                                                | `mu` — вероятность успеха, ограничена `(0, 1)`.                                                                                                                                               | `mu`: `Logit`                     | `0` или `1`                                                |
| `Beta`             | `mu`, `precision`                                                   | `mu` — среднее в `(0, 1)`; `precision` — положительная precision. Эквивалентные beta-shapes: `alpha = mu * precision` и `beta = (1 - mu) * precision`.                                        | `mu`: `Logit`; `precision`: `Log` | `0 < y < 1`                                                |
| `Exponential`      | `rate`                                                              | `rate` — положительная интенсивность exponential-распределения. Среднее равно `1 / rate`.                                                                                                     | `rate`: `Log`                     | `y >= 0`                                                   |
| `Gamma`            | `shape`, `rate`                                                     | `shape` и `rate` — положительные gamma-параметры. Среднее равно `shape / rate`.                                                                                                               | `shape`: `Log`; `rate`: `Log`     | `y > 0`                                                    |
| `Gumbel`           | `mu`, `sigma`                                                       | Maximum-type Gumbel в location-scale форме. `mu` — location; `sigma` — положительный scale.                                                                                                   | `mu`: `Identity`; `sigma`: `Log`  | конечное вещественное `y`                                  |
| `InverseGaussian`  | `mu`, `shape`                                                       | `mu` — положительное среднее; `shape` — положительный shape-параметр inverse Gaussian.                                                                                                        | `mu`: `Log`; `shape`: `Log`       | `y > 0`                                                    |
| `Laplace`          | `mu`, `sigma`                                                       | `mu` — location; `sigma` — положительный scale.                                                                                                                                               | `mu`: `Identity`; `sigma`: `Log`  | конечное вещественное `y`                                  |
| `LogNormal`        | `mu`, `sigma`                                                       | `mu` — location для `log(Y)`; `sigma` — положительный scale для `log(Y)`.                                                                                                                     | `mu`: `Identity`; `sigma`: `Log`  | `y > 0`                                                    |
| `Logistic`         | `mu`, `sigma`                                                       | `mu` — location; `sigma` — положительный scale.                                                                                                                                               | `mu`: `Identity`; `sigma`: `Log`  | конечное вещественное `y`                                  |
| `Lomax`            | `shape`, `scale`                                                    | Pareto type II форма. `shape` — положительный tail-shape; `scale` — положительный scale.                                                                                                      | `shape`: `Log`; `scale`: `Log`    | `y >= 0`                                                   |
| `NegativeBinomial` | `mu`, `shape`                                                       | `mu` — положительное среднее; `shape` — положительный overdispersion/size параметр. Дисперсия равна `mu + mu^2 / shape`.                                                                      | `mu`: `Log`; `shape`: `Log`       | неотрицательный целочисленный count                        |
| `Normal`           | `mu`, `sigma`                                                       | `mu` — location/mean; `sigma` — положительное стандартное отклонение.                                                                                                                         | `mu`: `Identity`; `sigma`: `Log`  | конечное вещественное `y`                                  |
| `Poisson`          | `mu`                                                                | `mu` — положительное среднее/rate распределения Пуассона.                                                                                                                                     | `mu`: `Log`                       | неотрицательный целочисленный count                        |
| `StudentT`         | `mu`, `sigma`; фиксированный `degrees_of_freedom` в значении family | `mu` — location; `sigma` — положительный scale; `degrees_of_freedom` конечен и положителен, но не является моделируемым `Eta`-параметром. Default family использует `degrees_of_freedom = 5`. | `mu`: `Identity`; `sigma`: `Log`  | конечное вещественное `y`                                  |
| `Weibull`          | `shape`, `scale`                                                    | `shape` и `scale` — положительные параметры Weibull.                                                                                                                                          | `shape`: `Log`; `scale`: `Log`    | `y > 0` для NLL, `y >= 0` для boundary handling в CDF/CRPS |

## Ограничения link-функций

Families кодируют домены параметров через link traits:

| Домен параметра                                                     | Link trait, используемый families | Типичный default |
| ------------------------------------------------------------------- | --------------------------------- | ---------------- |
| Неограниченные вещественные параметры, например location            | `Link<f64>`                       | `Identity`       |
| Положительные параметры, например scale, shape, rate и precision    | `PositiveLink<f64>`               | `Log`            |
| Параметры на единичном интервале, например вероятности и beta means | `UnitIntervalLink<f64>`           | `Logit`          |

Таким образом, поддерживаемая параметризация фиксируется типом family, а
link-функции остаются настраиваемыми через generic-параметры family.
