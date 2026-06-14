# gamlss

[![CI](https://github.com/hexqnt/gamlss/actions/workflows/ci.yml/badge.svg)](https://github.com/hexqnt/gamlss/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/gamlss.svg)](https://crates.io/crates/gamlss)[![docs.rs](https://img.shields.io/docsrs/gamlss)](https://docs.rs/gamlss)

Type-driven Rust crates for GAMLSS-style modeling.

> **Status:** This crate is an early MVP. It is published to make the project
> available and reserve the crate names, but it is not intended for production
> use yet. The public API, internals, numerical behavior, and crate structure
> may change substantially while the library is being developed.

## Структура crate-ов

Основная точка входа для пользователей — crate `gamlss`:

```toml
[dependencies]
gamlss = "0.1"
```

`gamlss` — batteries-included фасад: по умолчанию он реэкспортирует typed
core, готовые families, spline/predictor building blocks, target transforms и
dynamic formula/builder слой.

Workspace также публикует отдельные crate-ы для более явного контроля API и
зависимостей:

- `gamlss-core` — type-driven ядро: links, parameter blocks, objectives и
  compiled models. Это низкоуровневый Rust-native framework API: без
  optimizer-а, dataframe abstraction и тяжёлого matrix backend-а.
- `gamlss-family` — reusable typed building blocks: распределения,
  likelihoods, scores, CDF/quantile/simulation helpers.
- `gamlss-spline` — reusable typed building blocks для spline/Fourier
  predictors, penalties и spline metadata.
- `gamlss-transform` — target preprocessing transforms.
- `gamlss-formula` — dynamic convenience layer. Он предназначен для удобной
  сборки типизированных моделей из простых runtime specifications, но не
  задаёт архитектуру hot-path core API.

Эти crate-ы опубликованы отдельно, чтобы сохранить явные границы модулей и
легкие зависимости, но они не являются основной пользовательской поверхностью.
В обычном случае достаточно зависеть от `gamlss`; остальные crate-ы будут
подключены транзитивно. Если нужен минимальный низкоуровневый API для
integration crate-а или собственного optimizer-а, используйте `gamlss-core`
напрямую.

## Общая форма GAMLSS

В общем виде GAMLSS задает условное распределение отклика через набор
параметров выбранного семейства:

$$
Y_i \mid x_i \sim D(\theta_{i1}, \ldots, \theta_{iK}),
$$

где `D(...)` — выбранное параметрическое распределение, а каждый параметр
моделируется своим link-function и аддитивным предиктором:

$$
g_k(\theta_{ik}) = \eta_{ik}
  = X_{k,i}\beta_k + \sum_j f_{k,j}(x_i),
\qquad k = 1,\ldots,K.
$$

Классическое соглашение `gamlss` часто называет первые четыре параметра
`mu`, `sigma`, `nu` и `tau`:

$$
(\theta_{i1}, \theta_{i2}, \theta_{i3}, \theta_{i4})
  = (\mu_i, \sigma_i, \nu_i, \tau_i),
$$

$$
Y_i \mid x_i \sim D(\mu_i, \sigma_i, \nu_i, \tau_i).
$$

Здесь `mu`, `sigma`, `nu` и `tau` обычно отвечают за положение, масштаб,
асимметрию и форму распределения. Это соглашение об именах, а не обязательная
форма API: не каждое семейство использует все четыре параметра, а typed core
поддерживает пользовательские parameter markers для собственного числа и
смысла параметров.
