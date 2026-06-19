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
gamlss = "*"
```

`gamlss` — batteries-included фасад. Основной и наиболее стабильный путь сейчас
идет через низкоуровневое typed API: `gamlss-core`, `gamlss-family`,
`gamlss-spline` и `gamlss-transform`. По умолчанию фасад также реэкспортирует
`gamlss-formula`, но этот слой пока является экспериментальным optional
convenience crate, а не основным API библиотеки.

Workspace также публикует отдельные crate-ы для более явного контроля API и
зависимостей:

- `gamlss-core` — type-driven ядро для links, parameter blocks, objectives и
  compiled models.
- `gamlss-family` — распределения, likelihoods и score helpers.
- `gamlss-diagnostics` — post-fit PIT/CDF diagnostics и normalized quantile
  residuals для supported continuous CDF families.
- `gamlss-spline` — spline/Fourier predictors, penalties и spline metadata.
- `gamlss-transform` — target preprocessing transforms.
- `gamlss-formula` — экспериментальный optional formula/builder layer, который
  компилирует runtime specifications в typed models. Он покрывает curated
  high-level workflows и не обязан зеркалировать все families, links и
  parameterizations, доступные в низкоуровневых crate-ах.

В обычном случае достаточно зависеть от `gamlss`; остальные crate-ы будут
подключены транзитивно. Если нужен более строгий low-level surface без
экспериментального formula слоя, используйте `default-features = false` или
зависимости на отдельные crate-ы напрямую.

## Cargo features

- `formula` включена по умолчанию и реэкспортирует экспериментальный
  `gamlss-formula` namespace из facade crate.
- `rand` включает sampling API в `gamlss-family` через facade crate:
  `rand = ["gamlss-family/rand"]`.

## Общая ионформация о GAMLSS

GAMLSS можно читать как distributional regression: модель описывает не только
условное среднее отклика, а всё условное распределение. Это полезно, когда
разброс, асимметрия, хвосты или сама область допустимых значений меняются вместе
с признаками. Например, одна часть модели может описывать центр распределения,
другая — гетероскедастичный масштаб, а третья — форму хвостов.

В общем виде GAMLSS задает условное распределение отклика через набор
параметров выбранного family:

$$
Y_i \mid x_i \sim D(\theta_{i1}, \ldots, \theta_{iK}),
$$

где `D(...)` — выбранное параметрическое распределение. Каждый его параметр
моделируется своим link-function и отдельным предиктором:

$$
g_k(\theta_{ik}) = \eta_{ik}
  = X_{k,i}\beta_k + \sum_j f_{k,j}(x_i),
\qquad k = 1,\ldots,K.
$$

Иными словами, у разных параметров одного распределения могут быть разные
наборы признаков, разные spline terms, разные штрафы и разные domain
constraints. Link-function переводит unconstrained линейный предиктор `eta` в
допустимую область параметра: например, scale-параметры обычно требуют
положительности, probability/mean-параметры для beta family — значения внутри
`(0, 1)`.

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
смысла параметров. Поэтому библиотека может выражать как привычные
location-scale модели вроде normal/log-normal/Laplace/Student's t, так и
семейства с другой параметризацией, например gamma, Weibull, inverse Gaussian
или beta.
