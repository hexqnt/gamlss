# Структура проекта и уровни API

## Два уровня API

В библиотеке есть два основных уровня API.

### Низкоуровневое typed API

Низкоуровневое API — основной слой проекта. Оно строится вокруг типизированных parameter blocks, links, families и compiled models:

- `gamlss-core` задает базовые абстракции: link functions, parameter markers, `ParameterBlock`, `ParameterBlocks`, objective/model traits и compiled model evaluation;
- `gamlss-family` содержит распределения, likelihoods, score helpers и CDF/quantile utilities;
- `gamlss-spline` содержит spline/Fourier predictors, penalties и metadata;
- `gamlss-transform` содержит target transforms и persisted transform state;
- `gamlss-diagnostics` содержит post-fit diagnostics поверх fitted distributional models.

Этот уровень рассчитан на код, где важны compile-time guarantees, явный layout параметров, отсутствие строковых lookup-ов в hot path и минимальные зависимости. Именно typed API является базовой поверхностью для integration layers, optimizer adapters и более высокоуровневых builders.

### Высокоуровневое API

Высокоуровневое API — convenience layer для более компактного описания модели. Сейчас оно представлено optional crate-ом `gamlss-formula`, который доступен из facade crate `gamlss` при включенной feature `formula`.

`gamlss-formula` принимает runtime specifications и curated builder descriptions, затем компилирует их в typed core models. Этот слой предназначен для удобных workflows, но он не является заменой низкоуровневому API и не обязан зеркалировать все combinations of families, links, penalties и custom parameterizations, доступные напрямую через `gamlss-core` и соседние crate-ы.

## Facade crate `gamlss`

Crate `gamlss` — основная точка входа для большинства пользователей. Он реэкспортирует основные workspace crate-ы:

- [`core`](crate::core) для typed ядра;
- [`family`](crate::family) для distributions и likelihoods;
- [`spline`](crate::spline) для spline predictors и penalties;
- [`transform`](crate::transform) для target preprocessing;
- [`diagnostics`](crate::diagnostics) для post-fit diagnostics;
- `formula` для experimental formula/builder API, если включена feature `formula`.

Также facade crate предоставляет [`prelude`](crate::prelude), где собраны наиболее часто используемые типы.

## Поток данных

Типичный низкоуровневый workflow выглядит так:

1. Пользователь выбирает distribution family из `gamlss-family`.
2. Для каждого параметра распределения создается свой `ParameterBlock` с parameter marker, link function, design/predictor и penalty.
3. Несколько блоков объединяются в `ParameterBlocks`, который задает общий layout beta-вектора.
4. `gamlss-core` компилирует family, blocks, response и optional weights в objective/model object.
5. Внешний optimizer работает с objective/gradient surface, не владея modeling logic.
6. После fit-а пользователь вызывает prediction и diagnostics helpers.

Высокоуровневый builder слой должен в итоге приводить к той же compiled typed модели, а не выполнять отдельную интерпретацию формулы в hot path.

## Границы зависимостей

`gamlss-core` намеренно остается легким и независимым от optimizer crates, dataframe libraries и тяжелых matrix backends. Более тяжелые integrations должны жить за optional features или в отдельных integration crate-ах.

Это разделение позволяет использовать core abstractions в разных окружениях: от небольших embedded-style numeric loops до более крупных ML pipelines с собственными matrix backends и optimizer stacks.
