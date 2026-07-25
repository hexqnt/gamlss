# Структура проекта и уровни API

## Layout workspace

Workspace состоит из facade crate-а `gamlss` в корне репозитория и набора специализированных crate-ов:

- `gamlss` — основная публичная точка входа, re-exports и общий `prelude`;
- `gamlss-bayes` — lightweight Bayesian boundary: normalized coefficient priors, posterior potential и pointwise log-likelihood без sampler backend;
- `gamlss-core` — type-driven ядро: links, parameter markers, `ParameterBlock`, `ParameterBlocks`, predictor contracts, observation views, objectives, compiled models, prediction helpers и typed family/model interfaces;
- `gamlss-family` — distribution-specific слой: univariate families, optional multivariate families, likelihoods, analytical score helpers, CDF/quantile/density/CRPS capability implementations и optional sampling API;
- `gamlss-special` — shared scalar `f64` special functions и численные helpers для likelihood, CDF, quantile и transform кода;
- `gamlss-spline` — spline/Fourier predictor blocks, row-basis helpers, sparse row metadata, tensor products и smoothness/shape penalties;
- `gamlss-transform` — target transforms, fitted transform state и domain-aware inverse transforms;
- `gamlss-diagnostics` — post-fit extension APIs поверх compiled models и prediction views: PIT/CDF values, normalized quantile residuals, CRPS values/summaries и reusable diagnostics views;
- `gamlss-formula` — experimental optional formula/builder layer, который читает runtime data/specifications, материализует predictor designs и компилирует curated workflows в typed core models.

В репозитории также есть `examples` для end-to-end примеров, `tests` для facade/public API проверок, crate-local `tests` для контрактов и численных проверок, `docs` для документации, `.github/workflows` для CI и `issues` для проектных заметок.

## Два уровня API

В библиотеке есть два основных уровня API.

### Низкоуровневое typed API

Низкоуровневое API — основной слой проекта. Оно строится вокруг типизированных parameter blocks, links, families, observations, objectives и compiled models:

- `gamlss-core` задает базовые абстракции и не владеет конкретными распределениями;
- `gamlss-family` реализует конкретные family contracts из core и хранит distribution-specific likelihood, score, chain-rule и domain logic;
- `gamlss-special` содержит общие численные building blocks, которые не должны дублироваться внутри families;
- `gamlss-spline` предоставляет predictor blocks и penalties, совместимые с typed `ParameterBlock`;
- `gamlss-transform` живет вокруг target preprocessing и persisted state, а не внутри compiled model hot path;
- `gamlss-diagnostics` добавляет post-fit вычисления через extension traits, не расширяя базовый `Family` contract сверх необходимости.

Этот уровень рассчитан на код, где важны compile-time guarantees, явный layout параметров, отсутствие строковых lookup-ов в hot path, optimizer-agnostic objective surface и минимальные зависимости. Именно typed API является базовой поверхностью для integration layers, optimizer adapters, diagnostics и более высокоуровневых builders.

### Высокоуровневое API

Высокоуровневое API — convenience layer для более компактного описания модели. Сейчас оно представлено crate-ом `gamlss-formula`, который доступен из facade crate `gamlss` при включенной feature `formula`; эта feature включена по умолчанию.

`gamlss-formula` принимает typed runtime inputs через `DataView` и `Col<T>`, собирает curated term builders вроде intercept, linear, factor, interaction, offsets, P-splines, cyclic splines, Fourier terms, monotone terms и tensor P-splines, затем компилирует их в typed core models. Сейчас этот слой покрывает выбранные specs для normal, beta, gamma, inverse Gaussian, log-normal и Weibull workflows.

Этот слой предназначен для удобных workflows, но он не является заменой низкоуровневому API и намеренно не содержит string formula parser, fitting loop, optimizer integration, diagnostics, dataframe adapters или полного зеркала всех combinations of families, links, penalties и custom parameterizations, доступных напрямую через typed crates.

## Facade crate `gamlss`

Crate `gamlss` — основная точка входа для большинства пользователей. Он реэкспортирует основные workspace crate-ы:

- [`core`](crate::core) для typed ядра;
- [`family`](crate::family) для distributions и likelihoods;
- [`special`](crate::special) для special functions и численных helpers;
- [`spline`](crate::spline) для spline/Fourier predictors и penalties;
- [`transform`](crate::transform) для target preprocessing;
- [`diagnostics`](crate::diagnostics) для post-fit diagnostics;
- `bayes` для normalized coefficient priors и posterior potential, если включена feature `bayes`;
- `formula` для experimental formula/builder API, если включена feature `formula`.

Также facade crate предоставляет [`prelude`](crate::prelude), где собраны наиболее часто используемые типы из core, family, spline, transform и diagnostics, а при включённых соответствующих features — bayes и formula.

## Cargo features

Facade crate сейчас имеет четыре feature:

- `formula` включена по умолчанию и добавляет re-export namespace `gamlss::formula`;
- `bayes` добавляет opt-in re-export namespace `gamlss::bayes` и Bayesian-типы в `gamlss::prelude`;
- `rand` пробрасывает `gamlss-family/rand` и включает sampling API для families, где он реализован;
- `multivariate` пробрасывает `gamlss-family/multivariate` и включает optional multivariate distribution families.

Если нужен более строгий low-level dependency surface без experimental builder layer, можно использовать `gamlss` с `default-features = false` или зависеть от отдельных crate-ов напрямую.

## Поток данных

Типичный низкоуровневый workflow выглядит так:

1. Пользователь при необходимости подготавливает response через `gamlss-transform` и сохраняет fitted transform state рядом с модельными артефактами.
2. Пользователь выбирает distribution family из `gamlss-family` или реализует custom family поверх contracts из `gamlss-core`.
3. Для каждого параметра распределения создается свой `ParameterBlock` с parameter marker, design/predictor и penalty; links принадлежат family и не дублируются в block type.
4. Несколько блоков объединяются в `ParameterBlocks`, который задает общий layout beta-вектора и offsets.
5. `gamlss-core` компилирует family, blocks, response и optional weights в `Gamlss`/workspace-backed model object через constructors вроде `try_new`, `try_new_weighted` или strict observation paths.
6. Внешний optimizer работает с `Objective`/gradient surface и не владеет modeling logic.
7. После fit-а пользователь вызывает prediction helpers для training rows или compatible prediction blocks, затем diagnostics extension APIs из `gamlss-diagnostics`, если family exposes нужные capability traits.

Высокоуровневый builder слой должен приводить к той же compiled typed модели, а не выполнять отдельную интерпретацию formula/specification в hot path.

## Capability boundaries

Capabilities выражаются traits вместо глобальных runtime flags. Например, distribution functions, density/log-density, quantile, CRPS, simulation и multivariate transforms доступны только там, где family реализует соответствующий trait или включена нужная Cargo feature.

Compiled fitting является opt-in. `CompilableFamily` связывает family с sealed static shape tree (`Scalar`, `Vector`, `Lower`, `StrictLower`, `Simplex`, `Product`, `Repeated`, `Broadcast`), а `DynamicallyCompilableFamily` обслуживает отдельный runtime-dimensional path через `DynamicParameterBlocks`. Оба пути переиспользуют общую optimizer-independent модель и не переносят links или likelihood math в predictors.

Diagnostics построены поверх prediction views и capability traits вроде CDF/CRPS, поэтому их можно расширять без утяжеления базового family contract. Optimizer adapters должны оставаться тонкими: они адаптируют objective/gradient API, но не переносят modeling logic в optimizer crate.

## Границы зависимостей

`gamlss-core` намеренно остается легким и независимым от optimizer crates, dataframe libraries и тяжелых matrix backends. Более тяжелые integrations должны жить за optional features или в отдельных integration crate-ах.

`gamlss-family` зависит от core и special; sampling остается за optional `rand`. `gamlss-spline` зависит от core и реализует predictor/penalty pieces, а не отдельный modeling layer. `gamlss-transform` зависит от special для численных helpers и хранит transform-specific state отдельно от compiled model. `gamlss-formula` может быть динамическим boundary layer, но compiled model evaluation должна оставаться типизированной и эффективной.

Это разделение позволяет использовать core abstractions в разных окружениях: от небольших embedded-style numeric loops до более крупных ML pipelines с собственными matrix backends, dataframe adapters и optimizer stacks.
