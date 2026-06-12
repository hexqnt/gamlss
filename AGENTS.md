# AGENTS.md

## Область действия

* Этот файл применяется ко всему workspace.
* Более близкие к редактируемому коду `AGENTS.md` или `AGENTS.override.md` в поддиректориях переопределяют эти правила для своей области.
* Явные инструкции пользователя имеют приоритет над этим файлом.

## Обзор проекта

Это Rust-native workspace для GAMLSS / distributional regression.

Цель проекта — не прямой порт R `gamlss` или `gamlss2`. Их можно использовать как статистический ориентир, но архитектура должна оставаться Rust-идиоматичной: type-driven, zero-cost, backend-agnostic и optimizer-agnostic.

## Структура workspace

* `gamlss` — high-level public crate и re-exports.
* `gamlss-core` — type-driven ядро: links, parameter blocks, objectives, compiled models.
* `gamlss-family` — распределения, likelihoods, scores, CDF/quantile helpers.
* `gamlss-spline` — spline bases, penalties и spline metadata.
* `gamlss-formula` — optional dynamic formula/builder layer, который компилируется в typed models.

## Принципы дизайна

* Предпочитать type-driven API и compile-time гарантии.
* Hot path должен быть zero-cost: избегать `Box<dyn ...>`, строковых lookup-ов, скрытых allocations и runtime dispatch во внутренних циклах.
* `gamlss-core` должен оставаться независимым от оптимизаторов, dataframe-библиотек и тяжёлых matrix backends.
* Ограничения параметров по возможности выражать типами и traits, например positive links для scale-параметров.
* Предпочитать “parse, don’t validate”: на границе превращать внешний ввод в более строгие внутренние типы.
* Корректность, численная устойчивость и понятные ошибки важнее микропроизводительности.
* SIMD-oriented и sparse/dense backends желательны, но не должны преждевременно протекать в core abstractions.

## Rust-стиль

* Использовать Rust edition 2024.
* Предпочитать stable Rust. Nightly допустим, если он действительно улучшает дизайн или производительность.
* Писать идиоматичный Rust с явным ownership, typed errors и небольшими сфокусированными модулями.
* Публичные API должны иметь rustdoc, если поведение или инварианты не очевидны.
* Для recoverable errors использовать `Result`; не использовать panic для обычной валидации входных данных.
* `unsafe` не добавлять без явного согласования. Любой согласованный `unsafe` должен быть изолирован, задокументирован и покрыт тестами.

## Границы crate-ов

* `gamlss-core` должен оставаться минимальным и лёгким по зависимостям.
* Optimizer integrations должны жить вне workspace, если пользователь явно не просит добавить отдельный integration crate.
* Тяжёлые зависимости вроде `faer`, `sprs`, `polars`, `ndarray`, `serde` должны быть optional features или отдельными integration crates.
* Formula/builder layer может быть динамическим; compiled model evaluation должен оставаться типизированным и эффективным.
* Optimizer adapters должны быть тонкими: адаптировать objective/gradient traits, но не переносить modeling logic в optimizer crate.

## Тестирование и проверки

Перед завершением изменения, если toolchain доступен, выполнить релевантные проверки:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features
cargo test --workspace --all-features
```

Для численного кода:

* Добавлять или обновлять тесты для likelihoods, links, penalties и gradients.
* При добавлении новых families сравнивать аналитические градиенты с finite-difference checks.
* Тестировать invalid domains, boundary cases, `NaN`/`inf` handling и extreme parameter values.
* Предпочитать небольшие детерминированные тесты большим stochastic tests.

Если команду нельзя выполнить в текущей среде, явно указать это и перечислить, что осталось непроверенным.

## Зависимости

* Не добавлять production dependencies без понятной причины.
* Для backend integrations предпочитать optional features.
* Избегать новых зависимостей в `gamlss-core`, если они не маленькие, стабильные и необходимые.
* Не вводить global state, hidden threading или implicit randomness в core modeling code.

## Рабочий процесс

* Перед изменениями посмотреть соседний код и существующие тесты.
* Делать минимальные сфокусированные изменения, сохраняя границы crate-ов.
* При изменении публичного поведения обновлять examples, rustdoc или tests.
* Не переписывать архитектуру ради сходства с R-пакетами; сохранять Rust-native abstractions.
