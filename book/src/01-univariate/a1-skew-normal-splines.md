# Сплайн-предикторы положения, масштаба и асимметрии

Полный код примера находится в [`examples/a1_skew_normal.rs`](https://github.com/hexqnt/gamlss/blob/main/examples/a1_skew_normal.rs).

Предыдущие модели поочерёдно усложняли нормальное условное распределение: сначала признаки объясняли только линейное среднее, затем собственный predictor появился у масштаба. Теперь одновременно добавим три возможности: нелинейные predictors, явную асимметрию распределения и прикладной optimizer с line search. Архитектурный принцип при этом не меняется: каждому параметру family соответствует отдельный typed block.

## Данные `a1`

Встроенный датасет содержит 2000 пар $(x_i,y_i)$ с $x\in[0,1.2]$. Он загружается из `gamlss-datasets` без runtime I/O и дополнительных allocations:

```rust,ignore
{{#include ../../../examples/a1_skew_normal.rs:data}}
```

Зависимость среднего от `x` явно нелинейна. Ширина облака также меняется по диапазону признака, а форма отклонений не везде симметрична. На рисунке показаны результат fit и отдельная кривая параметра асимметрии:

![Сплайн-модель skew-normal для a1](../assets/figures/a1-skew-normal.svg "Наблюдения a1, условное среднее, 90%-интервал и параметр асимметрии")

Отклик принимает отрицательные значения, поэтому Gamma и log-normal на исходной шкале не подходят. Skew-normal поддерживает всю вещественную прямую и позволяет моделировать наблюдаемую асимметрию без преобразования `y`.

## Почему mean/SD-параметризация skew-normal

В примере используется `SkewNormalMeanSd`, для которой natural-scale параметры имеют прямой смысл:

$$
Y_i\mid x_i \sim \operatorname{SN}_{\text{mean/sd}}\bigl(m_i,s_i,\nu_i\bigr),
\qquad
\mathbb E[Y_i\mid x_i]=m_i,
\qquad
\operatorname{SD}(Y_i\mid x_i)=s_i.
$$

Параметр $\nu_i$ управляет направлением и силой асимметрии; при $\nu_i=0$ family совпадает с нормальным распределением. Mean/SD-параметризация выбрана потому, что первый spline остаётся условным средним даже при меняющемся $\nu(x)$.

Links в модели имеют вид

$$
m_i=\eta_{m,i},
\qquad
s_i=\exp\!\left(\operatorname{clamp}(\eta_{s,i},-8,2)\right),
\qquad
\nu_i=\eta_{\nu,i}.
$$

`ClampedLog<-8, 2>` гарантирует положительный standard deviation и ограничивает экстремальные значения масштаба во время оптимизации. В найденном решении границы clamp не активны.

## Три сплайн-предиктора

Для каждого параметра строится собственный кубический B-spline basis:

$$
\eta_{r}(x)=\sum_{j=0}^{K_r-1}\beta_{r,j}B_{r,j}(x),
\qquad
r\in\{m,s,\nu\}.
$$

`OpenUniformSplineDesign::from_data` запоминает диапазон обучающих `x`, размещает open-uniform basis на этом диапазоне и вычисляет только локально ненулевые функции для каждой строки. Полная плотная design matrix не материализуется.

| Параметр | Marker | Число basis functions | Link | Penalty weight |
| --- | --- | ---: | --- | ---: |
| Условное среднее $m(x)$ | `Mean` | 20 | `Identity` | 0.05 |
| Standard deviation $s(x)$ | `Sigma` | 14 | `ClampedLog<-8, 2>` | 0.10 |
| Shape $\nu(x)$ | `Nu` | 12 | `Identity` | 0.01 |

Всего optimizer видит $20+14+12=46$ коэффициентов.

```rust,ignore
{{#include ../../../examples/a1_skew_normal.rs:model}}
```

Offsets вычисляются из `len()` предыдущих blocks. В результате layout плоского вектора коэффициентов остаётся явным, а соответствие `(Mean, Sigma, Nu)` требованиям family проверяется при компиляции и при построении модели.

## P-spline penalties

Большой basis сам по себе только даёт модели возможность изгибаться. Гладкость задаёт `PreparedDifferencePenalty` второго порядка. Для одного блока с $K$ коэффициентами библиотека добавляет

$$
J_2(\boldsymbol\beta)
=\frac{\lambda}{K-2}
\sum_{j=0}^{K-3}
\left(\beta_{j+2}-2\beta_{j+1}+\beta_j\right)^2.
$$

Комбинацию B-spline basis и difference penalty обычно называют P-spline. Penalty подавляет резкие изменения наклона соседних коэффициентов, но не штрафует их постоянный и линейный профили.

В этом примере числа basis functions и значения $\lambda$ зафиксированы вручную.

Модель использует `ObjectiveScale::Mean`, поэтому оптимизируется

$$
\mathcal L(\boldsymbol\beta)
=\frac{1}{n}\sum_{i=1}^{n}-\log p(y_i\mid m_i,s_i,\nu_i)
+J_m+J_s+J_\nu.
$$

Mean scaling делает характерный масштаб likelihood и выбранных penalties менее зависимым от числа наблюдений. Penalties при этом не делятся на $n$: их веса уже заданы относительно средней negative log-likelihood.

## Инициализация и два старта

Ряды B-spline basis образуют partition of unity. Поэтому одинаковые коэффициенты одного блока задают константу для всех наблюдений. Пример начинает spline среднего с общей выборочной средней, spline log-scale — с логарифма общего standard deviation, а затем запускает shape block из двух констант: $\nu=-5$ и $\nu=5$.

Ненулевой старт здесь принципиален. В mean/SD-параметризации $\nu=0$ является стационарной точкой симметричной normal-подмодели. Gradient-based optimizer, запущенный ровно из нуля, может остаться в ней даже тогда, когда решение с асимметрией имеет меньшую negative log-likelihood. Два знака не гарантируют глобальный optimum, но устраняют наиболее очевидную зависимость от этой симметричной точки; пример сохраняет fit с меньшим penalized objective.

```rust,ignore
{{#include ../../../examples/a1_skew_normal.rs:argmin}}
```

Значения $\pm5$ используются только как начальные shape predictors; L-BFGS свободно изменяет все 12 коэффициентов `Nu`.

## Подключение `argmin`

`gamlss-core` намеренно не зависит от конкретного optimizer-а. Его `Objective` предоставляет значение и аналитический gradient над обычным `&[f64]`, а тонкий adapter в примере реализует traits `CostFunction` и `Gradient` из `argmin`.

```rust,ignore
{{#include ../../../examples/a1_skew_normal.rs:adapter}}
```

Есть небольшое различие в borrowing contracts. GAMLSS принимает `&mut self`, чтобы повторно использовать внутренние buffers, а callbacks `argmin` принимают `&self`. Локальный `RefCell` согласует эти интерфейсы без копирования workspace при каждом вызове.

В качестве solver используется L-BFGS с памятью из десяти пар обновлений. `MoreThuenteLineSearch` подбирает длину шага с условиями Wolfe, поэтому пример не фиксирует learning rate вручную.

```rust,ignore
{{#include ../../../examples/a1_skew_normal.rs:optimizer}}
```

Остановку задают tolerance по норме gradient, tolerance по изменению objective и верхняя граница 750 итераций. После `run()` пример печатает termination reason и заново вычисленную норму gradient.

Запустить обычный fit можно командой

```bash
cargo run --example a1_skew_normal
```

Для воспроизведения графика пример умеет печатать исходный отклик, fitted parameters и границы 90%-интервала как чистый CSV:

```bash
cargo run --quiet --example a1_skew_normal -- --csv > a1-fit.csv
```

## Результат и диагностика

Для текущих настроек один детерминированный запуск даёт приблизительно следующие значения:

| Величина | Результат |
| --- | ---: |
| Penalized mean objective | -0.7715 |
| Mean negative log-likelihood без penalties | -0.7931 |
| Сумма penalties | 0.0215 |
| Норма gradient | $1.6\cdot10^{-4}$ |
| Диапазон $m(x)$ | $[-0.478,\ 0.917]$ |
| Диапазон $s(x)$ | $[0.050,\ 0.175]$ |
| Диапазон $\nu(x)$ | $[-2.91,\ 7.33]$ |
| Покрытие 90%-интервала на training data | 0.888 |

Интервал на рисунке вычислен через квантили fitted skew-normal distribution. При $\nu(x)\ne0$ он в общем случае не симметричен относительно $m(x)$, поэтому формула `mean ± 1.645 * sigma` здесь неверна. Нижняя панель показывает, что направление асимметрии меняется по `x`: shape положителен на краях диапазона и отрицателен в средней части.

```rust,ignore
{{#include ../../../examples/a1_skew_normal.rs:diagnostics}}
```

Среднее PIT в текущем fit близко к 0.5, а normalized quantile residuals имеют среднее около нуля и standard deviation около единицы. Все приведённые значения вычислены на training data; в примере они служат компактной проверкой результата оптимизации.
