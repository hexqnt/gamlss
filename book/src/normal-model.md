# Первая нормальная модель

Полный код этого шага находится в [`examples/simple_fit.rs`](https://github.com/hexqnt/gamlss/blob/main/examples/simple_fit.rs).

Рассмотрим нормальную модель с линейным средним и постоянным стандартным отклонением:

$$
Y_i \sim \mathcal N(\mu_i, \sigma_i),
$$

$$
\mu_i = \eta_{\mu,i} = \beta_{\mu,0} + \beta_{\mu,1}x_i,
\qquad
\sigma_i = \exp(\eta_{\sigma,i}) = \exp(\beta_{\sigma,0}).
$$

В модели три глобальных коэффициента: интерсепт и наклон для `mu`, а также интерсепт на log-шкале для `sigma`. Экспонента гарантирует положительность стандартного отклонения.

## От коэффициентов к распределению

Для каждого наблюдения модель проходит один и тот же путь:

$$
\underbrace{
  \begin{pmatrix}
    \beta_{\mu,0} \\
    \beta_{\mu,1} \\
    \beta_{\sigma,0}
  \end{pmatrix}
}_{\text{3 глобальных коэффициента}}
\longrightarrow
\underbrace{
  \begin{pmatrix}
    \eta_{\mu,i} \\
    \eta_{\sigma,i}
  \end{pmatrix}
}_{\text{2 предиктора}}
\longrightarrow
\underbrace{
  \begin{pmatrix}
    \mu_i \\
    \sigma_i
  \end{pmatrix}
}_{\theta_i}
\longrightarrow
\mathcal N(\mu_i, \sigma_i)
\longrightarrow Y_i.
$$

Первый переход выполняют prediction blocks и дизайн-матрицы, второй — link-функции, а семейство `Normal` вычисляет likelihood выбранного распределения.

## Соответствие API и формул

| Математический объект | API |
| --- | --- |
| Нормальное семейство | `Normal::<Identity, Log>` |
| Параметр среднего | `Mu` |
| Параметр масштаба | `Sigma` |
| Дизайн для `mu`: `[1, x_i]` | `DenseDesign::from_rows(...)` |
| Дизайн для `sigma`: `[1]` | `DenseDesign::intercept(n)` |
| Блоки коэффициентов | `ParameterBlock` и `ParameterBlocks` |
