# Литература и ресурсы

Источники ниже полезны как статистический ориентир, коллекция удачных учебных ходов и набор edge cases.

## Книги и обзорные статьи

- [Generalized Additive Models for Location, Scale and Shape: A Distributional Regression Approach, with Applications](https://gamlssbook.bitbucket.io/) — современная книга 2024 года и открытый companion site с кодом, figures и datasets. Полезна разделением материала на основы, inference и полноценные case studies.
- [Flexible Regression and Smoothing: Using GAMLSS in R](https://www.routledge.com/Flexible-Regression-and-Smoothing-Using-GAMLSS-in-R/Stasinopoulos-Rigby-Heller-Voudouris-Bastiani/p/book/9781138197909) — практическая книга о model building, additive terms, selection, diagnostics и centile estimation. Особенно полезен приём сравнивать несколько классов регрессии на одних данных.
- [Distributions for Modeling Location, Scale, and Shape: Using GAMLSS in R](https://www.routledge.com/Distributions-for-Modeling-Location-Scale-and-Shape-Using-GAMLSS-in-R/Rigby-Stasinopoulos-Heller-De-Bastiani/p/book/9780429298547) — справочник по continuous, count и mixed distributions, их support, skewness и tails. Для этой книги он является reference, а не образцом линейного порядка глав.
- [Rigby and Stasinopoulos (2005), Generalized additive models for location, scale and shape](https://doi.org/10.1111/j.1467-9876.2005.00510.x) — исходная статья с общей постановкой GAMLSS, penalized likelihood и additive predictors.
- [Distributional regression using generalized additive models for location, scale and shape](https://www.nature.com/articles/s43586-026-00498-z) — недавний обзор 2026 года, связывающий GAMLSS с более широкой областью distributional regression.

## Воспроизводимые примеры анализа

- [gamlss2: First Steps](https://gamlss-dev.github.io/gamlss2/vignettes/firststeps.html) моделирует распределение дневной максимальной температуры и превращает его в вероятность жаркого дня. Это хороший образец главы, которая заканчивается решением прикладного вопроса, а не таблицей коэффициентов.
- [gamlss2: Centile (Quantile) Estimation](https://gamlss-dev.github.io/gamlss2/vignettes/quantiles.html) проходит маршрут `data -> fit -> diagnostics -> centile curves` на `dbbmi`. Этот маршрут положен в основу planned case study о кривых роста.
- [GAMLSS companion datasets](https://gamlssbook.bitbucket.io/Datasets.html) публикует datasets case studies в CSV для переноса в другое software. Перед включением snapshot в репозиторий всё равно нужно проверить attribution и лицензию исходного набора.
- [GAMLSS short course: Distributional regression](https://mstasinopoulos.github.io/ShortCourse/regression.html) показывает короткие сравнительные analyses и содержит полезные numerical references.

## Диагностика и оценка прогнозов

- [Dunn and Smyth (1996), Randomized quantile residuals](https://doi.org/10.1080/10618600.1996.10474708) объясняет quantile residuals и необходимую randomization для discrete responses. Это ключевой источник для будущей count diagnostics extension.
- [Gneiting and Raftery (2007), Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) даёт основу для log score, CRPS и корректного сравнения probabilistic forecasts.

## Наборы данных

- [`gamlss.data` reference manual](https://gamlss-dev.r-universe.dev/gamlss.data/doc/manual.html) описывает `dbbmi`, `abdom`, `speech` и другие классические datasets вместе с происхождением и структурой переменных.
- [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) содержит hourly и daily counts, погоду и календарные признаки и распространяется под CC BY 4.0.
- [`GasolineYield` documentation](https://search.r-project.org/CRAN/refmans/betareg/html/GasolineYield.html) описывает небольшой классический dataset с долей выхода бензина строго внутри единичного интервала.
- [French MTPL frequency analysis](https://dutangc.perso.math.cnrs.fr/RRepository/pub/web/vignettes/poisson_vignette.html) документирует actuarial frequency use case для `freMTPL`/`freMTPL2`; frequency и severity tables удобны для финального end-to-end примера.
