# Литература и дополнительные материалы

Источники ниже полезны как статистический ориентир, коллекция удачных учебных ходов и набор пограничных случаев.

Библиотека не привязана к конкретному оптимизатору, поэтому пайплайн оптимизации нужно реализовать самостоятельно. Для знакомства с методами математической оптимизации рекомендуем сайт [Fmin.xyz](https://fmin.xyz/).

## Книги и обзорные статьи

- [Generalized Additive Models for Location, Scale and Shape: A Distributional Regression Approach, with Applications](https://gamlssbook.bitbucket.io/) — современная книга 2024 года и открытый сайт-сопровождение с кодом, рисунками и наборами данных. Полезна разделением материала на основы, статистический вывод и полноценные прикладные разборы.
- [Flexible Regression and Smoothing: Using GAMLSS in R](https://www.routledge.com/Flexible-Regression-and-Smoothing-Using-GAMLSS-in-R/Stasinopoulos-Rigby-Heller-Voudouris-Bastiani/p/book/9781138197909) — практическая книга о построении моделей, аддитивных членах, отборе, диагностике и оценивании центилей. Особенно полезен приём сравнивать несколько классов регрессии на одних данных.
- [Distributions for Modeling Location, Scale, and Shape: Using GAMLSS in R](https://www.routledge.com/Distributions-for-Modeling-Location-Scale-and-Shape-Using-GAMLSS-in-R/Rigby-Stasinopoulos-Heller-De-Bastiani/p/book/9780429298547) — справочник по непрерывным, счётным и смешанным распределениям, их носителям, асимметрии и хвостам. Для этой книги он служит справочником, а не образцом линейного порядка глав.
- [Rigby and Stasinopoulos (2005), Generalized additive models for location, scale and shape](https://doi.org/10.1111/j.1467-9876.2005.00510.x) — исходная статья с общей постановкой GAMLSS, штрафным правдоподобием и аддитивными предикторами.
- [Distributional regression using generalized additive models for location, scale and shape](https://www.nature.com/articles/s43586-026-00498-z) — недавний обзор 2026 года, связывающий GAMLSS с более широкой областью вероятностной регрессии.

## Воспроизводимые примеры анализа

- [gamlss2: First Steps](https://gamlss-dev.github.io/gamlss2/vignettes/firststeps.html) моделирует распределение дневной максимальной температуры и превращает его в вероятность жаркого дня. Это хороший образец главы, которая заканчивается решением прикладного вопроса, а не таблицей коэффициентов.
- [gamlss2: Centile (Quantile) Estimation](https://gamlss-dev.github.io/gamlss2/vignettes/quantiles.html) проходит маршрут `данные -> оценивание -> диагностика -> центильные кривые` на `dbbmi`. Этот маршрут положен в основу запланированного прикладного разбора кривых роста.
- [GAMLSS companion datasets](https://gamlssbook.bitbucket.io/Datasets.html) публикует наборы данных из прикладных разборов в CSV для переноса в другие программы. Перед включением копии в репозиторий всё равно нужно проверить указание авторства и лицензию исходного набора.
- [GAMLSS short course: Distributional regression](https://mstasinopoulos.github.io/ShortCourse/regression.html) показывает короткие сравнительные разборы и содержит полезные численные эталоны.

## Диагностика и оценка прогнозов

- [Dunn and Smyth (1996), Randomized quantile residuals](https://doi.org/10.1080/10618600.1996.10474708) объясняет квантильные остатки и необходимую рандомизацию для дискретных откликов. Это ключевой источник для будущего расширения диагностики счётных моделей.
- [Gneiting and Raftery (2007), Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) даёт основу для логарифмической оценки, CRPS и корректного сравнения вероятностных прогнозов.

## Наборы данных

- [`gamlss.data` reference manual](https://gamlss-dev.r-universe.dev/gamlss.data/doc/manual.html) описывает `dbbmi`, `abdom`, `speech` и другие классические наборы данных вместе с происхождением и структурой переменных.
- [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) содержит почасовые и посуточные числа поездок, погоду и календарные признаки и распространяется под CC BY 4.0.
- [`GasolineYield` documentation](https://search.r-project.org/CRAN/refmans/betareg/html/GasolineYield.html) описывает небольшой классический набор данных с долей выхода бензина строго внутри единичного интервала.
- [French MTPL frequency analysis](https://dutangc.perso.math.cnrs.fr/RRepository/pub/web/vignettes/poisson_vignette.html) документирует задачу моделирования частоты страховых случаев для `freMTPL`/`freMTPL2`; таблицы частоты и тяжести убытка удобны для заключительного сквозного примера.
