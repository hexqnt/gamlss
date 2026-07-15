# Numerical-kernel benchmarks

Run the complete suite with:

```bash
cargo bench -p gamlss-special --bench numeric_kernels
```

Use a Criterion filter for one numerical family, for example:

```bash
cargo bench -p gamlss-special --bench numeric_kernels -- "normal_functions"
```

The suite measures batches of deterministic inputs so Criterion overhead does not dominate small scalar kernels. Gamma/beta benchmarks cover ordinary, reflection, large-argument and imbalanced-shape paths, including multivariate beta at `D = 4/32`. Regularized beta/gamma and Student-t CDF cases cover central, tail and large-shape saddlepoint regimes. Normal helpers cover central and far-tail CDF/log-CDF/Mills-ratio paths, central/tail quantiles and mixed-regime Owen's T evaluation.

Input generation happens outside timed iterations. Results are local comparison data, not accuracy checks, portable guarantees or CI thresholds; numerical correctness remains covered by deterministic and `statrs` reference tests.
