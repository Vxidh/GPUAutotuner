# Experiment protocol

## Research hypotheses

- **H1:** Screening removes inactive factors without materially reducing attainable performance.
- **H2:** A quadratic response surface predicts near-optimal feasible configurations within a small evaluation budget.
- **H3:** DOE/RSM achieves a comparable best latency to a baseline search at lower sample cost on the stated benchmark suite.

## Measurements

- Time kernels with CUDA events, after warm-up, in a controlled stream.
- Use at least 30 independent repetitions per configuration; report median and spread (IQR or 95% CI), not only one best run.
- Randomize experiment order. Re-run center points through the schedule to estimate drift/noise.
- Log: GPU model/compute capability, driver, CUDA/toolkit, compiler, flags, clocks/power mode, problem shape, parameters, validity status, elapsed time, and error/correctness result.

## Comparators

1. Exhaustive search over the same feasible discrete space (evaluation oracle).
2. Random search with identical evaluation budgets and multiple seeds.
3. One established autotuning baseline where feasible (for example Kernel Tuner, or a clearly documented simpler search baseline).
4. cuBLAS only as a production-library reference; it is not necessarily a fair equivalent parameter-space baseline.

## Analysis

- Screening: fractional factorial design, ANOVA/effect estimates, diagnostic checks.
- Modeling: CCD or Box–Behnken over retained factors; quadratic OLS; report adjusted R², cross-validated error, residual diagnostics, and lack-of-fit where applicable.
- Optimization: maximize predicted performance / minimize predicted latency subject to discrete feasibility; round candidates carefully, compile, and **measure** the final configuration.
- Report tuning cost (number of compilations/evaluations and wall-clock tuning time) separately from kernel runtime.

## Threats to validity

GPU clock variation, thermal throttling, compilation/JIT effects, a single GPU, a single kernel family, and an overly small or invalid factor domain can all invalidate broad claims. State these limits explicitly.
