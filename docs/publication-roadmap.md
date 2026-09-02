# From project to paper

## Is publication possible?

Yes, but the synopsis alone is not a paper. A publishable result needs a clear empirical contribution, rigorous baselines, reproducibility, and claims matched to the evidence. A class project can credibly become a workshop, student conference, or systems/performance venue submission if the results are strong and your supervisor supports the submission.

## What you need to learn

1. **CUDA performance:** memory coalescing, shared memory, occupancy, registers, warps, numerical correctness, Nsight Compute.
2. **Benchmarking:** CUDA events, warm-ups, repeated trials, randomization, clock/thermal controls, error bars.
3. **DOE/RSM:** factorial/fractional factorial designs, aliasing/resolution, ANOVA, CCD, quadratic regression, residual and lack-of-fit diagnostics.
4. **Constrained discrete optimization:** feasible-set encoding, integer rounding/repair, validation of predicted optima.
5. **Research practice:** literature review, baselines, ablations, preregistered/locked protocol where possible, Git, environment capture, and reproducible artifacts.

## A credible contribution statement

Aim to prove a narrow statement such as: “Across these N kernels, shapes, and G GPUs, our two-stage DOE/RSM tuner reaches at least X% of the exhaustive-search optimum with Y% fewer evaluations than random search, while exposing interpretable factor effects.”

Avoid claiming superiority over TVM/Ansor unless you run a fair, reproducible comparison under matched budgets and target workloads.

## Evidence required before writing

- At least 2–3 kernel families, several input shapes, and ideally ≥2 GPU architectures.
- An exhaustive oracle for tractable spaces; random search with multiple seeds; one practical baseline if available.
- An ablation: screening only, RSM only, full method, and sensitivity to evaluation budget.
- Statistical uncertainty and failure cases, not just the best plot.
- Public code, parameter files, raw/processed results, exact environment, and a one-command reproduction path.

## Paper structure

1. Problem and contribution (precise scope).
2. Background and related work.
3. Method: feasible space → screening → RSM → discrete validation.
4. Experimental setup and reproducibility.
5. Results: quality, sample efficiency, tuning time, ablations, diagnostics.
6. Limitations and threats to validity.
7. Artifact/repository link.

## Submission sequence

Discuss authorship, target venue, and data/code release with Prof. Sharma before any public upload. First write a 2-page internal results memo; then select a venue whose scope and deadline fit the finished evidence. Do not pick a venue before seeing your experimental results.
