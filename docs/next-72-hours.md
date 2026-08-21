# First 72 hours

## Day 1 — establish a trustworthy baseline

1. Install/verify NVIDIA driver, CUDA Toolkit, CMake, and a C++ compiler compatible with your CUDA version.
2. Create a Git repository in this folder and make an initial commit.
3. Choose **one** initial kernel: tiled FP32 GEMM. Use fixed, square matrices first (for example 1024×1024), then add two more realistic shapes.
4. Write a correctness oracle: compare against cuBLAS (or a CPU reference for small matrices) with a stated tolerance.
5. Benchmark only GPU execution using CUDA events; warm up before measurement and repeat each configuration many times.

## Day 2 — define the design space

Start with 3–4 factors only:

- `BM`, `BN`: output tile dimensions
- `BK`: reduction tile dimension
- `TM` or `TN`: per-thread work / unroll factor

Derive validity constraints: threads/block ≤ 1024, dimensions divisible where required, shared-memory use within the device limit, and register/occupancy constraints. Keep all rejected configurations with a reason.

## Day 3 — collect the first dataset

1. Run a small exhaustive sweep over the feasible space. This is your oracle for this initial study—not something the final tuner uses.
2. Run a screening design, fit a first model, and compare predicted optimum against the exhaustive optimum on held-out configurations.
3. Save raw results, random seeds, compiler flags, and hardware details. Generate one latency heatmap or contour plot.

## Success checkpoint

You should have: a correct kernel, a reproducible timing harness, one complete dataset, and a plot. Only then add fractional factorial screening, CCD, and constrained optimization.
