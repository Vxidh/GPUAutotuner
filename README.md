# Hardware-Aware CUDA Autotuning with Response Surface Methodology

This workspace turns the project synopsis into a reproducible research project. The core question is:

> Can a two-stage DOE/RSM tuner identify near-optimal CUDA kernel launch and tiling parameters with fewer measurements than established search baselines?

## Start here

1. Read `docs/next-72-hours.md`.
2. Record the exact GPU, driver, CUDA toolkit, compiler, and clock/power settings in `experiments/environment.md`.
3. Build one correct, parameterized matrix-multiplication kernel before implementing the optimizer.

## Day 1 implementation

Two correctness-first programs are now available:

- `kernels/vector_add.cu` — the basic CUDA execution model: one thread, one output value.
- `kernels/naive_gemm.cu` — a deliberately untuned matrix multiplication baseline; it is the reference point for the later shared-memory tiled kernel.

Run the commands in `experiments/environment.md`. Do not treat timings from either program as research data yet.

## Folder map

- `kernels/` — parameterized CUDA templates and correctness tests.
- `src/` — C++ tuning runner, design generation, and model fitting.
- `scripts/` — data analysis and plotting.
- `experiments/` — immutable experiment specifications and environment records.
- `data/raw/` — append-only measurement CSV/Parquet files.
- `data/processed/` — cleaned datasets and fitted-model outputs.
- `results/figures/` — paper-ready plots.
- `paper/` — manuscript, tables, and submission checklist.
- `references/` — reading notes and bibliography.

## Minimum viable research contribution

Do not claim an “absolute minimum” latency. The defensible claim is a measured comparison: on a fixed set of kernels, shapes, and GPUs, your DOE/RSM procedure reaches a specified fraction of the exhaustive-search best latency using fewer evaluations, with uncertainty reported.

See `docs/publication-roadmap.md` and `docs/experiment-protocol.md`.
