# GPUAutotuner

Milestone 1 of a hardware-aware CUDA GEMM autotuner: a reproducible correctness
foundation and one parameterized tiled GEMM kernel. This version intentionally has no
performance timing, search, DOE, or response-surface implementation.

## Requirements

- CMake 3.24 or newer
- A C++17 compiler
- NVIDIA CUDA Toolkit with `nvcc`
- A CUDA-capable GPU and compatible driver to execute the programs

No Python framework or binding is used.

## Start here

1. Read [the short-term project plan](docs/next-72-hours.md).
2. Record the GPU, driver, CUDA toolkit, compiler, and clock/power settings in
   [the environment record](experiments/environment.md).
3. Build and pass the Milestone 1 correctness suite before collecting performance data
   or implementing DOE/RSM.

## Configure and build

Choose the CUDA architecture for the target GPU at configure time. For example, use
`86` for a device whose desired CMake architecture is 86:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release
```

If `CMAKE_CUDA_ARCHITECTURES` is omitted, CMake uses the CUDA compiler's default. The
project does not hard-code a particular GPU.

## Run

The standalone program accepts `M N K [relative_tolerance]`:

```sh
./build/gemm_experiment 127 131 37 1e-4
```

It creates deterministic inputs, compares the tiled GPU result with a CPU reference,
and prints device, CUDA, configuration, launch, shared-memory, and error metadata.

Two introductory standalone examples from the project foundation are also retained:

- `vector_add` demonstrates one thread producing one output element.
- `naive_gemm` is a small square-matrix CUDA GEMM example. The tested naive
  reference used by Milestone 1 itself remains part of `gemm_kernels`.

Run all five correctness shapes through CTest:

```sh
ctest --test-dir build --output-on-failure -C Release
```

The correctness executable also accepts an optional relative tolerance, defaulting to
`1e-4`: `correctness_tests 1e-4`.

If Compute Sanitizer is installed, run:

```sh
compute-sanitizer --tool memcheck ./build/correctness_tests
```

On multi-configuration generators the executable may instead be under
`build/Release/`. See [the kernel mapping](docs/kernel_mapping.md) for the exact thread
and memory indexing model and [the frozen experiment configuration](configs/experiment.yaml)
for numerical semantics.

## Research planning documents

Future work is outlined in [the experiment protocol](docs/experiment-protocol.md),
[publication roadmap](docs/publication-roadmap.md), and
[reading list](references/reading-list.md). These documents describe later stages;
their timing, search, DOE, and RSM work is intentionally not implemented in Milestone 1.
