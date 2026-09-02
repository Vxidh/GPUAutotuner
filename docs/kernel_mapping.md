# Mapping a Matrix Multiplication onto CUDA

This milestone computes row-major `C = A * B` with FP32 inputs, output, and
accumulation. It establishes correctness only; it contains no timing or autotuning.

## Host code and device code

The **host** is the CPU. It parses dimensions, allocates memory, copies inputs to the
GPU, launches work, copies results back, and checks errors. See `main` in
`src/main.cu` and the checked calls wrapped by `CUDA_CHECK` in
`src/cuda_check.hpp`.

The **device** is the GPU. A function marked `__global__` is a **kernel**: host code
launches it, and many GPU threads execute it. This project has
`naive_gemm_kernel` and `tiled_gemm_kernel` in `src/gemm_kernel.cu`.

## Grid, blocks, threads, and warps

A kernel launch creates a grid of blocks. Each block contains threads. CUDA exposes
the current block as `blockIdx`, the current thread inside that block as `threadIdx`,
and the block size as `blockDim`. NVIDIA schedules threads in groups of 32 called
warps. The fixed tiled block is `(16, 16, 1)`, or 256 threads (eight warps).

Blocks do not synchronize with each other. Threads in one block can share fast
on-chip shared memory and synchronize with `__syncthreads()`.

## Why the naive kernel is correct

The naive launch uses 16 by 16 threads. One thread computes one output element:

```text
row = blockIdx.y * blockDim.y + threadIdx.y
col = blockIdx.x * blockDim.x + threadIdx.x
C[row, col] = sum from inner=0 to K-1 of A[row, inner] * B[inner, col]
```

For row-major arrays, element `(row, col)` of an array with `width` columns is at
`row * width + col`. Therefore the exact addresses are:

```text
A[row, inner] -> A[row * K + inner]
B[inner, col] -> B[inner * N + col]
C[row, col]   -> C[row * N + col]
```

The guard `row < M && col < N` prevents accesses by surplus threads in edge blocks.
These equations appear directly in `naive_gemm_kernel`.

Example: for `M=2, N=3, K=2`, thread `(col=1,row=0)` computes
`C[0*3+1] = A[0*2+0]*B[0*3+1] + A[0*2+1]*B[1*3+1]`.

## Tiling vocabulary

The fixed values are:

| Symbol | Value | Meaning |
|---|---:|---|
| `BM` | 64 | output rows owned by one block |
| `BN` | 64 | output columns owned by one block |
| `BK` | 16 | reduction (`K`) values loaded per phase |
| `TM` | 4 | output rows owned by one thread |
| `TN` | 4 | output columns owned by one thread |

The grid is `(ceil(N/BN), ceil(M/BM), 1)`. The block is
`(BN/TN, BM/TM, 1) = (16,16,1)`. Thus 256 threads each own 16 results, covering
the block's `64*64 = 4096` results exactly.

## From a thread to its 4 by 4 micro-tile

For a block and thread `(threadIdx.x, threadIdx.y)`, the implementation uses:

```text
block_row = blockIdx.y * BM
block_col = blockIdx.x * BN
micro_row = threadIdx.y * TM
micro_col = threadIdx.x * TN
global_row = block_row + micro_row + local_row
global_col = block_col + micro_col + local_col
```

Here `local_row` and `local_col` each range from 0 through 3. The thread holds
`accumulators[4][4]` in registers. For example, thread `(x=2,y=3)` in block
`(x=1,y=0)` starts at global output `(12,72)` because `0*64+3*4=12` and
`1*64+2*4=72`. It owns rows 12-15 and columns 72-75.

## Cooperative shared-memory loading

Each phase needs `BM*BK = 1024` A values and `BK*BN = 1024` B values. The 256
threads are flattened as:

```text
thread_linear = threadIdx.y * blockDim.x + threadIdx.x
thread_count = blockDim.x * blockDim.y = 256
```

Each thread loads indices `thread_linear`, `thread_linear+256`, and so on. A linear
A-tile index becomes `(local_row=linear/BK, local_k=linear%BK)`. A linear B-tile
index becomes `(local_k=linear/BN, local_col=linear%BN)`. Consequently every thread
loads four A values and four B values for this configuration.

Global coordinates in K phase `tile_k` are:

```text
A: global_row = block_row + local_row
   global_k   = tile_k + local_k
B: global_k   = tile_k + local_k
   global_col = block_col + local_col
```

An invalid coordinate stores zero into shared memory. This is what makes partial M,
N, and K tiles behave as though matrices were zero-padded.

## Accumulation and synchronization

After loading, `__syncthreads()` ensures all 256 threads have finished writing shared
memory before any thread reads it. For every `local_k` from 0 through 15, a thread
performs:

```text
accumulators[local_row][local_col] +=
    shared_a[micro_row + local_row][local_k] *
    shared_b[local_k][micro_col + local_col]
```

A second `__syncthreads()` ensures all reads are finished before any thread overwrites
shared memory for the next K phase. Omitting either barrier creates a data race.

If `K` is not divisible by 16, invalid loads in the last phase become zero, so all 16
loop iterations remain valid without changing the answer. Final stores independently
guard `global_row < M` and `global_col < N`, which supports matrices smaller than a
tile and partial edge tiles.

The compile-time divisibility assertions are in `tiled_gemm_kernel`; device thread and
shared-memory limits are checked by `validate_fixed_tiled_feasibility` before launch.

