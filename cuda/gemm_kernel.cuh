// Tunable 2D-blocktiled SGEMM kernel: C = alpha * A * B + beta * C,
// row-major A (MxK), B (KxN), C (MxN), FP32 inputs, FP32 accumulation,
// ordinary CUDA-core FMA arithmetic (no Tensor Cores).
//
// Tunable hardware parameters are injected at compile time via -D macros:
//   BM  - block output tile rows      (shared-mem tile size)
//   BN  - block output tile cols      (shared-mem tile size)
//   BK  - K-dim chunk loaded per iter (shared-mem tile depth)
//   TM  - per-thread output tile rows (register tile -> thread block shape)
//   TN  - per-thread output tile cols (register tile -> thread block shape)
//   UNROLL - pragma unroll factor for the inner dot-product loop
//
// The kernel requires M % BM == 0, N % BN == 0, K % BK == 0 (no boundary
// guards); the host harness rejects shapes that violate this.
#pragma once

#ifndef BM
#define BM 64
#endif
#ifndef BN
#define BN 64
#endif
#ifndef BK
#define BK 8
#endif
#ifndef TM
#define TM 4
#endif
#ifndef TN
#define TN 4
#endif
#ifndef UNROLL
#define UNROLL 1
#endif

#define NUM_THREADS ((BM * BN) / (TM * TN))
#define SMEM_BYTES ((BM * BK + BK * BN) * 4)

// Feasibility constraints (mirrored in python/design_space.py::is_valid).
// A violating configuration must fail to compile rather than run incorrectly.
static_assert(BM % TM == 0 && BN % TN == 0, "thread tile must divide block tile");
static_assert((BM * BN) % (TM * TN) == 0, "threads per block must be an integer");
static_assert(NUM_THREADS <= 1024, "threads per block exceeds 1024");
static_assert(SMEM_BYTES <= 49152, "static shared memory exceeds 48 KiB");
static_assert(NUM_THREADS % BK == 0, "A-tile load stride must be an integer");
static_assert(NUM_THREADS % BN == 0, "B-tile load stride must be an integer");
// Tile loads: each thread copies one element per pass, so the threads of one
// pass must fit inside the tile, and the passes must tile it exactly.
// Violating these writes past the end of As/Bs.
static_assert(NUM_THREADS <= BM * BK, "A-tile load overflow: threads > BM*BK");
static_assert(NUM_THREADS <= BK * BN, "B-tile load overflow: threads > BK*BN");
static_assert(BM % (NUM_THREADS / BK) == 0, "A-tile rows not covered exactly");
static_assert(BK % (NUM_THREADS / BN) == 0, "B-tile rows not covered exactly");

__global__ __launch_bounds__(NUM_THREADS) void sgemm_tunable(
    int M, int N, int K, float alpha, const float *A, const float *B,
    float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint threadCol = threadIdx.x % (BN / TN);
  const uint threadRow = threadIdx.x / (BN / TN);

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += cRow * BM * (uint)K;
  B += cCol * BN;
  C += cRow * BM * (uint)N + cCol * BN;

  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  constexpr uint strideA = NUM_THREADS / BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  constexpr uint strideB = NUM_THREADS / BN;

  float threadResults[TM * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

#ifdef INJECT_FAULT
  // Negative control for the correctness check: silently drop the last
  // K-tile. Must be detected as numerically_invalid.
  const uint kEnd = (uint)K - BK;
#else
  const uint kEnd = (uint)K;
#endif

  for (uint bkIdx = 0; bkIdx < kEnd; bkIdx += BK) {
#pragma unroll
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * (uint)K + innerColA];
    }
#pragma unroll
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * (uint)N + innerColB];
    }
    __syncthreads();

    A += BK;
    B += BK * (uint)N;

    constexpr int kUnroll = UNROLL;
#pragma unroll kUnroll
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
#pragma unroll
      for (uint i = 0; i < TM; ++i)
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
#pragma unroll
      for (uint i = 0; i < TN; ++i)
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
#pragma unroll
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
#pragma unroll
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
    }
    __syncthreads();
  }

#pragma unroll
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
#pragma unroll
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      uint idx = (threadRow * TM + resIdxM) * (uint)N + threadCol * TN + resIdxN;
      C[idx] = alpha * threadResults[resIdxM * TN + resIdxN] + beta * C[idx];
    }
  }
}
