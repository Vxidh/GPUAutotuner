// Host harness for one compiled kernel configuration (macros injected via
// nvcc -D, see python/orchestrator.py). Prints exactly one JSON line.
//
// Usage:
//   ./bench M N K iters warmup [--verify] [--compare B MODE] [--seed S]
//
//   iters     timed kernel launches per measurement block; each launch is
//             timed individually with its own CUDA event pair
//   warmup    untimed launches before timing
//   --verify  relative Frobenius error vs cuBLAS SGEMM on the same inputs
//   --compare B MODE
//             interleaved custom-vs-cuBLAS benchmark: B blocks, each timing
//             `iters` custom launches and `iters` cuBLAS launches, with the
//             order alternating per block. MODE = default | pedantic
//             (cuBLAS math mode; both are pure FP32, pedantic additionally
//             forbids any reduced-precision or reordering shortcuts)
//
// Timing boundary: only the kernel (or cublasSgemm) between CUDA events on
// the default stream. Allocation, initialisation, transfers, verification
// and compilation are all outside it.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "gemm_kernel.cuh"

static std::string g_status = "OK";

#define CUDA_TRY(call)                                                   \
  do {                                                                   \
    cudaError_t e__ = (call);                                            \
    if (e__ != cudaSuccess) {                                            \
      g_status = std::string("launch_invalid:") + cudaGetErrorString(e__); \
      emit_failure();                                                    \
      return 0;                                                          \
    }                                                                    \
  } while (0)

static int M_, N_, K_;
static void emit_failure() {
  printf("{\"status\":\"%s\",\"BM\":%d,\"BN\":%d,\"BK\":%d,\"TM\":%d,\"TN\":%d,"
         "\"UNROLL\":%d,\"M\":%d,\"N\":%d,\"K\":%d}\n",
         g_status.c_str(), BM, BN, BK, TM, TN, UNROLL, M_, N_, K_);
}

// Deterministic, order-independent input generation on the device:
// uniform in [-1, 1) from a splitmix64 hash of (seed, index).
__global__ void init_uniform(float *x, size_t n, unsigned long long seed) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long long z = seed + 0x9E3779B97F4A7C15ULL * (i + 1);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z ^= z >> 31;
  x[i] = (float)((z >> 40) * (1.0 / 16777216.0)) * 2.0f - 1.0f;
}

static void print_list(const char *name, const std::vector<float> &v) {
  printf("\"%s\":[", name);
  for (size_t i = 0; i < v.size(); ++i) printf(i ? ",%.6f" : "%.6f", v[i]);
  printf("]");
}

static void launch_custom(dim3 grid, dim3 block, int M, int N, int K,
                          const float *A, const float *B, float *C) {
  sgemm_tunable<<<grid, block>>>(M, N, K, 1.0f, A, B, 0.0f, C);
}

// Row-major C = A*B via column-major cuBLAS: C^T = B^T * A^T.
static cublasStatus_t launch_cublas(cublasHandle_t h, int M, int N, int K,
                                    const float *A, const float *B, float *C) {
  const float alpha = 1.0f, beta = 0.0f;
  return cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K,
                     &beta, C, N);
}

int main(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: %s M N K iters warmup [--verify] [--compare B MODE] [--seed S]\n", argv[0]);
    return 2;
  }
  M_ = atoi(argv[1]); N_ = atoi(argv[2]); K_ = atoi(argv[3]);
  const int M = M_, N = N_, K = K_;
  const int iters = atoi(argv[4]);
  const int warmup = atoi(argv[5]);
  bool verify = false, cpu_ref = false;
  int compare_blocks = 0;
  std::string cublas_mode = "default";
  unsigned long long seed = 42;
  for (int i = 6; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--verify") verify = true;
    else if (a == "--cpu-ref") { verify = true; cpu_ref = true; }
    else if (a == "--compare" && i + 2 < argc) { compare_blocks = atoi(argv[++i]); cublas_mode = argv[++i]; }
    else if (a == "--seed" && i + 1 < argc) seed = strtoull(argv[++i], nullptr, 10);
  }

  if (M % BM || N % BN || K % BK) {
    g_status = "shape_invalid:not_divisible_by_tile";
    emit_failure();
    return 0;
  }

  // Compiled resource usage: registers per thread and local memory per
  // thread (local memory > 0 means the compiler spilled registers).
  cudaFuncAttributes attr;
  CUDA_TRY(cudaFuncGetAttributes(&attr, sgemm_tunable));
  int blocks_per_sm = 0;
  CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, sgemm_tunable, NUM_THREADS, 0));
  cudaDeviceProp prop;
  CUDA_TRY(cudaGetDeviceProperties(&prop, 0));
  double occupancy = (double)(blocks_per_sm * NUM_THREADS) / prop.maxThreadsPerMultiProcessor;

  size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;
  float *dA, *dB, *dC, *dRef = nullptr;
  CUDA_TRY(cudaMalloc(&dA, nA * sizeof(float)));
  CUDA_TRY(cudaMalloc(&dB, nB * sizeof(float)));
  CUDA_TRY(cudaMalloc(&dC, nC * sizeof(float)));
  init_uniform<<<(unsigned)((nA + 255) / 256), 256>>>(dA, nA, seed);
  init_uniform<<<(unsigned)((nB + 255) / 256), 256>>>(dB, nB, seed ^ 0xABCDEF1234567ULL);
  CUDA_TRY(cudaMemset(dC, 0, nC * sizeof(float)));
  CUDA_TRY(cudaDeviceSynchronize());

  dim3 grid(N / BN, M / BM);
  dim3 block(NUM_THREADS);

  cublasHandle_t handle = nullptr;
  int cublas_version = 0;
  if (verify || compare_blocks > 0) {
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
      g_status = "launch_invalid:cublasCreate";
      emit_failure();
      return 0;
    }
    cublasGetVersion(handle, &cublas_version);
    cublasSetMathMode(handle, cublas_mode == "pedantic" ? CUBLAS_PEDANTIC_MATH : CUBLAS_DEFAULT_MATH);
  }

  // ---- correctness: relative Frobenius error vs cuBLAS ----
  double rel_fro = NAN, max_abs = NAN;
  double cpu_err_custom = NAN, cpu_err_cublas = NAN;
  bool all_finite = true;
  if (verify) {
    CUDA_TRY(cudaMalloc(&dRef, nC * sizeof(float)));
    CUDA_TRY(cudaMemset(dC, 0, nC * sizeof(float)));
    launch_custom(grid, block, M, N, K, dA, dB, dC);
    CUDA_TRY(cudaGetLastError());
    if (launch_cublas(handle, M, N, K, dA, dB, dRef) != CUBLAS_STATUS_SUCCESS) {
      g_status = "launch_invalid:cublasSgemm";
      emit_failure();
      return 0;
    }
    CUDA_TRY(cudaDeviceSynchronize());
    std::vector<float> hC(nC), hR(nC);
    CUDA_TRY(cudaMemcpy(hC.data(), dC, nC * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(hR.data(), dRef, nC * sizeof(float), cudaMemcpyDeviceToHost));
    double num = 0.0, den = 0.0;
    max_abs = 0.0;
    for (size_t i = 0; i < nC; ++i) {
      if (!std::isfinite(hC[i])) all_finite = false;
      double d = (double)hC[i] - (double)hR[i];
      num += d * d;
      den += (double)hR[i] * hR[i];
      max_abs = std::max(max_abs, std::fabs(d));
    }
    rel_fro = std::sqrt(num) / std::max(std::sqrt(den), 1e-30);

    // Independent check of both GPU results against a float64 CPU reference
    // (small shapes only), so agreement with cuBLAS is not the only evidence.
    if (cpu_ref) {
      std::vector<float> hA(nA), hB(nB);
      CUDA_TRY(cudaMemcpy(hA.data(), dA, nA * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_TRY(cudaMemcpy(hB.data(), dB, nB * sizeof(float), cudaMemcpyDeviceToHost));
      std::vector<double> ref(nC, 0.0);
      for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k) {
          double a = hA[(size_t)i * K + k];
          for (int j = 0; j < N; ++j) ref[(size_t)i * N + j] += a * hB[(size_t)k * N + j];
        }
      double nc = 0, nr = 0, dd = 0;
      for (size_t i = 0; i < nC; ++i) {
        nc += (hC[i] - ref[i]) * (hC[i] - ref[i]);
        nr += (hR[i] - ref[i]) * (hR[i] - ref[i]);
        dd += ref[i] * ref[i];
      }
      cpu_err_custom = std::sqrt(nc / dd);
      cpu_err_cublas = std::sqrt(nr / dd);
    }
  }

  // ---- timing ----
  std::vector<cudaEvent_t> ev0(iters), ev1(iters);
  for (int i = 0; i < iters; ++i) { cudaEventCreate(&ev0[i]); cudaEventCreate(&ev1[i]); }

  auto time_custom = [&](std::vector<float> &out) -> bool {
    for (int i = 0; i < iters; ++i) {
      cudaEventRecord(ev0[i]);
      launch_custom(grid, block, M, N, K, dA, dB, dC);
      cudaEventRecord(ev1[i]);
    }
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return false;
    for (int i = 0; i < iters; ++i) { float ms; cudaEventElapsedTime(&ms, ev0[i], ev1[i]); out.push_back(ms); }
    return true;
  };
  auto time_cublas = [&](std::vector<float> &out) -> bool {
    for (int i = 0; i < iters; ++i) {
      cudaEventRecord(ev0[i]);
      launch_cublas(handle, M, N, K, dA, dB, dC);
      cudaEventRecord(ev1[i]);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) return false;
    for (int i = 0; i < iters; ++i) { float ms; cudaEventElapsedTime(&ms, ev0[i], ev1[i]); out.push_back(ms); }
    return true;
  };

  for (int i = 0; i < warmup; ++i) launch_custom(grid, block, M, N, K, dA, dB, dC);
  if (compare_blocks > 0)
    for (int i = 0; i < warmup; ++i) launch_cublas(handle, M, N, K, dA, dB, dC);
  CUDA_TRY(cudaGetLastError());
  CUDA_TRY(cudaDeviceSynchronize());

  std::vector<float> t_custom, t_cublas;
  std::vector<int> block_order;  // 0 = custom first, 1 = cuBLAS first
  if (compare_blocks <= 0) {
    if (!time_custom(t_custom)) { g_status = "launch_invalid:timed_loop"; emit_failure(); return 0; }
  } else {
    for (int b = 0; b < compare_blocks; ++b) {
      bool cublas_first = (b % 2) == 1;
      block_order.push_back(cublas_first);
      bool ok = cublas_first ? (time_cublas(t_cublas) && time_custom(t_custom))
                             : (time_custom(t_custom) && time_cublas(t_cublas));
      if (!ok) { g_status = "launch_invalid:compare_loop"; emit_failure(); return 0; }
    }
  }

  if (verify && !all_finite) g_status = "numerically_invalid:non_finite";

  char rel_s[32] = "null", abs_s[32] = "null", cc_s[32] = "null", cb_s[32] = "null";
  if (verify) {
    snprintf(rel_s, sizeof rel_s, "%.6e", rel_fro);
    snprintf(abs_s, sizeof abs_s, "%.6e", max_abs);
  }
  if (cpu_ref) {
    snprintf(cc_s, sizeof cc_s, "%.6e", cpu_err_custom);
    snprintf(cb_s, sizeof cb_s, "%.6e", cpu_err_cublas);
  }

  printf("{\"status\":\"%s\",\"BM\":%d,\"BN\":%d,\"BK\":%d,\"TM\":%d,\"TN\":%d,\"UNROLL\":%d,"
         "\"M\":%d,\"N\":%d,\"K\":%d,\"threads\":%d,\"smem_bytes\":%d,"
         "\"regs_per_thread\":%d,\"local_bytes_per_thread\":%zu,\"blocks_per_sm\":%d,"
         "\"occupancy\":%.4f,\"verified\":%s,\"rel_fro_err\":%s,\"max_abs_err\":%s,"
         "\"cpu_f64_err_custom\":%s,\"cpu_f64_err_cublas\":%s,"
         "\"cublas_version\":%d,\"cublas_math_mode\":\"%s\",\"seed\":%llu,"
         "\"iters\":%d,\"warmup\":%d,",
         g_status.c_str(), BM, BN, BK, TM, TN, UNROLL, M, N, K, NUM_THREADS,
         SMEM_BYTES, attr.numRegs, attr.localSizeBytes, blocks_per_sm, occupancy,
         verify ? "true" : "false", rel_s, abs_s, cc_s, cb_s,
         cublas_version, (verify || compare_blocks > 0) ? cublas_mode.c_str() : "none",
         seed, iters, warmup);
  print_list("t_custom_ms", t_custom);
  printf(",");
  print_list("t_cublas_ms", t_cublas);
  printf(",\"block_order_cublas_first\":[");
  for (size_t i = 0; i < block_order.size(); ++i) printf(i ? ",%d" : "%d", block_order[i]);
  printf("]}\n");

  if (handle) cublasDestroy(handle);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  if (dRef) cudaFree(dRef);
  return 0;
}
