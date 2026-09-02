#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace gemm {

inline constexpr int kBM = 64;
inline constexpr int kBN = 64;
inline constexpr int kBK = 16;
inline constexpr int kTM = 4;
inline constexpr int kTN = 4;

struct LaunchMetadata {
    dim3 grid;
    dim3 block;
    std::size_t static_shared_bytes;
    std::size_t dynamic_shared_bytes;
};

void validate_problem(int m, int n, int k);
void validate_fixed_tiled_feasibility(int device);

LaunchMetadata launch_naive_gemm(const float* a, const float* b, float* c,
                                 int m, int n, int k,
                                 cudaStream_t stream = nullptr);

LaunchMetadata launch_fixed_tiled_gemm(const float* a, const float* b, float* c,
                                       int m, int n, int k,
                                       cudaStream_t stream = nullptr);

}  // namespace gemm

