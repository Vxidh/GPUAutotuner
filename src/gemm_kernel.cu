#include "gemm_kernel.cuh"

#include "cuda_check.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <sstream>
#include <stdexcept>

namespace gemm {
namespace {

__global__ void naive_gemm_kernel(const float* a, const float* b, float* c,
                                  int m, int n, int k) {
    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= m || col >= n) {
        return;
    }

    float accumulator = 0.0F;
    for (int inner = 0; inner < k; ++inner) {
        accumulator += a[static_cast<std::size_t>(row) * k + inner] *
                       b[static_cast<std::size_t>(inner) * n + col];
    }
    c[static_cast<std::size_t>(row) * n + col] = accumulator;
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void tiled_gemm_kernel(const float* a, const float* b, float* c,
                                  int m, int n, int k) {
    static_assert(BM % TM == 0, "BM must be divisible by TM");
    static_assert(BN % TN == 0, "BN must be divisible by TN");
    static_assert((BM / TM) * (BN / TN) <= 1024,
                  "CUDA blocks cannot exceed 1024 threads");

    __shared__ float shared_a[BM][BK];
    __shared__ float shared_b[BK][BN];

    float accumulators[TM][TN] = {};
    const int thread_linear = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
    const int thread_count = static_cast<int>(blockDim.x * blockDim.y);
    const int block_row = static_cast<int>(blockIdx.y) * BM;
    const int block_col = static_cast<int>(blockIdx.x) * BN;
    const int micro_row = static_cast<int>(threadIdx.y) * TM;
    const int micro_col = static_cast<int>(threadIdx.x) * TN;

    for (int tile_k = 0; tile_k < k; tile_k += BK) {
        for (int linear = thread_linear; linear < BM * BK; linear += thread_count) {
            const int local_row = linear / BK;
            const int local_k = linear % BK;
            const int global_row = block_row + local_row;
            const int global_k = tile_k + local_k;
            shared_a[local_row][local_k] =
                (global_row < m && global_k < k)
                    ? a[static_cast<std::size_t>(global_row) * k + global_k]
                    : 0.0F;
        }
        for (int linear = thread_linear; linear < BK * BN; linear += thread_count) {
            const int local_k = linear / BN;
            const int local_col = linear % BN;
            const int global_k = tile_k + local_k;
            const int global_col = block_col + local_col;
            shared_b[local_k][local_col] =
                (global_k < k && global_col < n)
                    ? b[static_cast<std::size_t>(global_k) * n + global_col]
                    : 0.0F;
        }
        __syncthreads();

        #pragma unroll
        for (int local_k = 0; local_k < BK; ++local_k) {
            #pragma unroll
            for (int row = 0; row < TM; ++row) {
                const float a_value = shared_a[micro_row + row][local_k];
                #pragma unroll
                for (int col = 0; col < TN; ++col) {
                    accumulators[row][col] +=
                        a_value * shared_b[local_k][micro_col + col];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int row = 0; row < TM; ++row) {
        const int global_row = block_row + micro_row + row;
        #pragma unroll
        for (int col = 0; col < TN; ++col) {
            const int global_col = block_col + micro_col + col;
            if (global_row < m && global_col < n) {
                c[static_cast<std::size_t>(global_row) * n + global_col] =
                    accumulators[row][col];
            }
        }
    }
}

}  // namespace

void validate_problem(int m, int n, int k) {
    if (m <= 0 || n <= 0 || k <= 0) {
        throw std::invalid_argument("M, N, and K must all be positive");
    }
}

void validate_fixed_tiled_feasibility(int device) {
    static_assert(kBM % kTM == 0, "BM must be divisible by TM");
    static_assert(kBN % kTN == 0, "BN must be divisible by TN");
    constexpr int threads = (kBM / kTM) * (kBN / kTN);
    constexpr std::size_t shared_bytes =
        static_cast<std::size_t>(kBM * kBK + kBK * kBN) * sizeof(float);

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (threads > properties.maxThreadsPerBlock) {
        std::ostringstream message;
        message << "configuration needs " << threads << " threads, device permits "
                << properties.maxThreadsPerBlock;
        throw std::runtime_error(message.str());
    }
    if (shared_bytes > properties.sharedMemPerBlock) {
        std::ostringstream message;
        message << "configuration needs " << shared_bytes
                << " shared-memory bytes, device permits "
                << properties.sharedMemPerBlock;
        throw std::runtime_error(message.str());
    }
}

LaunchMetadata launch_naive_gemm(const float* a, const float* b, float* c,
                                 int m, int n, int k, cudaStream_t stream) {
    validate_problem(m, n, k);
    const dim3 block(16, 16, 1);
    const dim3 grid((n + block.x - 1) / block.x,
                    (m + block.y - 1) / block.y, 1);
    naive_gemm_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    CUDA_CHECK_LAUNCH();
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, naive_gemm_kernel));
    return {grid, block, attributes.sharedSizeBytes, 0};
}

LaunchMetadata launch_fixed_tiled_gemm(const float* a, const float* b, float* c,
                                       int m, int n, int k, cudaStream_t stream) {
    validate_problem(m, n, k);
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    validate_fixed_tiled_feasibility(device);

    const dim3 block(kBN / kTN, kBM / kTM, 1);
    const dim3 grid((n + kBN - 1) / kBN, (m + kBM - 1) / kBM, 1);
    tiled_gemm_kernel<kBM, kBN, kBK, kTM, kTN>
        <<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    CUDA_CHECK_LAUNCH();
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(
        &attributes, tiled_gemm_kernel<kBM, kBN, kBK, kTM, kTN>));
    return {grid, block, attributes.sharedSizeBytes, 0};
}

}  // namespace gemm
