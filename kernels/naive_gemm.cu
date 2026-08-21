#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    const cudaError_t status = (call);                                           \
    if (status != cudaSuccess) {                                                 \
      std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__ << ": "     \
                << cudaGetErrorString(status) << '\n';                          \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

// C = A × B, row-major. One thread computes one element of C.
__global__ void naive_gemm(const float* a, const float* b, float* c, int n) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < n && col < n) {
    float sum = 0.0F;
    for (int k = 0; k < n; ++k) {
      sum += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
  }
}

int main() {
  constexpr int n = 128;
  constexpr int tile = 16;
  const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
  std::vector<float> h_a(n * n), h_b(n * n), h_c(n * n), h_reference(n * n, 0.0F);
  for (int i = 0; i < n * n; ++i) {
    h_a[i] = static_cast<float>((i % 13) - 6) / 7.0F;
    h_b[i] = static_cast<float>((i % 11) - 5) / 5.0F;
  }
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      for (int k = 0; k < n; ++k) h_reference[row * n + col] += h_a[row * n + k] * h_b[k * n + col];
    }
  }

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  const dim3 threads(tile, tile);
  const dim3 blocks((n + tile - 1) / tile, (n + tile - 1) / tile);
  naive_gemm<<<blocks, threads>>>(d_a, d_b, d_c, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  for (int i = 0; i < n * n; ++i) {
    if (std::fabs(h_c[i] - h_reference[i]) > 1e-3F) {
      std::cerr << "Validation failed at element " << i << '\n';
      return EXIT_FAILURE;
    }
  }
  std::cout << "Naive GEMM passed: " << n << "×" << n << ".\n";
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return EXIT_SUCCESS;
}
