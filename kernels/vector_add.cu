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

// One thread produces one output element. `idx` is the thread's global position.
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

int main() {
  constexpr int n = 1 << 20;
  constexpr int threads_per_block = 256;
  const int blocks = (n + threads_per_block - 1) / threads_per_block;

  std::vector<float> h_a(n), h_b(n), h_c(n);
  for (int i = 0; i < n; ++i) {
    h_a[i] = static_cast<float>(i) * 0.5F;
    h_b[i] = static_cast<float>(i) * -0.25F;
  }

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  const size_t bytes = n * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  for (int i = 0; i < n; ++i) {
    if (std::fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-6F) {
      std::cerr << "Validation failed at index " << i << '\n';
      return EXIT_FAILURE;
    }
  }

  std::cout << "Vector addition passed: " << n << " elements, " << blocks
            << " blocks × " << threads_per_block << " threads.\n";
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return EXIT_SUCCESS;
}
