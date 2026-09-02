#include "cuda_check.hpp"
#include "gemm_kernel.cuh"
#include "gemm_reference.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr unsigned int kSeed = 20260902U;

struct Shape {
    const char* name;
    int m;
    int n;
    int k;
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer_),
                              count * sizeof(float)));
    }
    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            const cudaError_t result = cudaFree(pointer_);
            if (result != cudaSuccess) {
                std::cerr << "cudaFree failed in DeviceBuffer destructor: "
                          << cudaGetErrorString(result) << '\n';
            }
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    float* get() { return pointer_; }

private:
    float* pointer_ = nullptr;
};

void fill_random(std::vector<float>& values, std::mt19937& generator) {
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    for (float& value : values) {
        value = distribution(generator);
    }
}

bool check_result(const Shape& shape, const char* implementation,
                  const std::vector<float>& expected,
                  const std::vector<float>& actual, double tolerance) {
    const ErrorMetrics metrics = calculate_error(expected, actual);
    const bool passed = metrics.relative_frobenius <= tolerance;
    std::cout << shape.name << " [" << shape.m << 'x' << shape.n << 'x'
              << shape.k << "] " << implementation << ": relative_frobenius="
              << std::scientific << std::setprecision(8)
              << metrics.relative_frobenius << ", max_absolute="
              << metrics.maximum_absolute << ", " << (passed ? "PASS" : "FAIL")
              << '\n';
    return passed;
}

bool run_shape(const Shape& shape, double tolerance) {
    const std::size_t a_count = static_cast<std::size_t>(shape.m) * shape.k;
    const std::size_t b_count = static_cast<std::size_t>(shape.k) * shape.n;
    const std::size_t c_count = static_cast<std::size_t>(shape.m) * shape.n;
    std::vector<float> a(a_count);
    std::vector<float> b(b_count);
    std::vector<float> expected(c_count);
    std::vector<float> naive(c_count);
    std::vector<float> tiled(c_count);
    std::mt19937 generator(kSeed);
    fill_random(a, generator);
    fill_random(b, generator);
    cpu_reference_gemm(a.data(), b.data(), expected.data(), shape.m, shape.n, shape.k);

    DeviceBuffer device_a(a_count);
    DeviceBuffer device_b(b_count);
    DeviceBuffer device_c(c_count);
    CUDA_CHECK(cudaMemcpy(device_a.get(), a.data(), a_count * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_b.get(), b.data(), b_count * sizeof(float),
                          cudaMemcpyHostToDevice));

    gemm::launch_naive_gemm(device_a.get(), device_b.get(), device_c.get(),
                            shape.m, shape.n, shape.k);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(naive.data(), device_c.get(), c_count * sizeof(float),
                          cudaMemcpyDeviceToHost));

    gemm::launch_fixed_tiled_gemm(device_a.get(), device_b.get(), device_c.get(),
                                  shape.m, shape.n, shape.k);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(tiled.data(), device_c.get(), c_count * sizeof(float),
                          cudaMemcpyDeviceToHost));

    const bool naive_ok = check_result(shape, "naive", expected, naive, tolerance);
    const bool tiled_ok = check_result(shape, "tiled", expected, tiled, tolerance);
    return naive_ok && tiled_ok;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const double tolerance = argc > 1 ? std::stod(argv[1]) : 1.0e-4;
        if (!(tolerance > 0.0)) {
            throw std::invalid_argument("tolerance must be positive");
        }
        const Shape shapes[] = {
            {"divisible_square", 64, 64, 32},
            {"non_divisible_all_tiles", 73, 91, 19},
            {"rectangular", 128, 192, 32},
            {"k_not_divisible_by_bk", 64, 64, 17},
            {"smaller_than_one_tile", 7, 9, 5},
        };
        bool all_passed = true;
        for (const Shape& shape : shapes) {
            all_passed = run_shape(shape, tolerance) && all_passed;
        }
        std::cout << "overall: " << (all_passed ? "PASS" : "FAIL") << '\n';
        return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
