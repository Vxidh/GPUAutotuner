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

int parse_positive(const char* text, const char* name) {
    const std::string value(text);
    std::size_t consumed = 0;
    const long parsed = std::stol(value, &consumed);
    if (consumed != value.size() || parsed <= 0 || parsed > 2147483647L) {
        throw std::invalid_argument(std::string(name) + " must be a positive int");
    }
    return static_cast<int>(parsed);
}

void fill_random(std::vector<float>& values, std::mt19937& generator) {
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    for (float& value : values) {
        value = distribution(generator);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int m = argc > 1 ? parse_positive(argv[1], "M") : 127;
        const int n = argc > 2 ? parse_positive(argv[2], "N") : 131;
        const int k = argc > 3 ? parse_positive(argv[3], "K") : 37;
        const double tolerance = argc > 4 ? std::stod(argv[4]) : 1.0e-4;
        if (!(tolerance > 0.0)) {
            throw std::invalid_argument("tolerance must be positive");
        }

        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        int runtime_version = 0;
        int driver_version = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
        CUDA_CHECK(cudaDriverGetVersion(&driver_version));

        std::vector<float> a(static_cast<std::size_t>(m) * k);
        std::vector<float> b(static_cast<std::size_t>(k) * n);
        std::vector<float> expected(static_cast<std::size_t>(m) * n);
        std::vector<float> actual(expected.size());
        std::mt19937 generator(kSeed);
        fill_random(a, generator);
        fill_random(b, generator);
        cpu_reference_gemm(a.data(), b.data(), expected.data(), m, n, k);

        float* device_a = nullptr;
        float* device_b = nullptr;
        float* device_c = nullptr;
        try {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_a),
                                  a.size() * sizeof(float)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_b),
                                  b.size() * sizeof(float)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_c),
                                  actual.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(device_a, a.data(), a.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(device_b, b.data(), b.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
            const auto launch = gemm::launch_fixed_tiled_gemm(
                device_a, device_b, device_c, m, n, k);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(actual.data(), device_c,
                                  actual.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            const ErrorMetrics metrics = calculate_error(expected, actual);

            std::cout << "gpu_name: " << properties.name << '\n'
                      << "compute_capability: " << properties.major << '.'
                      << properties.minor << '\n'
                      << "cuda_runtime_version: " << runtime_version << '\n'
                      << "cuda_driver_version: " << driver_version << '\n'
                      << "M,N,K: " << m << ',' << n << ',' << k << '\n'
                      << "BM,BN,BK,TM,TN: " << gemm::kBM << ',' << gemm::kBN
                      << ',' << gemm::kBK << ',' << gemm::kTM << ',' << gemm::kTN
                      << '\n'
                      << "grid: (" << launch.grid.x << ',' << launch.grid.y << ','
                      << launch.grid.z << ")\n"
                      << "block: (" << launch.block.x << ',' << launch.block.y << ','
                      << launch.block.z << ")\n"
                      << "static_shared_bytes: " << launch.static_shared_bytes << '\n'
                      << "dynamic_shared_bytes: " << launch.dynamic_shared_bytes << '\n'
                      << "seed: " << kSeed << '\n'
                      << std::scientific << std::setprecision(8)
                      << "relative_frobenius_error: " << metrics.relative_frobenius
                      << '\n'
                      << "maximum_absolute_error: " << metrics.maximum_absolute << '\n'
                      << "tolerance: " << tolerance << '\n'
                      << "result: "
                      << (metrics.relative_frobenius <= tolerance ? "PASS" : "FAIL")
                      << '\n';

            CUDA_CHECK(cudaFree(device_c));
            device_c = nullptr;
            CUDA_CHECK(cudaFree(device_b));
            device_b = nullptr;
            CUDA_CHECK(cudaFree(device_a));
            device_a = nullptr;
            return metrics.relative_frobenius <= tolerance ? EXIT_SUCCESS : EXIT_FAILURE;
        } catch (...) {
            if (device_c != nullptr && cudaFree(device_c) != cudaSuccess) {
                std::cerr << "warning: cudaFree(device_c) failed during cleanup\n";
            }
            if (device_b != nullptr && cudaFree(device_b) != cudaSuccess) {
                std::cerr << "warning: cudaFree(device_b) failed during cleanup\n";
            }
            if (device_a != nullptr && cudaFree(device_a) != cudaSuccess) {
                std::cerr << "warning: cudaFree(device_a) failed during cleanup\n";
            }
            throw;
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
