#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

struct ErrorMetrics {
    double relative_frobenius;
    float maximum_absolute;
};

inline void cpu_reference_gemm(const float* a, const float* b, float* c,
                               int m, int n, int k) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float accumulator = 0.0F;
            for (int inner = 0; inner < k; ++inner) {
                accumulator += a[static_cast<std::size_t>(row) * k + inner] *
                               b[static_cast<std::size_t>(inner) * n + col];
            }
            c[static_cast<std::size_t>(row) * n + col] = accumulator;
        }
    }
}

inline ErrorMetrics calculate_error(const std::vector<float>& reference,
                                    const std::vector<float>& candidate) {
    double squared_difference = 0.0;
    double squared_reference = 0.0;
    float maximum_absolute = 0.0F;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        const double difference = static_cast<double>(candidate[index]) -
                                  static_cast<double>(reference[index]);
        squared_difference += difference * difference;
        const double expected = static_cast<double>(reference[index]);
        squared_reference += expected * expected;
        maximum_absolute = std::fmax(maximum_absolute,
                                     static_cast<float>(std::fabs(difference)));
    }
    const double relative = squared_reference == 0.0
                                ? std::sqrt(squared_difference)
                                : std::sqrt(squared_difference / squared_reference);
    return {relative, maximum_absolute};
}

