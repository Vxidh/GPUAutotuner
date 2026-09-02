#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

inline void cuda_check(cudaError_t result, const char* expression,
                       const char* file, int line) {
    if (result == cudaSuccess) {
        return;
    }
    std::ostringstream message;
    message << file << ':' << line << ": CUDA call " << expression
            << " failed: " << cudaGetErrorName(result) << " ("
            << cudaGetErrorString(result) << ')';
    throw std::runtime_error(message.str());
}

#define CUDA_CHECK(expression) \
    cuda_check((expression), #expression, __FILE__, __LINE__)

// Call immediately after every launch; synchronization is deliberately separate.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

