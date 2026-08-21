cmake_minimum_required(VERSION 3.24)
project(cuda_autotuning_rsm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(vector_add kernels/vector_add.cu)
add_executable(naive_gemm kernels/naive_gemm.cu)

foreach(target vector_add naive_gemm)
  set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()
