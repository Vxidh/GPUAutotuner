cmake_minimum_required(VERSION 3.24)

project(GPUAutotuner VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Select at configure time, for example:
#   cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
# No GPU model is hard-coded; an unset value uses the CUDA compiler default.

add_library(gemm_kernels STATIC
    src/gemm_kernel.cu
)
target_include_directories(gemm_kernels PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
set_target_properties(gemm_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

function(enable_project_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
        )
    else()
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>
        )
    endif()
endfunction()
enable_project_warnings(gemm_kernels)

add_executable(gemm_experiment src/main.cu)
target_link_libraries(gemm_experiment PRIVATE gemm_kernels)
enable_project_warnings(gemm_experiment)

include(CTest)
if(BUILD_TESTING)
    add_executable(correctness_tests tests/correctness_tests.cu)
    target_link_libraries(correctness_tests PRIVATE gemm_kernels)
    enable_project_warnings(correctness_tests)
    add_test(NAME gemm_correctness COMMAND correctness_tests)
endif()
