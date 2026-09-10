cmake_minimum_required(VERSION 3.18)
project(genoaligner LANGUAGES CXX)

# genoaligner: hardware-agnostic sequence alignment (CUDA + ROCm).
#
# Strategy: write everything in HIP. hipcc targets AMD natively and NVIDIA
# through -D__HIP_PLATFORM_NVIDIA__, so one source tree yields both backends.
#
# IMPORTANT (learned on KU HPC, 2026-09-10):
#   * CMake has no native HIP language mode unless a HIP-capable CMake is used;
#     forcing CXX_COMPILER to hipcc does NOT work (g++ is still invoked and the
#     .hip file is treated as a linker input).
#   * The robust approach is to declare .hip as CXX and inject hipcc through
#     CMAKE_CXX_COMPILER at configure time, which the build scripts do via
#     -DCMAKE_CXX_COMPILER=<hipcc>. This file asserts that it happened.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(GENOALIGNER_BUILD_TESTS "Build tests" ON)
option(GENOALIGNER_BUILD_BENCH "Build benchmarks" ON)

# --- Backend detection -------------------------------------------------
if(NOT DEFINED GENOALIGNER_BACKEND)
    if(DEFINED ENV{ROCM_PATH} OR DEFINED ENV{HIP_PATH})
        set(GENOALIGNER_BACKEND "rocm")
    elseif(DEFINED ENV{CUDA_HOME} OR DEFINED ENV{CUDAToolkit_ROOT})
        set(GENOALIGNER_BACKEND "cuda")
    else()
        set(GENOALIGNER_BACKEND "rocm")
    endif()
endif()
message(STATUS "genoaligner backend: ${GENOALIGNER_BACKEND}")

# The CXX compiler MUST be hipcc for .hip sources to compile. The build
# scripts pass -DCMAKE_CXX_COMPILER=$(command -v hipcc); verify it here so a
# misconfigured build fails loudly instead of producing broken objects.
get_filename_component(_cxx_name "${CMAKE_CXX_COMPILER}" NAME)
message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER} (${_cxx_name})")
if(NOT _cxx_name MATCHES "^hipcc")
    message(FATAL_ERROR
        "CMAKE_CXX_COMPILER is '${_cxx_name}', not hipcc. Configure with:\n"
        "  cmake -DCMAKE_CXX_COMPILER=$(command -v hipcc) ...\n"
        "or use scripts/build_rocm.sh / scripts/build_cuda.sh")
endif()

set_source_files_properties(src/probe.hip PROPERTIES LANGUAGE CXX)

add_executable(genoaligner_probe src/probe.hip)

if(GENOALIGNER_BACKEND STREQUAL "cuda")
    target_compile_definitions(genoaligner_probe PRIVATE __HIP_PLATFORM_NVIDIA__)
endif()

if(GENOALIGNER_BUILD_TESTS)
    enable_testing()
    add_test(NAME probe_runs COMMAND genoaligner_probe)
endif()
