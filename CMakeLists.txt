cmake_minimum_required(VERSION 3.18)
project(genoaligner LANGUAGES CXX)

# genoaligner: hardware-agnostic sequence alignment (CUDA + ROCm).
#
# Strategy: write everything in HIP. hipcc targets AMD natively and NVIDIA
# through -D__HIP_PLATFORM_NVIDIA__, so one source tree yields both backends.
# This file detects which platform the modules/toolchain expose and configures
# accordingly.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# HIP sources: CMake does not know the .hip extension by default, so it cannot
# infer the link language and fails with "Cannot determine link language for
# target". Register .hip as a C++ source extension and route it to hipcc.
set_source_files_properties(
    src/probe.hip PROPERTIES LANGUAGE CXX)
set(CMAKE_HIP_ARCHITECTURES "" CACHE STRING "" FORCE)

option(GENOALIGNER_BUILD_TESTS "Build tests" ON)
option(GENOALIGNER_BUILD_BENCH "Build benchmarks" ON)

# --- Backend detection -------------------------------------------------
# Prefer an explicit override (GENOALIGNER_BACKEND=rocm|cuda). Otherwise detect
# from the environment: ROCM_PATH/HIP_PATH implies rocm, CUDA_HOME/CUDAToolkit
# implies cuda.
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

find_program(HIPCC NAMES hipcc)
if(NOT HIPCC)
    message(FATAL_ERROR
        "hipcc not found. Load a toolchain first, e.g.:\n"
        "  ml load rocm/6.4.3     (AMD)\n"
        "  ml load cuda/13.0      (NVIDIA)\n"
        "hipcc ships with both; on CUDA nodes it is provided by the ROCm "
        "toolchain or the nvidia HIP shim.")
endif()
message(STATUS "hipcc: ${HIPCC}")

add_library(genoaligner_types INTERFACE)
target_include_directories(genoaligner_types INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Compile a .hip source with hipcc for the selected platform.
function(add_hip_library name)
    set(sources ${ARGN})
    add_library(${name} STATIC ${sources})
    target_include_directories(${name} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(${name} PROPERTIES
        CXX_COMPILER ${HIPCC}
        LINKER_LANGUAGE CXX
        POSITION_INDEPENDENT_CODE ON)
    if(GENOALIGNER_BACKEND STREQUAL "cuda")
        target_compile_definitions(${name} PUBLIC __HIP_PLATFORM_NVIDIA__)
    endif()
endfunction()

# --- Platform probe ----------------------------------------------------
# tiny executable that reports which runtime it was built against; used as the
# H1 test (compiles and runs on both MI210/ROCm and NVIDIA/CUDA).
add_executable(genoaligner_probe src/probe.hip)
set_target_properties(genoaligner_probe PROPERTIES CXX_COMPILER ${HIPCC})
if(GENOALIGNER_BACKEND STREQUAL "cuda")
    target_compile_definitions(genoaligner_probe PRIVATE __HIP_PLATFORM_NVIDIA__)
endif()

if(GENOALIGNER_BUILD_TESTS)
    enable_testing()
    add_test(NAME probe_runs COMMAND genoaligner_probe)
endif()
