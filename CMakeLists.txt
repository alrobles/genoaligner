cmake_minimum_required(VERSION 3.18)
project(genoaligner LANGUAGES CXX)

# genoaligner: hardware-agnostic sequence alignment (CUDA + ROCm).
#
# ============================================================ ONE SOURCE, TWO COMPILERS
#
# The strategy is one HIP source tree and two compilers -- NOT one compiler:
#
#   AMD (MI210)   hipcc  (AMD clang wrapper)  -> ROCm native
#   NVIDIA        nvcc   (CUDA)               -> __HIP_PLATFORM_NVIDIA__
#
# The mechanism that makes the NVIDIA side work is a header layer inside ROCm,
# not a compiler shim:
#
#   hip/hip_runtime.h
#     #if defined(__HIP_PLATFORM_NVIDIA__) && !defined(__HIP_PLATFORM_AMD__)
#       #include <hip/nvidia_detail/nvidia_hip_runtime.h>  -> <cuda_runtime.h>
#
# So nvcc compiles the .hip source directly, given that define and ROCm's
# include dir. Verified end-to-end: 100% parity on both backends against three
# external oracles (jobs 29201335 MI210, 29202787 RTX 6000).
#
# HISTORY (this file was wrong twice, both worth recording):
#   1. An earlier version required CMAKE_CXX_COMPILER to be hipcc for BOTH
#      backends. That made the CUDA branch impossible by construction, because
#      no HIP-for-NVIDIA shim exists to install (four routes closed, 2026-09-10).
#      The correct model is two compilers over one source.
#   2. It also assumed hipcc could target NVCC via -D__HIP_PLATFORM_NVIDIA__.
#      It cannot: hipcc is AMD's clang wrapper and never invokes nvcc.
#
# ============================================================ CMAKE + HIP GOTCHAS
#
# CMake has no HIP language mode here, so .hip is declared as CXX (below).
# Merely setting the language is not enough for hipcc: it must also BE the CXX
# compiler, because setting CXX_COMPILER as a target property is ignored and
# g++ ends up treating .hip as a linker input.
#
# The KU cluster also needs a specific compiler load order (compiler/gcc/14.2
# BEFORE cmake, or cmake dies on GLIBCXX_3.4.32). See docs/CONTAINERS.md and the
# ku-hpc-gpu-toolchain skill; the build scripts handle it.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(GENOALIGNER_BUILD_TESTS "Build tests" ON)
option(GENOALIGNER_BUILD_BENCH "Build benchmarks" ON)

# --- Backend selection -------------------------------------------------
# Explicit is better than inferred here: the two backends need different
# compilers, and guessing wrong produces a confusing failure much later.
set(GENOALIGNER_BACKEND "rocm" CACHE STRING "Backend: rocm or cuda")
set_property(CACHE GENOALIGNER_BACKEND PROPERTY STRINGS rocm cuda)

if(NOT GENOALIGNER_BACKEND MATCHES "^(rocm|cuda)$")
    message(FATAL_ERROR
        "GENOALIGNER_BACKEND must be 'rocm' or 'cuda', got '${GENOALIGNER_BACKEND}'")
endif()
message(STATUS "genoaligner backend: ${GENOALIGNER_BACKEND}")

# --- CUDA toolkit location ---------------------------------------------
# Needed for the NVIDIA branch because ROCm's nvidia_detail layer includes
# <cuda_runtime.h> and its friends. The KU nvhpc install splits CUDA headers
# across include/ and targets/x86_64-linux/include/, so both are added.
#
# CUDA VERSION IS LOAD-BEARING: use 12.4, not 13.0. ROCm 6.4.3's nvidia_detail
# layer targets CUDA 12.x and fails to compile under 13.0 on cudaMemLocation and
# on cudaDeviceProp.clockRate / .computeMode.
set(GENOALIGNER_CUDA_ROOT "/kuhpc/sw/nvhpc/Linux_x86_64/2024/cuda/12.4"
    CACHE PATH "CUDA toolkit root (CUDA 12.x required for the HIP header layer)")

# --- ROCm location -----------------------------------------------------
set(GENOALIGNER_ROCM_ROOT "/kuhpc/sw/rocm/6.4.3"
    CACHE PATH "ROCm root (provides the HIP headers used by BOTH backends)")

# --- Host compiler for the CUDA backend ---------------------------------
# nvcc validates the host GCC version and CUDA 12.4 refuses anything after 13:
#   error: -- unsupported GNU version! gcc versions later than 13 are not supported!
# The cluster's gcc 14.2 (needed by cmake itself) is therefore NOT usable as
# nvcc's host compiler. The system gcc 11.5 is, and nvcc is only invoking it for
# host code -- the device compiler is a separate component.
#
# This only bites on the CUDA branch; hipcc brings its own clang and ignores it.
set(GENOALIGNER_HOST_CXX "/usr/bin/g++" CACHE FILEPATH
    "Host compiler for nvcc (CUDA backend). CUDA 12.4 rejects GCC > 13.")

if(NOT EXISTS "${GENOALIGNER_ROCM_ROOT}/include/hip/hip_runtime.h")
    message(FATAL_ERROR
        "HIP headers not found under ${GENOALIGNER_ROCM_ROOT}. Both backends need "
        "ROCm's headers: the NVIDIA path reaches CUDA through hip/nvidia_detail.")
endif()

# --- Compiler expectation ----------------------------------------------
# Assert the right compiler for the chosen backend. A mismatch here is the
# single most likely misconfiguration, so fail loudly and say what to do.
get_filename_component(_cxx_name "${CMAKE_CXX_COMPILER}" NAME)

if(GENOALIGNER_BACKEND STREQUAL "rocm")
    if(NOT _cxx_name MATCHES "^hipcc")
        message(FATAL_ERROR
            "backend=rocm needs CMAKE_CXX_COMPILER=hipcc, got '${_cxx_name}'.\n"
            "  cmake -DCMAKE_CXX_COMPILER=$(command -v hipcc) -DGENOALIGNER_BACKEND=rocm ...\n"
            "or use scripts/build_rocm.sh")
    endif()
else()
    if(NOT _cxx_name MATCHES "^nvcc")
        message(FATAL_ERROR
            "backend=cuda needs CMAKE_CXX_COMPILER=nvcc, got '${_cxx_name}'.\n"
            "hipcc CANNOT target NVIDIA (it is AMD's clang wrapper and never "
            "invokes nvcc), so the CUDA branch uses nvcc over the same source.\n"
            "  cmake -DCMAKE_CXX_COMPILER=${GENOALIGNER_CUDA_ROOT}/bin/nvcc "
            "-DGENOALIGNER_BACKEND=cuda ...\n"
            "or use scripts/build_cuda.sh")
    endif()
endif()
message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER} (${_cxx_name})")

# --- .hip must be treated as C++ ---------------------------------------
set(CMAKE_CXX_EXTENSIONS OFF)

# All .hip translation units in the project. Kept in one place because several
# per-source properties must be applied to every one of them (LANGUAGE CXX, and
# on the CUDA backend the -x cu override nvcc requires).
set(GENOALIGNER_ALL_HIP_SOURCES
    src/probe.hip
    include/genoaligner/backend/wfa_kernel.hip
    # api.cpp is a .cpp, but it includes the kernel headers, so on the CUDA backend
    # nvcc needs the same -x cu override. It belongs in this list for that reason.
    src/api/api.cpp
)

# The LANGUAGE property must be set on the SOURCE before add_executable, and
# from a scope that outlives it. Setting it inside a helper function called
# AFTER add_executable fails with "Cannot determine link language for target"
# -- the property has to be in effect when CMake evaluates the target.
set_source_files_properties(
    ${GENOALIGNER_ALL_HIP_SOURCES}
    PROPERTIES LANGUAGE CXX)

# Include dirs and the platform define are per-target, so they can be applied
# after the target exists.
function(genoaligner_configure_hip target)
    target_include_directories(${target} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${GENOALIGNER_ROCM_ROOT}/include")

    if(GENOALIGNER_BACKEND STREQUAL "cuda")
        # The define that routes hip_runtime.h to nvidia_detail.
        target_compile_definitions(${target} PRIVATE __HIP_PLATFORM_NVIDIA__)
        target_include_directories(${target} PRIVATE
            "${GENOALIGNER_CUDA_ROOT}/include"
            "${GENOALIGNER_CUDA_ROOT}/targets/x86_64-linux/include")

        # nvcc does not recognise the .hip extension:
        #   "nvcc fatal : Don't know what to do with '.../probe.hip'"
        # so every .hip translation unit must be forced to CUDA. hipcc needs no
        # equivalent flag (it owns .hip natively), which is why this lives in the
        # CUDA branch only.
        foreach(_src ${GENOALIGNER_ALL_HIP_SOURCES})
            set_source_files_properties(${_src}
                PROPERTIES COMPILE_OPTIONS "-x;cu")
        endforeach()

        # CUDA 12.4 refuses GCC > 13 as its host compiler, and the cluster's
        # gcc 14.2 (which cmake itself needs) is therefore unusable here. Point
        # nvcc at the system gcc for host code only.
        if(GENOALIGNER_HOST_CXX)
            target_compile_options(${target} PRIVATE
                "SHELL:-ccbin ${GENOALIGNER_HOST_CXX}")
        endif()

        # Position-independent code. Modern toolchains default to PIE linking,
        # but nvcc does not emit PIC/PIE objects on its own, which fails with:
        #   relocation R_X86_64_32S against `.rodata' can not be used when
        #   making a PIE object; recompile with -fPIE
        # Both halves are needed: -Xcompiler -fPIE for the objects and -pie for
        # the link step. hipcc handles this itself, hence CUDA-only.
        target_compile_options(${target} PRIVATE
            "SHELL:-Xcompiler -fPIE")
        target_link_options(${target} PRIVATE
            "SHELL:-Xcompiler -pie")
    endif()
endfunction()

# --- Targets ------------------------------------------------------------
#
# The library target is the one that matters to a consumer: until it existed,
# `cmake --build` produced only a probe executable, so "install this software"
# had no meaning. api.cpp is the public surface (align/align_batch); the kernels
# it launches are headers, so they need no separate translation unit.
add_library(genoaligner STATIC src/api/api.cpp)
genoaligner_configure_hip(genoaligner)
target_include_directories(genoaligner PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")

add_executable(genoaligner_probe src/probe.hip)
genoaligner_configure_hip(genoaligner_probe)

add_executable(genoaligner_api_test tests/api/test_api.cpp)
target_link_libraries(genoaligner_api_test PRIVATE genoaligner)
genoaligner_configure_hip(genoaligner_api_test)

# FASTA reading is header-only and pure host code, so this test needs no HIP flags.
add_executable(genoaligner_fasta_test tests/io/test_fasta.cpp)
target_include_directories(genoaligner_fasta_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

install(TARGETS genoaligner ARCHIVE DESTINATION lib)
install(FILES include/genoaligner/api.hpp DESTINATION include/genoaligner)
install(FILES include/genoaligner/io/fasta.hpp DESTINATION include/genoaligner/io)
install(DIRECTORY include/genoaligner/backend DESTINATION include/genoaligner)

# The README example, built as part of the tree so the documented usage cannot rot.
# It links the library and uses the consumer include form (<genoaligner/api.hpp>),
# which is the form that must work against an installed tree.
add_executable(genoaligner_example examples/align_fasta.cpp)
target_link_libraries(genoaligner_example PRIVATE genoaligner)
genoaligner_configure_hip(genoaligner_example)

if(GENOALIGNER_BUILD_TESTS)
    enable_testing()
    # These two need a GPU. On a host without one they exit 77 (the conventional
    # "skipped" code) rather than failing, because "no accelerator here" is not a
    # defect in the software. Without these properties a CPU-only CI runner reports
    # a green build as broken.
    add_test(NAME probe_runs COMMAND genoaligner_probe)
    set_tests_properties(probe_runs PROPERTIES SKIP_RETURN_CODE 77)

    add_test(NAME api_runs   COMMAND genoaligner_api_test)
    set_tests_properties(api_runs PROPERTIES SKIP_RETURN_CODE 77)

    # Pure host code: must pass everywhere, so no skip code.
    add_test(NAME fasta_runs COMMAND genoaligner_fasta_test)

    # Real biological sequences (tests/data/mtdna_*.fa). Needs a GPU; exits 77 on a
    # device-less host, so it is reported as skipped rather than failed.
    add_executable(genoaligner_real_test tests/real/test_real_sequences.cpp)
    target_link_libraries(genoaligner_real_test PRIVATE genoaligner)
    genoaligner_configure_hip(genoaligner_real_test)
    # Run from the repository root: the test reads tests/data/ by relative path.
    add_test(NAME real_runs COMMAND genoaligner_real_test
             WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(real_runs PROPERTIES SKIP_RETURN_CODE 77)

    # Concurrency (B3) and batch semantics (B4). Both need a device.
    add_executable(genoaligner_conc_test tests/concurrency/test_concurrency.cpp)
    target_link_libraries(genoaligner_conc_test PRIVATE genoaligner)
    find_package(Threads REQUIRED)
    target_link_libraries(genoaligner_conc_test PRIVATE Threads::Threads)
    genoaligner_configure_hip(genoaligner_conc_test)
    add_test(NAME conc_runs COMMAND genoaligner_conc_test
             WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(conc_runs PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(genoaligner_batch_test tests/api/test_batch_semantics.cpp)
    target_link_libraries(genoaligner_batch_test PRIVATE genoaligner)
    genoaligner_configure_hip(genoaligner_batch_test)
    add_test(NAME batch_runs COMMAND genoaligner_batch_test
             WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(batch_runs PROPERTIES SKIP_RETURN_CODE 77)
endif()

# --- Summary -----------------------------------------------------------
message(STATUS "")
message(STATUS "genoaligner configuration")
message(STATUS "  backend     : ${GENOALIGNER_BACKEND}")
message(STATUS "  compiler    : ${CMAKE_CXX_COMPILER}")
message(STATUS "  ROCm root   : ${GENOALIGNER_ROCM_ROOT}")
if(GENOALIGNER_BACKEND STREQUAL "cuda")
    message(STATUS "  CUDA root   : ${GENOALIGNER_CUDA_ROOT}")
    message(STATUS "  platform    : __HIP_PLATFORM_NVIDIA__ -> hip/nvidia_detail")
endif()
message(STATUS "  tests       : ${GENOALIGNER_BUILD_TESTS}")
message(STATUS "")
