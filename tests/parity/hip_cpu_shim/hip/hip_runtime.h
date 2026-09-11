// Minimal CPU shim for hip_runtime.h: lets the WFA kernel logic be compiled and
// executed on a machine with no GPU, so the cluster run is gated by a ~seconds
// CPU check first.
//
// This is a VERIFICATION harness, not a substitute for the MI210 run. It
// emulates the small surface the kernel uses. It does NOT emulate real
// barriers, warp behaviour, block-size limits, or memory coalescing.
//
// Design note: do NOT try `#define extern` or `#define smem` here to neutralise
// `extern __shared__ int smem[];`. That poisons <string>/<random> and the build
// fails with dozens of unrelated errors. Instead the kernel guards that single
// line with `#ifdef GENOALIGNER_HIP_SHIM` and takes a host pointer from below.
#ifndef GENOALIGNER_HIP_SHIM_H
#define GENOALIGNER_HIP_SHIM_H

#include <cstdint>
#include <vector>

#define __global__
#define __device__
#define __host__
#define __restrict__
#define __syncthreads() ((void)0)

struct uint3 { unsigned x, y, z; };
struct dim3  { unsigned x, y, z; };

extern uint3 threadIdx;
extern uint3 blockIdx;
extern dim3  blockDim;
extern dim3  gridDim;

typedef int hipError_t;
enum { hipSuccess = 0 };
inline const char* hipGetErrorString(hipError_t) { return "ok"; }

// Host backing store standing in for `extern __shared__ int smem[];`
namespace shim {
    inline std::vector<int>& smem_vec() { static std::vector<int> v; return v; }
}
inline int* shim_smem_ptr() { return shim::smem_vec().data(); }

#endif // GENOALIGNER_HIP_SHIM_H
