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

// The Fase 4 trace kernel is launched by the harness under the shim, so the
// launch surface must exist here too. Under the shim the "device" is the host,
// which lets the GATE exercise the shipped kernel body -- not a paraphrased
// copy of the walk. See tests/parity/wfa_parity.cpp.
//
// hipFuncSetAttribute is the runtime call that raises a block's dynamic shared
// memory past the 48 KB default on NVIDIA. Accepted, never enforced: there is
// no shared memory on the host to exceed.
typedef void* hipStream_t;
enum { hipFuncAttributeMaxDynamicSharedMemorySize = 8 };
inline hipError_t hipFuncSetAttribute(const void*, int, int) { return hipSuccess; }

// No-op launch. Calling it does NOT run the kernel; the harness calls the
// kernel body directly. Declared so the same harness source also compiles for
// a real backend, where it IS the launch.
inline hipError_t hipLaunchKernelGGLInternal(void (*)(), int, int, size_t,
                                             hipStream_t, const void*, ...)
{ return hipSuccess; }

// Host backing store standing in for `extern __shared__ int smem[];`
namespace shim {
    inline std::vector<int>& smem_vec() { static std::vector<int> v; return v; }
}
inline int* shim_smem_ptr() { return shim::smem_vec().data(); }

#endif // GENOALIGNER_HIP_SHIM_H
