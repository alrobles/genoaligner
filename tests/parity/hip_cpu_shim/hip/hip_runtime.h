// Minimal CPU shim for hip_runtime.h: lets the WFA kernel logic be compiled and
// executed on a machine with no GPU, so the cluster run is gated by a ~seconds
// CPU check first.
//
// This is a VERIFICATION harness, not a substitute for the MI210 run. It
// emulates the small surface the kernel uses. It does NOT emulate real
// barriers, warp behaviour, block-size limits, or memory coalescing -- IN THE
// DEFAULT DIRECT-CALL MODE. See the warp-emulation section below: when a test
// drives a kernel through shim::run_block, barriers and warp shuffles ARE
// emulated with real threads, which is what makes the blockDim>1 SW parity gate
// meaningful on a GPU-less host.
//
// Design note: do NOT try `#define extern` or `#define smem` here to neutralise
// `extern __shared__ int smem[];`. That poisons <string>/<random> and the build
// fails with dozens of unrelated errors. Instead the kernel guards that single
// line with `#ifdef GENOALIGNER_HIP_SHIM` and takes a host pointer from below.
#ifndef GENOALIGNER_HIP_SHIM_H
#define GENOALIGNER_HIP_SHIM_H

// <cstddef> is here for size_t, used by the hipLaunchKernelGGLInternal forward
// declaration below. It was missing and the shim only compiled when something
// else upstream happened to include it first -- the same self-contained-header
// bug the kernel header had, found in the same deployment test (fase8, clean
// clone). Both were fixed together; do not remove either.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <memory>

#define __global__
#define __device__
#define __host__
#define __restrict__
#define __shared__ static

struct uint3 { unsigned x, y, z; };
struct dim3  { unsigned x, y, z; };

// threadIdx is per-lane state, so under the shim it must be per-THREAD state:
// shim::run_block spawns one host thread per emulated lane and each needs its
// own value. Every launch-owner translation unit defines it as thread_local;
// in direct-call mode (the historic use) the main thread's copy is the one the
// kernel sees, which is behaviour-identical to the old plain global.
extern thread_local uint3 threadIdx;
extern uint3 blockIdx;
extern dim3  blockDim;
extern dim3  gridDim;

// warpSize is a compile-time constant on device (32 NVIDIA, 64 AMD). Under the
// shim it is a global the harness may set BEFORE run_block, which lets one CPU
// gate exercise BOTH warp widths -- a warp-width assumption caught on CPU is a
// portability bug that never reaches a GPU.
inline int warpSize = 32;

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

// ---------------------------------------------------------------------------
// Warp/block emulation. OFF by default: when no emulation context is active
// (the historic direct-call use), every primitive below is a no-op and the
// shim behaves exactly as it always has. When a test calls shim::run_block,
// nthreads real std::threads execute the kernel body with per-lane threadIdx,
// a REAL block barrier for __syncthreads, a per-warp barrier for __syncwarp,
// and publish-buffer exchanges for the __shfl_*_sync family -- so a kernel
// whose correctness depends on cross-lane dataflow is genuinely exercised,
// and a missing exchange shows up as a wrong result on CPU, in seconds.
//
// Deliberate limits of the emulation, stated so they are not mistaken for
// coverage: lockstep scheduling is approximated by barriers at each exchange,
// the mask argument is accepted but only full-warp exchange is implemented,
// and only int lanes are exchanged (every kernel use to date is int).
// ---------------------------------------------------------------------------
namespace shim {

// Largest warp width the emulation supports (gfx warps are 64 lanes).
constexpr int kMaxWarp = 64;

// When true, every __shfl_*_sync returns the caller's own value -- i.e. the
// cross-lane exchange silently does nothing. This exists so a gate can assert
// that it FAILS when synchronisation is removed: a gate that stays green with
// the exchange disabled was never testing the exchange. Tests flip it for a
// dedicated negative-control run only.
inline bool g_shuffle_passthrough = false;

// Sense-reversing centralised barrier for n threads, hybrid: spin briefly, then
// block on a condvar. A pure spin barrier convoys when the emulated block has
// more threads than the host has cores (the last arriver sits descheduled while
// 100+ threads spin), which made the first version of this gate take minutes.
// Sleeping waiters free the cores so the barrier completes in microseconds.
// (std::barrier is C++20; the gate compiles at C++17.)
struct SenseBarrier {
    std::atomic<int>        count{0};
    std::atomic<int>        phase{0};
    std::mutex              mu;
    std::condition_variable cv;
    int n = 0;
    void init(int n_) { n = n_; }
    void wait() {
        const int p = phase.load(std::memory_order_acquire);
        if (count.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
            count.store(0, std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> lk(mu);
                phase.store(p + 1, std::memory_order_release);
            }
            cv.notify_all();
        } else {
            for (int s = 0; s < 4000; ++s)
                if (phase.load(std::memory_order_acquire) != p) return;
            std::unique_lock<std::mutex> lk(mu);
            // Phase cannot advance more than one generation without this
            // thread's arrival, so a single generation check is exact.
            cv.wait(lk, [&]{ return phase.load(std::memory_order_relaxed) != p; });
        }
    }
};

struct WarpEmu {
    SenseBarrier bar;
    // Publish buffer for the __shfl_*_sync family, double-buffered by exchange
    // generation: generation g+1's writes cannot start until every lane has
    // passed g's barrier, and a lane only reaches that barrier after reading
    // its g value -- so one barrier per exchange suffices.
    int pub[2][kMaxWarp];
};

struct Emu {
    int nthreads = 0, nwarps = 0, wsize = 0;
    SenseBarrier block_bar;
    std::vector<std::unique_ptr<WarpEmu>> warps;
};

// Active emulation context (null in direct-call mode). Set by run_block before
// the threads are spawned; read by every primitive below.
inline Emu*& cur() { static Emu* p = nullptr; return p; }

inline int lane_of() { return (int)threadIdx.x % warpSize; }
inline int warp_of() { return (int)threadIdx.x / warpSize; }

// Per-lane exchange generation: every lane executes the same sequence of
// exchanges (kernels must keep shuffle calls uniform across the warp), so a
// thread_local counter stays consistent and selects the buffer parity.
inline int& xchg_gen() { static thread_local int g = 0; return g; }

// Core exchange: every lane publishes v into the current generation's buffer,
// waits once, then reads the SOURCE lane's slot. The generation parity makes a
// second barrier unnecessary (see WarpEmu::pub).
inline int exchange(int v, int src) {
    Emu* e = cur();
    if (!e || g_shuffle_passthrough) return v;
    WarpEmu& w = *e->warps[warp_of()];
    const int pb = xchg_gen() & 1;
    w.pub[pb][lane_of()] = v;
    w.bar.wait();
    const int r = w.pub[pb][src];
    ++xchg_gen();
    return r;
}

// Spawn nthreads host threads as one emulated block. fn runs in every thread
// with threadIdx.x = tid. warpSize must divide nthreads. One block at a time:
// a test emulating a grid loops over blockIdx between run_block calls.
template <class F>
inline void run_block(int nthreads, F&& fn) {
    Emu e;
    e.nthreads = nthreads;
    e.wsize    = warpSize;
    e.nwarps   = nthreads / e.wsize;
    if (nthreads % e.wsize != 0 || e.wsize <= 0 || e.wsize > kMaxWarp) {
        // A malformed launch must fail loudly, not emulate something untrue.
        std::abort();
    }
    e.block_bar.init(nthreads);
    e.warps.resize((size_t)e.nwarps);
    for (int w = 0; w < e.nwarps; ++w) {
        e.warps[(size_t)w] = std::unique_ptr<WarpEmu>(new WarpEmu());
        e.warps[(size_t)w]->bar.init(e.wsize);
    }
    cur() = &e;
    std::vector<std::thread> ts;
    ts.reserve((size_t)nthreads);
    for (int t = 0; t < nthreads; ++t) {
        ts.emplace_back([&e, &fn, t] {
            threadIdx = uint3{(unsigned)t, 0, 0};
            fn();
        });
    }
    for (auto& th : ts) th.join();
    cur() = nullptr;
}

inline void sync_block() { if (Emu* e = cur()) e->block_bar.wait(); }
inline void sync_warp()  { if (Emu* e = cur()) e->warps[warp_of()]->bar.wait(); }

}  // namespace shim

// ---------------------------------------------------------------------------
// Minimal device-memory surface, added for the MSA GPU driver (src/msa/
// msa_gpu.cpp): under the shim "device" allocations are host heap, copies are
// memcpy, and there is exactly one device. This lets the SHIPPED driver code
// -- packing arithmetic, buffer sizing, level batching -- run bit-exact on a
// GPU-less host instead of being duplicated under an #ifdef.
// ---------------------------------------------------------------------------
enum hipMemcpyKind { hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2,
                     hipMemcpyDeviceToDevice = 3, hipMemcpyHostToHost = 4 };

inline hipError_t hipMalloc(void** p, size_t n) {
    *p = std::malloc(n ? n : 1);
    return *p ? hipSuccess : hipError_t(-1);
}
inline hipError_t hipFree(void* p) { std::free(p); return hipSuccess; }
inline hipError_t hipMemcpy(void* d, const void* s, size_t n, hipMemcpyKind) {
    if (n) std::memcpy(d, s, n);
    return hipSuccess;
}
inline hipError_t hipGetDeviceCount(int* n) { *n = 1; return hipSuccess; }
inline hipError_t hipGetLastError() { return hipSuccess; }
inline hipError_t hipDeviceSynchronize() { return hipSuccess; }

// The intrinsic surface the kernels use. Under emulation these are real
// exchanges between real threads; with no context they degrade to returning
// the caller's own value, which is the correct degenerate case for the old
// direct-call use at blockDim=1.
#define __syncthreads() (::shim::sync_block())

inline int  __shfl_up_sync(unsigned long long, int v, int d) {
    const int l = ::shim::lane_of();
    return ::shim::exchange(v, l >= d ? l - d : l);
}
inline int  __shfl_sync(unsigned long long, int v, int src) {
    return ::shim::exchange(v, src);
}
inline int  __shfl_xor_sync(unsigned long long, int v, int m) {
    return ::shim::exchange(v, ::shim::lane_of() ^ m);
}
inline void __syncwarp(unsigned long long) { ::shim::sync_warp(); }

#endif // GENOALIGNER_HIP_SHIM_H
