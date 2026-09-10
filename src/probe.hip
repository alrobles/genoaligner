// genoaligner platform probe (H1 test).
//
// Purpose: prove that ONE HIP source compiles and runs correctly on both
// backends -- AMD (ROCm) and NVIDIA (CUDA via __HIP_PLATFORM_NVIDIA__).
// This is the cheapest possible test of the architecture decision, and it
// gates everything else: if this does not work, the pure-HIP strategy is
// wrong and we reconsider (SYCL / hipify).
//
// It also reports which runtime and device it is on, so the same binary
// source yields an auditable platform report on each cluster partition.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t _e = (call);                                             \
        if (_e != hipSuccess) {                                             \
            fprintf(stderr, "HIP error %s at %s:%d\n",                      \
                    hipGetErrorString(_e), __FILE__, __LINE__);             \
            return 1;                                                       \
        }                                                                   \
    } while (0)

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void vector_sub(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] - b[i];
}

int main() {
    int dev = 0;
    HIP_CHECK(hipGetDevice(&dev));

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));

    const char* runtime =
#if defined(__HIP_PLATFORM_NVIDIA__)
        "CUDA (via HIP)";
#elif defined(__HIP_PLATFORM_AMD__)
        "ROCm (native HIP)";
#else
        "unknown";
#endif

    printf("genoaligner probe\n");
    printf("  backend : %s\n", runtime);
    printf("  device  : %s\n", prop.name);
    printf("  arch    : %s\n", prop.gcnArchName);
    printf("  memory  : %.2f GB\n", prop.totalGlobalMem / 1073741824.0);

    // Functional check: add and subtract must round-trip.
    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);
    float *a, *b, *c, *d;
    HIP_CHECK(hipMalloc(&a, bytes));
    HIP_CHECK(hipMalloc(&b, bytes));
    HIP_CHECK(hipMalloc(&c, bytes));
    HIP_CHECK(hipMalloc(&d, bytes));

    float* h = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) h[i] = (float)(i % 1024);
    HIP_CHECK(hipMemcpy(a, h, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b, h, bytes, hipMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(a, b, c, n);
    vector_sub<<<blocks, threads>>>(c, b, d, n);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h, d, bytes, hipMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < n; ++i) {
        // (a + b) - b == a, within float tolerance
        if (h[i] < (float)(i % 1024) - 1e-3f ||
            h[i] > (float)(i % 1024) + 1e-3f) {
            if (bad < 5) fprintf(stderr, "  mismatch at %d: %f\n", i, h[i]);
            ++bad;
        }
    }

    free(h);
    hipFree(a); hipFree(b); hipFree(c); hipFree(d);

    if (bad) {
        printf("  RESULT  : FAIL (%d mismatches)\n", bad);
        return 1;
    }
    printf("  RESULT  : PASS (round-trip on %d elements)\n", n);
    return 0;
}
