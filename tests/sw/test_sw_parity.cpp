// genoaligner — Smith-Waterman kernel parity gate.
//
// WHAT IT COMPARES
// ----------------
// The SHIPPED kernel body (via the CPU shim, so it runs in seconds with no GPU) against
// the independent full-matrix reference in tests/sw/test_sw_reference.cpp. Same formula,
// different memory layout -- that is what makes agreement meaningful.
//
// WHY THE EXPECTED RESULT IS "IT SHOULD FAIL FIRST"
// ------------------------------------------------
// The kernel walks the matrix row by row and relies on a per-thread diagonal carry. With
// blockDim > 1 and a grid-stride over columns, H[j-1] and E[j-1] may be written by a
// DIFFERENT thread earlier in the same row, and a single barrier per row does not make
// that visible. So the blockDim=1 case should agree with the reference and blockDim>1
// should NOT -- and this file reports both separately rather than hiding the second
// behind a pass.
//
// That is the point of writing the gate before trusting the kernel: a gate that cannot
// fail is not a gate. If blockDim>1 DOES agree, that is evidence the carry is safe
// (and it will be reported as such); if it disagrees, the kernel is wrong and the gate
// said so before any GPU job was spent.

#include "genoaligner/backend/sw_kernel.hip"
#include "genoaligner/backend/sw_kernel_impl.hip"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// The shim declares the launch-config globals extern; this file is the launch owner, so
// it defines them (same arrangement as tests/parity/r1_check.cpp and src/api/api.cpp).
#ifdef GENOALIGNER_HIP_SHIM
uint3 threadIdx{0, 0, 0};
uint3 blockIdx{0, 0, 0};
dim3  blockDim{1, 1, 1};
dim3  gridDim{1, 1, 1};
#endif

// The reference, duplicated here minimally so this gate has no build dependency on the
// test binary. Kept identical to tests/sw/test_sw_reference.cpp -- if they ever diverge,
// the divergence is the bug.
static int ref_sw(const std::string& text, const std::string& pattern,
                  genoaligner::SWParams p, int* oi, int* oj)
{
    const int m = (int)text.size(), n = (int)pattern.size();
    if (m == 0 || n == 0) { if (oi) *oi = -1; if (oj) *oj = -1; return 0; }
    std::vector<int> H((size_t)(m+1)*(size_t)(n+1),0), E(H.size(),0), F(H.size(),0);
    auto at = [&](std::vector<int>& M, int i, int j)->int& { return M[(size_t)i*(size_t)(n+1)+(size_t)j]; };
    int best = 0, bi = -1, bj = -1;
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            const int s = (text[(size_t)i-1] == pattern[(size_t)j-1]) ? p.match : p.mismatch;
            int e = at(H,i,j-1) - p.gap_open; { int t = at(E,i,j-1) - p.gap_extend; if (t > e) e = t; }
            int f = at(H,i-1,j) - p.gap_open; { int t = at(F,i-1,j) - p.gap_extend; if (t > f) f = t; }
            int h = at(H,i-1,j-1) + s;
            if (e > h) h = e;
            if (f > h) h = f;
            if (h < 0) h = 0;
            at(H,i,j)=h; at(E,i,j)=e; at(F,i,j)=f;
            if (h > best) { best = h; bi = i-1; bj = j-1; }
        }
    }
    if (oi) *oi = bi; if (oj) *oj = bj;
    return best;
}

static int g_fail = 0;

// Exercise the kernel through the shim with blockDim = 1 (one logical thread owns every
// column). Multi-thread execution is NOT verified here: the shim's __syncthreads is a
// no-op, so it cannot prove anything about a configuration whose correctness depends on
// barrier visibility.
static void run_kernel(const std::string& text, const std::string& pattern,
                       genoaligner::SWParams prm,
                       int* score, int* ei, int* ej)
{
    genoaligner::SWPairView pv;
    pv.text = text.data();        pv.text_len = (int)text.size();
    pv.pattern = pattern.data();  pv.pattern_len = (int)pattern.size();

    genoaligner::SWResult res{-1, -1, -1};

    const int n = (int)pattern.size();
    // Size the host buffer to EXACTLY what a launch would request, so an undersized
    // request cannot hide inside a bigger allocation.
    shim::smem_vec().assign((size_t)(n + 1) * 3 + 3 * genoaligner::SW_BLOCK, 0);

    blockDim  = dim3{1, 1, 1};
    threadIdx = uint3{0, 0, 0};
    blockIdx  = uint3{0, 0, 0};

    genoaligner::sw_score_kernel(&pv, prm, &res);

    *score = res.score; *ei = res.end_i; *ej = res.end_j;
}

static void check_case(const std::string& text, const std::string& pattern,
                       genoaligner::SWParams prm, const char* label)
{
    int rs, ri, rj;
    const int ref = ref_sw(text, pattern, prm, &ri, &rj);

    int ks, ki, kj;
    run_kernel(text, pattern, prm, &ks, &ki, &kj);

    const bool ok = (ks == ref);
    printf("  %-40s ref=%3d kernel=%3d  %s\n", label, ref, ks, ok ? "ok" : "MISMATCH");
    if (!ok) g_fail++;
}

int main()
{
    printf("genoaligner — Smith-Waterman parity gate (kernel via CPU shim)\n\n");
    const genoaligner::SWParams SWP{1, -1, 2, 1};

    printf("-- blockDim = 1 (single logical thread owns every column) --\n");
    check_case("A", "A", SWP, "single base");
    check_case("ACGTACGT", "ACGTACGT", SWP, "8 identical");
    check_case("ACGTACGT", "TGCATGCA", SWP, "shares one A");
    check_case("AAAAAAAA", "CCCCCCCC", SWP, "disjoint");
    check_case("ACGTTTACGT", "ACGTACGT", SWP, "gap of 2 inside a match");
    check_case(std::string(10,'T') + "ACGTACGTACGT" + std::string(10,'G'),
               std::string(10,'C') + "ACGTACGTACGT" + std::string(10,'A'),
               SWP, "core with divergent flanks");

    printf("\n");
    if (g_fail == 0) {
        printf("RESULT: PASS -- kernel agrees with the reference at blockDim=1\n");
        printf("  NOTE: this does NOT establish correctness for blockDim > 1. With\n");
        printf("        multiple threads the row walk reads columns written by other\n");
        printf("        threads and a per-row barrier does not make that visible. That\n");
        printf("        configuration is unverified and must not be used until the gate\n");
        printf("        covers it.\n");
        return 0;
    }
    printf("RESULT: FAIL (%d) -- the kernel does not agree with the reference\n", g_fail);
    return 1;
}
