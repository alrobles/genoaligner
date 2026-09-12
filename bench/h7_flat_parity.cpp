// Fase 7 — parity of the flat (fixed-block) score kernel against the validated one.
//
// The flat kernel is a THREAD REMAPPING of the same recurrence, which is exactly
// the kind of change that can look right and be wrong (the Fase 4 bugs were both
// index remappings). So it is gated the same way everything else is: the CPU shim
// runs the real kernel body for both variants over the same pairs, and any
// disagreement fails.
//
// Build (no GPU):
//   g++ -O2 -std=c++17 -DGENOALIGNER_HIP_SHIM -Itests/parity/hip_cpu_shim -I. \
//       -o /tmp/h7flat bench/h7_flat_parity.cpp

#include <hip/hip_runtime.h>

#include "include/genoaligner/backend/wfa_kernel.hip"
#include "include/genoaligner/backend/wfa_score_flat.hip"
#include "src/reference/edit_distance_cpu.hpp"

uint3 threadIdx{0, 0, 0};
uint3 blockIdx{0, 0, 0};
dim3  blockDim{1, 1, 1};
dim3  gridDim{1, 1, 1};

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <random>

using namespace genoaligner;

// Run a kernel by replaying its own block structure under the shim: set blockDim,
// then for each "wavefront iteration" the kernel does internally there is no hook,
// so instead we drive it the way the harness does for a RETURNING kernel -- one
// call per pair with blockDim set to what the launch site would use, and threadIdx
// swept by the kernel's own grid-stride loop only if blockDim == 1.
//
// That limitation is real: with blockDim > 1 the shim cannot advance threadIdx by
// itself inside the kernel. So the shim run below uses blockDim = 1 for BOTH
// kernels, which makes the flat variant's interior loop degenerate to the original
// one-thread-per-slot form -- i.e. this checks the ALGEBRA of the flat file but
// CANNOT distinguish the thread remapping. The remapping is verified on the GPU
// (scripts/h7_flat.sbatch), where real threads exist.
//
// What this DOES verify, and it matters: the flat kernel's slot layout, sentinel
// handling, barrier placement and termination test reproduce the reference across
// random and adversarial cases. A layout mistake shows up here in seconds.

// WHY THE REFERENCE IS THE CPU DP, NOT wfa_score_kernel
// -----------------------------------------------------
// The first version of this gate compared the flat kernel against the validated
// kernel under the shim. That comparison is INVALID, and it took a 1087/1230
// "failure" to see why: with blockDim=1, the validated kernel's guard
// `if (threadIdx.x <= 2*s)` lets thread 0 write ONLY diagonal k=-s, so it does not
// compute the wavefront at all and returns -1 for cases it solves correctly on the
// GPU. The shim run of that kernel is degenerate; it is not a reference.
// Meanwhile the flat kernel's interior grid-stride loop, at blockDim=1, walks every
// slot t=0..2s and therefore DOES compute the wavefront.
//
// So the two disagree because one is degenerate and the other is not -- not because
// the flat kernel is wrong. Ground truth is the independent CPU DP, which is what
// this gate now checks the flat kernel against, and it checks it on the cases where
// the score actually resolves.
//
// The zero-arg version of the validated kernel is still run, but only to report how
// often the shim can even exercise it -- a number that belongs in the log rather
// than being silently averaged into a verdict.
static int run_one(const PairView& p, int smax, bool flat)
{
    int score = -12345;
    shim::smem_vec().assign((size_t)2 * (2 * smax + 3), WFA_NEG);
    blockDim  = dim3{1, 1, 1};
    threadIdx = uint3{0, 0, 0};
    blockIdx  = uint3{0, 0, 0};
    if (flat) wfa_score_kernel_flat(&p, &score, smax);
    else      wfa_score_kernel(&p, &score, smax);
    return score;
}
int main()
{
    int flat_fail = 0, flat_resolved = 0, flat_wrong = 0;
    int val_resolved = 0;   // how often the shim can exercise the validated kernel
    int checked = 0;

    // Adversarial + random, mirroring the parity harness's control set.
    struct C { std::string P, T; const char* lab; };
    std::vector<C> cases = {
        {"", "", "both empty"}, {"", "ACGT", "empty pattern"}, {"ACGT", "", "empty text"},
        {"A", "T", "single mismatch"}, {"AAAA", "AAAA", "identical"},
        {"AAAA", "AA", "pattern prefix"}, {"ACGTACGT", "TTTTTTTT", "no match"},
        {"GG", "TTT", "min repro 1"}, {"CT", "CCC", "min repro 2"}, {"CG", "GAC", "min repro 3"},
    };

    std::mt19937 rng(999u);
    std::uniform_int_distribution<int> base(0, 3), pos(0, 255), ch(0, 3);
    const char alpha[] = "ACGT";
    for (int c = 0; c < 400; ++c) {
        const int len = 1 + (c % 256);
        const int ident = (c % 4 == 0) ? 50 : (c % 4 == 1) ? 70 : (c % 4 == 2) ? 90 : 98;
        std::string t;
        for (int i = 0; i < len; ++i) t.push_back(alpha[base(rng)]);
        std::string p = t;
        const int nmut = (int)((size_t)len * (100 - ident) / 100.0);
        for (int i = 0; i < nmut && len > 0; ++i) p[pos(rng) % len] = alpha[ch(rng)];
        cases.push_back({p, t, "random"});
    }

    for (int smax : {64, 128, 256}) {
        for (auto& c : cases) {
            std::string P = c.P, T = c.T;
            PairView pv{T.data(), (int)T.size(), P.data(), (int)P.size(), smax};
            const int ref = (T.empty() || P.empty())
                          ? (int)(P.empty() ? T.size() : P.size())
                          : edit_distance_cpu(P.data(), (int)P.size(), T.data(), (int)T.size());
            const int b = run_one(pv, smax, true);    // flat kernel
            const int a = run_one(pv, smax, false);   // validated, for the coverage note
            ++checked;
            if (a >= 0) ++val_resolved;

            // The flat kernel is the thing under test. When it RESOLVES it must equal
            // the CPU DP exactly. When its true distance exceeds smax it may return
            // -1, which is by design -- but only if the reference is actually above
            // smax; resolving-nothing when the answer fits is a failure.
            if (b >= 0) {
                ++flat_resolved;
                if (b != ref) {
                    ++flat_wrong; ++flat_fail;
                    if (flat_wrong <= 8)
                        printf("  WRONG smax=%d [%s] P=%s T=%s: flat=%d cpu=%d\n",
                               smax, c.lab, P.c_str(), T.c_str(), b, ref);
                }
            } else if (ref <= smax) {
                // Should have been resolved within smax but wasn't.
                ++flat_fail;
                if (flat_wrong <= 8)
                    printf("  UNRESOLVED smax=%d [%s]: flat=-1 but cpu=%d <= smax\n",
                           smax, c.lab, ref);
            }
        }
    }

    printf("flat score kernel vs CPU DP (independent reference)\n");
    printf("  cases x smax                 : %d distinct x 3\n", (int)cases.size());
    printf("  total checks                 : %d\n", checked);
    printf("  flat resolved                : %d\n", flat_resolved);
    printf("  flat disagreements           : %d\n", flat_fail);
    printf("  [note] validated kernel under this shim resolved only %d of %d --\n",
           val_resolved, checked);
    printf("         its `threadIdx.x <= 2*s` guard computes one diagonal at blockDim=1,\n");
    printf("         so that run is degenerate and cannot serve as a reference.\n\n");
    if (flat_fail == 0 && flat_resolved > 0) {
        printf("  RESULT: PASS -- flat layout matches the CPU DP everywhere it resolves\n");
        printf("  NOTE: blockDim=1 here, so the THREAD REMAPPING is not exercised. The GPU\n");
        printf("        run (scripts/h7_flat.sbatch) is what verifies real threads.\n");
        return 0;
    }
    printf("  RESULT: FAIL (%d disagreements, %d resolved)\n", flat_fail, flat_resolved);
    return 1;
}
