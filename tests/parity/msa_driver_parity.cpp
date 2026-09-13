// Driver parity: the SHIPPED GPU driver (src/msa/msa_gpu.cpp) compiled under
// the CPU shim vs the sequential CPU reference driver. The shim executes the
// real packing arithmetic, real buffer sizing, real level batches and the
// shipped kernel body -- so agreement here means the GPU path as DEPLOYED
// equals the reference, not a re-implementation of it.
//
// Build:  g++ -O2 -std=c++17 -DGENOALIGNER_HIP_SHIM \
//           -I tests/parity/hip_cpu_shim -I include \
//           tests/parity/msa_driver_parity.cpp \
//           src/msa/msa_gpu.cpp src/msa/msa_ref.cpp
#include <genoaligner/msa/msa.hpp>
#include <cstdio>
#include <random>
#include <vector>
#include <string>

using namespace genomsa;

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); ++fails; } } while (0)

static std::mt19937 rng(31337);
static std::string rand_dna(int len) {
    static const char b[] = "ACGT";
    std::string s; for (int i = 0; i < len; ++i) s += b[rng() % 4]; return s;
}
static std::string mutate(const std::string& s, int ps, int pi, int pd) {
    static const char b[] = "ACGT";
    std::string o;
    for (char c : s) {
        int r = rng() % 100;
        if (r < pd) continue;
        o += (r < pd + ps) ? b[rng() % 4] : c;
        if (rng() % 100 < pi) o += b[rng() % 4];
    }
    return o;
}

static void run_suite(Params P, const char* tag0) {
    for (int n : {2, 3, 4, 5, 7, 8, 11, 16, 24}) {
        std::string anc = rand_dna(50 + rng() % 100);
        std::vector<std::string> in;
        for (int i = 0; i < n; ++i) in.push_back(mutate(anc, 12, 6, 6));
        if (n >= 5) in[3] = in[3].substr(3, in[3].size() / 2);      // fragment
        if (n >= 8) { std::string& q = in[6]; if (q.size() > 10) q[7] = 'N'; }

        auto ref = msa_align(in, P);
        std::vector<std::string> gpu; std::string err; GpuStats st;
        bool ok = msa_align_gpu(in, P, gpu, err, &st);
        CHECK(ok, "%sn=%d driver error: %s", tag0, n, err.c_str());
        if (!ok) continue;
        CHECK(gpu == ref, "%sn=%d driver != reference", tag0, n);
        CHECK(st.levels >= 1 && st.pairs == n - 1,
              "%sn=%d stats levels=%d pairs=%d", tag0, n, st.levels, st.pairs);
        for (int i = 0; i < n; ++i) {
            std::string u = gpu[i];
            u.erase(std::remove(u.begin(), u.end(), '-'), u.end());
            CHECK(u == in[i], "%sn=%d seq %d corrupted", tag0, n, i);
        }
    }
}

int main() {
    run_suite(Params{}, "default:");
    Params l; l.psgp = false; l.gappy = 0;
    run_suite(l, "legacy:");
    Params p; p.gappy = 0;
    run_suite(p, "psgp:");
    Params g; g.psgp = false; g.gappy = 0.9f;
    run_suite(g, "gappy:");
    Params pg; pg.gappy = 0.9f;
    run_suite(pg, "psgp+gappy:");
    // n=1 degenerate
    {
        std::vector<std::string> one = {"ACGTACGT"};
        std::vector<std::string> gpu; std::string err;
        CHECK(msa_align_gpu(one, Params{}, gpu, err) && gpu == one,
              "n=1 degenerate");
    }

    if (!fails) printf("RESULT: PASS -- shipped GPU driver == CPU reference, bit-exact\n");
    else        printf("%d FAILURES\n", fails);
    return fails;
}
