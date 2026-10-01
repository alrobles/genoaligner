// genoaligner — codon stage-2 phase-DP kernel, body.
//
// The DP below IS the reference: msa_ref.cpp's codon_realign_row calls the
// same codon_phase_dp() on host, so host/kernel parity is by construction
// (the CPU shim compiles this file under g++). All float ops go through
// CR_F*_RN -- non-contractible on device, plain ops on host. Expression
// order mirrors msa_ref.cpp exactly; a-b is written CR_FADD_RN(a, -b)
// (IEEE-identical to subtraction).

#ifndef GENOALIGNER_CODON_REFINE_KERNEL_IMPL_HIP
#define GENOALIGNER_CODON_REFINE_KERNEL_IMPL_HIP

#include "genoaligner/backend/codon_refine_kernel.hip"

namespace genoaligner {

__device__ __host__ inline int cr_min(int a, int b) { return a < b ? a : b; }
__device__ __host__ inline int cr_max(int a, int b) { return a > b ? a : b; }

// T,C,A,G -> 0..3, else -1 (identical to genomsa::nt4).
__device__ __host__ inline int cr_nt4(char c) {
    switch (c) {
        case 'T': case 't': case 'U': case 'u': return 0;
        case 'C': case 'c': return 1;
        case 'A': case 'a': return 2;
        case 'G': case 'g': return 3;
        default: return -1;
    }
}

// 3 nts -> codon index 0..63, 64 = ambiguous (identical to codon_token3).
__device__ __host__ inline int cr_token3(const char* p) {
    int a = cr_nt4(p[0]), b = cr_nt4(p[1]), c = cr_nt4(p[2]);
    if (a < 0 || b < 0 || c < 0) return 64;
    return a * 16 + b * 4 + c;
}

// psgp_open with the msa.hpp formula (occ<1 branch only differs).
__device__ __host__ inline float cr_psgp_open(float occ,
                                              const CodonRefineParams& p) {
    if (!p.psgp || occ >= 1.0f) return CR_FMUL_RN(p.gap_open, occ);
    const float q  = CR_FMUL_RN(CR_FMUL_RN(p.psgp_scale, p.gap_open), occ);
    const float fl = CR_FMUL_RN(p.psgp_min_open, p.gap_open);
    return q > fl ? q : fl;
}

// nt vs base marginal, weighted by match/ts/tv (identical to ntm_score).
__device__ __host__ inline float cr_ntm_score(int nt, const float* marg,
                                              const CodonRefineParams& p) {
    float s = 0.f;
    for (int b = 0; b < 4; ++b) {
        if (!marg[b]) continue;
        float m = (nt == b) ? p.match
                 : ((nt==0&&b==2)||(nt==2&&b==0)||(nt==1&&b==3)||(nt==3&&b==1)
                    ? p.ts : p.tv);
        s = CR_FADD_RN(s, CR_FMUL_RN(marg[b], m));
    }
    return s;
}

// The phase DP for one row. Fills tr (K*ND direction bytes) and returns the
// endpoint s via *bes_out (-1 = unreachable -> host dumps the raw seq).
// dpb must hold 2*ND floats of scratch. Identical to the DP section of
// genomsa::codon_realign_row -- keep expression order in sync.
__device__ __host__ inline void codon_phase_dp(
        const char* seq, int L,
        const int8_t* stok,
        const int* pi, int K,
        const float* dot, const float* ocnt, const float* ncnt,
        const float* sub,
        const CodonRefineParams& P,
        uint8_t* tr, float* dpb, int* bes_out) {
    const int C = P.C;
    const float NEG = -1e30f;
    const float inv_nm1 = 1.f / (P.nseq - 1);
    const int D = P.band, ND = 2 * D + 1;
    float* prev = dpb;          // rolling dp row k-1, indexed s-3(k-1)+D
    float* cur  = dpb + ND;     // row k

    auto cscore = [&](int t, int j) -> float {
        float s = dot[(size_t)t * C + j];
        int sj = stok[j];
        if (sj >= 0) s = CR_FADD_RN(s, -sub[t * 65 + sj]);
        return CR_FMUL_RN(s, inv_nm1);
    };
    // nt marginals of column j minus self (same nseq-1 scale as cscore).
    auto marginals = [&](int j, float m[3][4]) -> bool {
        float ng = CR_FADD_RN(ocnt[j], -(stok[j] >= 0 ? 1.f : 0.f));
        if (ng <= 0) return false;
        int st_ = stok[j];
        const float* nc = ncnt + (size_t)j * 12;
        for (int p = 0; p < 3; ++p)
            for (int b = 0; b < 4; ++b) {
                float v = nc[p * 4 + b];
                if (st_ >= 0 && st_ < 64 && ((st_ >> (4 - 2 * p)) & 3) == b)
                    v = CR_FADD_RN(v, -1.f);
                m[p][b] = CR_FMUL_RN(v, inv_nm1);
            }
        return true;
    };
    auto pscore_at = [&](const char* p, int len, int j) -> float {
        if (len >= 3) return cscore(cr_token3(p), j);
        float m[3][4];
        if (!marginals(j, m)) return 0.f;
        int n0 = cr_nt4(p[0]);
        if (n0 < 0) return 0.f;
        if (len == 1) {
            float bs = -1e30f;
            for (int q = 0; q < 3; ++q) {
                float v = cr_ntm_score(n0, m[q], P);
                bs = v > bs ? v : bs;
            }
            return bs;
        }
        int n1 = cr_nt4(p[1]);
        if (n1 < 0) return 0.f;
        float bs = CR_FADD_RN(cr_ntm_score(n0, m[0], P),
                              cr_ntm_score(n1, m[1], P));
        float v2 = CR_FADD_RN(cr_ntm_score(n0, m[0], P),
                              cr_ntm_score(n1, m[2], P));
        bs = v2 > bs ? v2 : bs;
        float v3 = CR_FADD_RN(cr_ntm_score(n0, m[1], P),
                              cr_ntm_score(n1, m[2], P));
        return v3 > bs ? v3 : bs;
    };
    auto fsc = [&](int pos) -> float {
        return (pos < 3 || pos > L - 3) ? P.fs_term : P.fs;
    };
    auto occm = [&](int j) -> float {
        return CR_FMUL_RN(CR_FADD_RN(ocnt[j], -(stok[j] >= 0 ? 1.f : 0.f)),
                          inv_nm1);
    };

    // codon 0 starts at s in [0, min(D, L)]; s == L allowed (the slot may
    // be vacated entirely, e.g. a stage-1 partial-codon artefact).
    for (int s = 0; s < ND; ++s) prev[s] = NEG;
    for (int s = 0; s <= cr_min(D, L); ++s) prev[s + D] = 0.f;
    const int STEP[6] = {3, 4, 2, 5, 1, 0};
    for (int k = 1; k < K; ++k) {
        int slo = cr_max(0, 3 * k - D), shi = cr_min(L, 3 * k + D);
        for (int s = 0; s < ND; ++s) cur[s] = NEG;
        uint8_t* trk = tr + (size_t)k * ND;
        for (int s = slo; s <= shi; ++s) {
            float bv = NEG; int bst = 0;
            for (int si = 0; si < 6; ++si) {
                int st = STEP[si];
                int sp = s - st;
                if (sp < 0) continue;
                int dpk = sp - 3 * (k - 1);
                if (dpk > D || dpk < -D) continue;
                float pv = prev[dpk + D];
                if (pv <= NEG / 2) continue;
                float cand;
                if (st == 0) {                          // pi[k-1] empty
                    cand = CR_FADD_RN(pv, -cr_psgp_open(occm(pi[k - 1]), P));
                } else {
                    int len = cr_min(3, st);
                    if (sp + len > L) len = L - sp;     // seq tail partial
                    if (len < 1) continue;
                    cand = CR_FADD_RN(pv, pscore_at(seq + sp, len, pi[k - 1]));
                    if (st > 3)      cand = CR_FADD_RN(cand, -fsc(sp + 3));
                    else if (st < 3) cand = CR_FADD_RN(cand, -fsc(sp));
                }
                if (cand > bv) { bv = cand; bst = st; }
            }
            if (bv > NEG / 2) {
                cur[s - 3 * k + D] = bv;
                trk[s - 3 * k + D] = (uint8_t)(bst + 1);
            }
        }
        float* tmp = prev; prev = cur; cur = tmp;
    }
    // endpoint: codon K-1 starts at s; s == L leaves the last slot empty.
    int bes = -1; float eb = NEG;
    {
        int slo = cr_max(0, 3 * (K - 1) - D), shi = cr_min(L, 3 * (K - 1) + D);
        for (int s = slo; s <= shi; ++s) {
            float pv = prev[s - 3 * (K - 1) + D];
            if (pv <= NEG / 2) continue;
            float v = s < L ? CR_FADD_RN(pv, pscore_at(seq + s,
                                                     cr_min(3, L - s),
                                                     pi[K - 1]))
                            : pv;
            if (v > eb) { eb = v; bes = s; }
        }
    }
    *bes_out = bes;
}

// One thread per row; writes tr + bes. The host traceback reconstructs the
// placement (fill + insertion blocks) from them. The definition is emitted
// only in the launch TU (msa_gpu.cpp defines GENOALIGNER_CODON_REFINE_DEF);
// host TUs get codon_phase_dp but no (empty-qualifier) duplicate symbol.
#ifdef GENOALIGNER_CODON_REFINE_DEF
__global__ void codon_refine_kernel(const char* __restrict__ seqs,
                                    const size_t* __restrict__ seq_base,
                                    const int* __restrict__ seq_len,
                                    const int8_t* __restrict__ stok,
                                    const int* __restrict__ pi,
                                    const size_t* __restrict__ pi_base,
                                    const int* __restrict__ pi_len,
                                    const float* __restrict__ dot,
                                    const float* __restrict__ ocnt,
                                    const float* __restrict__ ncnt,
                                    const float* __restrict__ sub,
                                    CodonRefineParams params,
                                    uint8_t* __restrict__ tr,
                                    const size_t* __restrict__ tr_base,
                                    float* __restrict__ scr,
                                    const size_t* __restrict__ scr_base,
                                    int* __restrict__ bes,
                                    int n_rows)
{
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_rows) return;
    const int L = seq_len[r];
    const int K = pi_len[r];
    if (L <= 0 || K <= 0 || params.nseq <= 1) { bes[r] = -2; return; }
    codon_phase_dp(seqs + seq_base[r], L,
                   stok + (size_t)r * params.C,
                   pi + pi_base[r], K,
                   dot, ocnt, ncnt, sub, params,
                   tr + tr_base[r], scr + scr_base[r], bes + r);
}
#endif  // GENOALIGNER_CODON_REFINE_DEF

}  // namespace genoaligner

#endif  // GENOALIGNER_CODON_REFINE_KERNEL_IMPL_HIP
