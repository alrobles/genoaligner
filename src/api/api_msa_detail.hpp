// genoaligner — internals shared by the two public MSA entries.
//
// api_msa.cpp (host engine) and api_msa_gpu.cpp (device engine) must run
// IDENTICAL validation and assemble IDENTICAL engine params, or the two
// entries drift: a request refused by one and served by the other is a
// contract break. They also verify the same result contract on the way out.
// Keeping those pieces here, inline, means there is exactly one copy.
//
// Private header: src/api only, never installed.

#ifndef GENOALIGNER_API_MSA_DETAIL_HPP
#define GENOALIGNER_API_MSA_DETAIL_HPP

#include "genoaligner/api.hpp"
#include "genoaligner/msa/msa.hpp"

#include <cctype>
#include <string>

namespace genoaligner {
namespace msa_detail {

// IUPAC nucleotide letters, uppercased on the fly. '-' is deliberately NOT
// in the set: input sequences must be unaligned raw data -- a pre-gapped
// sequence smuggles a different problem into the engine.
inline bool valid_nt(const std::string& s) {
    for (char ch : s) {
        switch (std::toupper((unsigned char)ch)) {
        case 'A': case 'C': case 'G': case 'T': case 'U':
        case 'R': case 'Y': case 'S': case 'W': case 'K': case 'M':
        case 'B': case 'D': case 'H': case 'V': case 'N': case '?':
            break;
        default:
            return false;
        }
    }
    return true;
}

// Amino-acid input: ASCII letters plus '*' and '?' (the engine maps
// B/Z/J/X/O/U itself; digits and punctuation are caller bugs).
inline bool valid_aa(const std::string& s) {
    for (char ch : s) {
        if (std::isalpha((unsigned char)ch) || ch == '*' || ch == '?') continue;
        return false;
    }
    return true;
}

inline genomsa::Params params_for(const MsaRequest& req) {
    if (req.mode == MsaMode::protein) return genomsa::protein_params();
    if (req.mode == MsaMode::codon) {
        genomsa::Params P = genomsa::codon_params(req.gc_def);
        P.codon_refine      = req.codon_refine;
        P.codon_local_frame = req.codon_local_frame ? 1 : 0;
        return P;
    }
    return genomsa::Params{};   // dna: engine defaults (alpha=4)
}

inline MsaResult fail(MsaResult::Status st, const char* msg) {
    MsaResult r;
    r.status = st;
    r.error  = msg;
    return r;
}

// Full request validation, identical for both engines. Returns a ready-made
// refusal result, or an ok-status result when the request is servable --
// check .ok() on the return value; mode flags come back through the outs.
inline MsaResult check_request(const MsaRequest& req,
                               bool* is_prot, bool* is_codon) {
    if (req.seqs.empty())
        return fail(MsaResult::Status::invalid_argument,
                    "msa_align: request carries no sequences");
    for (const std::string& s : req.seqs)
        if (s.empty())
            return fail(MsaResult::Status::invalid_argument,
                        "msa_align: request carries an empty sequence");

    // A casted-out-of-range mode would otherwise fall through to the dna
    // branch and get answered -- a guess, not a refusal.
    if (req.mode != MsaMode::dna && req.mode != MsaMode::protein &&
        req.mode != MsaMode::codon)
        return fail(MsaResult::Status::invalid_argument,
                    "msa_align: unknown mode");

    *is_prot  = req.mode == MsaMode::protein;
    *is_codon = req.mode == MsaMode::codon;
    for (const std::string& s : req.seqs) {
        const bool good = *is_prot ? valid_aa(s) : valid_nt(s);
        if (!good)
            return fail(MsaResult::Status::invalid_argument,
                        *is_prot ? "msa_align: character outside the amino-acid"
                                   " alphabet"
                                 : "msa_align: character outside the IUPAC"
                                   " nucleotide alphabet (input must be"
                                   " unaligned raw sequence)");
    }
    if (req.gc_def != 1 && req.gc_def != 2)
        return fail(MsaResult::Status::invalid_argument,
                    "msa_align: gc_def must be NCBI table 1 or 2");
    if (!*is_codon && req.gc_def != 1)
        return fail(MsaResult::Status::invalid_argument,
                    "msa_align: gc_def is only meaningful in codon mode");
    if (!*is_codon && req.codon_refine != 0)
        return fail(MsaResult::Status::invalid_argument,
                    "msa_align: codon_refine is only meaningful in codon mode");
    if (req.codon_refine < 0)
        return fail(MsaResult::Status::invalid_argument,
                    "msa_align: codon_refine must be >= 0");
    return MsaResult{};
}

// Codon pipeline shared by both engines: tokenize -> run the token MSA
// through ENG -> decode -> refine through REF, per the driver's rules
// (local_frame implies >= 1 round; its placeholder tokens only become real
// nts through the refine pass). ENG is msa_align on the host entry and
// msa_align_gpu on the device entry; REF is codon_refine / codon_refine_gpu
// respectively. Both take (in..., out, err) and return bool -- a false from
// ENG produces eng_fail; a false from REF falls back to the host pass, whose
// DP body is the same source as the device kernel (api.hpp documents why
// that one fallback is honest).
template <typename Eng, typename Ref>
MsaResult run_codon(const MsaRequest& req, const genomsa::Params& P,
                    Eng&& eng, Ref&& ref, const char* eng_fail) {
    std::vector<genomsa::CodonQc> cqc;
    std::vector<std::string> tok =
        req.codon_local_frame
            ? genomsa::codon_encode_local(req.seqs, P, &cqc)
            : genomsa::codon_encode(req.seqs, P.gc_def, &cqc);
    std::vector<std::string> msa;
    std::string err;
    if (!eng(tok, msa, err))
        return fail(MsaResult::Status::device_error, eng_fail);
    msa = genomsa::codon_decode(msa);
    int rounds = req.codon_refine;
    if (req.codon_local_frame && rounds < 1) rounds = 1;
    if (rounds > 0) {
        std::vector<std::string> refined;
        if (!ref(req.seqs, msa, P, rounds, refined, err))
            refined = genomsa::codon_refine(req.seqs, msa, P, rounds);
        msa = std::move(refined);
    }
    MsaResult res;
    res.aligned = std::move(msa);
    res.qc.assign(req.seqs.size(), MsaSeqQc{});
    for (size_t i = 0; i < cqc.size() && i < res.qc.size(); ++i) {
        res.qc[i].frame   = cqc[i].frame;
        res.qc[i].stops   = cqc[i].stops;
        res.qc[i].partial = cqc[i].partial;
    }
    return res;
}

// The contract is verified here, not trusted: the engine documents
// equal-width rows in input order, and a caller reading ragged output off
// this API would see a silent contract break -- the failure mode the flags
// exist for.
inline MsaResult verify_contract(MsaResult res, size_t nseq) {
    if (res.aligned.size() != nseq)
        return fail(MsaResult::Status::error,
                    "msa_align: engine returned a different row count");
    res.width = res.aligned.empty() ? 0 : (int)res.aligned[0].size();
    for (const std::string& row : res.aligned)
        if ((int)row.size() != res.width)
            return fail(MsaResult::Status::error,
                        "msa_align: engine returned ragged rows");
    return res;
}

} // namespace msa_detail
} // namespace genoaligner

#endif
