// genoaligner — public MSA API implementation.
//
// Thin composition layer over the genomsa engine (src/msa/msa_ref.cpp),
// kept to the same discipline as the pairwise section of this library: own
// the validation, the parameter assembly and the result contract, nothing
// else. The engine is pure host code, so this TU needs no GPU, no shim and
// no backend headers -- it compiles and runs identically on every platform.

#include "genoaligner/api.hpp"
#include "genoaligner/msa/msa.hpp"

#include <cctype>
#include <exception>

namespace genoaligner {
namespace {

// IUPAC nucleotide letters, uppercased on the fly. '-' is deliberately NOT
// in the set: input sequences must be unaligned raw data -- a pre-gapped
// sequence smuggles a different problem into the engine.
bool valid_nt(const std::string& s) {
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
bool valid_aa(const std::string& s) {
    for (char ch : s) {
        if (std::isalpha((unsigned char)ch) || ch == '*' || ch == '?') continue;
        return false;
    }
    return true;
}

genomsa::Params params_for(const MsaRequest& req) {
    if (req.mode == MsaMode::protein) return genomsa::protein_params();
    if (req.mode == MsaMode::codon) {
        genomsa::Params P = genomsa::codon_params(req.gc_def);
        P.codon_refine      = req.codon_refine;
        P.codon_local_frame = req.codon_local_frame ? 1 : 0;
        return P;
    }
    return genomsa::Params{};   // dna: engine defaults (alpha=4)
}

MsaResult fail(MsaResult::Status st, const char* msg) {
    MsaResult r;
    r.status = st;
    r.error  = msg;
    return r;
}

} // namespace

MsaResult msa_align(const MsaRequest& req) {
    // ------------------------------------------------------------ validation
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

    const bool is_prot  = req.mode == MsaMode::protein;
    const bool is_codon = req.mode == MsaMode::codon;
    for (const std::string& s : req.seqs) {
        const bool good = is_prot ? valid_aa(s) : valid_nt(s);
        if (!good)
            return fail(MsaResult::Status::invalid_argument,
                        is_prot ? "msa_align: character outside the amino-acid"
                                  " alphabet"
                                : "msa_align: character outside the IUPAC"
                                  " nucleotide alphabet (input must be"
                                  " unaligned raw sequence)");
    }
    if (req.gc_def != 1 && req.gc_def != 2)
        return fail(MsaResult::Status::invalid_argument,
                    "msa_align: gc_def must be NCBI table 1 or 2");
    if (!is_codon && req.gc_def != 1)
        return fail(MsaResult::Status::invalid_argument,
                    "msa_align: gc_def is only meaningful in codon mode");
    if (!is_codon && req.codon_refine != 0)
        return fail(MsaResult::Status::invalid_argument,
                    "msa_align: codon_refine is only meaningful in codon mode");
    if (req.codon_refine < 0)
        return fail(MsaResult::Status::invalid_argument,
                    "msa_align: codon_refine must be >= 0");

    // ------------------------------------------------------------------- run
    const genomsa::Params P = params_for(req);
    MsaResult res;
    try {
        if (is_codon) {
            std::vector<genomsa::CodonQc> cqc;
            std::vector<std::string> tok =
                req.codon_local_frame
                    ? genomsa::codon_encode_local(req.seqs, P, &cqc)
                    : genomsa::codon_encode(req.seqs, P.gc_def, &cqc);
            std::vector<std::string> msa = genomsa::msa_align(tok, P);
            msa = genomsa::codon_decode(msa);
            // Same rule as the driver's --local-frame: placeholder tokens
            // only become real nts through the refine pass, so local-frame
            // output without >=1 round is not content-correct.
            int rounds = req.codon_refine;
            if (req.codon_local_frame && rounds < 1) rounds = 1;
            if (rounds > 0)
                msa = genomsa::codon_refine(req.seqs, msa, P, rounds);
            res.aligned = std::move(msa);
            res.qc.assign(req.seqs.size(), MsaSeqQc{});
            for (size_t i = 0; i < cqc.size() && i < res.qc.size(); ++i) {
                res.qc[i].frame   = cqc[i].frame;
                res.qc[i].stops   = cqc[i].stops;
                res.qc[i].partial = cqc[i].partial;
            }
        } else {
            res.aligned = genomsa::msa_align(req.seqs, P);
            res.qc.assign(req.seqs.size(), MsaSeqQc{});
        }
    } catch (const std::exception&) {
        return fail(MsaResult::Status::error,
                    "msa_align: engine threw an exception");
    } catch (...) {
        return fail(MsaResult::Status::error,
                    "msa_align: engine failed");
    }

    // ------------------------------------------------------------- contract
    // Verified here, not trusted: the engine documents equal-width rows in
    // input order, and a caller that reads ragged output off this API would
    // see a silent contract break -- the failure mode the flags exist for.
    if (res.aligned.size() != req.seqs.size())
        return fail(MsaResult::Status::error,
                    "msa_align: engine returned a different row count");
    res.width = res.aligned.empty() ? 0 : (int)res.aligned[0].size();
    for (const std::string& row : res.aligned)
        if ((int)row.size() != res.width)
            return fail(MsaResult::Status::error,
                        "msa_align: engine returned ragged rows");
    return res;
}

} // namespace genoaligner
