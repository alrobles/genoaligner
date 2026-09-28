// genoaligner — public MSA API, host engine.
//
// Thin composition layer over the genomsa engine (src/msa/msa_ref.cpp),
// kept to the same discipline as the pairwise section of this library: own
// the validation, the parameter assembly and the result contract, nothing
// else. The engine is pure host code, so this TU needs no GPU, no shim and
// no backend headers -- it compiles and runs identically on every platform.
// Validation and the codon pipeline live in api_msa_detail.hpp, shared with
// api_msa_gpu.cpp so the two public entries cannot drift apart.

#include "genoaligner/api.hpp"
#include "genoaligner/msa/msa.hpp"
#include "api_msa_detail.hpp"

#include <exception>

namespace genoaligner {

MsaResult msa_align(const MsaRequest& req) {
    bool is_prot = false, is_codon = false;
    MsaResult bad = msa_detail::check_request(req, &is_prot, &is_codon);
    if (!bad.ok()) return bad;

    const genomsa::Params P = msa_detail::params_for(req);
    MsaResult res;
    try {
        if (is_codon) {
            res = msa_detail::run_codon(req, P,
                [&P](const std::vector<std::string>& tok,
                     std::vector<std::string>& out, std::string&) -> bool {
                    out = genomsa::msa_align(tok, P);
                    return true;
                },
                [](const std::vector<std::string>& s,
                   const std::vector<std::string>& a,
                   const genomsa::Params& PP, int r,
                   std::vector<std::string>& out, std::string&) -> bool {
                    out = genomsa::codon_refine(s, a, PP, r);
                    return true;
                },
                "msa_align: host engine failed");
        } else {
            res.aligned = genomsa::msa_align(req.seqs, P);
            res.qc.assign(req.seqs.size(), MsaSeqQc{});
        }
    } catch (const std::exception&) {
        return msa_detail::fail(MsaResult::Status::error,
                                "msa_align: engine threw an exception");
    } catch (...) {
        return msa_detail::fail(MsaResult::Status::error,
                                "msa_align: engine failed");
    }
    return msa_detail::verify_contract(std::move(res), req.seqs.size());
}

} // namespace genoaligner
