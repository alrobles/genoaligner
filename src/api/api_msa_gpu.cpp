// genoaligner — public MSA API, device engine.
//
// The device counterpart of api_msa.cpp: identical validation and codon
// pipeline (api_msa_detail.hpp), but the token MSA runs through
// genomsa::msa_align_gpu -- NJ guide tree on the device, one
// profile-profile kernel launch per guide-tree level, host merge.
//
// This TU itself is pure C++: it calls engine functions, it launches no
// kernels. The HIP requirement lives in src/msa/msa_gpu.cpp (genomsa_gpu),
// which callers link transitively through libgenoaligner. Under the CPU
// shim the engine's kernels execute on host, so this entry is exercisable
// in the gate exactly like the pairwise API is.
//
// NO device_available() early-check here on purpose: under the shim that
// query answers false while the engine path still works. The engine's own
// false+err return is the single source of "the device could not serve".

#include "genoaligner/api.hpp"
#include "genoaligner/msa/msa.hpp"
#include "api_msa_detail.hpp"

#include <exception>

namespace genoaligner {

MsaResult msa_align_device(const MsaRequest& req) {
    bool is_prot = false, is_codon = false;
    MsaResult bad = msa_detail::check_request(req, &is_prot, &is_codon);
    if (!bad.ok()) return bad;

    const genomsa::Params P = msa_detail::params_for(req);
    MsaResult res;
    try {
        if (is_codon) {
            res = msa_detail::run_codon(req, P,
                [&P](const std::vector<std::string>& tok,
                     std::vector<std::string>& out, std::string& err) -> bool {
                    return genomsa::msa_align_gpu(tok, P, out, err);
                },
                [](const std::vector<std::string>& s,
                   const std::vector<std::string>& a,
                   const genomsa::Params& PP, int r,
                   std::vector<std::string>& out, std::string& err) -> bool {
                    return genomsa::codon_refine_gpu(s, a, PP, r, out, err);
                },
                "msa_align_device: device engine failed on the token MSA");
            if (res.ok()) res.device = true;
        } else {
            std::string err;
            if (!genomsa::msa_align_gpu(req.seqs, P, res.aligned, err))
                return msa_detail::fail(
                    MsaResult::Status::device_error,
                    "msa_align_device: device engine failed");
            res.qc.assign(req.seqs.size(), MsaSeqQc{});
            res.device = true;
        }
    } catch (const std::exception&) {
        return msa_detail::fail(MsaResult::Status::error,
                                "msa_align_device: engine threw an exception");
    } catch (...) {
        return msa_detail::fail(MsaResult::Status::error,
                                "msa_align_device: engine failed");
    }
    return msa_detail::verify_contract(std::move(res), req.seqs.size());
}

} // namespace genoaligner
