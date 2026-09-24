// genoaligner — public MSA device-API test.
//
// WHAT THIS VERIFIES
// ------------------
// msa_align_device must serve rows BIT-IDENTICAL to msa_align on the same
// request: the public promise is "same answer, different processor", and a
// silent semantic difference between the two entries is exactly the drift
// the parity gates exist to catch. Under the CPU shim the device kernels
// execute on host, so every gate run asserts real results -- not mocks.
//
// On a device-less REAL build the entry must refuse with
// Status::device_error (it never silently serves host work). The test
// verifies that refusal is the right one, then exits 77 so ctest marks the
// environment, not the code, as the reason nothing ran.

#include "genoaligner/api.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace genoaligner;

static int g_fail = 0;
static void check(bool ok, const char* what)
{
    if (!ok) { printf("  FAIL: %s\n", what); ++g_fail; }
}

static bool same_result(const MsaResult& a, const MsaResult& b)
{
    if (a.ok() != b.ok() || a.width != b.width ||
        a.aligned.size() != b.aligned.size() ||
        a.qc.size() != b.qc.size()) return false;
    if (a.aligned != b.aligned) return false;
    for (size_t i = 0; i < a.qc.size(); ++i)
        if (a.qc[i].frame != b.qc[i].frame ||
            a.qc[i].stops != b.qc[i].stops ||
            a.qc[i].partial != b.qc[i].partial) return false;
    return true;
}

int main()
{
    printf("genoaligner public MSA device API test\n");

    // ------------------------------------------------------------- probe
    // One cheap call decides whether any device path exists. A refusal here
    // must be device_error specifically -- invalid_argument would mean the
    // request was blamed, error would mean an engine fault; only
    // device_error says "asked for the device, none could serve".
    {
        MsaRequest req;
        req.seqs = {"ACGTACGT", "ACGTTCGT"};
        MsaResult r = msa_align_device(req);
        if (r.status == MsaResult::Status::device_error) {
            printf("  no device path available -- refusal is device_error "
                   "(correct semantics); SKIP\n");
            return 77;
        }
        check(r.ok(), "probe: device entry resolves");
        check(r.device, "probe: device provenance recorded");
    }

    // -------------------------------------------------- device == host
    // The core promise: same request through both entries, identical rows.
    {
        MsaRequest req;
        req.mode = MsaMode::dna;
        req.seqs = {"ACGTACGTACGT", "ACGTACGTTCGT",
                    "ACGTTCGTACGT", "ACGTACGTACGTA"};
        MsaResult host = msa_align(req);
        MsaResult dev  = msa_align_device(req);
        check(dev.ok(), "dna device: resolves");
        check(dev.device && !host.device, "dna: provenance flags differ");
        check(same_result(host, dev),
              "dna: device rows bit-identical to host");
    }
    {
        MsaRequest req;
        req.mode = MsaMode::protein;
        req.seqs = {"MKTAYIAKQRQISFVK", "MKTAYIAKQRQISFVKE",
                    "MKTAZIAKQRQISFVK"};
        check(same_result(msa_align(req), msa_align_device(req)),
              "protein: device rows bit-identical to host");
    }
    {
        MsaRequest req;   // strict codon: whole-codon invariant on device too
        req.mode = MsaMode::codon;
        req.codon_local_frame = false;
        req.seqs = {"ATGAAACCCGGGTTT", "ATGAAAGGGTTT", "ATGAAACCCGGGTTTAAA"};
        MsaResult dev = msa_align_device(req);
        check(same_result(msa_align(req), dev),
              "codon strict: device rows bit-identical to host");
        check(dev.width % 3 == 0, "codon strict device: width multiple of 3");
        check(dev.qc.size() == 3 && dev.qc[0].frame >= 0,
              "codon strict device: qc carried through");
    }
    {
        MsaRequest req;   // local-frame + implied refine, the default path
        req.mode = MsaMode::codon;
        req.seqs = {"ATGAAACCCGGGTTT", "ATGAAAGGGTTT", "ATGAAACCCGGGTTTAAA"};
        check(same_result(msa_align(req), msa_align_device(req)),
              "codon local-frame: device rows bit-identical to host");
    }
    {
        MsaRequest req;
        req.seqs = {"ACGTACGT"};
        MsaResult dev = msa_align_device(req);
        check(dev.ok() && dev.aligned.size() == 1 &&
              dev.aligned[0] == req.seqs[0] && dev.device,
              "one sequence returns itself on device too");
    }

    // ------------------------------------------------- validation parity
    // The two entries share check_request: every refusal must be identical,
    // and a refused result must not claim device provenance.
    {
        MsaRequest req;
        check(msa_align_device(req).status ==
              MsaResult::Status::invalid_argument,
              "device validation: empty request refused");
        req.seqs = {"ACGT!"};
        MsaResult r = msa_align_device(req);
        check(r.status == MsaResult::Status::invalid_argument && !r.device,
              "device validation: bad char refused, no provenance");
        req.seqs = {"ACGT"};
        req.mode = static_cast<MsaMode>(99);
        check(msa_align_device(req).status ==
              MsaResult::Status::invalid_argument,
              "device validation: out-of-range mode refused");
    }

    // Determinism on the device path as well.
    {
        MsaRequest req;
        req.seqs = {"ACGTACGTACGT", "ACGTACGTTCGT", "ACGTTCGTACGT"};
        check(msa_align_device(req).aligned == msa_align_device(req).aligned,
              "device: deterministic on repeat call");
    }

    if (g_fail == 0) printf("MSA device API test: PASS\n");
    else             printf("MSA device API test: %d FAILURE(S)\n", g_fail);
    return g_fail ? 1 : 0;
}
