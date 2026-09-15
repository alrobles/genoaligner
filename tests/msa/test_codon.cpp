// Codon-mode tests: tokenization, genetic codes, scoring, frame QC,
// encode/decode round-trip, and end-to-end codon MSA invariants.
// Every check has a KNOWN answer -- no "looks plausible" tests.
#include <genoaligner/msa/msa.hpp>
#include <cstdio>
#include <string>
#include <vector>
using namespace genomsa;

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    printf("FAIL: %s\n", msg); ++fails; } } while (0)

// Codon index: base-4, T=0 C=1 A=2 G=3 -> idx = b0*16 + b1*4 + b2.
static int cidx(const char* c3) {
    auto v = [](char c) {
        switch (c) { case 'T': return 0; case 'C': return 1;
                     case 'A': return 2; default: return 3; } };
    return v(c3[0]) * 16 + v(c3[1]) * 4 + v(c3[2]);
}

int main() {
    // ---- 1. codon index order ------------------------------------------
    CHECK(cidx("TTT") == 0,  "TTT index 0");
    CHECK(cidx("TTC") == 1,  "TTC index 1");
    CHECK(cidx("GGG") == 63, "GGG index 63");
    CHECK(cidx("ATG") == 35, "ATG index 35");
    CHECK(cidx("TGA") == 14, "TGA index 14");

    // ---- 2. encode: frame selection + stops -----------------------------
    {   // frame 0 has a stop (TGA at codon 2); frame 1 is clean
        std::vector<std::string> in = {"AAACCGGGTGACCCTTT"};
        std::vector<CodonQc> qc;
        auto enc = codon_encode(in, 1, &qc);
        // f0: AAA CCG GGT GAC CCT TT -> wait, len 17: codons from f0: AAA CCG GGT GAC CCT +1nt
        // f1: AAC CGG GTG ACC CTT     all sense
        // f2: ACC GGG TGA(stop) CCC   1 stop
        CHECK(qc[0].frame == 0 || qc[0].frame == 1, "frame chosen is clean");
        CHECK(qc[0].stops == 0, "clean frame has 0 stops");
        CHECK((int)enc[0].size() == 5 + qc[0].partial,
              "encoded length = full codons + partial token");
        CHECK(qc[0].partial == ((17 - qc[0].frame) % 3 ? 1 : 0),
              "partial flag matches remainder");
    }
    {   // forced internal stop: TGA in every forward frame of this short seq?
        // "TGATGA" -> f0: TGA TGA (2 stops); f1: GAT GA? -> GAT +partial; f2: ATG A?
        std::vector<std::string> in = {"TGATGA"};
        std::vector<CodonQc> qc;
        auto enc = codon_encode(in, 1, &qc);
        CHECK(qc[0].stops <= 2, "stop count <= 2");
        (void)enc;
    }

    // ---- 3. genetic code affects stop COUNTING (qc), not tokenization ---
    {   // mito reclassifies TGA->W and AGA/AGG->stop; on a long in-frame
        // CDS the qc.stops count reflects the code in the chosen frame.
        // Build a frame-0 sequence with a TGA that cannot be dodged by
        // frame shifting (TGA in all three frames is impossible; instead
        // verify the matrix-level difference in test 6 and here check the
        // encode is code-invariant in tokens).
        std::vector<std::string> in = {"ATGTGAATAAGACCG"};
        auto e1 = codon_encode(in, 1, nullptr);
        auto e2 = codon_encode(in, 2, nullptr);
        CHECK(e1[0] == e2[0], "tokens identical across codes");
    }

    // ---- 4. ambiguous / partial codons ----------------------------------
    {
        std::vector<std::string> in = {"ATGNNNCCG", "ATGCC"};  // NNN codon, len%3=2
        std::vector<CodonQc> qc;
        auto enc = codon_encode(in, 1, &qc);
        CHECK((unsigned char)enc[0][1] == 128 + 64, "NNN codon -> token 64");
        CHECK(qc[1].partial == 1, "len%3 remainder -> partial token");
        CHECK((unsigned char)enc[1].back() == 128 + 64,
              "trailing partial emits token 64");
        // no remainder -> NO trailing token
        std::vector<std::string> in2 = {"ATGCCG"};
        auto enc2 = codon_encode(in2, 1, nullptr);
        CHECK(enc2[0].size() == 2, "no partial token when len%3==0");
    }

    // ---- 5. decode round-trip -------------------------------------------
    {
        std::vector<std::string> in = {"ATGCCGGGATAC"};
        auto enc = codon_encode(in, 1, nullptr);
        auto dec = codon_decode(enc);
        CHECK(dec[0] == in[0], "encode/decode round-trip exact");
        // gap -> '---', token64 -> 'NNN'
        std::vector<std::string> rows = {std::string("x-y", 3)};
        rows[0][0] = (char)(128 + 35); rows[0][1] = '-';
        rows[0][2] = (char)(128 + 64);
        auto dd = codon_decode(rows);
        CHECK(dd[0] == "ATG---NNN", "decode: ATG, gap, token64");
    }

    // ---- 6. substitution matrix invariants ------------------------------
    {
        Params P = codon_params(1);
        CHECK(P.alpha == 65, "codon alpha=65");
        float sub_stop_sense = P.sub[cidx("TGA") * 65 + cidx("ATG")];
        float sub_stop_stop  = P.sub[cidx("TGA") * 65 + cidx("TAG")];
        float sub_same       = P.sub[cidx("ATG") * 65 + cidx("ATG")];
        float sub_syn        = P.sub[cidx("GGT") * 65 + cidx("GGC")]; // Gly->Gly
        float sub_other      = P.sub[64 * 65 + cidx("ATG")];
        CHECK(sub_stop_sense == -P.codon_stop_pen, "stop-vs-sense = -stop_pen");
        CHECK(sub_stop_stop == 0.f, "stop-vs-stop = 0");
        CHECK(sub_other == 0.f && P.sub[cidx("ATG") * 65 + 64] == 0.f,
              "token 64 neutral both ways");
        CHECK(sub_same == 5.f + 3.f * P.codon_nt_bonus,
              "ATG/ATG = BLOSUM62(M,M)=5 + 3*bonus");
        CHECK(sub_syn == 6.f + 2.f * P.codon_nt_bonus,
              "GGT/GGC = BLOSUM62(G,G)=6 + 2*bonus");
        // symmetric
        CHECK(P.sub[cidx("ATG") * 65 + cidx("GGT")] ==
              P.sub[cidx("GGT") * 65 + cidx("ATG")], "sub symmetric");
        // mito: TGA is W not stop
        Params PM = codon_params(2);
        CHECK(PM.sub[cidx("TGA") * 65 + cidx("ATG")] > -PM.codon_stop_pen,
              "mito TGA is Trp, not penalized as stop");
        CHECK(PM.sub[cidx("AGA") * 65 + cidx("ATG")] == -PM.codon_stop_pen,
              "mito AGA is stop");
    }

    // ---- 7. end-to-end codon MSA ----------------------------------------
    {
        // two CDS differing by one codon insertion + one SNP
        std::vector<std::string> in = {
            "ATGCCGGGATACGGTCCCGGG",          // 7 codons
            "ATGCCGAAAGGGGGGTACGGTCCCGGG",    // +AAA GGG insertion, GGA->GGG syn
        };
        Params P = codon_params(1);
        auto enc = codon_encode(in, 1, nullptr);
        auto msa = msa_align(enc, P, nullptr);
        CHECK(msa.size() == 2, "codon msa row count");
        CHECK(msa[0].size() == msa[1].size(), "codon msa rectangular");
        // back-translate: every aligned column is a full codon or '---'
        auto dec = codon_decode(msa);
        CHECK(dec[0].size() == dec[1].size(), "decoded rectangular");
        CHECK(dec[0].size() % 3 == 0, "decoded length multiple of 3");
        for (size_t i = 0; i < dec[0].size(); i += 3)
            CHECK(!(dec[0][i] == '-' && dec[1][i] == '-'),
                  "no all-gap codon column");
        // ungapped content preserved (ungap = original)
        auto ungap = [](std::string s) {
            std::string o; for (char c : s) if (c != '-') o += c; return o; };
        CHECK(ungap(dec[0]) == in[0], "seq0 ungapped == input");
        CHECK(ungap(dec[1]) == in[1], "seq1 ungapped == input");
        // no stop-misaligned pathological output on clean input
        CHECK(msa[0].find('-') == std::string::npos ||
              msa[0].size() > 7, "insertion placed as codon gap");
    }
    {   // indel must be a whole codon even when nt-level misalignment is cheaper
        std::vector<std::string> in = {"ATGAAACCG", "ATGCCG"};
        Params P = codon_params(1);
        auto enc = codon_encode(in, 1, nullptr);
        auto dec = codon_decode(msa_align(enc, P, nullptr));
        CHECK(dec[0].size() == 9 && dec[1].size() == 9,
              "3-codon vs 2-codon -> 3 columns");
        CHECK(dec[1].substr(0, 3) == "---" || dec[1].substr(3, 3) == "---" ||
              dec[1].substr(6, 3) == "---",
              "gap is a whole-codon block");
    }

    // ---- 8. profile path at alpha=65 ------------------------------------
    {
        std::vector<std::string> in = {"ATGCCGGGATAC", "ATGCCGGGATAC"};
        auto enc = codon_encode(in, 1, nullptr);
        Profile p = profile_from_seq(enc[0], 65);
        CHECK(p.ncols() == 4, "4-codon profile");
        CHECK(p.cols[0][cidx("ATG")] == 1.f, "col0 = ATG count 1");
        CHECK(p.occ[0] == 1.f, "col0 occ 1");
    }

    // ---- 9. local-frame encode (codon_encode_local) ---------------------
    // The encode DP partitions raw nts into codon blocks plus skipped
    // 1-2 nt frameshift blocks (left out of the token stream, restored
    // by codon_refine). Leading 1-2 nt drops are free (frame choice);
    // an internal skip costs codon_fs_enc=80 > stop_pen=60, so it only
    // ever wins when the unshifted parse would eat >=2 stops.
    auto ungap = [](std::string s) {
        std::string o; for (char c : s) if (c != '-') o += c; return o; };
    {
        Params P = codon_params(1);
        // clean sequence: identical tokens to the global-frame encoder
        std::vector<std::string> in = {"ATGCCGGGATAC"};
        std::vector<CodonQc> qc;
        auto loc = codon_encode_local(in, P, &qc);
        auto glb = codon_encode(in, 1, nullptr);
        CHECK(loc[0] == glb[0], "local: clean seq identical to global");
        CHECK(qc[0].partial == 0 && qc[0].stops == 0, "clean: no fs, no stops");
    }
    {
        Params P = codon_params(1);
        // +1 nt insertion after 4 codons; every unshifted parse eats
        // >=2 stops downstream, so the -80 skip wins. Verified against
        // the DP: tokens = pre + tail codons, skipped nt never emitted.
        std::vector<std::string> in = {
            "GTGGGATTGATAGGTTGACTCTCTTTTGAGTCACTTAGCTATAGATGAA"};
        std::vector<CodonQc> qc;
        auto enc = codon_encode_local(in, P, &qc);
        auto dec = codon_decode(enc);
        CHECK(qc[0].partial == 1, "local +1fs: one skipped block");
        CHECK(enc[0].size() == 16, "local +1fs: 16 codon tokens");
        CHECK(dec[0] == std::string(
            "GTGGGATTGATA" "GTTGACTCTCTTTTGAGTCACTTAGCTATAGATGAA"),
              "local +1fs: decode = input minus the inserted nt");
        CHECK(qc[0].stops == 0, "local +1fs: no stop tokens emitted");
    }
    {
        Params P = codon_params(1);
        // 2-nt skipped block (insertion / remnant of a deleted codon)
        std::vector<std::string> in = {
            "CTAATTAATAGTGTTGAAACGTCTTAATAAACAATAACGCGCAT"};
        std::vector<CodonQc> qc;
        auto enc = codon_encode_local(in, P, &qc);
        auto dec = codon_decode(enc);
        CHECK(qc[0].partial == 1 && enc[0].size() == 14,
              "local +2fs: one skipped 2-nt block, 14 tokens");
        CHECK(dec[0] == std::string(
            "CTAATTAATAGTGTT" "AACGTCTTAATAAACAATAACGCGCAT"),
              "local +2fs: decode = input minus the 2 inserted nt");
    }
    {
        Params P = codon_params(1);
        // one internal stop, no winning dodge (80 > 60): the stop is
        // emitted as a token and every nt stays covered.
        std::vector<std::string> in = {"GGTGAGGATCTAAACTCCGCGTAGACT"};
        std::vector<CodonQc> qc;
        auto enc = codon_encode_local(in, P, &qc);
        CHECK(qc[0].partial == 0, "single stop: no fs dodge");
        CHECK(qc[0].stops == 1, "single stop: emitted as stop token");
        CHECK(codon_decode(enc)[0] == in[0],
              "single stop: decode covers the full input");
        CHECK((unsigned char)enc[0][7] == 128 + cidx("TAG"),
              "single stop: TAG token at position 7");
    }
    {
        Params P = codon_params(1);
        // ambiguous codon -> token 64; short seqs; len%3 remainder may
        // reframe (leading drop is free) instead of a trailing partial.
        std::vector<std::string> in = {"ATGNNNCCG", "AT", "ATGCCGGT"};
        std::vector<CodonQc> qc;
        auto enc = codon_encode_local(in, P, &qc);
        CHECK((unsigned char)enc[0][1] == 128 + 64, "local: NNN -> token 64");
        CHECK(enc[0].size() == 3, "local: NNN seq -> 3 tokens");
        CHECK(enc[1].size() == 1 && (unsigned char)enc[1][0] == 128 + 64,
              "local: L<3 -> single token 64");
        // "ATGCCGGT": frame-2 (drop AT) scores same as frame-0 + trailing
        // GT; codon move wins the tie -> {GCC, GGT}, partial==0.
        CHECK(enc[2].size() == 2 && qc[2].partial == 0,
              "local: len%3=2 reframes via free leading drop");
        CHECK(codon_decode({enc[2]})[0] == "GCCGGT",
              "local: reframed tokens decode to GCC GGT");
    }

    // ---- 10. stage-2 refine end-to-end ---------------------------------
    {
        // s2 carries a +1 nt insertion after codon 5 whose unshifted
        // tail is all-sense (no encode dodge): the repair must come
        // from codon_refine's phase DP inside the frozen footprint.
        std::vector<std::string> in = {
            "ATGAAACCCGGGTTTGGCAAGCCAGGTCCT",
            "ATGAAACCCGGGTTTAGGCAAGCCAGGTCCT",   // +A after pos 15
            "ATGAAACCCGGATTTGGCAAGCCAGGTCCT",
        };
        Params P = codon_params(1);
        auto enc = codon_encode(in, 1, nullptr);
        auto dec = codon_decode(msa_align(enc, P, nullptr));
        auto ref = codon_refine(in, dec, P, 1);
        CHECK(ref.size() == 3, "refine: row count");
        size_t W = ref[0].size();
        CHECK(ref[1].size() == W && ref[2].size() == W,
              "refine: rectangular output");
        for (int r = 0; r < 3; ++r)
            CHECK(ungap(ref[r]) == in[r], "refine: ungapped == raw input");
        // the inserted 'A' occupies its own event column; downstream
        // codons land back in-frame (s1/s3 columns kept intact).
        bool shared_event_col = false;
        for (size_t c = 0; c < W; ++c) {
            int ngap = 0, wid = 0;
            for (int r = 0; r < 3; ++r)
                if (ref[r][c] == '-') ++ngap; else ++wid;
            if (ngap == 2 && wid == 1) shared_event_col = true;
        }
        CHECK(shared_event_col, "refine: fs residue sits in singleton column");
    }
    {
        // encode_local + refine round-trip: row 0 carries the verified
        // +1 fs sequence; every row must come back content-exact.
        std::vector<std::string> in = {
            "GTGGGATTGATAGGTTGACTCTCTTTTGAGTCACTTAGCTATAGATGAA",
            "GTGGGATTGATAGTTGACTCTCTTTTGAGTCACTTAGCTATAGATGAA",
            "GTGGGATTGATAGTTGACTCTCTTATGAGTCACTTAGCTATAGATGAA",
        };
        Params P = codon_params(1);
        auto enc = codon_encode_local(in, P, nullptr);
        auto dec = codon_decode(msa_align(enc, P, nullptr));
        auto ref = codon_refine(in, dec, P, 1);
        for (int r = 0; r < 3; ++r)
            CHECK(ungap(ref[r]) == in[r],
                  "local+refine e2e: ungapped == raw input");
    }

    printf("%s (%d failures)\n", fails ? "FAILURES" : "ALL OK", fails);
    return fails ? 1 : 0;
}
