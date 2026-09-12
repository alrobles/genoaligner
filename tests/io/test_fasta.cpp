// genoaligner — B1: FASTA reader tests.
//
// Each case below corresponds to a promise made in include/genoaligner/io/fasta.hpp.
// The multi-line case is the one that matters most: a parser that assumes one line
// per record truncates almost every real genome, and it does so SILENTLY -- the
// record count looks right, so a count-only test would pass.

#include "include/genoaligner/io/fasta.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using genoaligner::io::FastaRecord;
using genoaligner::io::read_fasta;
using genoaligner::io::read_fasta_file;

static int g_fail = 0;

static void ok(bool cond, const char* what)
{
    if (!cond) { printf("  FAIL: %s\n", what); ++g_fail; }
}

// Parse a string as if it were a file.
static bool parse(const std::string& text, std::vector<FastaRecord>* recs, std::string* err)
{
    FILE* f = std::tmpfile();
    if (!f) return false;
    std::fwrite(text.data(), 1, text.size(), f);
    std::rewind(f);
    const bool r = read_fasta(f, recs, err);
    std::fclose(f);
    return r;
}

int main()
{
    printf("FASTA reader tests\n\n");

    // ---- the basic case, and MULTI-LINE sequence -------------------------
    printf("-- multi-line sequence (the case that silently truncates) --\n");
    {
        std::vector<FastaRecord> r;
        std::string err;
        const std::string text =
            ">seq1 first record\n"
            "ACGTACGTAC\n"
            "GTACGTACGT\n"
            "ACGT\n"
            ">seq2 second\n"
            "TTTT\n";
        ok(parse(text, &r, &err), "parses");
        ok(r.size() == 2, "two records");
        ok(r[0].id == "seq1", "id is the first token");
        ok(r[0].description == "first record", "description keeps the remainder");
        ok(r[0].sequence == "ACGTACGTACGTACGTACGTACGT", "multi-line sequence JOINED");
        ok(r[0].sequence.size() == 24, "sequence length 24, not 10");
        ok(r[1].id == "seq2" && r[1].sequence == "TTTT", "second record intact");
        printf("  records=%zu  seq1 len=%zu (expected 24)\n", r.size(), r[0].sequence.size());
    }

    // ---- CRLF ------------------------------------------------------------
    printf("\n-- CRLF line endings --\n");
    {
        std::vector<FastaRecord> r;
        std::string err;
        const std::string text = ">a\r\nACGT\r\nACGT\r\n>b\r\nTT\r\n";
        ok(parse(text, &r, &err), "parses CRLF");
        ok(r.size() == 2, "two records");
        ok(r[0].sequence == "ACGTACGT", "CR removed from sequence");
        ok(r[0].id == "a" && r[0].description.empty(), "CR removed from header");
        ok(r[1].sequence == "TT", "second record clean");
        printf("  seq '%s' id '%s'\n", r[0].sequence.c_str(), r[0].id.c_str());
    }

    // ---- blank lines, and a bare identifier header -----------------------
    printf("\n-- blank lines and bare headers --\n");
    {
        std::vector<FastaRecord> r;
        std::string err;
        const std::string text =
            "\n"
            ">only_id\n"
            "\n"
            "ACGT\n"
            "\n"
            "\n"
            ">with_desc  some words here\n"
            "NNNN\n";
        ok(parse(text, &r, &err), "parses blank lines");
        ok(r.size() == 2, "two records");
        ok(r[0].id == "only_id" && r[0].description.empty(), "bare header -> empty description");
        ok(r[0].sequence == "ACGT", "blank line does not break the record");
        ok(r[1].id == "with_desc" && r[1].description == "some words here",
           "tabs/spaces in header handled");
        printf("  '%s' / '%s'\n", r[1].id.c_str(), r[1].description.c_str());
    }

    // ---- real-world sequence content: lowercase, N, ambiguity ------------
    printf("\n-- non-ACGT content is preserved, not rejected --\n");
    {
        std::vector<FastaRecord> r;
        std::string err;
        const std::string text = ">masked\nacgtNNNRYSWacgt\n";
        ok(parse(text, &r, &err), "accepts lowercase and ambiguity codes");
        ok(r[0].sequence == "acgtNNNRYSWacgt", "case and ambiguity preserved byte-for-byte");
        printf("  '%s'\n", r[0].sequence.c_str());
    }

    // ---- errors: each must FAIL, not return something plausible ----------
    printf("\n-- malformed input must fail --\n");
    {
        std::vector<FastaRecord> r;
        std::string err;

        r.clear(); err.clear();
        ok(!parse("ACGT\n", &r, &err), "no '>' at all -> fail");
        printf("  no header   : err='%s'\n", err.c_str());

        r.clear(); err.clear();
        ok(!parse(">empty\n>next\nACGT\n", &r, &err), "header with no sequence -> fail");
        printf("  empty record: err='%s'\n", err.c_str());

        r.clear(); err.clear();
        ok(!parse("", &r, &err), "empty file -> fail (0 records is not a FASTA)");
        printf("  empty file  : err='%s'\n", err.c_str());

        r.clear(); err.clear();
        ok(!parse(">   \nACGT\n", &r, &err), "whitespace-only identifier -> fail");
        printf("  blank id    : err='%s'\n", err.c_str());
    }

    // ---- the error path must not leave a half-parsed record set ----------
    printf("\n-- a failed parse is reported, not silently partial --\n");
    {
        std::vector<FastaRecord> r;
        std::string err;
        // Two good records, then a broken one.
        const bool okp = parse(">a\nAC\n>b\nTT\n>c\n", &r, &err);
        ok(!okp, "a later bad record fails the whole parse");
        printf("  partial records collected=%zu (parser returned false)\n", r.size());
        printf("  NOTE: the caller must treat false as 'discard everything'.\n");
    }

    // ---- file-level API --------------------------------------------------
    printf("\n-- read_fasta_file --\n");
    {
        const char* path = "/tmp/genoaligner_test.fa";
        FILE* f = std::fopen(path, "wb");
        if (f) {
            std::fputs(">x\nACGT\n", f);
            std::fclose(f);
            std::vector<FastaRecord> r;
            std::string err;
            ok(read_fasta_file(path, &r, &err), "reads a real file");
            ok(r.size() == 1 && r[0].id == "x" && r[0].sequence == "ACGT", "content correct");
            printf("  '%s' -> %zu record(s)\n", path, r.size());
            std::remove(path);

            std::vector<FastaRecord> r2;
            std::string err2;
            ok(!read_fasta_file("/tmp/definitely_not_here_9f3a.fa", &r2, &err2),
               "missing file -> fail");
            printf("  missing     : err='%s'\n", err2.c_str());
        }
    }

    printf("\n");
    if (g_fail == 0) { printf("RESULT: PASS -- FASTA reader verified\n"); return 0; }
    printf("RESULT: FAIL (%d)\n", g_fail);
    return 1;
}
