// genoaligner — minimal FASTA reading.
//
// WHY THIS IS IN THE LIBRARY, NOT IN A TEST
// -----------------------------------------
// The public API takes (const char*, len). That is the right interface for a
// caller that already has sequences, but it means a *user* cannot get sequences in:
// there was no way to go from a .fa file to an align call without writing a parser.
// A library that cannot be fed is not installable, so the reader belongs here.
//
// SCOPE, DELIBERATELY NARROW
// --------------------------
//   - FASTA only. No FASTQ (a different format with a quality line, and mixing the
//     two silently is how quality strings end up aligned as sequence).
//   - No gzip. Decompression is a dependency and a policy choice; a caller can pipe.
//   - Sequences are returned as raw bytes, NOT validated as ACGT. Practically every
//     real FASTA contains N, ambiguity codes, lowercase soft-masking, and sometimes
//     protein. Rejecting them here would be wrong; the caller decides.
//
// WHAT IT MUST GET RIGHT (and the tests in tests/io/test_fasta.cpp pin):
//   - multi-line sequences (the common case; a parser that assumes one line per
//     record silently truncates almost every real genome)
//   - CRLF line endings (files produced on Windows, and some pipelines)
//   - blank lines between records
//   - a header line with extra description after the identifier
//   - the FIRST token of the header is the id; the rest is description
//   - an empty record (header, no sequence) is an error, not an empty sequence:
//     silently returning "" would let a broken file align as a deletion swarm

#ifndef GENOALIGNER_IO_FASTA_HPP
#define GENOALIGNER_IO_FASTA_HPP

#include <cstdio>
#include <string>
#include <vector>

namespace genoaligner {
namespace io {

struct FastaRecord {
    std::string id;           // first whitespace-delimited token of the header
    std::string description;  // the remainder of the header line (may be empty)
    std::string sequence;     // raw bytes, line breaks removed, case preserved
};

// Read every record from an already-open FILE*. Returns false on a malformed file.
// `error` receives a human-readable reason when it returns false.
//
// If `retain` is non-null, the records are appended to it; otherwise a local vector
// is used and discarded (useful for "just validate this file").
bool read_fasta(FILE* in, std::vector<FastaRecord>* out, std::string* error);

// Convenience: open a path and read it. Returns false if the file cannot be opened
// or is malformed. An empty file is an error (0 records is not a valid FASTA).
bool read_fasta_file(const std::string& path, std::vector<FastaRecord>* out,
                     std::string* error);

// --- implementation ---------------------------------------------------------

inline bool read_fasta(FILE* in, std::vector<FastaRecord>* out, std::string* error)
{
    if (!in) { if (error) *error = "null FILE*"; return false; }
    if (!out) { if (error) *error = "null output vector"; return false; }

    auto fail = [&](const std::string& why, long line) {
        if (error) {
            *error = why + " (line " + std::to_string(line) + ")";
        }
        return false;
    };

    // Read the whole stream ourselves instead of using fgets: a fixed buffer would
    // cap header length, and real headers exceed 4 KB routinely.
    std::string data;
    {
        char buf[65536];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof(buf), in)) > 0) data.append(buf, n);
        if (std::ferror(in)) return fail("read error", 0);
    }

    const size_t total = data.size();
    size_t i = 0;
    long line = 1;

    // Advance to the end of the current line, handling \n, \r\n and a lone \r.
    auto consume_eol = [&]() {
        while (i < total) {
            const char c = data[i];
            if (c == '\n') { ++i; ++line; return; }
            if (c == '\r') {
                ++i;
                if (i < total && data[i] == '\n') ++i;
                ++line;
                return;
            }
            ++i;
        }
    };

    bool   in_record = false;
    bool   have_seq  = false;   // does the current record have any sequence bytes?
    size_t record_line = 0;     // line the current header started on

    while (i < total) {
        if (!in_record) {
            // Between records: skip blank lines, expect '>'.
            const char c = data[i];
            if (c == '\n' || c == '\r' || c == ' ' || c == '\t') { consume_eol(); continue; }
            if (c != '>') {
                return fail("expected '>' at start of a record, found '" +
                            std::string(1, c) + "'", line);
            }
            ++i;   // past '>'
            std::string header;
            while (i < total && data[i] != '\n' && data[i] != '\r') header.push_back(data[i++]);
            // strip trailing whitespace from the header
            while (!header.empty() && (header.back() == ' ' || header.back() == '\t')) header.pop_back();

            FastaRecord rec;
            const size_t sp = header.find_first_of(" \t");
            if (sp == std::string::npos) {
                rec.id = header;
            } else {
                rec.id = header.substr(0, sp);
                size_t s = header.find_first_not_of(" \t", sp);
                if (s != std::string::npos) rec.description = header.substr(s);
            }
            if (rec.id.empty()) return fail("empty identifier in header", line);

            out->push_back(std::move(rec));
            in_record  = true;
            have_seq   = false;
            record_line = line;
            consume_eol();
            continue;
        }

        // Inside a record.
        const char c = data[i];
        if (c == '>') {
            // New record: the previous one must have had sequence.
            if (!have_seq) return fail("record has a header but no sequence", record_line);
            in_record = false;
            continue;
        }
        if (c == '\n' || c == '\r') { consume_eol(); continue; }

        // Sequence byte: append until end of line.
        std::string& seq = out->back().sequence;
        while (i < total && data[i] != '\n' && data[i] != '\r') seq.push_back(data[i++]);
        if (!seq.empty()) have_seq = true;
    }

    if (in_record && !have_seq) return fail("record has a header but no sequence", record_line);
    if (out->empty()) return fail("no records found", line);
    return true;
}

inline bool read_fasta_file(const std::string& path, std::vector<FastaRecord>* out,
                            std::string* error)
{
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        if (error) *error = "cannot open '" + path + "'";
        return false;
    }
    const bool ok = read_fasta(f, out, error);
    std::fclose(f);
    return ok;
}

}  // namespace io
}  // namespace genoaligner

#endif  // GENOALIGNER_IO_FASTA_HPP
