// genoaligner — CPU reference for edit distance (validation oracle).
//
// Deliberately a DIFFERENT algorithm from the GPU kernel: this is the classic
// O(nm) dynamic program, not a wavefront. Validating a wavefront computation
// against a second wavefront implementation would be circular and would hide
// exactly the class of bug we care about (mis-indexed diagonal shifts).
//
// Unit costs only, matching UnitCosts in the kernel. This is a Levenshtein
// distance: substitution = 1, insertion = 1, deletion = 1.

#include <string>
#include <vector>
#include <algorithm>
#include <climits>

namespace genoaligner {

// Classic Levenshtein via rolling row. Returns the edit distance.
inline int edit_distance_cpu(const char* pattern, int m,
                             const char* text,    int n)
{
    if (m == 0) return n;
    if (n == 0) return m;

    std::vector<int> prev(n + 1), cur(n + 1);
    for (int j = 0; j <= n; ++j) prev[j] = j;

    for (int i = 1; i <= m; ++i) {
        cur[0] = i;
        for (int j = 1; j <= n; ++j) {
            const int sub = prev[j - 1] + (pattern[i - 1] == text[j - 1] ? 0 : 1);
            const int del = prev[j] + 1;
            const int ins = cur[j - 1] + 1;
            cur[j] = std::min({sub, del, ins});
        }
        std::swap(prev, cur);
    }
    return prev[n];
}

} // namespace genoaligner
