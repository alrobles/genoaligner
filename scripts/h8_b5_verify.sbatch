#!/bin/bash -l
#SBATCH --job-name=h8_b5v
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h8b5v_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h8b5v_%j.err

# B5 final verification — an EXTERNAL consumer, built only against the install
# prefix, with no knowledge of the source tree.
#
# This is the last link the earlier runs left unverified:
#   - the README's install commands work from a CLEAN CLONE (done: rc=0 on all three);
#   - a program that includes <genoaligner/api.hpp> and links -lgenoaligner from the
#     INSTALLED prefix compiles (done: rc=0);
#   - and now: that program RUNS and produces correct results on a device.
#
# The consumer is written here, not taken from the repo, so it cannot accidentally
# pick up an in-tree header.
set -o pipefail

P=/beegfs/a474r867/pfx_b5
LOGD=/beegfs/a474r867/genoaligner/logs

echo "=== host: $(hostname) ==="
echo "=== H8-B5v: external consumer against the install ==="
echo

if [ ! -f "$P/lib/libgenoaligner.a" ]; then
    echo "no install at $P -- run the install first"; exit 1
fi
echo "--- installed prefix ---"
find "$P" -type f | sort

cat > /tmp/ext_consumer.cpp <<'EOF'
// Written by the job, not by the repo: it must resolve entirely from the install.
#include <genoaligner/api.hpp>
#include <cstdio>
#include <vector>
int main(){
  // 1. the documented example pair
  genoaligner::AlignRequest r;
  r.pattern="ACGTACGT"; r.pattern_len=8;
  r.text="ACGTTCGT";    r.text_len=8;
  r.smax=32; r.with_cigar=true;

  genoaligner::BatchResult b = genoaligner::align_batch({r});
  if(!b.ok()){ std::printf("FAILED: %s\n", b.error?b.error:"?"); return 1; }
  std::printf("pair: score=%d cigar=%s\n", b.results[0].score, b.results[0].cigar.c_str());
  if (b.results[0].score != 1) { std::printf("WRONG SCORE\n"); return 1; }

  // 2. batch, and the documented minimum-smax behaviour
  std::vector<genoaligner::AlignRequest> reqs(2);
  reqs[0].pattern="ACGTACGTACGT"; reqs[0].pattern_len=12;
  reqs[0].text   ="ACGTACGTACGT"; reqs[0].text_len   =12; reqs[0].smax=200;
  reqs[1].pattern="ACGTACGTACGT"; reqs[1].pattern_len=12;
  reqs[1].text   ="ACGTACGTACGT"; reqs[1].text_len   =12; reqs[1].smax=8;
  genoaligner::BatchResult b2 = genoaligner::align_batch(reqs);
  std::printf("batch: resolved=%d unresolved=%d\n", b2.resolved_count, b2.unresolved_count);
  if (b2.resolved_count != 2) { std::printf("WRONG RESOLVED COUNT\n"); return 1; }

  // 3. rejected input, as documented
  genoaligner::AlignRequest bad = reqs[0];
  bad.smax = 9999;
  genoaligner::BatchResult b3 = genoaligner::align_batch({bad});
  std::printf("smax=9999 -> ok=%d error=%s\n", (int)b3.ok(), b3.error?b3.error:"(null)");
  if (b3.ok()) { std::printf("SHOULD HAVE BEEN REJECTED\n"); return 1; }

  std::printf("\nRESULT: PASS -- external consumer works against the install\n");
  return 0;
}
EOF

/kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -std=c++17 -I"$P/include" \
    -o /tmp/ext_consumer /tmp/ext_consumer.cpp -L"$P/lib" -lgenoaligner
RC=$?
echo "BUILD_RC=$RC"
[ "$RC" -ne 0 ] && { echo "H8-B5v: BUILD FAILED"; exit 1; }

echo
echo "--- run ---"
/tmp/ext_consumer | tee "$LOGD/h8b5v_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}

echo
echo "=== H8-B5v RESULT ==="
if [ "$RC" -eq 0 ]; then echo "H8-B5v: PASS"; else echo "H8-B5v: FAIL (rc=$RC)"; fi
exit "$RC"
