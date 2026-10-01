#!/usr/bin/env python3
"""Compare genomsa --codon alignments against MACSE raw nt alignments.

Metric: aligned amino-acid pair agreement. For a pair of sequences (i,j),
each alignment induces a set of aligned codon pairs (columns where both
rows have a non-gap, translatable codon). We report the fraction of
MACSE-aligned codon pairs that genomsa also aligns (MACSE as
pseudo-reference, like an SP score), plus the symmetric recall.

'!' (MACSE frameshift marks) and codons containing non-ACGT are skipped.
Pairs are sampled with a fixed seed when nseq is large.

Usage: codon_vs_macse.py <gene> <genomsa_fasta> <macse_fasta> [gc_def]
"""
import random
import sys

STD = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','TCT':'S','TCC':'S','TCA':'S','TCG':'S',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','TGT':'C','TGC':'C','TGA':'*','TGG':'W',
    'CTT':'L','CTC':'L','CTA':'L','CTG':'L','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'CAT':'H','CAC':'H','CAA':'Q','CAG':'Q','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','ACT':'T','ACC':'T','ACA':'T','ACG':'T',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','AGT':'S','AGC':'S','AGA':'R','AGG':'R',
    'GTT':'V','GTC':'V','GTA':'V','GTG':'V','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'GAT':'D','GAC':'D','GAA':'E','GAG':'E','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}
MT = dict(STD, TGA='W', ATA='M', AGA='*', AGG='*')


def read_fa(path):
    recs, name, buf = {}, None, []
    for ln in open(path):
        ln = ln.strip()
        if ln.startswith('>'):
            if name is not None:
                recs[name] = ''.join(buf)
            name, buf = ln[1:].split()[0], []
        else:
            buf.append(ln)
    if name is not None:
        recs[name] = ''.join(buf)
    return recs


def pair_alignment(rec_i, rec_j, code):
    """Return set of residue-index pairs (i_res, j_res) co-aligned."""
    s = set()
    i = j = 0
    for k in range(0, len(rec_i) - 2, 3):
        ci3, cj3 = rec_i[k:k+3], rec_j[k:k+3]
        res_i, res_j = ci3 != '---', cj3 != '---'
        if (res_i and res_j and '!' not in ci3 and '!' not in cj3
                and ci3.upper() in code and cj3.upper() in code):
            s.add((i, j))
        i += res_i
        j += res_j
    return s


def main():
    gene, g_fa, m_fa = sys.argv[1], sys.argv[2], sys.argv[3]
    gc = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    code = MT if gc == 2 else STD
    G, M = read_fa(g_fa), read_fa(m_fa)
    ids = [i for i in G if i in M]
    print(f"{gene}: genomsa n={len(G)} macse n={len(M)} shared={len(ids)}")
    if len(ids) < 2:
        return
    rng = random.Random(7)
    pairs = [(a, b) for x, a in enumerate(ids) for b in ids[x+1:]]
    if len(pairs) > 40000:
        pairs = rng.sample(pairs, 40000)
    agree = ref_tot = our_tot = 0
    for a, b in pairs:
        sp = pair_alignment(M[a], M[b], code)
        gp = pair_alignment(G[a], G[b], code)
        agree += len(sp & gp)
        ref_tot += len(sp)
        our_tot += len(gp)
    rec = agree / ref_tot if ref_tot else 0
    prec = agree / our_tot if our_tot else 0
    print(f"{gene}: pairs={len(pairs)} aligned_pairs(macse)={ref_tot} "
          f"recall={rec:.4f} precision={prec:.4f} "
          f"aln_len genomsa={len(next(iter(G.values())))} "
          f"macse={len(next(iter(M.values())))}")


if __name__ == '__main__':
    main()
