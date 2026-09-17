#!/usr/bin/env python3
"""pipeline_test — modular pre-backbone pipeline harness.

Runs the four upstream stages as swappable variants per gene, logs
per-stage walltime and QC stats to metrics.tsv inside the run dir.

Stages and variants:
  ortho    blast                    baits x blastdb -> hits.tsv
  extract  extract_gene_regions.py  records + baits -> extracted.fasta
  align    macse|genomsa|genomsa_lf|mafft   genes -> {var}.aln.fasta
  trim     clipkit <mode> [+ strip-!]       aln -> {alnvar}.{mode}.clipkit.fasta

Run layout:
  $WORK/{GENE}/sample.fasta          input for align (or extract output)
  $WORK/{GENE}/hits.tsv              ortho output
  $WORK/{GENE}/extracted.fasta       extract output
  $WORK/{GENE}/{alnvar}.aln.fasta    align output
  $WORK/{GENE}/{alnvar}.{mode}.clipkit.fasta
  $WORK/metrics.tsv

Usage:
  pipeline_test.py --work RUN --genes CYTB,ND1 \
      --stages align,trim --align-variant genomsa_lf --trim-mode smart-gap \
      --genes-src /path/pilot_200_8genes [--strip-fs]
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--work', required=True)
ap.add_argument('--genes', required=True, help='comma list')
ap.add_argument('--stages', default='align,trim')
ap.add_argument('--align-variant', default='genomsa')
ap.add_argument('--trim-mode', default='smart-gap')
ap.add_argument('--strip-fs', action='store_true',
                help='replace MACSE ! frameshift markers with - in trimmed output')
ap.add_argument('--genes-src', default='',
                help='dir holding {GENE}.fasta or {GENE}/sample.fasta')
ap.add_argument('--records-src', default='',
                help='dir holding {GENE}.records.fasta (extract input)')
ap.add_argument('--baits-src', default='')
ap.add_argument('--blastdb', default='')
ap.add_argument('--genomsa', default='genomsa')
ap.add_argument('--genomsa-args', default='')
ap.add_argument('--macse', default='macse')
ap.add_argument('--extract-script', default='')
ap.add_argument('--timeout', type=int, default=5400)
a = ap.parse_args()

os.makedirs(a.work, exist_ok=True)
M = open(os.path.join(a.work, 'metrics.tsv'), 'a')
if os.stat(os.path.join(a.work, 'metrics.tsv')).st_size == 0:
    M.write('gene\tstage\tvariant\tkey\tvalue\tseconds\texit\n')


def emit(gene, stage, variant, key, value, secs, rc):
    M.write(f'{gene}\t{stage}\t{variant}\t{key}\t{value}\t{secs:.1f}\t{rc}\n')
    M.flush()


def fasta_stats(path):
    n = width = 0
    gaps = 0
    try:
        for line in open(path):
            if line.startswith('>'):
                n += 1
            else:
                width += len(line.strip())
                gaps += line.count('-') + line.count('!')
    except FileNotFoundError:
        return 0, 0, 0.0
    return n, width, (gaps / width if width else 0.0)


def run(cmd, log, timeout):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout)
        rc, secs = p.returncode, time.time() - t0
        log.write(f"$ {' '.join(map(str, cmd))}\n# {secs:.1f}s rc={rc}\n"
                  f"{p.stderr[-1500:]}\n")
        return secs, rc
    except subprocess.TimeoutExpired:
        secs = time.time() - t0
        log.write(f"$ {' '.join(map(str, cmd))}\n# TIMEOUT {secs:.0f}s\n")
        return secs, -9


def find_input(gene):
    """Locate the per-gene fasta in genes-src (flat or pilot layout)."""
    for c in (os.path.join(a.genes_src, f'{gene}.fasta'),
              os.path.join(a.genes_src, gene, 'sample.fasta'),
              os.path.join(a.genes_src, gene, f'{gene}.fasta')):
        if os.path.exists(c):
            return c
    return ''


def stage_ortho(gene, wd, log):
    if not a.blastdb or not os.path.exists(a.blastdb + '.nal'):
        emit(gene, 'ortho', 'blast', 'skipped', 'no-blastdb', 0, 0)
        return
    bait = os.path.join(a.baits_src, f'{gene}.baits.fasta')
    out = os.path.join(wd, 'hits.tsv')
    if not os.path.exists(bait):
        emit(gene, 'ortho', 'blast', 'skipped', 'no-baits', 0, 0)
        return
    secs, rc = run(['blastn', '-db', a.blastdb, '-query', bait,
                    '-out', out, '-outfmt', '6',
                    '-num_threads', '4', '-max_target_seqs', '500',
                    '-evalue', '1e-20'], log, a.timeout)
    nh = sum(1 for _ in open(out)) if os.path.exists(out) else 0
    recs = {l.split('\t')[1] for l in open(out)} if nh else set()
    emit(gene, 'ortho', 'blast', 'hits', nh, secs, rc)
    emit(gene, 'ortho', 'blast', 'records', len(recs), 0, rc)


def stage_extract(gene, wd, log):
    rec = os.path.join(a.records_src, f'{gene}.records.fasta')
    bait = os.path.join(a.baits_src, f'{gene}.baits.fasta')
    out = os.path.join(wd, 'extracted.fasta')
    rep = os.path.join(wd, 'extract.tsv')
    if not os.path.exists(rec):
        emit(gene, 'extract', 'extract', 'skipped', 'no-records', 0, 0)
        return
    secs, rc = run([sys.executable, a.extract_script, '--gene', gene,
                    '--fasta', rec, '--out', out, '--bait', bait,
                    '--report', rep], log, a.timeout)
    n, w, _ = fasta_stats(out)
    emit(gene, 'extract', 'extract', 'seqs', n, secs, rc)


def stage_align(gene, wd, log):
    src = os.path.join(wd, 'extracted.fasta')
    if not os.path.exists(src):
        src = find_input(gene)
        if not src:
            emit(gene, 'align', a.align_variant, 'skipped', 'no-input',
                 0, 0)
            return
    out = os.path.join(wd, f'{a.align_variant}.aln.fasta')
    if a.align_variant == 'mafft':
        # mafft writes the alignment to stdout
        t0 = time.time()
        try:
            p = subprocess.run(['mafft', '--auto', '--thread', '4', src],
                               capture_output=True, text=True,
                               timeout=a.timeout)
            secs, rc = time.time() - t0, p.returncode
            if rc == 0:
                open(out, 'w').write(p.stdout)
            log.write(f"$ mafft --auto {src}\n# {secs:.1f}s rc={rc}\n"
                      f"{p.stderr[-1500:]}\n")
        except subprocess.TimeoutExpired:
            secs, rc = time.time() - t0, -9
    else:
        if a.align_variant == 'macse':
            cmd = [a.macse, '-prog', 'alignSequences', '-seq', src,
                   '-out_NT', out]
        elif a.align_variant.startswith('genomsa'):
            cmd = [a.genomsa, src, out, '--codon', '--gc-def', '1', '--cpu']
            if a.align_variant == 'genomsa_lf':
                cmd.append('--local-frame')
            cmd += a.genomsa_args.split()
        else:
            emit(gene, 'align', a.align_variant, 'skipped', 'unknown',
                 0, 1)
            return
        secs, rc = run(cmd, log, a.timeout)
    n, w, gapf = fasta_stats(out)
    emit(gene, 'align', a.align_variant, 'seqs', n, secs, rc)
    emit(gene, 'align', a.align_variant, 'width',
         (w // n) if n else 0, 0, rc)
    emit(gene, 'align', a.align_variant, 'gap_frac', f'{gapf:.3f}', 0, rc)


def stage_trim(gene, wd, log):
    aln = os.path.join(wd, f'{a.align_variant}.aln.fasta')
    tag = f'{a.align_variant}.{a.trim_mode}'
    out = os.path.join(wd, f'{tag}.clipkit.fasta')
    if not os.path.exists(aln):
        emit(gene, 'trim', tag, 'skipped', 'no-aln', 0, 0)
        return
    secs, rc = run(['clipkit', aln, '-m', a.trim_mode, '-o', out],
                   log, a.timeout)
    if rc == 0 and a.strip_fs and os.path.exists(out):
        s = open(out).read().replace('!', '-')
        open(out, 'w').write(s)
    n, w, gapf = fasta_stats(out)
    emit(gene, 'trim', tag, 'seqs', n, secs, rc)
    emit(gene, 'trim', tag, 'width', (w // n) if n else 0, 0, rc)
    emit(gene, 'trim', tag, 'gap_frac', f'{gapf:.3f}', 0, rc)


def main():
    stages = set(a.stages.split(','))
    for gene in a.genes.split(','):
        wd = os.path.join(a.work, gene)
        os.makedirs(wd, exist_ok=True)
        log = open(os.path.join(wd, 'pipeline.log'), 'a')
        if 'ortho' in stages:
            stage_ortho(gene, wd, log)
        if 'extract' in stages:
            stage_extract(gene, wd, log)
        if 'align' in stages:
            stage_align(gene, wd, log)
        if 'trim' in stages:
            stage_trim(gene, wd, log)
        log.close()
    print('pipeline_test done ->', os.path.join(a.work, 'metrics.tsv'))


main()
