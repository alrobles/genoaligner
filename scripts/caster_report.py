#!/usr/bin/env python3
"""Consolidate CASTER runtime and RF results for full and mini backbones."""
import argparse
import csv
import glob
import importlib.util
import os
import re
import statistics


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location(
    "rf_distance", os.path.join(SCRIPT_DIR, "rf_distance.py"))
RF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RF)


def compare(scope, rep, variant, label, tree_a, tree_b):
    row = {
        "scope": scope,
        "rep": rep,
        "variant": variant,
        "comparison": label,
        "tree_a": tree_a,
        "tree_b": tree_b,
        "shared_taxa": "",
        "rf": "",
        "rf_norm": "",
        "status": "ok",
    }
    if not os.path.isfile(tree_a):
        row["status"] = "missing_caster"
        return row
    if not os.path.isfile(tree_b):
        row["status"] = "missing_reference"
        return row
    distance, normalized, shared = RF.rf_distance(tree_a, tree_b)
    row["shared_taxa"] = shared
    row["rf"] = distance
    row["rf_norm"] = f"{normalized:.6f}"
    if shared < 4:
        row["status"] = "insufficient_shared_taxa"
    return row


def read_meta(path):
    if not os.path.isfile(path):
        return {}
    values = {}
    with open(path, newline="") as handle:
        rows = csv.reader(handle, delimiter="\t")
        next(rows, None)
        for row in rows:
            if len(row) == 2:
                values[row[0]] = row[1]
    return values


def memory_to_kb(value):
    match = re.fullmatch(r"([0-9.]+)([KMGT]?)", value.strip())
    if not match:
        return ""
    number = float(match.group(1))
    multiplier = {
        "": 1,
        "K": 1,
        "M": 1024,
        "G": 1024 ** 2,
        "T": 1024 ** 3,
    }[match.group(2)]
    return str(round(number * multiplier))


def max_rss_kb(path, fallback=""):
    if not os.path.isfile(path):
        return memory_to_kb(fallback)
    with open(path) as handle:
        for line in handle:
            if "Maximum resident set size (kbytes):" in line:
                return line.rsplit(":", 1)[1].strip()
    return memory_to_kb(fallback)


def run_row(scope, rep, variant, outdir):
    meta = read_meta(os.path.join(outdir, "run.meta.tsv"))
    return {
        "scope": scope,
        "rep": rep,
        "variant": variant,
        "status": meta.get("status", "missing"),
        "exit_code": meta.get("exit_code", ""),
        "input": meta.get("input", ""),
        "input_bytes": meta.get("input_bytes", ""),
        "caster_bin": meta.get("caster_bin", ""),
        "caster_bin_sha256": meta.get("caster_bin_sha256", ""),
        "caster_backend": meta.get("caster_backend", ""),
        "caster_build_profile": meta.get("caster_build_profile", ""),
        "caster_aster_commit": meta.get("caster_aster_commit", ""),
        "caster_compiler": meta.get("caster_compiler", ""),
        "caster_flags": meta.get("caster_flags", ""),
        "host": meta.get("host", ""),
        "host_arch": meta.get("host_arch", ""),
        "threads": meta.get("threads", ""),
        "seed": meta.get("seed", ""),
        "chunk": meta.get("chunk", ""),
        "elapsed_seconds": meta.get("elapsed_seconds", ""),
        "max_rss_kb": max_rss_kb(
            os.path.join(outdir, "caster.time"),
            meta.get("slurm_max_rss", "")),
        "slurm_job_id": meta.get("slurm_job_id", ""),
        "slurm_array_task_id": meta.get("slurm_array_task_id", ""),
    }


def write_tsv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    groups = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        key = (row["scope"], row["variant"], row["comparison"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (scope, variant, comparison), values in sorted(groups.items()):
        normalized = [float(row["rf_norm"]) for row in values]
        shared = [int(row["shared_taxa"]) for row in values]
        summary.append({
            "scope": scope,
            "variant": variant,
            "comparison": comparison,
            "n": len(values),
            "rf_norm_mean": f"{statistics.mean(normalized):.6f}",
            "rf_norm_min": f"{min(normalized):.6f}",
            "rf_norm_max": f"{max(normalized):.6f}",
            "shared_taxa_min": min(shared),
            "shared_taxa_max": max(shared),
        })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="/beegfs/a474r867/phylogenyAI")
    parser.add_argument("--outdir")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    outdir = args.outdir or os.path.join(
        root, "results", "caster_backbone", "report")
    upham = os.path.join(
        root, "data", "upham_ref", "upham_pruned_4353.tre")

    comparisons = []
    runs = []
    full = {
        "genomsa": ["supermatrix_tree_genomsa/backbone.treefile"],
        "genomsa_lf": ["supermatrix_tree_genomsa_lf/backbone.treefile"],
        "codon": ["supermatrix_tree_codon/backbone.treefile"],
    }
    for variant, relative_iqtrees in full.items():
        caster_dir = os.path.join(
            root, "results", "caster_backbone", "full", variant)
        caster_tree = os.path.join(caster_dir, "caster.treefile")
        runs.append(run_row("full", "", variant, caster_dir))
        comparisons.append(compare(
            "full", "", variant, "upham", caster_tree, upham))
        iqtrees = [
            os.path.join(root, "results", path)
            for path in relative_iqtrees
        ]
        if variant == "genomsa_lf":
            iqtrees.extend(sorted(glob.glob(os.path.join(
                root, "results", "supermatrix_tree_genomsa_lf",
                "backbone_s*.treefile"))))
        seen = set()
        for iqtree in iqtrees:
            if iqtree in seen:
                continue
            seen.add(iqtree)
            label = "iqtree:" + os.path.basename(iqtree)
            comparisons.append(compare(
                "full", "", variant, label, caster_tree, iqtree))

    for rep in range(5):
        for variant in ("macse", "base", "lf"):
            caster_dir = os.path.join(
                root, "results", "caster_backbone", "mini_v2",
                f"rep{rep}", variant)
            caster_tree = os.path.join(caster_dir, "caster.treefile")
            iqtree = os.path.join(
                root, "results", "mini_backbone_v2", f"rep{rep}",
                variant, "tree", "backbone.treefile")
            runs.append(run_row("mini_v2", f"rep{rep}", variant, caster_dir))
            comparisons.append(compare(
                "mini_v2", f"rep{rep}", variant, "upham",
                caster_tree, upham))
            comparisons.append(compare(
                "mini_v2", f"rep{rep}", variant, "iqtree",
                caster_tree, iqtree))

    comparison_fields = [
        "scope", "rep", "variant", "comparison", "shared_taxa", "rf",
        "rf_norm", "status", "tree_a", "tree_b",
    ]
    run_fields = [
        "scope", "rep", "variant", "status", "exit_code", "input",
        "input_bytes", "caster_bin", "caster_bin_sha256",
        "caster_backend", "caster_build_profile", "caster_aster_commit",
        "caster_compiler", "caster_flags", "host", "host_arch", "threads",
        "seed", "chunk", "elapsed_seconds", "max_rss_kb", "slurm_job_id",
        "slurm_array_task_id",
    ]
    summary_fields = [
        "scope", "variant", "comparison", "n", "rf_norm_mean",
        "rf_norm_min", "rf_norm_max", "shared_taxa_min",
        "shared_taxa_max",
    ]
    write_tsv(
        os.path.join(outdir, "caster_rf.tsv"),
        comparisons, comparison_fields)
    write_tsv(
        os.path.join(outdir, "caster_runs.tsv"), runs, run_fields)
    write_tsv(
        os.path.join(outdir, "caster_rf_summary.tsv"),
        summarize(comparisons), summary_fields)
    print(f"wrote {outdir}")


if __name__ == "__main__":
    main()
