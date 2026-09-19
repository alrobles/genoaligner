#!/usr/bin/env python3
"""Summarize CASTER CPU thread-scaling runs."""
import argparse
import csv
import importlib.util
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location(
    "caster_report", os.path.join(SCRIPT_DIR, "caster_report.py"))
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def scaling_rows(out_root, thread_counts):
    runs = []
    baseline = None
    baseline_threads = thread_counts[0]
    baseline_tree = os.path.join(
        out_root, f"t{baseline_threads}", "caster.treefile")
    for threads in thread_counts:
        run_directory = os.path.join(out_root, f"t{threads}")
        run = REPORT.run_row(
            "benchmark", "", f"t{threads}", run_directory)
        elapsed = run["elapsed_seconds"]
        elapsed_value = float(elapsed) if elapsed else None
        if (threads == thread_counts[0]
                and run["result_state"] == "verified_complete"):
            baseline = elapsed_value
        speedup = baseline / elapsed_value if baseline and elapsed_value else None
        topology = REPORT.compare(
            "benchmark", "", f"t{threads}", "first",
            os.path.join(run_directory, "caster.treefile"), baseline_tree)
        runs.append({
            "threads": threads,
            "status": run["status"],
            "result_state": run["result_state"],
            "elapsed_seconds": elapsed,
            "speedup_vs_first": f"{speedup:.6f}" if speedup else "",
            "parallel_efficiency": (
                f"{speedup / (threads / baseline_threads):.6f}"
                if speedup else ""),
            "max_rss_kb": run["max_rss_kb"],
            "host": run["host"],
            "caster_bin_sha256": run["caster_bin_sha256"],
            "caster_config_sha256": run["caster_config_sha256"],
            "shared_taxa_vs_first": topology["shared_taxa"],
            "rf_vs_first": topology["rf"],
            "rf_norm_vs_first": topology["rf_norm"],
            "topology_status": topology["status"],
        })
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--threads", default="1,2,4,8,16,32",
        help="comma-separated thread counts")
    parser.add_argument("--output")
    args = parser.parse_args()
    thread_counts = [int(value) for value in args.threads.split(",")]
    output = args.output or os.path.join(args.out_root, "scaling.tsv")
    rows = scaling_rows(args.out_root, thread_counts)
    with open(output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
