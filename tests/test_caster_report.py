import hashlib
import importlib.util
import os
import signal
import subprocess
import tempfile
import time
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_script(name):
    path = os.path.join(ROOT, "scripts", name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RF = load_script("rf_distance.py")
REPORT = load_script("caster_report.py")
PENDING = load_script("caster_pending.py")
SAMPLE = load_script("caster_sample_alignment.py")
SCALE = load_script("caster_scale_report.py")


class RfDistanceTest(unittest.TestCase):
    def write_trees(self, directory, *newicks):
        paths = []
        for i, newick in enumerate(newicks):
            path = os.path.join(directory, f"tree{i}.tre")
            with open(path, "w") as handle:
                handle.write(newick + "\n")
            paths.append(path)
        return paths

    def test_equivalent_unrooted_representations_match(self):
        # the three audited representations share the single nontrivial
        # split {B,C}|{A,D}; RF must be 0 between every pair
        with tempfile.TemporaryDirectory() as directory:
            a, b, c = self.write_trees(
                directory,
                "(D,(A,(B,C)));",
                "(A,(D,(B,C)));",
                "((A,D),(B,C));",
            )
            for first, second in ((a, b), (a, c), (b, c)):
                distance, normalized, shared = RF.rf_distance(first, second)
                self.assertEqual(distance, 0)
                self.assertEqual(normalized, 0)
                self.assertEqual(shared, 4)

    def test_rerooting_preserves_unrooted_topology(self):
        with tempfile.TemporaryDirectory() as directory:
            internal, leaf_edge = self.write_trees(
                directory,
                "((A,B),(C,D));",
                "(A,(B,(C,D)));",
            )
            distance, normalized, shared = RF.rf_distance(internal, leaf_edge)
            self.assertEqual(distance, 0)
            self.assertEqual(normalized, 0)
            self.assertEqual(shared, 4)

    def test_pruning_to_shared_taxa(self):
        with tempfile.TemporaryDirectory() as directory:
            larger, smaller = self.write_trees(
                directory,
                "(E,(D,(A,(B,C))));",
                "((A,D),(B,C));",
            )
            distance, normalized, shared = RF.rf_distance(larger, smaller)
            self.assertEqual(distance, 0)
            self.assertEqual(normalized, 0)
            self.assertEqual(shared, 4)

    def test_label_order_permutation_preserves_topology(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = self.write_trees(
                directory,
                "((A,D),(B,C));",
                "((C,B),(D,A));",
            )
            distance, normalized, shared = RF.rf_distance(first, second)
            self.assertEqual(distance, 0)
            self.assertEqual(normalized, 0)
            self.assertEqual(shared, 4)

    def test_star_against_resolved_quartet_is_one_split(self):
        with tempfile.TemporaryDirectory() as directory:
            star, resolved = self.write_trees(
                directory,
                "(A,B,C,D);",
                "((A,B),(C,D));",
            )
            distance, normalized, shared = RF.rf_distance(star, resolved)
            self.assertEqual(distance, 1)
            self.assertEqual(normalized, 1)
            self.assertEqual(shared, 4)

    def test_beast_annotations_do_not_change_topology(self):
        with tempfile.TemporaryDirectory() as directory:
            plain = os.path.join(directory, "plain.tre")
            annotated = os.path.join(directory, "annotated.tre")
            with open(plain, "w") as handle:
                handle.write("((A,B),(C,D));\n")
            with open(annotated, "w") as handle:
                handle.write(
                    "((A:1[\\[&height={1,2}\\]],B:1),(C:1,D:1));\n")
            distance, normalized, shared = RF.rf_distance(plain, annotated)
            self.assertEqual(distance, 0)
            self.assertEqual(normalized, 0)
            self.assertEqual(shared, 4)

    def test_quartet_difference_is_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            first = os.path.join(directory, "first.tre")
            second = os.path.join(directory, "second.tre")
            with open(first, "w") as handle:
                handle.write("((A,B),(C,D));\n")
            with open(second, "w") as handle:
                handle.write("((A,C),(B,D));\n")
            distance, normalized, shared = RF.rf_distance(first, second)
            self.assertEqual(distance, 2)
            self.assertEqual(normalized, 1)
            self.assertEqual(shared, 4)


class CasterReportTest(unittest.TestCase):
    def test_runtime_metadata_and_memory_are_read(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "run.meta.tsv"), "w") as handle:
                handle.write(
                    "key\tvalue\nstatus\tcomplete\nelapsed_seconds\t42\n")
            with open(os.path.join(directory, "caster.time"), "w") as handle:
                handle.write("Maximum resident set size (kbytes): 12345\n")
            row = REPORT.run_row("full", "", "genomsa", directory)
            self.assertEqual(row["status"], "complete")
            self.assertEqual(row["elapsed_seconds"], "42")
            self.assertEqual(row["max_rss_kb"], "12345")

    def test_slurm_memory_is_used_without_gnu_time(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "run.meta.tsv"), "w") as handle:
                handle.write(
                    "key\tvalue\nstatus\tcomplete\nslurm_max_rss\t2.5G\n")
            with open(os.path.join(directory, "caster.time"), "w") as handle:
                handle.write("GNU time unavailable on this node\n")
            row = REPORT.run_row("full", "", "genomsa", directory)
            self.assertEqual(row["max_rss_kb"], "2621440")

    def test_provenance_is_included(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "run.meta.tsv"), "w") as handle:
                handle.write(
                    "key\tvalue\n"
                    "status\tfailed\n"
                    "exit_code\t132\n"
                    "caster_backend\tcpu-portable\n"
                    "caster_build_profile\tportable\n"
                    "caster_aster_commit\tabc123\n"
                    "caster_bin_sha256\tabc123\n"
                    "host\tnode1\n"
                    "host_arch\tx86_64\n")
            row = REPORT.run_row("full", "", "genomsa", directory)
            self.assertEqual(row["status"], "failed")
            self.assertEqual(row["exit_code"], "132")
            self.assertEqual(row["caster_backend"], "cpu-portable")
            self.assertEqual(row["caster_build_profile"], "portable")
            self.assertEqual(row["caster_aster_commit"], "abc123")
            self.assertEqual(row["caster_bin_sha256"], "abc123")
            self.assertEqual(row["host"], "node1")
            self.assertEqual(row["host_arch"], "x86_64")

    def test_report_writes_extended_provenance_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            outdir = os.path.join(directory, "report")
            result = subprocess.run(
                [
                    "python3",
                    os.path.join(ROOT, "scripts", "caster_report.py"),
                    "--root", directory,
                    "--outdir", outdir,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(os.path.join(outdir, "caster_runs.tsv")) as handle:
                header = handle.readline().rstrip("\n").split("\t")
            self.assertIn("caster_aster_commit", header)
            self.assertIn("host_arch", header)


class CasterPendingTest(unittest.TestCase):
    def make_input(self, directory, variant="genomsa"):
        path = os.path.join(
            directory, "data", f"supermatrix_{variant}", "supermatrix.fasta")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            handle.write(">A\nAAAA\n>B\nAAAA\n>C\nCCCC\n>D\nCCCC\n")
        return path

    def write_manifested_run(self, directory, variant, input_path):
        outdir = os.path.join(
            directory, "results", "caster_backbone", "full", variant)
        os.makedirs(outdir)
        with open(os.path.join(outdir, "caster.treefile"), "w") as handle:
            handle.write("((A,B),(C,D));\n")
        with open(input_path, "rb") as handle:
            payload = handle.read()
        with open(os.path.join(outdir, "run.meta.tsv"), "w") as handle:
            handle.write(
                "key\tvalue\n"
                "status\tcomplete\n"
                f"input\t{input_path}\n"
                f"input_bytes\t{len(payload)}\n"
                f"input_sha256\t{hashlib.sha256(payload).hexdigest()}\n"
                f"caster_config_sha256\t{'0' * 64}\n")

    def test_only_verified_cells_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = self.make_input(directory)
            self.write_manifested_run(directory, "genomsa", input_path)
            self.assertEqual(
                PENDING.pending_indices(directory, "full"), [1, 2])
            # legacy result: valid tree + complete meta but no manifest
            legacy = os.path.join(
                directory, "results", "caster_backbone", "full",
                "genomsa_lf")
            os.makedirs(legacy)
            with open(os.path.join(legacy, "caster.treefile"), "w") as handle:
                handle.write("((A,B),(C,D));\n")
            with open(os.path.join(legacy, "run.meta.tsv"), "w") as handle:
                handle.write("key\tvalue\nstatus\tcomplete\n")
            self.assertEqual(
                PENDING.pending_indices(directory, "full"), [1, 2])
            # same-length input rewrite breaks the recorded hash
            with open(input_path, "w") as handle:
                handle.write(">A\nAAAA\n>B\nAAAA\n>C\nCCCC\n>D\nCCCT\n")
            self.assertEqual(
                PENDING.pending_indices(directory, "full"), [0, 1, 2])

    def test_explain_reports_cell_states(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = self.make_input(directory)
            self.write_manifested_run(directory, "genomsa", input_path)
            legacy = os.path.join(
                directory, "results", "caster_backbone", "full",
                "genomsa_lf")
            os.makedirs(legacy)
            with open(os.path.join(legacy, "caster.treefile"), "w") as handle:
                handle.write("((A,B),(C,D));\n")
            with open(os.path.join(legacy, "run.meta.tsv"), "w") as handle:
                handle.write("key\tvalue\nstatus\tcomplete\n")
            result = subprocess.run(
                [
                    "python3",
                    os.path.join(ROOT, "scripts", "caster_pending.py"),
                    "--root", directory, "--scope", "full", "--explain",
                ],
                check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = dict(
                line.split("\t") for line in result.stdout.strip().split("\n"))
            self.assertEqual(lines["0"], "verified_complete")
            self.assertEqual(lines["1"], "legacy_unverified")
            self.assertEqual(lines["2"], "missing")


class CasterSampleAlignmentTest(unittest.TestCase):
    def test_evenly_spaced_columns_are_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "source.fasta")
            output = os.path.join(directory, "sample.fasta")
            with open(source, "w") as handle:
                handle.write(">A\nABCDEFGH\n>B\nabcdefgh\n")
            SAMPLE.sample_alignment(source, output, 4)
            with open(output) as handle:
                self.assertEqual(
                    handle.read(), ">A\nACEG\n>B\naceg\n")

    def test_mismatched_sequence_lengths_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "source.fasta")
            output = os.path.join(directory, "sample.fasta")
            with open(source, "w") as handle:
                handle.write(">A\nABCDEFGH\n>B\nabc\n")
            with self.assertRaisesRegex(ValueError, "expected 8"):
                SAMPLE.sample_alignment(source, output, 4)
            self.assertFalse(os.path.exists(output))


class CasterRunTest(unittest.TestCase):
    def make_input(self, directory):
        path = os.path.join(directory, "input.fasta")
        with open(path, "w") as handle:
            handle.write(">A\nAAAA\n>B\nAAAA\n>C\nCCCC\n>D\nCCCC\n")
        return path

    def make_caster(self, directory, exit_code):
        path = os.path.join(directory, "caster")
        with open(path, "w") as handle:
            handle.write(
                "#!/bin/bash\n"
                "set -euo pipefail\n"
                "output=\n"
                "while [ \"$#\" -gt 0 ]; do\n"
                "    if [ \"$1\" = -o ]; then\n"
                "        output=$2\n"
                "        shift 2\n"
                "    else\n"
                "        shift\n"
                "    fi\n"
                "done\n"
                "if [ -n \"${CASTER_TEST_STARTED:-}\" ]; then\n"
                "    : > \"$CASTER_TEST_STARTED\"\n"
                "fi\n"
                "sleep \"${CASTER_TEST_SLEEP:-0}\"\n"
                f"if [ {exit_code} -ne 0 ]; then exit {exit_code}; fi\n"
                "printf '((A,B),(C,D));\\n' > \"$output\"\n")
        os.chmod(path, 0o755)
        return path

    def run_caster(self, directory, caster, input_path=None,
                   **environment_overrides):
        output = os.path.join(directory, "output")
        if input_path is None:
            input_path = self.make_input(directory)
        environment = os.environ.copy()
        environment.update({
            "CASTER_BIN": caster,
            "CASTER_BACKEND": "cpu-test",
            "THREADS": "2",
            "TIME_BIN": os.path.join(directory, "missing-time"),
        })
        environment.update(environment_overrides)
        result = subprocess.run(
            [
                "bash", os.path.join(ROOT, "scripts", "caster_run.sh"),
                input_path, output,
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        return result, REPORT.read_meta(os.path.join(
            output, "run.meta.tsv"))

    def test_success_records_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            caster = self.make_caster(directory, 0)
            result, meta = self.run_caster(directory, caster)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(meta["status"], "complete")
            self.assertEqual(meta["exit_code"], "0")
            self.assertEqual(meta["caster_backend"], "cpu-test")
            self.assertEqual(len(meta["caster_bin_sha256"]), 64)
            self.assertEqual(len(meta["input_sha256"]), 64)
            self.assertEqual(len(meta["caster_config_sha256"]), 64)
            self.assertRegex(meta["elapsed_seconds"], r"^\d+\.\d+$")
            self.assertEqual(meta["result_state"]
                             if "result_state" in meta else
                             REPORT.result_state(
                                 os.path.join(directory, "output")),
                             "verified_complete")

    def test_reuse_skips_when_manifest_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            caster = self.make_caster(directory, 0)
            started = os.path.join(directory, "started")
            result, _ = self.run_caster(
                directory, caster, CASTER_TEST_STARTED=started)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(os.path.exists(started))
            os.remove(started)
            result, _ = self.run_caster(
                directory, caster, CASTER_TEST_STARTED=started)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("already complete", result.stdout)
            self.assertFalse(os.path.exists(started))

    def test_manifest_mismatch_archives_and_reruns(self):
        with tempfile.TemporaryDirectory() as directory:
            caster = self.make_caster(directory, 0)
            input_path = os.path.join(directory, "input.fasta")
            result, _ = self.run_caster(directory, caster)
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(input_path, "w") as handle:
                handle.write(">A\nAAAA\n>B\nAAAT\n>C\nCCCC\n>D\nCCCC\n")
            result, meta = self.run_caster(
                directory, caster, input_path=input_path)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(meta["status"], "complete")
            stale = os.listdir(os.path.join(directory, "output"))
            self.assertTrue(
                any(name.startswith("caster.treefile.stale.")
                    for name in stale))
            self.assertTrue(
                any(name.startswith("run.meta.tsv.stale.")
                    for name in stale))

    def test_unmanifested_result_is_archived_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            caster = self.make_caster(directory, 0)
            started = os.path.join(directory, "started")
            output = os.path.join(directory, "output")
            os.makedirs(output)
            with open(os.path.join(output, "caster.treefile"), "w") as handle:
                handle.write("((A,B),(C,D));\n")
            with open(os.path.join(output, "run.meta.tsv"), "w") as handle:
                handle.write("key\tvalue\nstatus\tcomplete\n")
            result, meta = self.run_caster(
                directory, caster, CASTER_TEST_STARTED=started)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(os.path.exists(started))
            self.assertTrue(
                any(name.startswith("caster.treefile.stale.")
                    for name in os.listdir(output)))
            self.assertEqual(len(meta["caster_config_sha256"]), 64)

    def test_invalid_input_records_meta(self):
        with tempfile.TemporaryDirectory() as directory:
            caster = self.make_caster(directory, 0)
            missing = os.path.join(directory, "missing.fasta")
            result, meta = self.run_caster(
                directory, caster, input_path=missing)
            self.assertEqual(result.returncode, 66)
            self.assertEqual(meta["status"], "invalid_input")
            self.assertEqual(meta["exit_code"], "66")

    def test_failure_records_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            caster = self.make_caster(directory, 132)
            result, meta = self.run_caster(directory, caster)
            self.assertEqual(result.returncode, 132)
            self.assertEqual(meta["status"], "failed")
            self.assertEqual(meta["exit_code"], "132")
            self.assertTrue(os.path.isfile(os.path.join(
                directory, "output", "caster.log")))
            self.assertFalse(os.path.exists(os.path.join(
                directory, "output", "caster.treefile.tmp")))

    def test_termination_records_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            caster = self.make_caster(directory, 0)
            output = os.path.join(directory, "output")
            started = os.path.join(directory, "started")
            environment = os.environ.copy()
            environment.update({
                "CASTER_BIN": caster,
                "CASTER_BACKEND": "cpu-test",
                "CASTER_TEST_SLEEP": "30",
                "CASTER_TEST_STARTED": started,
                "TIME_BIN": os.path.join(directory, "missing-time"),
            })
            process = subprocess.Popen(
                [
                    "bash", os.path.join(ROOT, "scripts", "caster_run.sh"),
                    self.make_input(directory), output,
                ],
                env=environment,
                preexec_fn=os.setsid,
            )
            for _ in range(100):
                if os.path.exists(started):
                    break
                time.sleep(0.01)
            self.assertTrue(os.path.exists(started))
            os.killpg(process.pid, signal.SIGTERM)
            self.assertEqual(process.wait(timeout=5), 143)
            meta = REPORT.read_meta(os.path.join(
                output, "run.meta.tsv"))
            self.assertEqual(meta["status"], "timeout")
            self.assertEqual(meta["exit_code"], "143")

    def test_declared_backend_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            caster = self.make_caster(directory, 0)
            with open(f"{caster}.build.tsv", "w") as handle:
                handle.write(
                    "key\tvalue\n"
                    "runtime_backend\tcpu-portable\n"
                    "profile\tportable\n")
            result, _ = self.run_caster(
                directory, caster, CASTER_BACKEND="hip-amd")
            self.assertEqual(result.returncode, 65)
            self.assertIn("does not match", result.stderr)


class CasterScaleReportTest(unittest.TestCase):
    def write_run(self, directory, threads, elapsed, input_path=None):
        run = os.path.join(directory, f"t{threads}")
        os.makedirs(run)
        meta = (
            "key\tvalue\n"
            "status\tcomplete\n"
            f"elapsed_seconds\t{elapsed}\n")
        if input_path:
            with open(input_path, "rb") as handle:
                payload = handle.read()
            meta += (
                f"input\t{input_path}\n"
                f"input_bytes\t{len(payload)}\n"
                f"input_sha256\t{hashlib.sha256(payload).hexdigest()}\n"
                f"caster_config_sha256\t{'f' * 64}\n")
        with open(os.path.join(run, "run.meta.tsv"), "w") as handle:
            handle.write(meta)
        with open(os.path.join(run, "caster.treefile"), "w") as handle:
            handle.write("((A,B),(C,D));\n")

    def test_speedup_and_efficiency_are_calculated(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = os.path.join(directory, "bench.fasta")
            with open(input_path, "w") as handle:
                handle.write(">A\nAAAA\n>B\nAAAA\n>C\nCCCC\n>D\nCCCC\n")
            for threads, elapsed in ((1, 100), (2, 60), (4, 50)):
                self.write_run(directory, threads, elapsed, input_path)
            rows = SCALE.scaling_rows(directory, [1, 2, 4])
            self.assertEqual(rows[0]["result_state"], "verified_complete")
            self.assertEqual(rows[1]["speedup_vs_first"], "1.666667")
            self.assertEqual(rows[1]["parallel_efficiency"], "0.833333")
            self.assertEqual(rows[2]["parallel_efficiency"], "0.500000")
            self.assertEqual(rows[2]["rf_vs_first"], 0)
            self.assertEqual(rows[2]["topology_status"], "ok")

    def test_legacy_baseline_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as directory:
            for threads, elapsed in ((1, 100), (2, 60)):
                self.write_run(directory, threads, elapsed)
            rows = SCALE.scaling_rows(directory, [1, 2])
            self.assertEqual(rows[0]["result_state"], "legacy_unverified")
            self.assertEqual(rows[0]["speedup_vs_first"], "")
            self.assertEqual(rows[1]["speedup_vs_first"], "")


if __name__ == "__main__":
    unittest.main()
