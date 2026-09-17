import importlib.util
import os
import subprocess
import tempfile
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
    def test_only_incomplete_cells_are_returned(self):
        with tempfile.TemporaryDirectory() as directory:
            complete = os.path.join(
                directory, "results", "caster_backbone", "full",
                "genomsa")
            os.makedirs(complete)
            with open(os.path.join(complete, "caster.treefile"), "w") as handle:
                handle.write("((A,B),(C,D));\n")
            self.assertEqual(
                PENDING.pending_indices(directory, "full"), [1, 2])


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
                f"if [ {exit_code} -ne 0 ]; then exit {exit_code}; fi\n"
                "printf '((A,B),(C,D));\\n' > \"$output\"\n")
        os.chmod(path, 0o755)
        return path

    def run_caster(self, directory, caster, **environment_overrides):
        output = os.path.join(directory, "output")
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
                self.make_input(directory), output,
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
    def test_speedup_and_efficiency_are_calculated(self):
        with tempfile.TemporaryDirectory() as directory:
            for threads, elapsed in ((1, 100), (2, 60), (4, 50)):
                run = os.path.join(directory, f"t{threads}")
                os.makedirs(run)
                with open(os.path.join(run, "run.meta.tsv"), "w") as handle:
                    handle.write(
                        "key\tvalue\n"
                        "status\tcomplete\n"
                        f"elapsed_seconds\t{elapsed}\n")
                with open(
                        os.path.join(run, "caster.treefile"), "w") as handle:
                    handle.write("((A,B),(C,D));\n")
            rows = SCALE.scaling_rows(directory, [1, 2, 4])
            self.assertEqual(rows[1]["speedup_vs_first"], "1.666667")
            self.assertEqual(rows[1]["parallel_efficiency"], "0.833333")
            self.assertEqual(rows[2]["parallel_efficiency"], "0.500000")
            self.assertEqual(rows[2]["rf_vs_first"], 0)
            self.assertEqual(rows[2]["topology_status"], "ok")


if __name__ == "__main__":
    unittest.main()
