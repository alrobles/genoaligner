import importlib.util
import os
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


if __name__ == "__main__":
    unittest.main()
