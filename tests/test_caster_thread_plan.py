import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "caster_thread_plan.py"
)
SPEC = importlib.util.spec_from_file_location("caster_thread_plan", SCRIPT)
PLAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLAN)


class CasterThreadPlanTest(unittest.TestCase):
    def test_fasta_sites_reads_first_wrapped_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            fasta = Path(directory) / "input.fasta"
            fasta.write_text(">one\nACGT\nAC\n>two\nACGTAC\n")
            self.assertEqual(PLAN.fasta_sites(fasta), 6)

    def test_fasta_sites_rejects_non_fasta(self):
        with tempfile.TemporaryDirectory() as directory:
            fasta = Path(directory) / "input.fasta"
            fasta.write_text("ACGT\n")
            with self.assertRaisesRegex(ValueError, "not FASTA"):
                PLAN.fasta_sites(fasta)

    def test_fasta_sites_rejects_unequal_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            fasta = Path(directory) / "input.fasta"
            fasta.write_text(">one\nACGT\n>two\nACG\n")
            with self.assertRaisesRegex(ValueError, "unequal lengths"):
                PLAN.fasta_sites(fasta)

    def test_effective_chunks(self):
        cases = [(3000, 10000, 1), (30000, 10000, 3), (30001, 10000, 4)]
        for sites, chunk_size, chunks in cases:
            with self.subTest(sites=sites, chunk_size=chunk_size):
                self.assertEqual(
                    PLAN.effective_chunks(sites, chunk_size),
                    chunks,
                )

    def test_default_threads_stops_at_effective_chunks(self):
        self.assertEqual(
            PLAN.default_threads(29, 32),
            [1, 2, 4, 8, 16, 29],
        )
        self.assertEqual(PLAN.default_threads(1, 32), [1])

    def test_requested_threads_rejects_idle_workers(self):
        with self.assertRaisesRegex(
            ValueError,
            "exceeds 3 scoring chunks",
        ):
            PLAN.requested_threads("1 2 4", 3, 32)

    def test_requested_threads_preserves_order_and_removes_duplicates(self):
        self.assertEqual(
            PLAN.requested_threads("3,1,3", 3, 32),
            [3, 1],
        )


if __name__ == "__main__":
    unittest.main()
