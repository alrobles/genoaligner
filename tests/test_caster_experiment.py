import json
import os
import random
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import caster_experiment as EXP  # noqa: E402


def tip_depths(nwk):
    """Return {tip: root-to-tip distance} for a parsed Newick tree."""
    nodes, root = EXP.parse_newick(nwk)
    out = {}

    def walk(nd, acc):
        acc += nodes[nd]["length"]
        if not nodes[nd]["children"]:
            out[nodes[nd]["name"]] = acc
            return
        for child in nodes[nd]["children"]:
            walk(child, acc)

    walk(root, 0.0)
    return out


class TreeTest(unittest.TestCase):
    def test_balanced_is_ultrametric(self):
        for n in (4, 5, 7, 128):
            depths = tip_depths(EXP.balanced_tree(n, 0.2))
            self.assertEqual(len(depths), n)
            for tip, depth in depths.items():
                self.assertTrue(tip.startswith("s"))
                self.assertAlmostEqual(depth, 0.2, places=6)

    def test_pectinate_is_ultrametric_caterpillar(self):
        nwk = EXP.pectinate_tree(6, 0.2)
        depths = tip_depths(nwk)
        self.assertEqual(len(depths), 6)
        for depth in depths.values():
            self.assertAlmostEqual(depth, 0.2, places=6)
        nodes, _ = EXP.parse_newick(nwk)
        internal = [nd for nd in nodes if nd["children"]]
        self.assertEqual(len(internal), 5)
        # every internal node has exactly one leaf child -> caterpillar
        leaf_children = [
            sum(1 for c in nd["children"] if not nodes[c]["children"])
            for nd in internal]
        self.assertEqual(sorted(leaf_children), [1] * 4 + [2])

    def test_yule_deterministic_and_ultrametric(self):
        a = EXP.yule_tree(32, 0.2, random.Random(7))
        b = EXP.yule_tree(32, 0.2, random.Random(7))
        self.assertEqual(a, b)
        depths = tip_depths(a)
        self.assertEqual(len(depths), 32)
        for depth in depths.values():
            self.assertAlmostEqual(depth, 0.2, places=6)
        c = EXP.yule_tree(32, 0.2, random.Random(8))
        self.assertNotEqual(a, c)


class SeedTest(unittest.TestCase):
    def test_sub_seed_is_deterministic_and_role_specific(self):
        self.assertEqual(EXP.sub_seed(11001, "topology"),
                         EXP.sub_seed(11001, "topology"))
        self.assertNotEqual(EXP.sub_seed(11001, "topology"),
                            EXP.sub_seed(11001, "mask"))
        self.assertNotEqual(EXP.sub_seed(11001, "topology"),
                            EXP.sub_seed(11002, "topology"))
        for role in ("topology", "evolution", "mask"):
            seed = EXP.sub_seed(11001, role)
            self.assertTrue(1 <= seed < 2 ** 31)

    def test_dataset_id_stable_and_spec_sensitive(self):
        spec = {"n": 8, "L": 100,
                "tree": {"shape": "balanced", "height": 0.2},
                "mask": {"kind": "none"}, "model": "JC",
                "generator_kind": "internal-jc", "generator_version": "",
                "data_seed": 11001}
        a = EXP.dataset_id(spec)
        self.assertEqual(a, EXP.dataset_id(dict(spec)))
        changed = dict(spec, L=200)
        self.assertNotEqual(a, EXP.dataset_id(changed))
        changed = dict(spec, data_seed=11002)
        self.assertNotEqual(a, EXP.dataset_id(changed))


class MaskTest(unittest.TestCase):
    def seqs(self):
        return {"a": "ACGTACGT", "b": "ACGTACGT", "c": "ACGTACGT"}

    def test_independent_mask_deterministic(self):
        spec = {"kind": "independent", "p": 0.5}
        m1, r1 = EXP.apply_mask(self.seqs(), spec, random.Random(3))
        m2, r2 = EXP.apply_mask(self.seqs(), spec, random.Random(3))
        self.assertEqual(m1, m2)
        self.assertEqual(r1, r2)
        self.assertTrue(0.0 < r1["effective_absence"] < 1.0)

    def test_locus_mask_removes_whole_blocks(self):
        class SeqRng:
            def __init__(self, values):
                self.values = iter(values)

            def random(self):
                return next(self.values)

        # 3 taxa x 2 blocks; mask only each taxon's second block
        rng = SeqRng([0.9, 0.1, 0.9, 0.9, 0.1, 0.9])
        spec = {"kind": "locus", "p": 0.5, "blocks": 2}
        masked, record = EXP.apply_mask(self.seqs(), spec, rng)
        self.assertEqual(masked["a"], "ACGT----")
        self.assertEqual(masked["b"], "ACGTACGT")
        self.assertEqual(masked["c"], "----ACGT")
        self.assertAlmostEqual(record["effective_absence"], 8 / 24)
        self.assertEqual(record["blocks"], 2)
        self.assertEqual(record["taxon_present_min"], 4)

    def test_none_mask_reports_existing_absence(self):
        seqs = {"a": "ACGT", "b": "ACNT"}
        masked, record = EXP.apply_mask(seqs, {"kind": "none"},
                                        random.Random(1))
        self.assertEqual(masked, seqs)
        self.assertAlmostEqual(record["effective_absence"], 1 / 8)


class ObservedTest(unittest.TestCase):
    def test_dimensions(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.fasta")
            with open(path, "w") as handle:
                handle.write(">a\nACGTNNNN\n>b\nACGT----\n")
            obs = EXP.observed_dimensions(path, 4)
            self.assertEqual(obs["n_taxa"], 2)
            self.assertEqual(obs["alignment_length"], 8)
            self.assertEqual(obs["sites_nonempty"], 4)
            self.assertEqual(obs["chunks_total"], 2)
            self.assertEqual(obs["chunks_nonempty"], 1)


class ManifestTest(unittest.TestCase):
    def base_manifest(self):
        return {
            "schema": EXP.SCHEMA, "protocol": "t", "stage": "exploration",
            "generator": {"kind": "internal-jc", "model": "JC"},
            "seeds": {"data": [11001], "search": 233, "order": 31001,
                      "bootstrap": 41001},
            "defaults": {"chunk": 10000},
            "cells": [{"case_id": "c1", "n": 8, "L": 120,
                       "tree": {"shape": "balanced", "height": 0.2},
                       "mask": {"kind": "none"}, "chunk": 10000,
                       "budget_seconds": 60, "regime": "n_lt_100"}],
            "variants": {"v1": {"caster_bin": "/x", "backend": "cpu-test",
                               "threads": 1, "extra_env": {}}},
            "baseline_variant": "v1", "repetitions": 1,
            "budgets": {"search_seconds": 60},
        }

    def write_manifest(self, directory, manifest=None):
        path = os.path.join(directory, "exp.json")
        with open(path, "w") as handle:
            json.dump(manifest or self.base_manifest(), handle)
        return path

    def test_rejects_missing_keys(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = self.base_manifest()
            del manifest["seeds"]
            with self.assertRaises(ValueError):
                EXP.load_manifest(self.write_manifest(d, manifest))

    def test_rejects_unknown_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = self.base_manifest()
            manifest["baseline_variant"] = "nope"
            with self.assertRaises(ValueError):
                EXP.load_manifest(self.write_manifest(d, manifest))

    def test_e1_panel_has_11_cells(self):
        manifest = EXP.emit_manifest("e1", "/iqtree2", "/caster")
        self.assertEqual(len(manifest["cells"]), 11)
        ids = [c["case_id"] for c in manifest["cells"]]
        self.assertIn("e1_n99_L40000", ids)
        self.assertIn("e1_n101_L40000", ids)
        self.assertEqual(manifest["seeds"]["data"], [11001, 11002, 11003])
        self.assertEqual(manifest["seeds"]["search"], 233)

    def test_e2_panel_has_9_cells(self):
        manifest = EXP.emit_manifest("e2", "/iqtree2", "/caster")
        self.assertEqual(len(manifest["cells"]), 9)


class MaterializeTest(unittest.TestCase):
    def manifest(self):
        m = ManifestTest().base_manifest()
        return m

    def test_materialize_is_idempotent_and_frozen(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = self.manifest()
            case = manifest["cells"][0]
            ds_dir, rec = EXP.materialize_dataset(
                manifest, case, 11001, d, log=lambda m: None)
            with open(os.path.join(ds_dir, "alignment.fasta"),
                      "rb") as handle:
                first = handle.read()
            ds_dir2, rec2 = EXP.materialize_dataset(
                manifest, case, 11001, d, log=lambda m: None)
            self.assertEqual(ds_dir, ds_dir2)
            self.assertEqual(rec["input_sha256"], rec2["input_sha256"])
            with open(os.path.join(ds_dir, "alignment.fasta"),
                      "rb") as handle:
                self.assertEqual(handle.read(), first)
            self.assertTrue(os.path.exists(
                os.path.join(ds_dir, "sim_true_tree.nwk")))
            self.assertEqual(rec["observed"]["n_taxa"], 8)
            self.assertEqual(rec["observed"]["alignment_length"], 120)

    def test_refuses_overwrite_of_different_spec(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = self.manifest()
            case = manifest["cells"][0]
            ds_dir, _ = EXP.materialize_dataset(
                manifest, case, 11001, d, log=lambda m: None)
            meta = os.path.join(ds_dir, "dataset.json")
            with open(meta) as handle:
                record = json.load(handle)
            record["spec"]["L"] = 999
            with open(meta, "w") as handle:
                json.dump(record, handle)
            with self.assertRaises(ValueError):
                EXP.materialize_dataset(manifest, case, 11001, d,
                                        log=lambda m: None)


class RunUnitTest(unittest.TestCase):
    def make_caster(self, directory):
        path = os.path.join(directory, "fake_caster")
        with open(path, "w") as handle:
            handle.write(
                "#!/bin/bash\n"
                "out=\n"
                "while [ $# -gt 0 ]; do\n"
                "  if [ \"$1\" = -o ]; then out=$2; shift 2; else shift; fi\n"
                "done\n"
                "printf '((s0,s1),(s2,s3));\\n' > \"$out\"\n")
        os.chmod(path, 0o755)
        return path

    def manifest(self, caster):
        m = ManifestTest().base_manifest()
        m["variants"]["v1"]["caster_bin"] = caster
        m["variants"]["v1"]["extra_env"] = {"TIME_BIN": "/nonexistent"}
        return m

    def test_run_unit_records_verified_result(self):
        with tempfile.TemporaryDirectory() as d:
            caster = self.make_caster(d)
            manifest = self.manifest(caster)
            case = manifest["cells"][0]
            ds_dir, rec = EXP.materialize_dataset(
                manifest, case, 11001, d, log=lambda m: None)
            result = EXP.run_unit(
                manifest, case, rec["dataset_id"], rec["spec"],
                "v1", 1, d, log=lambda m: None)
            self.assertEqual(result["result_state"], "verified_complete")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["rf_vs_truth"], 0)
            self.assertTrue(os.path.exists(os.path.join(
                d, "runs", "c1", rec["dataset_id"], "v1", "rep1",
                "result.json")))

    def test_timeout_terminates_process_group(self):
        with tempfile.TemporaryDirectory() as d:
            pidfile = os.path.join(d, "caster.pid")
            caster = os.path.join(d, "slow_caster")
            with open(caster, "w") as handle:
                handle.write("#!/bin/bash\n"
                             f"echo $$ > {pidfile}\n"
                             "sleep 60\n")
            os.chmod(caster, 0o755)
            manifest = self.manifest(caster)
            case = dict(manifest["cells"][0], budget_seconds=2)
            _, rec = EXP.materialize_dataset(
                manifest, case, 11001, d, log=lambda m: None)
            result = EXP.run_unit(
                manifest, case, rec["dataset_id"], rec["spec"],
                "v1", 1, d, log=lambda m: None)
            self.assertTrue(result["driver_timed_out"])
            self.assertEqual(result["result_state"], "timeout")
            with open(pidfile) as handle:
                caster_pid = int(handle.read().strip())
            with self.assertRaises(ProcessLookupError):
                os.kill(caster_pid, 0)   # no orphaned CASTER survives


class PairedAnalysisTest(unittest.TestCase):
    def row(self, ds, variant, rep, elapsed, state="verified_complete",
            block="nodeA"):
        return {"dataset_id": ds, "variant_id": variant, "rep": rep,
                "hardware_block": block, "result_state": state,
                "elapsed_seconds": str(elapsed)}

    def test_speedup_and_exclusions(self):
        results = []
        for ds in ("d1", "d2"):
            for rep in (1, 2, 3):
                results.append(self.row(ds, "base", rep, 100.0))
                results.append(self.row(ds, "cand", rep, 50.0))
        # one incomplete pair: candidate timed out -> excluded, reported
        results.append(self.row("d3", "base", 1, 100.0))
        results.append(self.row("d3", "cand", 1, 50.0, state="timeout"))
        summary = EXP.paired_analysis(results, "base", "cand")
        self.assertAlmostEqual(summary["speedup"], 2.0)
        self.assertEqual(summary["n_datasets"], 2)
        self.assertEqual(summary["n_pairs"], 6)
        self.assertEqual(len(summary["excluded"]), 1)
        self.assertEqual(summary["excluded"][0]["dataset_id"], "d3")
        self.assertLessEqual(summary["ci95"][0], summary["speedup"])
        self.assertLessEqual(summary["speedup"], summary["ci95"][1])

    def test_no_pairs_returns_empty_summary(self):
        results = [self.row("d1", "base", 1, 100.0)]
        summary = EXP.paired_analysis(results, "base", "cand")
        self.assertEqual(summary["speedup"], "")
        self.assertEqual(summary["n_datasets"], 0)
        self.assertEqual(len(summary["excluded"]), 1)

    def test_technical_reps_median_not_mean(self):
        results = []
        for rep, t in ((1, 50.0), (2, 60.0), (3, 1000.0)):
            results.append(self.row("d1", "base", rep, 100.0))
            results.append(self.row("d1", "cand", rep, t))
        summary = EXP.paired_analysis(results, "base", "cand")
        # median log-ratio = log(100/60), not inflated by the outlier rep
        self.assertAlmostEqual(summary["speedup"], 100.0 / 60.0)


if __name__ == "__main__":
    unittest.main()
