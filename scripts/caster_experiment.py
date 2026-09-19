#!/usr/bin/env python3
"""caster_experiment — manifest-driven harness for CASTER backbone cells.

One experiment manifest pins down cells, dataset seeds, generator, CASTER
configuration, variants and budgets. Datasets are materialized once under
a content-hash directory; analysis reads the frozen artifacts and never
regenerates the truth when scoring another backend.

    caster_experiment.py emit --panel e1 --alisim-bin IQTREE2 > exp.json
    caster_experiment.py materialize --manifest exp.json --out ROOT
    caster_experiment.py run --manifest exp.json --out ROOT \
        --case CASE [--dataset D --variant V --rep K]
    caster_experiment.py status --manifest exp.json --out ROOT
    caster_experiment.py analyze --manifest exp.json --out ROOT [--case C]

Layout produced under ROOT:

    experiment.json            frozen manifest copy + manifest sha
    datasets/<dataset_id>/     dataset.json, sim_true_tree.nwk,
                               alignment.fasta, optional mask.json
    runs/<case>/<ds>/<var>/rep<K>/out/    caster_run.sh output
    runs/<case>/<ds>/<var>/rep<K>/result.json
    analysis/<case>_<cand>_vs_<base>.tsv / .json

Dataset ids embed a short hash of the full dataset spec, so identical
specs across cells share one frozen dataset and a spec change produces a
new directory instead of overwriting the old truth.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import signal
import statistics
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import caster_report  # noqa: E402
import caster_thread_plan  # noqa: E402
import rf_distance  # noqa: E402

sys.setrecursionlimit(100000)

SCHEMA = "caster-experiment/1"
MISSING = set("-?NnXx")


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sub_seed(data_seed, role):
    """Deterministic per-role sub-seed, independent of PYTHONHASHSEED and
    of the panel name, so identical specs share one frozen dataset."""
    digest = sha256_text(f"caster-experiment:{data_seed}:{role}")
    return int(digest[:15], 16) % (2 ** 31 - 1) + 1


# ---------------- trees ---------------------------------------------------

def _height_assign(node):
    """Ultrametric heights in [0,1]: tips at 0, root at 1. Post-order
    depth_below is computed once so pectinate trees stay O(n)."""
    depths = {}
    stack = [(node, False)]
    while stack:
        nd, done = stack.pop()
        if done:
            depths[id(nd)] = 0 if not nd["children"] else \
                1 + max(depths[id(c)] for c in nd["children"])
        else:
            stack.append((nd, True))
            stack.extend((c, False) for c in nd["children"])
    root_depth = depths[id(node)] or 1

    def serialize(nd):
        if not nd["children"]:
            return f"{nd['name']}:{nd.get('length', 0.0):.8f}"
        inner = ",".join(serialize(c) for c in nd["children"])
        return f"({inner}):{nd.get('length', 0.0):.8f}"

    def assign(nd, height):
        for child in nd["children"]:
            child_height = depths[id(child)] / root_depth
            child["length"] = max(height - child_height, 0.0)
            assign(child, child_height)

    node["length"] = 0.0
    assign(node, 1.0)
    return serialize(node) + ";"


def balanced_tree(n, height):
    """Balanced binary topology, ultrametric, max root-to-tip = height."""
    def split(lo, k):
        if k == 1:
            return {"name": f"s{lo}", "children": []}
        mid = k // 2
        return {"name": None,
                "children": [split(lo, mid), split(lo + mid, k - mid)]}
    if n < 2:
        raise ValueError("n must be >= 2")
    return _scale_newick(_height_assign(split(0, n)), height)


def pectinate_tree(n, height):
    """Caterpillar (((s0,s1),s2),...); ultrametric, max depth = height."""
    if n < 2:
        raise ValueError("n must be >= 2")
    node = {"name": None, "children": [
        {"name": "s0", "children": []}, {"name": "s1", "children": []}]}
    for i in range(2, n):
        node = {"name": None, "children": [
            node, {"name": f"s{i}", "children": []}]}
    return _scale_newick(_height_assign(node), height)


def yule_tree(n, height, rng):
    """Pure-birth (Yule) tree conditioned on n tips. Waiting times are
    Exp(k) with k active lineages; each split picks a uniform tip. Branch
    lengths keep the generator's times, then the tree is normalized so
    the max root-to-tip distance equals height."""
    if n < 2:
        raise ValueError("n must be >= 2")
    counter = [0]

    def leaf(t):
        name = f"s{counter[0]}"
        counter[0] += 1
        return {"name": name, "children": [], "t_start": t, "t_end": t}

    root = {"name": None, "children": [leaf(0.0), leaf(0.0)],
            "t_start": 0.0, "t_end": 0.0}
    active = list(root["children"])
    t = 0.0
    while len(active) < n:
        t += rng.expovariate(len(active))
        parent = active.pop(rng.randrange(len(active)))
        parent["name"] = None
        parent["t_end"] = t
        for _ in range(2):
            child = leaf(t)
            parent["children"].append(child)
            active.append(child)
    for tip in active:
        tip["t_end"] = t

    def serialize(nd):
        length = nd["t_end"] - nd["t_start"]
        if not nd["children"]:
            return f"{nd['name']}:{length:.8f}"
        inner = ",".join(serialize(c) for c in nd["children"])
        return f"({inner}):{length:.8f}"

    factor = height / t if t > 0 else 1.0
    return _scale_newick(serialize(root) + ";", factor)


def _scale_newick(nwk, factor):
    return re.sub(r":([0-9.eE+-]+)",
                  lambda m: f":{float(m.group(1)) * factor:.8f}", nwk)


def build_tree(spec, rng):
    n = spec["n"]
    height = float(spec.get("height", 0.2))
    shape = spec.get("shape", "balanced")
    if shape == "balanced":
        nwk = balanced_tree(n, 1.0)
    elif shape == "pectinate":
        nwk = pectinate_tree(n, 1.0)
    elif shape == "yule":
        nwk = yule_tree(n, 1.0, rng)
    else:
        raise ValueError(f"unknown tree shape: {shape}")
    return _scale_newick(nwk, height)


# ---------------- internal JC simulator (tests / offline fallback) --------

BASES = "ACGT"


def parse_newick(text):
    """Parse the bifurcating Newick produced here -> (nodes, root)."""
    tokens = re.findall(r"\(|\)|;|,|:[^(),;]+|[^(),;:]+", text)
    pos = [0]
    nodes = []

    def subtree():
        tok = tokens[pos[0]]
        pos[0] += 1
        if tok == "(":
            children = [subtree()]
            while tokens[pos[0]] == ",":
                pos[0] += 1
                children.append(subtree())
            pos[0] += 1                      # ')'
            name = None
        else:
            children, name = [], tok
        length = 0.0
        if pos[0] < len(tokens) and tokens[pos[0]].startswith(":"):
            length = float(tokens[pos[0]][1:])
            pos[0] += 1
        nodes.append({"children": children, "name": name,
                      "length": length})
        return len(nodes) - 1

    root = subtree()
    return nodes, root


def internal_jc(tree_nwk, length, rng):
    """Minimal JC69 simulator over a Newick tree. For harness tests and
    offline checks; the campaign generator is AliSim."""
    nodes, root = parse_newick(tree_nwk)
    seqs = {}
    ancestor = [rng.choice(BASES) for _ in range(length)]

    def evolve(nd, seq):
        bl = nodes[nd]["length"]
        p_change = 0.75 * (1.0 - math.exp(-4.0 * bl / 3.0))
        out = [rng.choice([b for b in BASES if b != base])
               if rng.random() < p_change else base
               for base in seq]
        if not nodes[nd]["children"]:
            seqs[nodes[nd]["name"]] = "".join(out)
            return
        for child in nodes[nd]["children"]:
            evolve(child, out)

    evolve(root, ancestor)
    return seqs


# ---------------- fasta + masks -------------------------------------------

def read_fasta(path):
    seqs, cur = {}, None
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                cur = line[1:].split()[0]
                seqs[cur] = ""
            elif cur is not None:
                seqs[cur] += line.upper()
    return seqs


def write_fasta(path, seqs):
    with open(path, "w") as handle:
        for name, row in seqs.items():
            handle.write(f">{name}\n{row}\n")


def apply_mask(seqs, spec, rng):
    """Return (masked_seqs, mask_record). Absence is written as '-'."""
    kind = spec.get("kind", "none")
    names = sorted(seqs)
    width = len(seqs[names[0]]) if names else 0
    record = {"kind": kind, "nominal_p": spec.get("p", 0.0)}
    if kind == "none":
        record["effective_absence"] = sum(
            c in MISSING for s in seqs.values() for c in s) / max(
                1, width * len(names))
        return dict(seqs), record
    p = float(spec.get("p", 0.0))
    if not 0.0 <= p <= 1.0:
        raise ValueError("mask probability must be in [0,1]")
    masked = {k: list(v) for k, v in seqs.items()}
    if kind == "independent":
        for name in names:
            row = masked[name]
            for j in range(width):
                if rng.random() < p:
                    row[j] = "-"
        record["blocks"] = None
    elif kind == "locus":
        blocks = int(spec.get("blocks", 1))
        edges = [round(i * width / blocks) for i in range(blocks + 1)]
        record["blocks"] = blocks
        for name in names:
            for b in range(blocks):
                if rng.random() < p:
                    for j in range(edges[b], edges[b + 1]):
                        masked[name][j] = "-"
    else:
        raise ValueError(f"unknown mask kind: {kind}")
    out = {k: "".join(v) for k, v in masked.items()}
    total = width * len(names)
    absent = sum(c in MISSING for s in out.values() for c in s)
    record["effective_absence"] = absent / max(1, total)
    present = {k: sum(c not in MISSING for c in s) for k, s in out.items()}
    record["taxon_present_min"] = min(present.values(), default=0)
    record["taxon_present_mean"] = (
        sum(present.values()) / len(present) if present else 0)
    return out, record


def observed_dimensions(path, chunk):
    seqs = read_fasta(path)
    width = len(next(iter(seqs.values()))) if seqs else 0
    nonempty_cols = 0
    chunks_nonempty = 0
    for start in range(0, width, chunk):
        block_has_site = False
        for j in range(start, min(start + chunk, width)):
            if any(seqs[name][j] not in MISSING for name in seqs):
                nonempty_cols += 1
                block_has_site = True
        if block_has_site:
            chunks_nonempty += 1
    return {
        "n_taxa": len(seqs),
        "alignment_length": width,
        "sites_nonempty": nonempty_cols,
        "chunks_total": (caster_thread_plan.effective_chunks(width, chunk)
                         if width else 0),
        "chunks_nonempty": chunks_nonempty,
    }


# ---------------- manifest -------------------------------------------------

def load_manifest(path):
    with open(path) as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != SCHEMA:
        raise ValueError(
            f"unsupported schema: {manifest.get('schema')!r}")
    required = ["protocol", "stage", "seeds", "cells", "variants",
                "baseline_variant", "repetitions", "generator"]
    for key in required:
        if key not in manifest:
            raise ValueError(f"manifest missing required key: {key}")
    if manifest["baseline_variant"] not in manifest["variants"]:
        raise ValueError("baseline_variant must name a variant")
    ids = [c["case_id"] for c in manifest["cells"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case_id in manifest")
    return manifest


def dataset_spec(manifest, case, data_seed):
    """Everything that determines dataset content (not the bin path)."""
    gen = manifest["generator"]
    return {
        "n": case["n"],
        "L": case["L"],
        "tree": case["tree"],
        "mask": case.get("mask", {"kind": "none"}),
        "model": gen.get("model", "JC"),
        "generator_kind": gen.get("kind", "alisim"),
        "generator_version": gen.get("version", ""),
        "data_seed": data_seed,
    }


def dataset_id(spec):
    readable = (f"n{spec['n']}_L{spec['L']}_{spec['tree']['shape']}"
                f"_{spec['mask']['kind']}_s{spec['data_seed']}")
    return readable + "-" + sha256_text(canonical_json(spec))[:8]


def manifest_cases(manifest, only_case=None):
    for case in manifest["cells"]:
        if only_case and case["case_id"] != only_case:
            continue
        yield case


def iter_units(manifest, only_case=None, only_dataset=None,
               only_variant=None, only_rep=None):
    for case in manifest_cases(manifest, only_case):
        for seed in manifest["seeds"]["data"]:
            spec = dataset_spec(manifest, case, seed)
            ds = dataset_id(spec)
            if only_dataset and ds != only_dataset:
                continue
            for variant in manifest["variants"]:
                if only_variant and variant != only_variant:
                    continue
                for rep in range(1, int(manifest["repetitions"]) + 1):
                    if only_rep and rep != only_rep:
                        continue
                    yield case, ds, spec, variant, rep


# ---------------- materialize ----------------------------------------------

def materialize_dataset(manifest, case, data_seed, root, log=print):
    spec = dataset_spec(manifest, case, data_seed)
    ds = dataset_id(spec)
    ds_dir = os.path.join(root, "datasets", ds)
    meta_path = os.path.join(ds_dir, "dataset.json")
    if os.path.exists(meta_path):
        with open(meta_path) as handle:
            existing = json.load(handle)
        if existing.get("spec") == spec:
            log(f"dataset {ds}: already materialized")
            return ds_dir, existing
        raise ValueError(
            f"dataset dir {ds_dir} holds a different spec; refusing to "
            "overwrite a frozen dataset")
    os.makedirs(ds_dir, exist_ok=True)
    topo_seed = sub_seed(data_seed, "topology")
    evo_seed = sub_seed(data_seed, "evolution")
    mask_seed = sub_seed(data_seed, "mask")

    tree_nwk = build_tree(
        {**case["tree"], "n": case["n"]}, random.Random(topo_seed))
    tree_path = os.path.join(ds_dir, "sim_true_tree.nwk")
    with open(tree_path, "w") as handle:
        handle.write(tree_nwk + "\n")

    gen = manifest["generator"]
    aln_path = os.path.join(ds_dir, "alignment.fasta")
    raw_path = os.path.join(ds_dir, "alignment_raw.fasta")
    if gen.get("kind", "alisim") == "internal-jc":
        seqs = internal_jc(tree_nwk, case["L"], random.Random(evo_seed))
        write_fasta(raw_path, seqs)
    else:
        bin_path = gen["bin"]
        prefix = os.path.join(ds_dir, "alisim")
        cmd = [bin_path, "--alisim", prefix, "-m", gen.get("model", "JC"),
               "-t", tree_path, "--length", str(case["L"]),
               "--out-format", "fasta", "-seed", str(evo_seed)]
        cmd += gen.get("extra_args", [])
        with open(os.path.join(ds_dir, "alisim.log"), "w") as logh:
            proc = subprocess.run(cmd, stdout=logh, stderr=logh,
                                  text=True)
        produced = prefix + ".fa"
        if proc.returncode != 0 or not os.path.exists(produced):
            raise RuntimeError(
                f"AliSim failed for {ds} (rc={proc.returncode}); "
                "see alisim.log")
        os.replace(produced, raw_path)

    mask_spec = case.get("mask", {"kind": "none"})
    mask_seed_rng = random.Random(mask_seed)
    seqs = read_fasta(raw_path)
    masked, mask_record = apply_mask(seqs, mask_spec, mask_seed_rng)
    write_fasta(aln_path, masked)
    if mask_spec.get("kind", "none") != "none":
        with open(os.path.join(ds_dir, "mask.json"), "w") as handle:
            json.dump(mask_record, handle, indent=1, sort_keys=True)

    chunk = int(case.get("chunk",
                         manifest.get("defaults", {}).get("chunk", 10000)))
    record = {
        "dataset_id": ds,
        "spec": spec,
        "seeds": {"data": data_seed, "topology": topo_seed,
                  "evolution": evo_seed, "mask": mask_seed},
        "tree_sha256": file_sha256(tree_path),
        "input_sha256": file_sha256(aln_path),
        "raw_input_sha256": file_sha256(raw_path),
        "mask": mask_record,
        "observed": observed_dimensions(aln_path, chunk),
    }
    tmp = meta_path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(record, handle, indent=1, sort_keys=True)
    os.replace(tmp, meta_path)
    log(f"dataset {ds}: materialized "
        f"(n={record['observed']['n_taxa']} "
        f"L={record['observed']['alignment_length']})")
    return ds_dir, record


# ---------------- run ------------------------------------------------------

def run_unit(manifest, case, ds, spec, variant, rep, root, log=print):
    """Execute one (case, dataset, variant, rep) via caster_run.sh and
    harvest result.json. A unit that exhausts its budget is sent SIGTERM
    so the runner records status=timeout."""
    run_dir = os.path.join(root, "runs", case["case_id"], ds, variant,
                           f"rep{rep}")
    out_dir = os.path.join(run_dir, "out")
    result_path = os.path.join(run_dir, "result.json")
    ds_dir = os.path.join(root, "datasets", ds)
    input_path = os.path.join(ds_dir, "alignment.fasta")
    os.makedirs(run_dir, exist_ok=True)

    var = manifest["variants"][variant]
    env = os.environ.copy()
    env.update({
        "CASTER_BIN": var["caster_bin"],
        "CASTER_BACKEND": var.get("backend", ""),
        "THREADS": str(var.get("threads", 1)),
        "SEED": str(manifest["seeds"]["search"]),
        "CASTER_CHUNK": str(case.get(
            "chunk", manifest.get("defaults", {}).get("chunk", 10000))),
        "PHYLOGENY_ROOT": manifest.get("phylogeny_root",
                                       env.get("PHYLOGENY_ROOT", "")),
    })
    env.update(var.get("extra_env", {}))
    budget = int(case.get("budget_seconds",
                          manifest.get("budgets", {})
                          .get("search_seconds", 1200)))
    runner = os.path.join(SCRIPT_DIR, "caster_run.sh")
    proc = subprocess.Popen(["bash", runner, input_path, out_dir],
                            env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)
    timed_out = False
    try:
        output, _ = proc.communicate(timeout=budget)
    except subprocess.TimeoutExpired:
        timed_out = True
        # Signal the whole process group: bash defers TERM traps while a
        # foreground child runs, so signalling only the runner would leave
        # CASTER orphaned and racing the next unit.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            output, _ = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            output, _ = proc.communicate()
    with open(os.path.join(run_dir, "driver.log"), "w") as handle:
        handle.write(output or "")

    meta = caster_report.read_meta(os.path.join(out_dir, "run.meta.tsv"))
    state = caster_report.result_state(out_dir, input_path)
    tree = os.path.join(out_dir, "caster.treefile")
    rf_truth = rf_norm = shared = ""
    truth = os.path.join(ds_dir, "sim_true_tree.nwk")
    if state == "verified_complete" and os.path.exists(truth):
        rf_truth, rf_norm, shared = rf_distance.rf_distance(tree, truth)
    with open(os.path.join(ds_dir, "dataset.json")) as handle:
        observed = json.load(handle).get("observed", {})
    result = {
        "case_id": case["case_id"], "dataset_id": ds,
        "variant_id": variant, "rep": rep,
        "result_state": state, "rc": proc.returncode,
        "driver_timed_out": timed_out,
        "status": meta.get("status", ""),
        "exit_code": meta.get("exit_code", ""),
        "elapsed_seconds": meta.get("elapsed_seconds", ""),
        "input_sha256": meta.get("input_sha256", ""),
        "caster_config_sha256": meta.get("caster_config_sha256", ""),
        "slurm_job_id": meta.get("slurm_job_id", ""),
        "host": meta.get("host", ""),
        "hardware_block": var.get("hardware_block",
                                  meta.get("host", "local")),
        "rf_vs_truth": rf_truth, "rf_norm_vs_truth": rf_norm,
        "shared_taxa_vs_truth": shared,
        "observed": observed,
    }
    with open(result_path + ".tmp", "w") as handle:
        json.dump(result, handle, indent=1, sort_keys=True)
    os.replace(result_path + ".tmp", result_path)
    log(f"{case['case_id']}/{ds}/{variant}/rep{rep}: {state} "
        f"rc={proc.returncode}")
    return result


# ---------------- paired analysis ------------------------------------------

def load_results(root, only_case=None):
    runs_root = os.path.join(root, "runs")
    results = []
    for dirpath, _dirs, files in os.walk(runs_root):
        if "result.json" not in files:
            continue
        with open(os.path.join(dirpath, "result.json")) as handle:
            row = json.load(handle)
        if only_case and row["case_id"] != only_case:
            continue
        results.append(row)
    return results


def paired_analysis(results, baseline, candidate, bootstrap_seed=41001,
                    resamples=10000):
    """Median log-ratio per (dataset, hardware_block) over technical reps,
    then an equal-weight mean over datasets, exponentiated. Bootstrap the
    dataset set (seed 41001, 10k draws). Only verified_complete pairs."""
    pairs = {}
    excluded = []
    for row in results:
        if row["variant_id"] not in (baseline, candidate):
            continue
        key = (row["dataset_id"], row["rep"], row["hardware_block"])
        pairs.setdefault(key, {})[row["variant_id"]] = row
    log_ratios = {}
    for (ds, rep, block), sides in sorted(pairs.items()):
        base = sides.get(baseline)
        cand = sides.get(candidate)
        ok = (base and cand and
              base["result_state"] == "verified_complete" and
              cand["result_state"] == "verified_complete")
        if not ok:
            excluded.append({"dataset_id": ds, "rep": rep,
                             "hardware_block": block,
                             "baseline_state": (base or {}).get(
                                 "result_state", "absent"),
                             "candidate_state": (cand or {}).get(
                                 "result_state", "absent")})
            continue
        t_base = float(base["elapsed_seconds"])
        t_cand = float(cand["elapsed_seconds"])
        if t_base <= 0 or t_cand <= 0:
            excluded.append({"dataset_id": ds, "rep": rep,
                             "hardware_block": block,
                             "reason": "nonpositive elapsed"})
            continue
        log_ratios.setdefault((ds, block), []).append(
            math.log(t_base / t_cand))
    per_block = {k: statistics.median(v) for k, v in log_ratios.items()}
    per_dataset = {}
    for (ds, _block), value in per_block.items():
        per_dataset.setdefault(ds, []).append(value)
    dataset_effects = {ds: statistics.mean(v)
                       for ds, v in per_dataset.items()}
    effect_list = list(dataset_effects.values())
    if not effect_list:
        return {"baseline": baseline, "candidate": candidate,
                "n_datasets": 0, "n_pairs": 0, "excluded": excluded,
                "speedup": "", "ci95": ["", ""],
                "dataset_effects": dataset_effects}
    estimate = statistics.mean(effect_list)
    rng = random.Random(bootstrap_seed)
    boots = []
    for _ in range(resamples):
        draw = [rng.choice(effect_list) for _ in effect_list]
        boots.append(statistics.mean(draw))
    boots.sort()
    lo = boots[int(0.025 * resamples)]
    hi = boots[min(resamples - 1, int(0.975 * resamples))]
    return {
        "baseline": baseline, "candidate": candidate,
        "n_datasets": len(effect_list),
        "n_pairs": sum(len(v) for v in log_ratios.values()),
        "excluded": excluded,
        "speedup": math.exp(estimate),
        "ci95": [math.exp(lo), math.exp(hi)],
        "dataset_effects": {
            ds: math.exp(v) for ds, v in dataset_effects.items()},
    }


# ---------------- emit ------------------------------------------------------

def e1_cells():
    cells = []
    for length in (3000, 10000, 40000, 160000, 320000):
        cells.append({
            "case_id": f"e1_n128_L{length}",
            "n": 128, "L": length,
            "tree": {"shape": "balanced", "height": 0.2},
            "mask": {"kind": "none"},
            "chunk": 10000,
            "budget_seconds": 1200,
            "regime": "n_ge_100",
        })
    for n in (32, 60, 99, 100, 101, 256):
        cells.append({
            "case_id": f"e1_n{n}_L40000",
            "n": n, "L": 40000,
            "tree": {"shape": "balanced", "height": 0.2},
            "mask": {"kind": "none"},
            "chunk": 10000,
            "budget_seconds": 1200,
            "regime": "n_lt_100" if n < 100 else "n_ge_100",
        })
    return cells


def e2_cells():
    cells = []
    for shape in ("balanced", "pectinate", "yule"):
        for mask in ({"kind": "none"},
                     {"kind": "independent", "p": 0.8},
                     {"kind": "locus", "p": 0.8, "blocks": 31}):
            tag = mask["kind"]
            cells.append({
                "case_id": f"e2_{shape}_{tag}",
                "n": 128, "L": 40000,
                "tree": {"shape": shape, "height": 0.2},
                "mask": mask,
                "chunk": 10000,
                "budget_seconds": 1200,
                "regime": "n_ge_100",
            })
    return cells


def emit_manifest(panel, alisim_bin, caster_bin):
    cells = e1_cells() if panel == "e1" else e2_cells()
    return {
        "schema": SCHEMA,
        "protocol": f"caster-{panel}-panel",
        "stage": "exploration",
        "generator": {"kind": "alisim", "bin": alisim_bin,
                      "model": "JC", "extra_args": []},
        "seeds": {"data": [11001, 11002, 11003], "search": 233,
                  "order": 31001, "bootstrap": 41001},
        "defaults": {"chunk": 10000},
        "cells": cells,
        "variants": {
            "cpu_ref": {"caster_bin": caster_bin,
                        "backend": "cpu-portable", "threads": 1,
                        "extra_env": {}},
        },
        "baseline_variant": "cpu_ref",
        "repetitions": 3,
        "budgets": {"search_seconds": 1200},
    }


# ---------------- cli -------------------------------------------------------

def cmd_emit(args):
    manifest = emit_manifest(args.panel, args.alisim_bin, args.caster_bin)
    text = json.dumps(manifest, indent=1, sort_keys=True)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)


def cmd_plan(args):
    manifest = load_manifest(args.manifest)
    for case in manifest_cases(manifest, args.case):
        seeds = manifest["seeds"]["data"]
        print(f"{case['case_id']}\tn={case['n']}\tL={case['L']}\t"
              f"mask={case.get('mask', {}).get('kind', 'none')}\t"
              f"datasets={len(seeds)}\treps={manifest['repetitions']}")


def cmd_materialize(args):
    manifest = load_manifest(args.manifest)
    os.makedirs(os.path.join(args.out, "datasets"), exist_ok=True)
    frozen = os.path.join(args.out, "experiment.json")
    if not os.path.exists(frozen):
        with open(args.manifest, "rb") as handle:
            payload = handle.read()
        with open(frozen, "wb") as handle:
            handle.write(payload)
        record = {"manifest_sha256": hashlib.sha256(
            payload).hexdigest()}
        with open(os.path.join(args.out, "experiment.sha256"), "w") as h:
            h.write(record["manifest_sha256"] + "\n")
    seen = set()
    for case in manifest_cases(manifest, args.case):
        for seed in manifest["seeds"]["data"]:
            spec = dataset_spec(manifest, case, seed)
            ds = dataset_id(spec)
            if ds in seen or (args.dataset and ds != args.dataset):
                continue
            seen.add(ds)
            materialize_dataset(manifest, case, seed, args.out)


def cmd_run(args):
    manifest = load_manifest(args.manifest)
    if not args.case:
        raise SystemExit("run requires --case")
    for case, ds, spec, variant, rep in iter_units(
            manifest, args.case, args.dataset, args.variant, args.rep):
        ds_dir = os.path.join(args.out, "datasets", ds)
        if not os.path.exists(os.path.join(ds_dir, "dataset.json")):
            materialize_dataset(manifest, case, spec["data_seed"],
                                args.out)
        run_unit(manifest, case, ds, spec, variant, rep, args.out)


def cmd_status(args):
    manifest = load_manifest(args.manifest)
    rows = []
    for case, ds, spec, variant, rep in iter_units(
            manifest, args.case):
        ds_dir = os.path.join(args.out, "datasets", ds)
        input_path = os.path.join(ds_dir, "alignment.fasta")
        out_dir = os.path.join(args.out, "runs", case["case_id"], ds,
                               variant, f"rep{rep}", "out")
        state = caster_report.result_state(out_dir, input_path)
        rows.append((case["case_id"], ds, variant, rep, state))
    for row in rows:
        print("\t".join(map(str, row)))


def cmd_analyze(args):
    manifest = load_manifest(args.manifest)
    results = load_results(args.out, args.case)
    baseline = manifest["baseline_variant"]
    outdir = os.path.join(args.out, "analysis")
    os.makedirs(outdir, exist_ok=True)
    seed = manifest["seeds"].get("bootstrap", 41001)
    for candidate in manifest["variants"]:
        if candidate == baseline:
            continue
        summary = paired_analysis(results, baseline, candidate, seed)
        tag = f"{args.case or 'all'}_{candidate}_vs_{baseline}"
        with open(os.path.join(outdir, tag + ".json"), "w") as handle:
            json.dump(summary, handle, indent=1, sort_keys=True)
        print(f"{tag}\tspeedup={summary['speedup']}\t"
              f"ci95={summary['ci95']}\tdatasets={summary['n_datasets']}"
              f"\texcluded={len(summary['excluded'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("emit")
    p.add_argument("--panel", choices=["e1", "e2"], required=True)
    p.add_argument("--alisim-bin", required=True)
    p.add_argument("--caster-bin", required=True)
    p.add_argument("--out")
    p.set_defaults(func=cmd_emit)
    for name in ("plan", "materialize", "run", "status", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--case")
        if name in ("materialize", "run"):
            p.add_argument("--dataset")
        if name == "run":
            p.add_argument("--variant")
            p.add_argument("--rep", type=int)
        p.set_defaults(func={
            "plan": cmd_plan, "materialize": cmd_materialize,
            "run": cmd_run, "status": cmd_status,
            "analyze": cmd_analyze}[name])
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
