"""Randomized-smoothing certification of clean and quarantined classifiers.

Test-time evasion certificate (Cohen et al.), sigma=0.5, N=200 Monte Carlo
samples, per binary class, on the clean test split. Certifies the last
checkpoint of each clean tag and each quarantine tag produced by the campaign.

Parallel across cells (fork pool; datasets preloaded in the parent for
copy-on-write sharing). Usage:
    python3 -u -m src.s1.run_rs [--workers 4] [dataset]
"""
import json
import glob
import numpy as np
import torch
import os
import argparse
import multiprocessing
from pathlib import Path

from .data import (load_cic_unsw_nb15, load_nf_toniot, load_ciciot2023,
                   stratified_split, Standardizer)
from .train import MLP
from .certify_rs import certify, certified_accuracy_at_radius

# parent-preloaded for COW sharing across forked workers
_RS_DATA = {}
_RS_SPLITS = {}


def _load(dataset: str):
    if dataset == "cic":
        return load_cic_unsw_nb15()
    if dataset == "nf":
        return load_nf_toniot()
    if dataset == "ciciot2023":
        return load_ciciot2023()
    raise ValueError(dataset)


def _rs_one(task):
    dset, attack, ratio, seed = task
    if dset not in _RS_DATA:
        _RS_DATA[dset] = _load(dset)
    X, y, names = _RS_DATA[dset]
    if (dset, seed) not in _RS_SPLITS:
        rng = np.random.default_rng(seed)
        split = stratified_split(y, rng)
        sc = Standardizer().fit(X[split["train"]])
        Xs = sc.transform(X)
        _RS_SPLITS[(dset, seed)] = (Xs[split["test"]], y[split["test"]])
    Xte, yte = _RS_SPLITS[(dset, seed)]

    if attack == "none":
        tag = f"{dset}_clean"
    else:
        tag = f"{dset}_{attack}_r{ratio:g}_quar"
    ckpts_dir = Path(f"data/processed/checkpoints/{tag}_seed{seed}_K5")
    ckpts = sorted(ckpts_dir.glob("ckpt_*.pt")) if ckpts_dir.exists() else []
    if not ckpts:
        print(f"Checkpoint not found for {tag} seed{seed}; skipping")
        return None

    cert = certify(MLP, ckpts[-1], Xte, sigma=0.5, n_samples=200, seed=seed)
    radii_bins = [0.0, 0.25, 0.5, 0.75, 1.0]
    class_res = {}
    for c in np.unique(yte):
        class_res[int(c)] = {
            f"r>={r}": float(certified_accuracy_at_radius(cert, yte, r,
                                                          class_of_interest=int(c)))
            for r in radii_bins
        }
    key = f"{dset}_{attack}_r{ratio}_seed{seed}"
    print(f"RS done: {key}")
    return key, class_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default=None,
                    help="optional dataset filter (cic|nf|ciciot2023)")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent certification processes (GPU-light)")
    ap.add_argument("--max-cells", type=int, default=0,
                    help="cap on number of cells to certify (0 = all)")
    args = ap.parse_args()
    target_dset = args.dataset

    rs_results = {}
    rs_json_path = "results/rs_results.json"
    if os.path.exists(rs_json_path):
        try:
            with open(rs_json_path) as fp:
                rs_results = json.load(fp)
        except Exception:
            pass

    tasks = []
    for f in sorted(glob.glob("results/exp_*.json")):
        if "smoke" in f:
            continue
        with open(f) as fp:
            d = json.load(fp)
        dset = d.get("dataset", "unk")
        if target_dset and dset != target_dset:
            continue
        attack = d.get("attack", "unk")
        if attack != "none" and "n_quarantined" not in d:
            continue
        tasks.append((dset, attack, d.get("ratio", 0.0), d.get("seed", 0)))

    if args.max_cells > 0:
        tasks = tasks[:args.max_cells]

    # preload in parent so forked workers share the arrays copy-on-write
    for dset in sorted({t[0] for t in tasks}):
        _RS_DATA[dset] = _load(dset)

    if args.workers > 1 and len(tasks) > 1:
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(min(args.workers, len(tasks))) as pool:
            for out in pool.imap_unordered(_rs_one, tasks):
                if out is not None:
                    rs_results[out[0]] = out[1]
    else:
        for t in tasks:
            out = _rs_one(t)
            if out is not None:
                rs_results[out[0]] = out[1]

    with open(rs_json_path, "w") as fp:
        json.dump(rs_results, fp, indent=2)
    print(f"wrote {rs_json_path} with {len(rs_results)} RS cells")


if __name__ == "__main__":
    main()
