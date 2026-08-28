#!/usr/bin/env python3
"""Patch rs_cert OOM failures: recertify affected cells' quarantined models.

Some cells' inline RS certification hit CUDA OOM while sibling workers held
TRAK peaks; their rs_cert holds {"error": ...} and everything else is intact.
This script re-runs certify_dual on the cell's quarT checkpoint (GPU idle now),
rewrites the cell JSON, and re-records the manifest fragment (fresh sha).

Run on the pod that owns the checkpoints, from /workspace:
    python3 -u scripts/patch_rs.py
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from src.s1 import config, manifest  # noqa: E402
from src.s1.data import Standardizer, load_cic_unsw_nb15, load_nf_toniot, \
    load_ciciot2023, stratified_split  # noqa: E402
from src.s1.run_experiment import _rs_block, _ckpts  # noqa: E402

_LOADERS = {"cic": load_cic_unsw_nb15, "nf": load_nf_toniot,
            "ciciot2023": load_ciciot2023}
_DATA = {}


def main() -> None:
    n_patched = 0
    for f in sorted(glob.glob(str(config.RESULTS_DIR / "exp_*_K5.json"))):
        if "_smoke" in f:
            continue
        p = Path(f)
        d = json.loads(p.read_text())
        if d.get("attack") == "none":
            continue
        rc = d.get("rs_cert", {})
        if "error" not in rc.get("classes", {}):
            continue
        dataset, attack, ratio, seed = d["dataset"], d["attack"], d["ratio"], d["seed"]
        print(f"[patch] {p.name}: recertifying ...", flush=True)
        if dataset not in _DATA:
            X, y, _ = _LOADERS[dataset]()
            _DATA[dataset] = (X, y)
        X, y = _DATA[dataset]
        rng = np.random.default_rng(seed)
        split = stratified_split(y, rng)
        sc = Standardizer().fit(X[split["train"]])
        Xs = sc.transform(X)
        Xte, yte = Xs[split["test"]], y[split["test"]]

        tag = f"{dataset}_{attack}_r{ratio:g}_quarT"
        ckpts = _ckpts(tag, seed, 5)
        if not ckpts:
            print(f"[patch] {p.name}: NO CHECKPOINTS for {tag} — left as-is",
                  flush=True)
            continue
        t0 = time.time()
        classes = _rs_block(ckpts[-1], Xte, yte, seed, None)
        if "error" in classes:
            print(f"[patch] {p.name}: recert FAILED again: {classes['error'][:80]}",
                  flush=True)
            continue
        d["rs_cert"] = {"model": "tracincp_quarantine", "classes": classes,
                        "patched": True}
        p.write_text(json.dumps(d, indent=2))
        ck_all = (_ckpts(f"{dataset}_{attack}_r{ratio:g}", seed, 5)
                  + _ckpts(f"{dataset}_clean", seed, 5) + ckpts)
        manifest.record_cell(f"{dataset}_{attack}_r{ratio:g}_seed{seed}", p,
                             ck_all, time.time() - t0)
        n_patched += 1
        print(f"[patch] {p.name}: OK", flush=True)
    print(f"[patch] patched {n_patched} cells")


if __name__ == "__main__":
    main()
